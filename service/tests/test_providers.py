"""The backend seam as a setting — the Windows / cloud-primary path.

The contract under test: `[model] backend` selects who answers, defaults keep the
Mac local-first path byte-for-byte unchanged, cloud selection fails fast and
clearly (missing SDK, missing key, unknown name), and the MLX-shaped defaults are
retuned for a cloud primary without ever overriding an explicit setting.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import http.client
import io
import json
import os
import shutil
import socket
import socketserver
import ssl
import subprocess
import sys
import time
import types
import urllib.error
from pathlib import Path
import urllib.request

import pytest
from conftest import (
    PHOTO,
    REAL_CLAUDE_CODE_VERDICT,
    REAL_CODEX_VERDICT,
    BadStatusLine,
    FakeOllama,
    QuietHandler,
    Silent,
    TricklingPull,
    closed_port,
    fake_platform,
    loopback_server,
    ollama_chat_reply,
    proxy_in_the_environment,
    recording_handler,
    redirecting_handler,
    trickle,
)
from test_pipeline import ID_OK, ROUTING_OK

from melampus import providers
from melampus.backend import (
    AnthropicBackend,
    CommandBackend,
    CommandFailed,
    MLXBackend,
    OllamaBackend,
    OpenAIBackend,
    ScriptedBackend,
    _Bounded,
    _Deadline,
    _NotedHTTPS,
    _hang_up,
)
from melampus.config import load_config

ALL_KEY_VARIABLES = [name for names in providers.KEY_VARIABLES.values() for name in names]


def _cfg(**overrides):
    return load_config(use_local=False, **overrides)


@pytest.fixture()
def no_ambient_keys(monkeypatch):
    """A developer's real keys must not decide what these tests assert."""
    for name in ALL_KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def stub_sdks(monkeypatch):
    """Satisfy the factory's eager SDK import without installing either SDK.

    The backends themselves import their SDK lazily inside `_ensure_client`, so
    construction needs no real package — which is exactly what lets the local-only
    install stay local-only.
    """
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))


@pytest.mark.skipif(not providers.on_apple_silicon(), reason="the mlx default only constructs on Apple Silicon")
def test_default_backend_is_local_mlx():
    config = _cfg()
    assert config.model.backend == "mlx"
    assert not providers.is_cloud_primary(config)
    backend = providers.build_primary_backend(config)
    assert isinstance(backend, MLXBackend)
    assert backend.repo == config.model.repo


def test_mlx_is_refused_on_windows_with_directions(monkeypatch, no_ambient_ollama):
    fake_platform(monkeypatch, "win32", "AMD64")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    message = str(err.value)
    assert "claude" in message and "openai" in message and "--backend" in message


def test_mlx_is_refused_on_intel_mac(monkeypatch, no_ambient_ollama):
    """darwin alone is not enough — the pyproject marker also requires arm64,
    so an Intel Mac must get the helpful refusal, not a ModuleNotFoundError
    at warmup."""
    fake_platform(monkeypatch, "darwin", "x86_64")
    with pytest.raises(providers.BackendUnavailable):
        providers.build_primary_backend(_cfg())


def test_claude_primary_builds_the_anthropic_backend_with_its_default_model(
    monkeypatch, no_ambient_keys, stub_sdks
):
    """Card #403: `claude` is the one user-facing name of the Anthropic backend."""
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "key-from-env")
    config = _cfg(model={"backend": "claude"})
    assert providers.is_cloud_primary(config)
    backend = providers.build_primary_backend(config)
    assert isinstance(backend, AnthropicBackend)
    assert backend.model == providers.DEFAULT_MODELS["claude"]


def test_openai_primary_respects_name_and_base_url(monkeypatch, no_ambient_keys, stub_sdks):
    monkeypatch.setenv("MELAMPUS_OPENAI_KEY", "key-from-env")
    config = _cfg(
        model={"backend": "openai", "name": "my-model", "base_url": "http://localhost:1234/v1"}
    )
    backend = providers.build_primary_backend(config)
    assert isinstance(backend, OpenAIBackend)
    assert backend.model == "my-model"
    assert backend._base_url == "http://localhost:1234/v1"


def test_missing_key_fails_fast_and_names_the_variable(no_ambient_keys, stub_sdks):
    with pytest.raises(ValueError) as err:
        providers.build_primary_backend(_cfg(model={"backend": "claude"}))
    assert "MELAMPUS_ANTHROPIC_KEY" in str(err.value)


def test_unknown_backend_is_rejected_with_choices():
    with pytest.raises(ValueError) as err:
        providers.build_primary_backend(_cfg(model={"backend": "gemini"}))
    message = str(err.value)
    assert "gemini" in message and "claude" in message and "openai" in message


ENGINES = ("mlx", "ollama", "openai", "claude")


def test_backend_choices_are_the_owners_engine_names_in_order():
    """Card #403: the engines a user chooses between are mlx, ollama, openai,
    claude, in that order, plus the offline test fake. The order is the one
    card #404's detection will try them in for a default."""
    assert providers.BACKEND_CHOICES == (*ENGINES, providers.SCRIPTED)


@pytest.mark.parametrize("engine", ENGINES)
def test_cli_accepts_each_engine_name(engine, photos, tmp_path, no_ambient_keys, capsys):
    """Card #403, Done-when 1: given an engine name, when the plugin passes it
    as --backend, then the CLI receives it. --report-only stops before any
    engine is built, so this is the parse alone: the name is accepted."""
    from melampus.cli import main

    code = main([str(photos), "--backend", engine, "--report-only",
                 "--cache", str(tmp_path / "cache.jsonl")])

    assert code == 0, capsys.readouterr().err


@pytest.mark.parametrize("name", ["anthropic", "gemini", "MLX"])
def test_cli_rejects_a_name_that_is_not_an_engine(name, photos, tmp_path, capsys):
    from melampus.cli import main

    with pytest.raises(SystemExit) as exit_:
        main([str(photos), "--backend", name, "--report-only",
              "--cache", str(tmp_path / "cache.jsonl")])

    assert exit_.value.code == 2
    err = capsys.readouterr().err
    for engine in ENGINES:
        assert engine in err, f"the usage error does not name {engine!r}:\n{err}"


def test_cli_rejects_an_escalation_provider_that_is_not_in_the_registry(photos, tmp_path, capsys):
    """The clouds --escalate-provider offers are providers.KEY_VARIABLES, the
    registry escalate.py reads, not a list spelled in the CLI. `anthropic` is
    the SDK, not a provider name (card #403: `claude` everywhere a user names
    it), so it is refused as a usage error that names every registered
    provider, the way --backend's names every engine."""
    from melampus.cli import main

    with pytest.raises(SystemExit) as exit_:
        main([str(photos), "--report-only", "--escalate", "--escalate-provider", "anthropic",
              "--cache", str(tmp_path / "cache.jsonl")])

    assert exit_.value.code == 2
    err = capsys.readouterr().err
    for provider in providers.KEY_VARIABLES:
        assert provider in err, f"the usage error does not name {provider!r}:\n{err}"


def test_ollama_not_running_is_refused_before_any_image_is_read_and_names_the_fix(
    no_ambient_keys, no_ambient_ollama
):
    """Card #406, Done-when 2: given Ollama is not running, when the backend is
    asked for, then the refusal says so, names the address it tried and where
    to install Ollama, and names the backends that do work here, through the
    same BackendUnavailable path every refusal takes (exit 3 from the CLI).
    The check is the probe card #404 built, run at construction: no image is
    read first. Ollama is local, so nothing about a cloud primary applies to
    it: no retuned defaults, no cost gate, no cloud cache."""
    config = _cfg(model={"backend": "ollama"})
    assert not providers.is_cloud_primary(config)
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(config)
    message = str(err.value)
    assert "No Ollama server at" in message
    assert providers.OLLAMA_URL in message
    assert providers.OLLAMA_INSTALL in message
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in message, f"{works_here!r} is not named as working here:\n{message}"
    assert "--backend" in message


def test_ollama_not_running_refusal_names_the_configured_address(monkeypatch):
    """The address tried is the configured one, so the message and the probe
    cannot disagree about where Ollama was looked for. One probe: the refusal
    and its "what works" list come from the one detection run, so a port that
    hangs costs OLLAMA_PROBE_SECONDS once, not twice; and one sentence: the
    refusal says what the verdict (--detect-engines, the dialog) says."""
    probed: list[str] = []

    def answers(url=None):
        probed.append(url)
        return False

    monkeypatch.setattr(providers, "ollama_answers", answers)
    config = _cfg(model={"backend": "ollama", "ollama_url": "http://127.0.0.1:11435"})
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(config)
    message = str(err.value)
    assert "No Ollama server at http://127.0.0.1:11435" in message, message
    assert providers.OLLAMA_INSTALL in message, message
    assert probed == ["http://127.0.0.1:11435"], f"probed more than once: {probed}"


def test_cli_backend_ollama_exits_3_with_the_not_running_message(
    photos, tmp_path, capsys, no_ambient_keys, no_ambient_ollama
):
    from melampus.cli import main

    code = main([str(photos), "--backend", "ollama", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "No Ollama server at" in err
    assert providers.OLLAMA_INSTALL in err
    assert "cloud default" not in err, f"ollama is local; nothing was retuned for a cloud:\n{err}"


def test_ollama_primary_builds_the_backend_from_the_model_settings(monkeypatch):
    """Given engine ollama and a model name, the factory builds an OllamaBackend
    on the configured model, address, temperature and timeout; unset, the
    address is providers.OLLAMA_URL."""
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: True)
    backend = providers.build_primary_backend(_cfg(model={"backend": "ollama"}))
    assert isinstance(backend, OllamaBackend)
    assert backend.name == backend.model == "qwen3-vl:8b-instruct"
    assert backend.url == providers.OLLAMA_URL
    assert backend.temperature == 0.0
    assert backend.timeout == 180.0

    backend = providers.build_primary_backend(_cfg(model={
        "backend": "ollama", "ollama_model": "qwen3-vl:30b-a3b-instruct",
        "ollama_url": "http://127.0.0.1:11435/", "temperature": 0.2, "timeout_seconds": 30,
    }))
    assert backend.model == "qwen3-vl:30b-a3b-instruct"
    assert backend.url == "http://127.0.0.1:11435", "a trailing slash must not double up in the endpoint"
    assert backend.temperature == 0.2 and backend.timeout == 30.0


@pytest.mark.parametrize(
    ("arguments", "backend", "extra"),
    [(("--backend", "claude"), "claude", "cloud"),
     (("--backend", "openai"), "openai", "openai"),
     (("--report-only", "--escalate", "--escalate-provider", "claude"), "claude", "cloud"),
     (("--report-only", "--escalate", "--escalate-provider", "openai"), "openai", "openai")],
)
def test_cli_names_the_backend_whose_sdk_is_missing_and_the_extra_that_ships_it(
    arguments, backend, extra, photos, tmp_path, capsys, monkeypatch, no_ambient_keys
):
    """There is no "claude SDK": the package is anthropic and the extra is
    cloud. So the install hint says what it is, the SDK for the backend the
    user asked for, and names the extra that ships it; the primary path and
    the escalation path say it in one shape. The SDK is made absent the way
    Python reports an absent module, so this holds with or without it
    installed here."""
    from melampus.cli import main

    monkeypatch.setitem(sys.modules, "anthropic" if backend == "claude" else "openai", None)
    code = main([str(photos), *arguments, "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert f"The SDK for the {backend} backend is not installed. Run:" in err, err
    assert f'"./service[{extra}]"' in err, err
    assert f"The {backend} SDK" not in err, f"names an SDK that does not exist:\n{err}"


def test_mlx_refusal_does_not_name_ollama_as_working(monkeypatch, no_ambient_ollama):
    """Off Apple Silicon the mlx refusal lists what works here; with no Ollama
    server answering (card #404's detection decides), ollama stays off that
    list."""
    fake_platform(monkeypatch, "win32", "AMD64")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    assert "ollama" not in str(err.value)


def test_engine_round_trips_through_the_local_config(monkeypatch, tmp_path):
    """Card #403: the engine is a setting before it is a dialog. `[model]
    backend = "claude"` in melampus.local.toml is what load_config reads back,
    the same way the plugin's executable reads its config (card #436)."""
    local = tmp_path / "melampus.local.toml"
    local.write_text('[model]\nbackend = "claude"\n', encoding="utf-8")
    from melampus import config as config_module
    monkeypatch.setattr(config_module, "_local_config", lambda: local)

    assert load_config().model.backend == "claude"
    assert load_config(use_local=False).model.backend == "mlx"


def test_anthropic_is_not_a_backend_name():
    """Card #403: the engine names are mlx, ollama, openai, claude. The old
    spelling of the Anthropic backend is refused like any other unknown name,
    so there is exactly one name for it everywhere."""
    with pytest.raises(ValueError) as err:
        providers.build_primary_backend(_cfg(model={"backend": "anthropic"}))
    message = str(err.value)
    assert "anthropic" in message and "claude" in message
    assert "anthropic" not in providers.BACKEND_CHOICES


def test_key_resolution_prefers_config_then_specific_then_generic(monkeypatch):
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "specific")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic")
    from pydantic import SecretStr

    assert providers.resolve_provider_key("claude", SecretStr("explicit")) == "explicit"
    assert providers.resolve_provider_key("claude") == "specific"
    monkeypatch.delenv("MELAMPUS_ANTHROPIC_KEY")
    assert providers.resolve_provider_key("claude") == "generic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert providers.resolve_provider_key("claude") is None


def test_cloud_defaults_retuned_when_left_at_defaults():
    config = _cfg(model={"backend": "openai"})
    changed = providers.apply_cloud_primary_defaults(config)
    assert config.image.max_edge == 2048
    assert config.image.fallback_edges == []
    assert config.model.max_tokens == 1200
    assert config.model.routing_max_tokens == 900
    # Paid answers get their own file, or the next local pass would silently
    # overwrite them (the exact foot-gun EscalationConfig.cache_path documents).
    assert config.run.cache_path.name == "identifications-cloud.jsonl"
    assert len(changed) == 5


def test_cloud_defaults_never_override_explicit_settings(tmp_path):
    my_cache = tmp_path / "my-results.jsonl"
    config = _cfg(
        model={"backend": "openai", "max_tokens": 700},
        image={"max_edge": 1500},
        run={"cache_path": str(my_cache)},
    )
    providers.apply_cloud_primary_defaults(config)
    # Deliberate choices survive, even inconvenient ones.
    assert config.image.max_edge == 1500
    assert config.model.max_tokens == 700
    assert config.run.cache_path == my_cache
    # Untouched siblings are still retuned.
    assert config.model.routing_max_tokens == 900
    assert config.image.fallback_edges == []


def test_registry_is_shared_with_escalation():
    """One registry: a provider added for escalation exists for the primary too."""
    from melampus import escalate

    assert escalate.DEFAULT_MODELS is providers.DEFAULT_MODELS
    assert escalate._KEY_VARIABLES is providers.KEY_VARIABLES


def test_scripted_backend_is_selectable_and_local():
    """Card #399: the shipped executable must be smoke-testable on a machine with
    no weights, so the fake the unit tests use is reachable from the same
    `[model] backend` setting as the real ones. It answers nothing useful, and
    it is local: no cloud retuning, no cost prompt, no cloud cache file."""
    config = _cfg(model={"backend": "scripted"})
    assert not providers.is_cloud_primary(config)
    backend = providers.build_primary_backend(config)
    assert isinstance(backend, ScriptedBackend)
    assert backend.name == "scripted"


def test_cli_backend_scripted_writes_a_result_without_weights(photos, tmp_path, capsys):
    """`melampus-id FOLDER --backend scripted --json-out FILE` runs the whole
    pipeline (staging, prompts, retry, cache, export) and writes one result per
    image, attributed to the scripted backend."""
    from melampus.cli import main

    out = tmp_path / "results.json"

    code = main([
        str(photos), "--backend", "scripted",
        "--cache", str(tmp_path / "cache.jsonl"), "--json-out", str(out),
    ])

    assert code == 0, capsys.readouterr().err
    results = json.loads(out.read_text(encoding="utf-8"))
    assert [r["file"] for r in results] == [PHOTO]
    assert results[0]["model"] == "scripted"
    assert results[0]["status"] == "unprocessed"


def test_cli_backend_mlx_on_windows_names_apple_silicon_and_the_backends_that_work(
    photos, tmp_path, capsys, monkeypatch, no_ambient_ollama
):
    """Card #400, Done-when 2: given the Windows executable, when the local MLX
    engine is requested, then it says clearly that MLX needs Apple Silicon and
    names the engines that work here — every backend but mlx and, with no
    server answering, ollama (card #404's detection decides)."""
    from melampus.cli import main

    fake_platform(monkeypatch, "win32", "AMD64")

    code = main([str(photos), "--backend", "mlx", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code != 0
    assert "Apple Silicon" in err
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in err, f"{works_here!r} is not named as working here:\n{err}"
    assert "ollama" not in err, f"an engine with no server answering is named as working:\n{err}"


# ---------------------------------------------------------------------------
# Card #404: which engines can run on this machine.


@pytest.fixture()
def no_ambient_ollama(monkeypatch):
    """A developer's running Ollama must not decide what these tests assert."""
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: False)


def _verdict(engine: str) -> providers.EngineVerdict:
    (verdict,) = [v for v in providers.detect_engines() if v.engine == engine]
    return verdict


def test_detection_lists_the_engines_in_the_owners_order_then_the_subscription_clis(
    no_ambient_keys, no_ambient_ollama
):
    """The list the dialog (card #405) will show: one verdict per engine, in
    the order BACKEND_CHOICES names them, then claude-code and codex (cards
    #421, #422; the picker learns them in #423), never the test fake."""
    verdicts = providers.detect_engines()
    assert [v.engine for v in verdicts] == [*ENGINES, providers.CLAUDE_CODE, providers.CODEX]
    for verdict in verdicts:
        assert isinstance(verdict.available, bool)
        assert verdict.reason, f"{verdict.engine} has no reason"


def test_detection_mlx_is_available_on_apple_silicon(monkeypatch, no_ambient_ollama):
    """Done-when 1: given Apple Silicon, when detection runs, then mlx is available."""
    fake_platform(monkeypatch, "darwin", "arm64")
    assert _verdict("mlx").available


@pytest.mark.parametrize(("platform_name", "machine"), [("win32", "AMD64"), ("darwin", "x86_64"), ("linux", "x86_64")])
def test_detection_mlx_needs_apple_silicon_anywhere_else(monkeypatch, no_ambient_ollama, platform_name, machine):
    """Done-when 1: given anything else, then mlx is unavailable with the
    reason "needs Apple Silicon"."""
    fake_platform(monkeypatch, platform_name, machine)
    verdict = _verdict("mlx")
    assert not verdict.available
    assert verdict.reason == "needs Apple Silicon"


def test_detection_ollama_is_available_when_the_server_answers(monkeypatch):
    """Done-when 2: given Ollama answering on localhost, then ollama is available."""
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: True)
    assert _verdict("ollama").available


def test_detection_ollama_points_to_the_install_when_nothing_answers(no_ambient_ollama):
    """Done-when 2: given no answer, then ollama is unavailable with a pointer
    to install it, and the address that was tried."""
    verdict = _verdict("ollama")
    assert not verdict.available
    assert providers.OLLAMA_INSTALL in verdict.reason
    assert providers.OLLAMA_URL in verdict.reason


def test_ollama_address_is_one_constant_on_the_documented_default():
    """docs/faq.mdx in Ollama's repo: "Ollama binds 127.0.0.1 port 11434 by
    default." One constant, the one place the documented default is written:
    the default of `[model] ollama_url` (card #406), shared by the probe and
    the backend until a user names another address."""
    assert providers.OLLAMA_URL == "http://127.0.0.1:11434"


@pytest.mark.parametrize("engine", ["openai", "claude"])
def test_detection_cloud_engines_are_available_and_name_the_key_variable(
    engine, no_ambient_keys, no_ambient_ollama
):
    """Done-when 3: given any machine, then openai and claude are available and
    each says an API key is required, naming the variable it comes from."""
    verdict = _verdict(engine)
    assert verdict.available
    assert "API key required" in verdict.reason
    assert providers.KEY_VARIABLES[engine][0] in verdict.reason


def test_the_refusal_names_what_detection_says_is_available(monkeypatch):
    """One truth: the backends the refusal names as working here are the ones
    detection says are available, plus the test fake. Off Apple Silicon with
    Ollama answering, that is ollama, openai, claude, scripted, and not mlx."""
    fake_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: True)
    assert providers._works_here(providers.detect_engines()) == ("ollama", "openai", "claude", "scripted")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    assert "ollama, openai, claude, scripted" in str(err.value)


def test_default_engine_is_always_one_detection_names_available(
    monkeypatch, no_ambient_keys, no_ambient_ollama
):
    """`default_engine()` is the first verdict that is available, in the
    owner's order, and nothing else: the cloud engines are available
    everywhere, so there is always one, and there is no fallback that could
    quietly name an engine detection did not. Were the list ever to change so
    that nothing is available, the call raises rather than inventing mlx."""
    fake_platform(monkeypatch, "linux", "x86_64")
    verdicts = providers.detect_engines()
    assert providers.default_engine() == next(v.engine for v in verdicts if v.available) == "openai"

    nothing_available = [dataclasses.replace(v, available=False) for v in verdicts]
    monkeypatch.setattr(providers, "detect_engines", lambda ollama_at=None: nothing_available)
    with pytest.raises(StopIteration):
        providers.default_engine()


@contextlib.contextmanager
def _ollama_served_by(monkeypatch, handler: type[QuietHandler], prefix: str = ""):
    """`handler` on 127.0.0.1 at an ephemeral port, standing in for Ollama:
    detection is pointed at it for the block. `prefix` is a path in front of
    the endpoints, as a reverse proxy mounts Ollama under one, and is part
    of the address detection is pointed at."""
    with loopback_server(handler) as server:
        monkeypatch.setattr(
            providers, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}{prefix}")
        yield server


# Slack for thread scheduling in the deadline tests: the timer thread fires and
# the main thread's read returns some tens of milliseconds after the deadline,
# while waiting out the trickle takes over two seconds.
SCHEDULING_SLACK = 0.5


@contextlib.contextmanager
def _timed():
    """How long the block took, as `seconds` on the object yielded, read
    after the block: the one clock for every deadline assertion, whether
    the call inside returns or raises (a `pytest.raises` nested inside it
    swallows the raise before the clock stops)."""
    took = types.SimpleNamespace(seconds=None)
    started = time.monotonic()
    try:
        yield took
    finally:
        took.seconds = time.monotonic() - started


def _timed_probe(monkeypatch, handler: type[QuietHandler]) -> tuple[bool, float]:
    """The probe against `handler` standing in for Ollama: what it answered
    and how many seconds it took."""
    with _ollama_served_by(monkeypatch, handler), _timed() as took:
        answered = providers.ollama_answers()
    return answered, took.seconds


@contextlib.contextmanager
def _fake_ollama(
    monkeypatch, *, status: int = 200, delay: float = 0.0, replies: list[str] = (), prefix: str = ""
):
    """conftest's FakeOllama (Ollama's version and chat endpoints on
    127.0.0.1 at an ephemeral port) with detection pointed at it. `status`
    is what GET /api/version answers; `delay` holds the answer that long.
    `replies` are the texts POST /api/chat answers with, in order; every
    chat request's JSON body is kept on `server.chats`. `prefix` mounts both
    endpoints under a path, the way a reverse proxy does, and is part of
    the address detection is pointed at; any other path is Ollama's own
    404."""
    with FakeOllama(status=status, delay=delay, replies=replies, prefix=prefix).serve() as server:
        monkeypatch.setattr(providers, "OLLAMA_URL", f"{server.endpoint}{prefix}")
        yield server


def test_ollama_backend_returns_candidates_in_the_same_shape_as_mlx_on_the_fixture(
    monkeypatch, photos, tmp_path
):
    """Card #406, Done-when 1 and 3, at the real boundary: an HTTP server on
    loopback speaking Ollama's chat endpoint stands in for Ollama, no model
    and no network beyond 127.0.0.1. The factory builds the backend, the
    Identifier around it stages the committed fixture and asks the routing
    prompt then the bird prompt, and the result carries candidates in exactly
    the shape the same replies take through the mlx-shaped pipeline: same
    fields, ordered by confidence, attributed to the model name."""
    from melampus.identify import Identifier

    config = _cfg(model={"backend": "ollama"})
    with _fake_ollama(monkeypatch, replies=[ROUTING_OK, ID_OK]) as server:
        backend = providers.build_primary_backend(config)
        result = Identifier(backend, config).identify(photos / PHOTO)
    expected = Identifier(
        ScriptedBackend([ROUTING_OK, ID_OK], name="qwen3-vl:8b-instruct"), config
    ).identify(photos / PHOTO)

    assert result.status == "ok", result.error
    assert result.model == "qwen3-vl:8b-instruct"
    assert result.identification == expected.identification
    assert result.taxon_routing == expected.taxon_routing
    assert [c.common_name for c in result.identification.ranked()] == [
        "Tricolored Heron", "Little Blue Heron"]
    assert result.identification.top().scientific_name == "Egretta tricolor"

    routing, species = server.chats
    assert routing["model"] == species["model"] == "qwen3-vl:8b-instruct"
    assert routing["options"]["num_predict"] == config.model.routing_max_tokens
    assert species["options"]["num_predict"] == config.model.max_tokens
    for chat in (routing, species):
        (message,) = chat["messages"]
        (image,) = message["images"]
        assert base64.b64decode(image)[:2] == b"\xff\xd8", "the image is not the staged JPEG's bytes"
    assert "bird" in species["messages"][0]["content"].lower()
    assert PHOTO.split(".")[0] not in json.dumps(server.chats), "the filename travelled"


def _settings_naming_the_fake(monkeypatch, tmp_path):
    """Inside `_fake_ollama`: a settings file whose `[model] ollama_url` is the
    fake's address, with the default pointed at a closed port, so a test that
    passes the file proves the address came from it and not the default."""
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'[model]\nollama_url = "{providers.OLLAMA_URL}"\n', encoding="utf-8")
    monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{closed_port()}")
    return settings


def test_ollama_not_running_fires_before_any_image_is_read(monkeypatch, tmp_path, capsys):
    """Card #406, Done-when 2, at the real boundary: nothing listening on the
    port, and the folder's one image is a link to nowhere, so opening it
    would fail loudly. The CLI exits 3 on the not-running message, naming the
    address tried and the install pointer, and never mentions the file: the
    check ran before any image was read. `--no-local-config` keeps a
    developer's own `[model] ollama_url` in melampus.local.toml from being
    the address probed (Done-when 3)."""
    from melampus.cli import main

    port = closed_port()
    monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{port}")
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")

    code = main([
        str(folder), "--backend", "ollama", "--no-local-config",
        "--cache", str(tmp_path / "cache.jsonl"),
    ])

    err = capsys.readouterr().err
    assert code == 3, err
    assert f"No Ollama server at http://127.0.0.1:{port}" in err
    assert providers.OLLAMA_INSTALL in err
    for about_the_file in ("nowhere", "does-not-exist", "No such file", "unreadable"):
        assert about_the_file not in err, f"the image was touched before the Ollama check:\n{err}"


def test_cli_backend_ollama_writes_a_json_result_from_the_configured_address(
    monkeypatch, photos, tmp_path, capsys
):
    """Acceptance for Done-when 1: `melampus-id FOLDER --backend ollama` with
    the fake server's address as `[model] ollama_url` in a config file, on the
    committed fixture, runs the whole pipeline and writes a JSON result with
    the candidates, attributed to the model. The address comes from the
    config: detection's default is pointed at a closed port to prove it.
    `--no-local-config` keeps a developer's own `[model]` keys in
    melampus.local.toml (an `ollama_model`, say) out of what is asserted:
    `--config` overrides that file key by key, not whole (Done-when 3)."""
    from melampus.cli import main

    out = tmp_path / "results.json"
    with _fake_ollama(monkeypatch, replies=[ROUTING_OK, ID_OK]):
        settings = _settings_naming_the_fake(monkeypatch, tmp_path)
        code = main([
            str(photos), "--backend", "ollama", "--config", str(settings),
            "--no-local-config",
            "--cache", str(tmp_path / "cache.jsonl"), "--json-out", str(out),
        ])

    err = capsys.readouterr().err
    assert code == 0, err
    assert "loading qwen3-vl:8b-instruct" in err, err
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert result["file"] == PHOTO
    assert result["status"] == "ok"
    assert result["model"] == "qwen3-vl:8b-instruct"
    assert [c["common_name"] for c in result["identification"]["candidates"]] == [
        "Tricolored Heron", "Little Blue Heron"]


def test_cli_detection_probes_the_configured_ollama_address(monkeypatch, tmp_path, capsys, no_ambient_keys):
    """One source for the address: `[model] ollama_url` is what the default
    engine and `--detect-engines` probe too, not only the backend.
    `--no-local-config` says the intent: the config file, not a developer's
    melampus.local.toml, is the one source here (Done-when 3)."""
    from melampus.cli import main

    with _fake_ollama(monkeypatch) as server:
        settings = _settings_naming_the_fake(monkeypatch, tmp_path)
        assert main([
            "--detect-engines", "--config", str(settings), "--no-local-config",
        ]) == 0
    verdicts = {v["engine"]: v for v in json.loads(capsys.readouterr().out)}
    assert verdicts["ollama"]["available"] is True
    assert f"127.0.0.1:{server.server_port}" in verdicts["ollama"]["reason"]


def test_ollama_probe_finds_a_server_answering_on_localhost(monkeypatch):
    """Done-when 2 and 4, at the real boundary: an HTTP server on loopback
    standing in for Ollama, GET /api/version (Ollama's docs/api.md § Version)
    answering 200, and the probe says it is there. No network beyond 127.0.0.1."""
    with _fake_ollama(monkeypatch):
        assert providers.ollama_answers()
        assert _verdict("ollama").available


def test_ollama_probe_reports_a_closed_port_without_raising(monkeypatch):
    """Nothing listening: connection refused is "not installed or not
    running", never a traceback."""
    port = closed_port()
    monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{port}")
    assert providers.ollama_answers() is False
    verdict = _verdict("ollama")
    assert not verdict.available
    assert f"127.0.0.1:{port}" in verdict.reason


def test_ollama_probe_treats_a_non_200_as_unavailable(monkeypatch):
    with _fake_ollama(monkeypatch, status=503):
        assert providers.ollama_answers() is False


def test_ollama_probe_gives_up_after_its_timeout(monkeypatch):
    """A server that accepts and never answers must not stall detection: the
    probe waits OLLAMA_PROBE_SECONDS (one second in production) and reports
    unavailable."""
    monkeypatch.setattr(providers, "OLLAMA_PROBE_SECONDS", 0.2)
    with _fake_ollama(monkeypatch, delay=5.0):
        assert providers.ollama_answers() is False


class Trickling(QuietHandler):
    """A listener that sends a valid 200 with the headers trickled: each
    byte within the socket timeout, the whole well past the probe's
    deadline. A POST gets the same, once its body is read."""

    def do_GET(self):  # noqa: N802 - http.server's name
        with contextlib.suppress(OSError):
            self.wfile.write(b"HTTP/1.1 200 OK\r\n")
            trickle(self.wfile, b"Content-Length: 2\r\n\r\n")
            self.wfile.write(b"{}")

    def do_POST(self):  # noqa: N802 - http.server's name
        self.rfile.read(int(self.headers["Content-Length"]))
        self.do_GET()


def _trickling_body(status: int) -> type[QuietHandler]:
    """A listener that answers a POST's status line and headers at once,
    then a `status` body trickled: each byte within the socket timeout,
    the whole well past the deadline. 200 is a reply being read; 500 is an
    error body being read."""
    body = b'{"error": "slowly"}'

    class TricklingBody(QuietHandler):
        def do_POST(self):  # noqa: N802 - http.server's name
            self.rfile.read(int(self.headers["Content-Length"]))
            with contextlib.suppress(OSError):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                trickle(self.wfile, body)

    return TricklingBody


def test_ollama_probe_gives_up_at_its_deadline_when_the_headers_trickle(monkeypatch):
    """Security: OLLAMA_PROBE_SECONDS is a deadline on the whole probe, not on
    each read. A socket timeout is per operation, so whatever listens on the
    port when Ollama does not could send the status line and then one header
    byte every hundred milliseconds, each within the timeout, and hold
    detection, and the CLI's startup behind it, for as long as it liked. Given
    a server that trickles a valid 200 over two seconds, the probe reports
    unavailable and returns within its deadline, not after the trickle."""
    deadline = 0.3
    monkeypatch.setattr(providers, "OLLAMA_PROBE_SECONDS", deadline)
    answered, elapsed = _timed_probe(monkeypatch, Trickling)
    assert elapsed < deadline + SCHEDULING_SLACK, f"the probe read past its deadline: {elapsed:.2f}s"
    assert answered is False


def _hold_connect(monkeypatch, seconds: float) -> None:
    """The kernel's connect timing is not reproducible, so the TCP phase is
    held `seconds` here: HTTPConnection.connect sleeps that long, then
    connects as it would have."""
    connect = http.client.HTTPConnection.connect

    def held(connection):
        time.sleep(seconds)
        connect(connection)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", held)


def test_ollama_probe_gives_up_at_its_deadline_when_the_handshake_stalls_after_a_slow_connect(monkeypatch):
    """Codex round 3 (providers.py:139): the probe gave the deadline its
    socket once `connection.request()` had returned, and for https that is
    the TCP connection and then the TLS handshake, each bounded by the
    socket timeout on its own; a connection that took most of
    OLLAMA_PROBE_SECONDS, then a handshake that stalls, held detection a
    second whole probe timeout. Given a connect that returns just before
    the deadline (held 0.6s of 0.8s) and a listener behind an https address
    that accepts and never completes the handshake, the probe reports
    unavailable within its deadline."""
    deadline = 0.8
    monkeypatch.setattr(providers, "OLLAMA_PROBE_SECONDS", deadline)
    _hold_connect(monkeypatch, 0.6)
    with loopback_server(Silent) as server, _timed() as took:
        answered = providers.ollama_answers(f"https://127.0.0.1:{server.server_port}")
    assert took.seconds < deadline + SCHEDULING_SLACK, f"the probe shook hands past its deadline: {took.seconds:.2f}s"
    assert answered is False


def test_ollama_probe_gives_up_at_its_deadline_when_it_fires_during_connect(monkeypatch):
    """Security: the deadline must bound the probe whichever side of the
    connection it lands on. The timer hangs up the socket the deadline
    holds, and there is none while the probe is still connecting, so a
    deadline that fires then hangs up nothing; the socket is handed to the
    deadline the moment `_Noted` makes it, and `on()` hangs it up at once
    when the deadline has already passed, so the reads that follow a late
    connect never start, and a trickling listener cannot hold detection
    through them. The kernel's connect timing is not reproducible, so
    connect() is held past the deadline here. Given a connect that
    completes just after the deadline against a trickling server, the
    probe reports unavailable and returns within the deadline plus the
    hold, not after the trickle."""
    deadline = 0.3
    hold = 0.05  # how long past the deadline connect() is held
    monkeypatch.setattr(providers, "OLLAMA_PROBE_SECONDS", deadline)
    _hold_connect(monkeypatch, deadline + hold)
    answered, elapsed = _timed_probe(monkeypatch, Trickling)
    assert elapsed < deadline + hold + SCHEDULING_SLACK, f"the probe read past its deadline: {elapsed:.2f}s"
    assert answered is False


def test_ollama_probe_timeout_is_one_second():
    assert providers.OLLAMA_PROBE_SECONDS == 1.0


def test_hang_up_records_the_deadline_and_ends_the_stream_it_holds():
    """What `_hang_up` promises the deadline's timer, on the `_Deadline`
    it is given: with no socket yet (the caller is still connecting), the
    deadline is recorded as expired and nothing is touched, so `on()` can
    hang up as soon as there is a socket; with a socket, it is recorded
    and the stream is ended now, the other end reading end-of-stream; and
    a socket the main thread already closed (the probe's `finally`, or
    urllib once the headers are in) raises OSError, which is suppressed."""
    deadline = _Deadline(60.0)
    _hang_up(deadline, deadline.expired)
    assert deadline.expired.is_set()

    deadline = _Deadline(60.0)
    ours, theirs = socket.socketpair()
    with ours, theirs:
        deadline.on(ours)
        _hang_up(deadline, deadline.expired)
        assert deadline.expired.is_set()
        theirs.settimeout(1.0)
        assert theirs.recv(1) == b"", "the stream was not ended"

    deadline = _Deadline(60.0)
    closed = socket.socket()
    closed.close()
    deadline.on(closed)
    _hang_up(deadline, deadline.expired)
    assert deadline.expired.is_set()


def test_deadline_again_counts_the_bound_from_now():
    """Security (review round 4): a stream has no one exchange to bound, so
    `_Deadline.again` gives the next line the bound an exchange gets, from
    now. Given a deadline of one second armed, 0.6s in and again, it has
    not fired 0.6s after that (1.2s from the start: the first timer would
    have), and fires within the bound from the re-arm."""
    with _Deadline(1.0) as deadline:
        time.sleep(0.6)
        deadline.again()
        time.sleep(0.6)
        assert not deadline.expired.is_set(), "the deadline counted from the start, not from again()"
        assert deadline.expired.wait(2.0), "the deadline never fired after again()"


def test_ollama_probe_stays_on_loopback_whatever_proxy_the_environment_names(monkeypatch):
    """Security: the probe is a loopback call and must stay one. urlopen's
    default opener honours `http_proxy` (and, on a Mac, the system proxy
    settings, whose default bypass list does not cover 127.0.0.1), which
    would send the probe off the machine and let the proxy's answer stand in
    for Ollama's: a captive portal or a corporate proxy that answers 200 to
    anything would make detection report a server that is not there, and the
    default engine would follow it. Given a proxy in the environment that
    answers 200 to everything and nothing at OLLAMA_URL, the probe reports
    unavailable and the proxy never hears from it."""
    seen: list[str] = []
    monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{closed_port()}")
    with proxy_in_the_environment(monkeypatch, seen):
        assert providers.ollama_answers() is False
    assert seen == [], f"the probe left the machine through the proxy: {seen}"


def test_ollama_probe_refuses_a_redirect_off_loopback(monkeypatch):
    """Security: the probe asks one address and takes only that address's
    answer. build_opener installs HTTPRedirectHandler by default, so whatever
    listens on port 11434 (any local process can bind it when Ollama is not
    running) could answer 3xx with a Location anywhere, and the probe would
    make an outbound request there and let that server's 200 stand in for
    Ollama's. Given a server at OLLAMA_URL answering 302 towards a second
    server that records every request, the probe reports unavailable and the
    redirected destination never hears from it."""
    seen: list[str] = []
    with loopback_server(recording_handler(seen)) as destination:
        elsewhere = f"http://127.0.0.1:{destination.server_port}"
        with _ollama_served_by(monkeypatch, redirecting_handler(elsewhere)):
            answered = providers.ollama_answers()
    assert seen == [], f"the probe followed the redirect off loopback: {seen}"
    assert answered is False


def test_ollama_probe_asks_at_the_path_the_address_carries_as_the_backend_does(
    monkeypatch, tmp_path
):
    """Codex round 1 (providers.py:144): the probe read only the host and port
    of `[model] ollama_url` and asked /api/version at the root, while the
    backend POSTs to the address as typed plus /api/chat, so an Ollama behind
    a reverse proxy at `http://host/ollama` was refused as not running though
    every frame would have reached it. Given a fake Ollama mounted under
    /ollama, answering 404 anywhere else, detection at that address finds
    it, and the backend the factory builds from the same address posts to
    /ollama/api/chat: one address, read one way, from the probe to the
    frame."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with _fake_ollama(monkeypatch, replies=[ID_OK], prefix="/ollama") as server:
        address = providers.OLLAMA_URL
        assert address.endswith("/ollama")
        assert providers.ollama_answers(address) is True
        backend = providers.build_primary_backend(
            _cfg(model={"backend": "ollama", "ollama_url": address}))
        assert backend.complete(image, "prompt", 10).text == ID_OK
        assert len(server.chats) == 1


def test_ollama_probe_speaks_tls_for_an_https_address(monkeypatch):
    """Codex round 1 (providers.py:144), security: `https://host` is what a
    user types for an Ollama behind a TLS proxy, and the backend speaks TLS
    to it (urllib, certificate verified), but the probe dropped the scheme
    and asked port 80 in the clear. Given a plain HTTP server on loopback
    and an https address naming its port, the probe reports unavailable and
    the server never receives a request: a TLS handshake is not a GET, and
    the probe is not asking in plaintext."""
    seen: list[str] = []
    with loopback_server(recording_handler(seen)) as plain:
        answered = providers.ollama_answers(f"https://127.0.0.1:{plain.server_port}")
    assert answered is False
    assert seen == [], f"the probe asked an https address in plaintext: {seen}"


def test_ollama_probe_opens_a_verified_https_connection_for_an_https_address(monkeypatch):
    """The other half of the https regression: the probe does ask over TLS
    (not refuse https outright), at the address's host and its port as typed
    (none, here: HTTPSConnection's own 443), with http.client's default
    context, the one that verifies the certificate. Given the backend's
    HTTPS connection class faked at the http.client edge to answer 200
    without connecting, the probe reports the server found."""
    opened: list[tuple] = []

    class FakeHTTPS:
        sock = None

        def __init__(self, host, port=None, *, deadline, **kwargs):
            opened.append((host, port, kwargs))

        def request(self, method, path):
            opened.append((method, path))

        def getresponse(self):
            return types.SimpleNamespace(status=200)

        def close(self):
            return None

    monkeypatch.setattr(providers, "_NotedHTTPS", FakeHTTPS)
    assert providers.ollama_answers("https://ollama.example") is True
    (host, port, kwargs), asked = opened
    assert (host, port) == ("ollama.example", None)
    assert "context" not in kwargs, "the probe must verify the certificate: no context of its own"
    assert asked == ("GET", "/api/version")


# ---------------------------------------------------------------------------
# Card #404 through the CLI: `--detect-engines`, and the default engine.


def test_cli_detect_engines_prints_the_verdicts_as_json_in_order(
    monkeypatch, capsys, no_ambient_keys, no_ambient_ollama
):
    """Acceptance for Done-when 1 to 3: `melampus-id --detect-engines` needs no
    folder, prints one JSON list to stdout, in the owner's order, each item
    {engine, available, reason}, and exits 0. Faked off Apple Silicon with no
    Ollama: mlx and ollama say why not, the cloud engines say which key."""
    from melampus.cli import main

    fake_platform(monkeypatch, "win32", "AMD64")

    code = main(["--detect-engines"])

    out, err = capsys.readouterr()
    assert code == 0, err
    verdicts = json.loads(out)
    assert [v["engine"] for v in verdicts] == [*ENGINES, providers.CLAUDE_CODE, providers.CODEX]
    assert all(set(v) == {"engine", "title", "available", "reason"} for v in verdicts)
    assert all(v["title"] for v in verdicts), "a verdict with no title for the picker"
    by_engine = {v["engine"]: v for v in verdicts}
    assert by_engine["mlx"] == {
        "engine": "mlx", "title": "MLX — local, Apple Silicon", "available": False, "reason": "needs Apple Silicon"}
    assert by_engine["ollama"]["available"] is False
    assert providers.OLLAMA_INSTALL in by_engine["ollama"]["reason"]
    for engine in ("openai", "claude"):
        assert by_engine[engine]["available"] is True
        assert "API key required" in by_engine[engine]["reason"]
        assert providers.KEY_VARIABLES[engine][0] in by_engine[engine]["reason"]


def test_cli_detect_engines_reports_ollama_when_it_answers(monkeypatch, capsys):
    """The fake at the default address is what `--detect-engines` finds;
    `--no-local-config` keeps a developer's own `[model] ollama_url` in
    melampus.local.toml from being the address probed instead (Done-when 3)."""
    from melampus.cli import main

    with _fake_ollama(monkeypatch):
        assert main(["--detect-engines", "--no-local-config"]) == 0
    verdicts = {v["engine"]: v for v in json.loads(capsys.readouterr().out)}
    assert verdicts["ollama"]["available"] is True


def test_cli_detect_engines_reads_config_the_way_the_run_would(monkeypatch, tmp_path, capsys):
    """`--detect-engines` describes the run the same flags would make, so it
    reads config the same way: with `--no-local-config`, the `ollama_url` in
    melampus.local.toml is not probed (the default address is); without it,
    that address is the one probed. Otherwise the verdict and the run could
    disagree about where Ollama was looked for."""
    from melampus import config as config_module
    from melampus.cli import main

    local = tmp_path / "melampus.local.toml"
    local.write_text('[model]\nollama_url = "http://127.0.0.1:11436"\n', encoding="utf-8")
    monkeypatch.setattr(config_module, "_local_config", lambda: local)
    probed: list[str] = []

    def answers(url=None):
        probed.append(url)
        return False

    monkeypatch.setattr(providers, "ollama_answers", answers)

    assert main(["--detect-engines", "--no-local-config"]) == 0
    assert probed == [providers.OLLAMA_URL], f"the local config's address was probed: {probed}"
    assert main(["--detect-engines"]) == 0
    assert probed == [providers.OLLAMA_URL, "http://127.0.0.1:11436"], probed


def test_cli_still_requires_a_folder_without_detect_engines(capsys):
    from melampus.cli import main

    with pytest.raises(SystemExit) as exit_:
        main(["--backend", "scripted"])
    assert exit_.value.code == 2
    assert "folder" in capsys.readouterr().err


def _chosen_engine(monkeypatch, argv: list[str]) -> str:
    """Run the CLI to the backend seam and answer which engine it chose there;
    the seam refuses, so nothing loads or runs. `--no-local-config` keeps the
    developer's melampus.local.toml from setting the engine under a test about
    what happens when nothing sets it."""
    import melampus.cli

    chosen: list[str] = []

    def refuse(config):
        chosen.append(config.model.backend)
        raise providers.BackendUnavailable("stopped at the seam")

    monkeypatch.setattr(melampus.cli, "build_primary_backend", refuse)
    assert melampus.cli.main([*argv, "--no-local-config"]) == 3
    (engine,) = chosen
    return engine


def test_cli_default_engine_is_the_first_that_can_run_here(
    monkeypatch, photos, tmp_path, capsys, no_ambient_keys
):
    """No `--backend` and no `[model] backend`: the CLI picks the first engine
    detection says is available, in the owner's order. Off Apple Silicon with
    Ollama answering, that is ollama; the log line says so and how to choose."""
    fake_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: True)

    engine = _chosen_engine(monkeypatch, [str(photos), "--cache", str(tmp_path / "cache.jsonl")])

    assert engine == "ollama"
    err = capsys.readouterr().err
    assert "engine: ollama" in err and "--backend" in err, err


def test_cli_default_engine_is_mlx_on_apple_silicon_and_openai_with_nothing_local(
    monkeypatch, photos, tmp_path, no_ambient_keys, no_ambient_ollama
):
    argv = [str(photos), "--cache", str(tmp_path / "cache.jsonl")]
    fake_platform(monkeypatch, "darwin", "arm64")
    assert _chosen_engine(monkeypatch, argv) == "mlx"
    fake_platform(monkeypatch, "linux", "x86_64")
    assert _chosen_engine(monkeypatch, argv) == "openai"


def test_cli_detection_never_overrides_a_chosen_engine(
    monkeypatch, photos, tmp_path, no_ambient_keys
):
    """`--backend` and `[model] backend` are the user's word; detection only
    fills the blank. Faked so detection would say ollama."""
    fake_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: True)
    argv = [str(photos), "--cache", str(tmp_path / "cache.jsonl")]
    assert _chosen_engine(monkeypatch, [*argv, "--backend", "mlx"]) == "mlx"
    config = tmp_path / "settings.toml"
    config.write_text('[model]\nbackend = "claude"\n', encoding="utf-8")
    assert _chosen_engine(monkeypatch, [*argv, "--config", str(config)]) == "claude"


# ---------------------------------------------------------------------------
# Card #406: an Ollama backend behind the model seam.


def test_ollama_model_and_address_are_settings_in_the_model_section():
    """The model name and the server address are `[model]` settings, the way
    `repo` names the mlx model and `name` the cloud one. The model default is
    `qwen3-vl:8b-instruct` (ollama.com/library/qwen3-vl/tags: 6.1 GB, text and
    image input, the Instruct build of the family the mlx default uses). The
    address is unset by default, which means providers.OLLAMA_URL: the one
    place Ollama's documented default is written (card #404)."""
    config = _cfg()
    assert config.model.ollama_model == "qwen3-vl:8b-instruct"
    assert config.model.ollama_url is None
    assert providers.DEFAULT_MODELS.get("ollama") is None, "one default, in config"

    config = _cfg(model={"ollama_model": "qwen3-vl:30b-a3b-instruct",
                         "ollama_url": "http://127.0.0.1:11435"})
    assert config.model.ollama_model == "qwen3-vl:30b-a3b-instruct"
    assert config.model.ollama_url == "http://127.0.0.1:11435"


def test_ollama_address_setting_round_trips_through_a_config_file(tmp_path):
    settings = tmp_path / "settings.toml"
    settings.write_text(
        '[model]\nbackend = "ollama"\nollama_url = "http://127.0.0.1:11435"\n',
        encoding="utf-8",
    )
    config = load_config(settings, use_local=False)
    assert config.model.backend == "ollama"
    assert config.model.ollama_url == "http://127.0.0.1:11435"


class _FakeUrlopen:
    """Stands in for urllib.request.urlopen at the backend's HTTP edge: records
    every request, then answers with `reply` or raises `error`."""

    def __init__(self, reply: bytes = b"{}", error: Exception | None = None):
        self.reply, self.error = reply, error
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.reply)


OLLAMA_REPLY = json.dumps(ollama_chat_reply("qwen3-vl:8b-instruct", ID_OK)).encode("utf-8")


def _ollama_backend(client: _FakeUrlopen, **kwargs) -> OllamaBackend:
    return OllamaBackend("qwen3-vl:8b-instruct", "http://127.0.0.1:11435", client=client, **kwargs)


def test_ollama_backend_posts_the_chat_request_ollama_documents(tmp_path):
    """Request building, against Ollama's docs/api.md § Generate a chat
    completion: POST {url}/api/chat, JSON body with `model`, one user message
    carrying the prompt as `content` and the image as a base64 string in
    `images`, `stream` false so one object comes back, and `options` with
    `num_predict` (docs/modelfile.mdx: the maximum number of tokens to
    predict) and `temperature`. Only the staged file's bytes travel: no path,
    no filename."""
    image = tmp_path / "SECRET_SPECIES_NAME.jpg"
    image.write_bytes(b"\xff\xd8not really a jpeg\xff\xd9")
    client = _FakeUrlopen(OLLAMA_REPLY)
    backend = _ollama_backend(client, temperature=0.0, timeout=42.0)

    backend.complete(image, "what is in this image?", 900)

    ((request, timeout),) = client.requests
    assert request.full_url == "http://127.0.0.1:11435/api/chat"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 42.0
    body = json.loads(request.data)
    assert body == {
        "model": "qwen3-vl:8b-instruct",
        "messages": [{
            "role": "user",
            "content": "what is in this image?",
            "images": [base64.b64encode(image.read_bytes()).decode("ascii")],
        }],
        "stream": False,
        "options": {"num_predict": 900, "temperature": 0.0},
    }
    assert "SECRET_SPECIES_NAME" not in request.data.decode("utf-8")


def test_ollama_backend_reads_the_reply_into_a_completion(tmp_path):
    """Reply parsing: the text is `message.content`; the counts are
    `prompt_eval_count` and `eval_count` (docs/api.md, the final response
    object). The text is handed to the same JSON extraction and schema
    validation every backend's text goes through (identify.py); nothing here
    parses candidates."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _ollama_backend(_FakeUrlopen(OLLAMA_REPLY))

    completion = backend.complete(image, "prompt", 900)

    assert completion.text == ID_OK
    assert completion.prompt_tokens == 26
    assert completion.generated_tokens == 83
    assert completion.refused is False
    assert completion.seconds >= 0
    assert backend.name == "qwen3-vl:8b-instruct"


def test_ollama_backend_tolerates_a_reply_with_no_message(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _ollama_backend(_FakeUrlopen(b'{"done": true}'))
    assert backend.complete(image, "prompt", 10).text == ""


@pytest.mark.parametrize(
    ("error", "expected", "said"),
    [
        (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
         ConnectionError, "no Ollama server answering at http://127.0.0.1:11435"),
        (TimeoutError("timed out"), TimeoutError, "did not answer within 42s"),
        (urllib.error.URLError(socket.timeout("timed out")), TimeoutError, "did not answer within 42s"),
        (urllib.error.HTTPError("http://127.0.0.1:11435/api/chat", 404, "Not Found", {},
                                io.BytesIO(b'{"error": "model \'qwen3-vl:8b-instruct\' not found"}')),
         RuntimeError, "Ollama answered 404: model 'qwen3-vl:8b-instruct' not found"),
        (urllib.error.HTTPError("http://127.0.0.1:11435/api/chat", 500, "Internal Server Error", {},
                                io.BytesIO(b"not json")),
         RuntimeError, "Ollama answered 500: not json"),
    ],
    ids=["connection-refused", "timeout", "timeout-wrapped", "404-model-missing", "500-plain"],
)
def test_ollama_backend_maps_each_failure_to_a_plain_error(tmp_path, error, expected, said):
    """Error mapping: connection refused, a timeout (bare, or wrapped in
    URLError as urllib does), a non-200 (with Ollama's own `error` field when
    the body carries one), each raised as a plain message naming the address
    or the status, never a traceback into urllib. identify() turns any of
    them into an error result on that image and the batch continues."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _ollama_backend(_FakeUrlopen(error=error), timeout=42.0)
    with pytest.raises(expected) as err:
        backend.complete(image, "prompt", 10)
    assert said in str(err.value), str(err.value)


@pytest.mark.parametrize(
    "body, said",
    [
        (b"<html>proxy error</html>", "not JSON"),
        (b"[]", "not a JSON object"),
        (b'"text"', "not a JSON object"),
        (b'{"message": "just a string"}', "not a JSON object"),
    ],
    ids=["html", "list", "string", "message-not-an-object"],
)
def test_ollama_backend_reports_a_malformed_reply(tmp_path, body, said):
    """A 200 whose body is not JSON, is JSON but not an object, or whose
    `message` is not one, is a plain message naming the address and what
    came back, never an AttributeError out of `.get`; identify() records it
    on the frame and the batch continues."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _ollama_backend(_FakeUrlopen(body))
    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)
    assert said in str(err.value), str(err.value)
    assert backend.url in str(err.value), str(err.value)


def test_ollama_backend_stays_at_the_address_whatever_proxy_the_environment_names(
    monkeypatch, tmp_path
):
    """Security: readme.md § Windows promises that through Ollama nothing
    leaves the machine, and the probe (card #404) keeps that promise by
    consulting no proxy. The backend must keep it too: urlopen's default
    opener honours `http_proxy` (and, on a Mac, the system proxy settings,
    whose default bypass list does not cover 127.0.0.1), which would send
    every frame's bytes to the proxy and let the proxy's answer stand in for
    the model's. Given a proxy in the environment that answers 200 to
    everything and nothing at the address, the frame fails as not-running
    and the proxy never hears from it."""
    seen: list[str] = []
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with proxy_in_the_environment(monkeypatch, seen):
        backend = OllamaBackend(
            "qwen3-vl:8b-instruct", f"http://127.0.0.1:{closed_port()}", timeout=5.0)
        with pytest.raises(ConnectionError):
            backend.complete(image, "prompt", 10)
    assert seen == [], f"the frame left the machine through the proxy: {seen}"


def test_ollama_backend_refuses_a_redirect_off_the_address(tmp_path):
    """Security: the backend asks one address and takes only that address's
    answer, as the probe does. urlopen's default opener follows a 3xx, so
    whatever listens on the port when Ollama does not (any local process can
    bind it) could answer 302 with a Location anywhere, and the reply from
    there would stand in for the model's. Given a server at the address
    answering 302 towards a second server that records every request, the
    frame fails on the status and the destination never hears from it."""
    seen: list[str] = []
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with loopback_server(recording_handler(seen)) as destination:
        elsewhere = f"http://127.0.0.1:{destination.server_port}"
        with loopback_server(redirecting_handler(elsewhere)) as squatter:
            backend = OllamaBackend(
                "qwen3-vl:8b-instruct", f"http://127.0.0.1:{squatter.server_port}", timeout=5.0)
            with pytest.raises(RuntimeError) as err:
                backend.complete(image, "prompt", 10)
    assert "Ollama answered 302" in str(err.value), str(err.value)
    assert seen == [], f"the backend followed the redirect off the address: {seen}"


def test_https_contexts_verify_the_certificate_for_the_backend_and_the_probe():
    """The two verifying contexts the docstrings name, as they are: the
    backend's is urllib's HTTPSHandler's default, which `_Bounded` keeps and
    passes to `_NotedHTTPS`; the probe's is http.client's, which
    `_NotedHTTPS` makes for itself when given none. Both require the
    certificate and check the host name."""
    backend_context = _Bounded()._context
    probe_context = _NotedHTTPS("127.0.0.1", deadline=_Deadline(1.0))._context
    for context in (backend_context, probe_context):
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True


def test_ollama_backend_speaks_tls_for_an_https_address(tmp_path):
    """The backend's half of the https regression (Codex round 1,
    providers.py:144): an https address is spoken over TLS, never in the
    clear. Given a plain HTTP server on loopback and an https address naming
    its port, the frame fails as not answering and the server never receives
    a request."""
    seen: list[str] = []
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with loopback_server(recording_handler(seen)) as plain:
        backend = OllamaBackend(
            "qwen3-vl:8b-instruct", f"https://127.0.0.1:{plain.server_port}", timeout=5.0)
        with pytest.raises(ConnectionError):
            backend.complete(image, "prompt", 10)
    assert seen == [], f"the backend asked an https address in plaintext: {seen}"


@pytest.mark.parametrize(
    "handler", [Trickling, _trickling_body(200), _trickling_body(500)],
    ids=["headers", "body", "error-body"],
)
def test_ollama_backend_gives_up_at_its_deadline_when_the_server_trickles(tmp_path, handler):
    """Codex round 1 (backend.py:419), security: `timeout_seconds` is
    documented as the per-request ceiling, but urlopen's `timeout` is the
    socket's, bounding each operation and resetting on every read, so a
    server sending one byte within the timeout, then another, could hold a
    frame, and the batch behind it, for as long as it liked; the byte limit
    bounds how much, not how long. Given a server on loopback trickling a
    valid answer over two seconds, in the headers, in a 200 body, or in a
    500 body (the error text the backend reads for its message), and a
    timeout of 0.3s, the frame fails as timed out within the deadline, not
    after the trickle."""
    deadline = 0.3
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with loopback_server(handler) as server:
        backend = OllamaBackend(
            "qwen3-vl:8b-instruct", f"http://127.0.0.1:{server.server_port}", timeout=deadline)
        with _timed() as took, pytest.raises(TimeoutError) as err:
            backend.complete(image, "prompt", 10)
    assert took.seconds < deadline + SCHEDULING_SLACK, f"the frame read past its deadline: {took.seconds:.2f}s"
    assert f"did not answer within {deadline:g}s" in str(err.value), str(err.value)


def test_ollama_backend_stream_gives_up_at_its_deadline_when_a_line_trickles():
    """Security (review round 4, download.py:784): the pull's stream was read
    a line at a time with urlopen's socket timeout alone, which bounds each
    read and resets on every byte, so a listener writing a line a byte at a
    time within it held the pull, and the cancel marker read between lines,
    for as long as it liked (a probe: the marker written one second into a
    trickled line was read eight seconds later, at its newline). `stream`
    gives each line what `send` gives an exchange, the deadline armed again
    for it. Given a server writing two whole lines, each after a pause
    within the deadline and the two together past it, then a line trickled
    well past it, the stream yields both whole lines (the deadline counts
    per line, not per exchange) and ends as timed out within a deadline of
    the trickle's start, not at its newline."""
    deadline = 0.5
    with loopback_server(TricklingPull) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        backend = OllamaBackend("qwen3-vl:8b-instruct", url, timeout=deadline)
        request = urllib.request.Request(f"{url}/api/pull", data=b"{}", method="POST")
        lines = []
        with _timed() as took, pytest.raises(TimeoutError) as err:
            for line in backend.stream(request):
                lines.append(line)
    assert lines == [TricklingPull.WHOLE, TricklingPull.WHOLE]
    ceiling = 2 * TricklingPull.PAUSE + deadline + SCHEDULING_SLACK
    assert took.seconds < ceiling, f"the stream read past its deadline: {took.seconds:.2f}s"
    assert f"did not answer within {deadline:g}s" in str(err.value), str(err.value)


@pytest.mark.parametrize(
    ("hold", "ceiling"),
    [(0.85, 0.85), (0.6, 0.8)],
    ids=["connect-after-deadline", "connect-just-before"],
)
def test_ollama_backend_deadline_covers_the_handshake_whenever_connect_lands(
    monkeypatch, tmp_path, hold, ceiling
):
    """Codex round 2 (backend.py:406) and round 3 (backend.py:426), security:
    for https connect() is the TCP connection and then the TLS handshake,
    each bounded by the socket timeout on its own, and the deadline could
    not touch the handshake: first because `_Noted` handed the socket over
    only once connect() had returned, then because SSLContext.wrap_socket
    detaches the socket it was handed (its descriptor moves to the new
    SSLSocket) before the handshake runs inside it. Either way a
    connection that took most of the budget, or landed just after it, and
    then a handshake that stalls, held the frame a whole socket timeout
    more. The kernel's connect timing is not reproducible, so the TCP phase
    is held here, as the probe's tests hold it. Given a 0.8s timeout, a
    connect held `hold` seconds (just past the deadline, or 0.6s of it) and
    a listener behind an https address that accepts and never completes
    the handshake, the frame fails as timed out by `ceiling` (the deadline,
    or the late connect's own moment), not a whole timeout later."""
    deadline = 0.8
    _hold_connect(monkeypatch, hold)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with loopback_server(Silent) as server:
        backend = OllamaBackend(
            "qwen3-vl:8b-instruct", f"https://127.0.0.1:{server.server_port}", timeout=deadline)
        with _timed() as took, pytest.raises(TimeoutError) as err:
            backend.complete(image, "prompt", 10)
    assert took.seconds < ceiling + SCHEDULING_SLACK, f"the handshake ran past the deadline: {took.seconds:.2f}s"
    assert f"did not answer within {deadline:g}s" in str(err.value), str(err.value)


@pytest.mark.parametrize(
    "address", ["localhost:11434", "http://", "not an address", "http://127.0.0.1:99999"],
    ids=["no-scheme", "no-host", "not-a-url", "port-out-of-range"],
)
def test_ollama_probe_reports_unavailable_for_an_address_it_cannot_ask(address, no_ambient_keys):
    """`[model] ollama_url` is the user's typing, and the probe promises never
    to raise: a scheme left off, a host left out, a port out of range is an
    address no server answers at, reported as such, not a traceback out of
    detection, the default engine or `--detect-engines`. Asked for the
    backend there, the refusal names the address as typed (trailing slashes
    dropped, as providers.ollama_url documents)."""
    assert providers.ollama_answers(address) is False
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg(model={"backend": "ollama", "ollama_url": address}))
    assert f"No Ollama server at {address.rstrip('/')}" in str(err.value), str(err.value)


def test_ollama_backend_bounds_what_it_reads_of_an_error_body(tmp_path):
    """A non-200's body is the server's words, and they land in the frame's
    error record (identify.py), so in the cache and in --json-out: a proxy's
    error page or a squatter's megabyte must not land there whole. At most
    OllamaBackend.MAX_ERROR_BYTES of it are read; Ollama's own errors are
    one line."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    error = urllib.error.HTTPError(
        "http://127.0.0.1:11435/api/chat", 502, "Bad Gateway", {}, io.BytesIO(b"<p>" * (1 << 20)))
    backend = _ollama_backend(_FakeUrlopen(error=error))
    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)
    assert "Ollama answered 502" in str(err.value)
    assert len(str(err.value)) <= OllamaBackend.MAX_ERROR_BYTES + 40, len(str(err.value))


@pytest.mark.parametrize(
    ("error", "reason", "body", "said"),
    [
        ("\x1b[2K\rmodel 'x' not found\x07", "Not Found", None, "[2K model 'x' not found"),
        (None, "Bad Gateway", b"<p>\x1b[31mproxy\x00error\r\nline two</p>", "<p> [31mproxy error line two</p>"),
        (None, "Boom\x1b[0m\r\nfake log line", b"", "Boom [0m fake log line"),
    ],
    ids=["ollama-error-field", "plain-body", "status-reason"],
)
def test_ollama_backend_neutralises_control_characters_in_a_servers_error_text(
    tmp_path, error, reason, body, said
):
    """Codex round 1 (backend.py:451), security: a non-200's text is the
    server's, and it lands in the frame's error record, the log and the
    terminal. Escape sequences and control characters in it would move the
    cursor, recolour the terminal, erase a line, or fake a line of the log.
    Given Ollama's `error` field, a plain body, or the status line's own
    reason carrying ESC, BEL, NUL and line breaks, the message keeps the
    words and none of the control characters: each becomes a space, runs
    of whitespace collapse to one (what an escape sequence leaves behind,
    `[2K`, is words), and the status is still named."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    if body is None:
        body = json.dumps({"error": error}).encode("utf-8")
    exc = urllib.error.HTTPError("http://127.0.0.1:11435/api/chat", 502, reason, {}, io.BytesIO(body))
    backend = _ollama_backend(_FakeUrlopen(error=exc))
    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)
    message = str(err.value)
    assert message == f"Ollama answered 502: {said}", message
    assert all(c.isprintable() for c in message), message


def test_ollama_backend_reports_a_malformed_status_line_in_printable_words(tmp_path):
    """Codex round 2 (backend.py:545), security: a status line http.client
    cannot parse raises `BadStatusLine` carrying the line as sent, and it
    passed through `_exchange` uncaught, so its ESC and CR reached the
    frame's error record and the terminal report unchanged, past the rule
    round-1 finding 3 put on error bodies. Given a listener on loopback
    answering a status line with an escape sequence and a carriage return,
    the frame fails as a plain error naming the address and the line in
    printable characters only."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with loopback_server(BadStatusLine) as server:
        backend = OllamaBackend(
            "qwen3-vl:8b-instruct", f"http://127.0.0.1:{server.server_port}", timeout=5.0)
        with pytest.raises(RuntimeError) as err:
            backend.complete(image, "prompt", 10)
    message = str(err.value)
    assert message == (
        f"Ollama's reply from {backend.url} was not HTTP: [31mHTTP/9.9 OK fake log line"
    ), message
    assert all(c.isprintable() for c in message), message


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (http.client.BadStatusLine("\x1b[2K\rnot http\n"), "[2K not http"),
        (http.client.RemoteDisconnected("Remote end closed connection without response"),
         "Remote end closed connection without response"),
        (http.client.IncompleteRead(b"\x1b[31m", 40), "IncompleteRead(5 bytes read, 40 more expected)"),
        (http.client.LineTooLong("status line"), "got more than 65536 bytes when reading status line"),
        (http.client.BadStatusLine("x" * 5000), "x" * OllamaBackend.MAX_ERROR_BYTES),
    ],
    ids=["bad-status-line", "hung-up", "incomplete-read", "line-too-long", "bounded"],
)
def test_ollama_backend_maps_each_protocol_error_to_a_plain_bounded_message(tmp_path, error, said):
    """The same, at the urlopen edge, for each of http.client's protocol
    errors urllib lets through unwrapped (a status line that is not HTTP,
    a server hanging up before one, a body cut short, a line past
    http.client's limit): a plain RuntimeError naming the address, the
    words printable only and at most MAX_ERROR_BYTES of them, never an
    http.client exception into identify()."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _ollama_backend(_FakeUrlopen(error=error))
    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)
    assert str(err.value) == f"Ollama's reply from {backend.url} was not HTTP: {said}", str(err.value)


def test_ollama_backend_bounds_what_it_reads_of_a_reply(tmp_path):
    """The reply is one object read into memory whole (`stream` false); a
    server that keeps sending must not fill it. Ollama's reply is the text of
    at most `num_predict` tokens and a dozen counters, so one past
    OllamaBackend.MAX_REPLY_BYTES is refused as too long, not parsed."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    reply = b'{"message": {"content": "' + b"a" * OllamaBackend.MAX_REPLY_BYTES + b'"}}'
    backend = _ollama_backend(_FakeUrlopen(reply))
    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)
    assert str(OllamaBackend.MAX_REPLY_BYTES) in str(err.value), str(err.value)


# ---------------------------------------------------------------------------
# Card #420: a command backend behind the model seam, driving an installed CLI.


COMMAND = ["fake-vlm", "--image", "{image}", "--prompt", "{prompt}", "--quiet"]


def test_command_is_a_setting_in_the_model_section(tmp_path):
    """`[model] command` is the program to run, as a list of arguments with
    `{image}` and `{prompt}` placeholders: an argv list, never a shell string,
    so a prompt with spaces, quotes or newlines is one argument and nothing is
    quoted. Unset by default: no CLI is assumed installed. The reply is read
    from stdout; the per-request ceiling is the existing `timeout_seconds`."""
    assert _cfg().model.command == []

    settings = tmp_path / "settings.toml"
    settings.write_text(
        '[model]\nbackend = "command"\n'
        'command = ["fake-vlm", "--image", "{image}", "--prompt", "{prompt}", "--quiet"]\n'
        "timeout_seconds = 30\n",
        encoding="utf-8",
    )
    config = load_config(settings, use_local=False)
    assert config.model.backend == "command"
    assert config.model.command == COMMAND
    assert config.model.timeout_seconds == 30.0


@pytest.mark.parametrize(
    ("command", "missing"),
    [
        (["fake-vlm", "--prompt", "{prompt}"], "{image}"),
        (["fake-vlm", "--image", "{image}"], "{prompt}"),
        (["fake-vlm"], "{image}"),
    ],
    ids=["no-image", "no-prompt", "neither"],
)
def test_command_template_without_a_placeholder_is_refused_at_config_load(command, missing):
    """A template that never receives the image, or never asks the question,
    cannot answer anything: refused when the config loads, naming the
    placeholder it lacks, not per frame after the run has started."""
    with pytest.raises(ValueError) as err:
        _cfg(model={"backend": "command", "command": command})
    assert missing in str(err.value), str(err.value)


class _FakeRun:
    """Stands in for subprocess.run at the backend's process edge: records
    every call, then returns a completed process with `stdout`, `stderr` and
    `returncode`, or raises `error`."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0,
                 error: Exception | None = None):
        self.stdout, self.stderr, self.returncode, self.error = stdout, stderr, returncode, error
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)


def _command_backend(run: _FakeRun, command: list[str] = COMMAND, **kwargs) -> CommandBackend:
    return CommandBackend(command, executable="/opt/fake/bin/fake-vlm", run=run, **kwargs)


def test_command_backend_expands_the_template_into_one_argv(tmp_path):
    """Template expansion: each argument with `{image}` gets the image path as
    given (absolute, the staged file), each with `{prompt}` the prompt in
    full as one argument, spaces, quotes and newlines included, and every
    other argument is passed untouched. The resolved executable stands in
    for the bare name (shutil.which found it; on Windows that is how a
    `.cmd` shim runs without a shell). subprocess.run is given the list, no
    shell, the reply as text, the config's timeout, stdout and stderr
    captured, and nothing on stdin, so a program that reads it cannot hang."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK)
    backend = _command_backend(run, timeout=42.0)
    prompt = 'Identify the "bird".\n\nReply with JSON: {"taxon": ...}'

    backend.complete(image, prompt, 900)

    ((argv, kwargs),) = run.calls
    assert argv == ["/opt/fake/bin/fake-vlm", "--image", str(image), "--prompt", prompt, "--quiet"]
    assert kwargs["timeout"] == 42.0
    assert kwargs["text"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs.get("shell", False) is False
    assert "env" not in kwargs, "the child gets the parent's environment as it is; nothing is added"


def test_command_backend_expands_a_placeholder_inside_a_longer_argument(tmp_path):
    """`--image={image}` is one argument too: the placeholder is replaced
    wherever it sits, so a CLI that takes `--flag=value` works."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK)
    backend = _command_backend(run, ["fake-vlm", "--image={image}", "--prompt={prompt}"])

    backend.complete(image, "what is this?", 10)

    ((argv, _),) = run.calls
    assert argv == ["/opt/fake/bin/fake-vlm", f"--image={image}", "--prompt=what is this?"]


def test_command_backend_reads_stdout_into_a_completion(tmp_path):
    """Reply parsing: stdout is the text, handed as it came to the same JSON
    extraction and schema validation every backend's text goes through
    (identify.py); nothing here parses candidates. No token counts: a
    command reports none. The name is the template, so a changed flag is a
    changed run fingerprint and the cache cannot re-serve the old template's
    answers (architecture.md § Caching and resume)."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _command_backend(_FakeRun(stdout=f"Here you go:\n{ID_OK}\n", stderr="warning: slow"))

    completion = backend.complete(image, "prompt", 900)

    assert completion.text == f"Here you go:\n{ID_OK}\n"
    assert completion.prompt_tokens is None and completion.generated_tokens is None
    assert completion.refused is False
    assert completion.seconds >= 0
    assert backend.name == "fake-vlm --image {image} --prompt {prompt} --quiet"


@pytest.mark.parametrize(
    ("run", "expected", "said"),
    [
        (_FakeRun(returncode=2, stderr="not logged in\nrun `fake-vlm login` first\nmore\nand more"),
         CommandFailed, "fake-vlm exited 2: not logged in / run `fake-vlm login` first / more"),
        (_FakeRun(returncode=1), CommandFailed, "fake-vlm exited 1 with nothing on stderr"),
        (_FakeRun(error=subprocess.TimeoutExpired(["fake-vlm"], 42.0)),
         TimeoutError, "fake-vlm did not answer within 42s"),
        (_FakeRun(stdout="  \n", stderr="usage: fake-vlm ..."),
         RuntimeError, "fake-vlm printed nothing on stdout: usage: fake-vlm ..."),
        (_FakeRun(error=PermissionError(13, "Permission denied")),
         RuntimeError, "fake-vlm could not be run: [Errno 13] Permission denied"),
    ],
    ids=["non-zero", "non-zero-silent", "timeout", "empty-stdout", "not-runnable"],
)
def test_command_backend_maps_each_failure_to_a_plain_error(tmp_path, run, expected, said):
    """Error mapping: a non-zero exit is CommandFailed naming the command,
    the exit code and the first lines of stderr (the CLI surfaces it at
    exit 3, the way the other backend failures are); a timeout is
    TimeoutError naming the ceiling to raise; empty stdout and a program
    that cannot be started are plain RuntimeErrors naming the command,
    recorded on the frame while the batch goes on. Never a traceback into
    subprocess."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _command_backend(run, timeout=42.0)
    with pytest.raises(expected) as err:
        backend.complete(image, "prompt", 10)
    assert said in str(err.value), str(err.value)
    assert not isinstance(err.value, subprocess.SubprocessError)


def test_command_is_selectable_by_config_and_flag_but_not_a_picker_choice():
    """`[model] backend = "command"` and `--backend command` select the seam;
    the plugin's picker learns it in card #423, so BACKEND_CHOICES, the
    engines the picker offers in the owner's order, is unchanged. It is
    local: no cloud retuning, no cost prompt, no cloud cache file."""
    assert providers.COMMAND == "command"
    assert providers.BACKEND_CHOICES == (*ENGINES, providers.SCRIPTED)
    assert providers.COMMAND in providers.LOCAL_BACKENDS
    assert not providers.is_cloud_primary(_cfg(model={"backend": "command", "command": COMMAND}))


def test_cli_accepts_backend_command(photos, tmp_path, capsys):
    from melampus.cli import main

    code = main([str(photos), "--backend", "command", "--report-only",
                 "--cache", str(tmp_path / "cache.jsonl")])

    assert code == 0, capsys.readouterr().err


def test_command_not_configured_is_refused_with_the_shape(no_ambient_ollama):
    """Engine command with no `[model] command`: refused through the same
    BackendUnavailable path every refusal takes, saying what to set."""
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg(model={"backend": "command"}))
    message = str(err.value)
    assert "[model] command" in message and "{image}" in message and "{prompt}" in message
    assert "--backend" in message


def test_command_not_installed_is_refused_before_any_image_is_read_and_names_the_fix(
    monkeypatch, no_ambient_keys, no_ambient_ollama
):
    """Card #420, Done-when 2: given the command is missing, when the backend
    is asked for, then the refusal names the command, says it is not
    installed or not on PATH and how to fix that in general words (the
    specific CLI's install pointer is #421/#422), and names the backends
    that do work here, through BackendUnavailable (exit 3 from the CLI).
    shutil.which runs in the factory: no image is read first."""
    asked: list[str] = []

    def which(name):
        asked.append(name)
        return None

    monkeypatch.setattr(providers.shutil, "which", which)
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg(model={"backend": "command", "command": COMMAND}))
    message = str(err.value)
    assert asked == ["fake-vlm"]
    assert "'fake-vlm' is not installed or not on PATH" in message
    assert "install" in message.lower() and "PATH" in message
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in message, f"{works_here!r} is not named as working here:\n{message}"
    assert "--backend" in message


def test_command_primary_builds_the_backend_from_the_model_settings(monkeypatch):
    """Given engine command and a template, the factory builds a
    CommandBackend on the template, the path shutil.which resolved its
    first element to, and timeout_seconds."""
    monkeypatch.setattr(providers.shutil, "which", lambda name: f"/opt/fake/bin/{name}")
    backend = providers.build_primary_backend(
        _cfg(model={"backend": "command", "command": COMMAND, "timeout_seconds": 30}))
    assert isinstance(backend, CommandBackend)
    assert backend.command == COMMAND
    assert backend.executable == "/opt/fake/bin/fake-vlm"
    assert backend.timeout == 30.0
    assert backend.name == " ".join(COMMAND)


FAKE_CLI = "fake-vlm"

_FAKE_CLI_SCRIPT = '''#!{python}
"""A stand-in for a subscription CLI: takes an image and a prompt, prints a
reply on stdout in the shape the real models produce. Checks what it was
given the way a real program would notice: the image must exist, the prompt
must not be empty."""
import argparse
import os
import sys

ROUTING = {routing!r}
IDENTIFICATION = {identification!r}

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()
if not os.path.isfile(args.image):
    sys.exit(f"no such image: {{args.image}}")
if not args.prompt.strip():
    sys.exit("empty prompt")
if not os.path.isabs(args.image):
    sys.exit(f"image path is not absolute: {{args.image}}")
if {exit_code} != 0:
    print({stderr!r}, file=sys.stderr)
    sys.exit({exit_code})
print("Sure, here is the identification:")
print(ROUTING if "router" in args.prompt else IDENTIFICATION)
'''


def _fake_cli(monkeypatch, tmp_path, *, exit_code: int = 0, stderr: str = "") -> list[str]:
    """Write FAKE_CLI, an executable Python script, into a folder put first
    on PATH, and return the config template that runs it by its bare name:
    the real shutil.which, the real subprocess, no shell. `exit_code`
    non-zero makes it fail after reading its arguments, saying `stderr`."""
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    script = folder / FAKE_CLI
    script.write_text(_FAKE_CLI_SCRIPT.format(
        python=sys.executable, routing=ROUTING_OK, identification=ID_OK,
        exit_code=exit_code, stderr=stderr,
    ), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    assert shutil.which(FAKE_CLI) == str(script)
    return [FAKE_CLI, "--image", "{image}", "--prompt", "{prompt}", "--quiet"]


def _command_settings(tmp_path, command: list[str]) -> Path:
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'[model]\nbackend = "command"\ncommand = {json.dumps(command)}\n', encoding="utf-8")
    return settings


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the fake CLI is a shebang script")


@posix_only
def test_command_backend_returns_candidates_in_the_same_shape_as_mlx_on_the_fixture(
    monkeypatch, photos, tmp_path
):
    """Card #420, Done-when 1 and 3, at the real boundary: an executable
    script on PATH stands in for the CLI, resolved by the real shutil.which
    and run by the real subprocess, no shell; nothing real is installed or
    called. It reads its arguments, checks the image exists and the prompt
    is non-empty, and answers the routing prompt then the bird prompt in
    the shape the real models produce. The factory builds the backend, the
    Identifier stages the committed fixture, and the result carries
    candidates in exactly the shape the same replies take through the
    mlx-shaped pipeline: same fields, ordered by confidence."""
    from melampus.identify import Identifier

    command = _fake_cli(monkeypatch, tmp_path)
    config = _cfg(model={"backend": "command", "command": command})
    backend = providers.build_primary_backend(config)
    result = Identifier(backend, config).identify(photos / PHOTO)
    expected = Identifier(
        ScriptedBackend([ROUTING_OK, ID_OK], name=" ".join(command)), config
    ).identify(photos / PHOTO)

    assert result.status == "ok", result.error
    assert result.model == " ".join(command)
    assert result.identification == expected.identification
    assert result.taxon_routing == expected.taxon_routing
    assert [c.common_name for c in result.identification.ranked()] == [
        "Tricolored Heron", "Little Blue Heron"]
    assert result.identification.top().scientific_name == "Egretta tricolor"


def test_command_not_installed_fires_before_any_image_is_read(tmp_path, capsys, no_ambient_ollama):
    """Card #420, Done-when 2, at the real boundary: a command nothing on
    this machine is called, and the folder's one image is a link to
    nowhere, so opening it would fail loudly. The CLI exits 3 on the
    not-installed message, naming the command, and never mentions the file:
    the check ran before any image was read."""
    from melampus.cli import main

    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")
    settings = _command_settings(tmp_path, ["melampus-no-such-command-420", "{image}", "{prompt}"])

    code = main([str(folder), "--config", str(settings), "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "'melampus-no-such-command-420' is not installed or not on PATH" in err
    for about_the_file in ("nowhere", "does-not-exist", "No such file", "unreadable"):
        assert about_the_file not in err, f"the image was touched before the command check:\n{err}"


@posix_only
def test_cli_backend_command_exits_3_when_the_command_exits_non_zero(
    monkeypatch, photos, tmp_path, capsys
):
    """Card #420, Done-when 2: given the command exits non-zero, when the
    service analyzes, then the run stops at exit 3 on a message naming the
    command, its exit code and what it said on stderr, like the other
    backend failures, rather than recording the same failure on every
    frame in turn. Nothing is cached for the frame in flight."""
    from melampus.cli import main

    command = _fake_cli(monkeypatch, tmp_path, exit_code=2, stderr="not signed in\nrun fake-vlm login")
    out = tmp_path / "results.json"

    code = main([str(photos), "--backend", "command", "--config", str(_command_settings(tmp_path, command)),
                 "--cache", str(tmp_path / "cache.jsonl"), "--json-out", str(out)])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "fake-vlm exited 2: not signed in / run fake-vlm login" in err
    assert not out.exists()
    assert not (tmp_path / "cache.jsonl").exists(), "the failed frame was cached"


@posix_only
def test_cli_backend_command_writes_a_json_result_from_the_configured_template(
    monkeypatch, photos, tmp_path, capsys
):
    """Acceptance for Done-when 1: `melampus-id FOLDER --config FILE` with
    `[model] backend = "command"` and the fake CLI's template in the file,
    on the committed fixture, runs the whole pipeline and writes a JSON
    result with the candidates, attributed to the template."""
    from melampus.cli import main

    command = _fake_cli(monkeypatch, tmp_path)
    out = tmp_path / "results.json"

    code = main([str(photos), "--config", str(_command_settings(tmp_path, command)),
                 "--cache", str(tmp_path / "cache.jsonl"), "--json-out", str(out)])

    err = capsys.readouterr().err
    assert code == 0, err
    assert f"loading {' '.join(command)}" in err, err
    assert "cloud default" not in err, f"command is local; nothing was retuned for a cloud:\n{err}"
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert result["file"] == PHOTO
    assert result["status"] == "ok"
    assert result["model"] == " ".join(command)
    assert [c["common_name"] for c in result["identification"]["candidates"]] == [
        "Tricolored Heron", "Little Blue Heron"]


# --- card #421: Claude Code as an engine ------------------------------------

CLAUDE = "claude"

#: The routing reply Claude Code gave on the committed fixture in the one real
#: proof run (2.1.277, 2026-09-18), as the `result` field carried it: fenced.
_CLAUDE_RESULT = "```json\n{}\n```"

_FAKE_CLAUDE_SCRIPT = '''#!{python}
"""Stands in for Claude Code 2.1.277's documented non-interactive interface,
as `claude --help`, `claude auth status --help` and code.claude.com/docs/en/
headless describe it: `claude -p [flags] "prompt"` prints one result and
exits; `--output-format json` wraps it as a JSON object whose `result` is
the text and whose `is_error` says whether the run failed; a failure inside
the run, such as missing authentication, is printed as the result on stdout
with a non-zero exit and nothing on stderr; `claude auth status` exits 0
when signed in and 1 when not, `--json` carrying `loggedIn`. The image is
a file the prompt names, read by the Read tool, which needs no prompt only
when `--allowedTools Read` pre-approves it. MODE: "signed-in" answers;
"not-signed-in" fails the status check and every run the documented way;
"expired" passes the status check and fails the run, the way a session
that lapses mid-batch would; "hung" never answers the status check."""
import json
import os
import re
import sys

MODE = {mode!r}
ROUTING = {routing!r}
IDENTIFICATION = {identification!r}
LOG = {log!r}
NOT_LOGGED_IN = "Not logged in \\u00b7 Please run /login"

argv = sys.argv[1:]
with open(LOG, "a", encoding="utf-8") as log:
    log.write(json.dumps({{"argv": argv, "cwd": os.getcwd()}}) + "\\n")

if argv[:2] == ["auth", "status"]:
    if MODE == "hung":
        import time
        time.sleep(30)
    logged_in = MODE != "not-signed-in"
    if "--text" in argv:
        print("Login method: Claude Max account" if logged_in
              else "Not logged in. Run claude auth login to authenticate.")
    else:
        status = {{"loggedIn": logged_in, "authMethod": "claude.ai" if logged_in else "none",
                  "apiProvider": "firstParty"}}
        if logged_in:
            status["subscriptionType"] = "max"
        print(json.dumps(status, indent=2))
    sys.exit(0 if logged_in else 1)

import argparse

ap = argparse.ArgumentParser(prog="claude")
ap.add_argument("-p", "--print", action="store_true")
ap.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
ap.add_argument("--tools", default="default")
ap.add_argument("--allowedTools", "--allowed-tools", default="")
ap.add_argument("--permission-prompts", choices=["host", "none"], default="host")
ap.add_argument("--no-session-persistence", action="store_true")
ap.add_argument("--strict-mcp-config", action="store_true")
ap.add_argument("--setting-sources", default="user,project,local")
ap.add_argument("prompt")
args = ap.parse_args()
if not args.print:
    sys.exit("an interactive session needs a terminal; use -p")
for source in args.setting_sources.split(","):
    if source not in ("user", "project", "local"):
        sys.exit(f"Error processing --setting-sources: Invalid setting source: {{source}}. "
                 "Valid options are: user, project, local")


def result(text, is_error=False):
    if args.output_format == "json":
        print(json.dumps({{"type": "result", "subtype": "success", "is_error": is_error,
                          "result": text, "session_id": "00000000-0000-0000-0000-000000000000",
                          "num_turns": 1 if is_error else 2, "total_cost_usd": 0.0}}))
    else:
        print(text)


if MODE != "signed-in":
    result(NOT_LOGGED_IN, is_error=True)
    sys.exit(1)

# The Read tool, as the prompt names the file: outside the working directory
# it prompts unless pre-approved, and with nobody to answer it is denied.
match = re.search(r"(/\\S+\\.jpe?g)", args.prompt, re.IGNORECASE)
if match is None:
    result("I could not find an image path in the prompt.")
    sys.exit(0)
image = match.group(1)
if "Read" not in args.tools.split(","):
    result("I have no tool that can read files.")
    sys.exit(0)
if "Read" not in args.allowedTools.replace(",", " ").split():
    result("Permission to read " + image + " was denied.")
    sys.exit(0)
if not os.path.isfile(image):
    result("The file " + image + " does not exist.")
    sys.exit(0)
answer = ROUTING if "router" in args.prompt else IDENTIFICATION
result("```json\\n" + answer + "\\n```")
'''


def _fake_claude(monkeypatch, tmp_path, *, mode: str = "signed-in") -> Path:
    """Write a `claude` that imitates the real CLI's documented interface into
    a folder put first on PATH, so the real shutil.which finds it ahead of
    any real Claude Code and the real subprocess runs it, no shell. Returns
    the log it appends each invocation's argv and cwd to."""
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    log = tmp_path / "claude-calls.jsonl"
    script = folder / CLAUDE
    script.write_text(_FAKE_CLAUDE_SCRIPT.format(
        python=sys.executable, mode=mode, routing=ROUTING_OK, identification=ID_OK, log=str(log),
    ), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    # conftest's autouse fixture stubs detection out; this test wants the
    # real one, against the fake.
    monkeypatch.setattr(providers, "claude_code_verdict", REAL_CLAUDE_CODE_VERDICT)
    assert shutil.which(CLAUDE) == str(script)
    return log


def _no_claude(monkeypatch, tmp_path) -> None:
    """A PATH on which nothing is called `claude`, keeping the interpreter's
    own folder so a shebang script elsewhere on it still runs."""
    empty = tmp_path / "empty-bin"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", f"{empty}{os.pathsep}{Path(sys.executable).parent}")
    monkeypatch.setattr(providers, "claude_code_verdict", REAL_CLAUDE_CODE_VERDICT)
    assert shutil.which(CLAUDE) is None, "a real claude is still on the test PATH"


def test_claude_code_is_an_engine_name_on_the_command_seam():
    """Card #421: `claude-code` is the engine's name (the owner's words), a
    named configuration of the command seam and not a new backend: it is
    local (bills to a subscription, not per call: no cloud retuning, no
    cost prompt, no cloud cache file), selectable by config and --backend,
    and not yet a picker choice (the picker learns it in #423), so
    BACKEND_CHOICES is unchanged."""
    assert providers.CLAUDE_CODE == "claude-code"
    assert providers.CLAUDE_CODE in providers.LOCAL_BACKENDS
    assert providers.BACKEND_CHOICES == (*ENGINES, providers.SCRIPTED)
    assert not providers.is_cloud_primary(_cfg(model={"backend": "claude-code"}))


def test_cli_accepts_backend_claude_code(photos, tmp_path, capsys):
    from melampus.cli import main

    code = main([str(photos), "--backend", "claude-code", "--report-only",
                 "--cache", str(tmp_path / "cache.jsonl")])

    assert code == 0, capsys.readouterr().err


def test_claude_code_template_is_the_documented_print_mode_invocation():
    """The built-in template, from `claude --help` (2.1.277) and
    code.claude.com/docs/en/headless: `-p` runs non-interactively and
    exits; `--output-format json` puts the reply in the `result` field;
    `--tools Read` leaves it only the tool that reads files (which returns
    PNG and JPG "as visual content that Claude can see", tools-reference);
    `--allowedTools Read` pre-approves that tool everywhere, so the staged
    image in its temporary folder is read without a prompt;
    `--permission-prompts none` denies anything else that would wait for a
    person; `--no-session-persistence` keeps a thousand frames from writing
    a thousand transcripts; `--strict-mcp-config` connects no MCP server;
    `--setting-sources user` loads no project or local settings from
    wherever melampus was launched. The prompt is the last argument, the
    positional, and carries both placeholders: the image's path for the
    Read tool to read, then the pipeline's prompt in full. It is a valid
    `[model] command` by the config's own rule."""
    template = providers.CLAUDE_CODE_COMMAND
    assert template[0] == CLAUDE == providers.CLAUDE_CODE_PROGRAM
    flags = template[1:-1]
    assert flags == [
        "-p", "--output-format", "json", "--tools", "Read", "--allowedTools", "Read",
        "--permission-prompts", "none", "--no-session-persistence", "--strict-mcp-config",
        "--setting-sources", "user",
    ]
    assert "{image}" in template[-1] and "{prompt}" in template[-1]
    assert template[-1].index("{image}") < template[-1].index("{prompt}")
    assert "Read" in template[-1], "the prompt must say to read the file with the Read tool"
    assert _cfg(model={"command": template}).model.command == template


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (json.dumps({"type": "result", "subtype": "success", "is_error": False,
                     "result": _CLAUDE_RESULT.format(ROUTING_OK), "num_turns": 2}),
         _CLAUDE_RESULT.format(ROUTING_OK)),
        ("Sure:\n" + ID_OK, "Sure:\n" + ID_OK),
        ('{"taxon": "bird", "confidence": 0.9, "reasoning": "a heron"}',
         '{"taxon": "bird", "confidence": 0.9, "reasoning": "a heron"}'),
    ],
    ids=["json-result", "text", "bare-reply-json"],
)
def test_claude_code_reply_is_the_result_field_of_the_json_output(stdout, expected):
    """Reply extraction: `--output-format json` wraps the text in a result
    object (headless docs: "the text result in the `result` field"), and
    the shared JSON extraction must see only the text, not the wrapper. A
    stdout that is not that object, as a user's own `[model] command` with
    `--output-format text` prints, or a bare reply that happens to be JSON
    without a `result` key, passes through untouched."""
    assert providers.claude_code_reply(stdout) == expected


def test_claude_code_reply_maps_not_logged_in_to_the_sign_in_command():
    """Measured on 2.1.277 with an empty CLAUDE_CONFIG_DIR: exit 1, nothing on
    stderr, and on stdout the result object with `is_error` true,
    `subtype` still "success", and the result "Not logged in · Please run
    /login". `/login` is the interactive session's command; the refusal
    names the one that works from a shell, `claude auth login`, as
    CommandFailed: the engine is broken, not the frame."""
    stdout = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                         "result": "Not logged in · Please run /login", "num_turns": 1})
    with pytest.raises(CommandFailed) as err:
        providers.claude_code_reply(stdout)
    message = str(err.value)
    assert "not signed in" in message and providers.CLAUDE_CODE_SIGN_IN in message
    assert providers.CLAUDE_CODE_SIGN_IN == "claude auth login"


def test_claude_code_reply_surfaces_any_other_error_result():
    """Any other `is_error` result (a rate limit, a model that is not found)
    is CommandFailed carrying Claude Code's own words."""
    stdout = json.dumps({"type": "result", "is_error": True, "result": "API Error: 429 rate limited"})
    with pytest.raises(CommandFailed) as err:
        providers.claude_code_reply(stdout)
    assert "API Error: 429 rate limited" in str(err.value)


def test_command_backend_decodes_stdout_before_the_reply_is_read(tmp_path):
    """The seam's one extension for a CLI that wraps its reply: an optional
    `decode` on stdout, applied before the empty-reply check, so a wrapper
    whose text is empty is "printed nothing" and a decoder's CommandFailed
    stops the batch the way a non-zero exit does."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    seen: list[str] = []

    def decode(stdout: str) -> str:
        seen.append(stdout)
        return stdout.upper()

    backend = _command_backend(_FakeRun(stdout="reply"), decode=decode)
    assert backend.complete(image, "prompt", 10).text == "REPLY"
    assert seen == ["reply"]

    with pytest.raises(RuntimeError, match="printed nothing"):
        _command_backend(_FakeRun(stdout="wrapper"), decode=lambda _: "  ").complete(image, "p", 10)

    def refuse(stdout: str) -> str:
        raise CommandFailed("not signed in")

    with pytest.raises(CommandFailed, match="not signed in"):
        _command_backend(_FakeRun(stdout="wrapper"), decode=refuse).complete(image, "p", 10)


@posix_only
def test_claude_code_primary_builds_the_command_backend_on_the_built_in_template(
    monkeypatch, tmp_path
):
    """Given engine claude-code and Claude Code installed and signed in, the
    factory builds a CommandBackend on the built-in template, the path
    shutil.which resolved `claude` to, timeout_seconds, and the reply
    decoder; `[model] command`, when the user sets one, replaces the
    template (a different model flag, a full path) and keeps the rest."""
    _fake_claude(monkeypatch, tmp_path)

    backend = providers.build_primary_backend(
        _cfg(model={"backend": "claude-code", "timeout_seconds": 30}))
    assert isinstance(backend, CommandBackend)
    assert backend.command == providers.CLAUDE_CODE_COMMAND
    assert backend.executable == shutil.which(CLAUDE)
    assert backend.timeout == 30.0
    assert backend.name == " ".join(providers.CLAUDE_CODE_COMMAND)

    own = [CLAUDE, "-p", "--model", "sonnet", "--output-format", "json", "{image} {prompt}"]
    backend = providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": own}))
    assert backend.command == own
    assert backend.executable == shutil.which(CLAUDE)


@posix_only
def test_claude_code_returns_candidates_in_the_same_shape_as_mlx_on_the_fixture(
    monkeypatch, photos, tmp_path
):
    """Card #421, Done-when 1 and 3, at the real boundary: the fake `claude`
    on PATH takes the documented print-mode flags, reads the image path out
    of the prompt the way the Read tool would (refusing when the tool is
    not allowed or the file is not there), and answers the routing prompt
    then the bird prompt inside the JSON result object. The factory builds
    the backend, the Identifier stages the committed fixture, and the
    candidates equal, field for field, the scripted pipeline's for the same
    replies."""
    from melampus.identify import Identifier

    log = _fake_claude(monkeypatch, tmp_path)
    config = _cfg(model={"backend": "claude-code"})
    backend = providers.build_primary_backend(config)
    result = Identifier(backend, config).identify(photos / PHOTO)
    expected = Identifier(
        ScriptedBackend([ROUTING_OK, ID_OK], name=" ".join(providers.CLAUDE_CODE_COMMAND)), config
    ).identify(photos / PHOTO)

    assert result.status == "ok", result.error
    assert result.model == " ".join(providers.CLAUDE_CODE_COMMAND)
    assert result.identification == expected.identification
    assert result.taxon_routing == expected.taxon_routing
    assert [c.common_name for c in result.identification.ranked()] == [
        "Tricolored Heron", "Little Blue Heron"]
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    runs = [c["argv"] for c in calls if c["argv"][:1] == ["-p"]]
    assert len(runs) == 2, calls
    for argv in runs:
        assert argv[:-1] == providers.CLAUDE_CODE_COMMAND[1:-1]
        assert str(photos / PHOTO) not in argv[-1], "the original file's path reached the program"
        assert "melampus-" in argv[-1], "the staged copy's path is not in the prompt"


@posix_only
def test_detection_claude_code_is_available_when_installed_and_signed_in(monkeypatch, tmp_path):
    """Card #421, Done-when 2: given Claude Code installed and signed in, when
    detection runs, then claude-code is available and the reason says runs
    bill to the subscription. The check is the documented, cheap one:
    `claude auth status` "Exits with code 0 if logged in, 1 if not"
    (cli-reference), no model call."""
    log = _fake_claude(monkeypatch, tmp_path)
    verdict = _verdict("claude-code")
    assert verdict.available, verdict.reason
    assert "subscription" in verdict.reason
    calls = [json.loads(line)["argv"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls == [["auth", "status", "--json"]]


@posix_only
def test_detection_claude_code_not_signed_in_names_the_sign_in_command(monkeypatch, tmp_path):
    """Done-when 2: installed but not signed in, then unavailable with the
    reason "not signed in" and the command that signs in."""
    _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "not signed in" in verdict.reason and providers.CLAUDE_CODE_SIGN_IN in verdict.reason


def test_detection_claude_code_not_installed_points_to_the_install(monkeypatch, tmp_path):
    """Done-when 2: not installed, then unavailable with the reason "not
    installed" and where to get it."""
    _no_claude(monkeypatch, tmp_path)
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "not installed" in verdict.reason and providers.CLAUDE_CODE_INSTALL in verdict.reason


@posix_only
def test_detection_claude_code_gives_up_when_the_status_check_hangs(monkeypatch, tmp_path):
    """--detect-engines never hangs: a status check that does not answer
    within CLAUDE_CODE_PROBE_SECONDS is an unavailable verdict saying so,
    not a stalled dialog."""
    _fake_claude(monkeypatch, tmp_path, mode="hung")
    monkeypatch.setattr(providers, "CLAUDE_CODE_PROBE_SECONDS", 0.5)
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "did not answer" in verdict.reason


def test_claude_code_probe_timeout_is_short():
    """Short enough that a broken install cannot stall the settings dialog;
    long enough for a Node CLI's start (measured: 0.1 s)."""
    assert 1.0 <= providers.CLAUDE_CODE_PROBE_SECONDS <= 15.0


@posix_only
def test_the_refusal_names_claude_code_when_it_is_signed_in(monkeypatch, tmp_path, no_ambient_ollama):
    """One truth: the backends a refusal names as working here follow
    detection, so claude-code is named when Claude Code is signed in and
    not otherwise."""
    _fake_claude(monkeypatch, tmp_path)
    assert "claude-code" in providers._works_here(providers.detect_engines())
    _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")
    assert "claude-code" not in providers._works_here(providers.detect_engines())


@posix_only
def test_cli_detect_engines_prints_the_claude_code_verdict(monkeypatch, tmp_path, capsys, no_ambient_keys):
    """--detect-engines carries the fifth verdict after the four, as JSON."""
    from melampus.cli import main

    _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")
    assert main(["--detect-engines"]) == 0
    verdicts = json.loads(capsys.readouterr().out)
    assert [v["engine"] for v in verdicts] == [*ENGINES, "claude-code", "codex"]
    assert verdicts[-2]["available"] is False
    assert providers.CLAUDE_CODE_SIGN_IN in verdicts[-2]["reason"]


@posix_only
def test_claude_code_not_signed_in_is_refused_before_any_image_is_read(monkeypatch, tmp_path, capsys):
    """Card #421, Done-when 2 at analysis time, where detection can tell:
    Claude Code installed but not signed in, and the folder's one image is
    a link to nowhere, so opening it would fail loudly. The CLI exits 3 on
    a refusal that says to run the sign-in command and never mentions the
    file: the status check ran before any image was read."""
    from melampus.cli import main

    _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")

    code = main([str(folder), "--backend", "claude-code", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not signed in" in err and providers.CLAUDE_CODE_SIGN_IN in err
    for about_the_file in ("nowhere", "does-not-exist", "No such file", "unreadable"):
        assert about_the_file not in err, f"the image was touched before the sign-in check:\n{err}"


def test_claude_code_not_installed_is_refused_before_any_image_is_read(monkeypatch, tmp_path, capsys):
    """Done-when 2 at analysis time, not installed: exit 3 naming `claude`,
    where to install it and how to sign in, before any image is read."""
    from melampus.cli import main

    _no_claude(monkeypatch, tmp_path)
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")

    code = main([str(folder), "--backend", "claude-code", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not installed" in err and providers.CLAUDE_CODE_INSTALL in err
    assert providers.CLAUDE_CODE_SIGN_IN in err
    assert "nowhere" not in err and "does-not-exist" not in err


@posix_only
def test_claude_code_that_lapses_mid_run_stops_the_batch_at_the_first_reply(
    monkeypatch, photos, tmp_path, capsys
):
    """Done-when 2 where detection cannot tell: the status check passed, and
    the first run answers not-logged-in the documented way (exit 1, the
    result object with is_error, nothing on stderr). The run stops at exit
    3 on the same refusal, naming the sign-in command, rather than
    recording it on every frame; nothing is cached."""
    from melampus.cli import main

    _fake_claude(monkeypatch, tmp_path, mode="expired")
    out = tmp_path / "results.json"

    code = main([str(photos), "--backend", "claude-code", "--cache", str(tmp_path / "cache.jsonl"),
                 "--json-out", str(out)])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not signed in" in err and providers.CLAUDE_CODE_SIGN_IN in err
    assert not out.exists() and not (tmp_path / "cache.jsonl").exists()


@posix_only
def test_cli_backend_claude_code_writes_a_json_result(monkeypatch, photos, tmp_path, capsys):
    """Acceptance for Done-when 1: `melampus-id FOLDER --backend claude-code
    --json-out FILE` with the fake `claude` on PATH, signed in, on the
    committed fixture, runs the whole pipeline and writes a JSON result
    with the candidates, attributed to the template, nothing retuned for a
    cloud and no cost prompt."""
    from melampus.cli import main

    _fake_claude(monkeypatch, tmp_path)
    out = tmp_path / "results.json"

    code = main([str(photos), "--backend", "claude-code", "--cache", str(tmp_path / "cache.jsonl"),
                 "--json-out", str(out)])

    err = capsys.readouterr().err
    assert code == 0, err
    assert f"loading {' '.join(providers.CLAUDE_CODE_COMMAND)}" in err, err
    assert "cloud default" not in err and "estimate" not in err.lower()
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert result["file"] == PHOTO
    assert result["status"] == "ok"
    assert result["model"] == " ".join(providers.CLAUDE_CODE_COMMAND)
    assert [c["common_name"] for c in result["identification"]["candidates"]] == [
        "Tricolored Heron", "Little Blue Heron"]


# --- card #422: Codex CLI as an engine --------------------------------------

CODEX = "codex"

#: The usage-limit reply the real Codex CLI 0.154.0 gave on the one exec
#: attempt (2026-09-18, the owner's plan at its limit), verbatim.
_CODEX_USAGE_LIMIT = (
    "You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to "
    "purchase more credits or try again at Sep 19th, 2026 7:46 AM."
)

#: The not-signed-in failure measured with an empty CODEX_HOME: every
#: attempt is refused with a 401 and the turn fails on it, exit 1.
_CODEX_UNAUTHORIZED = (
    "unexpected status 401 Unauthorized: Missing bearer or basic authentication in "
    "header, url: https://api.openai.com/v1/responses, cf-ray: a3d4aafb6870e669-DEN, "
    "request id: req_8d75516fb1544d9e8b56bb59d3f4fa67"
)


def _codex_events(*events: dict) -> str:
    """A `--json` stdout: one event per line, as the docs' sample stream."""
    return "".join(json.dumps(event) + "\n" for event in events)


def _codex_answer(text: str) -> str:
    """The documented success stream around one agent message."""
    return _codex_events(
        {"type": "thread.started", "thread_id": "01a0b730-6b11-7720-9573-672388e3cc7e"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 1620, "cached_input_tokens": 0,
                                             "output_tokens": 61, "reasoning_output_tokens": 0}},
    )


def _codex_failure(message: str) -> str:
    """The measured failure stream: an error event, then the turn fails on it."""
    return _codex_events(
        {"type": "thread.started", "thread_id": "01a0b730-6b11-7720-9573-672388e3cc7e"},
        {"type": "turn.started"},
        {"type": "error", "message": message},
        {"type": "turn.failed", "error": {"message": message}},
    )


_FAKE_CODEX_SCRIPT = '''#!{python}
"""Stands in for Codex CLI 0.154.0's documented non-interactive interface,
as `codex exec --help`, `codex login status --help` and the docs
(developers.openai.com/codex: non-interactive-mode, developer-commands,
image-inputs) describe it, and as measured on 2026-09-18: `codex exec
[OPTIONS] [PROMPT]` runs once; `-i/--image <FILE>...` attaches the image to
the prompt (variadic: a prompt right after it is taken for a second file,
so a flag must come between); `--json` makes stdout a JSONL stream whose
`item.completed` agent_message carries the reply and whose `turn.failed`
carries a failure's message; without `--json` only the final message is
on stdout; a run that cannot proceed exits 1 with the failure in the
stream and progress on stderr; `codex login status` exits 0 when signed
in ("Logged in using ChatGPT" on stderr) and 1 when not ("Not logged in").
MODE: "signed-in" answers; "not-signed-in" fails the status check and
every run the measured way (401); "usage-limit" passes the status check
and fails every run with the measured usage-limit reply; "hung" never
answers the status check."""
import json
import os
import sys

MODE = {mode!r}
ROUTING = {routing!r}
IDENTIFICATION = {identification!r}
LOG = {log!r}
USAGE_LIMIT = {usage_limit!r}
UNAUTHORIZED = {unauthorized!r}

argv = sys.argv[1:]
with open(LOG, "a", encoding="utf-8") as log:
    log.write(json.dumps({{"argv": argv, "cwd": os.getcwd()}}) + "\\n")

if argv[:2] == ["login", "status"]:
    if MODE == "hung":
        import time
        time.sleep(30)
    if MODE == "not-signed-in":
        print("Not logged in", file=sys.stderr)
        sys.exit(1)
    print("Logged in using ChatGPT", file=sys.stderr)
    sys.exit(0)

if argv[:1] != ["exec"]:
    sys.exit("the fake codex knows `exec` and `login status` only")

import argparse

ap = argparse.ArgumentParser(prog="codex exec")
ap.add_argument("-i", "--image", nargs="+", action="extend", default=[])
ap.add_argument("--json", action="store_true")
ap.add_argument("--ephemeral", action="store_true")
ap.add_argument("--skip-git-repo-check", action="store_true")
ap.add_argument("--ignore-user-config", action="store_true")
ap.add_argument("-s", "--sandbox", choices=["read-only", "workspace-write", "danger-full-access"])
ap.add_argument("-c", "--config", action="append", default=[])
ap.add_argument("--color", choices=["always", "never", "auto"], default="auto")
ap.add_argument("-o", "--output-last-message")
ap.add_argument("-m", "--model")
ap.add_argument("prompt", nargs="?")
args = ap.parse_args(argv[1:])
if args.prompt is None:
    print("Reading prompt from stdin...", file=sys.stderr)
    print("No prompt provided via stdin.", file=sys.stderr)
    sys.exit(1)
print("Reading additional input from stdin...", file=sys.stderr)


def event(**fields):
    if args.json:
        print(json.dumps(fields))


def fail(message):
    event(type="thread.started", thread_id="01a0b730-6b11-7720-9573-672388e3cc7e")
    event(type="turn.started")
    event(type="error", message=message)
    event(type="turn.failed", error={{"message": message}})
    if not args.json:
        print("ERROR: " + message, file=sys.stderr)
    sys.exit(1)


if MODE == "not-signed-in":
    fail(UNAUTHORIZED)
if MODE == "usage-limit":
    fail(USAGE_LIMIT)
for image in args.image:
    if not os.path.isfile(image):
        fail("image not found: " + image)
answer = "```json\\n" + (ROUTING if "router" in args.prompt else IDENTIFICATION) + "\\n```"
event(type="thread.started", thread_id="01a0b730-6b11-7720-9573-672388e3cc7e")
event(type="turn.started")
event(type="item.completed", item={{"id": "item_0", "type": "agent_message", "text": answer}})
event(type="turn.completed", usage={{"input_tokens": 1620, "cached_input_tokens": 0,
                                    "output_tokens": 61, "reasoning_output_tokens": 0}})
if not args.json:
    print(answer)
if args.output_last_message:
    with open(args.output_last_message, "w", encoding="utf-8") as last:
        last.write(answer)
'''


def _fake_codex(monkeypatch, tmp_path, *, mode: str = "signed-in") -> Path:
    """Write a `codex` that imitates the real CLI's documented interface into
    a folder put first on PATH, so the real shutil.which finds it ahead of
    any real Codex and the real subprocess runs it, no shell. Returns the
    log it appends each invocation's argv and cwd to."""
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    log = tmp_path / "codex-calls.jsonl"
    script = folder / CODEX
    script.write_text(_FAKE_CODEX_SCRIPT.format(
        python=sys.executable, mode=mode, routing=ROUTING_OK, identification=ID_OK, log=str(log),
        usage_limit=_CODEX_USAGE_LIMIT, unauthorized=_CODEX_UNAUTHORIZED,
    ), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    # conftest's autouse fixture stubs detection out; this test wants the
    # real one, against the fake.
    monkeypatch.setattr(providers, "codex_verdict", REAL_CODEX_VERDICT)
    assert shutil.which(CODEX) == str(script)
    return log


def _no_codex(monkeypatch, tmp_path) -> None:
    """A PATH on which nothing is called `codex`, keeping the interpreter's
    own folder so a shebang script elsewhere on it still runs."""
    empty = tmp_path / "empty-bin"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", f"{empty}{os.pathsep}{Path(sys.executable).parent}")
    monkeypatch.setattr(providers, "codex_verdict", REAL_CODEX_VERDICT)
    assert shutil.which(CODEX) is None, "a real codex is still on the test PATH"


@posix_only
def test_detection_titles_every_engine_for_the_picker(monkeypatch, tmp_path, no_ambient_keys, no_ambient_ollama):
    """Card #423: the picker's titles come from the verdict, the one source,
    not a table in Lua. Every verdict carries a title; the four engines'
    are the plain names, and the two CLIs' name the program (CliEngine
    .title) and that no key is needed, the same in every state: not
    installed, installed but not signed in, signed in."""
    _no_claude(monkeypatch, tmp_path)
    _no_codex(monkeypatch, tmp_path)
    titles = {v.engine: v.title for v in providers.detect_engines()}
    assert titles == {
        "mlx": "MLX — local, Apple Silicon",
        "ollama": "Ollama — local",
        "openai": "OpenAI — cloud, needs an API key",
        "claude": "Claude — cloud, needs an API key",
        "claude-code": "Claude Code — subscription, no API key",
        "codex": "Codex CLI — subscription, no API key",
    }
    assert titles["claude-code"].startswith(providers.CLAUDE_CODE_CLI.title)
    assert titles["codex"].startswith(providers.CODEX_CLI.title)
    for mode in ("not-signed-in", "signed-in"):
        _fake_claude(monkeypatch, tmp_path, mode=mode)
        _fake_codex(monkeypatch, tmp_path, mode=mode)
        assert _verdict("claude-code").title == titles["claude-code"], mode
        assert _verdict("codex").title == titles["codex"], mode


def test_codex_is_an_engine_name_on_the_command_seam():
    """Card #422: `codex` is the engine's name (the owner's words), a named
    configuration of the command seam and not a new backend: it is local
    (bills to a subscription, not per call: no cloud retuning, no cost
    prompt, no cloud cache file), selectable by config and --backend, and
    not yet a picker choice (the picker learns it in #423), so
    BACKEND_CHOICES is unchanged."""
    assert providers.CODEX == "codex"
    assert providers.CODEX in providers.LOCAL_BACKENDS
    assert providers.BACKEND_CHOICES == (*ENGINES, providers.SCRIPTED)
    assert not providers.is_cloud_primary(_cfg(model={"backend": "codex"}))


def test_cli_accepts_backend_codex(photos, tmp_path, capsys):
    from melampus.cli import main

    code = main([str(photos), "--backend", "codex", "--report-only",
                 "--cache", str(tmp_path / "cache.jsonl")])

    assert code == 0, capsys.readouterr().err


def test_codex_template_is_the_documented_exec_invocation():
    """The built-in template, from `codex exec --help` (0.154.0) and
    developers.openai.com/codex (non-interactive-mode, developer-commands,
    image-inputs): `exec` runs non-interactively; `--image {image}` attaches
    the staged JPEG to the prompt ("Attach images to the first message";
    PNG and JPEG accepted), and sits first because the flag is variadic, so
    the prompt must not follow it directly; `--json` makes stdout a JSONL
    stream whose agent_message is the reply and whose turn.failed is a
    failure in the CLI's own words (codex_reply reads both); `--ephemeral`
    writes no session per frame; `--skip-git-repo-check` runs from wherever
    melampus was launched; `--ignore-user-config` loads no
    ~/.codex/config.toml (no MCP server per frame; auth still read);
    `--sandbox read-only` and `-c approval_policy="never"` let the run
    proceed with nobody to approve and nothing writable (exec 0.154.0 has
    no --ask-for-approval flag, measured; the config key is the documented
    equivalent); `-c project_doc_max_bytes=0` keeps the launch directory's
    AGENTS.md out of the prompt; `--color never` keeps ANSI out of stderr.
    The prompt is the positional argument, last, the pipeline's prompt in
    full. It is a valid `[model] command` by the config's own rule."""
    template = providers.CODEX_COMMAND
    assert template[0] == CODEX == providers.CODEX_PROGRAM
    assert template[1:4] == ["exec", "--image", "{image}"]
    assert template[4].startswith("--"), "a flag must follow the variadic --image"
    assert template[4:-1] == [
        "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
        "--sandbox", "read-only", "-c", 'approval_policy="never"',
        "-c", "project_doc_max_bytes=0", "--color", "never",
    ]
    assert template[-1] == "{prompt}"
    assert "--ask-for-approval" not in template, "codex exec 0.154.0 rejects it"
    assert _cfg(model={"command": template}).model.command == template


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (_codex_answer("```json\n" + ROUTING_OK + "\n```"), "```json\n" + ROUTING_OK + "\n```"),
        (_codex_events(
            {"type": "thread.started", "thread_id": "x"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning",
                                                 "text": "Looking at the image."}},
            {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message",
                                                 "text": "Let me look."}},
            {"type": "item.completed", "item": {"id": "item_2", "type": "agent_message",
                                                 "text": ID_OK}},
            {"type": "turn.completed", "usage": {}},
        ), ID_OK),
        ("Sure:\n" + ID_OK, "Sure:\n" + ID_OK),
        ('{"taxon": "bird", "confidence": 0.9, "reasoning": "a heron"}',
         '{"taxon": "bird", "confidence": 0.9, "reasoning": "a heron"}'),
    ],
    ids=["jsonl-agent-message", "last-of-several", "text", "bare-reply-json"],
)
def test_codex_reply_is_the_agent_message_of_the_jsonl_stream(stdout, expected):
    """Reply extraction: `--json` makes stdout a JSONL stream (docs: "every
    event Codex emits"), and the shared JSON extraction must see only the
    final agent message's text, not the events. Several agent messages: the
    last is the reply (the docs' -o writes "the final message"). A stdout
    that is not an event stream, as a user's own `[model] command` without
    `--json` prints, or a bare reply that happens to be JSON without a
    `type`, passes through untouched."""
    assert providers.codex_reply(stdout) == expected


def test_codex_reply_maps_the_usage_limit_to_a_refusal_naming_the_reset_time():
    """Measured on 0.154.0 with the owner's plan at its limit: exit 1, the
    stream's `error` and `turn.failed` both carrying "You've hit your usage
    limit ... try again at Sep 19th, 2026 7:46 AM." That is the engine
    refusing, not the frame: CommandFailed naming the limit and, as the CLI
    said it, when it resets, so the batch stops at the first reply."""
    with pytest.raises(CommandFailed) as err:
        providers.codex_reply(_codex_failure(_CODEX_USAGE_LIMIT))
    message = str(err.value)
    assert "usage limit" in message
    assert "Sep 19th, 2026 7:46 AM" in message
    assert _CODEX_USAGE_LIMIT in message


def test_codex_reply_maps_unauthorized_to_the_sign_in_command():
    """Measured on 0.154.0 with an empty CODEX_HOME (no credentials): every
    attempt is refused 401 Unauthorized and the turn fails on it, exit 1.
    The refusal names the command that signs in, `codex login`."""
    with pytest.raises(CommandFailed) as err:
        providers.codex_reply(_codex_failure(_CODEX_UNAUTHORIZED))
    message = str(err.value)
    assert "not signed in" in message and providers.CODEX_SIGN_IN in message
    assert providers.CODEX_SIGN_IN == "codex login"


def test_codex_reply_surfaces_any_other_failed_turn():
    """Any other turn.failed (a model that is not found, a network that is
    down) is CommandFailed carrying Codex's own words; transient `error`
    events before a completed turn are not failures."""
    with pytest.raises(CommandFailed) as err:
        providers.codex_reply(_codex_failure("model not found: gpt-0"))
    assert "model not found: gpt-0" in str(err.value)

    recovered = _codex_events(
        {"type": "thread.started", "thread_id": "x"},
        {"type": "turn.started"},
        {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected)"},
        {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "ok"}},
        {"type": "turn.completed", "usage": {}},
    )
    assert providers.codex_reply(recovered) == "ok"


def test_codex_reply_with_no_agent_message_is_an_empty_reply():
    """A stream that completes without an agent message is "printed
    nothing" for the seam (recorded on the frame, the batch goes on), not
    a crash on a missing key."""
    stream = _codex_events(
        {"type": "thread.started", "thread_id": "x"},
        {"type": "turn.started"},
        {"type": "turn.completed", "usage": {}},
    )
    assert providers.codex_reply(stream).strip() == ""


@posix_only
def test_codex_primary_builds_the_command_backend_on_the_built_in_template(monkeypatch, tmp_path):
    """Given engine codex and Codex installed and signed in, the factory
    builds a CommandBackend on the built-in template, the path shutil.which
    resolved `codex` to, timeout_seconds, and the reply decoder;
    `[model] command`, when the user sets one, replaces the template (a
    model flag, say) and keeps the rest."""
    _fake_codex(monkeypatch, tmp_path)

    backend = providers.build_primary_backend(_cfg(model={"backend": "codex", "timeout_seconds": 30}))
    assert isinstance(backend, CommandBackend)
    assert backend.command == providers.CODEX_COMMAND
    assert backend.executable == shutil.which(CODEX)
    assert backend.timeout == 30.0
    assert backend.name == " ".join(providers.CODEX_COMMAND)

    own = [CODEX, "exec", "-m", "gpt-5", "--image", "{image}", "--json", "{prompt}"]
    backend = providers.build_primary_backend(_cfg(model={"backend": "codex", "command": own}))
    assert backend.command == own
    assert backend.executable == shutil.which(CODEX)


@posix_only
def test_codex_returns_candidates_in_the_same_shape_as_mlx_on_the_fixture(
    monkeypatch, photos, tmp_path
):
    """Card #422, Done-when 1 and 3, at the real boundary: the fake `codex`
    on PATH takes the documented exec flags, checks the attached image is a
    file, and answers the routing prompt then the bird prompt as the
    agent_message of the JSONL stream. The factory builds the backend, the
    Identifier stages the committed fixture, and the candidates equal, field
    for field, the scripted pipeline's for the same replies."""
    from melampus.identify import Identifier

    log = _fake_codex(monkeypatch, tmp_path)
    config = _cfg(model={"backend": "codex"})
    backend = providers.build_primary_backend(config)
    result = Identifier(backend, config).identify(photos / PHOTO)
    expected = Identifier(
        ScriptedBackend([ROUTING_OK, ID_OK], name=" ".join(providers.CODEX_COMMAND)), config
    ).identify(photos / PHOTO)

    assert result.status == "ok", result.error
    assert result.model == " ".join(providers.CODEX_COMMAND)
    assert result.identification == expected.identification
    assert result.taxon_routing == expected.taxon_routing
    assert [c.common_name for c in result.identification.ranked()] == [
        "Tricolored Heron", "Little Blue Heron"]
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    runs = [c["argv"] for c in calls if c["argv"][:1] == ["exec"]]
    assert len(runs) == 2, calls
    for argv in runs:
        assert argv[:2] == ["exec", "--image"]
        assert argv[3:-1] == providers.CODEX_COMMAND[4:-1]
        image, prompt = argv[2], argv[-1]
        assert image != str(photos / PHOTO), "the original file's path reached the program"
        assert "melampus-" in image and image.endswith("image.jpg"), "not the staged copy"
        assert str(photos / PHOTO) not in prompt and "melampus-" not in prompt


@posix_only
def test_detection_codex_is_available_when_installed_and_signed_in(monkeypatch, tmp_path):
    """Card #422, Done-when 2: given Codex installed and signed in, when
    detection runs, then codex is available and the reason says runs bill
    to the subscription, naming the account kind the status check printed.
    The check is the documented, cheap one: `codex login status` "exit with
    0 when logged in" (developer-commands), no model call."""
    log = _fake_codex(monkeypatch, tmp_path)
    verdict = _verdict("codex")
    assert verdict.available, verdict.reason
    assert "subscription" in verdict.reason and "ChatGPT" in verdict.reason
    calls = [json.loads(line)["argv"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls == [["login", "status"]]


@posix_only
def test_detection_codex_at_its_usage_limit_is_still_signed_in(monkeypatch, tmp_path):
    """Done-when 2, the limit: `codex login status` does not know the plan's
    usage limit (measured: it says only "Logged in using ChatGPT"; `codex
    doctor` reports nothing on it either), and detection spends no model
    call to find out, so the verdict is signed in; the first run's refusal
    names the limit and the reset time (the test below)."""
    log = _fake_codex(monkeypatch, tmp_path, mode="usage-limit")
    verdict = _verdict("codex")
    assert verdict.available, verdict.reason
    calls = [json.loads(line)["argv"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls == [["login", "status"]], "detection spent a run to learn the limit"


@posix_only
def test_detection_codex_not_signed_in_names_the_sign_in_command(monkeypatch, tmp_path):
    """Done-when 2: installed but not signed in (`codex login status` exit 1,
    "Not logged in", measured), then unavailable with the reason "not
    signed in" and the command that signs in."""
    _fake_codex(monkeypatch, tmp_path, mode="not-signed-in")
    verdict = _verdict("codex")
    assert not verdict.available
    assert "not signed in" in verdict.reason and providers.CODEX_SIGN_IN in verdict.reason


def test_detection_codex_not_installed_points_to_the_install(monkeypatch, tmp_path):
    """Done-when 2: not installed, then unavailable with the reason "not
    installed" and where to get it."""
    _no_codex(monkeypatch, tmp_path)
    verdict = _verdict("codex")
    assert not verdict.available
    assert "not installed" in verdict.reason and providers.CODEX_INSTALL in verdict.reason


@posix_only
def test_detection_codex_gives_up_when_the_status_check_hangs(monkeypatch, tmp_path):
    """--detect-engines never hangs: a status check that does not answer
    within CODEX_PROBE_SECONDS is an unavailable verdict saying so."""
    _fake_codex(monkeypatch, tmp_path, mode="hung")
    monkeypatch.setattr(providers, "CODEX_PROBE_SECONDS", 0.5)
    verdict = _verdict("codex")
    assert not verdict.available
    assert "did not answer" in verdict.reason


def test_codex_probe_timeout_is_short():
    """Short enough that a broken install cannot stall the settings dialog;
    long enough for the binary's start (measured: 0.01 s)."""
    assert 1.0 <= providers.CODEX_PROBE_SECONDS <= 15.0


@posix_only
def test_the_refusal_names_codex_when_it_is_signed_in(monkeypatch, tmp_path, no_ambient_ollama):
    """One truth: the backends a refusal names as working here follow
    detection, so codex is named when Codex is signed in and not otherwise."""
    _fake_codex(monkeypatch, tmp_path)
    assert "codex" in providers._works_here(providers.detect_engines())
    _fake_codex(monkeypatch, tmp_path, mode="not-signed-in")
    assert "codex" not in providers._works_here(providers.detect_engines())


@posix_only
def test_cli_detect_engines_prints_the_codex_verdict(monkeypatch, tmp_path, capsys, no_ambient_keys):
    """--detect-engines carries the sixth verdict after claude-code's, as JSON."""
    from melampus.cli import main

    _fake_codex(monkeypatch, tmp_path, mode="not-signed-in")
    assert main(["--detect-engines"]) == 0
    verdicts = json.loads(capsys.readouterr().out)
    assert [v["engine"] for v in verdicts] == [*ENGINES, "claude-code", "codex"]
    assert verdicts[-1]["available"] is False
    assert providers.CODEX_SIGN_IN in verdicts[-1]["reason"]


@posix_only
def test_codex_not_signed_in_is_refused_before_any_image_is_read(monkeypatch, tmp_path, capsys):
    """Card #422, Done-when 2 at analysis time, where detection can tell:
    Codex installed but not signed in, and the folder's one image is a
    link to nowhere, so opening it would fail loudly. The CLI exits 3 on a
    refusal that says to run the sign-in command and never mentions the
    file: the status check ran before any image was read."""
    from melampus.cli import main

    _fake_codex(monkeypatch, tmp_path, mode="not-signed-in")
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")

    code = main([str(folder), "--backend", "codex", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not signed in" in err and providers.CODEX_SIGN_IN in err
    for about_the_file in ("nowhere", "does-not-exist", "No such file", "unreadable"):
        assert about_the_file not in err, f"the image was touched before the sign-in check:\n{err}"


def test_codex_not_installed_is_refused_before_any_image_is_read(monkeypatch, tmp_path, capsys):
    """Done-when 2 at analysis time, not installed: exit 3 naming `codex`,
    where to install it and how to sign in, before any image is read."""
    from melampus.cli import main

    _no_codex(monkeypatch, tmp_path)
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")

    code = main([str(folder), "--backend", "codex", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not installed" in err and providers.CODEX_INSTALL in err
    assert providers.CODEX_SIGN_IN in err
    assert "nowhere" not in err and "does-not-exist" not in err


@posix_only
def test_codex_at_its_usage_limit_stops_the_batch_at_the_first_reply(
    monkeypatch, photos, tmp_path, capsys
):
    """Done-when 2, the limit, where only a run can tell: the status check
    passed, and the first run fails the turn with the measured usage-limit
    message (exit 1, the stream on stdout). The run stops at exit 3 on a
    refusal naming the limit and, as the CLI said it, when it resets,
    rather than recording it on every frame; nothing is cached."""
    from melampus.cli import main

    _fake_codex(monkeypatch, tmp_path, mode="usage-limit")
    out = tmp_path / "results.json"

    code = main([str(photos), "--backend", "codex", "--cache", str(tmp_path / "cache.jsonl"),
                 "--json-out", str(out)])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "usage limit" in err and "Sep 19th, 2026 7:46 AM" in err
    assert not out.exists() and not (tmp_path / "cache.jsonl").exists()


@posix_only
def test_cli_backend_codex_writes_a_json_result(monkeypatch, photos, tmp_path, capsys):
    """Acceptance for Done-when 1: `melampus-id FOLDER --backend codex
    --json-out FILE` with the fake `codex` on PATH, signed in, on the
    committed fixture, runs the whole pipeline and writes a JSON result
    with the candidates, attributed to the template, nothing retuned for a
    cloud and no cost prompt."""
    from melampus.cli import main

    _fake_codex(monkeypatch, tmp_path)
    out = tmp_path / "results.json"

    code = main([str(photos), "--backend", "codex", "--cache", str(tmp_path / "cache.jsonl"),
                 "--json-out", str(out)])

    err = capsys.readouterr().err
    assert code == 0, err
    assert f"loading {' '.join(providers.CODEX_COMMAND)}" in err, err
    assert "cloud default" not in err and "estimate" not in err.lower()
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert result["file"] == PHOTO
    assert result["status"] == "ok"
    assert result["model"] == " ".join(providers.CODEX_COMMAND)
    assert [c["common_name"] for c in result["identification"]["candidates"]] == [
        "Tricolored Heron", "Little Blue Heron"]
