"""The backend seam as a setting — the Windows / cloud-primary path.

The contract under test: `[model] backend` selects who answers, defaults keep the
Mac local-first path byte-for-byte unchanged, cloud selection fails fast and
clearly (missing SDK, missing key, unknown name), and the MLX-shaped defaults are
retuned for a cloud primary without ever overriding an explicit setting.
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import os
import shlex
import shutil
import signal
import socket
import socketserver
import ssl
import subprocess
import sys
import threading
import time
import types
import urllib.error
from pathlib import Path
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from conftest import (
    PHOTO,
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


@pytest.mark.parametrize("engine", (*ENGINES, providers.COMMAND))
def test_cli_accepts_each_engine_name(engine, photos, tmp_path, no_ambient_keys, capsys):
    """Card #403, Done-when 1: given an engine name, when the plugin passes it
    as --backend, then the CLI receives it. --report-only stops before any
    engine is built, so this is the parse alone: the name is accepted. Card
    #420's `command` is accepted the same way (chosen by config or --backend
    until the picker learns it in #423)."""
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


def test_detection_lists_the_four_engines_in_the_owners_order(no_ambient_keys, no_ambient_ollama):
    """The list the dialog (card #405) will show: one verdict per engine, in
    the order BACKEND_CHOICES names them, never the test fake."""
    verdicts = providers.detect_engines()
    assert [v.engine for v in verdicts] == list(ENGINES)
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

    nothing_available = [providers.EngineVerdict(v.engine, False, v.reason) for v in verdicts]
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


AGAIN_UNDER_A_SIGNAL = """\
import json, signal, time
from melampus.backend import _Deadline


class Interrupted(Exception):
    pass


def raise_interrupted(signum, frame):
    raise Interrupted()


signal.signal(signal.SIGALRM, raise_interrupted)
ended_with = []
for _ in range({rounds}):
    try:
        with _Deadline(60.0) as deadline:
            signal.setitimer(signal.ITIMER_REAL, 0.002)
            until = time.monotonic() + 2.0
            while time.monotonic() < until:
                deadline.again()
        ended_with.append("nothing: the signal never landed")
    except BaseException as exc:
        ended_with.append(type(exc).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
print(json.dumps(ended_with))
"""


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"),
                    reason="SIGALRM and setitimer are POSIX; the window this pins is the thread module's, "
                           "the same on every platform, and the pull's handler on Windows is SIGBREAK's")
def test_deadline_again_under_a_signal_handlers_exception_leaves_that_exception_alone_and_nothing_on_stderr():
    """Code review round 12 (backend.py:419), Done-when 1's signal path: the
    pull runs under `cancel_on_signals`, whose handler raises wherever the
    main thread is, and `again()`, called before every line of the stream,
    started a `threading.Timer` thread for each: raised inside
    `Thread.start()`, the handler's exception left the thread module's
    limbo table to be deleted twice (the new thread dying with a KeyError
    traceback on stderr, the log's tail the plugin shows) or the start
    event's lock held (`RuntimeError: release unlocked lock` out of the
    block, named as Ollama's failure, exit 3, the cancellation lost; or a
    thread stuck on that lock, the process hanging at exit). Given a
    SIGALRM handler that raises while the main thread loops `again()`
    inside a block, `rounds` times in a fresh interpreter (a hang is then
    the timeout, not the suite's), every block ends with the handler's
    exception alone and the process writes nothing on stderr. Measured
    at f398718, the timer per line: 20 of 20 runs failed, 19 with the
    timer thread's KeyError traceback on stderr and 1 hung to the timeout;
    on the one timer per block, 40 of 40 passed, and 40 of 40 under six
    CPU-bound processes loading the machine."""
    rounds = 40
    proc = subprocess.run(
        [sys.executable, "-c", AGAIN_UNDER_A_SIGNAL.format(rounds=rounds)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.stderr == "", f"the process wrote on stderr:\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0, f"exit {proc.returncode}"
    ended_with = json.loads(proc.stdout)
    assert ended_with == ["Interrupted"] * rounds, [e for e in ended_with if e != "Interrupted"]


def test_deadline_watcher_runs_for_the_block_alone():
    """Code review round 10 (backend.py:386), rule 7: 36147fd gave the
    deadline a second thread with a lifecycle, the watcher of `cancel`,
    started in __enter__ and ended by `_over` and joined in __exit__, and
    nothing asserted it. Given a deadline with a predicate, the block has
    the timer and the watcher running beside the caller, and after it the
    watcher is not alive and the timer, once joined, is not either.

    Code review round 11 (test_providers.py:935): the threads the block
    owns, not the process count. __exit__ joins the watcher but only
    cancels the timer, which exits on its own a moment later, and the
    count before the block can include a timer winding down from the
    deadline before, so the counts raced both; the timer is joined here,
    with a bound, before it is read. Round 12 (backend.py:419): the timer
    is one thread for the block, ended by `_over` and joined in __exit__
    as the watcher is, so the join here is already done."""
    with _Deadline(5.0, cancel=lambda: False) as deadline:
        assert deadline._timer.is_alive(), "the timer"
        assert deadline._watcher.is_alive(), "the watcher"
    assert not deadline._watcher.is_alive(), "the watcher outlived the block"
    deadline._timer.join(1.0)
    assert not deadline._timer.is_alive(), "the timer outlived the block"


def test_deadline_hangs_up_a_late_socket_for_a_cancel_as_for_the_timeout():
    """Code review round 10 (backend.py:386), rule 7: `on()` hangs up a
    socket given after the event, for the cancellation as for the
    deadline (a cancel that fired while the caller was still connecting
    has no socket to hang up). Given the watcher's hang-up with no socket
    yet, `cancelled` is set and, once the socket is given, the other end
    reads end-of-stream."""
    deadline = _Deadline(60.0, cancel=lambda: True)
    _hang_up(deadline, deadline.cancelled)
    assert deadline.cancelled.is_set() and not deadline.expired.is_set()
    ours, theirs = socket.socketpair()
    with ours, theirs:
        deadline.on(ours)
        theirs.settimeout(1.0)
        assert theirs.recv(1) == b"", "the stream was not ended for the cancellation"


def test_ollama_backend_stream_ends_cleanly_within_a_watch_when_its_cancel_turns_true():
    """Code review round 10 (backend.py:386), rule 7: the stream's `cancel`
    predicate, watched while a read blocks, had its proof only through
    pull_model and the entry point at 10 s timeouts. Given a listener
    writing one line and then nothing, and a predicate true once that
    line is out, the stream yields the line that arrived and ends with no
    exception within WATCH plus slack, `cancelled` set and `expired` not:
    the cancellation, whatever shape the hung-up read took."""
    from conftest import chunked_pull_answer, stalling_handler

    line = b'{"status": "pulling manifest"}\n'
    lines: list[bytes] = []
    with loopback_server(stalling_handler(chunked_pull_answer(line)), ThreadingHTTPServer) as stalled:
        url = f"http://127.0.0.1:{stalled.server_port}"
        backend = OllamaBackend("qwen3-vl:8b-instruct", url, timeout=10.0)
        request = urllib.request.Request(f"{url}/api/pull", data=b"{}", method="POST")
        with _timed() as took:
            lines.extend(backend.stream(request, cancel=lambda: bool(lines)))
    assert lines == [line]
    assert took.seconds < _Deadline.WATCH + SCHEDULING_SLACK, f"the cancel waited on the read: {took.seconds:.2f}s"
    assert request.deadline.cancelled.is_set() and not request.deadline.expired.is_set()


def _bounded_pull(timeout: float, cancel):
    """A `_bounded` block of a backend at a closed port, with `cancel` as
    the stream gives it: what the block does once it ends is the whole of
    the test, so nothing is sent."""
    backend = OllamaBackend("qwen3-vl:8b-instruct", "http://127.0.0.1:9", timeout=timeout)
    request = urllib.request.Request("http://127.0.0.1:9/api/pull", data=b"{}", method="POST")
    return backend._bounded(request, cancel)


@pytest.mark.parametrize("raising", [True, False], ids=["read-raised", "read-returned"])
def test_bounded_block_ends_cleanly_for_a_cancel_whose_deadline_has_fired_too(raising):
    """Code review round 10 (backend.py:686), Done-when 1: the timer and
    the watcher hang up the same socket and set their own event, and
    `_bounded` read `expired` first on both of its paths, so a cancel asked
    in the last WATCH before a line's deadline was reported as the timeout,
    exit 3 with its message, not `cancelled`, exit 4: the outcome the
    watcher was written to end. Given a block whose deadline (0.3s) and
    cancel (true from the start) have both fired, ending as the hung-up
    read ends it, raising or returning, the block ends cleanly with
    `cancelled` set and nothing raised."""
    with _bounded_pull(0.3, cancel=lambda: True) as deadline:
        time.sleep(0.5)
        assert deadline.expired.is_set() and deadline.cancelled.is_set(), "the case needs both fired"
        if raising:
            raise ConnectionResetError("the hung-up read")
    assert deadline.cancelled.is_set()


@pytest.mark.parametrize("raising", [True, False], ids=["read-raised", "read-returned"])
def test_bounded_block_asks_the_cancel_once_more_when_the_deadline_fired_before_the_watchers_tick(
    monkeypatch, raising
):
    """Code review round 10 (backend.py:686), Done-when 1, the marker the
    watcher has not seen: written after its last tick and before the timer
    fired, a window of up to WATCH at the end of each line's deadline. The
    hang-up is the same hang-up, so once the timer has fired the caller's
    predicate decides which it was. Given a watcher that never ticks (WATCH
    held large), a deadline fired, and the predicate true only after it,
    the block ends cleanly with `cancelled` not set by the watcher and
    nothing raised, for `_lines_until_cancelled` to name the marker."""
    monkeypatch.setattr(_Deadline, "WATCH", 60.0)
    marker = threading.Event()
    with _bounded_pull(0.3, cancel=marker.is_set) as deadline:
        assert deadline.expired.wait(2.0), "the deadline never fired"
        marker.set()
        if raising:
            raise ConnectionResetError("the hung-up read")
    assert not deadline.cancelled.is_set(), "the watcher ticked: the case needs the marker unseen"


def test_bounded_block_asks_the_cancel_once_more_when_the_socket_timed_out_before_the_timer(monkeypatch):
    """Code review round 10 (backend.py:686), Done-when 1, the same window
    by the socket's clock: `stream` gives urlopen `timeout` as the socket
    timeout too, and a blocked read is armed with both at once
    (`again()`, then readline), so at the deadline the socket's timeout
    and the timer race, and half the time the read ends with the socket's
    TimeoutError (named by `_naming`) before the timer has fired. That is
    the timeout the user sees as much as the timer's, so the predicate is
    asked then too. Given a watcher that never ticks, a deadline not
    fired, the predicate true, and the read ending with the timeout, the
    block ends cleanly and nothing is raised."""
    monkeypatch.setattr(_Deadline, "WATCH", 60.0)
    with _bounded_pull(60.0, cancel=lambda: True) as deadline:
        raise TimeoutError("the socket's timeout, as _naming names it")
    assert not deadline.expired.is_set() and not deadline.cancelled.is_set()


def test_bounded_block_re_raises_a_failure_that_is_not_the_timeout_though_the_cancel_is_true(monkeypatch):
    """Code review round 12 (backend.py:718), rule 7: `_cancelled` asks the
    predicate once more only once the block has timed out, the timer's
    `expired` or the socket's TimeoutError (c099782: "once the block has
    timed out the predicate decides"), and no test pinned the guard: with
    it dropped, a failure of any shape while the marker is present would
    be read as the cancellation, exit 4 and `cancelled` for a 400 Ollama
    answered. Given a watcher that never ticks, a deadline not fired, the
    predicate true, and the block ending with a failure that is not the
    timeout, that failure is raised as it was, and neither event is set.
    Red against the guard dropped (`return cancel is not None and
    cancel()`), green at head."""
    monkeypatch.setattr(_Deadline, "WATCH", 60.0)
    with pytest.raises(RuntimeError, match="answered 400"):
        with _bounded_pull(60.0, cancel=lambda: True) as deadline:
            raise RuntimeError("Ollama answered 400: bad")
    assert not deadline.cancelled.is_set() and not deadline.expired.is_set()


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
    assert [v["engine"] for v in verdicts] == list(ENGINES)
    assert all(set(v) == {"engine", "available", "reason"} for v in verdicts)
    by_engine = {v["engine"]: v for v in verdicts}
    assert by_engine["mlx"] == {"engine": "mlx", "available": False, "reason": "needs Apple Silicon"}
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
    """Stands in for subprocess.Popen at the backend's process edge: records
    every call, then returns a started process whose stdout and stderr
    pipes carry `stdout` and `stderr` (text, or bytes as they came, or an
    open binary pipe the test holds the other end of) and that exits
    `returncode`, or raises `error` on starting. `hangs` is a process that
    never finishes on its own: it has not exited, and `wait(timeout=...)`
    raises TimeoutExpired, until it is killed. `stopped` records, for every
    process tree the backend stopped (the OS edge, faked), the command's
    pid and its returncode at that moment: None means it had not been
    reaped, so the pid was still its own and its group's."""

    def __init__(self, stdout: str | bytes | io.BufferedReader = "",
                 stderr: str | bytes | io.BufferedReader = "", returncode: int = 0,
                 error: Exception | None = None, hangs: bool = False):
        self.stdout, self.stderr, self.returncode, self.error = stdout, stderr, returncode, error
        self.hangs = hangs
        self.calls: list[tuple[list[str], dict]] = []
        self.processes: list[_FakeProcess] = []
        self.stopped: list[tuple[int, int | None]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error
        self.processes.append(_FakeProcess(argv, self))
        return self.processes[-1]

    def stop_tree(self, pid: int) -> None:
        """The OS edge faked: os.killpg or taskkill on the command's pid."""
        (process,) = [process for process in self.processes if process.pid == pid]
        self.stopped.append((pid, process.returncode))

    def exited(self, process: _FakeProcess, within: float) -> bool:
        """The OS edge faked: whether `process` has exited, seen without
        reaping it (kqueue, waitid); a process that has not is given
        `within` seconds to, the way the OS call waits."""
        if process.exited:
            return True
        time.sleep(within)
        return False


class _FakeProcess:
    """What _FakeRun starts: the Popen surface the backend uses."""

    def __init__(self, argv, run: _FakeRun):
        self.args, self._run = argv, run
        self.pid = 4242
        self.returncode: int | None = None
        self.killed = False
        self.waited: list[float | None] = []
        self.stdout = self._pipe(run.stdout)
        self.stderr = self._pipe(run.stderr)

    @staticmethod
    def _pipe(said: str | bytes | io.BufferedReader) -> io.BufferedIOBase:
        if isinstance(said, io.BufferedReader):
            return said
        return io.BytesIO(said if isinstance(said, bytes) else said.encode("utf-8"))

    @property
    def exited(self) -> bool:
        """Whether the process has ended, reaped or not: a process that hangs
        has not until it is killed."""
        return not self._run.hangs or self.killed

    def wait(self, timeout: float | None = None):
        self.waited.append(timeout)
        if not self.exited:
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = -9 if self.killed else self._run.returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True


def _command_backend(run: _FakeRun, command: list[str] = COMMAND, **kwargs) -> CommandBackend:
    backend = CommandBackend(command, executable="/opt/fake/bin/fake-vlm", run=run, **kwargs)
    # The process-tree stop (os.killpg, taskkill) and the sight of the exit
    # without reaping (kqueue, waitid) are the OS edge: the fake records the
    # pid it would have stopped, and answers for its own process, instead.
    backend._stop_tree = run.stop_tree
    backend._exited = run.exited
    return backend


def test_command_backend_expands_the_template_into_one_argv(tmp_path):
    """Template expansion: each argument with `{image}` gets the image path as
    given (absolute, the staged file), each with `{prompt}` the prompt in
    full as one argument, spaces, quotes and newlines included, and every
    other argument is passed untouched. The resolved executable stands in
    for the bare name (shutil.which found it, so what was checked is what
    runs). subprocess.Popen is given the list, no
    shell, stdout and stderr piped back as bytes (read here with a ceiling,
    and decoded as UTF-8 with replacement, so a stray byte cannot fail the
    frame), nothing on stdin, so a program that reads it cannot hang, and
    its own session (POSIX) or process group (Windows), so a timeout can
    stop every process it started and not just the first; the config's
    timeout is the wait's ceiling."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK)
    backend = _command_backend(run, timeout=42.0)
    prompt = 'Identify the "bird".\n\nReply with JSON: {"taxon": ...}'

    backend.complete(image, prompt, 900)

    ((argv, kwargs),) = run.calls
    assert argv == ["/opt/fake/bin/fake-vlm", "--image", str(image), "--prompt", prompt, "--quiet"]
    assert "text" not in kwargs and "encoding" not in kwargs, "the pipes are read as bytes, with a ceiling"
    assert kwargs["stdout"] is subprocess.PIPE and kwargs["stderr"] is subprocess.PIPE
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs.get("shell", False) is False
    assert "env" not in kwargs, "the child gets the parent's environment as it is; nothing is added"
    if sys.platform == "win32":
        assert kwargs["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True
    (process,) = run.processes
    assert backend.timeout == 42.0
    assert run.stopped == [(process.pid, None)], (
        "what the command started is stopped at its exit, by its pid while it is still the command's own")


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
    assert backend.name == "fake-vlm --image '{image}' --prompt '{prompt}' --quiet"

    stray = _command_backend(_FakeRun(stdout=b"\xff" + ID_OK.encode("utf-8"))).complete(image, "prompt", 900)
    assert stray.text == "\ufffd" + ID_OK, "a byte that is not UTF-8 is replaced, not a failed frame"


@pytest.mark.parametrize(
    ("run", "expected", "said"),
    [
        (_FakeRun(returncode=2, stderr="not logged in\nrun `fake-vlm login` first\nmore\nand more"),
         CommandFailed, "fake-vlm exited 2: not logged in / run `fake-vlm login` first / more"),
        (_FakeRun(returncode=1), CommandFailed, "fake-vlm exited 1 with nothing on stderr"),
        (_FakeRun(hangs=True), TimeoutError, "fake-vlm did not answer within 0.2s"),
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
    backend = _command_backend(run, timeout=0.2)
    with pytest.raises(expected) as err:
        backend.complete(image, "prompt", 10)
    assert said in str(err.value), str(err.value)
    assert not isinstance(err.value, subprocess.SubprocessError)


def test_command_templates_that_differ_only_in_argument_boundaries_cannot_share_cached_answers():
    """`["--label", "bird --mode precise"]` (one argument) and `["--label",
    "bird", "--mode", "precise"]` (three) run the program differently, so
    their answers are different answers. The backend's name is what the
    run fingerprint carries for the engine (identify.py), so it has to keep
    the argument boundaries: joined with plain spaces, both templates were
    one name, one fingerprint, and a run under the second re-served the
    first's cached answers. shlex.join keeps the boundaries (an argument
    with a space is quoted, and `shlex.split` gives the list back) and
    stays a readable command line for the `loading ...` line and
    `result["model"]`."""
    from melampus.identify import Identifier

    one_argument = [*COMMAND, "--label", "bird --mode precise"]
    three_arguments = [*COMMAND, "--label", "bird", "--mode", "precise"]
    config = _cfg(model={"backend": "command", "command": one_argument})

    fingerprints = {
        Identifier(_command_backend(_FakeRun(), template), config).fingerprint
        for template in (one_argument, three_arguments)
    }

    assert len(fingerprints) == 2, "the two templates share a run fingerprint"
    assert _command_backend(_FakeRun(), one_argument).name == (
        "fake-vlm --image '{image}' --prompt '{prompt}' --quiet --label 'bird --mode precise'")
    assert shlex.split(_command_backend(_FakeRun(), three_arguments).name) == [
        "fake-vlm", *three_arguments[1:]]


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_command_backend_stops_a_command_that_writes_past_the_ceiling(tmp_path, stream):
    """A reply of candidates is kilobytes, so a program streaming megabytes
    on either pipe is broken, not answering: it is stopped, with everything
    it started, the moment the ceiling is passed, not read into memory
    until the timeout, and the frame's error is a RuntimeError naming the
    program, the stream and the ceiling, recorded on the frame while the
    batch goes on. A stream that stops at the ceiling is read whole."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    ceiling = CommandBackend.MAX_OUTPUT_BYTES
    run = _FakeRun(**{stream: "x" * (ceiling + 1)})
    backend = _command_backend(run)

    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)

    assert not isinstance(err.value, subprocess.SubprocessError)
    for named in ("fake-vlm", stream, str(ceiling)):
        assert named in str(err.value), str(err.value)
    (process,) = run.processes
    assert run.stopped == [(process.pid, None)], (
        "the tree is stopped by the command's pid while it is still the command's own, before the reap")
    assert process.returncode is not None, "the stopped command was not reaped"

    within = (_FakeRun(stdout=ID_OK.ljust(ceiling)) if stream == "stdout"
              else _FakeRun(stdout=ID_OK, stderr="x" * ceiling))
    assert _command_backend(within).complete(image, "prompt", 10).text.strip() == ID_OK
    assert within.stopped == [(within.processes[0].pid, None)], "stopped at its exit, not at the ceiling"


def test_command_backend_stops_the_whole_process_tree_on_timeout(tmp_path):
    """A timeout stops the command and everything it started, not just the
    first process: the tree is stopped by the command's pid (its session or
    process group), the command itself is killed, and it is reaped before
    the TimeoutError is raised, so a CLI whose worker outlives it cannot
    leave that worker running while the batch goes on."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(hangs=True)
    backend = _command_backend(run, timeout=0.5)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.complete(image, "prompt", 10)

    assert 0.5 <= time.monotonic() - started < 5, "the config's timeout is the ceiling"
    (process,) = run.processes
    assert run.stopped == [(process.pid, None)], (
        "the tree is stopped by the command's pid while it is still the command's own, before the reap")
    assert process.killed
    assert process.returncode is not None, "the killed command was not reaped"


def test_command_backend_uses_the_reply_of_a_command_that_exited_leaving_its_pipe_held(tmp_path):
    """The command's exit ends its answer. A command that printed its reply,
    exited 0 and left a process it started holding its stdout is not a
    timeout (the old outcome, after the whole timeout, advising a longer
    one that could not help, the reply discarded): the tree it heads is
    stopped the moment its exit is seen, which frees the pipe, and what
    was read is the reply. The stop happens before the command is reaped
    (its returncode still None at that moment), so the pid the tree is
    stopped by is still the command's own and cannot have been given to
    another process. The fake's stdout is a real pipe whose write end the
    test holds, as the worker would, until the tree is stopped."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    read_end, write_end = os.pipe()
    os.write(write_end, ID_OK.encode("utf-8"))
    run = _FakeRun(stdout=os.fdopen(read_end, "rb"))
    backend = _command_backend(run, timeout=5.0)

    def stop_tree(pid: int) -> None:
        run.stop_tree(pid)
        os.close(write_end)  # the worker dies with the tree and the pipe ends
    backend._stop_tree = stop_tree

    started = time.monotonic()
    completion = backend.complete(image, "prompt", 10)

    assert completion.text == ID_OK
    assert time.monotonic() - started < 2, "the reply was held to the timeout"
    (process,) = run.processes
    assert run.stopped == [(process.pid, None)], (
        "the tree is stopped by the command's pid while it is still the command's own, before the reap")
    assert process.returncode == 0, "the command was reaped last"


def test_command_is_selectable_by_config_and_flag_but_not_a_picker_choice():
    """`[model] backend = "command"` and `--backend command` select the seam;
    the plugin's picker learns it in card #423, so BACKEND_CHOICES, the
    engines the picker offers in the owner's order, is unchanged. It is
    local: no cloud retuning, no cost prompt, no cloud cache file."""
    assert providers.COMMAND == "command"
    assert providers.BACKEND_CHOICES == (*ENGINES, providers.SCRIPTED)
    assert providers.COMMAND in providers.LOCAL_BACKENDS
    assert not providers.is_cloud_primary(_cfg(model={"backend": "command", "command": COMMAND}))


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


@pytest.mark.parametrize("shim", [r"C:\Users\me\AppData\Roaming\npm\fake-vlm.cmd",
                                  r"C:\Tools\fake-vlm.BAT"], ids=["cmd", "bat"])
def test_command_resolving_to_a_batch_shim_is_refused_and_names_the_real_entry(
    monkeypatch, no_ambient_keys, no_ambient_ollama, shim
):
    """Given the template's first element resolves to a `.cmd` or `.bat`
    file (an npm-installed shim; any case), when the backend is asked for,
    then it is refused up front through the same shape, naming the file
    found and the fix: name the program's real entry in `[model] command`,
    its `.exe` or `node` and the script the shim wraps. Windows launches a
    batch file through cmd.exe whatever subprocess is told (Python's own
    subprocess docs, Security Considerations), so the prompt, with its
    newlines, quotes and braces, would be parsed by a shell rather than
    delivered as one argument, which is the promise the argv list makes."""
    monkeypatch.setattr(providers.shutil, "which", lambda name: shim)
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg(model={"backend": "command", "command": COMMAND}))
    message = str(err.value)
    assert shim in message, message
    assert "cmd.exe" in message, message
    assert "[model] command" in message and ".exe" in message and "node" in message, message
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
    assert backend.name == shlex.join(COMMAND)


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


_RUNAWAY_SCRIPT = '''#!{python}
"""A CLI that never stops writing to one stream: a broken program, not an
answer, however long the timeout. Its pid goes to a file so the test can
look for it afterwards."""
import os
import sys

with open({pid_file!r}, "w") as handle:
    handle.write(str(os.getpid()))
stream = getattr(sys, {stream!r})
while True:
    stream.write("x" * 65536)
    stream.flush()
'''

_LAUNCHER_SCRIPT = '''#!{python}
"""A CLI that hands the work to a worker, the way one wrapping a language
server or a daemon does: it waits for the worker (`waits`), or prints its
reply and exits at once, leaving the worker running with the CLI's own
stdout and stderr. The worker's pid goes to a file so the test can look
for the worker after the CLI is stopped or has exited."""
import subprocess
import sys

worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open({pid_file!r}, "w") as handle:
    handle.write(str(worker.pid))
if {waits}:
    worker.wait()
else:
    print({identification!r})
'''


def _fake_cli(monkeypatch, tmp_path, *, exit_code: int = 0, stderr: str = "",
              script: str = _FAKE_CLI_SCRIPT, **fields) -> list[str]:
    """Write FAKE_CLI, an executable Python script, into a folder put first
    on PATH, and return the config template that runs it by its bare name:
    the real shutil.which, the real subprocess, no shell. `exit_code`
    non-zero makes it fail after reading its arguments, saying `stderr`.
    `script` is another body in place of the answering CLI's, with `fields`
    filled in."""
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    path = folder / FAKE_CLI
    path.write_text(script.format(
        python=sys.executable, routing=ROUTING_OK, identification=ID_OK,
        exit_code=exit_code, stderr=stderr, **fields,
    ), encoding="utf-8")
    path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    assert shutil.which(FAKE_CLI) == str(path)
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
        ScriptedBackend([ROUTING_OK, ID_OK], name=shlex.join(command)), config
    ).identify(photos / PHOTO)

    assert result.status == "ok", result.error
    assert result.model == shlex.join(command)
    assert result.identification == expected.identification
    assert result.taxon_routing == expected.taxon_routing
    assert [c.common_name for c in result.identification.ranked()] == [
        "Tricolored Heron", "Little Blue Heron"]
    assert result.identification.top().scientific_name == "Egretta tricolor"


def _gone(pid: int, within: float) -> bool:
    """Whether process `pid` is gone (or a zombie no longer running) within
    `within` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@posix_only
def test_command_backend_timeout_stops_the_worker_the_command_started(monkeypatch, tmp_path):
    """At the real boundary: the command starts a worker and waits for it,
    the way a CLI wrapping a daemon does, and neither answers within the
    timeout. Then the run is a TimeoutError as before, and the worker is
    gone too: stopping only the command would leave a worker per timed-out
    frame running while the batch goes on."""
    pid_file = tmp_path / "worker.pid"
    command = _fake_cli(monkeypatch, tmp_path, script=_LAUNCHER_SCRIPT,
                        pid_file=str(pid_file), waits=True)
    backend = providers.build_primary_backend(
        _cfg(model={"backend": "command", "command": command, "timeout_seconds": 1}))
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")

    try:
        with pytest.raises(TimeoutError) as err:
            backend.complete(image, "prompt", 10)
        assert "fake-vlm did not answer within 1s" in str(err.value)
        worker = int(pid_file.read_text(encoding="utf-8"))
        assert _gone(worker, within=10.0), f"worker {worker} is still running after the timeout"
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), 9)


@posix_only
def test_command_backend_uses_the_reply_of_a_command_that_exits_leaving_a_worker(monkeypatch, tmp_path):
    """At the real boundary: the command prints its reply, starts a worker
    that inherits its stdout and stderr, and exits 0 at once, the way a
    CLI that leaves a helper behind does. Its exit ends its answer: the
    reply is used and the call returns well inside the timeout (the old
    outcome was a TimeoutError at the whole timeout, advising a longer one
    that could not help, the reply discarded), the worker is gone, and
    the group was stopped by a pid that was still the command's own, its
    exited process unreaped (a signal 0 still reaches it)."""
    pid_file = tmp_path / "worker.pid"
    command = _fake_cli(monkeypatch, tmp_path, script=_LAUNCHER_SCRIPT,
                        pid_file=str(pid_file), waits=False)
    backend = providers.build_primary_backend(
        _cfg(model={"backend": "command", "command": command, "timeout_seconds": 4}))
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    killpg, leader_unreaped = os.killpg, []

    def spy(pgid, sig):
        try:
            os.kill(pgid, 0)
            leader_unreaped.append(True)
        except ProcessLookupError:
            leader_unreaped.append(False)
        return killpg(pgid, sig)
    monkeypatch.setattr(os, "killpg", spy)

    try:
        started = time.monotonic()
        completion = backend.complete(image, "prompt", 10)
        assert time.monotonic() - started < 2, "the reply was held to the timeout"
        assert completion.text.strip() == ID_OK
        worker = int(pid_file.read_text(encoding="utf-8"))
        assert _gone(worker, within=10.0), f"worker {worker} outlived the command's exit"
        assert leader_unreaped == [True], "the tree was stopped by a pid the command no longer held"
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), 9)


def test_command_backend_sees_the_exit_without_reaping_the_command():
    """At the real boundary, the sight of the exit: a real child that has
    ended is seen as exited at once, and one still running is not, within
    the time given; on POSIX the child is not reaped by the look (its
    returncode is still None afterwards, so its pid is still its own and
    its group's until the backend reaps it last); on Windows the Popen
    handle keeps the pid reserved and the look may reap."""
    backend = CommandBackend(["fake-vlm", "{image}", "{prompt}"])
    ended = subprocess.Popen([sys.executable, "-c", "pass"], **CommandBackend.OWN_GROUP)
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               **CommandBackend.OWN_GROUP)
    try:
        assert backend._exited(ended, 10.0) is True
        if sys.platform != "win32":
            assert ended.returncode is None, "the look reaped the command"
        started = time.monotonic()
        assert backend._exited(running, 0.2) is False
        assert 0.2 <= time.monotonic() - started < 2
    finally:
        running.kill()
        running.wait()
        ended.wait()


@posix_only
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_command_backend_stops_a_runaway_command_before_the_timeout(monkeypatch, tmp_path, stream):
    """At the real boundary: the command writes without end to one stream,
    under a timeout of a minute. It is stopped as soon as it passes the
    ceiling, seconds in, not read into memory until the timeout: the frame's
    error names the program, the stream and the ceiling, and the process is
    gone."""
    pid_file = tmp_path / "runaway.pid"
    command = _fake_cli(monkeypatch, tmp_path, script=_RUNAWAY_SCRIPT,
                        pid_file=str(pid_file), stream=stream)
    backend = providers.build_primary_backend(
        _cfg(model={"backend": "command", "command": command, "timeout_seconds": 60}))
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")

    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError) as err:
            backend.complete(image, "prompt", 10)
        assert time.monotonic() - started < 20, "the runaway was read until the timeout"
        for named in ("fake-vlm", stream, str(CommandBackend.MAX_OUTPUT_BYTES)):
            assert named in str(err.value), str(err.value)
        pid = int(pid_file.read_text(encoding="utf-8"))
        assert _gone(pid, within=10.0), f"the runaway {pid} is still running"
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), 9)


def test_command_not_installed_fires_before_any_image_is_read(tmp_path, capsys, no_ambient_ollama):
    """Card #420, Done-when 2, at the real boundary: a command nothing on
    this machine is called, and the folder's one image is a link to
    nowhere, so opening it would fail loudly. The CLI exits 3 on the
    not-installed message, naming the command, and never mentions the file:
    the check ran before any image was read. `--no-local-config` keeps a
    developer's own melampus.local.toml keys out of the run (Done-when 3)."""
    from melampus.cli import main

    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")
    settings = _command_settings(tmp_path, ["melampus-no-such-command-420", "{image}", "{prompt}"])

    code = main([str(folder), "--config", str(settings), "--no-local-config",
                 "--cache", str(tmp_path / "cache.jsonl")])

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
    frame in turn. Nothing is cached for the frame in flight.
    `--no-local-config` keeps a developer's own melampus.local.toml keys
    (a `[run] profile`, a `[model] timeout_seconds`) out of the fake's run
    (Done-when 3)."""
    from melampus.cli import main

    command = _fake_cli(monkeypatch, tmp_path, exit_code=2, stderr="not signed in\nrun fake-vlm login")
    out = tmp_path / "results.json"

    code = main([str(photos), "--backend", "command", "--config", str(_command_settings(tmp_path, command)),
                 "--no-local-config",
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
    result with the candidates, attributed to the template.
    `--no-local-config` keeps a developer's own melampus.local.toml keys out
    of the run: a `[run] profile` there would swap the routing prompt for one
    the fake does not recognise (Done-when 3)."""
    from melampus.cli import main

    command = _fake_cli(monkeypatch, tmp_path)
    out = tmp_path / "results.json"

    code = main([str(photos), "--config", str(_command_settings(tmp_path, command)),
                 "--no-local-config",
                 "--cache", str(tmp_path / "cache.jsonl"), "--json-out", str(out)])

    err = capsys.readouterr().err
    assert code == 0, err
    assert f"loading {shlex.join(command)}" in err, err
    assert "cloud default" not in err, f"command is local; nothing was retuned for a cloud:\n{err}"
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert result["file"] == PHOTO
    assert result["status"] == "ok"
    assert result["model"] == shlex.join(command)
    assert [c["common_name"] for c in result["identification"]["candidates"]] == [
        "Tricolored Heron", "Little Blue Heron"]
