"""The backend seam as a setting — the Windows / cloud-primary path.

The contract under test: `[model] backend` selects who answers, defaults keep the
Mac local-first path byte-for-byte unchanged, cloud selection fails fast and
clearly (missing SDK, missing key, unknown name), and the MLX-shaped defaults are
retuned for a cloud primary without ever overriding an explicit setting.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import http.client
import io
import json
import os
import select
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
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import (
    PHOTO,
    REAL_CLAUDE_CODE_VERDICT,
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


@pytest.fixture()
def link_to_nowhere(tmp_path: Path) -> Path:
    """A folder whose one image is a symlink to nowhere, so opening it would
    fail loudly: the proof that a refusal fired before any image was read."""
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")
    return folder


def assert_no_image_was_touched(err: str, check: str) -> None:
    """The stderr of a run on `link_to_nowhere` never mentions the file: the
    `check` (named for the message) ran before any image was read."""
    for about_the_file in ("nowhere", "does-not-exist", "No such file", "unreadable"):
        assert about_the_file not in err, f"the image was touched before the {check}:\n{err}"


def test_detection_lists_the_engines_in_the_owners_order_then_claude_code(
    no_ambient_keys, no_ambient_ollama
):
    """The list the dialog (card #405) will show: one verdict per engine, in
    the order BACKEND_CHOICES names them, then claude-code (card #421;
    the picker learns it in #423), never the test fake."""
    verdicts = providers.detect_engines()
    assert [v.engine for v in verdicts] == [*ENGINES, providers.CLAUDE_CODE]
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


def test_ollama_not_running_fires_before_any_image_is_read(
    monkeypatch, tmp_path, capsys, link_to_nowhere
):
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

    code = main([
        str(link_to_nowhere), "--backend", "ollama", "--no-local-config",
        "--cache", str(tmp_path / "cache.jsonl"),
    ])

    err = capsys.readouterr().err
    assert code == 3, err
    assert f"No Ollama server at http://127.0.0.1:{port}" in err
    assert providers.OLLAMA_INSTALL in err
    assert_no_image_was_touched(err, "Ollama check")


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
    assert [v["engine"] for v in verdicts] == [*ENGINES, providers.CLAUDE_CODE]
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
        (["fake-vlm-{image}", "--prompt", "{prompt}"], "{image}"),
        (["fake-vlm-{prompt}", "--image", "{image}"], "{prompt}"),
    ],
    ids=["no-image", "no-prompt", "neither", "image-only-in-program", "prompt-only-in-program"],
)
def test_command_template_without_a_placeholder_is_refused_at_config_load(command, missing):
    """A template that never receives the image, or never asks the question,
    cannot answer anything: refused when the config loads, naming the
    placeholder it lacks, not per frame after the run has started. The first
    element is the program, which `_argv` replaces with the resolved
    executable, so a placeholder there never reaches the program: it counts
    only in the arguments after it."""
    with pytest.raises(ValueError) as err:
        _cfg(model={"backend": "command", "command": command})
    assert f"has no argument carrying {missing}" in str(err.value), str(err.value)


@pytest.mark.parametrize(
    ("command", "position", "codepoint"),
    [
        (["fake-vlm\x1b[2J", "--image", "{image}", "--prompt", "{prompt}"], 0, "U+001B"),
        (["fake-vlm", "--image", "{image}", "--prompt\n", "{prompt}"], 3, "U+000A"),
        (["fake-vlm", "--image", "{image}", "--prompt", "{prompt}", "--label\x07"], 5, "U+0007"),
    ],
    ids=["escape-in-program", "newline-in-argument", "bell-in-last-argument"],
)
def test_command_template_with_a_control_character_is_refused_at_config_load(
    command, position, codepoint
):
    """Codex round 18 (config.py:130), security: the template is printed as
    it is, in the CLI's `loading ...` line, in every error message the
    backend raises (`program`, its first element) and in the refusals
    naming the program, so an element carrying an escape sequence, a line
    break or a bell would reach the terminal, the log and the frame's error
    record through them. Given an element with any character that is not
    printable (str.isprintable, the one rule `plain` applies to what a
    program wrote), when the config loads, then it is refused there,
    naming the element's position and the character, in the shape the
    placeholder refusal uses, so nothing displayed later can carry one."""
    with pytest.raises(ValueError) as err:
        _cfg(model={"backend": "command", "command": command})
    clause = next(line for line in str(err.value).splitlines() if "[model] command element" in line)
    assert f"[model] command element {position} carries a character that is not printable ({codepoint})" in clause, clause
    assert "escape sequence" in clause and "line break" in clause, clause
    assert all(c.isprintable() for c in clause), clause


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


def _given(monkeypatch, module, **constants) -> None:
    """Give `module` each named constant where this platform has none, so a
    branch written for another platform can run here; where the platform
    has the constant, its real value is kept, so the test means the same
    thing on every platform and asserts against the real names. The value
    given is the other platform's own."""
    for name, value in constants.items():
        monkeypatch.setattr(module, name, getattr(module, name, value), raising=False)


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
    stop every process it started and not just the first (that the
    config's timeout is the wait's ceiling is proven by the timeout
    tests, not here)."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK)
    backend = _command_backend(run)
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
    assert run.stopped == [(process.pid, None)], (
        "what the command started is stopped at its exit, by its pid while it is still the command's own")


def test_command_backend_hands_the_program_the_real_path_of_the_image(tmp_path):
    """The path that crosses the command line is the staged file's real
    one, symlinks resolved: on macOS the temp folder is under /var, a link
    to /private/var, and a program that checks a path-scoped permission
    rule against both the path as given and where it resolves (Claude
    Code's allow rules, permissions § Read and Edit) must see one path
    that is the same either way."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "image.jpg").write_bytes(b"jpeg")
    (tmp_path / "link").symlink_to(real)
    through_link = tmp_path / "link" / "image.jpg"
    run = _FakeRun(stdout=ID_OK)

    _command_backend(run).complete(through_link, "what is this?", 10)

    ((argv, _),) = run.calls
    assert argv[2] == str(real.resolve() / "image.jpg")
    assert "link" not in argv[2]


def test_command_backend_runs_the_program_in_the_staged_images_folder(tmp_path):
    """Where the program runs (security review, round 2): its working
    directory is the staged image's own folder, the temporary one that
    holds that file and nothing else, never melampus's cwd as inherited
    from wherever it was launched (from Lightroom that is the app's, `/`
    on a Mac). A program that reads files freely inside its working
    directory and asks for anything outside it (Claude Code's Read tool:
    permissions § Working directories, "access to files in the directory
    where you launched it") is thereby confined to the one file it was
    handed; a photograph carrying text that asks for ~/.ssh or .env gets
    that read refused whatever folder melampus was started in. The folder
    is the real one, symlinks resolved, like the path itself."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "image.jpg").write_bytes(b"jpeg")
    (tmp_path / "link").symlink_to(real)
    run = _FakeRun(stdout=ID_OK)

    _command_backend(run).complete(tmp_path / "link" / "image.jpg", "what is this?", 10)

    ((_, kwargs),) = run.calls
    assert kwargs["cwd"] == str(real.resolve())


def test_command_backend_resolves_a_program_named_by_a_relative_path_before_moving(tmp_path, monkeypatch):
    """A `[model] command` whose program is a relative path (`./tools/vlm`,
    which shutil.which hands back as given) must still be the file the user
    named once the run moves into the staged image's folder: the argv
    carries it as an absolute path, resolved against melampus's cwd. A bare
    name is left for PATH, as before."""
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK)
    relative = os.path.join(".", "tools", "vlm")

    CommandBackend([relative, "{image}", "{prompt}"], executable=relative, run=run).complete(image, "p", 10)
    CommandBackend(["vlm", "{image}", "{prompt}"], run=run).complete(image, "p", 10)

    (by_path, _), (by_name, _) = run.calls
    assert by_path[0] == str(tmp_path / "tools" / "vlm")
    assert by_name[0] == "vlm"


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


def test_command_backend_closes_both_pipes_after_a_completion(tmp_path):
    """The two pipes Popen opens are closed by the backend once their readers
    have read them to their ends, not left to the garbage collector (which
    `python -X dev` reports as an unclosed file per pipe per frame): the
    Ollama backend scopes its response the same way, and subprocess.run its
    child's pipes."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK, stderr="warning: slow")
    backend = _command_backend(run)

    backend.complete(image, "prompt", 900)

    (process,) = run.processes
    assert process.stdout.closed and process.stderr.closed, "the pipes were left to the garbage collector"


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


@pytest.mark.parametrize(
    ("run", "expected", "said"),
    [
        (_FakeRun(returncode=2, stderr="\x1b[2J\rnot logged in\x07\n\x9brun `fake-vlm login`\tfirst\x85"),
         CommandFailed, "fake-vlm exited 2: [2J / not logged in / run `fake-vlm login` first"),
        (_FakeRun(stdout="  \n", stderr="\x1b[31musage:\x1b[0m fake-vlm\x07 ...\x9b"),
         RuntimeError, "fake-vlm printed nothing on stdout: [31musage: [0m fake-vlm ..."),
        (_FakeRun(returncode=1, stderr="x" * 5000),
         CommandFailed, "fake-vlm exited 1: " + "x" * CommandBackend.MAX_ERROR_BYTES),
        (_FakeRun(returncode=2, stderr="\x07\n\x1b\n\x00\nnot logged in\n"),
         CommandFailed, "fake-vlm exited 2: not logged in"),
        (_FakeRun(returncode=2, stderr="\x1b[?25l\n\x07\n\x00\nnot logged in\nrun `fake-vlm login` first\n"),
         CommandFailed, "fake-vlm exited 2: [?25l / not logged in / run `fake-vlm login` first"),
    ],
    ids=["non-zero", "empty-stdout", "bounded", "control-only-lines", "controls-among-words"],
)
def test_command_backend_keeps_only_the_printable_words_of_stderr(tmp_path, run, expected, said):
    """The same, for the program's stderr: what it wrote lands in the frame's
    error record, the log and the terminal, so an escape sequence in it
    would clear the screen or recolour the terminal, a BEL would ring it,
    and a C1 control or a disguised line break would fake a line of the
    log. Only its printable characters reach the message, by the one rule
    the Ollama backend's messages read through (`plain`): each kept line
    is words, at most MAX_ERROR_BYTES of them, and the three-line cap and
    the " / " joining stay as they are."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _command_backend(run)
    with pytest.raises(expected) as err:
        backend.complete(image, "prompt", 10)
    message = str(err.value)
    assert message == said, message
    assert message.isprintable(), message
    assert not any(control in message for control in "\x1b\x07\x9b\x85"), message


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


def test_command_backend_counts_starting_the_program_against_the_timeout(tmp_path):
    """Codex round 21 (backend.py:999, :1002): the timeout is the ceiling on
    the call as a whole, starting the program included. A launch that is
    slow (a program that takes long to exec, a PATH on a slow volume)
    counts against it, so a launch that consumed the whole timeout, 0.3s
    here against a 0.6s launch, is reported as the timeout it is, not
    answered as a success with a fresh full timeout after it. The fake's
    process factory sleeps past the timeout before returning a process
    that has already exited with a good reply."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK)
    backend = _command_backend(run, timeout=0.3)

    def slow_launch(argv, **kwargs):
        time.sleep(0.6)
        return run(argv, **kwargs)
    backend._run = slow_launch

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not answer within 0.3s"):
        backend.complete(image, "prompt", 10)

    assert time.monotonic() - started < 1.5, "the launch was given a fresh timeout on top of its own"
    (process,) = run.processes
    assert process.returncode is not None, "the command was not reaped"


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


@pytest.mark.parametrize(
    ("timeout", "at_least", "under"),
    [(5.0, 4.5, 7), (1.5, 1.5, 2.5)],
    ids=["five-second-bound-wins", "timeout-wins"],
)
def test_command_backend_leaves_a_pipe_still_held_past_the_reader_bound_to_its_holder(
        tmp_path, timeout, at_least, under):
    """A pipe still held when the readers' bound runs out (by something the
    tree stop could not reach: a daemon that left the session) is left to
    its holder, not closed from here: closing a BufferedReader from another
    thread waits for the read1 in flight to return, which is a wait on the
    holder for as long as it lives, past the config's timeout. The reply
    is used at the bound, the pipe that did end is closed, and the one
    still held is not. The config's timeout is the ceiling on the call as
    a whole, the readers' bound included: a command that exited in time
    and left its stdout held by something the tree stop could not reach
    gives the readers until the timeout or five seconds, whichever comes
    first, not five seconds past a shorter timeout. The fake's stdout is
    a real pipe whose write end the test holds for longer than either
    bound, so a close that waits shows as elapsed time rather than a
    hang; under a timeout longer than five seconds the reply is used at
    the five seconds, and under one shorter, 1.5s here, at the timeout,
    the call never nearing the five seconds a bound clamped to nothing
    would take."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    read_end, write_end = os.pipe()
    os.write(write_end, ID_OK.encode("utf-8"))
    stdout = os.fdopen(read_end, "rb")
    run = _FakeRun(stdout=stdout, stderr="warning: slow")
    backend = _command_backend(run, timeout=timeout)
    held = [write_end]

    def let_go() -> None:
        os.close(held.pop())  # the holder dies and the pipe ends
    holder = threading.Timer(8.0, let_go)
    holder.start()
    try:
        started = time.monotonic()
        completion = backend.complete(image, "prompt", 10)
        took = time.monotonic() - started

        assert completion.text == ID_OK
        assert at_least <= took < under, f"the call took {took:.2f}s: held past the reader bound"
        (process,) = run.processes
        assert process.stderr.closed, "the pipe read to its end is closed"
        assert not process.stdout.closed, "the pipe still held was closed from under its reader"
        assert process.returncode == 0, "the command was reaped last"
    finally:
        holder.cancel()
        holder.join()
        if held:
            let_go()
        stdout.close()


@pytest.mark.parametrize("system_root", [r"D:\Win", None], ids=["SystemRoot", "default"])
def test_command_backend_stops_a_tree_on_windows_with_the_system_taskkill(monkeypatch, system_root):
    """`_stop_tree` on Windows runs taskkill by its absolute path under
    System32 (SystemRoot's, or C:\\Windows's), never by bare name, which
    CreateProcess would look for in the current directory before System32:
    a taskkill.exe planted where melampus was started from would run with
    its rights the first time a command timed out. `/T /F /PID <pid>`,
    nothing on stdin, the exit code not checked (a tree already gone is
    nothing to do). Runs on every platform: the platform and the run are
    faked."""
    monkeypatch.setattr(sys, "platform", "win32")
    if system_root is None:
        monkeypatch.delenv("SystemRoot", raising=False)
    else:
        monkeypatch.setenv("SystemRoot", system_root)
    calls: list[tuple[list[str], dict]] = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 128, b"", b"ERROR: The process \"4242\" not found.")
    monkeypatch.setattr(subprocess, "run", run)

    CommandBackend(["fake-vlm", "{image}", "{prompt}"])._stop_tree(4242)

    ((argv, kwargs),) = calls
    assert argv == [rf"{system_root or r'C:\Windows'}\System32\taskkill.exe", "/T", "/F", "/PID", "4242"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["check"] is False and kwargs["capture_output"] is True


def test_command_backend_stops_a_tree_on_posix_by_the_group_the_command_heads(monkeypatch):
    """`_stop_tree` on POSIX: SIGKILL through os.killpg to the group whose
    id is the command's pid (OWN_GROUP started it in its own session). A
    group already gone (ESRCH) and one holding nothing but the command's
    own exited process (EPERM, macOS's answer) are nothing to do; any
    other failure is raised. Runs on every platform: os.killpg is faked,
    and SIGKILL is given where there is none."""
    monkeypatch.setattr(sys, "platform", "linux")
    _given(monkeypatch, signal, SIGKILL=9)
    calls: list[tuple[int, int]] = []
    answer: list[BaseException | None] = [None]

    def killpg(pgid, sig):
        calls.append((pgid, sig))
        if answer[0] is not None:
            raise answer[0]
    monkeypatch.setattr(os, "killpg", killpg, raising=False)
    backend = CommandBackend(["fake-vlm", "{image}", "{prompt}"])

    backend._stop_tree(4242)
    assert calls == [(4242, signal.SIGKILL)]
    for answer[0] in (ProcessLookupError(3, "No such process"), PermissionError(1, "Operation not permitted")):
        backend._stop_tree(4242)
    assert calls == [(4242, signal.SIGKILL)] * 3
    answer[0] = OSError(22, "Invalid argument")
    with pytest.raises(OSError):
        backend._stop_tree(4242)


def test_command_backend_starts_the_command_in_its_own_process_group_on_windows(monkeypatch, tmp_path):
    """OWN_GROUP on Windows: the command is started with
    CREATE_NEW_PROCESS_GROUP, the flag `taskkill /T` walks from, and not
    with start_new_session, which Windows has no idea of. Runs on every
    platform: the platform and the flag are faked."""
    monkeypatch.setattr(sys, "platform", "win32")
    _given(monkeypatch, subprocess, CREATE_NEW_PROCESS_GROUP=0x200)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    run = _FakeRun(stdout=ID_OK)

    _command_backend(run).complete(image, "prompt", 10)

    ((_, kwargs),) = run.calls
    assert kwargs["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    assert "start_new_session" not in kwargs


def test_command_backend_sees_the_exit_through_wait_on_windows(monkeypatch):
    """`_exited` on Windows is Popen.wait with the step as its timeout: the
    Popen handle keeps the pid reserved, so reaping there is safe and the
    order does not matter. Runs on every platform: the platform is faked
    and the process is the fake."""
    monkeypatch.setattr(sys, "platform", "win32")
    backend = CommandBackend(["fake-vlm", "{image}", "{prompt}"])
    ended = _FakeProcess([], _FakeRun())
    hung = _FakeProcess([], _FakeRun(hangs=True))

    assert backend._exited(ended, 0.01) is True
    assert backend._exited(hung, 0.01) is False
    assert ended.waited == [0.01] and hung.waited == [0.01]


def test_command_backend_sees_the_exit_through_kqueue_and_names_what_each_event_is(monkeypatch):
    """`_exited` where there is kqueue (macOS) registers a one-shot
    NOTE_EXIT on the command's pid and waits the step for one event, on a
    queue of its own closed after the look. Three answers can come back
    and each is named: no event is not yet; a NOTE_EXIT event is the exit,
    seen during the wait; an EV_ERROR event carrying ESRCH is the command
    already exited before the look (XNU's proc_find does not see an exited
    process, so the registration is refused, and for this process's own
    unreaped child that can only mean it has exited), and is the exit too.
    An EV_ERROR event carrying any other errno is that error, raised as
    the OSError it is, never counted as the exit. Runs on every platform:
    kqueue and kevent are faked, with the constants where there are none."""
    monkeypatch.setattr(sys, "platform", "darwin")
    _given(monkeypatch, select, KQ_FILTER_PROC=-5, KQ_EV_ADD=1, KQ_EV_ONESHOT=0x10, KQ_EV_CLEAR=0x20,
           KQ_EV_EOF=0x8000, KQ_EV_ERROR=0x4000, KQ_NOTE_EXIT=0x80000000)
    monkeypatch.setattr(select, "kevent", lambda *fields: types.SimpleNamespace(fields=fields), raising=False)
    controls: list[tuple[list, int, float]] = []
    answer: list[list] = [[]]
    closed: list[bool] = []

    class kqueue:
        def control(self, changes, max_events, timeout):
            controls.append((changes, max_events, timeout))
            return answer[0]

        def close(self):
            closed.append(True)
    monkeypatch.setattr(select, "kqueue", kqueue, raising=False)
    backend = CommandBackend(["fake-vlm", "{image}", "{prompt}"])
    process = _FakeProcess([], _FakeRun())

    assert backend._exited(process, 0.05) is False, "no event is not yet"
    # The flags a real NOTE_EXIT event carries (0x8031 on macOS): the registration's own ADD and
    # ONESHOT, CLEAR and EOF set by the kernel, and no EV_ERROR, which is all the branch reads of them.
    answer[0] = [types.SimpleNamespace(flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT | select.KQ_EV_CLEAR
                                       | select.KQ_EV_EOF, fflags=select.KQ_NOTE_EXIT, data=0)]
    assert backend._exited(process, 0.05) is True, "NOTE_EXIT is the exit, seen during the wait"
    answer[0] = [types.SimpleNamespace(flags=select.KQ_EV_ERROR | select.KQ_EV_ONESHOT, fflags=0,
                                       data=errno.ESRCH)]
    assert backend._exited(process, 0.05) is True, "EV_ERROR with ESRCH is the command already exited"
    answer[0] = [types.SimpleNamespace(flags=select.KQ_EV_ERROR | select.KQ_EV_ONESHOT, fflags=0,
                                       data=errno.ENOMEM)]
    with pytest.raises(OSError) as err:
        backend._exited(process, 0.05)
    assert err.value.errno == errno.ENOMEM, "any other EV_ERROR is the error it is, not the exit"
    assert len(controls) == 4 and len(closed) == 4, "one queue per look, closed after it"
    for changes, max_events, timeout in controls:
        (registered,) = changes
        assert registered.fields == (4242, select.KQ_FILTER_PROC, select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                                     select.KQ_NOTE_EXIT), "one-shot NOTE_EXIT on the command's pid"
        assert (max_events, timeout) == (1, 0.05), "one event, within the step"
    assert process.waited == [], "kqueue, never Popen.wait, which reaps"


def test_command_backend_sees_the_exit_through_waitid_where_there_is_no_kqueue(monkeypatch):
    """`_exited` on a POSIX without kqueue (Linux) asks waitid for the
    command by pid with WEXITED, WNOWAIT (seen, not reaped) and WNOHANG:
    None is not yet, anything else is exited. It looks first, so a command
    already exited is seen at once (as the kqueue and Windows looks see
    it), and only when the command has not exited sleeps the step and
    looks once more, so an exit during the step is seen at its end.
    Runs on every platform: kqueue is taken away, waitid is faked, and its
    constants are given where there are none."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delattr(select, "kqueue", raising=False)
    _given(monkeypatch, os, P_PID=1, WEXITED=4, WNOWAIT=0x1000000, WNOHANG=1)
    calls: list[tuple[int, int, int]] = []
    answer: list[object] = [None]

    def waitid(idtype, pid, options):
        calls.append((idtype, pid, options))
        return answer[0]
    monkeypatch.setattr(os, "waitid", waitid, raising=False)
    backend = CommandBackend(["fake-vlm", "{image}", "{prompt}"])
    process = _FakeProcess([], _FakeRun())

    started = time.monotonic()
    assert backend._exited(process, 0.05) is False
    assert time.monotonic() - started >= 0.05, "a not-yet look waits the step"
    answer[0] = object()  # a siginfo: the process has exited
    started = time.monotonic()
    assert backend._exited(process, 0.05) is True
    assert time.monotonic() - started < 0.05, "an exited command is seen before the step has passed"
    assert calls == [(os.P_PID, 4242, os.WEXITED | os.WNOWAIT | os.WNOHANG)] * 3, (
        "a not-yet look looks, sleeps the step, and looks once more; an exited look looks once")
    assert process.waited == [], "waitid, never Popen.wait, which reaps"


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


def test_command_resolved_through_a_path_entry_with_a_control_character_is_named_in_printable_words(
    monkeypatch, no_ambient_keys, no_ambient_ollama
):
    """Codex round 18 (providers.py:324), security: the resolved path the
    batch-shim refusal names is the template's first element joined to a
    PATH directory, and PATH is inherited from whatever launched melampus
    (a supervisor, the plugin's host), so a directory carrying an escape
    sequence would reach the terminal through the message even though the
    config's element is printable. Given shutil.which resolves the program
    under such a directory to a `.cmd`, when the backend is asked for,
    then the refusal names the path in printable words only, through
    `plain`, the one rule for text the program's side wrote."""
    shim = "C:\\Tools\x1b[2J\\fake-vlm.cmd"
    monkeypatch.setattr(providers.shutil, "which", lambda name: shim)
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg(model={"backend": "command", "command": COMMAND}))
    clause = next(line for line in str(err.value).splitlines() if "resolves to" in line)
    assert "C:\\Tools [2J\\fake-vlm.cmd, a batch file" in clause, clause
    assert all(c.isprintable() for c in clause), clause


def test_command_in_a_process_that_ignores_sigchld_is_refused_and_names_the_fix(
    monkeypatch, no_ambient_keys, no_ambient_ollama
):
    """Given the process melampus runs in ignores SIGCHLD (SIG_IGN, inherited
    across exec from whatever launched it: a supervisor, a Python parent
    that set it to avoid zombies), when the command backend is asked for,
    then it is refused up front through the same shape, naming SIGCHLD and
    the fix (start melampus from a shell, or restore the default), because
    the kernel reaps such a process's children the moment they exit: the
    command's exit could only be seen after its pid was freed, and the tree
    it started would be stopped by a number that may be someone else's by
    then. Refused once at exit 3 as a missing program is, not once per
    frame. Runs on every platform: the disposition is faked, and the
    signal's name is given where there is none."""
    monkeypatch.setattr(providers.shutil, "which", lambda name: f"/opt/fake/bin/{name}")
    _given(monkeypatch, signal, SIGCHLD=20)
    asked: list[int] = []

    def getsignal(signalnum):
        asked.append(signalnum)
        return signal.SIG_IGN
    monkeypatch.setattr(signal, "getsignal", getsignal)
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg(model={"backend": "command", "command": COMMAND}))
    message = str(err.value)
    assert asked == [signal.SIGCHLD]
    assert "ignores SIGCHLD" in message, message
    assert "shell" in message and "default" in message, message
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


def _script_on_path(monkeypatch, tmp_path, name: str, text: str) -> Path:
    """Write `text` as an executable script called `name` into a folder put
    first on PATH, so the real shutil.which finds it by its bare name ahead
    of anything else and the real subprocess runs it, no shell."""
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    script = folder / name
    script.write_text(text, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    assert shutil.which(name) == str(script)
    return script


def _real_detection(monkeypatch) -> None:
    """Run the real Claude Code detection against what this test put on
    PATH: conftest's autouse fixture stubs it out for every test, so one
    that has placed its own `claude` (or none) restores the real one."""
    monkeypatch.setattr(providers, "claude_code_verdict", REAL_CLAUDE_CODE_VERDICT)


def _fake_cli(monkeypatch, tmp_path, *, exit_code: int = 0, stderr: str = "",
              script: str = _FAKE_CLI_SCRIPT, **fields) -> list[str]:
    """Put FAKE_CLI on PATH and return the config template that runs it by
    its bare name. `exit_code` non-zero makes it fail after reading its
    arguments, saying `stderr`. `script` is another body in place of the
    answering CLI's, with `fields` filled in."""
    _script_on_path(monkeypatch, tmp_path, FAKE_CLI, script.format(
        python=sys.executable, routing=ROUTING_OK, identification=ID_OK,
        exit_code=exit_code, stderr=stderr, **fields,
    ))
    return [FAKE_CLI, "--image", "{image}", "--prompt", "{prompt}", "--quiet"]


def _command_settings(tmp_path, command: list[str]) -> Path:
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'[model]\nbackend = "command"\ncommand = {json.dumps(command)}\n', encoding="utf-8")
    return settings


posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the fake CLI is a shebang script, and _gone's look is signal 0, which on Windows "
           "is CTRL_C_EVENT then TerminateProcess, not a probe, and never a ProcessLookupError",
)


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


#: The pids `_gone` has proven gone during the current test (the `pid_file`
#: fixture empties it at setup), so its teardown never signals one of them.
_PROVEN_GONE: set[int] = set()


def _gone(pid: int, within: float) -> bool:
    """Whether the pid `pid` is gone within `within` seconds: no process
    holds it, so a signal 0 finds nothing. Looked for at least once. A
    zombie is not gone: it holds its pid until its parent reaps it, and
    signal 0 reaches it (macOS answers it with success, as POSIX asks). A
    pid proven gone is recorded in `_PROVEN_GONE`, so the fixture's
    teardown never signals it: it may be another process's by then."""
    deadline = time.monotonic() + within
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _PROVEN_GONE.add(pid)
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


@pytest.fixture
def pid_file(tmp_path):
    """The file a fake CLI's script writes a pid into (its worker's, or its
    own) so the test can look for that process afterwards. On teardown,
    the process it names is killed if it is still there, which is the
    failing case. A pid the test proved gone with `_gone` is never
    signalled, whatever a look at teardown would answer: between the
    proof and the teardown it may have been given to a process of
    someone else's, and a fresh look by number could not tell."""
    _PROVEN_GONE.clear()
    path = tmp_path / "worker.pid"
    yield path
    if path.exists():
        pid = int(path.read_text(encoding="utf-8"))
        if pid not in _PROVEN_GONE and not _gone(pid, within=0.0):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 9)


@posix_only
def test_pid_file_teardown_never_signals_a_pid_the_test_proved_gone(monkeypatch, tmp_path):
    """The `pid_file` fixture's teardown stops a process a failing test left
    behind, and never a pid the test proved gone with `_gone`, however a
    look at teardown answers: between the assertion and the teardown that
    pid can be given to a process of someone else's, and a fresh look by
    number cannot tell the two apart. Driven here with the fixture's own
    generator, a real child proven gone, and os.kill faked to answer
    "alive" for its pid, as a recycled pid would; and, the other way, a
    pid never proven gone that answers alive is signalled, which is the
    failing case the teardown exists for."""
    signals: list[tuple[int, int]] = []

    def kill(pid, sig):
        signals.append((pid, sig))
    teardown = pid_file.__wrapped__(tmp_path)
    path = next(teardown)
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    path.write_text(str(ended.pid), encoding="utf-8")
    assert _gone(ended.pid, within=10.0)
    monkeypatch.setattr(os, "kill", kill)
    with pytest.raises(StopIteration):
        next(teardown)
    assert signals == [], f"the teardown signalled a pid the test proved gone: {signals}"

    teardown = pid_file.__wrapped__(tmp_path)
    path = next(teardown)
    path.write_text(str(ended.pid + 1), encoding="utf-8")
    with pytest.raises(StopIteration):
        next(teardown)
    assert signals == [(ended.pid + 1, 0), (ended.pid + 1, 9)], "a leftover never proven gone is stopped"


def _real_command_backend(monkeypatch, tmp_path, script: str, timeout: float, **fields) -> CommandBackend:
    """The backend the factory builds for the fake CLI `script` (a body for
    _fake_cli, `fields` filled in) under a `timeout` of that many seconds:
    the real shutil.which, the real subprocess, no shell."""
    command = _fake_cli(monkeypatch, tmp_path, script=script, **fields)
    return providers.build_primary_backend(
        _cfg(model={"backend": "command", "command": command, "timeout_seconds": timeout}))


@posix_only
def test_command_backend_timeout_stops_the_worker_the_command_started(monkeypatch, tmp_path, pid_file):
    """At the real boundary: the command starts a worker and waits for it,
    the way a CLI wrapping a daemon does, and neither answers within the
    timeout. Then the run is a TimeoutError as before, and the worker is
    gone too: stopping only the command would leave a worker per timed-out
    frame running while the batch goes on."""
    backend = _real_command_backend(monkeypatch, tmp_path, _LAUNCHER_SCRIPT, timeout=1,
                                    pid_file=str(pid_file), waits=True)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")

    with pytest.raises(TimeoutError) as err:
        backend.complete(image, "prompt", 10)

    assert "fake-vlm did not answer within 1s" in str(err.value)
    worker = int(pid_file.read_text(encoding="utf-8"))
    assert _gone(worker, within=10.0), f"worker {worker} is still running after the timeout"


@posix_only
def test_command_backend_closes_both_pipes_after_a_timeout_at_the_real_boundary(
        monkeypatch, tmp_path, pid_file):
    """At the real boundary, the promise of
    test_command_backend_closes_both_pipes_after_a_completion on the path
    that recurs in a batch (every frame of a program that hangs): a command
    that does not answer within the timeout is stopped with its tree, its
    two pipes end, the readers reach those ends, and both pipes are closed
    by the backend, not left to the garbage collector (`python -X dev`
    reports an unclosed file per pipe per timed-out frame). The Popen is
    recorded by wrapping the backend's process factory, and the pipes are
    looked at once the readers have reached the ends, within a short
    bound: the readers end within milliseconds of the stop, and the close
    is theirs to make whatever bound _wait gave them."""
    backend = _real_command_backend(monkeypatch, tmp_path, _LAUNCHER_SCRIPT, timeout=0.5,
                                    pid_file=str(pid_file), waits=True)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    launch, processes = backend._run, []

    def recording_launch(argv, **kwargs):
        processes.append(launch(argv, **kwargs))
        return processes[-1]
    backend._run = recording_launch

    try:
        with pytest.raises(TimeoutError):
            backend.complete(image, "prompt", 10)

        (process,) = processes
        ends = time.monotonic() + 2.0
        while not (process.stdout.closed and process.stderr.closed) and time.monotonic() < ends:
            time.sleep(0.05)
        assert process.stdout.closed and process.stderr.closed, (
            "the pipes of a timed-out command were left to the garbage collector")
    finally:
        for process in processes:
            process.stdout.close()
            process.stderr.close()


@posix_only
def test_command_backend_uses_the_reply_of_a_command_that_exits_leaving_a_worker(
        monkeypatch, tmp_path, pid_file):
    """At the real boundary: the command prints its reply, starts a worker
    that inherits its stdout and stderr, and exits 0 at once, the way a
    CLI that leaves a helper behind does. Its exit ends its answer: the
    reply is used and the call returns well inside the timeout (the old
    outcome was a TimeoutError at the whole timeout, advising a longer one
    that could not help, the reply discarded), the worker is gone, and
    the group was stopped by a pid that was still the command's own, its
    exited process unreaped (a signal 0 still reaches it)."""
    backend = _real_command_backend(monkeypatch, tmp_path, _LAUNCHER_SCRIPT, timeout=4,
                                    pid_file=str(pid_file), waits=False)
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

    started = time.monotonic()
    completion = backend.complete(image, "prompt", 10)

    assert time.monotonic() - started < 2, "the reply was held to the timeout"
    assert completion.text.strip() == ID_OK
    worker = int(pid_file.read_text(encoding="utf-8"))
    assert _gone(worker, within=10.0), f"worker {worker} outlived the command's exit"
    assert leader_unreaped == [True], "the tree was stopped by a pid the command no longer held"


def test_command_backend_sees_the_exit_without_reaping_the_command():
    """At the real boundary, the sight of the exit: a real child that has
    ended is seen as exited at once, and one still running is not, within
    the time given; on POSIX the child is not reaped by the look (its
    returncode is still None afterwards, so its pid is still its own and
    its group's until the backend reaps it last); on Windows the Popen
    handle keeps the pid reserved and the look may reap. A second look at
    the same ended child (an unreaped zombie by then, which kqueue on
    macOS refuses to register on: the first look may have seen the exit
    the same way, depending on whether it landed before or during that
    look) is the same at-once yes, and still no reap."""
    backend = CommandBackend(["fake-vlm", "{image}", "{prompt}"])
    ended = subprocess.Popen([sys.executable, "-c", "pass"], **backend.OWN_GROUP)
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               **backend.OWN_GROUP)
    try:
        assert backend._exited(ended, 10.0) is True
        started = time.monotonic()
        assert backend._exited(ended, 10.0) is True, "a second look at an exited command"
        assert time.monotonic() - started < 1, "the second look waited on a command already exited"
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
def test_command_backend_stops_a_runaway_command_before_the_timeout(
        monkeypatch, tmp_path, pid_file, stream):
    """At the real boundary: the command writes without end to one stream,
    under a timeout of a minute. It is stopped as soon as it passes the
    ceiling, seconds in, not read into memory until the timeout: the frame's
    error names the program, the stream and the ceiling, and the process is
    gone."""
    backend = _real_command_backend(monkeypatch, tmp_path, _RUNAWAY_SCRIPT, timeout=60,
                                    pid_file=str(pid_file), stream=stream)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")

    started = time.monotonic()
    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)

    assert time.monotonic() - started < 20, "the runaway was read until the timeout"
    for named in ("fake-vlm", stream, str(CommandBackend.MAX_OUTPUT_BYTES)):
        assert named in str(err.value), str(err.value)
    pid = int(pid_file.read_text(encoding="utf-8"))
    assert _gone(pid, within=10.0), f"the runaway {pid} is still running"


@posix_only
def test_command_is_refused_at_the_real_boundary_when_sigchld_is_ignored(
    monkeypatch, tmp_path, no_ambient_keys, no_ambient_ollama
):
    """At the real boundary: SIGCHLD really set to SIG_IGN in this process
    (restored afterwards), a real fake CLI on PATH found by the real
    shutil.which, and the factory refuses through BackendUnavailable naming
    SIGCHLD before anything is started, since with children reaped by the
    kernel the backend's every stop would signal a pid the command no
    longer holds."""
    command = _fake_cli(monkeypatch, tmp_path)
    before = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    try:
        with pytest.raises(providers.BackendUnavailable) as err:
            providers.build_primary_backend(
                _cfg(model={"backend": "command", "command": command, "timeout_seconds": 30}))
    finally:
        signal.signal(signal.SIGCHLD, before)
    message = str(err.value)
    assert "ignores SIGCHLD" in message, message
    assert "shell" in message and "default" in message, message
    assert "--backend" in message


def test_command_not_installed_fires_before_any_image_is_read(
    tmp_path, capsys, no_ambient_ollama, link_to_nowhere
):
    """Card #420, Done-when 2, at the real boundary: a command nothing on
    this machine is called, and the folder's one image is a link to
    nowhere, so opening it would fail loudly. The CLI exits 3 on the
    not-installed message, naming the command, and never mentions the file:
    the check ran before any image was read. `--no-local-config` keeps a
    developer's own melampus.local.toml keys out of the run (Done-when 3)."""
    from melampus.cli import main

    settings = _command_settings(tmp_path, ["melampus-no-such-command-420", "{image}", "{prompt}"])

    code = main([str(link_to_nowhere), "--config", str(settings), "--no-local-config",
                 "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "'melampus-no-such-command-420' is not installed or not on PATH" in err
    assert_no_image_was_touched(err, "command check")


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
when signed in and 1 when not, `--json` carrying `loggedIn` and
`authMethod` (the shapes in STATUS, each measured on 2.1.278 or read from
the program's own source: `claude.ai` is the subscription sign-in and
carries `subscriptionType`; `api_key`, `oauth_token` and `third_party` are
other credentials; `apiKeySource` names a key the login is set aside for,
and then `subscriptionType` is null and `--text` says "not in use"), and
takes the global flags before the subcommand, as the real one does, the
settings-deciding ones deciding what the check reports (Codex round 2,
C1, each measured on 2.1.278): an `apiKeyHelper` in a loaded settings
file, the user's `$CLAUDE_CONFIG_DIR/settings.json` unless `--restricted`
or `--bare` or a `--setting-sources` without `user` leaves it out, or the
file or JSON `--settings` names, is reported as authMethod
api_key_helper, apiKeySource apiKeyHelper; `--bare` never reads the
keychain, so the login is not seen (loggedIn false, exit 1). The image is
a file the prompt names, read by the Read tool; every file the prompt
names is read, as text rendered in a photograph could ask, and the first
denial is the reply. A read inside a working directory needs no rule
(permissions § Working directories: "By default, Claude has access to
files in the directory where you launched it", and `additionalDirectories`
"become readable without prompts"; Codex round 2, C2). Outside them a read
needs a prompt unless an `--allowedTools` rule pre-approves it
(permissions § Read and Edit: a bare `Read` matches everywhere;
`Read(//path)` is one absolute path, and an allow rule applies only when
both the path as given and the file it resolves to match), and with
nobody to answer it is denied. Allow rules and `additionalDirectories`
from every loaded settings file count (permissions § Settings precedence:
rules from every loaded settings file merge, only a deny wins);
`--restricted` loads no user, project or local settings file
(cli-reference: "loads only managed settings and --settings") and
confines the file tools to the working directories, rule or no rule.
MODE: "signed-in" answers;
"not-signed-in" fails the status check and every run the documented way;
"expired" passes the status check and fails the run, the way a session
that lapses mid-batch would; "hung" never answers the status check;
"no-auth-command" is an older CLI with no `auth` subcommand, a usage
error on stderr at exit 2; "silent-not-signed-in" is the documented
exit 1 alone, nothing printed; "api-key", "login-and-api-key",
"oauth-token", "third-party" and "console" pass the status check on
another credential than the subscription (Codex round 1, C1 and S2) and
answer runs, which would bill it."""
import json
import os
import re
import sys

MODE = {mode!r}
ROUTING = {routing!r}
IDENTIFICATION = {identification!r}
LOG = {log!r}
NOT_LOGGED_IN = "Not logged in \\u00b7 Please run /login"
STATUS = {{
    "signed-in": {{"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
                  "subscriptionType": "max"}},
    "api-key": {{"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty",
                "apiKeySource": "ANTHROPIC_API_KEY"}},
    "login-and-api-key": {{"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
                          "apiKeySource": "ANTHROPIC_API_KEY", "subscriptionType": None}},
    "oauth-token": {{"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"}},
    "third-party": {{"loggedIn": True, "authMethod": "third_party", "apiProvider": "bedrock"}},
    "console": {{"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
                "apiKeySource": "/login managed key", "subscriptionType": None}},
    "not-signed-in": {{"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"}},
}}

argv = sys.argv[1:]
with open(LOG, "a", encoding="utf-8") as log:
    log.write(json.dumps({{"argv": argv, "cwd": os.getcwd()}}) + "\\n")


def loaded_settings(restricted, bare, sources, named):
    """The settings the real CLI would load: the user's file when `user`
    is among the sources and neither --restricted nor --bare leaves it
    out, then the file or JSON string --settings names, which "still
    appl[ies]" under --restricted (`claude --help`)."""
    loaded = []
    if "user" in sources and not restricted and not bare:
        path = os.path.join(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"),
                            "settings.json")
        try:
            with open(path, encoding="utf-8") as f:
                loaded.append(json.load(f))
        except (OSError, ValueError):
            pass
    if named:
        try:
            if named.lstrip().startswith("{{"):
                loaded.append(json.loads(named))
            else:
                with open(named, encoding="utf-8") as f:
                    loaded.append(json.load(f))
        except (OSError, ValueError):
            sys.exit("Error: --settings could not be read: " + named)
    return loaded


def account(restricted, bare, sources, named):
    """What `auth status` reports under these flags: MODE's status, unless
    a loaded settings file names an apiKeyHelper (measured: reported over
    the login) or --bare leaves the keychain unread (measured: loggedIn
    false on the signed-in Mac; an environment key is still seen)."""
    if any("apiKeyHelper" in settings for settings in loaded_settings(restricted, bare, sources, named)):
        return {{"loggedIn": True, "authMethod": "api_key_helper", "apiProvider": "firstParty",
                "apiKeySource": "apiKeyHelper"}}
    status = STATUS.get(MODE, STATUS["signed-in"])
    if bare and status["authMethod"] in ("claude.ai", "oauth_token"):
        return STATUS["not-signed-in"]
    return status


# The global flags the real CLI takes before a subcommand, the
# settings-deciding ones read off, in either spelling (`--settings file`
# or `--settings=file`; measured on 2.1.278, both report the same status);
# a print-mode run parses argv whole below.
command = list(argv)
global_flags = {{"restricted": False, "bare": False, "sources": ["user", "project", "local"], "named": None}}
while command and command[0].startswith("--"):
    flag, attached, value = command.pop(0).partition("=")
    if flag == "--restricted":
        global_flags["restricted"] = True
    elif flag == "--bare":
        global_flags["bare"] = True
    elif flag == "--setting-sources":
        global_flags["sources"] = (value if attached else command.pop(0)).split(",")
    elif flag == "--settings":
        global_flags["named"] = value if attached else command.pop(0)
if command[:2] == ["auth", "status"]:
    if MODE == "hung":
        import time
        time.sleep(30)
    if MODE == "no-auth-command":
        print("error: unknown command 'auth'", file=sys.stderr)
        print("(Did you mean --help?)", file=sys.stderr)
        sys.exit(2)
    if MODE == "silent-not-signed-in":
        sys.exit(1)
    status = account(**global_flags)
    logged_in = status["loggedIn"]
    if "--text" in command:
        print("Login method: Claude Max account" if logged_in
              else "Not logged in. Run claude auth login to authenticate.")
    else:
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
ap.add_argument("--settings")
ap.add_argument("--restricted", action="store_true")
ap.add_argument("--bare", action="store_true")
ap.add_argument("prompt")
args = ap.parse_args()
if not args.print:
    sys.exit("an interactive session needs a terminal; use -p")
sources = args.setting_sources.split(",") if args.setting_sources else []
for source in sources:
    if source not in ("user", "project", "local"):
        sys.exit(f"Error processing --setting-sources: Invalid setting source: {{source}}. "
                 "Valid options are: user, project, local")
settings_loaded = loaded_settings(args.restricted, args.bare, sources, args.settings)


def result(text, is_error=False):
    if args.output_format == "json":
        print(json.dumps({{"type": "result", "subtype": "success", "is_error": is_error,
                          "result": text, "session_id": "00000000-0000-0000-0000-000000000000",
                          "num_turns": 1 if is_error else 2, "total_cost_usd": 0.0}}))
    else:
        print(text)


if MODE == "expired" or not account(args.restricted, args.bare, sources, args.settings)["loggedIn"]:
    result(NOT_LOGGED_IN, is_error=True)
    sys.exit(1)

# The Read tool, on every file the prompt names: inside a working directory
# (the cwd, and additionalDirectories from the loaded settings) a read
# needs no rule; outside, --restricted confines the tool, else a rule must
# pre-approve the file or it prompts, and with nobody to answer it is denied.
images = re.findall(r"(/\\S+\\.jpe?g)", args.prompt, re.IGNORECASE)
if not images:
    result("I could not find an image path in the prompt.")
    sys.exit(0)
if "Read" not in args.tools.split(","):
    result("I have no tool that can read files.")
    sys.exit(0)


def allows(rule, image):
    if rule == "Read":
        return True
    if rule.startswith("Read(//") and rule.endswith(")"):
        allowed = rule[len("Read(/"):-1]
        return image == allowed and os.path.realpath(image) == allowed
    return False


def inside(image, folder):
    return os.path.realpath(image).startswith(os.path.realpath(folder).rstrip("/") + "/")


rules = args.allowedTools.replace(",", " ").split()
folders = [os.getcwd()]
for settings in settings_loaded:
    permissions = settings.get("permissions", {{}})
    rules += permissions.get("allow", [])
    folders += permissions.get("additionalDirectories", [])
for image in images:
    if any(inside(image, folder) for folder in folders):
        pass
    elif args.restricted:
        result("Permission to read " + image + " was denied: outside the working directory.")
        sys.exit(0)
    elif not any(allows(rule, image) for rule in rules):
        result("Permission to read " + image + " was denied.")
        sys.exit(0)
    if not os.path.isfile(image):
        result("The file " + image + " does not exist.")
        sys.exit(0)
answer = ROUTING if "router" in args.prompt else IDENTIFICATION
result("```json\\n" + answer + "\\n```")
'''


def _fake_claude(monkeypatch, tmp_path, *, mode: str = "signed-in") -> Path:
    """Put a `claude` that imitates the real CLI's documented interface on
    PATH, ahead of any real Claude Code, with a settings folder of the
    test's own (CLAUDE_CONFIG_DIR, empty until a test writes a
    settings.json into it) so no test reads a developer's real one. Returns
    the log it appends each invocation's argv and cwd to."""
    log = tmp_path / "claude-calls.jsonl"
    _script_on_path(monkeypatch, tmp_path, CLAUDE, _FAKE_CLAUDE_SCRIPT.format(
        python=sys.executable, mode=mode, routing=ROUTING_OK, identification=ID_OK, log=str(log),
    ))
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    _real_detection(monkeypatch)
    return log


def _no_claude(monkeypatch, tmp_path) -> None:
    """A PATH on which nothing is called `claude`, keeping the interpreter's
    own folder so a shebang script elsewhere on it still runs."""
    empty = tmp_path / "empty-bin"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", f"{empty}{os.pathsep}{Path(sys.executable).parent}")
    _real_detection(monkeypatch)
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
    `--allowedTools Read(/{image})` pre-approves reading the one staged
    file and nothing else (permissions § Read and Edit: `//path` is
    "Absolute path from filesystem root", and the staged path begins with
    `/`), so the image in its temporary folder is read without a prompt
    while a photograph whose rendered text asks for ~/.ssh or .env gets
    that read denied, not answered (security review, round 1);
    `--permission-prompts none` denies anything else that would wait for a
    person; `--no-session-persistence` keeps a thousand frames from writing
    a thousand transcripts; `--strict-mcp-config` connects no MCP server;
    `--restricted` loads no user, project or local settings file (Codex
    round 1, S1: cli-reference, "loads only managed settings and
    --settings", and "confines the built-in file tools to the working
    directories", the staged image's own folder) while the keychain login
    is not a settings file and stays (measured: `claude --restricted auth
    status --json` reports the claude.ai login; `--bare` is the mode that
    skips keychain reads). The prompt is the last argument, the
    positional, and carries both placeholders: the image's path for the
    Read tool to read, then the pipeline's prompt in full. It is a valid
    `[model] command` by the config's own rule."""
    template = providers.CLAUDE_CODE_COMMAND
    assert template[0] == CLAUDE == providers.CLAUDE_CODE_PROGRAM
    flags = template[1:-1]
    assert flags == [
        "-p", "--output-format", "json", "--tools", "Read", "--allowedTools", "Read(/{image})",
        "--permission-prompts", "none", "--no-session-persistence", "--strict-mcp-config",
        "--restricted",
    ]
    assert "--setting-sources" not in template, "a settings file would be loaded"
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
    assert backend.name == shlex.join(providers.CLAUDE_CODE_COMMAND)

    own = [CLAUDE, "-p", "--model", "sonnet", "--output-format", "json", "{image} {prompt}"]
    backend = providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": own}))
    assert backend.command == own
    assert backend.executable == shutil.which(CLAUDE)


def _status_checks(log: Path) -> list[list[str]]:
    """The `auth status` invocations the fake `claude` logged, as received,
    global flags before the subcommand included."""
    return [argv for argv in (json.loads(line)["argv"] for line in log.read_text(encoding="utf-8").splitlines())
            if "auth" in argv and argv[argv.index("auth"):argv.index("auth") + 2] == ["auth", "status"]]


@posix_only
def test_claude_code_primary_asks_claude_code_once_refused_or_built(monkeypatch, tmp_path, no_ambient_ollama):
    """One probe, one sentence, as the ollama branch does it: the factory's
    claude-code verdict is the one out of the single detect_engines call
    whose verdicts also make the refusal's "what works" list, so Claude
    Code is asked its status once whether the run is refused or built (a
    hung install costs one probe timeout, not two), and the backend runs
    the executable that verdict resolved, not a second lookup. Detection is
    run on the template's program, so a user's own program in [model]
    command is the one asked, once, and the built-in `claude` is not."""
    built_in = providers.claude_code_status(providers.CLAUDE_CODE_COMMAND)
    log = _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")
    with pytest.raises(providers.BackendUnavailable, match="not signed in"):
        providers.build_primary_backend(_cfg(model={"backend": "claude-code"}))
    assert _status_checks(log) == [built_in]

    log.unlink()
    _fake_claude(monkeypatch, tmp_path)
    backend = providers.build_primary_backend(_cfg(model={"backend": "claude-code"}))
    assert _status_checks(log) == [built_in]
    assert backend.executable == shutil.which(CLAUDE)

    log.unlink()
    own = [shutil.which(CLAUDE), "-p", "--output-format", "json", "{image} {prompt}"]
    backend = providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": own}))
    assert _status_checks(log) == [providers.claude_code_status(own)], "the built-in's verdict was asked too"
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
        ScriptedBackend([ROUTING_OK, ID_OK], name=shlex.join(providers.CLAUDE_CODE_COMMAND)), config
    ).identify(photos / PHOTO)

    assert result.status == "ok", result.error
    assert result.model == shlex.join(providers.CLAUDE_CODE_COMMAND)
    assert result.identification == expected.identification
    assert result.taxon_routing == expected.taxon_routing
    assert [c.common_name for c in result.identification.ranked()] == [
        "Tricolored Heron", "Little Blue Heron"]
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    runs = [c for c in calls if c["argv"][:1] == ["-p"]]
    assert len(runs) == 2, calls
    for call in runs:
        argv = call["argv"]
        rule = argv[argv.index("--allowedTools") + 1]
        assert rule.startswith("Read(//") and rule.endswith(")"), rule
        staged = rule[len("Read(/"):-1]
        assert argv[:-1] == [a.replace("{image}", staged) for a in providers.CLAUDE_CODE_COMMAND[1:-1]]
        assert str(photos / PHOTO) not in argv[-1], "the original file's path reached the program"
        assert "melampus-" in staged and staged in argv[-1], "the rule and the prompt name different files"
        assert staged == os.path.realpath(staged), "the rule names a path through a symlink"
        assert call["cwd"] == os.path.dirname(staged), \
            "the program's working directory is not the staged image's folder"


@posix_only
def test_the_fake_claude_models_the_documented_working_directory_rule(monkeypatch, tmp_path):
    """Done-when 3, corrected (Codex round 2, C2): the fake imitates the
    documented permission check for the Read tool. Inside the working
    directory a read needs no rule (permissions § Working directories:
    "By default, Claude has access to files in the directory where you
    launched it"), and the run's working directory is the staged image's
    folder, so the staged file is read whatever the rule names. Outside
    it, a read is a prompt unless an allow rule covers the file (`Read`
    bare, everywhere; `Read(//path)`, that one file), and with
    `--permission-prompts none` a prompt is a denial; under `--restricted`
    the file tools are confined to the working directories (cli-reference)
    and the outside read is denied even with a rule naming it. The fake
    reads every file the prompt names, as text rendered in a photograph
    could ask it to, and the first denial is the reply."""
    _fake_claude(monkeypatch, tmp_path)
    image = tmp_path / "staged" / "image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg")
    elsewhere = tmp_path / "elsewhere.jpg"
    elsewhere.write_bytes(b"jpeg")
    unrestricted = [a for a in providers.CLAUDE_CODE_COMMAND if a != "--restricted"]
    assert unrestricted != providers.CLAUDE_CODE_COMMAND

    def reply(rule: str, prompt: str, template: list[str] = unrestricted) -> str:
        own = [a.replace("Read(/{image})", rule) for a in template]
        backend = providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": own}))
        return backend.complete(image, prompt, 10).text

    asks_for_both = f"router: also read {elsewhere}"
    assert ROUTING_OK in reply(f"Read(/{elsewhere})", "router"), "a read inside the working directory needs no rule"
    assert reply(f"Read(/{image})", asks_for_both) == f"Permission to read {elsewhere} was denied."
    assert ROUTING_OK in reply(f"Read(/{elsewhere})", asks_for_both)
    assert ROUTING_OK in reply("Read", asks_for_both)
    assert reply(f"Read(/{elsewhere})", asks_for_both, providers.CLAUDE_CODE_COMMAND) == (
        f"Permission to read {elsewhere} was denied: outside the working directory.")


@posix_only
def test_claude_code_runs_without_the_users_own_permission_grants(monkeypatch, tmp_path):
    """Codex round 1, S1, the regression as round 2, C2 corrected it: a
    user's own settings file can allow Read everywhere
    (`permissions.allow`) and open more folders (`additionalDirectories`),
    and rules from every loaded settings file merge with --allowedTools
    (permissions § Settings precedence: only a deny wins), so loaded, such
    a grant lets text rendered in a photograph reach files outside the
    staged folder. The attempt: the built-in template, its rule naming
    the staged image as the run builds it, and a prompt that also asks
    for a synthetic file outside the staged folder. With
    `--setting-sources user` in the flag's place that read is denied
    while no grant exists (nothing names the file) and goes through once
    the broad grant is in the user's settings: the fake models the grant.
    The template loads no settings file and confines reads to the staged
    folder: with the grant present, the outside read is denied."""
    _fake_claude(monkeypatch, tmp_path)
    image = tmp_path / "staged" / "image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg")
    elsewhere = tmp_path / "elsewhere.jpg"
    elsewhere.write_bytes(b"jpeg")
    asks_for_both = f"router: also read {elsewhere}"

    def reply(command: list[str]) -> str:
        backend = providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": command}))
        return backend.complete(image, asks_for_both, 10).text

    own = list(providers.CLAUDE_CODE_COMMAND)
    loaded = [a for a in own if a != "--restricted"]
    loaded[1:1] = ["--setting-sources", "user"]
    assert loaded != own, "the template has no --restricted to take out"
    assert reply(loaded) == f"Permission to read {elsewhere} was denied.", "no grant, yet read"

    settings = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "settings.json"
    settings.write_text(json.dumps({"permissions": {
        "allow": ["Read"], "additionalDirectories": [str(tmp_path)]}}), encoding="utf-8")
    assert ROUTING_OK in reply(loaded), "the fake does not model the user's grant"

    assert reply(own) == f"Permission to read {elsewhere} was denied: outside the working directory.", \
        "the user's grant was loaded"


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
    assert calls == [providers.claude_code_status(providers.CLAUDE_CODE_COMMAND)]


@posix_only
def test_detection_claude_code_not_signed_in_names_the_sign_in_command(monkeypatch, tmp_path):
    """Done-when 2: installed but not signed in, then unavailable with the
    reason "not signed in", the check as run (flags included, like every
    other verdict that ran one; review round 6, 1) and the command that
    signs in."""
    _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "not signed in" in verdict.reason and providers.CLAUDE_CODE_SIGN_IN in verdict.reason
    assert "`claude --restricted auth status --json`" in verdict.reason, verdict.reason


@posix_only
def test_detection_claude_code_not_signed_in_is_the_documented_exit_alone(monkeypatch, tmp_path):
    """Done-when 2: the documented check is "Exits with code 0 if logged in,
    1 if not" (cli-reference), so exit 1 with nothing on stderr is not
    signed in even when no status JSON came back to say `loggedIn`."""
    _fake_claude(monkeypatch, tmp_path, mode="silent-not-signed-in")
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "not signed in" in verdict.reason and providers.CLAUDE_CODE_SIGN_IN in verdict.reason


@posix_only
def test_detection_claude_code_reports_a_status_check_that_failed_some_other_way(monkeypatch, tmp_path):
    """A status check that fails for a reason other than not being signed in
    (an older `claude` with no `auth` subcommand: a usage error at exit 2)
    is not "not signed in", and `claude auth login` would not help. The
    verdict says what ran, the exit code and the CLI's own first words
    on stderr, the way a failed run's CommandFailed does."""
    _fake_claude(monkeypatch, tmp_path, mode="no-auth-command")
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "not signed in" not in verdict.reason
    assert providers.CLAUDE_CODE_SIGN_IN not in verdict.reason
    assert "claude --restricted auth status --json" in verdict.reason
    assert "exited 2" in verdict.reason
    assert "error: unknown command 'auth'" in verdict.reason


@posix_only
def test_detection_claude_code_that_cannot_be_run_says_so_in_the_systems_words(monkeypatch, tmp_path):
    """A `claude` that PATH finds but the system cannot start (its
    interpreter is missing: `#!/nonexistent`, the "not-runnable" case the
    command seam has) is neither not installed nor not signed in: the
    verdict is unavailable, says it could not be run with the system's own
    error, and does not point at the sign-in, which would not help."""
    _script_on_path(monkeypatch, tmp_path, CLAUDE, "#!/nonexistent\n")
    _real_detection(monkeypatch)
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "could not be run" in verdict.reason
    assert "No such file or directory" in verdict.reason
    assert providers.CLAUDE_CODE_SIGN_IN not in verdict.reason


def test_detection_claude_code_not_installed_points_to_the_install(monkeypatch, tmp_path):
    """Done-when 2: not installed, then unavailable with the reason "not
    installed" and where to get it, then the sign-in; for a template
    carrying `--bare`, which no sign-in can help (review round 6, 1), the
    next step after installing is to remove the flag, not to sign in
    (review round 7, 1), so one round trip tells the whole story."""
    _no_claude(monkeypatch, tmp_path)
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "not installed" in verdict.reason and providers.CLAUDE_CODE_INSTALL in verdict.reason
    assert f"then sign in with `{providers.CLAUDE_CODE_SIGN_IN}`" in verdict.reason

    template = providers.CLAUDE_CODE_COMMAND
    bare = [*template[:-1], providers.CLAUDE_CODE_BARE, template[-1]]
    with pytest.raises(providers.BackendUnavailable, match="not installed") as err:
        providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": bare}))
    reason = str(err.value)
    assert providers.CLAUDE_CODE_INSTALL in reason, reason
    assert f"then {providers.CLAUDE_CODE_BARE_FIX}" in reason, reason
    assert providers.CLAUDE_CODE_SIGN_IN not in reason, reason


@posix_only
@pytest.mark.parametrize(
    ("mode", "names"),
    [
        ("api-key", ("authMethod api_key", "apiKeySource ANTHROPIC_API_KEY", "unset ANTHROPIC_API_KEY")),
        ("login-and-api-key", ("authMethod claude.ai", "apiKeySource ANTHROPIC_API_KEY", "unset ANTHROPIC_API_KEY")),
        ("oauth-token", ("authMethod oauth_token", "unset CLAUDE_CODE_OAUTH_TOKEN")),
        ("third-party", ("authMethod third_party", "unset CLAUDE_CODE_USE_BEDROCK")),
        ("console", ("apiKeySource /login managed key", "claude auth logout")),
    ],
)
def test_detection_claude_code_signed_in_but_not_to_the_subscription_is_refused(
    monkeypatch, tmp_path, mode, names
):
    """Codex round 1, C1 and S2 (one defect): the engine promises every frame
    bills the subscription, and a print-mode run uses whatever credential
    Claude Code's precedence puts first, the environment melampus runs
    from included (authentication § Authentication precedence: "In
    non-interactive mode (-p), the key is always used when present"). So a
    status check that passes on another credential (an API key; the
    subscription login set aside for one, which the real CLI reports as
    authMethod claude.ai with apiKeySource named and subscriptionType
    null; an OAuth or bearer token from the environment; a cloud provider;
    the Console sign-in without a key) is not available: the verdict is a
    sixth shape, signed in but not to the subscription, naming what the
    status check said, what to remove, and the sign-in, and never "not
    signed in", which is another fix."""
    _fake_claude(monkeypatch, tmp_path, mode=mode)
    verdict = _verdict("claude-code")
    assert not verdict.available
    assert "not to a Claude subscription" in verdict.reason, verdict.reason
    assert "not signed in" not in verdict.reason
    for name in names:
        assert name in verdict.reason, verdict.reason
    assert f"`{providers.CLAUDE_CODE_SIGN_IN}`" in verdict.reason


@posix_only
def test_detection_claude_code_available_only_on_a_verified_subscription(monkeypatch, tmp_path):
    """The available verdict is positive evidence, not the absence of a
    refusal: `authMethod` claude.ai with no `apiKeySource`, the shape
    measured on a signed-in Mac, and the reason names the account kind."""
    _fake_claude(monkeypatch, tmp_path)
    verdict = _verdict("claude-code")
    assert verdict.available, verdict.reason
    assert "claude.ai, max" in verdict.reason and "subscription" in verdict.reason


@posix_only
def test_the_status_check_runs_under_the_templates_isolation(monkeypatch, tmp_path):
    """S2, "under the same effective configuration used for inference": the
    status check carries the template's --restricted before the
    subcommand (the real CLI processes the global flag there: measured,
    `claude --setting-sources bogus auth status --json` is refused as an
    invalid setting source), so the credential it reports is read under
    the settings the run loads, and the same inherited environment."""
    log = _fake_claude(monkeypatch, tmp_path)
    assert _verdict("claude-code").available
    assert _status_checks(log) == [["--restricted", "auth", "status", "--json"]]


@posix_only
def test_the_status_check_is_probed_under_the_templates_own_settings_flags(monkeypatch, tmp_path):
    """Codex round 2, C1: a `[model] command` of the user's own under
    claude-code can leave out `--restricted` or add `--settings` naming an
    `apiKeyHelper`, and a print-mode run then bills that helper's key
    (authentication § Authentication precedence: apiKeyHelper ranks above
    "Subscription OAuth credentials from /login"); a status check under
    fixed flags reports the subscription. So the check carries the
    template's own settings-deciding global flags (`--restricted`,
    `--bare`, `--settings`, `--setting-sources`, values included, in the
    template's order) before `auth status --json`, and the verdict is the
    run's configuration whatever the template. Measured on 2.1.278, no
    model call: `claude --restricted --settings '{"apiKeyHelper": ...}'
    auth status --json` reports authMethod api_key_helper, apiKeySource
    apiKeyHelper; a user settings file with apiKeyHelper reports the same
    with no flag and authMethod none under `--restricted`."""
    log = _fake_claude(monkeypatch, tmp_path)
    helper = tmp_path / "helper.json"
    helper.write_text(json.dumps({"apiKeyHelper": "/usr/bin/true"}), encoding="utf-8")
    template = providers.CLAUDE_CODE_COMMAND

    def build(command: list[str]):
        log.write_text("", encoding="utf-8")
        return providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": command}))

    def refused(command: list[str]) -> str:
        with pytest.raises(providers.BackendUnavailable) as err:
            build(command)
        reason = str(err.value)
        assert "not to a Claude subscription" in reason, reason
        assert "apiKeySource apiKeyHelper" in reason and "remove apiKeyHelper from the settings" in reason
        return reason

    # `--settings` added to the built-in template: the helper is loaded even
    # under --restricted ("managed settings and --settings still apply").
    own = [*template[:-1], "--settings", str(helper), template[-1]]
    reason = refused(own)
    assert f"claude --restricted --settings {helper} auth status --json" in reason
    assert _status_checks(log) == [["--restricted", "--settings", str(helper), "auth", "status", "--json"]]

    # The same helper in the CLI's other spelling, `--settings=file` (Codex
    # round 3, C1 and S1; measured on 2.1.278: `claude --restricted
    # --settings=helper.json auth status --json` reports authMethod
    # api_key_helper, apiKeySource apiKeyHelper, as the two-argument form
    # does): carried as given, and refused the same way.
    equals = [*template[:-1], f"--settings={helper}", template[-1]]
    reason = refused(equals)
    assert f"claude --restricted --settings={helper} auth status --json" in reason
    assert _status_checks(log) == [["--restricted", f"--settings={helper}", "auth", "status", "--json"]]

    # The user's settings file names the helper: the built-in template loads
    # no settings file and is available; a template without --restricted
    # loads it and is refused; one with `--setting-sources user` in the
    # flag's place, likewise; one carrying both keeps --restricted's
    # exclusion (measured on 2.1.278, in either order: authMethod none).
    (Path(os.environ["CLAUDE_CONFIG_DIR"]) / "settings.json").write_text(
        json.dumps({"apiKeyHelper": "/usr/bin/true"}), encoding="utf-8")
    assert build(list(template)).executable == shutil.which(CLAUDE)
    assert _status_checks(log) == [["--restricted", "auth", "status", "--json"]]

    without = [a for a in template if a != "--restricted"]
    refused(without)
    assert _status_checks(log) == [["auth", "status", "--json"]]

    user = [*without[:-1], "--setting-sources", "user", without[-1]]
    refused(user)
    assert _status_checks(log) == [["--setting-sources", "user", "auth", "status", "--json"]]

    user_equals = [*without[:-1], "--setting-sources=user", without[-1]]
    refused(user_equals)
    assert _status_checks(log) == [["--setting-sources=user", "auth", "status", "--json"]]

    both = [*template[:-1], "--setting-sources", "user", template[-1]]
    assert build(both).executable == shutil.which(CLAUDE)
    assert _status_checks(log) == [["--restricted", "--setting-sources", "user", "auth", "status", "--json"]]


@posix_only
def test_a_bare_template_is_never_signed_in_to_the_subscription(monkeypatch, tmp_path):
    """C1, the other way a template can leave the subscription: `--bare`
    (`claude --help`: "OAuth and keychain are never read"; headless: "bare
    mode doesn't use your subscription login"). Probed under it, the check
    reports not signed in (measured on 2.1.278: `claude --bare auth status
    --json` on the signed-in Mac says loggedIn false, authMethod none,
    exit 1), and the refusal quotes the check it ran, `--bare` included,
    and says to remove `--bare` from `[model] command`, never to sign in:
    signing in cannot help a template that never reads the login (review
    round 6, 1). The same under a key in the environment, which bare mode
    does read (headless: "set ANTHROPIC_API_KEY ... because bare mode
    doesn't use your subscription login"): the key to unset, and `--bare`
    to remove."""
    log = _fake_claude(monkeypatch, tmp_path)
    template = providers.CLAUDE_CODE_COMMAND
    bare = [*template[:-1], "--bare", template[-1]]
    with pytest.raises(providers.BackendUnavailable, match="not signed in") as err:
        providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": bare}))
    assert _status_checks(log) == [["--restricted", "--bare", "auth", "status", "--json"]]
    reason = str(err.value)
    assert "`claude --restricted --bare auth status --json`" in reason, reason
    assert "remove `--bare` from `[model] command`" in reason, reason
    assert providers.CLAUDE_CODE_SIGN_IN not in reason, reason

    _fake_claude(monkeypatch, tmp_path, mode="api-key")
    with pytest.raises(providers.BackendUnavailable, match="not to a Claude subscription") as err:
        providers.build_primary_backend(_cfg(model={"backend": "claude-code", "command": bare}))
    reason = str(err.value)
    assert "`claude --restricted --bare auth status --json`" in reason, reason
    assert "unset ANTHROPIC_API_KEY" in reason and "remove `--bare` from `[model] command`" in reason, reason
    assert providers.CLAUDE_CODE_SIGN_IN not in reason, reason


def test_the_status_check_argv_is_derived_from_the_template():
    """C1, the one source: the status check's global flags are read off the
    template that will run, so the built-in's check carries its
    --restricted and nothing else, a template with none of the flags is
    checked with none, and the flags keep their values and order, in
    either spelling the CLI takes: `--flag value` or `--flag=value`
    (Codex round 3, C1 and S1; measured on 2.1.278, `claude --restricted
    --settings=helper.json auth status --json` reports apiKeySource
    apiKeyHelper exactly as the two-argument form does, and
    `--setting-sources=bogus` is refused as an invalid setting source, so
    the `=` spelling is carried as given, one argument)."""
    assert providers.claude_code_status(providers.CLAUDE_CODE_COMMAND) == [
        "--restricted", "auth", "status", "--json"]
    assert providers.claude_code_status([CLAUDE, "-p", "--output-format", "json", "{image} {prompt}"]) == [
        "auth", "status", "--json"]
    own = [CLAUDE, "-p", "--settings", '{"apiKeyHelper": "x"}', "--add-dir", "/tmp", "--setting-sources",
           "user,project", "--bare", "--restricted", "{image} {prompt}"]
    assert providers.claude_code_status(own) == [
        "--settings", '{"apiKeyHelper": "x"}', "--setting-sources", "user,project", "--bare", "--restricted",
        "auth", "status", "--json"]
    assert providers.claude_code_status([CLAUDE, "-p", "--settings", "{image} {prompt}"]) == [
        "--settings", "{image} {prompt}", "auth", "status", "--json"]
    equals = [CLAUDE, "-p", '--settings={"apiKeyHelper": "x"}', "--add-dir=/tmp",
              "--setting-sources=user,project", "--restricted", "{image} {prompt}"]
    assert providers.claude_code_status(equals) == [
        '--settings={"apiKeyHelper": "x"}', "--setting-sources=user,project", "--restricted",
        "auth", "status", "--json"]
    assert providers.claude_code_status([CLAUDE, "-p", "--settings=a=b", "--bare", "{image} {prompt}"]) == [
        "--settings=a=b", "--bare", "auth", "status", "--json"]


@posix_only
def test_cli_detect_engines_probes_the_users_own_template_under_its_flags(
    monkeypatch, tmp_path, capsys, no_ambient_keys
):
    """C1 at the dialog (Done-when 2, "the same verdict"): --detect-engines
    probes the configured template under that template's flags, so a
    `[model] command` whose `--settings` names an apiKeyHelper is shown
    as signed in but not to the subscription, the verdict its run gets."""
    from melampus.cli import main

    log = _fake_claude(monkeypatch, tmp_path)
    helper = tmp_path / "helper.json"
    helper.write_text(json.dumps({"apiKeyHelper": "/usr/bin/true"}), encoding="utf-8")
    template = providers.CLAUDE_CODE_COMMAND
    own = [*template[:-1], "--settings", str(helper), template[-1]]
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'[model]\nbackend = "claude-code"\ncommand = {json.dumps(own)}\n', encoding="utf-8")

    assert main(["--detect-engines", "--config", str(settings)]) == 0

    verdicts = json.loads(capsys.readouterr().out)
    assert verdicts[-1]["engine"] == "claude-code"
    assert verdicts[-1]["available"] is False, verdicts[-1]["reason"]
    assert "not to a Claude subscription" in verdicts[-1]["reason"]
    assert "remove apiKeyHelper from the settings" in verdicts[-1]["reason"]
    assert _status_checks(log) == [["--restricted", "--settings", str(helper), "auth", "status", "--json"]]


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
    assert [v["engine"] for v in verdicts] == [*ENGINES, "claude-code"]
    assert verdicts[-1]["available"] is False
    assert providers.CLAUDE_CODE_SIGN_IN in verdicts[-1]["reason"]


@posix_only
def test_cli_detect_engines_probes_the_program_the_claude_code_run_would(
    monkeypatch, tmp_path, capsys, no_ambient_keys
):
    """Done-when 2, "the same verdict": as --detect-engines probes the
    configured Ollama address, it probes the configured Claude Code
    program, read the way the run reads it, so the dialog cannot disagree
    with what --backend claude-code would run. A signed-in `claude` at a
    full path on no PATH, named by [model] command under claude-code, is
    reported available (the built-in `claude` alone would say not
    installed), and it is the one asked, once."""
    from melampus.cli import main

    log = _fake_claude(monkeypatch, tmp_path)
    script = shutil.which(CLAUDE)
    _no_claude(monkeypatch, tmp_path)
    own = [script, "-p", "--output-format", "json", "{image} {prompt}"]
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'[model]\nbackend = "claude-code"\ncommand = {json.dumps(own)}\n', encoding="utf-8")

    assert main(["--detect-engines", "--config", str(settings)]) == 0

    verdicts = json.loads(capsys.readouterr().out)
    assert verdicts[-1]["engine"] == "claude-code"
    assert verdicts[-1]["available"] is True, verdicts[-1]["reason"]
    assert _status_checks(log) == [providers.claude_code_status(own)]


@posix_only
def test_claude_code_not_signed_in_is_refused_before_any_image_is_read(
    monkeypatch, tmp_path, capsys, link_to_nowhere
):
    """Card #421, Done-when 2 at analysis time, where detection can tell:
    Claude Code installed but not signed in, and the folder's one image is
    a link to nowhere, so opening it would fail loudly. The CLI exits 3 on
    a refusal that says to run the sign-in command and never mentions the
    file: the status check ran before any image was read."""
    from melampus.cli import main

    _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")

    code = main([str(link_to_nowhere), "--backend", "claude-code", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not signed in" in err and providers.CLAUDE_CODE_SIGN_IN in err
    assert_no_image_was_touched(err, "sign-in check")


@posix_only
def test_claude_code_on_an_api_key_is_refused_before_any_image_is_read(
    monkeypatch, tmp_path, capsys, link_to_nowhere
):
    """C1 and S2 at analysis time: the subscription login set aside for an
    API key in the environment (the shape a `claude` cloud engine's key
    leaves behind), and the folder's one image a link to nowhere. Exit 3
    on a refusal naming the key to unset and the sign-in, before any image
    is read, and never mentioning the file."""
    from melampus.cli import main

    _fake_claude(monkeypatch, tmp_path, mode="login-and-api-key")

    code = main([str(link_to_nowhere), "--backend", "claude-code", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not to a Claude subscription" in err and "unset ANTHROPIC_API_KEY" in err
    assert providers.CLAUDE_CODE_SIGN_IN in err
    assert_no_image_was_touched(err, "sign-in check")


@posix_only
def test_claude_code_on_a_helper_named_by_settings_is_refused_before_any_image_is_read(
    monkeypatch, tmp_path, capsys, link_to_nowhere
):
    """Codex round 3, C1 and S1, at analysis time: a `[model] command` of
    the user's own carrying `--settings=helper.json` in the CLI's `=`
    spelling, the file naming an `apiKeyHelper`, on a signed-in Mac, and
    the folder's one image a link to nowhere. The status check carries the
    flag as spelled (measured on 2.1.278, no model call: `claude
    --restricted --settings=helper.json auth status --json` reports
    authMethod api_key_helper, apiKeySource apiKeyHelper), so the run is
    refused as the two-argument spelling is: exit 3 naming the helper to
    remove, before any image is read, never mentioning the file."""
    from melampus.cli import main

    log = _fake_claude(monkeypatch, tmp_path)
    helper = tmp_path / "helper.json"
    helper.write_text(json.dumps({"apiKeyHelper": "/usr/bin/true"}), encoding="utf-8")
    template = providers.CLAUDE_CODE_COMMAND
    own = [*template[:-1], f"--settings={helper}", template[-1]]
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'[model]\nbackend = "claude-code"\ncommand = {json.dumps(own)}\n', encoding="utf-8")

    code = main([str(link_to_nowhere), "--config", str(settings), "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not to a Claude subscription" in err and "remove apiKeyHelper from the settings" in err, err
    assert _status_checks(log) == [["--restricted", f"--settings={helper}", "auth", "status", "--json"]]
    assert_no_image_was_touched(err, "sign-in check")


@posix_only
def test_cli_detect_engines_prints_the_not_subscription_verdict(monkeypatch, tmp_path, capsys, no_ambient_keys):
    """Done-when 2 gains the sixth shape: --detect-engines shows it."""
    from melampus.cli import main

    _fake_claude(monkeypatch, tmp_path, mode="api-key")
    assert main(["--detect-engines"]) == 0
    verdicts = json.loads(capsys.readouterr().out)
    assert verdicts[-1]["engine"] == "claude-code"
    assert verdicts[-1]["available"] is False
    assert "not to a Claude subscription" in verdicts[-1]["reason"]
    assert "unset ANTHROPIC_API_KEY" in verdicts[-1]["reason"]


def test_claude_code_not_installed_is_refused_before_any_image_is_read(
    monkeypatch, tmp_path, capsys, link_to_nowhere
):
    """Done-when 2 at analysis time, not installed: exit 3 naming `claude`,
    where to install it and how to sign in, before any image is read."""
    from melampus.cli import main

    _no_claude(monkeypatch, tmp_path)

    code = main([str(link_to_nowhere), "--backend", "claude-code", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "not installed" in err and providers.CLAUDE_CODE_INSTALL in err
    assert providers.CLAUDE_CODE_SIGN_IN in err
    assert_no_image_was_touched(err, "install check")


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
    assert f"loading {shlex.join(providers.CLAUDE_CODE_COMMAND)}" in err, err
    assert "cloud default" not in err and "estimate" not in err.lower()
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert result["file"] == PHOTO
    assert result["status"] == "ok"
    assert result["model"] == shlex.join(providers.CLAUDE_CODE_COMMAND)
    assert [c["common_name"] for c in result["identification"]["candidates"]] == [
        "Tricolored Heron", "Little Blue Heron"]
