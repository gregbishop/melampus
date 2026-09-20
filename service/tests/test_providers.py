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
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import pytest
from conftest import (
    PHOTO,
    QuietHandler,
    closed_port,
    fake_platform,
    loopback_server,
    recording_handler,
)
from test_pipeline import ID_OK, ROUTING_OK

from melampus import providers
from melampus.backend import (
    AnthropicBackend,
    MLXBackend,
    OllamaBackend,
    OpenAIBackend,
    ScriptedBackend,
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
    default." One constant, so card #406 can turn it into a config value."""
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
def _ollama_served_by(monkeypatch, handler: type[QuietHandler]):
    """`handler` on 127.0.0.1 at an ephemeral port, standing in for Ollama:
    detection is pointed at it for the block."""
    with loopback_server(handler) as server:
        monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
        yield server


# Slack for thread scheduling in the deadline tests: the timer thread fires and
# the main thread's read returns some tens of milliseconds after the deadline,
# while waiting out the trickle takes over two seconds.
SCHEDULING_SLACK = 0.5


def _timed_probe(monkeypatch, handler: type[QuietHandler]) -> tuple[bool, float]:
    """The probe against `handler` standing in for Ollama: what it answered
    and how many seconds it took."""
    with _ollama_served_by(monkeypatch, handler):
        started = time.monotonic()
        answered = providers.ollama_answers()
        return answered, time.monotonic() - started


@contextlib.contextmanager
def _fake_ollama(monkeypatch, *, status: int = 200, delay: float = 0.0, replies: list[str] = ()):
    """A server speaking Ollama's version and chat endpoints, standing in for
    Ollama. `status` is what GET /api/version answers; `delay` holds the
    answer that long. `replies` are the texts POST /api/chat answers with, in
    order, each wrapped in the final response object docs/api.md § Generate a
    chat completion shows; every chat request's JSON body is kept on
    `server.chats`."""
    release = threading.Event()
    pending = list(replies)

    class Ollama(QuietHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            assert self.path == "/api/version", self.path
            if delay:
                release.wait(delay)
            self._answer(status, {"version": "0.0.0-fake"})

        def do_POST(self):  # noqa: N802 - http.server's name
            assert self.path == "/api/chat", self.path
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.chats.append(body)
            if not pending:
                self._answer(404, {"error": f"model '{body.get('model')}' not found"})
                return
            self._answer(200, {
                "model": body["model"], "created_at": "2026-09-18T00:00:00Z",
                "message": {"role": "assistant", "content": pending.pop(0)},
                "done_reason": "stop", "done": True, "total_duration": 1668506709,
                "prompt_eval_count": 26, "eval_count": 83,
            })

        def _answer(self, code: int, payload: dict) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

    with _ollama_served_by(monkeypatch, Ollama) as server:
        server.chats = []
        try:
            yield server
        finally:
            release.set()


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
    config: detection's default is pointed at a closed port to prove it."""
    from melampus.cli import main

    out = tmp_path / "results.json"
    settings = tmp_path / "settings.toml"
    with _fake_ollama(monkeypatch, replies=[ROUTING_OK, ID_OK]):
        settings.write_text(
            f'[model]\nollama_url = "{providers.OLLAMA_URL}"\n', encoding="utf-8")
        monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{closed_port()}")
        code = main([
            str(photos), "--backend", "ollama", "--config", str(settings),
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
    engine and `--detect-engines` probe too, not only the backend."""
    from melampus.cli import main

    settings = tmp_path / "settings.toml"
    with _fake_ollama(monkeypatch) as server:
        settings.write_text(
            f'[model]\nollama_url = "{providers.OLLAMA_URL}"\n', encoding="utf-8")
        monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{closed_port()}")
        assert main(["--detect-engines", "--config", str(settings)]) == 0
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
    """A listener that sends a valid 200 one header byte every hundred
    milliseconds, over two seconds: each byte within the socket timeout, the
    whole well past the probe's deadline."""

    def do_GET(self):  # noqa: N802 - http.server's name
        # Once the probe hangs up, the next write raises; that ends the trickle.
        with contextlib.suppress(OSError):
            self.wfile.write(b"HTTP/1.1 200 OK\r\n")
            for byte in b"Content-Length: 2\r\n\r\n":
                time.sleep(0.1)
                self.wfile.write(bytes([byte]))
            self.wfile.write(b"{}")


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


def test_ollama_probe_gives_up_at_its_deadline_when_it_fires_during_connect(monkeypatch):
    """Security: the deadline must bound the probe whichever side of the
    handshake it lands on. The timer hangs up the connection's socket, and
    there is none until connect() returns; a deadline that fires while the
    probe is still connecting hangs up nothing, and if the handshake then
    completes, the reads that follow are bounded per byte only, and a
    trickling listener holds detection for as long as it likes again. The
    kernel's handshake timing is not reproducible, so connect() is held past
    the deadline here. Given a connect that completes just after the deadline
    against a trickling server, the probe reports unavailable and returns
    within the deadline plus the hold, not after the trickle."""
    deadline = 0.3
    hold = 0.05  # how long past the deadline connect() is held
    monkeypatch.setattr(providers, "OLLAMA_PROBE_SECONDS", deadline)
    connect = http.client.HTTPConnection.connect

    def held(connection):
        time.sleep(deadline + hold)
        connect(connection)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", held)
    answered, elapsed = _timed_probe(monkeypatch, Trickling)
    assert elapsed < deadline + hold + SCHEDULING_SLACK, f"the probe read past its deadline: {elapsed:.2f}s"
    assert answered is False


def test_ollama_probe_timeout_is_one_second():
    assert providers.OLLAMA_PROBE_SECONDS == 1.0


def test_hang_up_survives_the_connection_closing_between_its_reads():
    """Security: `_hang_up` runs on the timer thread while the main thread may
    be closing the connection (`ollama_answers`'s `finally`, or `getresponse`
    itself when the response says close), and `HTTPConnection.close()` sets
    `sock` to None. Given a connection whose socket is there on the first
    read and gone on the second, `_hang_up` returns without raising, as its
    docstring says, and the deadline is still recorded."""
    reads = 0

    class Closing:
        def __init__(self, sock):
            self._sock = sock

        @property
        def sock(self):
            nonlocal reads
            reads += 1
            return self._sock if reads == 1 else None

    expired = threading.Event()
    with socket.socket() as sock:
        providers._hang_up(Closing(sock), expired)
    assert expired.is_set()


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
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    # urlopen builds its default opener once, reading the proxy variables then;
    # start it fresh so the environment set here is the one it would see.
    monkeypatch.setattr(urllib.request, "_opener", None)
    with loopback_server(recording_handler(seen)) as proxy:
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_port}")
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
        elsewhere = f"http://127.0.0.1:{destination.server_port}/api/version"

        class Redirecting(QuietHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                self.send_response(302)
                self.send_header("Location", elsewhere)
                self.end_headers()

        with _ollama_served_by(monkeypatch, Redirecting):
            answered = providers.ollama_answers()
    assert seen == [], f"the probe followed the redirect off loopback: {seen}"
    assert answered is False


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

    def __init__(self, reply: bytes = b"{}", error: Exception | None = None, status: int = 200):
        self.reply, self.error, self.status = reply, error, status
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.reply)


OLLAMA_REPLY = json.dumps({
    "model": "qwen3-vl:8b-instruct",
    "created_at": "2026-09-18T00:00:00Z",
    "message": {"role": "assistant", "content": ID_OK},
    "done_reason": "stop",
    "done": True,
    "total_duration": 1668506709,
    "prompt_eval_count": 26,
    "eval_count": 83,
}).encode("utf-8")


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


def test_ollama_backend_reports_a_malformed_reply(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    backend = _ollama_backend(_FakeUrlopen(b"<html>proxy error</html>"))
    with pytest.raises(RuntimeError) as err:
        backend.complete(image, "prompt", 10)
    assert "not JSON" in str(err.value)


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
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    # urlopen builds its default opener once, reading the proxy variables then;
    # start it fresh so the environment set here is the one it would see.
    monkeypatch.setattr(urllib.request, "_opener", None)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    with loopback_server(recording_handler(seen)) as proxy:
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_port}")
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
        elsewhere = f"http://127.0.0.1:{destination.server_port}/api/chat"

        class Redirecting(QuietHandler):
            def do_POST(self):  # noqa: N802 - http.server's name
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(302)
                self.send_header("Location", elsewhere)
                self.end_headers()

        with loopback_server(Redirecting) as squatter:
            backend = OllamaBackend(
                "qwen3-vl:8b-instruct", f"http://127.0.0.1:{squatter.server_port}", timeout=5.0)
            with pytest.raises(RuntimeError) as err:
                backend.complete(image, "prompt", 10)
    assert "Ollama answered 302" in str(err.value), str(err.value)
    assert seen == [], f"the backend followed the redirect off the address: {seen}"


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
