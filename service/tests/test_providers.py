"""The backend seam as a setting — the Windows / cloud-primary path.

The contract under test: `[model] backend` selects who answers, defaults keep the
Mac local-first path byte-for-byte unchanged, cloud selection fails fast and
clearly (missing SDK, missing key, unknown name), and the MLX-shaped defaults are
retuned for a cloud primary without ever overriding an explicit setting.
"""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import threading
import time
import types
import urllib.request
from http.server import BaseHTTPRequestHandler

import pytest
from conftest import PHOTO, fake_platform, loopback_server

from melampus import providers
from melampus.backend import AnthropicBackend, MLXBackend, OpenAIBackend, ScriptedBackend
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


def test_ollama_is_refused_as_not_built_yet_and_names_what_works(no_ambient_keys, no_ambient_ollama):
    """Card #403: `ollama` is one of the four names and the CLI must accept it,
    but its backend is card #406's. Until then it is refused the way mlx is
    refused off Apple Silicon: what is wrong, in plain words, and the backends
    that do work on this machine. Ollama is local, so nothing about a cloud
    primary applies to it: no retuned defaults, no cost gate, no cloud cache."""
    config = _cfg(model={"backend": "ollama"})
    assert not providers.is_cloud_primary(config)
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(config)
    message = str(err.value)
    assert "The Ollama engine is not built yet" in message
    assert "#406" in message
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in message, f"{works_here!r} is not named as working here:\n{message}"
    assert "--backend" in message


def test_cli_backend_ollama_exits_3_with_the_refusal(photos, tmp_path, capsys, no_ambient_keys, no_ambient_ollama):
    from melampus.cli import main

    code = main([str(photos), "--backend", "ollama", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "The Ollama engine is not built yet" in err
    assert "cloud default" not in err, f"ollama is local; nothing was retuned for a cloud:\n{err}"


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
    names the engines that work here — every backend but mlx and the one that
    is not built yet (card #403)."""
    from melampus.cli import main

    fake_platform(monkeypatch, "win32", "AMD64")

    code = main([str(photos), "--backend", "mlx", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code != 0
    assert "Apple Silicon" in err
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in err, f"{works_here!r} is not named as working here:\n{err}"
    assert "ollama" not in err, f"an engine that is not built yet is named as working:\n{err}"


# ---------------------------------------------------------------------------
# Card #404: which engines can run on this machine.


@pytest.fixture()
def no_ambient_ollama(monkeypatch):
    """A developer's running Ollama must not decide what these tests assert."""
    monkeypatch.setattr(providers, "ollama_answers", lambda: False)


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
    monkeypatch.setattr(providers, "ollama_answers", lambda: True)
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
    monkeypatch.setattr(providers, "ollama_answers", lambda: True)
    assert providers._works_here() == ("ollama", "openai", "claude", "scripted")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    assert "ollama, openai, claude, scripted" in str(err.value)


def _closed_port() -> int:
    """A loopback port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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
    monkeypatch.setattr(providers, "detect_engines", lambda: nothing_available)
    with pytest.raises(StopIteration):
        providers.default_engine()


@contextlib.contextmanager
def _fake_ollama(monkeypatch, *, status: int = 200, delay: float = 0.0):
    """A server speaking Ollama's version endpoint on 127.0.0.1 at an
    ephemeral port, with detection pointed at it. `status` is what
    GET /api/version answers; `delay` holds the answer that long."""
    release = threading.Event()

    class Version(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            assert self.path == "/api/version", self.path
            if delay:
                release.wait(delay)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"version": "0.0.0-fake"}')

        def log_message(self, *_):
            return None

    with loopback_server(Version) as server:
        monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
        try:
            yield server
        finally:
            release.set()


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
    closed_port = _closed_port()
    monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{closed_port}")
    assert providers.ollama_answers() is False
    verdict = _verdict("ollama")
    assert not verdict.available
    assert f"127.0.0.1:{closed_port}" in verdict.reason


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


def test_ollama_probe_gives_up_at_its_deadline_when_the_headers_trickle(monkeypatch):
    """Security: OLLAMA_PROBE_SECONDS is a deadline on the whole probe, not on
    each read. A socket timeout is per operation, so whatever listens on the
    port when Ollama does not could send the status line and then one header
    byte every hundred milliseconds, each within the timeout, and hold
    detection, and the CLI's startup behind it, for as long as it liked. Given
    a server that trickles a valid 200 over two seconds, the probe reports
    unavailable and returns within its deadline."""
    monkeypatch.setattr(providers, "OLLAMA_PROBE_SECONDS", 0.3)

    class Trickling(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            # Once the probe hangs up, the next write raises; that ends the trickle.
            with contextlib.suppress(OSError):
                self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                for byte in b"Content-Length: 2\r\n\r\n":
                    time.sleep(0.1)
                    self.wfile.write(bytes([byte]))
                self.wfile.write(b"{}")

        def log_message(self, *_):
            return None

    with loopback_server(Trickling) as ollama:
        monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{ollama.server_port}")
        started = time.monotonic()
        answered = providers.ollama_answers()
        elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"the probe read past its deadline: {elapsed:.2f}s"
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

    class AnythingGoes(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_):
            return None

    monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{_closed_port()}")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    # urlopen builds its default opener once, reading the proxy variables then;
    # start it fresh so the environment set here is the one it would see.
    monkeypatch.setattr(urllib.request, "_opener", None)
    with loopback_server(AnythingGoes) as proxy:
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

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_):
            return None

    with loopback_server(Destination) as destination:
        elsewhere = f"http://127.0.0.1:{destination.server_port}/api/version"

        class Redirecting(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                self.send_response(302)
                self.send_header("Location", elsewhere)
                self.end_headers()

            def log_message(self, *_):
                return None

        with loopback_server(Redirecting) as ollama:
            monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{ollama.server_port}")
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
    from melampus.cli import main

    with _fake_ollama(monkeypatch):
        assert main(["--detect-engines"]) == 0
    verdicts = {v["engine"]: v for v in json.loads(capsys.readouterr().out)}
    assert verdicts["ollama"]["available"] is True


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
    monkeypatch.setattr(providers, "ollama_answers", lambda: True)

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
    monkeypatch.setattr(providers, "ollama_answers", lambda: True)
    argv = [str(photos), "--cache", str(tmp_path / "cache.jsonl")]
    assert _chosen_engine(monkeypatch, [*argv, "--backend", "mlx"]) == "mlx"
    config = tmp_path / "settings.toml"
    config.write_text('[model]\nbackend = "claude"\n', encoding="utf-8")
    assert _chosen_engine(monkeypatch, [*argv, "--config", str(config)]) == "claude"
