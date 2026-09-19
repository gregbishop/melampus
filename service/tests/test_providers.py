"""The backend seam as a setting — the Windows / cloud-primary path.

The contract under test: `[model] backend` selects who answers, defaults keep the
Mac local-first path byte-for-byte unchanged, cloud selection fails fast and
clearly (missing SDK, missing key, unknown name), and the MLX-shaped defaults are
retuned for a cloud primary without ever overriding an explicit setting.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import types
import urllib.error
from pathlib import Path
import urllib.request

import pytest
from conftest import PHOTO, REAL_CLAUDE_CODE_VERDICT, FakeOllama
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
    monkeypatch.setattr(providers.sys, "platform", "win32")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    message = str(err.value)
    assert "claude" in message and "openai" in message and "--backend" in message


def test_mlx_is_refused_on_intel_mac(monkeypatch, no_ambient_ollama):
    """darwin alone is not enough — the pyproject marker also requires arm64,
    so an Intel Mac must get the helpful refusal, not a ModuleNotFoundError
    at warmup."""
    monkeypatch.setattr(providers.sys, "platform", "darwin")
    monkeypatch.setattr(providers.platform, "machine", lambda: "x86_64")
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
    assert "No Ollama server is answering at" in message
    assert providers.OLLAMA_URL in message
    assert providers.OLLAMA_INSTALL in message
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in message, f"{works_here!r} is not named as working here:\n{message}"
    assert "--backend" in message


def test_ollama_not_running_refusal_names_the_configured_address(monkeypatch):
    """The address tried is the configured one, so the message and the probe
    cannot disagree about where Ollama was looked for."""
    probed: list[str] = []

    def answers(url=None):
        probed.append(url)
        return False

    monkeypatch.setattr(providers, "ollama_answers", answers)
    config = _cfg(model={"backend": "ollama", "ollama_url": "http://127.0.0.1:11435"})
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(config)
    assert "http://127.0.0.1:11435" in str(err.value)
    # Probed for the backend, then again by detection for the "what works"
    # list: every probe went to the configured address.
    assert probed and set(probed) == {"http://127.0.0.1:11435"}


def test_cli_backend_ollama_exits_3_with_the_not_running_message(
    photos, tmp_path, capsys, no_ambient_keys, no_ambient_ollama
):
    from melampus.cli import main

    code = main([str(photos), "--backend", "ollama", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert "No Ollama server is answering at" in err
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


def test_mlx_refusal_does_not_name_ollama_as_working(monkeypatch, no_ambient_ollama):
    """Off Apple Silicon the mlx refusal lists what works here; with no Ollama
    server answering (card #404's detection decides), ollama stays off that
    list."""
    monkeypatch.setattr(providers.sys, "platform", "win32")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    assert "ollama" not in str(err.value)


def test_engine_round_trips_through_the_local_config(monkeypatch, tmp_path):
    """Card #403: the engine is a setting before it is a dialog. `[model]
    backend = "claude"` in melampus.local.toml is what load_config reads back,
    the same way the plugin's executable reads its config (card #436)."""
    local = tmp_path / "melampus.local.toml"
    local.write_text('[model]\nbackend = "claude"\n', encoding="utf-8")
    monkeypatch.setattr(providers.MelampusConfig.__module__ and __import__("melampus.config").config,
                        "_local_config", lambda: local)

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

    monkeypatch.setattr(providers.sys, "platform", "win32")
    monkeypatch.setattr(providers.platform, "machine", lambda: "AMD64")

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


def _fake_platform(monkeypatch, platform_name: str, machine: str) -> None:
    monkeypatch.setattr(providers.sys, "platform", platform_name)
    monkeypatch.setattr(providers.platform, "machine", lambda: machine)


def _verdict(engine: str) -> providers.EngineVerdict:
    (verdict,) = [v for v in providers.detect_engines() if v.engine == engine]
    return verdict


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
    _fake_platform(monkeypatch, "darwin", "arm64")
    assert _verdict("mlx").available


@pytest.mark.parametrize(("platform_name", "machine"), [("win32", "AMD64"), ("darwin", "x86_64"), ("linux", "x86_64")])
def test_detection_mlx_needs_apple_silicon_anywhere_else(monkeypatch, no_ambient_ollama, platform_name, machine):
    """Done-when 1: given anything else, then mlx is unavailable with the
    reason "needs Apple Silicon"."""
    _fake_platform(monkeypatch, platform_name, machine)
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
    _fake_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: True)
    assert providers._works_here() == ("ollama", "openai", "claude", "scripted")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    assert "ollama, openai, claude, scripted" in str(err.value)


@contextlib.contextmanager
def _fake_ollama(monkeypatch, *, status: int = 200, delay: float = 0.0, replies: list[str] = ()):
    """conftest's FakeOllama (Ollama's version and chat endpoints on
    127.0.0.1 at an ephemeral port) with detection pointed at it. `status`
    is what GET /api/version answers; `delay` holds the answer that long.
    `replies` are the texts POST /api/chat answers with, in order; every
    chat request's JSON body is kept on `server.chats`."""
    server = FakeOllama(status=status, delay=delay, replies=replies)
    server.start()
    monkeypatch.setattr(providers, "OLLAMA_URL", server.endpoint)
    try:
        yield server
    finally:
        server.stop()


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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
    check ran before any image was read."""
    from melampus.cli import main

    port = _closed_port()
    monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{port}")
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "nowhere.jpg").symlink_to(tmp_path / "does-not-exist.jpg")

    code = main([str(folder), "--backend", "ollama", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code == 3, err
    assert f"No Ollama server is answering at http://127.0.0.1:{port}" in err
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
        monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{_closed_port()}")
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
        monkeypatch.setattr(providers, "OLLAMA_URL", f"http://127.0.0.1:{_closed_port()}")
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
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
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


def test_ollama_probe_timeout_is_one_second():
    assert providers.OLLAMA_PROBE_SECONDS == 1.0


# ---------------------------------------------------------------------------
# Card #404 through the CLI: `--detect-engines`, and the default engine.


@pytest.fixture()
def no_local_config(monkeypatch, tmp_path):
    """The developer's melampus.local.toml must not set the engine under a
    test about what happens when nothing sets it."""
    import melampus.config

    monkeypatch.setattr(melampus.config, "_local_config", lambda: tmp_path / "absent.toml")


def test_cli_detect_engines_prints_the_verdicts_as_json_in_order(
    monkeypatch, capsys, no_ambient_keys, no_ambient_ollama
):
    """Acceptance for Done-when 1 to 3: `melampus-id --detect-engines` needs no
    folder, prints one JSON list to stdout, in the owner's order, each item
    {engine, available, reason}, and exits 0. Faked off Apple Silicon with no
    Ollama: mlx and ollama say why not, the cloud engines say which key."""
    from melampus.cli import main

    _fake_platform(monkeypatch, "win32", "AMD64")

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
    the seam refuses, so nothing loads or runs."""
    import melampus.cli

    chosen: list[str] = []

    def refuse(config):
        chosen.append(config.model.backend)
        raise providers.BackendUnavailable("stopped at the seam")

    monkeypatch.setattr(melampus.cli, "build_primary_backend", refuse)
    assert melampus.cli.main(argv) == 3
    (engine,) = chosen
    return engine


def test_cli_default_engine_is_the_first_that_can_run_here(
    monkeypatch, photos, tmp_path, capsys, no_ambient_keys, no_local_config
):
    """No `--backend` and no `[model] backend`: the CLI picks the first engine
    detection says is available, in the owner's order. Off Apple Silicon with
    Ollama answering, that is ollama; the log line says so and how to choose."""
    _fake_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: True)

    engine = _chosen_engine(monkeypatch, [str(photos), "--cache", str(tmp_path / "cache.jsonl")])

    assert engine == "ollama"
    err = capsys.readouterr().err
    assert "engine: ollama" in err and "--backend" in err, err


def test_cli_default_engine_is_mlx_on_apple_silicon_and_openai_with_nothing_local(
    monkeypatch, photos, tmp_path, no_ambient_keys, no_local_config, no_ambient_ollama
):
    argv = [str(photos), "--cache", str(tmp_path / "cache.jsonl")]
    _fake_platform(monkeypatch, "darwin", "arm64")
    assert _chosen_engine(monkeypatch, argv) == "mlx"
    _fake_platform(monkeypatch, "linux", "x86_64")
    assert _chosen_engine(monkeypatch, argv) == "openai"


def test_cli_detection_never_overrides_a_chosen_engine(
    monkeypatch, photos, tmp_path, no_ambient_keys, no_local_config
):
    """`--backend` and `[model] backend` are the user's word; detection only
    fills the blank. Faked so detection would say ollama."""
    _fake_platform(monkeypatch, "win32", "AMD64")
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
    assert "claude-code" in providers._works_here()
    _fake_claude(monkeypatch, tmp_path, mode="not-signed-in")
    assert "claude-code" not in providers._works_here()


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
