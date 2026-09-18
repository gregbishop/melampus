"""The backend seam as a setting — the Windows / cloud-primary path.

The contract under test: `[model] backend` selects who answers, defaults keep the
Mac local-first path byte-for-byte unchanged, cloud selection fails fast and
clearly (missing SDK, missing key, unknown name), and the MLX-shaped defaults are
retuned for a cloud primary without ever overriding an explicit setting.
"""

from __future__ import annotations

import json
import platform
import sys
import types

import pytest
from conftest import PHOTO

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


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="the mlx default only constructs on Apple Silicon",
)
def test_default_backend_is_local_mlx():
    config = _cfg()
    assert config.model.backend == "mlx"
    assert not providers.is_cloud_primary(config)
    backend = providers.build_primary_backend(config)
    assert isinstance(backend, MLXBackend)
    assert backend.repo == config.model.repo


def test_mlx_is_refused_on_windows_with_directions(monkeypatch):
    monkeypatch.setattr(providers.sys, "platform", "win32")
    with pytest.raises(providers.BackendUnavailable) as err:
        providers.build_primary_backend(_cfg())
    message = str(err.value)
    assert "anthropic" in message and "openai" in message and "--backend" in message


def test_mlx_is_refused_on_intel_mac(monkeypatch):
    """darwin alone is not enough — the pyproject marker also requires arm64,
    so an Intel Mac must get the helpful refusal, not a ModuleNotFoundError
    at warmup."""
    monkeypatch.setattr(providers.sys, "platform", "darwin")
    monkeypatch.setattr(providers.platform, "machine", lambda: "x86_64")
    with pytest.raises(providers.BackendUnavailable):
        providers.build_primary_backend(_cfg())


def test_anthropic_primary_uses_provider_default_model(
    monkeypatch, no_ambient_keys, stub_sdks
):
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "key-from-env")
    config = _cfg(model={"backend": "anthropic"})
    assert providers.is_cloud_primary(config)
    backend = providers.build_primary_backend(config)
    assert isinstance(backend, AnthropicBackend)
    assert backend.model == providers.DEFAULT_MODELS["anthropic"]


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
        providers.build_primary_backend(_cfg(model={"backend": "anthropic"}))
    assert "MELAMPUS_ANTHROPIC_KEY" in str(err.value)


def test_unknown_backend_is_rejected_with_choices():
    with pytest.raises(ValueError) as err:
        providers.build_primary_backend(_cfg(model={"backend": "gemini"}))
    message = str(err.value)
    assert "gemini" in message and "anthropic" in message and "openai" in message


def test_key_resolution_prefers_config_then_specific_then_generic(monkeypatch):
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "specific")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic")
    from pydantic import SecretStr

    assert providers.resolve_provider_key("anthropic", SecretStr("explicit")) == "explicit"
    assert providers.resolve_provider_key("anthropic") == "specific"
    monkeypatch.delenv("MELAMPUS_ANTHROPIC_KEY")
    assert providers.resolve_provider_key("anthropic") == "generic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert providers.resolve_provider_key("anthropic") is None


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
    tmp_path, capsys, monkeypatch
):
    """Card #400, Done-when 2: given the Windows executable, when the local MLX
    engine is requested, then it says clearly that MLX needs Apple Silicon and
    names the engines that work here — every backend but mlx, so the list can
    never drift from BACKEND_CHOICES."""
    from PIL import Image

    from melampus.cli import main

    monkeypatch.setattr(providers.sys, "platform", "win32")
    monkeypatch.setattr(providers.platform, "machine", lambda: "AMD64")
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (640, 480), (90, 120, 70)).save(photos / "frame.jpg")

    code = main([str(photos), "--backend", "mlx", "--cache", str(tmp_path / "cache.jsonl")])

    err = capsys.readouterr().err
    assert code != 0
    assert "Apple Silicon" in err
    for works_here in ("anthropic", "openai", "scripted"):
        assert works_here in err, f"{works_here!r} is not named as working here:\n{err}"
    assert tuple(b for b in providers.BACKEND_CHOICES if b != "mlx") == ("anthropic", "openai", "scripted")
