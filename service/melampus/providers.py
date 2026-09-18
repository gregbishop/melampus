"""Cloud provider registry, and the primary-backend factory.

Two places build a cloud backend: escalation (the second-opinion tail on a Mac)
and the primary pipeline on machines with no local runtime (CLAUDE.md §3 made
the backend a seam; Windows is why the seam is now also a setting). Key lookup,
default model names and provider validation must not fork between them, so they
live here and both callers import them.
"""

from __future__ import annotations

import platform
import sys

from pydantic import SecretStr

from .backend import VLMBackend
from .config import MelampusConfig

#: Where each provider's key is looked for, in order, when the config has none.
#: Keys never cross providers: an Anthropic key must not silently authorise a
#: request to OpenAI, or the "which cloud am I using" question has no answer.
KEY_VARIABLES = {
    "anthropic": ("MELAMPUS_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("MELAMPUS_OPENAI_KEY", "OPENAI_API_KEY"),
}

#: Starting points only. Vision model names move faster than this file does —
#: check the provider's current listing and override in config when they age.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5",
}

#: The fake the unit tests run against, reachable from the CLI so the shipped
#: executable can be smoke-tested on a machine with no weights (card #399). It
#: answers nothing useful; it is here to prove the pipeline around it runs.
SCRIPTED = "scripted"

#: What `[model] backend` may be set to.
BACKEND_CHOICES = ("mlx", *sorted(KEY_VARIABLES), SCRIPTED)

#: The backends that run on this machine and bill nobody.
LOCAL_BACKENDS = ("mlx", SCRIPTED)


class BackendUnavailable(RuntimeError):
    """This machine cannot run the configured backend; the message says what to do."""


def normalise_provider(provider: str | None) -> str:
    name = (provider or "").strip().lower()
    if name not in KEY_VARIABLES:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {', '.join(sorted(KEY_VARIABLES))}. "
            "For any other OpenAI-compatible endpoint, use 'openai' with base_url set."
        )
    return name


def resolve_provider_key(provider: str, explicit: SecretStr | None = None) -> str | None:
    """Find the key without ever letting it live in tracked source.

    Order: explicit config (from the git-ignored local file or an override), then
    the provider's own environment variables.
    """
    import os

    if explicit:
        # SecretStr keeps it out of reprs and tracebacks; unwrap only here.
        return explicit.get_secret_value()
    for name in KEY_VARIABLES.get((provider or "").strip().lower(), ()):
        value = os.environ.get(name)
        if value:
            return value
    return None


def is_cloud_primary(config: MelampusConfig) -> bool:
    return (config.model.backend or "mlx").strip().lower() not in LOCAL_BACKENDS


def build_primary_backend(config: MelampusConfig) -> VLMBackend:
    """The backend the main pipeline talks to, per `[model] backend`.

    `mlx` is the default and the local-first path; it exists only on Apple
    Silicon. The cloud choices are for machines without a local runtime —
    they reuse the exact classes escalation uses, so prompts, schema validation
    and the corrective retry are identical wherever the answer comes from.
    """
    kind = (config.model.backend or "mlx").strip().lower()

    if kind == "mlx":
        # Both halves of the pyproject marker, or an Intel Mac passes the OS
        # check and then dies on a raw ModuleNotFoundError at warmup instead of
        # this message.
        if sys.platform != "darwin" or platform.machine() != "arm64":
            works_here = ", ".join(b for b in BACKEND_CHOICES if b != "mlx")
            raise BackendUnavailable(
                "The local MLX backend only runs on Apple Silicon Macs. The "
                f"backends that work on this machine are: {works_here}. Set "
                "[model] backend in the config (with the matching API key for a "
                "cloud provider), or pass --backend. See readme.md § Windows."
            )
        from .backend import MLXBackend

        return MLXBackend(config.model.repo, config.model.temperature)

    if kind == SCRIPTED:
        from .backend import ScriptedBackend

        return ScriptedBackend([])

    provider = normalise_provider(kind)

    # Import now, not on first request: a missing SDK should fail once, up front,
    # with an install hint — not once per frame mid-run.
    if provider == "anthropic":
        import anthropic  # noqa: F401
    else:
        import openai  # noqa: F401

    settings = config.model
    key = resolve_provider_key(provider, settings.api_key)
    model = settings.name or DEFAULT_MODELS[provider]

    from .backend import AnthropicBackend, OpenAIBackend

    if provider == "anthropic":
        return AnthropicBackend(
            key, model, effort=settings.effort, timeout=settings.timeout_seconds
        )
    return OpenAIBackend(
        key, model, base_url=settings.base_url, timeout=settings.timeout_seconds
    )


def apply_cloud_primary_defaults(config: MelampusConfig) -> list[str]:
    """Retune MLX-shaped defaults for a cloud primary, respecting explicit settings.

    Several defaults encode workarounds for the local runtime — the 1280 px ceiling
    exists because of an mlx-vlm token-window bug, the fallback ladder exists for
    its empty-generation failure, and the tight token caps assume a runtime that
    does not think before answering. None of that applies to a cloud model, and
    escalation already retunes them (escalate.build_cloud_identifier); a cloud
    primary deserves the same treatment.

    Only fields the user did not set are touched: `model_fields_set` distinguishes
    "the default" from "deliberately configured to the same number". Returns a
    description of each change, for the startup log.
    """
    changed: list[str] = []
    image, model = config.image, config.model

    if "max_edge" not in image.model_fields_set:
        image.max_edge = 2048
        changed.append("image.max_edge -> 2048 (the 1280 ceiling is an mlx-vlm bug)")
    if "fallback_edges" not in image.model_fields_set:
        image.fallback_edges = []
        changed.append("image.fallback_edges -> [] (retry ladder is for an mlx-vlm bug)")
    if "max_tokens" not in model.model_fields_set:
        model.max_tokens = 1200
        changed.append("model.max_tokens -> 1200 (thinking models spend tokens before JSON)")
    if "routing_max_tokens" not in model.model_fields_set:
        model.routing_max_tokens = 900
        changed.append("model.routing_max_tokens -> 900 (200 starves a thinking model)")
    if "cache_path" not in config.run.model_fields_set:
        # The reason escalation has its own cache file (config.py, EscalationConfig
        # .cache_path) applies with more force to a cloud primary: results carry
        # this backend's fingerprint, so sharing the local file would let the next
        # mlx pass silently overwrite answers that were paid for — and flipping
        # back would re-bill every one of them.
        config.run.cache_path = config.run.cache_path.with_name("identifications-cloud.jsonl")
        changed.append(
            "run.cache_path -> identifications-cloud.jsonl "
            "(cloud answers must not overwrite local ones)"
        )
    return changed
