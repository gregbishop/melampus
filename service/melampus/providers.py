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
import urllib.request
from dataclasses import dataclass

from pydantic import SecretStr

from .backend import VLMBackend
from .config import MelampusConfig

#: Where each provider's key is looked for, in order, when the config has none.
#: Keys never cross providers: an Anthropic key must not silently authorise a
#: request to OpenAI, or the "which cloud am I using" question has no answer.
#: The providers carry the engine names the user chooses between (card #403):
#: `claude` is the Anthropic backend everywhere a user names it; the key
#: variables keep the vendor's name.
KEY_VARIABLES = {
    "claude": ("MELAMPUS_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("MELAMPUS_OPENAI_KEY", "OPENAI_API_KEY"),
}

#: Starting points only. Vision model names move faster than this file does —
#: check the provider's current listing and override in config when they age.
DEFAULT_MODELS = {
    "claude": "claude-opus-5",
    "openai": "gpt-5",
}

#: The fake the unit tests run against, reachable from the CLI so the shipped
#: executable can be smoke-tested on a machine with no weights (card #399). It
#: answers nothing useful; it is here to prove the pipeline around it runs.
SCRIPTED = "scripted"

#: Named so the CLI accepts it (card #403: the user's choice of engine needs a
#: home before it needs a dialog); its backend is card #406's. Until that
#: lands, asking for it is refused below, the way mlx is refused off Apple
#: Silicon.
OLLAMA = "ollama"

#: The engines the user chooses between, in the owner's order, then the fake.
#: `detect_engines` tries them in this order for a default (card #404): the
#: first that can run on this machine.
BACKEND_CHOICES = ("mlx", OLLAMA, "openai", "claude", SCRIPTED)

#: Where the local Ollama server listens. Ollama's docs/faq.mdx: "Ollama binds
#: 127.0.0.1 port 11434 by default." One constant, so card #406 can make it a
#: config value.
OLLAMA_URL = "http://127.0.0.1:11434"

#: How long the probe waits for the local server. Loopback answers in
#: milliseconds or not at all; a second is a firewall's silence, not Ollama's.
OLLAMA_PROBE_SECONDS = 1.0

#: Where to get Ollama when nothing answers at OLLAMA_URL.
OLLAMA_INSTALL = "https://ollama.com/download"

#: The backends that run on this machine and bill nobody.
LOCAL_BACKENDS = ("mlx", OLLAMA, SCRIPTED)


class BackendUnavailable(RuntimeError):
    """This machine cannot run the configured backend; the message says what to do."""


def on_apple_silicon() -> bool:
    """The pyproject marker for mlx-vlm, as a predicate: the one place the
    runtime check, the build script and the tests' skips ask whether MLX
    exists here. Both halves, or an Intel Mac passes the OS check and then
    dies on a raw ModuleNotFoundError at warmup instead of the message."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _refusal(reason: str, *, works_here: tuple[str, ...]) -> BackendUnavailable:
    """One shape for every refusal: what is wrong, what works here, how to switch."""
    return BackendUnavailable(
        f"{reason} The backends that work on this machine are: "
        f"{', '.join(works_here)}. Set [model] backend in the config (with the "
        "matching API key for a cloud provider), or pass --backend. See "
        "readme.md § Windows."
    )


def ollama_answers() -> bool:
    """Whether an Ollama server answers at OLLAMA_URL: GET /api/version
    (Ollama's docs/api.md § Version) within OLLAMA_PROBE_SECONDS, status 200.
    Connection refused, a timeout, a non-200: unavailable. Never raises; a
    probe reports."""
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/version", timeout=OLLAMA_PROBE_SECONDS
        ) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - every failure means the same thing: not here
        return False


@dataclass(frozen=True, slots=True)
class EngineVerdict:
    """Whether one engine can run on this machine, and why or why not, in the
    words a user sees: the reason is what makes an unavailable engine a
    greyed-out choice rather than a mystery (card #404)."""

    engine: str
    available: bool
    reason: str


def _key_required(engine: str) -> str:
    specific, generic = KEY_VARIABLES[engine]
    return f"API key required: set {specific} (or {generic})"


def detect_engines() -> list[EngineVerdict]:
    """One verdict per engine, in the owner's order (BACKEND_CHOICES without the
    test fake). This is the one place that knows whether an engine can run
    here: the refusals' "what works" list and the CLI's default both come from
    it, so they cannot disagree with what the dialog (card #405) shows."""
    apple_silicon = on_apple_silicon()
    ollama = ollama_answers()
    return [
        EngineVerdict(
            "mlx", apple_silicon,
            "runs locally on this Apple Silicon Mac" if apple_silicon else "needs Apple Silicon",
        ),
        EngineVerdict(
            OLLAMA, ollama,
            f"Ollama is answering at {OLLAMA_URL}" if ollama
            else f"no Ollama server at {OLLAMA_URL}; install it from {OLLAMA_INSTALL}",
        ),
        EngineVerdict("openai", True, _key_required("openai")),
        EngineVerdict("claude", True, _key_required("claude")),
    ]


def _works_here() -> tuple[str, ...]:
    """The backends this machine can run, as detection says, plus the fake."""
    return (*(v.engine for v in detect_engines() if v.available), SCRIPTED)


def default_engine() -> str:
    """What runs when nothing names an engine: the first detection says is
    available, in the owner's order. Were none available, mlx, whose refusal
    already says what to do."""
    return next((v.engine for v in detect_engines() if v.available), "mlx")


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
        if not on_apple_silicon():
            raise _refusal(
                "The local MLX backend only runs on Apple Silicon Macs.",
                works_here=_works_here(),
            )
        from .backend import MLXBackend

        return MLXBackend(config.model.repo, config.model.temperature)

    if kind == OLLAMA:
        raise _refusal(
            "The Ollama engine is not built yet (card #406).", works_here=_works_here()
        )

    if kind == SCRIPTED:
        from .backend import ScriptedBackend

        return ScriptedBackend([])

    provider = normalise_provider(kind)

    # Import now, not on first request: a missing SDK should fail once, up front,
    # with an install hint — not once per frame mid-run.
    if provider == "claude":
        import anthropic  # noqa: F401
    else:
        import openai  # noqa: F401

    settings = config.model
    key = resolve_provider_key(provider, settings.api_key)
    model = settings.name or DEFAULT_MODELS[provider]

    from .backend import AnthropicBackend, OpenAIBackend

    if provider == "claude":
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
