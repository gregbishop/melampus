"""Cloud escalation for the low-confidence tail (CLAUDE.md §6.6).

The local model handles the bulk of a catalog well and costs nothing to run. What
it does not handle is the tail: frames where it abstains, splits between two
similar species, or names something that does not occur within a thousand miles.
That tail is small — a few hundred frames out of a few thousand — which is exactly
the shape where a frontier model is worth paying for.

Everything here exists to keep that from becoming a foot-gun:

* **Off by default, and needs a key.** Two deliberate acts before a photograph
  leaves the machine, in a tool whose whole premise is that it does not.
* **Same pixels-only guarantee.** Escalated frames go through
  `images.staged_pixels` exactly as local ones do. No filename, no EXIF, no
  keywords — over the network least of all.
* **A hard cap, and no silent truncation.** When the cap bites, the most uncertain
  frames go first and the ones left behind are counted and reported.
* **Its own cache file.** Cloud answers never overwrite local ones, and a second
  run re-bills nothing.
* **Provenance kept.** Each escalated result records which model answered, why it
  was asked, and what the local model had said — which is what turns "is this
  worth the money?" into a number instead of an opinion.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .cache import ResultCache
from .config import EscalationConfig, MelampusConfig
from .images import content_hash
from .schema import ImageResult, Taxon

#: Rough token usage per image, for the estimate printed before spending anything.
#:
#: Two calls per image (taxon routing, then the taxon prompt). A 2048 px frame is
#: on the order of 2.5k image tokens, the prompt a few hundred more, and replies
#: run a few hundred out. Deliberately an over-estimate: a bill smaller than the
#: warning is a good surprise, the other way round is not. Per-token prices are
#: config, not constants, because they differ by provider and by model.
_INPUT_TOKENS_PER_IMAGE = 5_400
_OUTPUT_TOKENS_PER_IMAGE = 900

#: Where each provider's key is looked for, in order, when the config has none.
#: Keys never cross providers: an Anthropic key must not silently authorise a
#: request to OpenAI, or the "which cloud am I using" question has no answer.
_KEY_VARIABLES = {
    "anthropic": ("MELAMPUS_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("MELAMPUS_OPENAI_KEY", "OPENAI_API_KEY"),
}

#: Starting points only. Vision model names move faster than this file does —
#: check the provider's current listing and override with --escalate-model.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5",
}


class EscalationRefused(RuntimeError):
    """Raised instead of quietly doing nothing, or quietly doing something costly."""


def _known_provider(config: EscalationConfig) -> str:
    provider = (config.provider or "").strip().lower()
    if provider not in _KEY_VARIABLES:
        raise EscalationRefused(
            f"Unknown escalation provider '{config.provider}'. "
            f"Supported: {', '.join(sorted(_KEY_VARIABLES))}. "
            "For any other OpenAI-compatible endpoint, use provider = 'openai' "
            "with base_url set."
        )
    return provider


def resolve_api_key(config: EscalationConfig) -> str | None:
    """Find the key without ever letting it live in tracked source.

    Order: explicit config (from the git-ignored local file or an override), then
    the provider's own environment variables.
    """
    if config.api_key:
        return config.api_key
    for name in _KEY_VARIABLES.get((config.provider or "").strip().lower(), ()):
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_model(config: EscalationConfig) -> str:
    """The configured model, or this provider's default."""
    if config.model:
        return config.model
    return DEFAULT_MODELS.get((config.provider or "").strip().lower(), "")


def estimate_cost_usd(images: int, config: EscalationConfig | None = None) -> float:
    """Approximate spend for escalating `images` frames, at the configured rates."""
    input_rate = config.input_usd_per_mtok if config else 5.0
    output_rate = config.output_usd_per_mtok if config else 25.0
    per_image = (
        _INPUT_TOKENS_PER_IMAGE * input_rate / 1_000_000
        + _OUTPUT_TOKENS_PER_IMAGE * output_rate / 1_000_000
    )
    return max(0, images) * per_image


def should_escalate(
    result: ImageResult,
    *,
    config: EscalationConfig,
    range_flagged: bool = False,
) -> str | None:
    """Why this frame deserves a second opinion, or None if it does not.

    Returning the reason rather than a bare boolean means every paid call can say
    what it was for, which is what makes the run auditable afterwards.
    """
    if result.escalated:
        return None  # idempotency (§5.3): never re-bill a frame

    # An unreadable file is not a hard identification. A better model cannot open it
    # either, so this would be spending money to receive the same error.
    if result.status == "error":
        return None

    if result.status != "ok" or result.identification is None:
        return "unprocessed locally"

    ident = result.identification

    # `taxon: none` abstains by construction — that is the pipeline being certain
    # there is no organism, not uncertain about which one. Escalating those would
    # spend the entire budget rediscovering that empty frames are empty.
    if ident.taxon is Taxon.NONE:
        return None

    if ident.abstain or not ident.candidates:
        return "local model abstained" if config.on_abstain else None

    top = max(c.confidence for c in ident.candidates)
    if top < config.confidence_below:
        return f"top confidence {top:.2f} below {config.confidence_below:.2f}"

    # §4.3 already routes these to human review. A second opinion first is cheap
    # relative to the photographer's time, and it is the pile most likely to
    # contain either a real model error or a genuinely notable record.
    if range_flagged and config.on_range_flag:
        return "top candidate flagged out of range"

    return None


def _priority(result: ImageResult, range_flagged: bool) -> float:
    """Sort key for spending order. Lower means less certain, so it goes first."""
    if result.status != "ok" or result.identification is None:
        return -1.0
    ident = result.identification
    if ident.abstain or not ident.candidates:
        return 0.0
    top = max(c.confidence for c in ident.candidates)
    # A confident but geographically impossible answer is less trustworthy than its
    # number suggests, so it queues ahead of a merely middling one.
    return top * 0.5 if range_flagged else top


def select_for_escalation(
    results: Iterable[ImageResult],
    *,
    config: EscalationConfig,
    range_flagged: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[tuple[ImageResult, str]], list[tuple[ImageResult, str]]]:
    """Split candidates into what will be sent and what the cap excluded.

    Both halves are returned. A cap that silently drops work reads as "covered
    everything" when it did not, which is the failure this project has already had
    once with a counter that reported writes it never made.
    """
    scored: list[tuple[float, int, ImageResult, str]] = []
    for order, result in enumerate(results):
        flagged = result.file in range_flagged
        reason = should_escalate(result, config=config, range_flagged=flagged)
        if reason is not None:
            scored.append((_priority(result, flagged), order, result, reason))

    scored.sort(key=lambda row: (row[0], row[1]))
    ordered = [(result, reason) for _, _, result, reason in scored]

    cap = max(0, config.max_images)
    return ordered[:cap], ordered[cap:]


@dataclass
class EscalationRun:
    selected: int = 0
    processed: int = 0
    dropped: int = 0
    changed: int = 0        # cloud disagreed with, or resolved, the local answer
    resolved: int = 0       # local abstained or failed; cloud produced a candidate
    refused: int = 0
    errors: int = 0
    seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


def build_cloud_identifier(config: MelampusConfig, api_key: str | None = None):
    """An `Identifier` wired to a cloud provider instead of MLX.

    Same prompts, same schema, same corrective retry — only the backend and the
    staging size differ. The larger long edge is the point: these are the frames
    resolution might actually rescue, and the local ceiling that forced 1280 px is
    a runtime bug that does not apply here.
    """
    from .backend import AnthropicBackend, OpenAIBackend
    from .identify import Identifier

    settings = config.escalation
    provider = _known_provider(settings)
    key = api_key or resolve_api_key(settings)
    model = resolve_model(settings)

    if provider == "anthropic":
        backend = AnthropicBackend(
            key, model, effort=settings.effort, timeout=settings.timeout_seconds,
        )
    else:
        backend = OpenAIBackend(
            key, model, base_url=settings.base_url, timeout=settings.timeout_seconds,
        )
    cloud_config = config.model_copy(deep=True)
    cloud_config.image.max_edge = settings.max_edge
    cloud_config.image.fallback_edges = []
    cloud_config.model.max_tokens = settings.max_tokens
    return Identifier(backend, cloud_config)


def escalate(
    paths: Sequence[Path],
    local_cache: ResultCache,
    cloud_cache: ResultCache,
    *,
    identifier,
    config: MelampusConfig,
    range_flagged: frozenset[str] | set[str] = frozenset(),
    dry_run: bool = False,
    on_result: Callable[[ImageResult, str], None] | None = None,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> EscalationRun:
    """Re-run the uncertain frames against the Claude API.

    `paths` is the same folder listing the local pass used; local results are looked
    up by content hash, so a rename between runs changes nothing.
    """
    settings = config.escalation
    _known_provider(settings)  # fail on a typo'd provider before any work happens
    if not settings.enabled:
        raise EscalationRefused(
            "Cloud escalation is not enabled. It sends photographs to the Claude API, "
            "so it stays off until you ask for it: set escalation.enabled = true in "
            "melampus.local.toml, or pass --escalate."
        )
    # A dry run sends nothing, so it must work before you have a key — seeing the
    # count and the likely cost is exactly how you decide whether to get one.
    if not dry_run and resolve_api_key(settings) is None:
        raise EscalationRefused(
            "Cloud escalation needs an Anthropic API key. Export MELAMPUS_ANTHROPIC_KEY, "
            "or add it to melampus.local.toml (git-ignored). Never commit it. "
            "Add --escalate-dry-run to see what it would send and cost without one."
        )

    run = EscalationRun()

    by_hash: dict[str, Path] = {}
    for path in paths:
        try:
            by_hash[content_hash(path)] = path
        except OSError as exc:
            run.notes.append(f"{path.name}: unreadable ({exc})")

    pending: list[ImageResult] = []
    for digest, path in by_hash.items():
        if cloud_cache.get(digest) is not None:
            continue  # already paid for
        record = local_cache.get(digest)
        if record is None:
            run.notes.append(f"{path.name}: no local result yet, run identification first")
            continue
        pending.append(record)

    selected, dropped = select_for_escalation(
        pending, config=settings, range_flagged=range_flagged
    )
    run.selected = len(selected)
    run.dropped = len(dropped)
    run.estimated_cost_usd = estimate_cost_usd(len(selected), settings)

    if dropped:
        run.notes.append(
            f"{len(dropped)} further frame(s) qualified but exceed max_images="
            f"{settings.max_images}; raise it to include them"
        )

    if dry_run:
        return run

    started = time.perf_counter()
    for record, reason in selected:
        path = by_hash.get(record.content_hash)
        if path is None:  # pragma: no cover - pending is built from by_hash
            continue
        try:
            result = identifier.identify(path)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end the run
            run.errors += 1
            run.notes.append(f"{record.file}: {type(exc).__name__}: {exc}")
            log(f"{record.file}: escalation failed: {exc}")
            continue

        result.escalated = True
        result.escalation_model = identifier.backend.name
        result.escalation_reason = reason
        result.local_identification = record.identification

        cloud_cache.put(result)  # checkpoint before anything else can fail
        run.processed += 1

        if result.status != "ok":
            run.refused += 1
        else:
            local_top = record.identification.top() if record.identification else None
            cloud_top = result.identification.top() if result.identification else None
            if cloud_top is not None and local_top is None:
                run.resolved += 1
                run.changed += 1
            elif cloud_top is not None and local_top is not None:
                if cloud_top.common_name.strip().lower() != local_top.common_name.strip().lower():
                    run.changed += 1

        if on_result is not None:
            on_result(result, reason)

    run.seconds = time.perf_counter() - started
    return run


def merged_results(local_cache: ResultCache, cloud_cache: ResultCache) -> list[ImageResult]:
    """Local results with escalated ones substituted in. Cloud wins where present."""
    merged = {r.content_hash: r for r in local_cache.results()}
    merged.update({r.content_hash: r for r in cloud_cache.results() if r.status == "ok"})
    return list(merged.values())
