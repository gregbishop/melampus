"""Configuration. Data, not code — everything injectable for the eventual service."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr


# The checkout root: two levels up from this file. Both roots below are it
# unless the program is running frozen.
_CHECKOUT = Path(__file__).resolve().parents[2]


def _bundle() -> Path | None:
    """PyInstaller's unpack directory when running inside the executable
    (tools/build_binary.py); None in a checkout. The bootloader sets
    sys._MEIPASS before the package runs."""
    bundle = getattr(sys, "_MEIPASS", None)
    return Path(bundle) if bundle else None


def _repo_root() -> Path:
    """Where prompts/ sits relative to the code: what ships with the program.

    In a checkout, the checkout root. Inside the executable, the unpack
    directory, which the build lays out the same way: prompts/ at its top
    level. Nothing the user owns belongs here: the directory is deleted when
    the process exits. That is `_data_root()`.
    """
    return _bundle() or _CHECKOUT


def _data_root() -> Path:
    """Where the caches and melampus.local.toml sit: what the user owns.

    In a checkout, the checkout root, as before. Inside the executable, the
    directory the executable is in, which outlives the process; the unpack
    directory does not, so a cache written there is thrown away at exit and a
    local config there is never read. Beside the executable mirrors the
    checkout layout and needs no per-platform decision.
    """
    return Path(sys.executable).resolve().parent if _bundle() else _CHECKOUT


def _cache(name: str) -> Path:
    return _data_root() / ".melampus_cache" / name


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Base):
    # Which inference engine the *primary* pipeline talks to. CLAUDE.md §3 built
    # the backend seam; this makes it a setting, which is what lets the same
    # repo run on a machine with no local runtime at all (Windows).
    #   mlx       — local, Apple Silicon only. The default; the local-first path.
    #   anthropic — the Claude API. Every frame billed: see docs/config.md.
    #   openai    — OpenAI, or anything chat-completions-compatible via base_url.
    backend: str = "mlx"
    # CLAUDE.md §3 wants the model to be a setting, never a hardcode. (mlx only.)
    repo: str = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"
    # Cloud model name; None means the provider's default (providers.DEFAULT_MODELS).
    name: str | None = None
    # OpenAI-compatible endpoint override: OpenRouter, LM Studio, vLLM, a proxy, …
    base_url: str | None = None
    # Never set here in tracked source. Comes from MELAMPUS_ANTHROPIC_KEY /
    # MELAMPUS_OPENAI_KEY, the provider's own variable, or melampus.local.toml.
    api_key: SecretStr | None = None
    # Anthropic-only; ignored elsewhere.
    effort: str = "high"
    timeout_seconds: float = 180.0
    # Cloud primary only; the mlx backend ignores it. Same rationale as
    # escalation.max_images: a cloud primary bills every frame, and a mistyped
    # flag or an over-broad selection must not turn into an unexpected invoice
    # — the ceiling is low enough to notice and must be raised deliberately.
    # Unlike escalation there is no "most uncertain first" ordering to salvage a
    # truncated run, so exceeding the cap refuses outright rather than billing
    # an arbitrary subset.
    max_images: int = Field(default=200, ge=0, le=5000)
    max_tokens: int = 900
    # Identification wants determinism, not creativity.
    temperature: float = 0.0
    routing_max_tokens: int = 200


class ImageConfig(_Base):
    # Long edge sent to the model.
    #
    # mlx-vlm 0.6.7 + Qwen3-VL collapses to an immediate EOS once the prompt passes
    # roughly 2.1k tokens (measured: 2109 generates fine, 2183 returns one token).
    # That is far below the model's real context window, so it is a bug in the
    # runtime rather than a model limit. Vision tokens dominate that budget, so the
    # image size is the practical lever: 1280px keeps a full taxon prompt near
    # ~1.8k tokens with usable headroom.
    max_edge: int = 1280
    # Tried in order if generation comes back empty. The exact threshold shifts with
    # image aspect ratio, so a fixed size alone is not reliable.
    fallback_edges: list[int] = Field(default_factory=lambda: [1024, 768])
    jpeg_quality: int = 92


class QualityConfig(_Base):
    """Technical quality scoring (CLAUDE.md §4.1). Every weight lives here."""

    # Sharpness is scale dependent, so everything is measured at one working size.
    working_long_edge: int = 1600
    focus_window: int = 15
    focus_percentile: float = 85.0
    # p99, not the mean: measured on the corpus, a fully defocused frame has a
    # HIGHER mean focus value than a sharp one (2.30 vs 2.08) because smooth
    # bokeh is quiet while uniform softness is noisy. Only the high percentiles
    # separate them.
    # p99.9, not p99. A smooth pale subject — a Snowy Egret — has very little
    # texture to measure, so p99 lands on plumage rather than on the sharp bill
    # and eye, and a genuinely excellent frame scored one star. p99.9 finds the
    # sharpest structures that are actually present, and it still separates
    # frames *within* a burst (38.4 vs 24.5) where max cannot (230 vs 220).
    region_percentile: float = 99.9

    # Subject detection
    merge_dilate: int = 9
    min_blob_area_frac: float = 0.002
    area_exponent: float = 0.35
    centrality_strength: float = 0.30
    box_pad_frac: float = 0.08
    saliency_resize: int = 64
    saliency_blur_sigma: float = 2.5

    # Raw focus energy -> 0-100. Retune these first if real photographs cluster
    # at one end of the range.
    # Calibrated against real frames from the corpus. Subject p99 values run
    # roughly 4 (fully defocused) to 48 (tack sharp), so these are the knees.
    # Note that smooth pale subjects such as a Snowy Egret genuinely carry less
    # high-frequency detail and will score lower than a patterned bird at the
    # same focus accuracy.
    # Sampled across the real corpus (194 frames): raw subject p99 runs p10=13.5,
    # p50=30, p90=52, p99=63. Knees at 14 and 55 spread the corpus across the
    # range instead of pinning three quarters of it at 100.
    # Sampled over the corpus at p99.9: p10=23.7, p50=47.4, p90=75.9.
    knee_low: float = 22.0
    knee_high: float = 78.0
    size_reference_frac: float = 0.08
    size_gain_strength: float = 0.25
    size_gain_max: float = 1.35

    # Motion vs defocus
    anisotropy_floor: float = 0.15
    anisotropy_ceiling: float = 0.55

    # Catchlight / eye
    search_percentile: float = 99.0
    min_absolute_brightness: int = 200
    min_area_frac_of_subject: float = 0.00005
    max_area_frac_of_subject: float = 0.02
    min_circularity: float = 0.55
    min_ring_contrast: float = 45.0
    ring_dilate: int = 5
    patch_radius_mult: float = 6.0
    min_eye_confidence: float = 0.45

    # Exposure. Some clipping is always present on specular highlights and sky,
    # so a penalty only accrues past the tolerance.
    highlight_threshold: int = 254
    shadow_threshold: int = 1
    highlight_tolerance_pct: float = 0.5
    shadow_tolerance_pct: float = 0.5
    highlight_full_penalty_pct: float = 12.0
    shadow_full_penalty_pct: float = 12.0

    edge_margin_frac: float = 0.01

    # Composite weights. Eye sharpness dominates because it is what a wildlife
    # photographer actually culls on; motion is near-neutral because a
    # directional wingbeat is often the point of the photograph.
    weight_eye_sharpness: float = 0.40
    weight_subject_sharpness: float = 0.35
    weight_focus: float = 0.20
    weight_exposure: float = 0.05
    weight_motion: float = 0.00


class OccurrenceConfig(_Base):
    """Location and season re-ranking (CLAUDE.md §4.3)."""

    enabled: bool = True
    # GBIF needs no key and covers every taxon. eBird has far denser bird data but
    # requires a free token from https://ebird.org/api/keygen — optional, additive.
    # SecretStr: pydantic renders these as "**********" in repr, so a debug print,
    # a validation error on a sibling field, or a bug-report attachment cannot
    # leak the key. The docs promise the key never reaches a loggable object;
    # this makes that structural rather than a convention.
    ebird_token: SecretStr | None = None
    # Bodies without a GPS receiver produce photos with no coordinates, so a
    # default location is the only way §4.3 can run on them. There is no sane
    # universal default — unset, occurrence re-ranking is skipped for photos
    # without GPS. Per-photo GPS, when present, always wins over this.
    default_latitude: float | None = None
    default_longitude: float | None = None
    default_location_name: str | None = None
    # 50 km comfortably covers a refuge and its surroundings without reaching into
    # a different faunal region.
    radius_km: float = 50.0
    cache_path: Path = Field(default_factory=lambda: _cache("occurrence.json"))
    # Below this many regional records a species is present but scarce: demote
    # gently and mark notable, rather than treating it as absent.
    notable_threshold: int = 25
    # Multipliers applied to an ordinal confidence. Not probabilities.
    absent_penalty: float = 0.15
    notable_penalty: float = 0.6


class EscalationConfig(_Base):
    """Optional second opinion from the Claude API on the hard tail (CLAUDE.md §6.6).

    The local model is right about most frames and cheap about all of them. It is
    wrong, or abstains, on a minority — and that minority is where a frontier model
    earns its cost, precisely because the volume is small.

    Two things make this safe to ship in a tool whose selling point is that it runs
    locally. It is **off by default**, and it is the only path in the project that
    sends a photograph off the machine, so it is opt-in twice over: a setting and a
    key. And it obeys the same pixels-only rule as everything else — escalated frames
    go through `images.staged_pixels` exactly as local ones do, so filenames, EXIF and
    keywords do not travel either.
    """

    enabled: bool = False

    # Which cloud to ask. The backend seam in backend.py is what makes this a
    # one-line choice rather than a second pipeline: both providers get the same
    # prompts, the same schema validation and the same corrective retry.
    #   anthropic — the Claude API
    #   openai    — the OpenAI API, or anything speaking its chat-completions shape
    #               (set base_url for OpenRouter, LM Studio, vLLM, a proxy, …)
    provider: str = "anthropic"
    base_url: str | None = None

    # None means "this provider's default" — see escalate.DEFAULT_MODELS. Vision
    # model names change often, so treat the defaults as a starting point and
    # override with --escalate-model rather than assuming they are current.
    model: str | None = None
    # Never set here in tracked source. Comes from MELAMPUS_ANTHROPIC_KEY /
    # MELAMPUS_OPENAI_KEY, the provider's own variable, or the git-ignored
    # melampus.local.toml.
    api_key: SecretStr | None = None
    # Anthropic-only; ignored by other providers.
    effort: str = "high"
    max_tokens: int = 1200
    # Stage A (taxon routing) budget. The local default is 200, which is sized for
    # a runtime that does not think before answering. On a model where thinking is
    # on by default, max_tokens caps thinking AND output together, so 200 is
    # consumed before any JSON appears — every frame would fail, twice, billed.
    routing_max_tokens: int = 900

    # Used only for the estimate printed before spending anything. Defaults are the
    # Claude Opus 5 rate; change them when you change provider or model, or the
    # warning will be confidently wrong.
    input_usd_per_mtok: float = 5.0
    output_usd_per_mtok: float = 25.0
    # These are the frames the local model could not call, so resolution is the
    # cheapest lever left. Claude reads up to 2576 px on the long edge; 2048 keeps
    # most of that benefit without paying for the top of the image-token curve.
    max_edge: int = 2048
    timeout_seconds: float = 180.0

    # Triggers. Confidence is an ordinal hint from the local model, not a
    # probability — this threshold is a ranking cut, not a calibrated one.
    confidence_below: float = 0.80
    on_abstain: bool = True
    on_range_flag: bool = True

    # A hard ceiling on how many frames one run may bill for. The corpus has 836
    # frames needing review; a mistyped flag should not turn into an unexpected
    # invoice, so the cap is low enough to notice and must be raised deliberately.
    max_images: int = Field(default=200, ge=0, le=5000)

    # Cloud answers live in their own file. Merging them into the local cache would
    # give them a foreign run fingerprint, and the next local pass would decide they
    # were stale and quietly overwrite work that was paid for.
    cache_path: Path = Field(default_factory=lambda: _cache("escalations.jsonl"))


class RunConfig(_Base):
    # Which routing prompt to use. 'wildlife' asks what organism this is;
    # 'sport' asks what activity this is. Keeping them separate avoids the
    # obvious failure of a footballer being routed to 'mammal' and asked for a
    # species, and keeps each prompt short enough to stay under the runtime's
    # token ceiling.
    profile: str = "wildlife"
    prompts_dir: Path = Field(default_factory=lambda: _repo_root() / "prompts")
    cache_path: Path = Field(default_factory=lambda: _cache("identifications.jsonl"))
    # One corrective retry on schema-validation failure, per CLAUDE.md §4.2.
    max_retries: int = 1


class MelampusConfig(_Base):
    model: ModelConfig = Field(default_factory=ModelConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    occurrence: OccurrenceConfig = Field(default_factory=OccurrenceConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _local_config() -> Path:
    """melampus.local.toml beside the data, unless MELAMPUS_LOCAL_CONFIG names
    another file, in which case that file is the local layer and the one
    beside the data is not read. The CLI and the executable each have their
    own data root, so the variable is how the executable smoke tests hand
    both one synthetic configuration."""
    named = os.environ.get("MELAMPUS_LOCAL_CONFIG")
    return Path(named).expanduser() if named else _data_root() / "melampus.local.toml"


def _secrets_from_environment() -> dict[str, Any]:
    """Credentials come from the environment, never from tracked source.

    Precedence, lowest to highest: packaged defaults, melampus.local.toml
    (git-ignored), explicit --config file, environment variable, keyword override.
    """
    token = os.environ.get("MELAMPUS_EBIRD_TOKEN")
    secrets: dict[str, Any] = {}
    if token:
        secrets["occurrence"] = {"ebird_token": token}
    # The Anthropic key is deliberately *not* merged in here. It is resolved at the
    # point of use by escalate.resolve_api_key, so it never sits in a config object
    # that something might serialise into a log, a report or a bug attachment.
    return secrets


def load_config(
    path: str | Path | None = None, *, use_local: bool = True, **overrides: Any
) -> MelampusConfig:
    """Build config from defaults, the local file, an explicit file, then overrides.

    `use_local=False` skips melampus.local.toml. Tests need that: the file is
    git-ignored, so a suite that asserts on defaults would otherwise be asserting
    on whatever the developer happens to have configured — and would break for
    anyone who follows the documentation and puts their API key there.
    """
    data: dict[str, Any] = {}
    local = _local_config()
    if use_local and local.is_file():
        with local.open("rb") as handle:
            data = _deep_merge(data, tomllib.load(handle))
    if path is not None:
        file = Path(path).expanduser()
        if not file.is_file():
            raise FileNotFoundError(f"Config file not found: {file}")
        with file.open("rb") as handle:
            data = _deep_merge(data, tomllib.load(handle))
    data = _deep_merge(data, _secrets_from_environment())
    if overrides:
        data = _deep_merge(data, overrides)
    return MelampusConfig.model_validate(data)
