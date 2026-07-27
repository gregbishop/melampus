"""Configuration. Data, not code — everything injectable for the eventual service."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Base):
    # CLAUDE.md §3 wants the model to be a setting, never a hardcode.
    repo: str = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"
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


class OccurrenceConfig(_Base):
    """Location and season re-ranking (CLAUDE.md §4.3)."""

    enabled: bool = True
    # GBIF needs no key and covers every taxon. eBird has far denser bird data but
    # requires a free token from https://ebird.org/api/keygen — optional, additive.
    ebird_token: str | None = None
    # The Canon R3 has no GPS receiver and nothing in this catalog carries
    # coordinates, so a default location is the only way §4.3 can run at all.
    # Per-photo GPS, when present, always wins over this.
    default_latitude: float | None = 28.65
    default_longitude: float | None = -80.72
    default_location_name: str = "Merritt Island National Wildlife Refuge, FL"
    # 50 km comfortably covers a refuge and its surroundings without reaching into
    # a different faunal region.
    radius_km: float = 50.0
    cache_path: Path = REPO_ROOT / ".melampus_cache" / "occurrence.json"
    # Below this many regional records a species is present but scarce: demote
    # gently and mark notable, rather than treating it as absent.
    notable_threshold: int = 25
    # Multipliers applied to an ordinal confidence. Not probabilities.
    absent_penalty: float = 0.15
    notable_penalty: float = 0.6


class RunConfig(_Base):
    prompts_dir: Path = REPO_ROOT / "prompts"
    cache_path: Path = REPO_ROOT / ".melampus_cache" / "identifications.jsonl"
    # One corrective retry on schema-validation failure, per CLAUDE.md §4.2.
    max_retries: int = 1


class MelampusConfig(_Base):
    model: ModelConfig = Field(default_factory=ModelConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    occurrence: OccurrenceConfig = Field(default_factory=OccurrenceConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, **overrides: Any) -> MelampusConfig:
    data: dict[str, Any] = {}
    if path is not None:
        file = Path(path).expanduser()
        if not file.is_file():
            raise FileNotFoundError(f"Config file not found: {file}")
        with file.open("rb") as handle:
            data = tomllib.load(handle)
    if overrides:
        data = _deep_merge(data, overrides)
    return MelampusConfig.model_validate(data)
