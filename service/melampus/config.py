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
    # Long edge sent to the model. ~1600px is ample for ID per CLAUDE.md §5.1 and
    # keeps vision-token cost (and therefore runtime) down.
    max_edge: int = 1600
    jpeg_quality: int = 92


class RunConfig(_Base):
    prompts_dir: Path = REPO_ROOT / "prompts"
    cache_path: Path = REPO_ROOT / ".melampus_cache" / "identifications.jsonl"
    # One corrective retry on schema-validation failure, per CLAUDE.md §4.2.
    max_retries: int = 1


class MelampusConfig(_Base):
    model: ModelConfig = Field(default_factory=ModelConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
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
