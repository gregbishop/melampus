"""Batch execution: resumable, checkpointed, and tolerant of individual bad files."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .backend import CommandFailed
from .cache import ResultCache
from .identify import Identifier
from .images import content_hash
from .schema import ImageResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass
class BatchStats:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    ok: int = 0
    unprocessed: int = 0
    errors: int = 0
    seconds: float = 0.0
    per_image_seconds: list[float] = field(default_factory=list)

    def note(self, result: ImageResult) -> None:
        self.processed += 1
        self.seconds += result.seconds
        self.per_image_seconds.append(result.seconds)
        if result.status == "ok":
            self.ok += 1
        elif result.status == "unprocessed":
            self.unprocessed += 1
        else:
            self.errors += 1

    @property
    def mean_seconds(self) -> float:
        return self.seconds / len(self.per_image_seconds) if self.per_image_seconds else 0.0

    @property
    def median_seconds(self) -> float:
        if not self.per_image_seconds:
            return 0.0
        ordered = sorted(self.per_image_seconds)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2


def run_batch(
    paths: Sequence[Path],
    identifier: Identifier,
    cache: ResultCache,
    *,
    force: bool = False,
    limit: int | None = None,
    on_result: Callable[[ImageResult, BatchStats], None] | None = None,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> BatchStats:
    """Process images, skipping anything already completed unless forced.

    A failure on one file is logged and the batch continues — never abort a
    multi-thousand-image run over a single unreadable frame.
    """
    selected = list(paths)[: limit if limit is not None else len(paths)]
    stats = BatchStats(total=len(selected))
    started = time.perf_counter()

    for index, path in enumerate(selected, start=1):
        try:
            digest = content_hash(path)
        except OSError as exc:
            log(f"[{index}/{len(selected)}] {path.name}: unreadable ({exc})")
            stats.errors += 1
            continue

        if not force and cache.has_success(digest, identifier.fingerprint):
            stats.skipped += 1
            continue

        try:
            result = identifier.identify(path)
        except CommandFailed:
            # The engine, not the file: nothing to record, the batch stops.
            raise
        except Exception as exc:  # noqa: BLE001 - batch resilience is the whole point
            result = ImageResult(
                file=path.name, content_hash=digest, status="error",
                error=f"{type(exc).__name__}: {exc}", model=identifier.backend.name,
            )
            log(f"[{index}/{len(selected)}] {path.name}: {result.error}")

        cache.put(result)  # checkpoint before anything else can fail
        stats.note(result)
        if on_result is not None:
            on_result(result, stats)

    stats.seconds = time.perf_counter() - started
    return stats


def stratify_by_prediction(
    paths: Iterable[Path], cache: ResultCache, per_species: int
) -> list[Path]:
    """Sample up to N images per previously-predicted species.

    Requires a prior pass: with no labels and no predictions there is nothing to
    stratify on. Use stratify_by_encounter for a cold corpus.
    """
    buckets: dict[str, list[Path]] = {}
    for path in paths:
        try:
            record = cache.get(content_hash(path))
        except OSError:
            continue
        if record is None or record.identification is None:
            key = "__unprocessed__"
        else:
            top = record.identification.top()
            key = top.common_name.lower() if top else "__abstained__"
        buckets.setdefault(key, []).append(path)
    chosen: list[Path] = []
    for key in sorted(buckets):
        chosen.extend(buckets[key][:per_species])
    return sorted(chosen)
