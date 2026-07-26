"""Content-hash result cache with continuous checkpointing.

Append-only JSONL: each result is flushed and fsynced as soon as it is produced, so a
crash loses at most the image currently in flight. Keyed on file content rather than
path, so re-running after a rename, move or re-export is still a no-op.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import ImageResult


class ResultCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._by_hash: dict[str, ImageResult] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ImageResult.model_validate_json(line)
                except Exception:  # noqa: BLE001 - a truncated tail is expected after a crash
                    continue
                self._by_hash[record.content_hash] = record

    def __len__(self) -> int:
        return len(self._by_hash)

    def get(self, content_hash: str) -> ImageResult | None:
        return self._by_hash.get(content_hash)

    def has_success(self, content_hash: str) -> bool:
        record = self._by_hash.get(content_hash)
        return record is not None and record.status == "ok"

    def put(self, result: ImageResult) -> None:
        self._by_hash[result.content_hash] = result
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def results(self) -> list[ImageResult]:
        return list(self._by_hash.values())

    def export_json(self, destination: Path) -> None:
        payload = [json.loads(r.model_dump_json()) for r in self.results()]
        Path(destination).write_text(json.dumps(payload, indent=1), encoding="utf-8")
