"""Group frames into shooting encounters by capture-time proximity.

Wildlife shooting produces long bursts of one subject. Clustering on EXIF capture
time recovers those encounters, which gives us three things:

  1. A tractable labelling unit -- one identification per encounter instead of one
     per frame.
  2. A principled way to pick a diverse development subset, by sampling across
     encounters rather than at random (random sampling from 1,743 burst frames
     would return many near-duplicates of the same few subjects).
  3. The unit the Lightroom plugin's write gates reason over: burst agreement
     and range flags are per encounter (plugin_results.py), which is why this
     lives in the service package rather than in tools/ (card #436).

Reads capture time from the XMP packet. No pixels are needed, and nothing here
ever reaches the model. `tools/cluster_encounters.py` is the corpus report over
this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_CREATE_DATE = re.compile(rb'xmp:CreateDate="([^"]+)"')


def capture_time(path: Path) -> datetime | None:
    """Parse xmp:CreateDate. Returns None if absent or unparseable."""
    raw = path.read_bytes()
    match = _CREATE_DATE.search(raw)
    if not match:
        return None
    text = match.group(1).decode("ascii", "replace")
    # Lightroom writes fractional seconds and a numeric offset; fromisoformat on
    # 3.12 handles both, but tolerate odd variants rather than crash a scan.
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text.split(".")[0])
        except ValueError:
            return None


@dataclass
class Encounter:
    index: int
    frames: list[Path] = field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None

    @property
    def size(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return (self.end - self.start).total_seconds()

    def representative(self) -> Path:
        """Middle frame -- past the ragged start of a burst, before it trails off."""
        return self.frames[len(self.frames) // 2]


def cluster(paths: list[Path], gap_seconds: float) -> list[Encounter]:
    timed = [(capture_time(p), p) for p in paths]
    known = sorted((t, p) for t, p in timed if t is not None)
    unknown = [p for t, p in timed if t is None]

    encounters: list[Encounter] = []
    current: Encounter | None = None
    previous: datetime | None = None

    for stamp, path in known:
        if current is None or (stamp - previous).total_seconds() > gap_seconds:
            current = Encounter(index=len(encounters), start=stamp)
            encounters.append(current)
        current.frames.append(path)
        current.end = stamp
        previous = stamp

    if unknown:
        orphan = Encounter(index=len(encounters))
        orphan.frames = unknown
        encounters.append(orphan)

    return encounters
