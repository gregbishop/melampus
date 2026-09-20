"""Enrich identification results with the fields the Lightroom plugin needs.

The raw run output is per frame and knows nothing about bursts or geography. The
plugin's write gates need both, because neither confidence alone nor a species
name alone is enough to justify writing a keyword into a catalog:

* `burst_agreement` — how consistently the model called this subject across every
  frame of its encounter. The stronger of the two gates by a wide margin.
* `range_flag` — whether the top candidate actually occurs near the shoot
  location in that month, via GBIF.
* `encounter` — so a correction made on one frame can be traced to its burst.
* `quality`, `quality_rank`, `encounter_frames` — technical quality, and where
  the frame ranks within its burst, which is what the star rating is set from.

This began as tools/make_plugin_results.py, a second Python step the shipped
executable did not carry (card #436). It is now the `--plugin-out` half of
`melampus-id`; the tool is a thin caller.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, TextIO

from .config import MelampusConfig
from .encounters import Encounter, cluster
from .occurrence import GBIFClient, Location, applies_to
from .quality import QualityResult, analyze_quality
from .report import taxon_key

#: What MelampusImport.lua reads beyond the raw result.
PLUGIN_FIELDS = ("quality", "quality_rank", "burst_agreement", "range_flag",
                 "encounter", "encounter_frames")

#: Seconds between frames that separate one encounter from the next. Bursts of
#: one subject are far tighter than this; a walk to the next bird is far longer.
DEFAULT_GAP_SECONDS = 10.0

Scorer = Callable[[Path, MelampusConfig], QualityResult]


@dataclass
class Enrichment:
    rows: list[dict] = field(default_factory=list)
    flagged_encounters: int = 0
    quality_scores: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  records          : {len(self.rows)}",
            f"  with agreement   : {sum(1 for r in self.rows if 'burst_agreement' in r)}",
            f"  range-flagged    : {self.flagged_encounters} encounters",
        ]
        if self.quality_scores:
            vals = sorted(self.quality_scores.values())
            lines.append(f"  quality scored   : {len(vals)} (median {vals[len(vals) // 2]:.0f})")
        return "\n".join(lines)


def progress_printer(stream: TextIO) -> Callable[[int, int], None]:
    """An `on_progress` for `enrich` that narrates quality scoring to `stream`."""

    def progress(done: int, total: int) -> None:
        if done == 1:
            print(f"scoring quality for {total} frames ...", file=stream)
        if done % 250 == 0:
            print(f"  {done}/{total}", file=stream)

    return progress


def _top(rec: dict) -> dict | None:
    ident = rec.get("identification") or {}
    cands = sorted(ident.get("candidates", []), key=lambda c: -c.get("confidence", 0))
    return cands[0] if cands else None


def _score_quality(
    frames: list[Path], config: MelampusConfig, score: Scorer,
    on_progress: Callable[[int, int], None] | None,
) -> dict[str, float]:
    """Composite quality per frame name, rounded; frames that cannot be scored are absent."""
    quality_scores: dict[str, float] = {}
    for index, frame in enumerate(frames, 1):
        result = score(frame, config)
        if result.error is None:
            quality_scores[frame.name] = round(result.composite, 1)
        if on_progress is not None:
            on_progress(index, len(frames))
    return quality_scores


def _agreement(members: list[dict]) -> float | None:
    """Share of the encounter's voting frames that name the majority subject.

    An abstained frame casts no vote; an encounter where nothing voted has no
    agreement (None), and the row carries no `burst_agreement` at all, which
    MelampusRules.lua treats as absent, not as failure.
    """
    names: list[str] = []
    for rec in members:
        ident = rec.get("identification") or {}
        top = _top(rec)
        if not ident.get("abstain") and top is not None:
            names.append(taxon_key(top.get("common_name")))
    if not names:
        return None
    return Counter(names).most_common(1)[0][1] / len(names)


def _range_flag(client: GBIFClient, location: Location, enc: Encounter, members: list[dict]) -> bool:
    """Whether GBIF says the encounter's subject does not occur here in this month.

    One lookup per encounter, not per frame: every frame in an encounter is the
    same subject, and the cache would collapse them anyway. A failed lookup
    (None) is not absence.
    """
    first_ident = members[0].get("identification") or {}
    if not applies_to(first_ident.get("taxon")):
        return False
    month = enc.start.month if enc.start else None
    for rec in members:
        top = _top(rec)
        if top is None:
            continue
        sci = (top.get("scientific_name") or "").strip()
        if not sci:
            return False
        count = client.count(sci, location, month)
        return count is not None and count == 0
    return False


def _quality_ranks(members: list[dict], quality_scores: dict[str, float]) -> dict[str, float]:
    """Percentile rank of each scored frame WITHIN the encounter, 0.0 worst .. 1.0 best.

    Absolute sharpness is not comparable across subjects — a smooth white egret
    has far less texture to resolve than a patterned heron at identical focus
    accuracy — and culling is a within-burst question anyway: which frame of
    these forty is the keeper. Ranking answers that; an absolute score does not.
    """
    scored = sorted(
        [(rec["file"], quality_scores[rec["file"]]) for rec in members
         if rec["file"] in quality_scores],
        key=lambda m: m[1],
    )
    return {name: position / max(len(scored) - 1, 1) for position, (name, _) in enumerate(scored)}


def enrich(
    frames: Iterable[Path],
    records: Iterable[dict],
    config: MelampusConfig,
    *,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    lookup: tuple[GBIFClient, Location] | None = None,
    score: Scorer | None = analyze_quality,
    on_progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> Enrichment:
    """Add the plugin's fields to every record that has a frame in `frames`.

    `records` are raw results as the cache exports them (dicts, keyed here by
    `file`). `lookup` is the GBIF client and place from `occurrence.range_lookup`,
    or None to skip range checks. `score` is the quality scorer, or None to skip
    quality. Rows come back in encounter order, each carrying its raw record.

    A frame that cannot be read — gone since it was listed, or unreadable — is
    logged and left out, as `run_batch` does: one bad file never aborts the pass.
    """
    by_file = {r["file"]: r for r in records}
    readable: list[Path] = []
    for frame in sorted(frames):
        try:
            with frame.open("rb"):
                pass
        except OSError as exc:
            log(f"{frame.name}: unreadable ({exc}); left out of the enrichment")
            continue
        readable.append(frame)
    encounters = cluster(readable, gap_seconds)
    client, location = lookup if lookup is not None else (None, None)

    quality_scores: dict[str, float] = {}
    if score is not None:
        scorable = [f for e in encounters for f in e.frames if f.name in by_file]
        quality_scores = _score_quality(scorable, config, score, on_progress)

    out = Enrichment(quality_scores=quality_scores)

    for enc in encounters:
        members = [by_file[f.name] for f in enc.frames if f.name in by_file]
        if not members:
            continue
        agreement = _agreement(members)
        range_flag = client is not None and _range_flag(client, location, enc, members)
        if range_flag:
            out.flagged_encounters += 1
        rank_of = _quality_ranks(members, quality_scores)

        for rec in members:
            row = dict(rec)
            row["encounter"] = enc.index
            if rec["file"] in rank_of:
                row["quality_rank"] = round(rank_of[rec["file"]], 3)
                row["encounter_frames"] = len(rank_of)
            if agreement is not None:
                row["burst_agreement"] = round(agreement, 3)
            row["range_flag"] = range_flag
            if rec["file"] in quality_scores:
                row["quality"] = quality_scores[rec["file"]]
            out.rows.append(row)

    if client is not None and client.cache is not None:
        client.cache.flush()
    return out


def write_plugin_results(destination: Path, rows: list[dict]) -> None:
    """The file MelampusImport.lua reads. One shape, whoever writes it."""
    Path(destination).write_text(json.dumps(rows, indent=1), encoding="utf-8")
