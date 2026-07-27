"""Enrich identification results with the fields the Lightroom plugin needs.

The raw run output is per frame and knows nothing about bursts or geography. The
plugin's write gates need both, because neither confidence alone nor a species
name alone is enough to justify writing a keyword into a catalog:

* `burst_agreement` — how consistently the model called this subject across every
  frame of its encounter. The stronger of the two gates by a wide margin.
* `range_flag` — whether the top candidate actually occurs near the shoot
  location in that month, via GBIF.
* `encounter` — so a correction made on one frame can be traced to its burst.

    python tools/make_plugin_results.py fixtures_full stage1_full_results.json \\
        plugin_results.json --occurrence
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_encounters import cluster  # noqa: E402


def norm(name: str | None) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gap", type=float, default=10.0)
    ap.add_argument("--occurrence", action="store_true",
                    help="check candidates against GBIF (needs network)")
    ap.add_argument("--quality", action="store_true",
                    help="score technical quality so the plugin can set star ratings")
    args = ap.parse_args(argv[1:])

    records = {r["file"]: r for r in json.loads(args.results.read_text("utf-8"))}
    encounters = cluster(sorted(args.folder.glob("*.jpg")), args.gap)

    client = None
    location = None
    if args.occurrence:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
        from melampus.config import load_config
        from melampus.occurrence import GBIFClient, Location, OccurrenceCache

        config = load_config()
        occ = config.occurrence
        if occ.default_latitude is None or occ.default_longitude is None:
            print("no default location configured; skipping range checks", file=sys.stderr)
        else:
            location = Location(occ.default_latitude, occ.default_longitude, occ.radius_km)
            client = GBIFClient(cache=OccurrenceCache(occ.cache_path))

    quality_scores: dict[str, float] = {}
    if args.quality:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
        from melampus.config import load_config as _load
        from melampus.quality import analyze_quality

        qcfg = _load()
        frames = sorted(args.folder.glob("*.jpg"))
        print(f"scoring quality for {len(frames)} frames ...", file=sys.stderr)
        for index, frame in enumerate(frames, 1):
            score = analyze_quality(frame, qcfg)
            if score.error is None:
                quality_scores[frame.name] = round(score.composite, 1)
            if index % 250 == 0:
                print(f"  {index}/{len(frames)}", file=sys.stderr)

    enriched: list[dict] = []
    flagged = 0

    for enc in encounters:
        names: list[str] = []
        members: list[dict] = []
        for frame in enc.frames:
            rec = records.get(frame.name)
            if rec is None:
                continue
            members.append(rec)
            ident = rec.get("identification") or {}
            cands = sorted(ident.get("candidates", []), key=lambda c: -c.get("confidence", 0))
            if not ident.get("abstain") and cands:
                names.append(norm(cands[0].get("common_name")))

        agreement = None
        if names:
            agreement = Counter(names).most_common(1)[0][1] / len(names)

        # One range lookup per encounter, not per frame: every frame in an
        # encounter is the same subject, and the cache would collapse them anyway.
        range_flag = False
        if client is not None and location is not None and members:
            month = enc.start.month if enc.start else None
            for rec in members:
                ident = rec.get("identification") or {}
                cands = sorted(ident.get("candidates", []), key=lambda c: -c.get("confidence", 0))
                if not cands:
                    continue
                sci = (cands[0].get("scientific_name") or "").strip()
                if not sci:
                    break
                count = client.count(sci, location, month)
                if count is not None and count == 0:
                    range_flag = True
                    flagged += 1
                break

        # Rank quality WITHIN the encounter. Absolute sharpness is not
        # comparable across subjects — a smooth white egret has far less texture
        # to resolve than a patterned heron at identical focus accuracy — and
        # culling is a within-burst question anyway: which frame of these forty
        # is the keeper. Ranking answers that; an absolute score does not.
        member_quality = [(rec["file"], quality_scores.get(rec["file"]))
                          for rec in members]
        scored = sorted([m for m in member_quality if m[1] is not None],
                        key=lambda m: m[1])
        rank_of: dict[str, float] = {}
        if scored:
            for position, (name, _) in enumerate(scored):
                # Percentile rank within the burst, 0.0 worst .. 1.0 best.
                rank_of[name] = position / max(len(scored) - 1, 1)

        for rec in members:
            row = dict(rec)
            row["encounter"] = enc.index
            if rec["file"] in rank_of:
                row["quality_rank"] = round(rank_of[rec["file"]], 3)
                row["encounter_frames"] = len(scored)
            if agreement is not None:
                row["burst_agreement"] = round(agreement, 3)
            row["range_flag"] = range_flag
            if rec["file"] in quality_scores:
                row["quality"] = quality_scores[rec["file"]]
            enriched.append(row)

    if client is not None and client.cache is not None:
        client.cache.flush()

    args.out.write_text(json.dumps(enriched, indent=1), encoding="utf-8")
    with_agreement = sum(1 for r in enriched if "burst_agreement" in r)
    print(f"wrote {args.out}")
    print(f"  records          : {len(enriched)}")
    print(f"  with agreement   : {with_agreement}")
    print(f"  range-flagged    : {flagged} encounters")
    if quality_scores:
        vals = sorted(quality_scores.values())
        print(f"  quality scored   : {len(quality_scores)} "
              f"(median {vals[len(vals)//2]:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
