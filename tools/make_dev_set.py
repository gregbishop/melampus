"""Split the corpus: full validation set aside, small development set to work against.

Selection samples evenly across time-ordered encounters (see cluster_encounters.py)
rather than uniformly at random. Random sampling from a corpus that is mostly burst
frames returns many near-duplicates of the same few subjects; sampling across
encounters maximises subject diversity for a fixed budget.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_encounters import cluster  # noqa: E402


def choose(encounters, n: int):
    """Evenly spaced picks across the time-ordered encounter list."""
    if n >= len(encounters):
        return list(encounters)
    step = len(encounters) / n
    return [encounters[int(i * step)] for i in range(n)]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("fixtures"))
    ap.add_argument("--full", type=Path, default=Path("fixtures_full"))
    ap.add_argument("--dev", type=Path, default=Path("fixtures"))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--gap", type=float, default=10.0)
    ap.add_argument("--manifest", type=Path, default=Path("fixtures_dev_manifest.json"))
    args = ap.parse_args(argv[1:])

    paths = sorted(args.source.glob("*.jpg"))
    if not paths:
        print(f"no JPEGs under {args.source}", file=sys.stderr)
        return 1

    encounters = cluster(paths, args.gap)
    picks = choose(encounters, args.n)
    chosen = {e.index: e.representative() for e in picks}

    # Move the whole corpus aside first, then copy the development picks back.
    args.full.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in paths:
        shutil.move(str(p), str(args.full / p.name))
        moved += 1

    args.dev.mkdir(parents=True, exist_ok=True)
    records = []
    for e in picks:
        name = chosen[e.index].name
        shutil.copy2(args.full / name, args.dev / name)
        # Deliberately no capture timestamp. The manifest is committed so the dev-set
        # selection is auditable, and nothing reads it back — but a list of
        # millisecond-precision capture times is a record of where the photographer
        # was and when, which is not something a public repository should carry.
        # Encounter index already orders these chronologically; frame count and
        # duration carry the analytically useful part.
        records.append(
            {
                "file": name,
                "encounter": e.index,
                "encounter_frames": e.size,
                "encounter_duration_s": round(e.duration_s, 1),
            }
        )

    args.manifest.write_text(json.dumps({"gap_seconds": args.gap,
                                         "encounters": len(encounters),
                                         "selected": records}, indent=1))

    print(f"moved to {args.full}/ : {moved} frames")
    print(f"encounters (gap {args.gap:.0f}s): {len(encounters)}")
    print(f"copied to {args.dev}/  : {len(records)} representatives")
    print(f"manifest             : {args.manifest}")
    print()
    # Capture times are shown here, on your own machine, because they are useful
    # while selecting. They are not written to the manifest — see above.
    print(f"{'file':<18} {'enc':>4} {'frames':>7}  start")
    for e in picks:
        start = e.start.isoformat()[:19].replace("T", " ") if e.start else "unknown"
        print(f"{chosen[e.index].name:<18} {e.index:>4} {e.size:>7}  {start}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
