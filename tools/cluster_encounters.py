"""Report the corpus's shooting encounters (see melampus.encounters).

    python tools/cluster_encounters.py fixtures_full --gap 120 --json encounters.json

The clustering itself lives in the service package, because the plugin's
enrichment pass (melampus.plugin_results) depends on it and that pass ships
inside the executable. The other corpus tools import `cluster` from here, which
re-exports it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from melampus.encounters import Encounter, capture_time, cluster  # noqa: E402,F401


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--gap", type=float, default=120.0, help="seconds between encounters")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv[1:])

    paths = sorted(args.source.glob("*.jpg"))
    if not paths:
        print(f"no JPEGs under {args.source}", file=sys.stderr)
        return 1

    encounters = cluster(paths, args.gap)
    sizes = sorted((e.size for e in encounters), reverse=True)
    print(f"frames     : {len(paths)}")
    print(f"gap        : {args.gap:.0f}s")
    print(f"encounters : {len(encounters)}")
    print(f"largest    : {sizes[:10]}")
    print(f"singletons : {sum(1 for s in sizes if s == 1)}")
    print()
    print(f"{'enc':>4} {'frames':>7} {'dur_s':>8}  {'start':<20} representative")
    for e in encounters:
        start = e.start.strftime("%Y-%m-%d %H:%M:%S") if e.start else "unknown"
        print(f"{e.index:>4} {e.size:>7} {e.duration_s:>8.0f}  {start:<20} {e.representative().name}")

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "index": e.index,
                        "size": e.size,
                        "start": e.start.isoformat() if e.start else None,
                        "duration_s": e.duration_s,
                        "representative": e.representative().name,
                        "frames": [p.name for p in e.frames],
                    }
                    for e in encounters
                ],
                indent=1,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
