"""Enrich identification results with the fields the Lightroom plugin needs.

    python tools/make_plugin_results.py fixtures_full stage1_full_results.json \\
        plugin_results.json --occurrence --quality

A thin caller of `melampus.plugin_results`, which is what `melampus-id
--plugin-out` runs inside the executable (card #436). This entry point stays
because the Lightroom plugin still invokes it until card #401 rewires the
plugin; the enrichment itself lives in the service package, so the two write
the same bytes from the same inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from melampus.config import load_config  # noqa: E402
from melampus.occurrence import range_lookup  # noqa: E402
from melampus.plugin_results import (  # noqa: E402
    DEFAULT_GAP_SECONDS, enrich, progress_printer, write_plugin_results,
)
from melampus.quality import analyze_quality  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gap", type=float, default=DEFAULT_GAP_SECONDS)
    ap.add_argument("--occurrence", action="store_true",
                    help="check candidates against GBIF (needs network)")
    ap.add_argument("--quality", action="store_true",
                    help="score technical quality so the plugin can set star ratings")
    args = ap.parse_args(argv[1:])

    config = load_config()
    records = json.loads(args.results.read_text("utf-8"))
    frames = sorted(args.folder.glob("*.jpg"))

    lookup = None
    if args.occurrence:
        lookup = range_lookup(config.occurrence)
        if lookup is None:
            print("no default location configured; skipping range checks", file=sys.stderr)

    outcome = enrich(
        frames, records, config, gap_seconds=args.gap, lookup=lookup,
        score=analyze_quality if args.quality else None,
        on_progress=progress_printer(sys.stderr),
    )
    write_plugin_results(args.out, outcome.rows)
    print(f"wrote {args.out}\n{outcome.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
