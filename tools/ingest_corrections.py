"""Turn a reviewed corrections file into a label set covering the whole corpus.

Each encounter verdict is propagated to every frame in that encounter, because all
frames in an encounter are the same individual. Reviewing 43 encounters therefore
labels ~1,743 photographs.

    python tools/ingest_corrections.py fixtures_full melampus_corrections.json \\
        fixtures_full_labels.json

Then score against it:

    melampus-id fixtures_full --report-only --labels fixtures_full_labels.json

A caution about what these labels are. Confirming a call you were shown is not the
same as identifying the bird cold — anchoring is real, and "correct" verdicts will
skew optimistic. The "wrong" corrections are the trustworthy part, and they are what
feeds prompt tuning and the CLAUDE.md §6.4 correction dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_encounters import cluster  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    folder = Path(argv[1])
    corrections_path = Path(argv[2])
    out_path = Path(argv[3]) if len(argv) > 3 else Path("fixtures_full_labels.json")
    gap = float(argv[4]) if len(argv) > 4 else 10.0

    payload = json.loads(corrections_path.read_text("utf-8"))
    verdicts = {int(c["encounter"]): c for c in payload["corrections"]}
    encounters = {e.index: e for e in cluster(sorted(folder.glob("*.jpg")), gap)}

    labels: list[dict] = []
    stats = {"correct": 0, "wrong": 0, "unsure": 0, "frames": 0, "skipped_encounters": 0}

    for index, verdict in sorted(verdicts.items()):
        enc = encounters.get(index)
        if enc is None:
            stats["skipped_encounters"] += 1
            continue
        kind = verdict["verdict"]
        stats[kind] = stats.get(kind, 0) + 1

        if kind == "unsure":
            name, expected = None, "abstain"
        elif kind == "wrong":
            name = (verdict.get("corrected_to") or "").strip()
            if not name:
                stats["skipped_encounters"] += 1
                continue
            expected = "identify"
        else:
            name, expected = verdict["model_call"], "identify"

        for frame in enc.frames:
            labels.append({
                "file": frame.name,
                "taxon": "unknown",
                "common_name": name,
                "scientific_name": None,
                "reference_confidence": "human-reviewed",
                "expected_outcome": expected,
                "source_encounter": index,
                "review_verdict": kind,
            })
            stats["frames"] += 1

    out_path.write_text(json.dumps({
        "_about": {
            "source": "Human review of one representative frame per shooting encounter, "
                      "propagated to every frame in that encounter.",
            "caveat": "Reviewers saw the model's answer before judging, so 'correct' "
                      "verdicts carry anchoring bias and will read optimistic. The "
                      "corrections are the reliable signal.",
        },
        "labels": labels,
    }, indent=1), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  encounters reviewed : {stats['correct'] + stats['wrong'] + stats['unsure']}")
    print(f"    confirmed correct : {stats['correct']}")
    print(f"    corrected         : {stats['wrong']}")
    print(f"    can't tell        : {stats['unsure']}")
    print(f"  frames labelled     : {stats['frames']}")
    if stats["skipped_encounters"]:
        print(f"  skipped             : {stats['skipped_encounters']} (unknown or blank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
