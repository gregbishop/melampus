"""Encounter-aware analysis of identification results.

Two things the per-image report cannot tell you:

1. **Accuracy without pseudo-replication.** A 1,743-frame corpus is roughly 43
   shooting encounters. Scoring per frame mostly measures how long you held the
   shutter down, not identification skill: one subject photographed 216 times would
   dominate any average. Aggregating to one verdict per encounter fixes that.

2. **Stability, with no labels at all.** Every frame within an encounter is the same
   individual, so any disagreement between frames is unambiguous instability. That
   makes it a genuine quality signal on a corpus with no ground truth — which is
   exactly the situation here.

Read-only: safe to run against a batch that is still in progress.

    python tools/analyze_encounters.py fixtures_full stage1_full_results.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_encounters import cluster  # noqa: E402


def norm(name: str | None) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def display(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    folder, results_path = Path(argv[1]), Path(argv[2])
    gap = float(argv[3]) if len(argv) > 3 else 10.0

    records = {r["file"]: r for r in json.loads(results_path.read_text("utf-8"))}
    encounters = cluster(sorted(folder.glob("*.jpg")), gap)

    print(f"{'enc':>4} {'frames':>7} {'done':>5} {'abst':>5} {'agree':>6}  majority ID")
    print("-" * 78)

    total_frames = total_done = total_agree = total_abstain = 0
    encounter_ids: list[tuple[int, str, int, float]] = []

    for enc in encounters:
        names: list[str] = []
        abstains = 0
        done = 0
        for frame in enc.frames:
            rec = records.get(frame.name)
            if rec is None or rec.get("status") != "ok":
                continue
            done += 1
            ident = rec.get("identification") or {}
            cands = sorted(ident.get("candidates", []), key=lambda c: -c.get("confidence", 0))
            if ident.get("abstain") or not cands:
                abstains += 1
                continue
            names.append(display(cands[0].get("common_name")))

        if not done:
            continue
        counts = Counter(norm(n) for n in names)
        if counts:
            top_key, top_n = counts.most_common(1)[0]
            majority = next(n for n in names if norm(n) == top_key)
            agreement = top_n / len(names)
        else:
            majority, agreement, top_n = "<all abstained>", 1.0, 0

        distinct = len(counts)
        flag = "  <-- unstable" if distinct > 1 and agreement < 0.9 else ""
        print(
            f"{enc.index:>4} {enc.size:>7} {done:>5} {abstains:>5} {agreement:>6.0%}  "
            f"{majority[:34]:<34}{'' if distinct <= 1 else f'({distinct} distinct)'}{flag}"
        )

        total_frames += enc.size
        total_done += done
        total_abstain += abstains
        total_agree += top_n
        encounter_ids.append((enc.index, majority, done, agreement))

    named = [e for e in encounter_ids if not e[1].startswith("<")]
    print()
    print("SUMMARY")
    print("=" * 60)
    print(f"encounters with results   : {len(encounter_ids)} of {len(encounters)}")
    print(f"frames processed          : {total_done} of {total_frames}")
    if total_done:
        print(f"abstention rate (frames)  : {total_abstain / total_done:6.1%}")
    committed = total_done - total_abstain
    if committed:
        print(f"within-encounter agreement: {total_agree / committed:6.1%}   "
              "(same subject, so disagreement is pure instability)")
    unstable = [e for e in named if e[3] < 0.9]
    print(f"unstable encounters (<90%): {len(unstable)} of {len(named)}")
    print()
    print("DISTINCT SPECIES CALLED (one vote per encounter, not per frame)")
    print("-" * 60)
    for name, n in Counter(e[1] for e in named).most_common():
        print(f"  {n:>3} encounters  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
