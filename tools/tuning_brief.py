"""Turn your corrections into a prompt-tuning brief.

The confusion report says *which* discriminations the model gets wrong. This turns
that into an ordered worklist: which pairs cost the most, whether occurrence data
can settle each one for free, and which therefore need words added to a prompt.

That distinction is the useful part. A confusion between two species that both
occur here is a vision problem and belongs in prompts/. A confusion where one of
them does not occur here is a range problem that §4.3 already fixes, and writing
prompt text for it wastes effort.

    python tools/tuning_brief.py fixtures_full_labels.json stage1_full_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def norm(name: str | None) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def display(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path, help="from tools/ingest_corrections.py")
    ap.add_argument("results", type=Path)
    ap.add_argument("--check-range", action="store_true",
                    help="ask GBIF whether each confused species occurs here")
    args = ap.parse_args(argv[1:])

    labels = {row["file"]: row for row in json.loads(args.labels.read_text())["labels"]}
    results = {r["file"]: r for r in json.loads(args.results.read_text())}

    confusions: Counter = Counter()
    correct = Counter()
    scientific: dict[str, str] = {}

    for file, label in labels.items():
        truth = label.get("common_name")
        if not truth:
            continue
        record = results.get(file)
        if record is None or record.get("status") != "ok":
            continue
        ident = record.get("identification") or {}
        cands = sorted(ident.get("candidates", []), key=lambda c: -c.get("confidence", 0))
        if ident.get("abstain") or not cands:
            continue
        predicted = cands[0].get("common_name")
        scientific[norm(predicted)] = cands[0].get("scientific_name") or ""
        if norm(predicted) == norm(truth):
            correct[display(truth)] += 1
        else:
            confusions[(display(truth), display(predicted))] += 1

    if not confusions:
        print("No confusions found. Either the model agreed with every correction,"
              " or there are no corrections yet.")
        return 0

    ranges: dict[str, int | None] = {}
    if args.check_range:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
        from melampus.config import load_config
        from melampus.occurrence import GBIFClient, Location, OccurrenceCache

        cfg = load_config().occurrence
        client = GBIFClient(cache=OccurrenceCache(cfg.cache_path))
        where = Location(cfg.default_latitude, cfg.default_longitude, cfg.radius_km)
        for (_, predicted) in confusions:
            sci = scientific.get(norm(predicted), "")
            if sci and predicted not in ranges:
                ranges[predicted] = client.count(sci, where, None)
        client.cache.flush()

    print("PROMPT TUNING BRIEF")
    print("=" * 74)
    print(f"{'n':>4}  {'you said':<26} {'model said':<26} action")
    print("-" * 74)

    prompt_work, range_work = [], []
    for (truth, predicted), n in confusions.most_common(25):
        records = ranges.get(predicted)
        if records == 0:
            action = "range data already fixes this"
            range_work.append((truth, predicted, n))
        else:
            action = "needs prompt text"
            prompt_work.append((truth, predicted, n))
        print(f"{n:>4}  {truth[:26]:<26} {predicted[:26]:<26} {action}")

    if prompt_work:
        print()
        print("WRITE THESE INTO prompts/bird.md")
        print("-" * 74)
        for truth, predicted, n in prompt_work[:8]:
            print(f"\n  {truth} vs {predicted}  ({n} frames)")
            print(f"    Add a line naming the single visible feature that separates them.")
            print(f"    Say what to look at, not which is more likely.")

    if range_work:
        print()
        print(f"{len(range_work)} confusion pair(s) need no prompt work — the species does")
        print("not occur here and occurrence re-ranking already demotes it.")

    total = sum(confusions.values()) + sum(correct.values())
    print()
    print(f"scored {total} frames, {sum(correct.values())} correct, "
          f"{sum(confusions.values())} confused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
