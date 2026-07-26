"""Reporting: the raw per-image table, and scoring against a reference label set.

The headline accuracy number is MACRO-averaged (per-species, then averaged) because a
wildlife corpus is heavily class-imbalanced — overall accuracy would mostly measure the
photographer's commonest subject. Overall is reported too, but secondary.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .schema import ImageResult


def _norm(name: str | None) -> str:
    return (name or "").strip().lower()


def raw_table(results: list[ImageResult], max_candidates: int = 3) -> str:
    """Every photo, every candidate, every confidence. No summarising."""
    lines: list[str] = []
    header = f"{'file':<18} {'rank':>4} {'conf':>5}  {'taxon':<9} {'candidate':<30} {'sci name':<28} reasoning"
    lines.append(header)
    lines.append("-" * len(header))

    for record in sorted(results, key=lambda r: r.file):
        ident = record.identification
        if record.status != "ok" or ident is None:
            lines.append(
                f"{record.file:<18} {'-':>4} {'-':>5}  {'-':<9} "
                f"{'<' + record.status.upper() + '>':<30} {'':<28} {record.error or ''}"
            )
            continue

        if ident.abstain or not ident.candidates:
            reason = ident.abstain_reason or "no candidates returned"
            lines.append(
                f"{record.file:<18} {'-':>4} {'-':>5}  {ident.taxon.value:<9} "
                f"{'<ABSTAIN>':<30} {'':<28} {reason[:80]}"
            )
            continue

        for rank, cand in enumerate(ident.ranked()[:max_candidates], start=1):
            name = f"{record.file:<18}" if rank == 1 else " " * 18
            lines.append(
                f"{name} {rank:>4} {cand.confidence:>5.2f}  {ident.taxon.value:<9} "
                f"{cand.common_name[:30]:<30} {cand.scientific_name[:28]:<28} "
                f"{cand.reasoning[:100]}"
            )
    return "\n".join(lines)


@dataclass
class Scores:
    n_scored: int = 0
    top1: int = 0
    top3: int = 0
    abstained: int = 0
    unprocessed: int = 0
    per_species: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    per_taxon: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    confusions: Counter = field(default_factory=Counter)
    abstain_expected_met: int = 0
    abstain_expected_total: int = 0


def score(results: list[ImageResult], labels_path: Path) -> tuple[Scores, str]:
    payload = json.loads(Path(labels_path).read_text("utf-8"))
    labels = {row["file"]: row for row in payload["labels"]}
    by_file = {r.file: r for r in results}

    s = Scores()
    for file, label in sorted(labels.items()):
        record = by_file.get(file)
        if record is None:
            continue
        expected = label.get("expected_outcome", "identify")
        truth = _norm(label.get("common_name"))
        ident = record.identification

        if record.status != "ok" or ident is None:
            s.unprocessed += 1
            continue

        predicted_names = [_norm(c.common_name) for c in ident.ranked()]
        abstained = ident.abstain or not predicted_names
        if abstained:
            s.abstained += 1

        if expected in {"abstain", "none"}:
            s.abstain_expected_total += 1
            correct_behaviour = abstained or (
                expected == "none" and ident.taxon.value == "none"
            )
            if correct_behaviour:
                s.abstain_expected_met += 1
            continue

        # Only species-level rows with a real reference name are scored for accuracy.
        if not truth:
            continue
        s.n_scored += 1
        s.per_species[truth][1] += 1
        s.per_taxon[label.get("taxon", "unknown")][1] += 1

        if predicted_names and predicted_names[0] == truth:
            s.top1 += 1
            s.per_species[truth][0] += 1
            s.per_taxon[label.get("taxon", "unknown")][0] += 1
        elif predicted_names:
            s.confusions[(truth, predicted_names[0])] += 1
        if truth in predicted_names[:3]:
            s.top3 += 1

    return s, _render_scores(s)


def _render_scores(s: Scores) -> str:
    lines: list[str] = []
    macro_values = [hits / total for hits, total in s.per_species.values() if total]
    macro = sum(macro_values) / len(macro_values) if macro_values else 0.0
    overall = s.top1 / s.n_scored if s.n_scored else 0.0
    top3 = s.top3 / s.n_scored if s.n_scored else 0.0

    lines.append("SCORING (reference set = independent visual IDs, not ground truth)")
    lines.append("=" * 72)
    lines.append(f"macro-averaged top-1 (headline) : {macro:6.1%}   over {len(s.per_species)} species")
    lines.append(f"overall top-1                   : {overall:6.1%}   ({s.top1}/{s.n_scored})")
    lines.append(f"overall top-3                   : {top3:6.1%}   ({s.top3}/{s.n_scored})")
    lines.append(f"abstained (any reason)          : {s.abstained}")
    lines.append(f"unprocessed / failed            : {s.unprocessed}")
    if s.abstain_expected_total:
        lines.append(
            f"correct on abstain/none cases   : {s.abstain_expected_met}/{s.abstain_expected_total}"
        )

    lines.append("")
    lines.append("PER-SPECIES")
    lines.append(f"{'species':<32} {'n':>4} {'top-1':>7}")
    lines.append("-" * 46)
    for name in sorted(s.per_species, key=lambda k: -s.per_species[k][1]):
        hits, total = s.per_species[name]
        lines.append(f"{name[:32]:<32} {total:>4} {hits / total:>7.1%}")

    lines.append("")
    lines.append("PER-TAXON")
    lines.append(f"{'taxon':<32} {'n':>4} {'top-1':>7}")
    lines.append("-" * 46)
    for name in sorted(s.per_taxon, key=lambda k: -s.per_taxon[k][1]):
        hits, total = s.per_taxon[name]
        lines.append(f"{name[:32]:<32} {total:>4} {hits / total:>7.1%}")

    lines.append("")
    lines.append("CONFUSIONS (reference -> predicted, most frequent first)")
    lines.append("-" * 72)
    if not s.confusions:
        lines.append("(none)")
    for (truth, predicted), count in s.confusions.most_common(25):
        lines.append(f"{count:>4}x  {truth[:30]:<30} -> {predicted[:30]}")
    return "\n".join(lines)
