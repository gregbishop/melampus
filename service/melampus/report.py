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
    """Comparison key for a taxon name.

    Hyphenation and spacing of English bird names is genuinely inconsistent between
    authorities — "Tricolored Heron" / "Tri-colored Heron", "Night-Heron" /
    "Night Heron" — so comparing raw strings manufactures confusion pairs that are
    really one species. Every separator is dropped rather than normalised to a space,
    because the variants differ in both directions. Use _display for anything a human
    reads; this output is deliberately unreadable.
    """
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _display(name: str | None) -> str:
    """Readable lowercase form, for the confusion report and per-species rows."""
    return " ".join((name or "").strip().lower().split())


def _matches(predicted_common: str, predicted_sci: str, ref_common: str, ref_sci: str) -> bool:
    """A prediction is correct if either name agrees.

    Scientific names are the more reliable signal: a model can return an unusual
    common name while still having the taxon right. Requiring the common name alone
    would score those as errors.
    """
    if ref_common and predicted_common and predicted_common == ref_common:
        return True
    return bool(ref_sci and predicted_sci and predicted_sci == ref_sci)


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
    n_results: int = 0          # images that produced any result at all
    n_labelled: int = 0         # reference rows matched to a result
    n_scored: int = 0           # species-level rows eligible for accuracy
    top1: int = 0
    top3: int = 0
    abstained: int = 0
    unprocessed: int = 0
    # Accuracy restricted to rows where the model actually committed to an answer.
    # Reported separately because abstaining is desirable behaviour, not an error,
    # and folding abstentions into one accuracy number hides which is happening.
    n_committed: int = 0
    top1_committed: int = 0
    per_species: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    per_taxon: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    confusions: Counter = field(default_factory=Counter)
    abstain_expected_met: int = 0
    abstain_expected_total: int = 0
    # Top-1 confidence bucket -> [hits, total]. Feeds the §4.4 band thresholds:
    # bands are only defensible if confidence tracks correctness at all.
    calibration: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))


def _bucket(confidence: float) -> str:
    edges = [(0.9, "0.90-1.00"), (0.8, "0.80-0.89"), (0.7, "0.70-0.79"), (0.5, "0.50-0.69")]
    for low, label in edges:
        if confidence >= low:
            return label
    return "<0.50"


def score(results: list[ImageResult], labels_path: Path) -> tuple[Scores, str]:
    payload = json.loads(Path(labels_path).read_text("utf-8"))
    labels = {row["file"]: row for row in payload["labels"]}
    by_file = {r.file: r for r in results}

    s = Scores()
    s.n_results = len(results)
    for file, label in sorted(labels.items()):
        record = by_file.get(file)
        if record is None:
            continue
        s.n_labelled += 1
        expected = label.get("expected_outcome", "identify")
        truth = _norm(label.get("common_name"))
        truth_label = _display(label.get("common_name"))
        ident = record.identification

        if record.status != "ok" or ident is None:
            s.unprocessed += 1
            continue

        ranked = ident.ranked()
        predicted_names = [_norm(c.common_name) for c in ranked]
        predicted_sci = [_norm(c.scientific_name) for c in ranked]
        ref_sci = _norm(label.get("scientific_name"))
        abstained = ident.abstain or not ranked
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
        s.per_species[truth_label][1] += 1
        s.per_taxon[label.get("taxon", "unknown")][1] += 1

        hits = [
            _matches(common, sci, truth, ref_sci)
            for common, sci in zip(predicted_names, predicted_sci, strict=True)
        ]
        if hits and hits[0]:
            s.top1 += 1
            s.per_species[truth_label][0] += 1
            s.per_taxon[label.get("taxon", "unknown")][0] += 1
        elif ranked:
            s.confusions[(truth_label, _display(ranked[0].common_name))] += 1
        if any(hits[:3]):
            s.top3 += 1

        if not abstained and ranked:
            s.n_committed += 1
            bucket = _bucket(ranked[0].confidence)
            s.calibration[bucket][1] += 1
            if hits[0]:
                s.top1_committed += 1
                s.calibration[bucket][0] += 1

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
    committed = s.top1_committed / s.n_committed if s.n_committed else 0.0
    lines.append(
        f"top-1 when NOT abstaining       : {committed:6.1%}   ({s.top1_committed}/{s.n_committed})"
    )
    rate = s.abstained / s.n_labelled if s.n_labelled else 0.0
    lines.append(f"abstention rate                 : {rate:6.1%}   ({s.abstained}/{s.n_labelled})")
    lines.append(f"unprocessed / failed            : {s.unprocessed}")
    if s.abstain_expected_total:
        lines.append(
            f"correct on abstain/none cases   : {s.abstain_expected_met}/{s.abstain_expected_total}"
        )
    lines.append(f"coverage                        : {s.n_labelled} labelled of {s.n_results} results")

    lines.append("")
    lines.append("CONFIDENCE CALIBRATION (top-1 confidence vs whether it was right)")
    lines.append(f"{'bucket':<12} {'n':>5} {'correct':>9}")
    lines.append("-" * 30)
    for bucket in ("0.90-1.00", "0.80-0.89", "0.70-0.79", "0.50-0.69", "<0.50"):
        hits, total = s.calibration.get(bucket, [0, 0])
        if total:
            lines.append(f"{bucket:<12} {total:>5} {hits / total:>9.1%}")

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


def name_quality(results: list[ImageResult]) -> str:
    """Flag suspect scientific names before anything is written to a catalog.

    Two failure modes seen in practice, both of which would pollute a catalog and
    both of which scoring alone will not catch:

    * malformed binomials — not "Genus species", so almost certainly invented;
    * one common name mapped to several different binomials across the corpus,
      which means at least one is wrong even without consulting an authority.

    This is a local consistency check, not taxonomic validation. Checking against
    GBIF is Stage 2 work, once network access is on the table.
    """
    from collections import defaultdict

    malformed: Counter = Counter()
    by_common: dict[str, Counter] = defaultdict(Counter)
    missing = 0
    total = 0

    for record in results:
        ident = record.identification
        if record.status != "ok" or ident is None:
            continue
        for cand in ident.candidates:
            total += 1
            sci = (cand.scientific_name or "").strip()
            if not sci:
                missing += 1
                continue
            parts = sci.split()
            if len(parts) != 2 or not parts[0][:1].isupper() or not parts[1][:1].islower():
                malformed[sci] += 1
            by_common[_display(cand.common_name)][sci] += 1

    inconsistent = {
        common: counts for common, counts in by_common.items() if len(counts) > 1
    }

    lines = ["SCIENTIFIC NAME QUALITY (local consistency, not taxonomic validation)",
             "=" * 72,
             f"candidate names examined     : {total}",
             f"missing scientific name      : {missing}",
             f"malformed binomial           : {sum(malformed.values())}",
             f"common names with >1 binomial: {len(inconsistent)}"]

    if malformed:
        lines += ["", "MALFORMED", "-" * 72]
        lines += [f"{n:>4}x  {name[:60]}" for name, n in malformed.most_common(10)]

    if inconsistent:
        lines += ["", "ONE COMMON NAME, SEVERAL BINOMIALS (at least one is wrong)", "-" * 72]
        for common, counts in sorted(inconsistent.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
            rendered = ", ".join(f"{s} x{n}" for s, n in counts.most_common())
            lines.append(f"  {common[:28]:<28} -> {rendered[:80]}")

    return "\n".join(lines)
