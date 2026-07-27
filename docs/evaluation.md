# Evaluation methodology

What the accuracy numbers mean, what they don't, and which of them to trust.

This is the most important document in the project. Species identification is easy to
measure badly, and two measurement bugs have already produced numbers that looked like
model results. Both are described below, because knowing how a measurement failed is
more useful than the corrected figure.

---

## Headline results

Development set, 25 frames, `Qwen3-VL-30B-A3B-Instruct-4bit`:

| metric | value | |
|---|---|---|
| **macro-averaged top-1** | **79.4%** | headline; averaged per species |
| overall top-1 | 68.2% (15/22) | secondary |
| top-3 | 68.2% | identical to top-1 — see below |
| top-1 when not abstaining | 68.2% | |
| abstention rate | 8.0% | 11.0% on the wider corpus |
| majority-class baseline | 40.9% | always guess the commonest species |
| within-encounter agreement | 76.4% | label-free; preliminary |
| throughput | 6.8 s/frame | M4 Max |

**Read every one of these as provisional.** n = 22 scored frames. Three of the seven
species have n = 1. The 95% confidence interval on 68% at n = 22 is roughly ±20 points.

---

## Why macro-averaged is the headline

A wildlife corpus is heavily class-imbalanced. In the dev set, Tricolored Heron is 9 of
22 scored frames; four species appear once or twice. Overall accuracy on such a corpus
mostly measures performance on the photographer's commonest subject.

Macro-averaging computes accuracy per species and then averages those, so a rare
species counts as much as a common one. That is the number that predicts how the tool
behaves on the next species you photograph.

Overall accuracy is still reported, because macro-averaging over species with n = 1 is
itself noisy — a single frame flips that species between 0% and 100%.

---

## The pseudo-replication trap

**This is the single easiest way to produce a badly wrong number here.**

1,743 frames are not 1,743 independent samples. They are roughly **43 shooting
encounters**, and every frame within an encounter is the same individual photographed
repeatedly. One subject shot 216 times would dominate any per-frame average.

Per-frame accuracy therefore largely measures *how long you held the shutter down*, not
identification skill.

A live demonstration, from the same data and the same model:

| aggregation | result |
|---|---|
| per frame | **92.4%** |
| macro-averaged | **46.5%** |

The gap is entirely one large, correctly-identified encounter swamping the average.

**Always aggregate to one verdict per encounter before computing accuracy.**
`tools/analyze_encounters.py` does this.

---

## Within-encounter agreement: accuracy without labels

Every frame inside an encounter shows the same individual. So any disagreement between
frames is unambiguous **instability** — no reference labels required, and no judgement
call to dispute.

Measured: **76.4% agreement** (preliminary, first 200 frames). Roughly **one frame in
four of the same bird receives a different species call.**

This is arguably the most trustworthy number in the project, because it depends on no
reference set at all. It also reframes what the errors are: this is not a model with a
stable-but-wrong view of Tricolored Herons, it is a model whose answer moves from frame
to frame on the same subject.

**Design consequence.** If per-frame identifications wobble this much, writing a species
keyword per photo produces a catalog where one bird carries three species tags across a
single burst. Voting across an encounter before writing anything looks less like an
optimisation and more like a requirement — and it composes naturally with the burst
culling already specified in CLAUDE.md §6.1.

---

## The reference set is not ground truth

The corpus arrived with **no keywords and no GPS** — a full-byte scan of all 1,743 files
found zero `dc:subject`, zero `lr:hierarchicalSubject`, zero IPTC 2:25 and zero
coordinates. The Lightroom export had stripped them, and re-exporting was not available.

The reference labels in `fixtures_dev_labels.json` are therefore **independent visual
identifications made by Claude**, not verified ground truth. That has a specific and
uncomfortable failure mode:

> Both the reference and the system under test are vision-language models. Where they
> share a blind spot, the error is invisible and the score is **inflated**. Where the
> reference is wrong and the model is right, the model is penalised unfairly.

Rows carry a `reference_confidence` field; `low` rows should be discounted.

**The mitigation is human review**, via `review.html`. That converts anchored-but-real
human judgement into labels. Note the residual bias: a reviewer sees the model's answer
before judging, so *confirmations* skew optimistic. The **corrections** are the reliable
signal, and they are what feeds prompt tuning.

---

## Confidence is ordinal, not calibrated

| top-1 confidence | n | actually correct |
|---|---|---|
| 0.90 – 1.00 | 17 | **82.4%** |
| 0.80 – 0.89 | 3 | 33.3% |
| 0.70 – 0.79 | 2 | 0.0% |

Two conclusions, and they point in different directions:

- The model is **overconfident**: it claims 0.95 where it is right 82% of the time. The
  numbers must never be presented to a user as probabilities.
- The ranking nevertheless **carries real signal**. Accuracy collapses sharply below
  0.90, which makes confidence usable as an ordinal gate even though it is uncalibrated.

On this evidence a High band at ≥ 0.90, writing no species keyword below it, is
defensible for CLAUDE.md §4.4. The sample is small; confirm against the full corpus
before fixing thresholds.

---

## The runner-up candidate is currently decorative

Top-3 accuracy equals top-1 accuracy exactly. When the model's first answer is wrong,
its second is essentially never right.

The prompt asks for at least two candidates and gets two, but the second is a token
entry rather than a genuine competing hypothesis. CLAUDE.md §5.4.3 wants alternates
displayed so that correcting an ID is one gesture — that only works if the alternates
are real. The prompt should request three and explicitly demand the strongest
*competing* hypothesis.

---

## Out-of-range hallucination

The most damaging observed failure mode. Species offered for Florida wetland birds
have included **Long-tailed Cuckoo** (Asia), **Great Bowerbird** (Australia),
**White-faced Heron** (Australia), **Striped Heron** (Old World) and **Rufous-tailed
Nightjar** (Neotropics) — one at 0.90 confidence.

This is precisely what CLAUDE.md §4.3 occurrence re-ranking exists to catch, and it
makes that the highest-value remaining accuracy work rather than a refinement.

Two genuine confusions are also range problems in disguise:

- **Boat-tailed vs Great-tailed Grackle** (0/3). Great-tailed is scarce in peninsular
  Florida; occurrence data resolves it for free.
- **Tricolored vs Little Blue Heron** (the dominant confusion, 3 occurrences). Both are
  in range — this one is a genuine vision problem and belongs in the prompt.

---

## Scientific names look authoritative and are sometimes wrong

Across 927 candidate names: **zero missing, zero malformed** — every binomial is
well-formed `Genus species`. But three common names carry conflicting binomials, which
proves at least one is wrong without consulting any authority:

| common name | binomials returned |
|---|---|
| double-crested cormorant | `Nannopterum auritum` ×7, `Nannopterum brasilianum` ×2, `Phalacrocorax auritus` ×1 |
| long-tailed cuckoo | `Cuculus micropterus` ×2, `Cercococcyx olivinus` ×1 |
| rufous-tailed nightjar | `Uropsalis segmentata` ×1, `Uropsalis longicauda` ×1 |

`Nannopterum brasilianum` is **Neotropic Cormorant**, attached to the Double-crested
common name. `Phalacrocorax auritus` is a valid older synonym.

The important part is that nothing was malformed. **No syntactic check would flag any
of this** — the names always look right. Only cross-referencing catches it, which
argues for validating against GBIF before any catalog write.

`report.name_quality()` performs the local consistency half of this today.

---

## Two measurement bugs, and what they cost

Both produced plausible numbers that were wrong. Recorded because the failure mode
generalises.

**1. Hyphenation counted as a different species.** The first report showed 71.4% macro
and 45.5% overall. The top "confusion" was `tricolored heron → tri-colored heron` —
one species, two hyphenations, and the model had returned the correct binomial anyway.
Corrected: **79.4% macro, 68.2% overall.** A 23-point swing that was pure measurement
error.

**2. The first fix was incomplete.** Stripping hyphens unified `Tri-colored` with
`Tricolored` but left `Night-Heron` and `Night Heron` distinct. Caught by a test
written specifically around the first bug.

Comparison keys now drop every separator, and a match on *either* the common or the
scientific name counts. Display names are kept separate so reports stay legible.
`service/tests/test_report.py` exists so a measurement bug cannot masquerade as a model
result a third time.

---

## What this corpus cannot tell you

- **Five of seven taxa are untested.** The corpus is birds plus one reptile. There are
  no fish, amphibians, insects, mammals or plants, so those prompts are unvalidated
  regardless of how many frames are processed.
- **Two shooting dates, one lens, one location.** No seasonal variation, no focal-length
  variation, and no test of the location re-ranking that Stage 2 depends on.
- **No GPS anywhere**, so §4.3 cannot be exercised on this data at all.
- **The dev set is a subset of the full set.** Tuning against `fixtures/` and then
  "validating" on `fixtures_full/` is contaminated, though only mildly at 25/1743.

---

## Reproducing

```bash
# Per-image table, name quality, and scoring
.venv/bin/melampus-id fixtures/ --report-only --labels fixtures_dev_labels.json

# Encounter-aggregated accuracy and within-encounter stability
.venv/bin/python tools/analyze_encounters.py fixtures_full stage1_full_results.json
```

Inference is deterministic (`temperature = 0.0`), so identical inputs reproduce
identical outputs.
