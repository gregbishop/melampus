# Melampus

Local AI species identification and photo-quality triage for Adobe Lightroom Classic,
running entirely on Apple Silicon via MLX. No cloud dependency in the default path, and
no image ever leaves the machine.

> Named for the Greek seer who, after serpents cleaned his ears as he slept, could
> understand the speech of animals — birds especially. Pronounced *meh-LAM-pus*.

`CLAUDE.md` is the build specification. This README is how to run what exists today.

---

## Status

| Stage | Scope | State |
|---|---|---|
| **1** | Local VLM species identification | **Working.** 43 tests passing |
| 2 | Quality scoring + location/season re-ranking | Not started |
| 3 | HTTP service + frozen binary | Not started |
| 4 | Lightroom Classic plugin | Not started |

Stage 1 exists to answer one question before anything else gets built: *can a local
VLM identify species well enough to be worth wiring into a catalog?* The current
answer is **a qualified yes** — see [docs/evaluation.md](docs/evaluation.md) for what
the numbers do and don't support.

---

## Requirements

- **Apple Silicon Mac.** MLX is arm64-only. Developed on an M4 Max / 128 GB.
- **Python 3.12** — not 3.13+. The `mlx-vlm` dependency stack publishes wheels for
  3.12; 3.13 runs ahead of parts of it.
- ~20 GB of disk for the default model.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "./service[dev]"
```

Model weights are **not bundled**. They download on first use into the standard
HuggingFace cache, or fetch them ahead of time:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/hf download mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
```

`HF_HUB_DISABLE_XET=1` is not optional on some networks — see
[docs/troubleshooting.md](docs/troubleshooting.md).

### Available models

Verified against the HuggingFace API. The heavily-downloaded `unsloth/*-bnb-4bit`
repositories are bitsandbytes/CUDA builds and **will not run on MLX** — use the
`mlx-community` conversions.

| Model | Size | Notes |
|---|---|---|
| `Qwen3-VL-30B-A3B-Instruct-4bit` | 18.3 GB | MoE, ~3B active. **Default.** |
| `Qwen3-VL-32B-Instruct-4bit` | 19.6 GB | Dense |
| `Qwen3-VL-32B-Instruct-8bit` | 36.0 GB | Dense, minimal quantization loss |
| `Qwen3-VL-8B-Instruct-4bit` | 5.8 GB | Small baseline |
| `Qwen3-VL-2B-Instruct-4bit` | 1.8 GB | Too weak for stage B; smoke tests only |
| `Qwen3-VL-235B-A22B-Instruct-4bit` | 133.4 GB | Exceeds 128 GB — not viable |

---

## Quickstart

```bash
# Identify every JPEG in a folder, printing the raw per-image candidate table
.venv/bin/melampus-id fixtures/

# Score against a reference label set
.venv/bin/melampus-id fixtures/ --labels fixtures_dev_labels.json

# Re-print tables from cache without running the model (free, instant)
.venv/bin/melampus-id fixtures/ --report-only --labels fixtures_dev_labels.json

# A different model
.venv/bin/melampus-id fixtures/ --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Stratified sample: at most 3 images per previously-predicted species
.venv/bin/melampus-id fixtures_full/ --per-species 3
```

Expect roughly **7 seconds per image** on an M4 Max with the default model, plus about
40 seconds of one-time model loading.

### Runs are resumable

Results are cached by **file content hash**, fsynced after every image. A crash loses
at most the frame in flight, and re-running the same folder is a no-op. Renaming or
moving a file still hits the cache. `--force` reprocesses regardless.

The cache key also covers the model, the prompt set, the image size and the sampling
settings — so editing a prompt correctly invalidates prior results instead of silently
re-serving the old wording's answers.

---

## Reviewing results locally

Identification output is only useful if you can correct it. `review.html` is a
single self-contained page — thumbnails embedded, opens from disk, no server and no
network, and it writes nothing until you press Download.

```bash
# Build the sheet
.venv/bin/python tools/make_review_sheet.py fixtures_full stage1_full_results.json review.html
open review.html

# Feed your judgements back in
.venv/bin/python tools/ingest_corrections.py fixtures_full \
    ~/Downloads/melampus_corrections.json fixtures_full_labels.json

# Rescore against them
.venv/bin/melampus-id fixtures_full --report-only --labels fixtures_full_labels.json
```

**Review is per encounter, not per photo.** Wildlife shooting produces bursts, so
1,743 frames are roughly 43 shooting encounters, and every frame within one is the
same individual. Judging 43 representatives labels the entire corpus in minutes.

Encounters where the model called different species on different frames of the same
bird are flagged **unstable** — that is where a human eye is worth the most.

This is also the correction dataset CLAUDE.md §6.4 wants for prompt tuning and
eventual fine-tuning, so the effort compounds.

---

## How identification works

Two stages, both validated against a strict JSON schema:

1. **Taxon routing** — a cheap coarse classification: bird / fish / reptile /
   amphibian / insect / arachnid / mammal / plant / fungus / none. A frame routed to
   `none` short-circuits, because there is no point asking a species question about an
   empty frame. It also runs ~2.5× faster.
2. **Species identification** — a taxon-specialised prompt from `prompts/`, returning
   ranked candidates with confidence and reasoning, plus age/sex, count, behaviour, and
   an explicit abstention flag.

On a schema-validation failure the model gets **one** corrective retry, and then the
image is recorded as `unprocessed` rather than having a guess written to it.

**Abstention is a first-class outcome**, not a failure. A bird facing away, a submerged
reptile, or a flower with no visible reproductive parts should abstain, because *a
wrong keyword is worse than no keyword* — a polluted twenty-year catalog is expensive
to clean.

Confidence from a VLM is **not calibrated** and is treated as an ordinal hint only.
Measured behaviour is in [docs/evaluation.md](docs/evaluation.md).

---

## No metadata reaches the model

Only pixels are ever sent. Not filenames, keywords, subject tags, EXIF, XMP or IPTC.
This is enforced structurally rather than by convention, so a careless call site
cannot undo it:

- **`images.staged_pixels`** re-encodes every image to a temporary file named exactly
  `image.jpg`, rebuilt from raw pixel bytes. EXIF, XMP and IPTC do not survive that
  round trip, and the backend only ever receives the staged path — the original
  filename never appears in any argument.
- **`prompts.render`** refuses any substitution key outside an explicit allowlist, so
  a filename or keyword cannot be interpolated into a prompt.

Three tests treat this as load-bearing: `test_staged_image_is_renamed_and_stripped`,
`test_backend_never_receives_original_filename`, `test_prompt_rejects_unapproved_context`.

Verified end to end: inference runs correctly with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

---

## Prompts

Editable Markdown in `prompts/` — one per taxon, plus `taxon_routing.md` and a
`generic.md` fallback. Deliberately not string literals in code: the confusion report
tells you which discriminations to tighten, and you tighten them here without touching
Python.

Editing a prompt changes the cache fingerprint, so the next run genuinely re-runs.

---

## Tests

```bash
.venv/bin/python -m pytest -q
```

43 tests, no model weights required — everything runs against a scripted backend, so
parsing, validation, retry, caching, the downscale ladder and the no-leak guarantee are
all verifiable in under a second.

---

## Layout

```
service/melampus/     Python package. Library first; the CLI is a thin shell over it
  backend.py          The model call, behind one swappable interface
  identify.py         Two-stage identification, defensive parsing, retry
  images.py           Pixel staging — and the metadata-leak enforcement point
  cache.py            Content-hash result cache, resumable, checkpointed
  report.py           Raw table, scoring, calibration, name-quality checks
prompts/              Editable per-taxon prompt templates
tools/                Corpus utilities: clustering, dev split, review sheet, ingest
docs/                 Architecture, evaluation methodology, config, troubleshooting
fixtures/             Small development set
fixtures_full/        Full validation corpus
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit and why
- [docs/evaluation.md](docs/evaluation.md) — what the accuracy numbers mean, and don't
- [docs/config.md](docs/config.md) — every setting, with rationale
- [docs/troubleshooting.md](docs/troubleshooting.md) — known failure modes
