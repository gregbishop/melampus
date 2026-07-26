# Melampus

Local AI species identification and quality triage for Lightroom Classic, running
entirely on Apple Silicon via MLX. No cloud dependency in the default path.

> Named for the Greek seer who, after serpents cleaned his ears as he slept, could
> understand the speech of animals — birds especially. Pronounced *meh-LAM-pus*.

See `CLAUDE.md` for the full build specification.

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | Local VLM species identification | pipeline complete, tests passing |
| 2 | Quality scoring + location/season re-ranking | not started |
| 3 | HTTP service + frozen binary | not started |
| 4 | Lightroom Classic plugin | not started |

## Requirements

- Apple Silicon Mac (MLX is arm64-only). Developed on an M4 Max / 128 GB.
- **Python 3.12** — not 3.13+. `mlx-vlm` and its dependency stack publish wheels for
  3.12; 3.13 is ahead of parts of that stack.
- Roughly 20 GB of disk for the default model.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "./service[dev]"
```

### Model weights

Weights are **not** bundled — they download on first use into the standard
HuggingFace cache. To fetch the default model ahead of time:

```bash
.venv/bin/hf download mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
```

If the download stalls at zero bytes, disable the Xet transfer backend, which is a
known failure on some networks:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/hf download mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
```

Published MLX builds, verified against the HuggingFace API. Note that the widely
downloaded `unsloth/*-bnb-4bit` repositories are bitsandbytes/CUDA and **will not run
on MLX** — use the `mlx-community` conversions.

| Model | Size | Notes |
|---|---|---|
| `Qwen3-VL-30B-A3B-Instruct-4bit` | 18.3 GB | MoE, ~3B active. Default. |
| `Qwen3-VL-32B-Instruct-4bit` | 19.6 GB | Dense |
| `Qwen3-VL-32B-Instruct-8bit` | 36.0 GB | Dense, minimal quantization loss |
| `Qwen3-VL-8B-Instruct-4bit` | 5.8 GB | Small baseline |
| `Qwen3-VL-2B-Instruct-4bit` | 1.8 GB | Smoke-testing only |
| `Qwen3-VL-235B-A22B-Instruct-4bit` | 133.4 GB | Exceeds 128 GB — not viable |

## Use

```bash
# Identify every JPEG in a folder, printing the raw per-image candidate table
.venv/bin/melampus-id fixtures/

# Score against a reference label set
.venv/bin/melampus-id fixtures/ --labels fixtures_dev_labels.json

# Re-print tables from cache without running the model
.venv/bin/melampus-id fixtures/ --report-only --labels fixtures_dev_labels.json

# Try a different model
.venv/bin/melampus-id fixtures/ --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Stratified sample: at most 3 images per previously-predicted species
.venv/bin/melampus-id fixtures_full/ --per-species 3
```

Runs are **resumable**. Results are cached by file content hash and fsynced after every
image, so a crash loses at most the image in flight and a second run over the same
folder is a no-op. Pass `--force` to reprocess.

## How identification works

Two stages, both validated against a strict JSON schema:

1. **Taxon routing** — a cheap coarse classification (bird / fish / reptile / amphibian
   / insect / arachnid / mammal / plant / fungus / none). A frame routed to `none`
   short-circuits: there is no point asking a species question about an empty frame.
2. **Species identification** — a taxon-specialised prompt from `./prompts/`, returning
   ranked candidates with confidence and reasoning, age/sex, count, behaviour, and an
   explicit abstention flag.

On a schema-validation failure the model gets **one** corrective retry, and then the
image is marked `unprocessed` rather than having a guess written to it. Confidence from
a VLM is not calibrated and is treated as an ordinal hint only.

Abstention is a first-class outcome. A bird facing away, a submerged reptile, or a
flower with no visible reproductive parts should abstain, because **a wrong keyword is
worse than no keyword** — a polluted 20-year catalog is very expensive to clean.

### Known runtime limit

`mlx-vlm` 0.6.7 stops generating — returns a single EOS token and an empty string —
once the prompt exceeds roughly 2,100 tokens, far below Qwen3-VL's real context window.
Vision tokens dominate that budget, so images are sent at 1280 px rather than the
1600 px CLAUDE.md §5.1 suggests, with an automatic retry ladder at 1024 and 768 px.
If you see a run come back mostly `unprocessed`, this is almost certainly the cause.
Full measurements and the reasoning are in [docs/CONFIG.md](docs/CONFIG.md).

## No metadata reaches the model

Only pixels are ever sent. This is enforced structurally rather than by convention, so
that it cannot be undone by a careless call site:

- `images.staged_pixels` re-encodes every image to a temporary file with the fixed
  neutral name `image.jpg`, rebuilt from raw pixel bytes. EXIF, XMP and IPTC do not
  survive. The backend only ever receives that staged path.
- `prompts.render` refuses any substitution key outside an explicit allowlist, so a
  filename or keyword cannot be interpolated into a prompt.

Both are covered by tests (`test_staged_image_is_renamed_and_stripped`,
`test_backend_never_receives_original_filename`, `test_prompt_rejects_unapproved_context`).

## Prompts

Editable Markdown files in `./prompts/`, one per taxon plus `taxon_routing.md` and a
`generic.md` fallback. They are deliberately not string literals in code: the confusion
report tells you which discriminations to tighten, and you tighten them here without
touching Python.

## Tests

```bash
cd service && ../.venv/bin/python -m pytest -q
```

The suite runs against a scripted backend and needs no model weights.

## Layout

```
service/melampus/    Python package (library; CLI is a thin shell over it)
prompts/             Editable per-taxon prompt templates
tools/               Corpus utilities (encounter clustering, dev-set split)
fixtures/            Small development set
fixtures_full/       Full validation corpus
```
