# Troubleshooting

Failure modes hit during development, with the diagnosis that identified each. Every
one of these cost real time, and several fail *silently* — which is why they are worth
writing down rather than rediscovering.

---

## A run comes back mostly `unprocessed`

**Almost certainly the mlx-vlm prompt-token ceiling.**

`mlx-vlm` 0.6.7 driving Qwen3-VL stops generating once the prompt passes roughly 2,100
tokens: the model emits a single EOS token and returns an empty string. Measured on
`Qwen3-VL-30B-A3B-Instruct-4bit`, varying only image size:

| Long edge | Prompt tokens | Outcome |
|---|---|---|
| 768 | 1,217 | generates normally |
| 1,300 | 1,940 | generates normally |
| 1,400 | 2,109 | generates normally |
| 1,450 | 2,183 | **empty** |
| 1,600 | 2,483 | **empty** |

This is far below Qwen3-VL's real context window, so it is a defect in the runtime
rather than a model limit. It reproduces on `Qwen3-VL-2B-Instruct-4bit` too, so it is
not specific to the MoE build, and 0.6.7 is the latest published release.

**Two traps when diagnosing it.**

- At `temperature = 0` the failure is *silent and total* — an empty reply, no error.
- Raising temperature to 0.7 makes the model generate, but it then tends to ignore the
  output schema and invent its own keys, surfacing as `taxon: Field required`
  validation errors. That looks like a different bug. It is not, and it is not a
  workaround.

**The fix in place.** Vision tokens dominate the budget, so image size is the lever,
not prompt wording. `max_edge` defaults to 1280 with `fallback_edges = [1024, 768]`
retried automatically when generation returns empty. `ImageResult.image_max_edge`
records the size that actually worked, so degradation is visible.

This diverges from CLAUDE.md §5.1, which specifies ~1600 px. That value is not safely
usable with this runtime and a full taxon prompt.

If you need more resolution, shorten the taxon prompts to free token budget — do not
raise `max_edge` past the ceiling.

---

## Model download stalls at zero bytes

The HuggingFace **Xet** transfer backend hangs on some networks. The symptom is
distinctive: `hf download` reports progress, `.incomplete` blobs appear in the cache,
and they stay at exactly 0 bytes indefinitely while plain `curl` against the same URL
works fine.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/hf download mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
```

**If a download was interrupted**, stale lock files can deadlock the retry — you will
see `Still waiting to acquire lock on ...` climbing indefinitely. Clear them:

```bash
pkill -f "hf download"
rm -rf ~/.cache/huggingface/hub/.locks/models--mlx-community--Qwen3-VL-30B-A3B-Instruct-4bit
```

Downloads resume from where they stopped; nothing is lost.

---

## `Qwen3VLVideoProcessor requires the Torchvision library`

Raised by `transformers.AutoProcessor.from_pretrained()`, which tries to construct a
*video* processor this project never uses, and that sub-processor wants PyTorch.

Do not install PyTorch to satisfy it. Load through `mlx_vlm.load()` instead, which
returns a working processor without the video path. `MLXBackend` already does this;
the error only appears in ad-hoc scripts that reach for `AutoProcessor` directly.

---

## Tests fail with `ModuleNotFoundError: No module named 'melampus.quality'`

Or any other import error mentioning `_old/`.

Running `pytest` from the repository root collects the archived `_old/` tree, whose
stale `melampus` package shadows the real one on `sys.path`. The root `pytest.ini`
sets `norecursedirs` to prevent this. If you have moved or copied that config, restore
it, or run from `service/`.

---

## A prompt edit seems to have changed nothing

It used to be possible for this to happen silently: the result cache was keyed on image
content alone, so editing a prompt and re-running re-served the previous wording's
answers. A tuning pass would appear to work while measuring nothing.

Fixed — the cache key now covers the model repo, prompt-set hash, image size,
`max_tokens` and temperature. If you suspect a stale result anyway:

```bash
.venv/bin/melampus-id fixtures/ --force
```

Note that results written before fingerprinting existed carry an empty fingerprint and
are honoured indefinitely, so they will never invalidate on their own. `--force` is the
only way to refresh those.

---

## Accuracy looks implausibly high or low

Three measurement traps, all of which have produced wrong numbers here. Full detail in
[evaluation.md](evaluation.md).

**Per-frame accuracy on a burst corpus is close to meaningless.** The same data scored
92.4% per frame and 46.5% macro-averaged. One large, correctly-identified encounter
swamped the average. Aggregate per encounter with
`tools/analyze_encounters.py`.

**Name comparison is a real source of false errors.** English bird name hyphenation
varies between authorities. `Tricolored Heron` vs `Tri-colored Heron` and
`Night-Heron` vs `Night Heron` are the same species and once scored as confusions,
costing 23 accuracy points of pure measurement error. Comparison keys now strip every
separator and match on either the common or the scientific name.

**A reference set built by a vision model shares blind spots with the model under
test.** Scores against `fixtures_dev_labels.json` are agreement, not correctness.

---

## Scientific names look right but are wrong

Across 927 candidate names, **zero were malformed** — every binomial was well-formed
`Genus species` — yet three common names carried conflicting binomials, including
`Nannopterum brasilianum` (Neotropic Cormorant) returned for Double-crested Cormorant.

No syntactic check will catch this. `report.name_quality()` performs the local
consistency half — flagging one common name mapped to several binomials — but genuine
validation needs an authority such as GBIF, which arrives with Stage 2.

Treat any binomial as unverified until then, and do not write one to a catalog.

---

## Checking whether inference is really local

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python tools/smoke_backend.py \
  mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit fixtures/some.jpg
```

If this succeeds, weights are being read from the local cache and nothing is reaching
the network. It should complete in ~7 seconds after model load.
