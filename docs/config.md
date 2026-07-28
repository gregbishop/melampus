# Configuration

Every setting, with the reasoning behind its default. Config is data, never code —
nothing in the Python hardcodes a model name, threshold or weight.

Settings load from an optional TOML file passed with `--config`, layered over the
built-in defaults. A user file need only contain the keys it changes. Programmatic
callers can also pass overrides directly to `load_config(...)`, which is how the HTTP
service in Stage 3 will accept per-request configuration.

```bash
melampus-id fixtures/ --config my-settings.toml
```

```toml
# my-settings.toml — only what differs from the defaults
[model]
repo = "mlx-community/Qwen3-VL-32B-Instruct-8bit"

[image]
max_edge = 1280
```

---

## `[model]`

| Key | Default | Why |
|---|---|---|
| `repo` | `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` | CLAUDE.md §3 requires the model be a setting, not a hardcode. This is the MoE build named in the spec: 18.3 GB with roughly 3B active parameters, so it runs far faster than a dense model of similar quality. 128 GB of unified memory allows going considerably larger — see the table in the README. |
| `max_tokens` | `900` | Enough for a full identification with two or three candidates, each carrying a reasoning string. Too low truncates mid-JSON and forces a wasted retry. |
| `temperature` | `0.0` | Identification is a discrimination task, not a creative one. Deterministic decoding also makes re-runs reproducible, which matters when tuning prompts against the confusion report. |
| `routing_max_tokens` | `200` | Stage A returns three short fields. Capping it low keeps the cheap stage cheap. |

### Choosing a model

Prompt length interacts with model capability more sharply than expected. The taxon
prompts run to roughly 3,300 characters. Measured on `Qwen3-VL-2B-Instruct-4bit`:
prompts under ~1,200 characters return valid JSON, while prompts beyond ~1,600 cause
the model to emit an immediate EOS or degenerate into a repeated list marker. Small
models are therefore unsuitable for Stage B regardless of how the prompt is phrased.
If you switch to a smaller model and see a spike in `unprocessed`, this is the cause.

---

## `[image]`

| Key | Default | Why |
|---|---|---|
| `max_edge` | `1280` | Long edge sent to the model. Lower than the ~1600 px CLAUDE.md §5.1 suggests — see the runtime limit below, which makes 1600 unsafe. Set to `0` to disable resizing. |
| `fallback_edges` | `[1024, 768]` | Tried in order when generation returns empty. |
| `jpeg_quality` | `92` | High enough that re-encoding does not soften the fine plumage and scale detail that identification depends on. |

Resizing happens inside `images.staged_pixels`, which is also the enforcement point for
the no-metadata rule — see below.

### The prompt-token ceiling (important)

`mlx-vlm` 0.6.7 driving Qwen3-VL stops generating once the prompt passes roughly 2,100
tokens: the model emits a single EOS token and returns an empty string. Measured on
`Qwen3-VL-30B-A3B-Instruct-4bit` with a full bird prompt, varying only image size:

| Long edge | Prompt tokens | Outcome |
|---|---|---|
| 768 | 1,217 | generates normally |
| 1,300 | 1,940 | generates normally |
| 1,400 | 2,109 | generates normally |
| 1,450 | 2,183 | **empty** |
| 1,600 | 2,483 | **empty** |

This is far below Qwen3-VL's real context window, so it is a defect in the runtime
rather than a model limit, and 0.6.7 is the latest published release. It reproduces on
`Qwen3-VL-2B-Instruct-4bit` too, so it is not specific to the MoE build.

Two consequences worth knowing:

- At `temperature = 0` the failure is silent and total — an empty reply. Raising the
  temperature to 0.7 makes the model generate, but it then tends to ignore the schema
  and invent its own keys, which surfaces as `taxon: Field required` validation errors.
  Neither is a usable workaround.
- Vision tokens dominate the budget, so **image size is the lever, not prompt wording**.
  Shortening a taxon prompt buys back roughly 800 tokens; halving the image buys back
  far more.

`max_edge` plus `fallback_edges` handle this: identification is retried at each
successively smaller size until generation succeeds, because the exact threshold shifts
with image aspect ratio. `ImageResult.image_max_edge` records the size that actually
worked, so silent degradation is visible in the results.

The tradeoff is real. Less resolution means less detail for small-in-frame subjects —
distant birds and partly submerged reptiles are the cases that suffer. If accuracy on
those is poor, shorten the prompts to free budget for pixels rather than raising
`max_edge` past the ceiling.

---

## `[run]`

| Key | Default | Why |
|---|---|---|
| `prompts_dir` | `<repo>/prompts` | Per-taxon prompt templates as editable files (CLAUDE.md §4.2). Point this elsewhere to A/B a prompt set without touching the installed package. |
| `cache_path` | `<repo>/.melampus_cache/identifications.jsonl` | Append-only JSONL, fsynced after every image. |
| `max_retries` | `1` | One corrective retry on schema-validation failure, exactly as §4.2 specifies. Then the image is marked `unprocessed` rather than having a guess written to it. More retries mostly burn time on images the model cannot parse anyway. |

### Caching and resume

Results are keyed on the **SHA-256 of the file contents**, not the path. Renaming,
moving or re-exporting a file to a different folder still hits the cache, so a second
run over the same corpus is a no-op. `--force` reprocesses regardless.

Each result is written and fsynced as soon as it is produced, so a crash loses at most
the image in flight. A truncated final line from an abrupt kill is skipped on reload
rather than aborting the run.

---

## `[escalation]` — optional cloud second opinion

CLAUDE.md §6.6. The local model handles most of a catalog well and costs nothing to
run. What it does not handle is the tail — frames where it abstains, splits between
two similar species, or names something that does not occur within a thousand miles.
That tail is a few hundred frames out of a few thousand, which is exactly the shape
where a frontier model is worth paying for.

**This is the only path in the project that sends a photograph off your machine.** It
is off by default and additionally needs an API key, so it takes two deliberate acts
to turn on. It obeys the same pixels-only rule as everything else: escalated frames go
through `images.staged_pixels` exactly as local ones do, so no filename, EXIF or
keyword travels with them.

| Key | Default | Why |
|---|---|---|
| `enabled` | `false` | Off unless you ask. `--escalate` turns it on for one run. |
| `provider` | `"anthropic"` | `anthropic` (Claude API) or `openai`. For any other OpenAI-compatible endpoint — OpenRouter, LM Studio, vLLM, a proxy — use `openai` with `base_url`. |
| `base_url` | *(none)* | OpenAI-compatible endpoint override. Ignored by the Anthropic provider. |
| `model` | *(provider default)* | `claude-opus-5` or `gpt-5`. **Vision model names move faster than this file does** — check the provider's current listing and override with `--escalate-model`. |
| `api_key` | *(none)* | Never set this in tracked source. See below. |
| `effort` | `"high"` | Anthropic only. These are the frames the local model could not resolve, so thinking depth is where the money should go. |
| `max_tokens` | `1200` | Per reply (Stage B). |
| `routing_max_tokens` | `900` | Stage A budget. **Do not lower this to the local default of 200.** On a model where thinking is on by default, `max_tokens` caps thinking *and* output together, so 200 is consumed before any JSON appears and every frame fails twice while being billed. |
| `max_edge` | `2048` | Long edge sent to the cloud. Higher than the local `1280`, because the 1280 ceiling exists to dodge an mlx-vlm token bug that does not apply here, and resolution is the cheapest lever left on a hard frame. |
| `confidence_below` | `0.80` | Escalate when the local top candidate scores under this. Ordinal, not calibrated — a ranking cut, not a probability. |
| `on_abstain` | `true` | Escalate frames the local model declined to call. |
| `on_range_flag` | `true` | Escalate frames whose top candidate does not occur locally (§4.3). |
| `max_images` | `200` | Hard ceiling per run, bounded 0–5000 by the schema. A mistyped flag should not become an unexpected invoice. When the cap bites, the most uncertain frames go first and the rest are counted and reported — never silently dropped. |
| `input_usd_per_mtok` | `5.0` | Estimate only. Defaults are Claude Opus 5's rate. |
| `output_usd_per_mtok` | `25.0` | **Change both when you change provider or model**, or the printed estimate will be confidently wrong. The CLI prints the rates alongside the dollars so the assumption is visible. |
| `cache_path` | `<repo>/.melampus_cache/escalations.jsonl` | Cloud answers live in their own file. Merged into the local cache they would carry a foreign run fingerprint, and the next local pass would decide they were stale and quietly overwrite work you paid for. |

### Keys

Resolved at the point of use, in this order, and never merged into the config object
that reports and logs are built from:

1. `escalation.api_key` in `melampus.local.toml` (git-ignored)
2. `MELAMPUS_ANTHROPIC_KEY` / `MELAMPUS_OPENAI_KEY`
3. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`

Keys never cross providers: an Anthropic key will not silently authorise a request to
OpenAI.

### Usage

```bash
# See what it would send and roughly what it would cost. Needs no key, sends nothing.
melampus-id fixtures --report-only --escalate-dry-run

# Actually run it
export MELAMPUS_ANTHROPIC_KEY=sk-ant-...
melampus-id fixtures --escalate --escalate-max 50

# Or against OpenAI, with the rates corrected so the estimate means something
export MELAMPUS_OPENAI_KEY=sk-...
melampus-id fixtures --escalate --escalate-provider openai --escalate-model gpt-5
```

Install the provider SDK first — neither is a dependency of the local pipeline, so a
local-only install stays local-only:

```bash
uv pip install --python .venv/bin/python "./service[cloud]"   # anthropic
uv pip install --python .venv/bin/python "./service[openai]"  # openai
```

### What gets cached, and what gets retried

Only **permanent** outcomes are written to the escalation cache: a successful
identification, or a refusal by the provider's safety classifier. A refusal is a
decision about the image, so re-asking would spend money to be declined again.

**Transient failures are deliberately not cached.** A network drop, a bad key, a
529, or a missing SDK leaves the frame untouched, and the CLI says how many were
affected so a re-run picks them up. This matters more than it sounds: `identify()`
returns an error result rather than raising, so an earlier version cached those
failures into the "already paid for" file and skipped them forever — one outage
silently burned the whole tail, recoverable only by hand-editing JSONL.

A refusal also stops the corrective retry and the downscale ladder, both of which
would otherwise pay for repeat calls guaranteed to be refused again.

### Cost confirmation

A real run prints the selection and the estimate **before** sending anything, and
asks for confirmation. Non-interactive callers get nothing sent unless they pass
`--escalate-yes` — a confirmation that auto-answers itself when no one is watching
is decorative.

### Provenance

Each escalated result records `escalated`, `escalation_model`, `escalation_reason` and
`local_identification` — what the local model had said. That last field is the point:
it turns "is the cloud pass worth the money?" into a measurable local-vs-cloud
agreement rate rather than an impression.

A second escalation run over the same frames is a no-op; nothing is re-billed.

---

## What never reaches the model

Only pixels. Not filenames, keywords, subject tags, EXIF, XMP or IPTC. This is
structural rather than conventional, so a careless call site cannot undo it:

- **`images.staged_pixels`** re-encodes every image to a temporary file named exactly
  `image.jpg`, rebuilt from raw pixel bytes via `Image.frombytes`. EXIF, XMP and IPTC
  do not survive that round trip. The backend is only ever handed the staged path, so
  the original filename never appears in any argument.
- **`prompts.render`** refuses any substitution key outside `ALLOWED_CONTEXT_KEYS`
  (currently `season_context` and `location_context`, both reserved for Stage 2 and
  both empty today). Attempting to interpolate a filename raises `PromptError`.

Three tests cover this and should be treated as load-bearing:
`test_staging_strips_every_metadata_channel`,
`test_backend_never_receives_original_filename`,
`test_prompt_rejects_unapproved_context`.

> **Open question for Stage 2.** CLAUDE.md §4.3 asks for a season hint to be passed
> into the Stage B prompt, because molt state and bloom timing are genuine
> identification signals. That hint derives from EXIF capture date, which is metadata.
> The allowlist exists so this can be enabled deliberately and visibly rather than by
> accident, but whether to enable it at all is a decision to confirm before Stage 2,
> since it does weaken the "pixels only" guarantee.

---

## Not yet implemented

The following are specified in CLAUDE.md §7 but belong to later stages, and are listed
here so the gap is explicit rather than silent:

- Quality composite weights and per-sub-metric tuning (Stage 2)
- Rating breakpoints, flag thresholds, colour-label mapping (Stage 2 / 4)
- Confidence band thresholds for the High / Medium / Low policy in §4.4 (Stage 2)
- Occurrence API keys, cache TTL, offline mode (Stage 2)
- Keyword root and hierarchy style, per-field overwrite permissions, dry-run toggle,
  service autostart and idle-shutdown, smart-collection creation (Stages 3 and 4)
