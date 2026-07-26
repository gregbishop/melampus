# Melampus — Local AI Species ID & Quality Triage for Lightroom Classic

> Named for the Greek seer who, after serpents cleaned his ears as he slept, could understand the
> speech of animals — birds especially. Pronounced *meh-LAM-pus*.

## Purpose of this document

This is a build spec to hand to Claude Code. Read it fully before writing code. Where it says
**VERIFY**, do not trust the spec — check the actual SDK/API docs or probe the environment, then
proceed. Several details here are stated from general knowledge and may be wrong in specifics.

Build in the phase order given. Phase 1 must work end-to-end before starting Phase 2.

---

## 1. Goal

A Lightroom Classic plugin that, for a selected batch of photos:

1. Identifies the organism (bird, fish, reptile, amphibian, insect, mammal, plant/flower) with
   ranked candidates and calibrated confidence.
2. Scores technical quality — sharpness measured **on the subject, not the frame**.
3. Writes results into native LrC ratings, flags, color labels, and hierarchical keywords, plus
   custom metadata fields.
4. Uses capture location and date to re-rank species candidates against real occurrence data.
5. Never destroys existing user metadata. Ever.

All inference is local, on Apple Silicon, via MLX. No cloud dependency in the default path.

## 2. Target environment

- Mac Studio M4 Max, 128 GB unified memory, macOS current
- Adobe Lightroom Classic (recent version) — **VERIFY** the installed SDK version and target it
- Python 3.11 or 3.12 for the backend (**not** 3.13+ — some ML wheels lag; verify what your chosen
  deps actually support before pinning)
- Camera: Canon R3, CR3 raw files

## 3. Architecture

Two processes, thin boundary between them.

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│  LrC Plugin (Lua)            │        │  Melampus Service (Python)    │
│                              │  HTTP  │                               │
│  • menu items                │◄──────►│  • VLM inference (MLX)        │
│  • config dialog             │  JSON  │  • sharpness / quality CV     │
│  • reads GPS + capture date  │        │  • range & season re-ranking  │
│  • exports JPEG previews     │        │  • occurrence API clients     │
│  • writes catalog metadata   │        │  • result cache (SQLite)      │
└──────────────────────────────┘        └───────────────────────────────┘
```

**Why split:** the Python side is independently testable without launching Lightroom. Build and
test it first (Phase 1). The Lua side is thin glue.

**Service transport:** a local HTTP service on a configurable port (default 8765). Plain
JSON in, JSON out. Consider `vllm-mlx` as the model server underneath it — it exposes both
OpenAI- and Anthropic-compatible endpoints on Apple Silicon, and has content-based prefix caching
for repeated image queries. **VERIFY** its current API surface and whether it fits; if it adds
friction, call `mlx-vlm` directly from the service instead. Either way, keep the model call behind
a single interface so it can be swapped.

**Model:** Qwen3-VL, MoE variant (e.g. 30B-A3B) in 4-bit MLX quantization as the default.
128 GB allows going much larger — make the model a config setting, not a hardcode. **VERIFY**
current MLX-converted VLM availability on Hugging Face; the landscape moves fast, so check what's
actually published rather than assuming a specific repo name exists.

## 4. Phase 1 — Python service, no Lightroom

Deliverable: a CLI that takes a JPEG path plus optional lat/lon/date and prints result JSON.

### 4.1 Quality assessment (pure CV, no model)

This is the part where naive implementations fail on wildlife. Requirements:

- **Subject-localized sharpness.** Detect the primary subject, then compute sharpness *inside*
  that region. A whole-frame variance-of-Laplacian is wrong for long-lens wildlife work — shallow
  depth of field means background blur drags the global score down on excellent images.
- **Eye sharpness where an eye is detectable.** For vertebrates this is the metric a wildlife
  photographer actually culls on. If an eye is found, weight it heavily. If not, fall back to the
  subject box.
- **Subject size fraction.** Percentage of frame the subject occupies. Sharpness scores are not
  comparable across a full-frame portrait and a distant speck — normalize, and report the fraction
  so it can be used in filtering.
- **Motion blur vs defocus discrimination.** Directional blur (a wingbeat) is often desirable;
  isotropic defocus is not. Report them separately rather than collapsing to one number.
- **Exposure diagnostics.** Clipped highlights percentage, clipped shadows percentage. Report
  raw numbers, not a verdict.
- **Subject position** relative to frame edges — flag subjects clipped by the frame boundary.

Output a `quality` block with each sub-metric exposed **individually** plus a composite 0–100.
Do not hide the sub-metrics. The user needs to tune weights and will not trust an opaque score.

Composite weights must be config-driven, not hardcoded.

### 4.2 Species identification

Two-stage:

**Stage A — taxon routing.** Classify the coarse group: bird / fish / reptile / amphibian /
insect / arachnid / mammal / plant / fungus / none. Cheap, and it lets Stage B use a
taxon-appropriate prompt.

**Stage B — species ID** with a prompt specialized per taxon. Birds get plumage/structure
guidance; plants get inflorescence/leaf-arrangement guidance; snakes get scale/pattern/head-shape
guidance. Write these prompts as separate editable files, not string literals buried in code.

**Required output contract** — the model must return strict JSON:

```json
{
  "taxon": "bird",
  "candidates": [
    {"common_name": "...", "scientific_name": "...", "confidence": 0.0, "reasoning": "..."},
    {"common_name": "...", "scientific_name": "...", "confidence": 0.0, "reasoning": "..."}
  ],
  "age_sex": "adult male | juvenile | breeding plumage | indeterminate",
  "count": 1,
  "behavior": ["perched", "in-flight", "feeding", "displaying"],
  "diagnostic_features_visible": true,
  "abstain": false,
  "abstain_reason": null
}
```

Hard requirements on this stage:

- **Always return ranked candidates, never a single answer.** Top-1-only output is how you get
  confidently wrong tags.
- **Abstention must be a first-class outcome.** If diagnostic features aren't visible — bird
  facing away, flower with no visible reproductive parts, blurry snake — the correct answer is
  `abstain: true`, not a guess. Prompt for this explicitly and reward it.
- **Parse defensively.** Local VLMs are less reliable at strict JSON than frontier cloud models.
  Validate against a schema, retry once with a corrective message on failure, then give up
  gracefully and mark the photo unprocessed rather than writing garbage.
- Confidence from a VLM is **not calibrated**. Treat it as an ordinal hint. Do not present it as
  a probability in the UI. See §4.4.

### 4.3 Location and season re-ranking

This is the highest-value accuracy feature. Do not skip it.

Given lat/lon and capture date from EXIF:

1. Query occurrence data for species recorded near that location, in that part of the year.
   - **Birds:** eBird API 2.0 has far denser data than anything else. Requires a free API key.
     **VERIFY** current endpoints and terms of use.
   - **Everything else:** GBIF occurrence API (open, no key) and/or iNaturalist API. **VERIFY**
     endpoints and rate limits.
2. Re-rank the VLM's candidate list. A candidate with strong local records in that month is
   promoted; one with zero regional records is demoted.
3. **Do not silently delete out-of-range candidates.** If the VLM's top pick is regionally
   improbable, that is *interesting* — it's either a model error or a genuinely notable record.
   Emit a `range_flag` and route the photo to human review either way.
4. Cache occurrence queries aggressively — keyed on rounded coordinates plus month. Most of a
   photographer's shots cluster in a handful of locations, so hit rates will be high. Respect
   rate limits and back off politely.
5. **Degrade gracefully.** No GPS, no network, or API down must not break the pipeline — just
   skip re-ranking and note it in the output.

Also derive a `season` field (month, and optionally a plumage-relevant hint like "fall migration")
and pass it into the Stage B prompt as context. Molt state and bloom timing are real ID signals.

### 4.4 Confidence policy

Convert model confidence plus range agreement into three bands, thresholds config-driven:

- **High** — auto-tag with the species.
- **Medium** — tag, but also add a review keyword.
- **Low / abstain / range-flagged** — do **not** write a species keyword. Add a review marker only.

The single most important rule in this document: **a wrong keyword is worse than no keyword.**
A polluted 20-year catalog is very expensive to clean. Bias hard toward abstention.

## 5. Phase 2 — Lightroom plugin

### 5.1 Reading from the catalog

- GPS: **VERIFY** the metadata key and return shape for coordinates via the photo metadata API.
- Capture date: **VERIFY** the field name for original capture time.
- Pixels for the model: the SDK can produce a JPEG preview/thumbnail for a photo. **VERIFY** the
  exact call, its async signature, and its size limits. If preview generation proves unreliable
  for CR3, fall back to a temporary JPEG export at a bounded long edge (~1600 px is ample for ID
  and keeps inference fast). Do not send full-resolution files to the model.
- Note: some raw formats give plugins trouble. Test CR3 specifically, early. If it fights you,
  the export path is the escape hatch.

### 5.2 Writing to the catalog

Use LrC's native systems — this is the explicit requirement:

- **Star rating** from the composite quality score. Config-driven breakpoints.
- **Pick / reject flags** from thresholds. **Reject must be opt-in and off by default** — an
  automated reject pass across a real library is a destructive-feeling operation.
- **Color labels** by quality band, optional.
- **Hierarchical keywords**, structured so they're useful in the keyword panel:

```
Melampus > Taxon > Bird
Melampus > Species > Osprey
Melampus > Confidence > High
Melampus > Behavior > In-flight
Melampus > Review > Needs ID
Melampus > Notable > Out of range
```

  Consider mirroring true taxonomy (Class > Order > Family > Species) as an option — it makes the
  keyword tree browsable the way a field guide is organized.

- **Custom plugin metadata fields** for the numeric detail that shouldn't be keywords: species
  confidence, eye-sharpness score, subject size fraction, motion-vs-defocus values, clipping
  percentages, range-agreement score, model name and version, processing timestamp, schema version.
  Include the model/version — when a better model lands, you need to know what was tagged by what
  to decide what merits reprocessing.

### 5.3 Non-negotiable safety rules

- **Never overwrite an existing user rating, flag, label, or keyword.** Default to writing only
  where the field is empty. Overwrite must be explicit opt-in per field type.
- **Dry-run mode** that reports every intended change without writing. Make this the default for
  the first run on any catalog.
- **Idempotency.** Track processed state in a custom field. Re-running must be a no-op unless
  forced.
- **Chunked batches with resume.** Process in modest batches; a crash at photo 4,000 must not
  lose the first 3,999.
- Wrap all catalog writes in the SDK's write-access mechanism. **VERIFY** the correct pattern —
  and be aware that mixing async operations inside write-access blocks is a known source of
  "yielding is not allowed" errors in LrC plugins. Get model inference and network calls *fully
  completed* before opening a write transaction.

## 6. Phase 3 — features worth having

Ordered roughly by value-to-effort.

1. **Burst culling.** Wildlife shooting means 40-frame bursts of the same subject. Group by
   capture-time proximity plus visual similarity, rank within group by subject sharpness, mark the
   best, optionally stack. This may save more time than the species ID does.
2. **Life-list / first-of-species detection.** Flag when a species is new to the catalog. Report
   first-observed date per species.
3. **Native vs. non-native flagging for plants.** Given Florida's invasive pressure, tagging
   Brazilian pepper, air potato, etc. as invasive — and natives as natives — is useful beyond
   photography. Needs a regional species list; **VERIFY** what authoritative source is available
   (state / UF-IFAS lists, USDA PLANTS).
4. **Correction logging.** When the user fixes an ID, record `(image, wrong ID, correct ID,
   location, date)` to a local dataset. Two payoffs: prompt tuning, and eventually a fine-tune of
   the VLM on local taxa. MLX-VLM supports fine-tuning, so this is a real path, not a fantasy.
5. **Observation export.** Emit a CSV of species / date / coordinates / count from photo metadata,
   shaped for eBird or iNaturalist import. Turns the photo library into a records source.
6. **Optional cloud escalation for the low-confidence tail.** Since the local server can speak an
   Anthropic-compatible protocol, low-confidence and range-flagged photos can be re-run against
   the Claude API with the same request shape and a base-URL swap. Keep it **off by default**,
   clearly labeled, and scoped to the handful of hard cases — that's where cloud accuracy is worth
   it and where the volume is small enough to be cheap.

## 7. Config

Single config file, editable outside Lightroom, with a dialog in-plugin for the common knobs:

- model name/path, service URL and port
- quality composite weights, per sub-metric
- rating breakpoints, flag thresholds, color-label mapping
- confidence band thresholds
- keyword root and hierarchy style (flat / taxonomic)
- per-field overwrite permissions (default: all off)
- occurrence API keys, cache TTL, offline mode
- dry-run toggle (default on for first run)
- taxa to attempt (allow disabling groups)

## 8. Testing

- Python service tested standalone with a fixture set: sharp/soft pairs, a distant small subject,
  a backlit bird, a bird facing away (must abstain), a burst sequence, a non-organism frame
  (must return `taxon: none`), a photo with no GPS.
- Explicitly assert that a sharp subject against a blurred background scores **high**. This is
  the regression test that catches the global-sharpness mistake.
- Assert idempotency: run twice, second run writes nothing.
- Assert non-destruction: pre-set a rating and keyword, run, confirm untouched.
- Test against real CR3 files early, before building out features.

## 9. Deliverables

1. `service/` — Python package, CLI entry point, HTTP server, test suite
2. `prompts/` — editable per-taxon prompt templates
3. `plugin/Melampus.lrplugin/` — Lua plugin
4. `README.md` — install steps for both halves, including Python version pinning and macOS
   Gatekeeper notes for any unsigned binary
5. `docs/CONFIG.md` — every setting, with rationale for defaults

## 10. Start here

Phase 1, §4.1 only: build the subject-localized sharpness scorer and prove it on a folder of real
CR3-derived JPEGs. No model, no Lightroom, no network. Get that right, show the numbers, and only
then move to species ID.
