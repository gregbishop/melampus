# Architecture

How the pieces fit, and why they are shaped this way. Design decisions are recorded
with their reasoning so they can be revisited deliberately rather than by accident.

---

## The two-process split

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│  LrC Plugin (Lua)            │        │  Melampus Service (Python)    │
│                              │  HTTP  │                               │
│  • menu items                │◄──────►│  • VLM inference (MLX)        │
│  • config dialog             │  JSON  │  • sharpness / quality CV     │
│  • reads GPS + capture date  │        │  • range & season re-ranking  │
│  • exports JPEG previews     │        │  • occurrence API clients     │
│  • writes catalog metadata   │        │  • result cache               │
└──────────────────────────────┘        └───────────────────────────────┘
        Stage 4                                  Stages 1–3
```

**Why split.** The Python side is independently testable without launching Lightroom,
which is the difference between a fast feedback loop and a slow one. Everything in
Stage 1 was built and measured with no Lightroom involvement at all.

Only Stages 1–3 exist today; the Lua side is Stage 4.

---

## Library first, CLI second

`service/melampus/` is an importable library. `cli.py` is a thin shell that parses
arguments and calls into it. Nothing in the library prompts interactively, assumes a
working directory, or writes to stdout for control flow.

This is not stylistic. Stage 3 wraps the same library in an HTTP service, and Stage 4
drives that service from Lua. Anything that only works from a terminal would have to be
rebuilt.

Config is data, not code: `load_config()` accepts a TOML path *and* keyword overrides,
so the HTTP layer can accept per-request settings without touching the analysis modules.

---

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic settings, TOML layering, override merging |
| `schema.py` | The CLAUDE.md §4.2 output contract, with defensive coercion |
| `images.py` | Pixel staging, content hashing — **and the leak enforcement point** |
| `prompts.py` | Prompt loading, allowlisted substitution, prompt-set fingerprint |
| `backend.py` | The model call, behind one swappable interface |
| `identify.py` | Two-stage identification, JSON extraction, retry, downscale ladder |
| `cache.py` | Append-only JSONL result cache, fsynced per image |
| `runner.py` | Batch execution, resume, per-file failure isolation |
| `report.py` | Raw table, scoring, calibration, name-quality checks |
| `encounters.py` | Burst clustering by capture time, from the XMP packet, never the pixels |
| `plugin_results.py` | The enrichment the Lightroom plugin gates on: burst agreement, range flag, encounter, quality and its rank within the burst (`melampus-id --plugin-out`) |
| `download.py` | The MLX model into the HuggingFace cache, resumed across runs, the progress protocol the plugin parses and the cancel marker it writes, its status and its removal; Ollama's model through its pull, list and delete endpoints, mapped onto the same protocol (`melampus-id --download-model`, `--model-status`, `--remove-model`, for the engine `--backend` names; docs/config.md § Downloading the model) |

---

## One seam for the model

Everything above `backend.py` talks to `VLMBackend`, an abstract class with a single
method: take an image path and a prompt, return text.

```python
class VLMBackend(ABC):
    @abstractmethod
    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion: ...
```

Six implementations exist, and `[model] backend` picks one (`providers.py`,
docs/config.md § `[model]`). `MLXBackend` runs Qwen3-VL locally on Apple Silicon.
`OllamaBackend` runs whatever vision model a local Ollama server holds, over its
documented chat endpoint with the standard library, which is the local path on
Windows and Linux. `AnthropicBackend` and `OpenAIBackend` are the cloud path,
built for the §6.6 escalation tail and reused as a primary on machines with no
local runtime. `CommandBackend` runs an installed command-line program once per
frame with the image path and the prompt in its arguments and reads the reply
from its stdout: a subscription CLI such as Claude Code or Codex is vision with
no API key (the templates for those two are cards #421 and #422; the seam knows
no program). `ScriptedBackend` returns canned responses, which is what lets the
pipeline tests cover parsing, validation, retry, caching and the downscale ladder
in under a second with no weights on disk. Each is a class here and no change
anywhere else: the prompts, the JSON extraction, the schema validation and the
corrective retry live above the seam and are the same whoever answers.

`MLXBackend` loads weights lazily, so `--help` does not pull 18 GB. Whether an
engine can run on this machine at all is `providers.detect_engines`' question,
answered before any image is read; an Ollama that is not running is refused
there with the address tried and where to install it, and a command that
`shutil.which` cannot find is refused the same way, naming it. The one failure
that stops a batch rather than being recorded on the frame is a command exiting
non-zero (`CommandFailed`): that is a broken engine, not a bad file, and every
frame would fail the same way.

---

## Two-stage identification

**Stage A — taxon routing.** A short prompt returning one of bird / fish / reptile /
amphibian / insect / arachnid / mammal / plant / fungus / none.

Two payoffs. Stage B can use a taxon-appropriate prompt — plumage and structure for
birds, scale and head shape for reptiles, inflorescence and leaf arrangement for plants
— which would be incoherent as a single generic instruction. And a frame routed to
`none` **short-circuits entirely**: there is no point asking a species question about an
empty frame, and skipping stage B makes those frames ~2.5× faster.

**Stage B — species identification.** A taxon-specialised prompt returning ranked
candidates, age/sex, count, behaviour, and an explicit abstention flag.

Both stages validate against Pydantic models. On failure the model gets **one**
corrective retry carrying the specific validation error, and then the image is recorded
`unprocessed`. It is never given a guess.

### Defensive parsing

Local VLMs are markedly less reliable at strict JSON than frontier models. `extract_json`
recovers from the three common shapes without a round trip: markdown fences, a prose
preamble, and a trailing summary sentence. It scans for a *balanced* object and is
string-aware, so braces inside a `reasoning` field do not break it.

### The downscale ladder

`mlx-vlm` 0.6.7 stops generating past roughly 2,100 prompt tokens (see
[troubleshooting.md](troubleshooting.md)). Vision tokens dominate that budget, so image
size is the lever.

`identify()` tries the configured `max_edge`, then each `fallback_edges` entry, until
generation succeeds. The successful size is recorded on the result, so degradation is
visible rather than silent. A fixed size alone is not sufficient — the threshold shifts
with aspect ratio.

---

## Metadata isolation

The rule is that only pixels reach the model. It is enforced at two structural
chokepoints rather than by asking call sites to remember.

**Pixels.** `images.staged_pixels` is a context manager that re-encodes each image to a
temporary file named exactly `image.jpg`, rebuilt via `Image.frombytes` from raw pixel
bytes. EXIF, XMP and IPTC cannot survive that round trip. The backend only ever receives
the staged path, so the original filename never appears in any argument.

**Prompts.** `prompts.render` raises on any substitution key outside
`ALLOWED_CONTEXT_KEYS`. A filename cannot be interpolated into a prompt even by mistake.

> **Open question for Stage 2.** CLAUDE.md §4.3 wants a season hint in the stage B
> prompt, because molt state and bloom timing are genuine identification signals. That
> hint derives from EXIF capture date, which is metadata. The allowlist exists so
> enabling it is a deliberate, visible decision rather than an accident — but whether to
> enable it at all should be confirmed, since it weakens the pixels-only guarantee.

---

## Caching and resume

Append-only JSONL, flushed and fsynced per result. A crash loses at most the image in
flight, and a truncated final line is skipped on reload rather than aborting.

**The key is not just the image.** It is a fingerprint over the image content hash plus
the model repo, prompt-set hash, image size, `max_tokens` and temperature. Keying on the
image alone was an early bug: editing a prompt and re-running silently re-served the
previous wording's answers, so a tuning pass would have looked like it worked while
measuring nothing.

Content hashing rather than path means renaming, moving or re-exporting a file still
hits the cache. Records predating fingerprinting carry an empty value and are still
honoured, so introducing the change discarded no prior work.

---

## Batch resilience

`run_batch` isolates failures per file: an exception is recorded on that image and the
batch continues. A multi-thousand-image run must not abort on one unreadable frame.

Checkpointing happens *before* anything else can fail — the cache write is the first
thing after identification returns.

Known gap: there is no per-image **timeout**. Exceptions are handled; a hang is not.

---

## Encounter-aware analysis

Wildlife shooting produces bursts. `encounters.py` groups frames by EXIF capture-time
proximity (`tools/cluster_encounters.py` is the corpus report over it), which recovers
the shooting encounters and unlocks three things that per-frame processing cannot do:

1. **A tractable review unit.** One judgement per encounter labels every frame in it, so
   43 human decisions cover 1,743 photographs.
2. **Accuracy without pseudo-replication.** See [evaluation.md](evaluation.md).
3. **A label-free stability measure.** Frames within an encounter show the same
   individual, so disagreement between them is unambiguous instability.

The development set was sampled *across* encounters rather than at random, because
random sampling from a burst corpus returns near-duplicates of the same few subjects.

The plugin's write gates reason per encounter too. `plugin_results.py` adds
`burst_agreement` (the share of a burst's frames that agree with its majority call),
`range_flag` (one GBIF lookup per encounter, in the month it was shot, against the
configured default location), `encounter`, and `quality` with `quality_rank` and
`encounter_frames` — quality is ranked *within* the burst, because absolute sharpness
is not comparable across subjects and culling is a within-burst question anyway.
`melampus-id --plugin-out` writes it in the same run as the identification, and
that is the one command the Lightroom plugin runs, against the executable that
ships inside its own folder (`MelampusAnalyze.lua`); `tools/make_plugin_results.py`
is a thin caller of the same module for a `--json-out` file already on disk.

---

## The Lightroom plugin

Same split as the Python side, for the same reason. All write decisions live in
`MelampusRules.lua`, a module that imports nothing from Lightroom, so CLAUDE.md §5.3's
safety rules are testable in a second by a local Lua interpreter rather than discoverable
only by damaging a real catalog. The Lightroom layer reads existing state into a plain
table, calls `planFor`, and applies the returned plan.

Two ordering constraints shape `MelampusImport.lua`. Everything async — file reading,
JSON parsing, reading photo state — completes in a first pass before any write
transaction opens, because file I/O yields and yielding inside `withWriteAccessDo` is
what produces "yielding is not allowed". And writes are chunked at 100 photos per
transaction so a cancel or crash keeps completed work.

SDK verification results, including the one §5.4.5 requirement that turns out to be
impossible, are recorded in [plugin.md](plugin.md).

## Reporting

Three independent views, deliberately not blended into a single score:

- **Raw table** — every photo, every candidate, every confidence. No summarising.
- **Scoring** — macro-averaged headline, per-species and per-taxon breakdowns, the
  confusion report, abstention rate, and a confidence calibration table.
- **Name quality** — malformed binomials, and common names carrying conflicting
  binomials, which proves an error without consulting a taxonomic authority.

The confusion report is the highest-value output: it says which discriminations to write
into the prompts, which is why the prompts are editable files.
