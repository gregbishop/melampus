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
| **1** | Local VLM species identification | **Working.** Run over a 1,743-frame corpus |
| **2** | Quality scoring + location/season re-ranking | **Working.** Subject-localised sharpness and GBIF re-ranking both in the pipeline |
| 3 | HTTP service + frozen binary | Not started. The CLI is the interface today |
| **4** | Lightroom Classic plugin | **Working.** Analyses and writes to a real catalog |
| — | Optional cloud escalation for the hard tail (§6.6) | **Working.** Off by default |

182 tests: 126 Python, 56 Lua. None need model weights or a network.

Stage 1 exists to answer one question before anything else gets built: *can a local
VLM identify species well enough to be worth wiring into a catalog?* The current
answer is **a qualified yes** — see [docs/evaluation.md](docs/evaluation.md) for what
the numbers do and don't support.

---

## Requirements

- **Apple Silicon Mac** for local inference. MLX is arm64-only. Developed on an
  M4 Max / 128 GB. (Windows and Linux work too — locally through Ollama, or
  with cloud inference; see § Windows.)
- **Python 3.12** — not 3.13+. The `mlx-vlm` dependency stack publishes wheels for
  3.12; 3.13 runs ahead of parts of it.
- ~20 GB of disk for the default model.

## Install

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv sync --project service --locked --extra dev --active
```

That installs exactly `service/uv.lock`, the pinned dependency set CI installs
from too; `--locked` fails rather than re-resolve if the lock has drifted from
`service/pyproject.toml`.

Model weights are **not bundled**. They download on first use into the standard
HuggingFace cache; the Lightroom plugin's Settings dialog has a Download button
for the model (docs/plugin.md § The engine); or fetch them ahead of time with
the command that button drives:

```bash
.venv/bin/melampus-id --download-model
```

It fetches `[model] repo` (the default, or `--model` names another) into the
HuggingFace cache, prints `progress <bytes done> <bytes total>` as it goes and
`done <path>` at the end, and resumes where it stopped if interrupted (Ctrl+C
prints `cancelled`, exit 4); the format, the exit codes and what it does about
the Xet stall are in [docs/config.md](docs/config.md) § Downloading the model.
Some networks stall the HuggingFace Xet transfer — see
[docs/troubleshooting.md](docs/troubleshooting.md); the command downloads over
plain HTTP, so it is not affected.

## Windows (Ollama, or cloud inference)

There is no MLX on Windows — it is Apple-Silicon-only — so on Windows (and
Linux, and a Mac that prefers it) the primary backend is one of two things.

**Local, through Ollama.** Install Ollama from [ollama.com/download](https://ollama.com/download),
pull a vision model (`ollama pull qwen3-vl:8b-instruct`, the default; see
`ollama_model` in [docs/config.md](docs/config.md) § `[model]`; the
Lightroom plugin's Settings dialog has a Download button that asks Ollama to
pull it for you, docs/plugin.md § The engine), and the
`ollama` engine talks to it at Ollama's default address, `http://127.0.0.1:11434`
(`ollama_url` names another). Same prompts, same schema validation, same
corrective retry as the Mac's MLX path, and like it nothing leaves the
machine and nothing is billed. With no `--backend` and no `[model] backend`,
a running Ollama is what the CLI picks on a machine without Apple Silicon.
If Ollama is not running, the run stops before any image is read and says
so, naming the address it tried and where to install Ollama.

**A subscription CLI.** An installed command-line program can be the engine
too: `backend = "command"` runs the program `[model] command` names once per
frame with the image path and the prompt in its arguments and reads the reply
from its stdout, so a CLI that bills to a subscription rather than per call,
such as Claude Code or Codex CLI, is vision with no API key
([docs/config.md](docs/config.md) § `[model]`). The seam is generic. For
Claude Code it comes configured as the `claude-code` engine:
`backend = "claude-code"` (or `--backend claude-code`) runs `claude` in its
print mode with the built-in template docs/config.md § Claude Code quotes, on
whatever plan Claude Code is signed in to — install it from [code.claude.com/docs/en/setup](https://code.claude.com/docs/en/setup),
sign in with `claude auth login`, and every frame bills to that
subscription, not to an API key (card #421). For Codex CLI it is the `codex`
engine: `backend = "codex"` (or `--backend codex`) runs `codex exec` with the
image attached and the built-in template docs/config.md § Codex CLI quotes,
on the ChatGPT plan Codex is signed in to — install it from [developers.openai.com/codex/cli](https://developers.openai.com/codex/cli),
sign in with `codex login`, and every frame bills to that plan (card #422).
Both appear in the Lightroom plugin's engine picker, greyed until installed
and signed in, with no key field (card #423). A program that is not
installed is refused before any image is read, naming it; a CLI not signed
in, likewise, naming the sign-in command; a Codex plan at its usage limit
stops the batch at the first reply, naming when the limit resets.

**Cloud.** The same two backends the Mac uses for escalation, promoted to
answering everything. Same prompts, same schema validation, same corrective
retry; the only difference is who runs the model. Be aware of what that
trades away: **every analysed frame leaves the machine and is billed**,
where the local paths send nothing anywhere. Three guards keep that
predictable: the CLI prints a cost estimate and asks before spending (the
plugin passes `--yes` because it cannot ask — its CLI output, estimate
included, is written to `melampus-cli.log` in the OS temp directory);
`model.max_images` (default 200) hard-caps any single run, `--yes` or not;
and cloud results live in their own cache file so a later local pass cannot
overwrite answers you paid for. Set `escalation.input_usd_per_mtok` /
`output_usd_per_mtok` to your model's rates so the estimate means something.

Install (PowerShell, from the repo folder; this installs exactly
`service/uv.lock`, whose `mlx-vlm` entry is marked Apple-Silicon-only and so
is skipped on Windows; drop `--extra cloud` or `--extra openai` if you'll only
ever use the other provider):

```powershell
uv venv --python 3.12 .venv
$env:VIRTUAL_ENV = ".venv"; uv sync --project service --locked --extra dev --extra cloud --extra openai --active
```

Configure the backend in `melampus.local.toml` (git-ignored):

```toml
[model]
backend = "ollama"   # local; ollama_model and ollama_url change the defaults
# backend = "claude" # or "openai"; add base_url for any compatible endpoint
```

with, for a cloud backend, `MELAMPUS_ANTHROPIC_KEY` (or `MELAMPUS_OPENAI_KEY`)
set in your environment. Then everything works as on the Mac, plugin included:

```powershell
.venv\Scripts\melampus-id.exe fixtures\ --limit 3
```

One-off runs can skip the config file: `--backend ollama`, `--backend claude`,
`--backend claude-code`, `--backend codex`, `--backend command` (with
`[model] command` set).
With no `--backend` and no `[model] backend`, the CLI takes the first engine
that can run on this machine, in the order mlx, ollama, openai, claude
(docs/config.md § `[model]`); `--detect-engines` (`melampus-id
--detect-engines`, no folder needed) prints those verdicts as JSON, each with
the reason an engine cannot run here or what it needs, and after the four,
whether Claude Code and Codex CLI are installed and signed in. The Lightroom plugin's `engine` preference (docs/plugin.md § The
engine) passes the same name as `--backend`, and left unset it defers to the
backend and key above, else to that detection.

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

Three tests treat this as load-bearing:
`test_staging_strips_every_metadata_channel` builds a file carrying EXIF (maker, model,
artist, caption, GPS IFD), an XMP packet with a keyword, a JFIF comment and an ICC
profile, asserts the fixture really carries them, then asserts none survive staging;
`test_backend_never_receives_original_filename` and
`test_prompt_rejects_unapproved_context` cover the other two routes in.

Verified end to end: inference runs correctly with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

---

## Optional: a cloud second opinion on the hard tail

Everything above runs locally and stays local. The one exception is opt-in, and it
exists because the local model's failure mode is concentrated: it is right about most
frames and uncertain about a few hundred. That tail is small enough to be worth a
frontier model, and only the tail is sent.

```bash
# What would be sent, and roughly what it would cost. Needs no key. Sends nothing.
melampus-id fixtures --report-only --escalate-dry-run
#   would escalate 3 frame(s) to claude/claude-opus-5, est. $0.15 ...

export MELAMPUS_ANTHROPIC_KEY=sk-ant-...
melampus-id fixtures --escalate --escalate-max 50

# Or OpenAI, or anything speaking its chat-completions shape
melampus-id fixtures --escalate --escalate-provider openai --escalate-model gpt-5
melampus-id fixtures --escalate --escalate-provider openai \
  --escalate-base-url https://openrouter.ai/api/v1
```

Off by default, and additionally requires a key — two deliberate acts before a
photograph leaves the machine. A frame is escalated when the local model abstained,
scored its top candidate below `confidence_below`, failed outright, or named something
that does not occur locally. Confident empty frames are never escalated. There is a
hard per-run cap, and when it bites the most uncertain frames go first and the rest are
reported rather than silently dropped.

The same pixels-only guarantee applies: escalated frames go through the identical
staging path, so no filename or EXIF travels with them. Cloud answers are written to
their own cache file and never overwrite local ones, and each records what the local
model had said — which is what makes "is this worth paying for?" a measurable
agreement rate instead of an impression.

Install the provider SDK you want, pinned from `service/uv.lock` like everything
else: `cloud` is the anthropic SDK, `openai` the openai one; drop the extra you
will not use. Neither is a dependency of the local pipeline. `uv sync` installs
exactly the extras named, so keep `--extra dev` here, and note that re-running
the `## Install` block above (which names only `dev`) removes the SDKs again:

```bash
VIRTUAL_ENV=.venv uv sync --project service --locked --extra dev --extra cloud --extra openai --active
```

Full settings in [docs/config.md](docs/config.md#escalation--optional-cloud-second-opinion).

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

182 tests (126 Python, 56 Lua), no model weights required — everything runs against a scripted backend, so
parsing, validation, retry, caching, the downscale ladder and the no-leak guarantee are
all verifiable in under a second. The plugin's Lua suites run from the same command,
skipping cleanly if no Lua interpreter is installed.

The executable's smoke tests (`service/tests/test_binary.py`) run against
`dist/melampus` when it exists and skip when it does not. To build it first and
run everything, which takes about a minute (CI always does this):

```bash
.venv/bin/python -m pytest -q --build-binary
```

---

## Building the executable

The service ships to users as one file, `dist/melampus`, so they install neither
Python nor uv. It carries the Python runtime, the service, the prompts, the
anthropic and openai SDKs (`--backend claude` and `--backend openai` work on
every platform; with no key set the executable asks for one) and, on Apple
Silicon, the MLX runtime; model weights are not bundled and come from the
HuggingFace cache as before. PyInstaller bundles what the build venv has, so the
sync names every extra the executable needs: `build` for the pinned PyInstaller,
`cloud` and `openai` for the SDKs.

```bash
# once: the pinned PyInstaller and both SDKs, from the same lockfile as everything else
VIRTUAL_ENV=.venv uv sync --project service --locked --extra dev --extra build --extra cloud --extra openai --active
.venv/bin/python tools/build_binary.py    # writes dist/melampus, ~200 MB, about a minute
```

The build is a PyInstaller one-file bundle, which unpacks itself to a temporary
directory at every launch (a few seconds). `dist/` and `build/` are git-ignored.
PyInstaller's own cache, which it normally keeps per user and shares between
every checkout on the machine, goes under `build/pyinstaller-config/` here so two
checkouts can build at once (set `PYINSTALLER_CONFIG_DIR` yourself to choose
another; a build that fails on a corrupt cache names the directory to delete).
Try it without any Python on the path — the scripted backend needs no weights:

```bash
env -i PATH=/nonexistent HOME="$HOME" dist/melampus fixtures/ --backend scripted --limit 1
```

The executable keeps its results and reads its local config under the per-user
data directory, not the unpack directory that is deleted at exit:
`~/Library/Application Support/Melampus/` on macOS, `%LOCALAPPDATA%\Melampus\`
on Windows, `$XDG_DATA_HOME/Melampus/` elsewhere, with the caches in its
`cache/` subfolder (details in [docs/config.md](docs/config.md)). `--cache` and
`--config` override both, as they do for the CLI.

### Building on Windows

The same script on Windows writes `dist\melampus.exe`, which carries everything
but MLX: there is no MLX there, so `--backend` (or `[model] backend`) selects
Ollama or a cloud provider, exactly as in § Windows above. Asked for `--backend
mlx`, it says MLX needs Apple Silicon and names the backends that do work;
asked for `--backend ollama` with no server answering, it says so and where
to install Ollama.
CI builds it on every pull request (the `build-windows` job in
`.github/workflows/ci.yml`), smoke-tests it against the committed fixture frame
in `service/tests/fixtures/`, and uploads it as the `melampus-windows` artifact.
To build it yourself (PowerShell, from the repo folder):

```powershell
uv venv --python 3.12 .venv
$env:VIRTUAL_ENV = ".venv"; uv sync --project service --locked --extra dev --extra build --extra cloud --extra openai --active
.venv\Scripts\python.exe tools\build_binary.py    # writes dist\melampus.exe
```

---

## Reviewing in Lightroom

The plugin lives in `plugin/Melampus.lrplugin` and ships with the executable
inside its folder. Selected photos it has not seen are exported as previews and
run through that executable, `melampus` on macOS or `melampus.exe` on Windows,
with `--plugin-out`; the results are written into the catalog, so review happens
in Lightroom's own grid and loupe. No Python environment is involved. Dry run is
on by default and nothing the user set is ever overwritten. Install steps and the
SDK verification are in [docs/plugin.md](docs/plugin.md).

**From a release.** Each tagged release on the
[Releases page](https://github.com/gregbishop/melampus/releases) carries one
zip per platform, `Melampus-macOS.zip` and `Melampus-Windows.zip`, holding the
plugin folder with the executable already inside it. Unpack it (double-click,
or `unzip Melampus-macOS.zip`), keep the `Melampus.lrplugin` folder somewhere
it can stay, then in Lightroom: **File → Plug-in Manager → Add** and select
that folder. It should report *Installed and running*; **Library → Plug-in
Extras → Melampus: Settings…** opens the settings, where **Where
identification runs** picks the engine (mlx, ollama, openai, claude,
claude-code, codex; the ones this machine cannot run are greyed with the
reason) and takes the API key for a cloud engine, kept in your keychain; a
subscription CLI takes none and says what it bills to:

![The Melampus settings dialog, with the engine picker](docs/settings-dialog.png)

The release workflow
(`.github/workflows/release.yml`) builds both zips with the same commands CI
runs, through `tools/package_plugin.py`, on every pushed `v*` tag. The
executable is not yet signed or notarized (card #438).

**From source.**

```bash
# 1. the executable goes in the plugin folder (built as in § Building the
#    executable)
cp dist/melampus plugin/Melampus.lrplugin/     # dist\melampus.exe on Windows
# 2. Lightroom -> File -> Plug-in Manager -> Add -> plugin/Melampus.lrplugin
```

If the file is not there, the plugin says so, naming the folder and the file it
expects. The executable reads `melampus.local.toml` from the per-user data
directory (see [docs/config.md](docs/config.md)); on Windows that is where
`[model] backend` selects a cloud provider.

`--plugin-out` writes the results enriched with burst agreement, the range flag,
the encounter, and quality with its rank within the burst; the range check needs a
default location in `melampus.local.toml` and is the only step that uses the
network. The same flag works from the CLI (`.venv/bin/melampus-id fixtures_full
--plugin-out plugin_results.json`) to produce a results file the plugin can read
through Settings; `tools/make_plugin_results.py` is a thin caller of the same code
for a `--json-out` file already on disk.

## Layout

```
service/melampus/     Python package. Library first; the CLI is a thin shell over it
  backend.py          The model call, behind one swappable interface
  identify.py         Two-stage identification, defensive parsing, retry
  images.py           Pixel staging — and the metadata-leak enforcement point
  cache.py            Content-hash result cache, resumable, checkpointed
  report.py           Raw table, scoring, calibration, name-quality checks
  encounters.py       Burst clustering by capture time
  plugin_results.py   The enrichment the Lightroom plugin reads (--plugin-out)
prompts/              Editable per-taxon prompt templates
tools/                Corpus utilities: clustering, dev split, review sheet, ingest
plugin/               Lightroom Classic plugin, plus its dependency-free Lua tests
docs/                 Architecture, evaluation methodology, config, troubleshooting
fixtures/             Small development set
fixtures_full/        Full validation corpus
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit and why
- [docs/evaluation.md](docs/evaluation.md) — what the accuracy numbers mean, and don't
- [docs/config.md](docs/config.md) — every setting, with rationale
- [docs/plugin.md](docs/plugin.md) — the Lightroom plugin: install, SDK findings, safety
- [docs/troubleshooting.md](docs/troubleshooting.md) — known failure modes

---

## License

MIT — see [LICENSE](LICENSE).

The photo corpus is **not** part of this repository and is not covered by that
licence. `fixtures/` and `fixtures_full/` are git-ignored, and no image has ever been
committed.
