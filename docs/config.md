# Configuration

Every setting, with the reasoning behind its default. Config is data, never code —
nothing in the Python hardcodes a model name, threshold or weight.

Settings load from an optional TOML file passed with `--config`, layered over the
built-in defaults and the git-ignored `melampus.local.toml`. A user file need only
contain the keys it changes. `--no-local-config` leaves `melampus.local.toml`
unread, so `--config` alone, over the defaults, is the whole configuration: the
executable smoke tests use it to hand the CLI and the executable one synthetic
file, so a developer's own settings never decide whether the two agree.
Programmatic callers can also pass overrides directly to `load_config(...)`, which
is how the HTTP service in Stage 3 will accept per-request configuration.

```bash
melampus-id fixtures/ --config my-settings.toml
```

Before any `--config` file, `melampus.local.toml` is read if it exists — the
git-ignored place for keys and your home location. In a checkout it sits at the
repo root, next to the `.melampus_cache/` folder the caches default to. The
shipped executable unpacks itself into a temporary directory at every launch, so
there both resolve under the per-user data directory instead (card #436):
`~/Library/Application Support/Melampus/` on macOS, `%LOCALAPPDATA%\Melampus\`
on Windows, `$XDG_DATA_HOME/Melampus/` (or `~/.local/share/Melampus/`) elsewhere,
with the caches in its `cache/` subfolder (so `<data>/.melampus_cache/occurrence.json`
below is `…/Melampus/cache/occurrence.json` there). `--config` and `--cache` still
name any path explicitly.

```toml
# my-settings.toml — only what differs from the defaults
[model]
repo = "mlx-community/Qwen3-VL-32B-Instruct-8bit"

[image]
max_edge = 1280
```

`<repo>` in the defaults below is the checkout root; inside the shipped
executable (readme.md § Building the executable) it is the unpack directory,
where `prompts/` ships in the bundle. `<data>` is the checkout root; the
executable's equivalents are the per-user `cache/` files described above.

---

## `[model]`

| Key | Default | Why |
|---|---|---|
| `backend` | *(the first engine that can run here)* | Which engine answers. The engines are `mlx` (local, Apple Silicon only — the local-first choice), `ollama` (local, through an Ollama server: Windows, Linux, or a Mac that prefers it; `ollama_model` and `ollama_url` below), `openai`, and `claude` (the Anthropic API); `scripted` is the test fake, not an engine (answers nothing, needs no weights; it exists so the shipped executable can be smoke-tested — see readme.md § Building the executable). The Lightroom plugin's `engine` preference passes the same names as `--backend` (docs/plugin.md § The engine); unset, the plugin passes nothing and this setting decides. Left unset here too, the CLI runs detection (card #404) and takes the first engine that can run on this machine, in the order above: `mlx` on Apple Silicon, else `ollama` when a server answers at `ollama_url` (unset, Ollama's documented default `http://127.0.0.1:11434`), else `openai`. Asked for `ollama` with no server answering there, the run is refused before any image is read: the message names the address tried and where to install Ollama, exit 3, the way `mlx` is refused off Apple Silicon. `--detect-engines` (`melampus-id --detect-engines`, no folder needed) prints the same verdicts as JSON, one per engine with a plain-words reason: `needs Apple Silicon`, where to install Ollama, or which key variable a cloud engine needs. The refusal for an engine that cannot run here names the ones that can, from the same detection. CLAUDE.md §3 built the backend seam; making it a setting is what lets the same repo run on a machine with no local runtime at all (Windows). A cloud primary bills **every** frame, not just an escalated tail, so three guards apply: the CLI prints an estimate and asks before spending (`--yes` skips the question for non-interactive callers such as the plugin), `max_images` hard-caps the run regardless, and results go to their own cache file (`identifications-cloud.jsonl`) so a later local pass cannot silently overwrite answers that were paid for. When a cloud backend is selected, MLX-shaped defaults you have not overridden are retuned: `max_edge` 2048, no fallback ladder, `max_tokens` 1200, `routing_max_tokens` 900 — the same treatment escalation applies, for the same reasons. The estimate is priced by `escalation.input_usd_per_mtok` / `output_usd_per_mtok`; set them to your model's rates or the number is confidently wrong. |
| `repo` | `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` | CLAUDE.md §3 requires the model be a setting, not a hardcode. This is the MoE build named in the spec: 18.3 GB with roughly 3B active parameters, so it runs far faster than a dense model of similar quality. 128 GB of unified memory allows going considerably larger — see the table in the README. Used by the `mlx` backend only. |
| `name` | *(provider default)* | Cloud model name, for `backend = "claude"` or `"openai"`. Unset means the provider's default (`providers.DEFAULT_MODELS`) — vision model names age quickly, so treat that as a starting point. |
| `ollama_model` | `qwen3-vl:8b-instruct` | The model the `ollama` backend asks, as a tag from [ollama.com/library](https://ollama.com/library); it must take image input. The default is the Instruct build of the same family the `mlx` default uses, 6.1 GB, from [ollama.com/library/qwen3-vl/tags](https://ollama.com/library/qwen3-vl/tags); `qwen3-vl:30b-a3b-instruct` (20 GB) there is the Mac default's twin for a machine that can hold it. Not the library's bare `qwen3-vl` tag: that is the thinking build, which spends the token budget thinking before any JSON appears (the trap `escalation.routing_max_tokens` documents). The Lightroom plugin's Settings dialog has a Download button for it, which asks Ollama to pull it (§ Downloading the model, card #409), as does `melampus-id --download-model --backend ollama`; `ollama pull qwen3-vl:8b-instruct` does the same by hand. A model that is not there fails each frame with Ollama's own "not found" message. |
| `ollama_url` | *(unset)* | Where the `ollama` backend's server listens. Unset means Ollama's documented default, `http://127.0.0.1:11434` (Ollama's docs/faq.mdx: "Ollama binds 127.0.0.1 port 11434 by default"), written once as `providers.OLLAMA_URL`; set this for a server on another port or host. It is read the way a request uses it: `https://` is spoken as TLS with the certificate verified, and a path in front of the endpoints (a reverse proxy's `http://host/ollama`) is kept, so `/api/chat` and `/api/version` go on the end of the address as typed. One address, read one way: it is what detection probes for the default engine and `--detect-engines`, what the not-running refusal names, and what every request goes to. |
| `base_url` | *(unset)* | OpenAI-compatible endpoint override: OpenRouter, LM Studio, vLLM, a proxy. Turns the `openai` backend into a general escape hatch rather than one vendor's client. |
| `api_key` | *(unset)* | Cloud key for the primary backend. Never set it in tracked source — prefer `MELAMPUS_ANTHROPIC_KEY` / `MELAMPUS_OPENAI_KEY` (or the provider's own variable), or put it in the git-ignored `melampus.local.toml`. Stored as a `SecretStr` so a repr or traceback cannot leak it. |
| `effort` | `high` | Anthropic-only thinking effort for a cloud primary; ignored elsewhere. Same rationale as `escalation.effort`: these frames deserve the model actually thinking. |
| `timeout_seconds` | `180` | Per-request ceiling for a cloud primary, and for `ollama`. Long enough for a thinking model on a hard frame, short enough that a hung connection cannot stall a batch for minutes per image. Ollama loads the model on the first request, which counts against it; a frame that times out is recorded as an error and the batch goes on. |
| `max_images` | `200` | Cloud primary only; `mlx` ignores it. Same rationale as `escalation.max_images`: a run must not turn into an unexpected invoice, so the ceiling is low enough to notice and must be raised deliberately. Unlike escalation there is no most-uncertain-first ordering to make a truncated batch meaningful, so exceeding it refuses the whole run (use `--limit` to narrow instead). Applies even with `--yes` — the plugin passes `--yes`, and this cap is what bounds it. |
| `max_tokens` | `900` | Enough for a full identification with two or three candidates, each carrying a reasoning string. Too low truncates mid-JSON and forces a wasted retry. |
| `temperature` | `0.0` | `mlx` and `ollama`. Identification is a discrimination task, not a creative one. Deterministic decoding also makes re-runs reproducible, which matters when tuning prompts against the confusion report. |
| `routing_max_tokens` | `200` | Stage A returns three short fields. Capping it low keeps the cheap stage cheap. |

### Choosing a model

Prompt length interacts with model capability more sharply than expected. The taxon
prompts run to roughly 3,300 characters. Measured on `Qwen3-VL-2B-Instruct-4bit`:
prompts under ~1,200 characters return valid JSON, while prompts beyond ~1,600 cause
the model to emit an immediate EOS or degenerate into a repeated list marker. Small
models are therefore unsuitable for Stage B regardless of how the prompt is phrased.
If you switch to a smaller model and see a spike in `unprocessed`, this is the cause.

---

## Downloading the model

The flag `--download-model` (`melampus-id --download-model`, no folder
needed) fetches the picked engine's model: for `mlx`, `[model] repo`, or the
repo `--model` names, into the HuggingFace cache (`HF_HOME`, the same cache
`mlx` loads from); for `ollama`, it asks the Ollama server to pull
`[model] ollama_model` (§ The same flags for Ollama below). The engine is
`--backend` or `[model] backend`; with neither, `--model` means `mlx` (it
names a hub repo), else the first engine *with a model* that detection says
can run here, `mlx` then `ollama`, and `mlx` when neither can (its hub
download works on every platform), not a run's default, which may be a cloud
engine; `openai`, `claude` and `scripted` have no model to fetch and are
refused with exit 3 naming the two that have. The Lightroom plugin's download
button (card #408, and card #409 for Ollama) drives it, so what it prints on
stdout is a protocol, defined once in `download.py` (`Update`) and stable:

| Line | When |
|---|---|
| `progress <bytes_done> <bytes_total>` | One per chunk received (the hub library's 10 MiB), and one before any byte moves so the total is known at once. `bytes_total` is the whole model as the hub serves it: two files with the same bytes share one etag, so one blob in the cache, counted once; `bytes_done` counts what the cache already holds, complete files and the partial one being resumed included, so a re-run of a finished model prints one line with both equal. `bytes_done` never exceeds `bytes_total`: when a host answers the resume's Range request with the whole file instead, one update steps back by the partial's bytes before the file is counted from byte zero. |
| `done <path>` | Last line on success: the snapshot folder in the cache. The path is the rest of the line; it may hold spaces. |
| `cancelled` | Last line when a signal stopped it. |

Nothing else goes to stdout; errors and the hub library's own warnings go to
stderr, and a URL in either is named by its path alone: the hub serves a
weights file's bytes from its CDN at a signed URL, whose query is a credential
for that file, and neither the library's retry warning nor the failure message
carries it. Exit codes: **exit 0** once the model is complete (`done`); **exit 3**
on a failure, with a message on stderr naming the fix (the repo the hub does not
have, or an id that is not a repo id at all, a pasted hub URL say, so check
`[model] repo` or `--model`; a gated repo, so request access to
it on the hub and sign in with `hf auth login` or `HF_TOKEN`; the network, so
check it and re-run, a hub that accepts the connection and never answers
included, since every request to it is bounded by the hub library's own
metadata timeout, `HF_HUB_ETAG_TIMEOUT`, ten seconds;
a file whose bytes do not match the checksum the hub names for it, so its partial
is discarded and the re-run fetches it whole; a file that arrived at a size other
than the one the hub named, so its partial is kept and the re-run resumes it (the
message names the file and both sizes, never the hub library's own wording, which
after its own retry of a dropped connection names the file by the tail of its
URL); a hub whose answers are not a hub's,
an etag that is not a checksum or a commit that is not a hash, neither of which
is let become a path in the cache, so check `HF_ENDPOINT`; another run holding
the model, a download of it, an identification run loading it or a removal,
whose lock this run waits five seconds
for, so wait for it to finish and re-run; a cancel marker, below,
the command cannot remove, so remove it by hand);
**exit 4** when it was cancelled (`cancelled`), by a signal or by the cancel
marker below. The signals are SIGINT (Ctrl+C), SIGTERM and, on Windows,
Ctrl+Break: the download stops within the current chunk and leaves the partial
file in the cache as the hub's `<etag>.incomplete` blob, and the next run
**resumes** it, asking the hub for the rest by Range from the byte it has. A
failed run leaves the same partial file, so re-running after a network drop
resumes too.

### The cancel marker

The Lightroom plugin cannot signal the executable (`LrTasks.execute` blocks,
returns only the exit code, and the SDK kills nothing), so the download also
stops when a file named `download-cancel` appears beside the caches, under the
per-user data directory (`~/Library/Application Support/Melampus/cache/download-cancel`
on macOS, `%LOCALAPPDATA%\Melampus\cache\download-cancel` on Windows; in a
checkout `.melampus_cache/download-cancel`). It is looked for before each chunk
is counted, and its appearance ends the run exactly as a signal does:
`cancelled`, exit 4, the partial file kept. The command removes a stale marker
when it starts and the marker when it exits, whatever the outcome; a marker it
cannot remove (a folder at that path, say) is **exit 3** naming the path, on
start before the hub is asked, and on exit when the download completed (the
model is there; remove the marker by hand: until it is gone every
`--download-model` exits 3 naming it before the hub is asked). A cancellation
or failure already under way is the outcome reported, and the next run names
the marker. The name is
`download.CANCEL_MARKER`, the path `download.cancel_marker_path()`, and
`--model-status` reports it as `cancel_path`, so the plugin's Cancel button
writes where the executable looks without deriving the directory itself.

### `--model-status` and `--remove-model`

Two more flags that need no folder and act for the same engine (`[model]
repo` or `--model` for `mlx`; `ollama_model` for `ollama`, below), for the
Settings dialog's Download button (card #408). For `mlx`, both refuse an id
that is not a repo id (a pasted hub URL, a path) the way `--download-model`
does: **exit 3**, the message on stderr naming `[model] repo` or `--model`,
before the hub is asked anything. Both read the cache through the hub
library's scan of it, which walks every model folder in the shared cache,
other tools' models included; a folder the process cannot search or list
(the cache itself, when its parent cannot be searched; or, for the status,
the repo's own `blobs` folder it counts) is **exit 3** too, the message
naming the cache and the folder the OS named: check that folder's
permissions.

- `--model-status` prints one JSON object and exits 0:
  `{"repo", "installed", "bytes_total", "bytes_done", "path", "cancel_path"}`.
  `installed` means whole: the snapshot the cache's `refs/main` names (from
  the hub library's scan of the local cache) holds every file the hub's
  listing names that the model's load needs, each at the hub's size. The
  files the load needs are those matching mlx-vlm's own patterns
  (`*.json`, `*.safetensors`, `*.py`, `*.model`, `*.tiktoken`, `*.txt`,
  `*.jinja`, named once as `download.MODEL_FILE_PATTERNS`), which is what
  mlx-vlm's load fetches on first use: the `.gitattributes` the hub writes
  into every repo and the model card (the repo's readme) are listed, counted in
  `bytes_total`, and not needed, so a model the first identification run
  fetched reads installed. A download stopped while the
  snapshot was being laid out, this command's or the hub library's own, which
  mlx-vlm's load runs, leaves a snapshot with some of the files, which reads
  as not installed (its blobs still count in `bytes_done`, so the next
  download lays out the rest without fetching them again). When the hub
  cannot be reached nothing on the machine names the files the repo should
  hold, and `installed` is what the cache lays out: the snapshot `refs/main`
  names, whole as far as the cache knows. `path` is the installed snapshot
  folder, or null; `bytes_done` is
  every byte the cache holds for the repo, complete files and the partial one
  alike, which is what the next download starts from; `bytes_total` is the
  whole model from the hub's file listing, **null when the hub cannot be
  reached**, does not answer within 10 seconds (`download.STATUS_TIMEOUT`,
  the hub library's own request timeout: the dialog waits for this command
  as it opens, so a stalled connection ends here rather than in Lightroom),
  or answers with something that is not a hub's answer (a captive portal's
  page, a proxy's block page or JSON error, a listing whose files have no
  name or a name that is not a string, whose sizes are not non-negative
  integers of at most 2^53 (a string, a negative number, a fraction, or one
  larger than any file, `download.MAX_SIZE`, the largest integer the
  plugin's JSON decoder reads exactly; sizes that add up to more than it
  count the same), or
  whose dates or evaluation results the hub library cannot read: whatever
  the answer does wrong), which counts as the hub not reached, `installed`
  then what the cache lays out; the status never fails for the network
  (the button then says "size unknown"). Example, absent:
  `{"repo": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit", "installed": false, "bytes_total": 18300000000, "bytes_done": 0, "path": null, "cancel_path": "~/Library/Application Support/Melampus/cache/download-cancel"}` (the path is absolute).
- `--remove-model` deletes the repo from the cache, all or nothing from the
  cache's point of view: the repo's folder is first set aside within the
  cache as `<folder>.incomplete`, which the cache no longer lists as the
  model, then deleted through the hub library's own cache deletion (every
  revision, so the whole folder goes); it prints `removed <path>` (the
  folder the model was in) and exits 0. It is refused with **exit 3** and
  the reason on stderr when nothing is installed, or while another run
  holds the model: a download of it (it holds the model's own lock,
  `repo.lock` in the cache's `.locks` folder for the model, for the whole
  run, and the hub library's per-file lock on the file it is fetching; the
  removal takes the model's lock, then the per-file ones, itself and holds
  them until the deletion is done, so no download starts on it in between:
  one that tries is refused, or waits and starts from nothing once the
  model is gone), an identification run loading the model (it holds the
  same lock from the start of its load until the model is in memory,
  mlx-vlm's load fetching what the cache lacks the while; a load starting
  under a removal waits for it and then fetches the model from nothing),
  or another removal. The removal cannot tell which holds the lock, so the
  message names all three, the same sentence `--download-model` refuses
  with, and says to wait for it to finish (cancelling a download first),
  then remove.
  It also exits 3, the model untouched,
  when the repo's folder in the cache is a symbolic link (a model laid out
  on another disk and linked into the cache, which the status accepts as
  installed): nothing is deleted through a link, and the message names
  where the link points, which is where to remove the model; when a lock
  file the running-download check must open cannot be; when the set-aside
  name is taken by the folder an earlier refused removal left (see below:
  the message names it again, to delete by hand before removing again,
  and nothing moves until it is gone); and when the folder
  cannot be set aside (on Windows, while another program holds a file in
  it open). And it exits 3 when the set-aside folder is still
  there after the deletion: the hub library's deletion deletes what it can,
  logs a permission error at what it cannot and carries on rather than
  raising, so the model is gone from the cache (`--model-status` reads
  absent, a second `--remove-model` has nothing to remove) and the message
  names the set-aside folder: check its permissions, and on Windows that no
  other program holds a file in it open, and delete it by hand.

### The same flags for Ollama

With `--backend ollama` (what the plugin passes for its Ollama row; or
`[model] backend = "ollama"`), the same three flags act on the Ollama server
named by `ollama_url` (`[model] ollama_url`) and the model named by `ollama_model` (`[model] ollama_model`), through the
endpoints Ollama's own docs/api.md describes, with the standard library, as
the backend speaks its chat endpoint (card #409):

- `--download-model` is `/api/pull` (`POST`, § Pull a Model) with the model's
  name, its stream of JSON objects read line by line and mapped onto the
  protocol above: each layer line (`pulling <digest>` with `digest`, `total`
  and `completed`, the layers one after another) becomes
  `progress <bytes_done> <bytes_total>` summed over every layer seen so far,
  so the total climbs as layers appear (Ollama gives no whole size before
  the pull); `success` ends it with **`done <model>`** and exit 0, the model
  living in Ollama under that name. The other statuses (manifest, verifying,
  writing, removing) print nothing. An error object in the stream is
  **exit 3** with Ollama's words on stderr: an unknown model
  (`pull model manifest: file does not exist`) names the model and
  `[model] ollama_model`; no server answering is the backend's own
  not-running message, naming `[model] ollama_url`. A signal or the cancel
  marker ends it as for `mlx`, **exit 4** and `cancelled`, by closing the
  stream, which is how Ollama learns to stop; Ollama keeps the layers it has
  and the next pull resumes them by itself (the docs: "Cancelled pulls are
  resumed from where they left off"), so the next `progress` line starts
  from what was kept.
- `--model-status` asks `/api/tags` (`GET`, § List Local Models) whether the
  model is held: `installed` true with the listed `size` as `bytes_total` and
  `bytes_done` and the model's name as `path`; else false with `bytes_total`
  **null** (Ollama lists sizes for held models only, so the button says
  "size unknown" until the model is there) and `bytes_done` 0. A name
  without a tag matches Ollama's `<name>:latest`. With no server answering
  the model is reported absent, size unknown, exit 0, so Settings opens.
- `--remove-model` is `/api/delete` (`DELETE`, § Delete a Model) with the model's
  name: `removed <model>` and exit 0; exit 3 naming the model when Ollama
  does not hold it, or with the not-running message.

The tests prove all three against a fake Ollama on 127.0.0.1 that speaks
those endpoints and keeps what a cut-off pull had; no model is ever pulled.

The bytes move over plain HTTP, through the hub library's own file download
(its Range request, its size check, its per-file lock), and every finished file
is checked against the checksum the hub names in its etag (the sha256 of a
weights file, git's blob sha1 of a regular one) before it becomes a blob in
the cache, never through the Xet
transfer that stalls on some networks (docs/troubleshooting.md); the command
sets `HF_HUB_DISABLE_XET=1` for itself. It also sets
`HF_HUB_DISABLE_TELEMETRY=1` for itself (a value you set, or `DO_NOT_TRACK`,
stands): the hub library would otherwise ask the hub which AI coding agents
exist and name the one it runs under, and the torch version, in every
request; the command sends the hub nothing about the machine but the
requests the download needs. The commit `main` points at is
resolved once, at the start, and recorded in the cache's `refs/main`; every
file is fetched at that commit, so a branch that moves during the run changes
nothing. Once every file is in the cache and checked, the command lays out the
snapshot of that commit from those checked files alone, its pointers made by
the hub library's own helper exactly as `mlx` will look for them (where
symlinks are unavailable, Windows without developer mode, the helper copies
each file into the snapshot instead: the copy is made under a staging name
and renamed into place once whole, so a cancel or a full disk mid-copy leaves
nothing under the file's name, and a short copy an earlier run left is
replaced, never taken as complete); the hub is
asked nothing more, so what it answers after the plan (another etag for a
file, say) reaches no path in the cache. A file name the listing gives that
is a path (absolute, a drive, or traversing) is refused the same way an etag
that is not a checksum is.
`HF_ENDPOINT` points the command at another hub,
which is how the tests prove it against a fake on 127.0.0.1 without ever
fetching real weights. The user's hub token (`hf auth login`, or `HF_TOKEN`)
goes only to that hub's own origin, and only over `https://` or to a loopback
host (127.0.0.1, ::1, localhost): an `http://` hub on another machine gets
every request without it, rather than the token in cleartext on the wire.

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

## `[quality]`

CLAUDE.md §4.1. Every weight and threshold in the quality scorer lives here — the
composite is deliberately not opaque, and these are the knobs to turn when scores
cluster at one end of the range on your corpus. The commentary in
`service/melampus/config.py` records how each default was calibrated against real
frames; this table is the summary.

Measurement scale and focus:

| Key | Default | Why |
|---|---|---|
| `working_long_edge` | `1600` | Sharpness is scale-dependent, so everything is measured at one working size. |
| `focus_window` | `15` | Local window for the focus-energy map. |
| `focus_percentile` | `85.0` | Percentile of the focus map used for the frame-level focus score. |
| `region_percentile` | `99.9` | Subject sharpness reads p99.9, not the mean or p99: smooth bokeh is quiet while uniform softness is noisy, so only the extreme percentiles separate defocus from sharp — and p99.9 still finds the sharp bill and eye on a low-texture subject like a Snowy Egret where p99 lands on plumage. |

Subject detection:

| Key | Default | Why |
|---|---|---|
| `merge_dilate` | `9` | Dilation joining nearby salient blobs into one subject. |
| `min_blob_area_frac` | `0.002` | Blobs below this fraction of the frame are noise, not subjects. |
| `area_exponent` | `0.35` | How strongly blob area counts when picking the primary subject. |
| `centrality_strength` | `0.30` | Central blobs win ties — wildlife framing favours the middle. |
| `box_pad_frac` | `0.08` | Padding around the detected box so wingtips are not cropped out of the measurement. |
| `saliency_resize` | `64` | Saliency runs at thumbnail size; detail is not needed to find the subject. |
| `saliency_blur_sigma` | `2.5` | Smoothing before thresholding the saliency map. |

Mapping raw focus energy to a 0–100 score — **retune these first** if your
photographs cluster at one end of the range:

| Key | Default | Why |
|---|---|---|
| `knee_low` | `22.0` | Below this raw p99.9 value the score is 0. Sampled across a 194-frame corpus (p10=23.7, p50=47.4, p90=75.9) so the range spreads rather than pinning most frames at 100. |
| `knee_high` | `78.0` | At and above this the score is 100. |
| `size_reference_frac` | `0.08` | Subject-size fraction treated as the reference for sharpness comparability. |
| `size_gain_strength` | `0.25` | How much small subjects are compensated — a distant speck cannot score like a full-frame portrait without help. |
| `size_gain_max` | `1.35` | Cap on that compensation. |

Motion vs defocus (reported separately per §4.1 — a directional wingbeat is often
the point of the photograph):

| Key | Default | Why |
|---|---|---|
| `anisotropy_floor` | `0.15` | Below this the blur is treated as isotropic defocus. |
| `anisotropy_ceiling` | `0.55` | Above this it is confidently directional motion. |

Eye / catchlight detection (weighted heavily when found, per §4.1):

| Key | Default | Why |
|---|---|---|
| `search_percentile` | `99.0` | Catchlights live in the brightest sliver of the subject. |
| `min_absolute_brightness` | `200` | A catchlight is near-specular; dimmer bright spots are plumage. |
| `min_area_frac_of_subject` | `0.00005` | Lower bound — smaller is sensor noise. |
| `max_area_frac_of_subject` | `0.02` | Upper bound — larger is sky through wings, not an eye. |
| `min_circularity` | `0.55` | Catchlights are round; reflections on water are not. |
| `min_ring_contrast` | `45.0` | A real catchlight sits inside a dark iris ring. |
| `ring_dilate` | `5` | Ring sampled just outside the candidate blob. |
| `patch_radius_mult` | `6.0` | Sharpness is then measured on a patch this many blob-radii wide — the eye region, not just the highlight. |
| `min_eye_confidence` | `0.45` | Below this the eye result is discarded and subject sharpness is used instead. |

Exposure — raw numbers are always reported; a penalty only accrues past the
tolerance, because some clipping is always present on specular highlights and sky:

| Key | Default | Why |
|---|---|---|
| `highlight_threshold` | `254` | Pixel value counted as clipped white. |
| `shadow_threshold` | `1` | Pixel value counted as clipped black. |
| `highlight_tolerance_pct` | `0.5` | Clipping up to this percentage is free. |
| `shadow_tolerance_pct` | `0.5` | Same for shadows. |
| `highlight_full_penalty_pct` | `12.0` | Clipping at this level zeroes the exposure component. |
| `shadow_full_penalty_pct` | `12.0` | Same for shadows. |

Framing and the composite:

| Key | Default | Why |
|---|---|---|
| `edge_margin_frac` | `0.01` | Subjects inside this margin of the frame edge are flagged as clipped by the boundary. |
| `weight_eye_sharpness` | `0.40` | Dominates because eye sharpness is what a wildlife photographer actually culls on. |
| `weight_subject_sharpness` | `0.35` | The fallback signal when no eye is found. |
| `weight_focus` | `0.20` | Frame-level focus placement. |
| `weight_exposure` | `0.05` | Diagnostics carry the detail; the composite only nudges. |
| `weight_motion` | `0.00` | Near-neutral by design — directional blur is often desirable, so it informs the separate motion report, not the score. |

---

## `[occurrence]`

CLAUDE.md §4.3: candidates are re-scored against real occurrence records for the
photo's place and month, via the GBIF occurrence API (open, no key). **No candidate
is ever dropped** — a top pick with zero regional records is multiplied down and
`range_flag`ged for review, because it is either a model error or a genuinely
notable record, and both deserve eyes. Present-but-scarce species are demoted more
gently and marked notable. Re-ranking degrades gracefully: no GPS, no network, or a
non-organism profile (sport, fitness) simply skips it, with the reason recorded in
the result.

| Key | Default | Why |
|---|---|---|
| `enabled` | `true` | The highest-value accuracy feature (§4.3); on unless you are offline. |
| `ebird_token` | *(none)* | Reserved for the eBird species-list cross-check — accepted and kept out of logs (`SecretStr`), but **not queried yet**; GBIF is the only source today. Free token from ebird.org/api/keygen. Set it in `melampus.local.toml` or `MELAMPUS_EBIRD_TOKEN`, never in tracked source. |
| `default_latitude` | *(none)* | Used only for photos with no GPS of their own — per-photo GPS always wins. Unset, GPS-less photos skip re-ranking entirely. There is no sane universal default, so set your home patch here (in `melampus.local.toml`) if your camera does not embed coordinates. |
| `default_longitude` | *(none)* | Pairs with `default_latitude`; both must be set to take effect. |
| `default_location_name` | *(none)* | Human-readable label for reports; not used in queries. |
| `radius_km` | `50.0` | Comfortably covers a refuge and its surroundings without reaching into a different faunal region. |
| `cache_path` | `<data>/.melampus_cache/occurrence.json` | Lookups are cached keyed on species, rounded coordinates and month (§4.3 — most shots cluster in a handful of places, so hit rates are high). No TTL: occurrence data moves slowly; delete the file to refresh. |
| `notable_threshold` | `25` | Below this many regional records a species is present but scarce: demote gently and mark notable rather than treating it as absent. |
| `absent_penalty` | `0.15` | Multiplier on confidence for zero-record candidates. Ordinal, not a probability. |
| `notable_penalty` | `0.6` | Multiplier for scarce candidates — the pile worth looking at. |

---

## `[run]`

| Key | Default | Why |
|---|---|---|
| `profile` | `"wildlife"` | Which routing prompt runs Stage A: `wildlife` asks what organism this is, `sport` what activity. Separate profiles keep a footballer from being routed to `mammal` and asked for a species, and keep each prompt under the runtime's token ceiling. |
| `prompts_dir` | `<repo>/prompts` | Per-taxon prompt templates as editable files (CLAUDE.md §4.2). Point this elsewhere to A/B a prompt set without touching the installed package. |
| `cache_path` | `<data>/.melampus_cache/identifications.jsonl` | Append-only JSONL, fsynced after every image. |
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
| `provider` | `"claude"` | `claude` (the Anthropic API) or `openai`. For any other OpenAI-compatible endpoint — OpenRouter, LM Studio, vLLM, a proxy — use `openai` with `base_url`. |
| `base_url` | *(none)* | OpenAI-compatible endpoint override. Ignored by the Anthropic provider. |
| `model` | *(provider default)* | `claude-opus-5` or `gpt-5`. **Vision model names move faster than this file does** — check the provider's current listing and override with `--escalate-model`. |
| `api_key` | *(none)* | Never set this in tracked source. See below. |
| `effort` | `"high"` | Anthropic only. These are the frames the local model could not resolve, so thinking depth is where the money should go. |
| `max_tokens` | `1200` | Per reply (Stage B). |
| `routing_max_tokens` | `900` | Stage A budget. **Do not lower this to the local default of 200.** On a model where thinking is on by default, `max_tokens` caps thinking *and* output together, so 200 is consumed before any JSON appears and every frame fails twice while being billed. |
| `timeout_seconds` | `180.0` | Per-request ceiling. High-effort thinking on a hard frame is slow; a hung connection should still fail before the run does. |
| `max_edge` | `2048` | Long edge sent to the cloud. Higher than the local `1280`, because the 1280 ceiling exists to dodge an mlx-vlm token bug that does not apply here, and resolution is the cheapest lever left on a hard frame. |
| `confidence_below` | `0.80` | Escalate when the local top candidate scores under this. Ordinal, not calibrated — a ranking cut, not a probability. |
| `on_abstain` | `true` | Escalate frames the local model declined to call. |
| `on_range_flag` | `true` | Escalate frames whose top candidate does not occur locally (§4.3). |
| `max_images` | `200` | Hard ceiling per run, bounded 0–5000 by the schema. A mistyped flag should not become an unexpected invoice. When the cap bites, the most uncertain frames go first and the rest are counted and reported — never silently dropped. |
| `input_usd_per_mtok` | `5.0` | Estimate only. Defaults are Claude Opus 5's rate. |
| `output_usd_per_mtok` | `25.0` | **Change both when you change provider or model**, or the printed estimate will be confidently wrong. The CLI prints the rates alongside the dollars so the assumption is visible. |
| `cache_path` | `<data>/.melampus_cache/escalations.jsonl` | Cloud answers live in their own file. Merged into the local cache they would carry a foreign run fingerprint, and the next local pass would decide they were stale and quietly overwrite work you paid for. |

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

Install the provider SDK first, pinned from `service/uv.lock` like everything else:
`cloud` is the anthropic SDK, `openai` the openai one; drop the extra you will not
use. Neither is a dependency of the local pipeline, so a local-only install stays
local-only. `uv sync` installs exactly the extras named, so keep `--extra dev` here,
and re-running the README's `## Install` block (which names only `dev`) removes the
SDKs again:

```bash
VIRTUAL_ENV=.venv uv sync --project service --locked --extra dev --extra cloud --extra openai --active
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

- Rating breakpoints, flag thresholds, colour-label mapping (Stage 2 / 4)
- Confidence band thresholds for the High / Medium / Low policy in §4.4 (Stage 2)
- The eBird species-list cross-check — `occurrence.ebird_token` is accepted and
  stored safely, but no code queries eBird yet; GBIF is the only occurrence source
- Occurrence cache TTL (the cache currently persists until deleted)
- Keyword root and hierarchy style, per-field overwrite permissions, dry-run toggle,
  service autostart and idle-shutdown, smart-collection creation (Stages 3 and 4)
