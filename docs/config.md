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
below is `…/Melampus/cache/occurrence.json` there) and the Lightroom plugin's
log in its `logs/` subfolder (docs/plugin.md). `--config` and `--cache` still
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
| `backend` | *(the first engine that can run here)* | Which engine answers. The engines are `mlx` (local, Apple Silicon only — the local-first choice), `ollama` (local, through an Ollama server: Windows, Linux, or a Mac that prefers it; `ollama_model` and `ollama_url` below), `openai`, and `claude` (the Anthropic API); `command` runs an installed program named by `command` below, once per frame, and reads its stdout — how a subscription CLI becomes the engine with no key (card #420); `claude-code` is that seam configured for Claude Code (card #421; see *Claude Code* below), billing to the subscription it is signed in to; `codex` is the same seam configured for Codex CLI (card #422; see *Codex CLI* below), billing to the ChatGPT plan it is signed in to; both are in the plugin's picker (card #423), greyed until installed and signed in; `scripted` is the test fake, not an engine (answers nothing, needs no weights; it exists so the shipped executable can be smoke-tested — see readme.md § Building the executable). The Lightroom plugin's `engine` preference passes the same names as `--backend` (docs/plugin.md § The engine); unset, the plugin passes nothing and this setting decides. Left unset here too, the CLI runs detection (card #404) and takes the first engine that can run on this machine, in the order above: `mlx` on Apple Silicon, else `ollama` when a server answers at `ollama_url` (unset, Ollama's documented default `http://127.0.0.1:11434`), else `openai`. Asked for `ollama` with no server answering there, the run is refused before any image is read: the message names the address tried and where to install Ollama, exit 3, the way `mlx` is refused off Apple Silicon; asked for `command` with a program that is not installed or not on PATH, likewise, naming the program; asked for `claude-code` with Claude Code not installed or not signed in, likewise, naming where to install it or the command that signs in; `codex` the same, and a Codex plan at its usage limit stops the batch at the first reply, naming when the limit resets. A program that exits non-zero mid-run stops the batch at exit 3 with its exit code and the first lines of its stderr (a broken engine, not a bad frame); a timeout or empty stdout is recorded on that frame and the batch goes on. `--detect-engines` (`melampus-id --detect-engines`, no folder needed) prints the same verdicts as JSON, one per engine with a plain-words reason: `needs Apple Silicon`, where to install Ollama, which key variable a cloud engine needs, and after the four, whether Claude Code and Codex CLI are installed and signed in. The refusal for an engine that cannot run here names the ones that can, from the same detection. CLAUDE.md §3 built the backend seam; making it a setting is what lets the same repo run on a machine with no local runtime at all (Windows). A cloud primary bills **every** frame, not just an escalated tail, so three guards apply: the CLI prints an estimate and asks before spending (`--yes` skips the question for non-interactive callers such as the plugin), `max_images` hard-caps the run regardless, and results go to their own cache file (`identifications-cloud.jsonl`) so a later local pass cannot silently overwrite answers that were paid for. When a cloud backend is selected, MLX-shaped defaults you have not overridden are retuned: `max_edge` 2048, no fallback ladder, `max_tokens` 1200, `routing_max_tokens` 900 — the same treatment escalation applies, for the same reasons. The estimate is priced by `escalation.input_usd_per_mtok` / `output_usd_per_mtok`; set them to your model's rates or the number is confidently wrong. |
| `repo` | `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` | CLAUDE.md §3 requires the model be a setting, not a hardcode. This is the MoE build named in the spec: 18.3 GB with roughly 3B active parameters, so it runs far faster than a dense model of similar quality. 128 GB of unified memory allows going considerably larger — see the table in the README. Used by the `mlx` backend only. |
| `name` | *(provider default)* | Cloud model name, for `backend = "claude"` or `"openai"`. Unset means the provider's default (`providers.DEFAULT_MODELS`) — vision model names age quickly, so treat that as a starting point. |
| `ollama_model` | `qwen3-vl:8b-instruct` | The model the `ollama` backend asks, as a tag from [ollama.com/library](https://ollama.com/library); it must take image input. The default is the Instruct build of the same family the `mlx` default uses, 6.1 GB, from [ollama.com/library/qwen3-vl/tags](https://ollama.com/library/qwen3-vl/tags); `qwen3-vl:30b-a3b-instruct` (20 GB) there is the Mac default's twin for a machine that can hold it. Not the library's bare `qwen3-vl` tag: that is the thinking build, which spends the token budget thinking before any JSON appears (the trap `escalation.routing_max_tokens` documents). The Lightroom plugin's Settings dialog has a Download button for it, which asks Ollama to pull it (§ Downloading the model, card #409), as does `melampus-id --download-model --backend ollama`; `ollama pull qwen3-vl:8b-instruct` does the same by hand. A model that is not there fails each frame with Ollama's own "not found" message. |
| `ollama_url` | *(unset)* | Where the `ollama` backend's server listens. Unset means Ollama's documented default, `http://127.0.0.1:11434` (Ollama's docs/faq.mdx: "Ollama binds 127.0.0.1 port 11434 by default"), written once as `providers.OLLAMA_URL`; set this for a server on another port or host. One address: it is what detection probes for the default engine and `--detect-engines`, what the not-running refusal names, and what every request goes to. |
| `command` | *(unset)* | The program the `command` backend runs (card #420), one list element per argument, with `{image}` and `{prompt}` placeholders, for example `command = ["my-vlm", "--image", "{image}", "--prompt", "{prompt}"]`. It is an argv list, never a shell string: the prompt, spaces, quotes and newlines included, is one argument, and nothing is quoted or escaped. `{image}` is replaced by the absolute path of the staged, metadata-free JPEG (never the original file); `{prompt}` by the prompt in full. The program's stdout is the reply, parsed exactly as MLX's is; its stderr is kept for error messages only. A template lacking either placeholder is refused when the config loads. Unset by default: no program is assumed installed. With `backend = "claude-code"` or `"codex"` it is optional and replaces the built-in template (*Claude Code* and *Codex CLI* below). |
| `base_url` | *(unset)* | OpenAI-compatible endpoint override: OpenRouter, LM Studio, vLLM, a proxy. Turns the `openai` backend into a general escape hatch rather than one vendor's client. |
| `api_key` | *(unset)* | Cloud key for the primary backend. Never set it in tracked source — prefer `MELAMPUS_ANTHROPIC_KEY` / `MELAMPUS_OPENAI_KEY` (or the provider's own variable), or put it in the git-ignored `melampus.local.toml`. Stored as a `SecretStr` so a repr or traceback cannot leak it. |
| `effort` | `high` | Anthropic-only thinking effort for a cloud primary; ignored elsewhere. Same rationale as `escalation.effort`: these frames deserve the model actually thinking. |
| `timeout_seconds` | `180` | Per-request ceiling for a cloud primary, for `ollama`, and for `command` (the program is stopped and the frame recorded as an error when it runs past this). Long enough for a thinking model on a hard frame, short enough that a hung connection cannot stall a batch for minutes per image. Ollama loads the model on the first request, which counts against it; a frame that times out is recorded as an error and the batch goes on. |
| `max_images` | `200` | Cloud primary only; `mlx` ignores it. Same rationale as `escalation.max_images`: a run must not turn into an unexpected invoice, so the ceiling is low enough to notice and must be raised deliberately. Unlike escalation there is no most-uncertain-first ordering to make a truncated batch meaningful, so exceeding it refuses the whole run (use `--limit` to narrow instead). Applies even with `--yes` — the plugin passes `--yes`, and this cap is what bounds it. |
| `max_tokens` | `900` | Enough for a full identification with two or three candidates, each carrying a reasoning string. Too low truncates mid-JSON and forces a wasted retry. |
| `temperature` | `0.0` | `mlx` and `ollama`. Identification is a discrimination task, not a creative one. Deterministic decoding also makes re-runs reproducible, which matters when tuning prompts against the confusion report. |
| `routing_max_tokens` | `200` | Stage A returns three short fields. Capping it low keeps the cheap stage cheap. |

### Claude Code

`backend = "claude-code"` (card #421) is the `command` engine with Claude Code's
template built in: one place holds it, `providers.CLAUDE_CODE_COMMAND`, and this
is what it says, as the `[model] command` you would set to change it (a
different `--model`, say):

```toml
[model]
backend = "claude-code"
command = [
  "claude",
  "-p",
  "--output-format",
  "json",
  "--tools",
  "Read",
  "--allowedTools",
  "Read",
  "--permission-prompts",
  "none",
  "--no-session-persistence",
  "--strict-mcp-config",
  "--setting-sources",
  "user",
  "The photograph is the file {image}. Read it with the Read tool, then answer this about it:\n\n{prompt}"
]
```

Every flag is from `claude --help` (2.1.277) and Claude Code's own
documentation ([Run Claude Code programmatically](https://code.claude.com/docs/en/headless),
the [CLI reference](https://code.claude.com/docs/en/cli-reference)): `-p` prints
one reply and exits; `--output-format json` puts the reply in the result
object's `result` field, which is unwrapped before the shared JSON extraction
sees it; `--tools Read` leaves Claude Code only the tool that reads files (it
returns PNG and JPG as an image the model sees), and `--allowedTools Read`
pre-approves that tool everywhere, so the staged image, in a temporary folder
outside any working directory, is read without a permission prompt;
`--permission-prompts none` denies anything else that would wait for a person;
`--no-session-persistence` writes no transcript per frame; `--strict-mcp-config`
connects no MCP server; `--setting-sources user` loads no project or local
settings from wherever melampus was launched. The prompt is the last argument:
the staged image's path for the Read tool, then the pipeline's prompt in full.
Not `--bare`, which never reads the subscription login. On the committed
fixture the routing prompt came back in 5.6 s and two turns (one Read, one
answer), the JSON fenced, which the extraction already handles.

What to install: Claude Code, from [code.claude.com/docs/en/setup](https://code.claude.com/docs/en/setup),
so that `claude` is on the PATH melampus runs from. How to sign in:
`claude auth login` (the default, `--claudeai`, is the subscription). What it
costs: **every frame bills to that Claude subscription**, its rate limits
included, and no API key is read or needed here; nothing is charged per call,
so the cloud guards (the estimate, `max_images`, the cloud cache file) do not
apply. Detection (`--detect-engines`) reports `claude-code` as not installed
when nothing on PATH is called `claude`, as not signed in when
`claude auth status` exits non-zero (its documented, cheap check: no model
call), and otherwise as available, naming the account kind; a run asked for
`claude-code` is refused the same way before any image is read, exit 3. A
session that lapses mid-batch is caught at the first reply (Claude Code prints
`Not logged in` as its result, exit 1) and stops the batch at exit 3 with the
same sign-in pointer. On winpc Claude Code is not installed yet; the real run
there is card #424.

### Codex CLI

`backend = "codex"` (card #422) is the `command` engine with Codex CLI's
template built in, the second CLI behind the same seam for when Claude is at
its limit or you prefer it: one place holds it, `providers.CODEX_COMMAND`, and
this is what it says, as the `[model] command` you would set to change it (a
`-m` model, say):

```toml
[model]
backend = "codex"
command = [
  "codex",
  "exec",
  "--image",
  "{image}",
  "--json",
  "--ephemeral",
  "--skip-git-repo-check",
  "--ignore-user-config",
  "--sandbox",
  "read-only",
  "-c",
  'approval_policy="never"',
  "-c",
  "project_doc_max_bytes=0",
  "--color",
  "never",
  "{prompt}"
]
```

Every flag is from `codex exec --help` (0.154.0) and Codex's own
documentation ([Non-interactive mode](https://developers.openai.com/codex/non-interactive-mode),
the [CLI reference](https://developers.openai.com/codex/developer-commands?surface=cli),
[Image inputs](https://developers.openai.com/codex/image-inputs?surface=cli)):
`exec` runs "non-interactively" and prints the final message alone; `--image`
attaches the staged, metadata-free JPEG to the prompt ("PNG and JPEG"
accepted), and it comes first because the flag takes several files, so a
prompt placed right after it would be read as a second file (measured: the
prompt was then expected on stdin); `--json` makes stdout a JSONL event
stream, and the reply is the last `agent_message` in it, unwrapped before
the shared JSON extraction sees it; `--ephemeral` writes no session per
frame; `--skip-git-repo-check` lets it run from wherever melampus was
launched; `--ignore-user-config` loads no `~/.codex/config.toml`, so no MCP
server starts per frame and the run is the same on every machine (the
sign-in is still read); `--sandbox read-only` and
`-c approval_policy="never"` let the run proceed with nothing writable and
nobody to approve (`codex exec` 0.154.0 has no `--ask-for-approval` flag;
the config key is the same documented policy); `-c project_doc_max_bytes=0`
keeps the launch directory's `AGENTS.md` out of the prompt; `--color never`
keeps ANSI out of the stderr an error message quotes. The prompt is the last
argument, the pipeline's prompt in full; the image needs no mention.

What to install: Codex CLI, from [developers.openai.com/codex/cli](https://developers.openai.com/codex/cli),
so that `codex` is on the PATH melampus runs from. How to sign in:
`codex login` (with no flags, the ChatGPT plan). What it costs: **every frame
bills to that ChatGPT plan**, its usage limits included, and no API key is
read or needed here; nothing is charged per call, so the cloud guards (the
estimate, `max_images`, the cloud cache file) do not apply. Detection
(`--detect-engines`) reports `codex` as not installed when nothing on PATH is
called `codex`, as not signed in when `codex login status` exits non-zero
(its documented, cheap check: no model call), and otherwise as available,
naming the account kind; a run asked for `codex` is refused the same way
before any image is read, exit 3. The plan's usage limit is not knowable
without a model call (the status check does not report it, nor does
`codex doctor`), so detection does not try: a plan at its limit is caught
at the first reply (Codex fails the turn with "You've hit your usage limit
... try again at <time>", exit 1) and the batch stops at exit 3 naming the
limit and the reset time as Codex said it, with nothing cached; a session
that lapses mid-batch (a 401 in the stream) stops it the same way, naming
`codex login`. The one real run so far, on 2026-09-18, was exactly that
usage-limit reply on the committed fixture (the owner's plan was at its
limit until the next morning); the success path is covered by the fake
until a run after the reset proves it.

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
| `progress <bytes_done> <bytes_total>` | One per chunk received (the hub library's 10 MiB), and one before any byte moves so the total is known at once. `bytes_total` is the whole model; `bytes_done` counts what the cache already holds, complete files and the partial one being resumed included, so a re-run of a finished model prints one line with both equal. |
| `done <path>` | Last line on success: the snapshot folder in the cache. The path is the rest of the line; it may hold spaces. |
| `cancelled` | Last line when a signal stopped it. |

Nothing else goes to stdout; errors and the hub library's own warnings go to
stderr. Exit codes: **exit 0** once the model is complete (`done`); **exit 3**
on a failure, with a message on stderr naming the fix (the repo the hub does not
have, so check `[model] repo` or `--model`; the network, so check it and re-run);
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
when it starts and the marker when it exits, whatever the outcome. The name is
`download.CANCEL_MARKER`, the path `download.cancel_marker_path()`, and
`--model-status` reports it as `cancel_path`, so the plugin's Cancel button
writes where the executable looks without deriving the directory itself.

### `--model-status` and `--remove-model`

Two more flags that need no folder and act for the same engine (`[model]
repo` or `--model` for `mlx`; `ollama_model` for `ollama`, below), for the
Settings dialog's Download button (card #408):

- `--model-status` prints one JSON object and exits 0:
  `{"repo", "installed", "bytes_total", "bytes_done", "path", "cancel_path"}`.
  `installed` and `path` (the snapshot folder, or null) come from the hub
  library's scan of the local cache, without the network; `bytes_done` is
  every byte the cache holds for the repo, complete files and the partial one
  alike, which is what the next download starts from; `bytes_total` is the
  whole model from the hub's file listing, **null when the hub cannot be
  reached**, and the status never fails for the network being down (the
  button then says "size unknown"). Example, absent:
  `{"repo": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit", "installed": false, "bytes_total": 18300000000, "bytes_done": 0, "path": null, "cancel_path": "~/Library/Application Support/Melampus/cache/download-cancel"}` (the path is absolute).
- `--remove-model` deletes the repo from the cache through the hub library's
  own cache deletion (every revision, so the whole repo folder goes), prints
  `removed <path>` (that folder) and exits 0. It is refused with **exit 3**
  and the reason on stderr when nothing is installed, or while a download of
  the model is running (it holds the hub library's per-file lock the fetch
  takes): cancel the download first.

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
(its Range request, its size check, its per-file lock), never through the Xet
transfer that stalls on some networks (docs/troubleshooting.md); the command
sets `HF_HUB_DISABLE_XET=1` for itself. Once every file is in the cache the hub
library lays out the snapshot, the pointers and `refs/main` exactly as
`mlx` will look for them. `HF_ENDPOINT` points the command at another hub,
which is how the tests prove it against a fake on 127.0.0.1 without ever
fetching real weights.

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
