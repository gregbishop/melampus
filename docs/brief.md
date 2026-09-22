# melampus, project brief

Local AI species ID and quality triage for Lightroom Classic. For a selected
batch of photos it identifies the organism with ranked candidates and
calibrated confidence, scores technical quality (sharpness measured **on the
subject, not the frame**), re-ranks candidates against real occurrence data
using capture location and date, and writes the result into native LrC
ratings, flags, colour labels and keywords.

Named for the Greek seer who could understand the speech of animals — birds
especially. *meh-LAM-pus*.

All inference is local, on Apple Silicon, via MLX. No cloud dependency in the
default path.

The original 368-line build spec now lives in `docs/build-spec.md`. It is
still the reference for architecture and phasing; it is not a page to reload
every session.

## stack contract

- stack: python
- build: `.venv/bin/python tools/build_binary.py` — from the repo root; writes
  `dist/melampus`, the one-file executable, with MLX on Apple Silicon and
  without it elsewhere (`dist/melampus.exe` on Windows, where the local option
  is Ollama, the cloud engines the other; needs the `build` extra, see
  readme.md § Building the executable)
- test: `.venv/bin/python -m pytest` — from the repo root, locally; add
  `--build-binary` to build the executable first and smoke-test it
- test in CI: `uv sync --locked --extra dev --extra build --extra cloud --extra openai && uv run pytest -q --build-binary`
  — from `service/`, in `.github/workflows/ci.yml`, on macOS; this is the run
  that gates merges, so it builds the executable and smoke-tests it on every
  run; the `cloud` and `openai` extras are there so the executable carries
  both SDKs (card #434)
- build in CI: `uv sync --locked --extra dev --extra build --extra cloud --extra openai && uv run pytest -v -rs tests/test_binary.py tests/test_lua_plugin.py tests/test_package_plugin.py --build-binary`
  — from `service/`, in the same workflow's `build-windows` job on a Windows
  runner; builds `melampus.exe` the way the test command does, runs the
  executable's smoke tests against it, runs the plugin tests with the Lua the
  job installs (Lua 5.1, Lightroom's own) so the command the plugin builds for
  cmd.exe is run by cmd.exe against `melampus.exe`, runs the packaging tests
  so the Windows zip ships from a script tested on Windows (card #402), and
  uploads it as the `melampus-windows` artifact; both jobs then package the
  plugin zip through `tools/package_plugin.py`, and on a pushed `v*` tag the
  same workflow's `release` job attaches `Melampus-macOS.zip` and
  `Melampus-Windows.zip` to the GitHub release (card #402)
- lint: none adopted
- run: the service half, per `docs/architecture.md`

Two things here differ from every other python repo, both deliberately:

- **The lockfile is `service/uv.lock`, not at the repo root.** A tool that
  looks only at the root concludes there's no lockfile.
- **There are two test commands, and they collect the same tests.** Locally
  it is `.venv/bin/python -m pytest` from the repo root: `pytest.ini` pins
  `testpaths = service/tests` and excludes `_old/`, whose stale `melampus`
  package would otherwise shadow the real one on `sys.path`. CI runs
  `uv sync --locked --extra dev --extra build --extra cloud --extra openai && uv run pytest -q --build-binary`
  from `service/`, where `service/pyproject.toml` pins the same `tests/` path,
  because the runner has no `.venv` and `uv sync --locked` builds one from
  `service/uv.lock` (installing exactly the lockfile, and failing if it has
  drifted from `service/pyproject.toml`). CI always passes `--build-binary`,
  so the executable is built and smoke-tested on the run that gates merges;
  locally it is opt-in because the build takes a minute. The two runs differ
  only in skips: without `--build-binary` the executable's smoke tests skip
  unless an existing build is there to run against. `test_escalation.py`
  skips either way (with the anthropic and openai SDKs absent, the tests that
  need them; with the SDKs installed, as CI's `cloud` and `openai` extras do,
  those that assert their absence); in CI the corpus-backed tests in
  `test_quality.py` skip too, because `fixtures/` is gitignored and absent on
  the runner, as does the installed-checkout test in `test_docs.py`, because
  the runner has no on-purpose install outputs.

## the split, and why

Two processes with a thin boundary: a **Lua plugin** inside Lightroom Classic
(menu items, config dialog, reads GPS and capture date, exports JPEG
previews, writes catalog metadata) talking JSON over local HTTP to a
**python service** (MLX inference, sharpness/quality CV, range and season
re-ranking, occurrence API clients, SQLite result cache).

The point of the split is that the python side is independently testable
without launching Lightroom. Test it there first; the Lua side is thin glue.

## hard rules

- **Never destroy existing user metadata.** Ever. This is greg's photo
  catalogue, and the plugin writes into it. Additive changes only, and a way
  back for anything it does write.
- **No model downloads.** Weights arrive through a pinned fetch script for
  greg to run, never fetched by an agent.
- Tests never need the network or model weights.

## the macos CI deviation is correct

CI runs on `macos-latest`, not the house linux runner, because `mlx-vlm`
ships Apple Silicon wheels only — on anything else the tests would not run at
all. It installs uv and lua through brew (lua enables the plugin tests, which
self-skip without it, and `luac -p` syntax-checks every plugin file).

This is a recorded deviation, not drift. Don't standardize it away. It is
also, as of 2026-08-19, the only CI in the estate that actually runs.

## conventions

- Docs are lowercase: `readme.md`, `docs/config.md`.
- `_old/` is archived. Don't collect it, don't import from it, don't tidy it.
- Fixtures (`fixtures/`, `fixtures_full/`) are test data — large and
  intentional.
