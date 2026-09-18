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
- build: none
- test: `.venv/bin/python -m pytest` — from the repo root
- lint: none adopted
- run: the service half, per `docs/architecture.md`

Two things here differ from every other python repo, both deliberately:

- **The lockfile is `service/uv.lock`, not at the repo root.** A tool that
  looks only at the root concludes there's no lockfile.
- **The test command is `.venv/bin/python -m pytest`, not `uv run pytest`.**
  `pytest.ini` pins `testpaths = service/tests` and excludes `_old/`, whose
  stale `melampus` package would otherwise shadow the real one on `sys.path`.
  140 tests collect as of 2026-08-19.

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
