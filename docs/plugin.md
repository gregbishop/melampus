# The Lightroom Classic plugin

It analyses the selected photos with the executable that sits in its own folder
and writes the identifications into the catalog, so review happens inside
Lightroom's own grid and loupe rather than a separate window. It can also read
results from a JSON file produced elsewhere. There is no HTTP service.

Developed against **Lightroom Classic 15.4.1**.

---

## Install

```bash
# 1. Put the executable in the plugin folder: melampus on macOS, melampus.exe
#    on Windows (readme.md § Building the executable, or a release download).
cp dist/melampus plugin/Melampus.lrplugin/
```

The plugin runs that file, from its own folder, to analyse photos it has not seen:
identification and the enrichment (burst agreement, range flag, encounter,
quality) in one run, through `--plugin-out`. It never looks for a Python
environment; if the file is missing it says which folder should hold it and what
the file is called.

2. In Lightroom: **File → Plug-in Manager… → Add**, and select
   `plugin/Melampus.lrplugin`. It should report *Installed and running*.

3. **Library → Plug-in Extras → Melampus: Settings…** Leave **Dry Run** on. To
   import results produced elsewhere (`melampus-id … --plugin-out
   plugin_results.json`), Choose → select that file; otherwise leave it empty.

4. Select photos, then **Plug-in Extras → Melampus: Import Identifications…**
   It reports what it would change and writes nothing.

5. Satisfied? Turn off Dry Run in Settings and run again.

6. **Plug-in Extras → Melampus: Review Queue** builds the smart collections.

7. Open `Melampus > Needs Review`. In the Metadata panel switch the preset
   dropdown to **Melampus**. Set *Review Verdict* per photo in loupe; type into
   *Correct Species* when it is wrong. The collection empties as you work.

8. **Plug-in Extras → Melampus: Log Corrections** exports your judgements, which
   `tools/ingest_corrections.py` consumes to rescore everything.

Results come from exported JPEGs while the catalog holds raws, so matching is by
basename: `0A1A2475.jpg` finds `0A1A2475.CR3`. Verified 300/300 on this corpus.

---

## The engine

Where inference runs is the user's choice (card #403): a preference named
`engine`, one of `mlx`, `ollama`, `openai`, `claude`, in the plugin's
preferences beside `profile`, and since card #405 a picker in Settings under
**Where identification runs** (readme.md § Reviewing in Lightroom shows it,
`docs/settings-dialog.png`).

When the dialog opens it runs the executable beside the plugin once with
`--detect-engines` (card #404) and shows what it said: the four engines in
that order after *Let Melampus choose*, the ones that cannot run here greyed
with their reason under the picker, and when the reason names a web address
(Ollama's download page), a line that opens it in the browser.
`Rules.engineItems` turns the executable's JSON into those items, so the
dialog holds no engine knowledge of its own and the rules tests cover it
without Lightroom. Without the executable nothing is greyed and the note is
the missing-executable message; the dialog never fails to open.

`openai` and `claude` need an API key. Picking one shows a password field for
it. The key is kept through the SDK's `LrPasswords` (`store` / `retrieve` by
key string; the OS keychain on macOS), under the name of the variable the
executable reads, `MELAMPUS_OPENAI_KEY` or `MELAMPUS_ANTHROPIC_KEY`. It is
never in the preferences, never in `melampus.local.toml` or any other file,
and never logged. When a run starts, `MelampusAnalyze.lua` sets that variable
in the executable's environment for the picked engine only: `LrTasks.execute`
takes one shell line and nothing else, so the line begins `VAR='key'` (sh) or
`set "VAR=key" &&` (cmd.exe) ahead of the executable, and the log carries the
line with the value blanked. The key is not an argument of the executable, but
the shell line is the child's command line for the run's duration.

When the preference is set, `MelampusAnalyze.lua` passes it to the executable
as `--backend <engine>`, and the executable's own rules apply: `ollama` is
refused as not built yet (card #406), `openai` and `claude` need their key
(docs/config.md § `[model]`). When it is unset — the default, *Let Melampus
choose* — the command carries no `--backend` and the executable decides:
`[model] backend` in `melampus.local.toml`, else the first engine that can
run on this machine. A value that is not one of the four is refused before
anything runs, with the four named, so a stale preference never reaches the
shell.

---

## SDK verification

CLAUDE.md §5.1 and §5.2 mark several items VERIFY. Confirmed against the API
reference before any Lua was written:

| Item | Finding |
|---|---|
| GPS | `photo:getRawMetadata('gps')` returns a table `{ latitude, longitude }`, or nil |
| Capture date | `dateTimeOriginal` is **seconds since 2001-01-01 GMT** — not the Unix epoch. `dateTimeOriginalISO8601` returns a string, and is what this plugin uses to avoid an off-by-31-years bug. |
| Preview | `photo:requestJpegThumbnail(width, height, callback)` is **asynchronous**. The returned object must be retained until the callback fires or it is collected mid-flight. Requested sizes are minimums; a larger preview may come back. |
| Write access | `catalog:withWriteAccessDo(name, func)`. `catalog:withPrivateWriteAccessDo(func)` writes plugin-only fields **without touching the undo stack** — correct for our own metadata. |
| Custom metadata | Exactly three data types: `string`, `enum`, `URL`. |

### The one thing that is not achievable

§5.4.5 asks for the metadata panel to list alternate candidates as selectable, so
correction is one gesture. **It cannot work.** Enum values are fixed in `Info.lua`
at definition time and cannot vary per photo, and every photo's alternates differ.

What replaces it: a fixed `verdict` enum (`unreviewed / confirmed / wrong /
uncertain`) which *is* a closed set and so is legal, plus a free-text
`correction` field. Reviewing stays entirely inside grid and loupe, which is what
§5.4.3 actually cares about. The cost is that correcting an ID takes a dropdown
and some typing rather than one click.

---

## Why the code is shaped this way

**All write decisions live in `MelampusRules.lua`, which imports nothing from
Lightroom.** That is the whole testability strategy: the §5.3 safety rules are
verified in under a second by a local Lua interpreter, instead of being
discovered by damaging a real catalog. The Lightroom layer only reads existing
state into a plain table, calls `planFor`, and applies the returned plan.

**Everything async finishes before a write transaction opens.** File reading,
JSON parsing, and reading existing photo state all happen in a first pass;
writes happen in a second. File I/O yields, and yielding inside
`withWriteAccessDo` is what produces "yielding is not allowed" errors.

**Writes are chunked at 100 photos**, each chunk its own transaction, so a crash
or a cancel keeps completed work.

**The SDK ships no JSON library**, so `MelampusJson.lua` is a decode-only parser
written for Lua 5.1 — no goto, no integer division, no bitwise operators. It
returns `nil, message` on malformed input rather than raising, because a
truncated results file should produce a clear dialog, not a stack trace.

---

## Tests

```bash
.venv/bin/python -m pytest -q     # runs the Lua suites too
```

The Lua tests are driven from pytest so one command covers both languages, and
skip cleanly when no interpreter is present:

- **37 rules tests** — never-overwrite for ratings, labels and flags; dry-run;
  idempotency; force; auto-reject staying off; the confidence and burst-agreement
  gates; range-flag routing; abstention; keyword sanitisation; graceful handling
  of sparse records; the engine preference, every value and the default; the
  engine picker's items from detection, greyed states, reasons and links.
- **The settings dialog against the mock SDK** — the real `MelampusSettings.lua`
  executed: the picker's items and bindings, the greyed states from a fake
  detection, the Ollama link, the key field visible only for a cloud engine,
  the key landing in `LrPasswords` and nowhere else, the missing executable.
- **8 JSON tests** plus a parse of 1,093 real records.
- **`luac -p` over every plugin file**, which has already caught a real bug.

Lightroom itself runs **Lua 5.1**. A local run and the macOS CI job use
whatever Lua is installed (brew's, currently 5.5), so there these catch logic
errors only; the Windows CI job installs Lua 5.1, Lightroom's own, and runs the
same suites on it, so dialect differences are caught there.

---

## What is not tested

Honestly: every Lightroom API call. They are syntax-checked and never executed,
because they cannot run outside Lightroom. The likely failure points, in order:

1. **Smart collection criteria.** Built as `sdktext:net.gregbishop.melampus.verdict`.
   If collections come up empty despite photos carrying verdicts, this is why —
   and it fails **silently**.
2. **Metadata panel rendering.** If the "Melampus" preset does not appear in the
   Metadata panel dropdown, the field ID format in `MelampusTagset.lua` is wrong.
3. **Keyword tree creation.** `createKeyword` with `returnExisting = true` should
   make re-runs idempotent; unverified against a live catalog.

---

## Safety

Defaults, all enforced in `Rules.defaultSettings()` and covered by tests:

| Setting | Default | |
|---|---|---|
| `dryRun` | **on** | first run against any catalog writes nothing |
| `autoReject` | **off** | an automated reject pass feels destructive |
| `overwriteRating` / `overwriteLabel` / `overwriteFlags` | **off** | write only where empty |
| `force` | off | re-runs are a no-op |
| `writeRating` / `writeLabel` / `writeFlags` | off | these need quality scores, which are Stage 2 |

A species keyword is written only when the identification clears **every** gate:
not abstained, not range-flagged, confidence ≥ 0.90, and burst agreement ≥ 0.80.
Anything failing a gate gets `Melampus > Review > Needs ID` instead. A wrong
keyword is worse than no keyword.
