"""Run the Lua plugin test suites from pytest, so one command covers everything.

The plugin's decision logic lives in dependency-free Lua modules precisely so it
can be tested without Lightroom. Skips cleanly if no interpreter is installed.
Lightroom itself runs Lua 5.1. The macOS job's Lua is whatever brew installs,
so there this catches logic errors only; the Windows job's is Lua 5.1,
Lightroom's own, so there it catches dialect errors as well.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import PHOTO

from melampus import providers
from test_binary import no_python_environment, per_user_config

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin" / "Melampus.lrplugin"
TESTS = REPO / "plugin" / "tests"

pytestmark = pytest.mark.skipif(shutil.which("lua") is None, reason="lua not installed")

# The mock keeps its temp directory through sh (mktemp -d, mkdir -p, ls,
# rm -rf), which cmd.exe does not speak; the suites that fake a macOS
# Lightroom run on the macOS runner. The Windows runner runs what it can
# run as itself: the pure suites, luac, and the plugin's own command line.
needs_sh = pytest.mark.skipif(
    sys.platform == "win32", reason="the mock keeps its temp directory through sh, which cmd.exe cannot run")


def run_lua(script: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    # Forward slashes: the paths land in a Lua string literal, where a
    # backslash starts an escape, and Windows opens either kind.
    return subprocess.run(
        ["lua", "-e", f'package.path="{TESTS.as_posix()}/?.lua;{PLUGIN.as_posix()}/?.lua;"..package.path',
         str(script)],
        capture_output=True, text=True, cwd=TESTS, env=env,
    )


def as_the_shell_receives_it(path: Path | str) -> str:
    """A path or a word as an argument on the line LrTasks.execute hands to the
    shell:
    double-quoted for cmd.exe (a Windows filename cannot hold a double quote),
    single-quoted for sh with an apostrophe closed, escaped and reopened.
    Spelled here on its own, so the plugin's quote() is checked, not
    repeated."""
    if sys.platform == "win32":
        return f'"{path}"'
    return "'" + str(path).replace("'", "'\\''") + "'"


def run_lua_suite(script: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run one of the plugin's Lua test suites and assert its verdict: the
    interpreter exited 0 and the suite reported 0 failed."""
    proc = run_lua(script, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed" in proc.stdout, proc.stdout
    return proc


def assert_scripted_results(results: Path) -> None:
    """Assert the enriched results the executable writes on the committed
    fixture under the scripted backend: one row, for PHOTO. The scripted fake
    answers nothing, so the row has no identification and therefore no
    burst_agreement (that is agreement between calls); every other enrichment
    field is scored from the pixels and the capture times."""
    from melampus.plugin_results import PLUGIN_FIELDS

    rows = json.loads(results.read_text(encoding="utf-8"))
    assert [r["file"] for r in rows] == [PHOTO]
    expected = set(PLUGIN_FIELDS) - {"burst_agreement"}
    assert expected <= set(rows[0]), f"missing {expected - set(rows[0])}"
    assert 0 < rows[0]["quality"] <= 100, "quality was not scored on the pixels"


def run_as_lightroom_would(command: str, **kwargs) -> subprocess.CompletedProcess:
    """Hand the line to the shell LrTasks.execute hands it to: `cmd.exe /c`
    on Windows, as the C runtime's system() does, and `sh -c` elsewhere.

    Not shell=True on Windows: Python wraps the line in a pair of quotes of
    its own before cmd.exe sees it, and cmd.exe strips only that pair, so
    the pair the plugin wrapped the line in would still be on it."""
    if sys.platform == "win32":
        return subprocess.run(f'{os.environ["COMSPEC"]} /c {command}', **kwargs)
    return subprocess.run(command, shell=True, **kwargs)


def test_write_rules():
    """CLAUDE.md §5.3 safety rules: no overwrites, dry run, idempotency."""
    run_lua_suite(TESTS / "test_rules.lua")


@needs_sh
def test_import_runs_against_a_mock_lightroom():
    """Executes the real plugin files against a mock SDK: MelampusImport.lua
    end to end, and MelampusAnalyze.lua and MelampusSettings.lua loaded fresh
    under it, on a fake macOS and a fake Windows Lightroom.

    Catches what unit tests could not: keywords failing to attach, ratings not
    written, overwrite protection, dry run writing nothing, idempotency, file
    I/O inside a write gate, the command built for the executable beside the
    plugin, and what the dialogs say.

    It does NOT reproduce every real SDK behaviour — see docs/plugin.md for
    what remains unverified.
    """
    run_lua_suite(TESTS / "test_import_integration.lua")


def test_settings_dialog_against_a_mock_lightroom():
    """Card #405: executes the real MelampusSettings.lua against the mock SDK.
    The engine picker lists the four engines in order with the ones detection
    says cannot run here greyed and their reasons shown; the Ollama link is
    there exactly when ollama is unavailable; the API key field shows only for
    the picked cloud engine and stores through LrPasswords, never the
    preferences, a file, or the log; a missing executable greys nothing."""
    proc = run_lua(TESTS / "test_settings_dialog.lua")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed" in proc.stdout, proc.stdout


def test_json_decoder():
    run_lua_suite(TESTS / "test_json.lua")


def test_the_plugin_names_the_engines_the_cli_accepts(tmp_path: Path):
    """Card #403: the four engine names are spelled once per language, in
    `Rules.ENGINES` for the plugin and `providers.BACKEND_CHOICES` for the
    CLI, and this is what binds them: the plugin's list, read through lua, is
    the CLI's list without the offline test fake, in the same order. A rename
    on either side fails here rather than as a usage error the user never
    sees."""
    script = tmp_path / "engines.lua"
    script.write_text(
        "local Rules = require('MelampusRules')\n"
        "io.write(table.concat(Rules.ENGINES, '\\n'))\n",
        encoding="utf-8",
    )
    proc = run_lua(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    engines = [b for b in providers.BACKEND_CHOICES if b != providers.SCRIPTED]
    assert proc.stdout.split("\n") == engines


def test_every_plugin_file_compiles():
    """luac -p on the whole plugin, so a syntax error never reaches Lightroom."""
    if shutil.which("luac") is None:
        pytest.skip("luac not installed")
    files = sorted(str(p) for p in PLUGIN.glob("*.lua"))
    assert files, "no plugin Lua files found"
    proc = subprocess.run(["luac", "-p", *files], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


WINDOWS = sys.platform == "win32"


def _plugin_folder_holding(executable: Path, tmp_path: Path) -> Path:
    """A Melampus.lrplugin folder with the executable beside the Lua, as
    installed: copied, as the user copies it there from the download. Under a
    name with an apostrophe, the one character sh's own quoting cannot hold
    as it is: the command must close, escape and reopen it."""
    plugin_dir = tmp_path / "O'Brien" / "Melampus.lrplugin"
    plugin_dir.mkdir(parents=True)
    shutil.copy(executable, plugin_dir / executable.name)
    return plugin_dir


def _command_the_plugin_builds(
    plugin_dir: Path, previews: Path, results: Path, tmp_path: Path, *, engine: str
) -> str:
    """The one shell command MelampusAnalyze.lua builds under the mock SDK for
    the platform Lightroom reports (a fake Windows Lightroom on a Windows
    host), with `_PLUGIN.path` at `plugin_dir` and the engine preference set
    to `engine` ("" is the default: no preference).

    The mock's temp directory: under TMPDIR on a fake macOS Lightroom, the
    Windows temp folder (TEMP, as Lightroom reports it) on a fake Windows
    one, so the CLI log the command names lands under tmp_path either way."""
    script = tmp_path / "command.lua"
    script.write_text(
        "local mock = require('lrmock')\n"
        "mock.reset()\n"
        "mock.install(os.getenv('MELAMPUS_PLUGIN_DIR'),"
        " { windows = os.getenv('MELAMPUS_WINDOWS') == '1' })\n"
        "local Analyze = dofile(os.getenv('MELAMPUS_ANALYZE'))\n"
        "local ok, message = Analyze.run(os.getenv('MELAMPUS_PREVIEWS'),"
        " os.getenv('MELAMPUS_RESULTS'), 'wildlife', os.getenv('MELAMPUS_ENGINE'))\n"
        "assert(ok, message)\n"
        "io.write(mock.state.executed[1])\n",
        encoding="utf-8",
    )
    built = run_lua(script, env=os.environ | {
        "MELAMPUS_PLUGIN_DIR": str(plugin_dir),
        "MELAMPUS_ANALYZE": str(PLUGIN / "MelampusAnalyze.lua"),
        "MELAMPUS_PREVIEWS": str(previews),
        "MELAMPUS_RESULTS": str(results),
        "MELAMPUS_ENGINE": engine,
        "MELAMPUS_WINDOWS": "1" if WINDOWS else "0",
        "TMPDIR": str(tmp_path),
        "TEMP": str(tmp_path),
    })
    assert built.returncode == 0, built.stdout + built.stderr
    return built.stdout


def _cli_log_tail(tmp_path: Path) -> str:
    """The end of the CLI log the command sent the executable's output to, in
    the mock's temp directory under tmp_path, or a line saying no run wrote
    one. Read as the executable wrote it: its stderr is a file here, which
    Python encodes in the locale's encoding, the ANSI code page on Windows
    (readme.md's "§" is one byte there) and UTF-8 elsewhere."""
    log = next(tmp_path.rglob("melampus-cli.log"), None)
    if log is None:
        return "no CLI log"
    return log.read_text(encoding="mbcs" if WINDOWS else "utf-8", errors="replace")[-3000:]


def test_the_engine_preference_reaches_the_executable_through_the_command_the_plugin_builds(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Card #403, Done-when 1 at the real boundary. With the engine preference
    set to ollama, the command the plugin builds carries `--backend ollama`,
    and run through the shell LrTasks.execute hands it to (sh against
    dist/melampus, cmd.exe against dist/melampus.exe) with no python on the
    path the executable receives it: it answers with its own refusal for an
    engine that is not built yet (card #406), exit 3, written to the CLI log
    the plugin points a failed run at. Nothing is sent anywhere and no
    weights are read."""
    plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
    env = no_python_environment(tmp_path)

    command = _command_the_plugin_builds(
        plugin_dir, photos, photos / "results.json", tmp_path, engine="ollama")
    assert f"--backend {as_the_shell_receives_it('ollama')}" in command, command

    proc = run_as_lightroom_would(command, env=env, cwd=tmp_path,
                                  capture_output=True, text=True, timeout=600)

    assert proc.returncode == 3, f"exit {proc.returncode}: {proc.stderr[-2000:]}"
    tail = _cli_log_tail(tmp_path)
    assert "The Ollama engine is not built yet" in tail, tail
    assert "invalid choice" not in tail, f"the executable does not accept ollama:\n{tail}"


def test_the_command_the_plugin_builds_runs_the_executable_beside_it(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Card #401, Done-when 1 and 3 at the real boundary. The plugin's Analyze
    module, run under the mock SDK with `_PLUGIN.path` pointing at a plugin
    folder that holds the executable, builds one shell command for the
    platform Lightroom reports; that exact command is then run through the
    shell LrTasks.execute hands it to on this host: sh against dist/melampus
    on macOS, cmd.exe against dist/melampus.exe on Windows (the mock fakes a
    Windows Lightroom there, so `quote` and `shellLine` take their Windows
    branch and cmd.exe's own quote rule is what runs the line), with no
    python on the path. The scripted backend is selected the way a user
    selects any backend for the plugin: `[model] backend` in
    melampus.local.toml under the per-user data directory. The enriched
    results the plugin reads must land where the command said."""
    plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
    # Lightroom's previews folder: the committed frame, from conftest's fixture.
    previews = photos
    results = previews / "results.json"
    env = per_user_config(tmp_path, "[model]\nbackend = 'scripted'\n")

    command = _command_the_plugin_builds(plugin_dir, previews, results, tmp_path, engine="")
    assert as_the_shell_receives_it(plugin_dir / built_executable.name) in command, command

    proc = run_as_lightroom_would(command, env=env, cwd=tmp_path,
                                  capture_output=True, text=True, timeout=600)

    assert proc.returncode == 0, (
        f"exit {proc.returncode}: {proc.stderr[-2000:]}\n{_cli_log_tail(tmp_path)}")
    assert_scripted_results(results)


@needs_sh
def test_the_mock_hands_its_temp_paths_to_sh_as_data(tmp_path: Path):
    """The mock SDK makes, lists and removes its temp directory through sh
    (`mktemp -d`, `mkdir -p`, `ls`, `rm -rf`), under TMPDIR. TMPDIR comes
    from the environment, not from the plugin, so whatever it holds must reach
    sh as one quoted argument and come back from `mktemp` whole: a space, a
    quote, a `$`, a backtick and a newline in it make a directory of that name,
    a sibling that an unquoted `rm -rf` of the first word would remove stays,
    the parent that a path cut at the newline would name stays, and nothing in
    the name runs."""
    marker = tmp_path / "marker"
    parent = tmp_path / "a b'c$HOME`touch $MARKER`"
    base = parent / "\nd"
    base.mkdir(parents=True)
    # What `rm -rf` would take if the path came back cut at the newline.
    (parent / "canary").write_text("", encoding="utf-8")
    sibling = tmp_path / "a"
    sibling.mkdir()
    (sibling / "canary").write_text("", encoding="utf-8")

    script = tmp_path / "temp.lua"
    script.write_text(
        "local mock = require('lrmock')\n"
        "mock.reset()\n"
        "mock.install('/nowhere/Melampus.lrplugin')\n"
        "local LrFileUtils, LrPathUtils = import('LrFileUtils'), import('LrPathUtils')\n"
        "local temp = LrPathUtils.getStandardFilePath('temp')\n"
        "local work = LrPathUtils.child(temp, 'melampus-previews-1')\n"
        "LrFileUtils.createAllDirectories(work)\n"
        "assert(io.open(LrPathUtils.child(work, 'one.jpg'), 'w')):close()\n"
        "local listed = 0\n"
        "for _ in LrFileUtils.files(work) do listed = listed + 1 end\n"
        "assert(listed == 1, 'listed ' .. listed .. ' files in ' .. work)\n"
        "mock.cleanUp()\n"
        "io.write(temp)\n",
        encoding="utf-8",
    )
    proc = run_lua(script, env=os.environ | {"TMPDIR": str(base), "MARKER": str(marker)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    temp = Path(proc.stdout)
    assert temp.parent == base, f"the temp directory is not under TMPDIR: {temp}"
    assert not temp.exists(), f"cleanUp left {temp}"
    assert (parent / "canary").exists(), "cleanUp removed the parent of TMPDIR"
    assert (sibling / "canary").exists(), "cleanUp removed a sibling of TMPDIR"
    assert not marker.exists(), "a backtick in TMPDIR ran through sh"
