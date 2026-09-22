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
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import (
    FAKE_FILES,
    FAKE_FOLDER,
    FAKE_REPO,
    FAKE_TOTAL,
    PHOTO,
    REAL_CLAUDE_CODE_VERDICT,
    REAL_CODEX_VERDICT,
    FakeHub,
    closed_port,
    fake_bytes,
    snapshot_files,
)
from huggingface_hub.constants import DOWNLOAD_CHUNK_SIZE

from melampus import providers
from melampus.download import CANCEL_MARKER, EXIT_CANCELLED, Update
from test_binary import no_python_environment, per_user_config, per_user_data_dir

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


SUMMARY = re.compile(r"^(\d+) passed, (\d+) failed$", re.MULTILINE)


def summary_counts(stdout: str) -> tuple[int, int] | None:
    """The (passed, failed) counts from the harness's summary line, parsed
    here and nowhere else; None when the output has no summary line."""
    summary = SUMMARY.search(stdout)
    if summary is None:
        return None
    passed, failed = (int(n) for n in summary.groups())
    return passed, failed


def assert_suite_green(proc: subprocess.CompletedProcess) -> None:
    """The gate every Lua suite passes through. Card #443: the harness's
    summary line is parsed into its counts, never substring-matched ("10
    failed" contains "0 failed"), and the exit code is checked as well."""
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    counts = summary_counts(proc.stdout)
    assert counts is not None, f"no summary line from the harness:\n{output}"
    passed, failed = counts
    assert failed == 0, output
    assert passed > 0, output


def run_lua_suite(script: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run one of the plugin's Lua test suites and assert its verdict through
    assert_suite_green."""
    proc = run_lua(script, env=env)
    assert_suite_green(proc)
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


@needs_sh
def test_settings_dialog_against_a_mock_lightroom(tmp_path: Path):
    """Card #405: executes the real MelampusSettings.lua against the mock SDK.
    The engine picker lists the engines in the executable's order with the
    ones detection says cannot run here greyed and their reasons shown; the
    Ollama link is there exactly when ollama is unavailable; the API key
    field shows only for the picked cloud engine and stores through
    LrPasswords, never the preferences, a file, or the log; a missing
    executable greys nothing. Card #423: claude-code and codex after the
    four, greyed as not installed or not signed in, offered when signed in,
    no key field for either, and the picked engine's reason under the
    picker, the billing sentence for a signed-in CLI.
    Card #408: the download plumbing, stepped through the mock's tasks: the
    command with stdout redirected on both shells, the poller reading the
    progress file, Cancel writing the marker, exit 3 with the log's tail.
    Card #409: the same row once per engine with a model, mlx and ollama,
    each shown for its engine and each command carrying it as --backend.
    The files the fake executable writes land under tmp_path (the mock's
    temp directory is TMPDIR), and Cancel's marker folder is made by the
    mock through sh, so this suite runs where the import suite does."""
    run_lua_suite(TESTS / "test_settings_dialog.lua", env=os.environ | {"TMPDIR": str(tmp_path)})


def test_json_decoder():
    run_lua_suite(TESTS / "test_json.lua")


@pytest.mark.parametrize("stdout, counts", [
    ("10 passed, 10 failed\n", (10, 10)),
    ("  FAIL a case: boom\n1 passed, 1 failed\n", (1, 1)),
    ("no summary line at all\n", None),
])
def test_the_summary_line_is_parsed_once_into_ints(stdout: str, counts: tuple[int, int] | None):
    """Card #443: the harness's summary line is parsed in one place, into
    ints, and a missing line is None rather than an AttributeError."""
    assert summary_counts(stdout) == counts


def _summary(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["lua"], returncode=returncode, stdout=stdout, stderr="")


@pytest.mark.parametrize("stdout", [
    "  FAIL a case: boom\n10 passed, 10 failed\n",
    "0 passed, 0 failed\n",
    "no summary line at all\n",
])
def test_the_gate_fails_a_summary_that_is_not_green(stdout: str):
    """Card #443: "10 failed" contains "0 failed", so the counts are parsed,
    never substring-matched; no passes is not green either."""
    with pytest.raises(AssertionError):
        assert_suite_green(_summary(stdout))


def test_the_gate_fails_a_green_summary_from_a_process_that_exited_non_zero():
    with pytest.raises(AssertionError):
        assert_suite_green(_summary("3 passed, 0 failed\n", returncode=1))


def test_the_gate_passes_a_green_summary():
    assert_suite_green(_summary("3 passed, 0 failed\n"))


def test_a_failing_suite_exits_non_zero_under_lua(tmp_path: Path):
    """Card #443, done-when 2: the harness itself ends the interpreter with
    exit 1 after a failing summary, so a failure shows without the wrapper
    and the summary line is still printed; the gate refuses it either way."""
    suite = tmp_path / "test_tiny.lua"
    suite.write_text(
        "local t = require 'harness'\n"
        "t.test('passes', function() t.isTrue(true) end)\n"
        "t.test('fails', function() t.isTrue(false, 'boom') end)\n"
        "return t.summary()\n"
    )
    proc = run_lua(suite)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert summary_counts(proc.stdout) == (1, 1), proc.stdout
    assert "FAIL fails: " in proc.stdout and "boom" in proc.stdout, proc.stdout
    with pytest.raises(AssertionError):
        assert_suite_green(proc)


def test_the_plugin_names_the_engines_the_cli_accepts(tmp_path: Path):
    """Card #403: the engine names are spelled once per language, in
    `Rules.ENGINES` for the plugin and `providers.BACKEND_CHOICES` for the
    CLI, and this is what binds them: the plugin's list, read through lua, is
    the CLI's list without the offline test fake, then the two subscription
    CLIs (card #423; the command seam is not a picker choice), in the same
    order. A rename on either side fails here rather than as a usage error
    the user never sees."""
    script = tmp_path / "engines.lua"
    script.write_text(
        "local Rules = require('MelampusRules')\n"
        "io.write(table.concat(Rules.ENGINES, '\\n'))\n",
        encoding="utf-8",
    )
    proc = run_lua(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    engines = [b for b in providers.BACKEND_CHOICES if b != providers.SCRIPTED]
    assert proc.stdout.split("\n") == [*engines, providers.CLAUDE_CODE, providers.CODEX]


def test_the_plugin_offers_a_download_row_for_exactly_the_engines_the_cli_fetches_a_model_for(tmp_path: Path):
    """Card #409: the engines with a local model to fetch are spelled once
    per language, in `Rules.MODEL_ENGINES` for the plugin (one Download row
    each) and `cli.MODEL_ENGINES` for the executable (the engines the three
    model flags act for), and this is what binds them, the way the #403 test
    above binds the engine names: the plugin's list, read through lua, is the
    CLI's list in the same order. An engine added on one side alone would
    otherwise get a Download row the executable refuses, or a dispatch the
    dialog never shows."""
    from melampus import cli

    script = tmp_path / "model-engines.lua"
    script.write_text(
        "local Rules = require('MelampusRules')\n"
        "io.write(table.concat(Rules.MODEL_ENGINES, '\\n'))\n",
        encoding="utf-8",
    )
    proc = run_lua(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert proc.stdout.split("\n") == list(cli.MODEL_ENGINES)


def test_the_plugin_stores_each_key_under_the_variable_the_executable_reads(tmp_path: Path):
    """Card #405, Done-when 2: the variable a cloud engine's key travels in is
    spelled once per language, in `Rules.KEY_VARIABLES` for the plugin and
    `providers.KEY_VARIABLES` for the executable, and this is what binds
    them, the way the #403 test above binds the engine names: the plugin's
    table, read through lua for every engine it names, is the first variable
    the executable looks in for each cloud engine, and nothing for the rest.
    A rename on either side fails here rather than as a stored key the
    executable never sees."""
    script = tmp_path / "keys.lua"
    script.write_text(
        "local Rules = require('MelampusRules')\n"
        "for _, engine in ipairs(Rules.ENGINES) do\n"
        "  io.write(engine, '\\t', tostring(Rules.keyVariable(engine)), '\\n')\n"
        "end\n",
        encoding="utf-8",
    )
    proc = run_lua(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    rows = [line.split("\t") for line in proc.stdout.splitlines()]
    plugin_variables = {engine: variable for engine, variable in rows if variable != "nil"}
    executable_variables = {
        engine: variables[0] for engine, variables in providers.KEY_VARIABLES.items()}
    assert plugin_variables == executable_variables, proc.stdout


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


def _plugin_under_the_mock(
    plugin_dir: Path, tmp_path: Path, body: str, *, home: Path | None = None, **env: str
) -> str:
    """Run `body`, Lua, with the mock SDK installed for `plugin_dir` (so
    `_PLUGIN.path` is there) on the platform Lightroom reports for this host
    (a fake Windows Lightroom on a Windows host), and MelampusAnalyze.lua
    loaded fresh under it through the mock's own loader as `Analyze`, with
    the MelampusRules.lua instance it uses as `Rules`. `home` is the home
    folder the fake Lightroom reports, when a test names one; else the
    mock's own. `env` is what the body reads through os.getenv. Hands back
    what it wrote.

    The mock's temp directory: under TMPDIR on a fake macOS Lightroom, the
    Windows temp folder (TEMP, as Lightroom reports it) on a fake Windows
    one, so what a command writes there lands under tmp_path either way."""
    script = tmp_path / "under-the-mock.lua"
    script.write_text(
        "local mock = require('lrmock')\n"
        "local Analyze = mock.loadUnderMock('MelampusAnalyze',"
        " { home = os.getenv('MELAMPUS_HOME') },"
        " os.getenv('MELAMPUS_PLUGIN_DIR'),"
        " { windows = os.getenv('MELAMPUS_WINDOWS') == '1' })\n"
        "local Rules = require('MelampusRules')\n"
        + body,
        encoding="utf-8",
    )
    ran = run_lua(script, env=os.environ | {
        "MELAMPUS_PLUGIN_DIR": str(plugin_dir),
        "MELAMPUS_WINDOWS": "1" if WINDOWS else "0",
        "TMPDIR": str(tmp_path),
        "TEMP": str(tmp_path),
    } | ({"MELAMPUS_HOME": str(home)} if home is not None else {}) | env)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    return ran.stdout


def _command_the_plugin_builds(
    plugin_dir: Path, previews: Path, results: Path, tmp_path: Path, *, engine: str,
    stored_key: str = "",
) -> str:
    """The one shell command MelampusAnalyze.lua builds under the mock SDK
    with `_PLUGIN.path` at `plugin_dir` and the engine preference set to
    `engine` ("" is the default: no preference). `stored_key` is what
    LrPasswords holds for `engine`'s key variable (card #405); "" means
    nothing stored."""
    return _plugin_under_the_mock(
        plugin_dir, tmp_path,
        "local variable = Rules.keyVariable(os.getenv('MELAMPUS_ENGINE'))\n"
        "if variable and os.getenv('MELAMPUS_STORED_KEY') ~= '' then\n"
        "  mock.state.passwords[variable] = os.getenv('MELAMPUS_STORED_KEY')\n"
        "end\n"
        "local ok, message = Analyze.run(os.getenv('MELAMPUS_PREVIEWS'),"
        " os.getenv('MELAMPUS_RESULTS'), 'wildlife', os.getenv('MELAMPUS_ENGINE'))\n"
        "assert(ok, message)\n"
        "io.write(mock.state.executed[1])\n",
        MELAMPUS_PREVIEWS=str(previews), MELAMPUS_RESULTS=str(results),
        MELAMPUS_ENGINE=engine, MELAMPUS_STORED_KEY=stored_key)


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
    Ollama that is not running (card #406; the per-user config under the fake
    HOME points it at a closed port, so a developer's Ollama cannot answer),
    exit 3, written to the CLI log the plugin points a failed run at. Nothing
    is sent anywhere and no weights are read."""
    plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
    port = closed_port()
    env = per_user_config(tmp_path, f'[model]\nollama_url = "http://127.0.0.1:{port}"\n')

    command = _command_the_plugin_builds(
        plugin_dir, photos, photos / "results.json", tmp_path, engine="ollama")
    assert f"--backend {as_the_shell_receives_it('ollama')}" in command, command

    proc = run_as_lightroom_would(command, env=env, cwd=tmp_path,
                                  capture_output=True, text=True, timeout=600)

    assert proc.returncode == 3, f"exit {proc.returncode}: {proc.stderr[-2000:]}"
    tail = _cli_log_tail(tmp_path)
    assert f"No Ollama server at http://127.0.0.1:{port}" in tail, tail
    assert "invalid choice" not in tail, f"the executable does not accept ollama:\n{tail}"


def test_the_log_lands_under_the_data_directory_the_executable_reports(
    built_executable: Path, tmp_path: Path
):
    """Card #442, Done-when 1 at the real boundary. The plugin needs the
    per-user Melampus data directory before it can run anything (it logs the
    first command), so MelampusLog.lua derives it in Lua from the SDK's home
    folder and the platform; the executable derives its own from HOME (or
    %LOCALAPPDATA%) in config._data_root and reports it through
    `--model-status` as the parent of `cancel_path`. Given the same home,
    the two rules name the same directory on this host, so the plugin's log
    sits beside the executable's config and caches: <root>/logs/Melampus.log.
    The status is asked for ollama at a closed port, so no server, hub or
    network is involved; nothing is downloaded."""
    env = per_user_config(tmp_path, f'[model]\nollama_url = "http://127.0.0.1:{closed_port()}"\n')
    proc = subprocess.run(
        [str(built_executable), "--model-status", "--backend", "ollama"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    data_root = Path(json.loads(proc.stdout)["cancel_path"]).parent.parent
    assert data_root == per_user_data_dir(Path(env["HOME"]))

    lua_root, lua_log = _plugin_under_the_mock(
        PLUGIN, tmp_path,
        "io.write(require('MelampusLog').dataRoot() .. '\\n' .. require('MelampusLog').path())\n",
        home=Path(env["HOME"]),
    ).splitlines()
    assert Path(lua_root) == data_root, f"the plugin's root {lua_root} is not the executable's {data_root}"
    assert Path(lua_log) == data_root / "logs" / "Melampus.log"


def test_the_engines_the_plugin_knows_are_the_executables_in_its_order(
    built_executable: Path, tmp_path: Path
):
    """Card #423: the plugin validates the engine preference before the shell
    (Rules.ENGINES, the picker's order too), and the executable's
    `--detect-engines` is the list the picker is built from, so the two are
    one order in two places. Held to each other here against dist/melampus
    with no python on the path: a name added to one without the other fails
    CI, and the picker can never offer an engine the run would refuse."""
    script = tmp_path / "engines.lua"
    script.write_text(
        "local Rules = require('MelampusRules')\n"
        "io.write(table.concat(Rules.ENGINES, '\\n'))\n",
        encoding="utf-8",
    )
    known = run_lua(script)
    assert known.returncode == 0, known.stdout + known.stderr

    proc = subprocess.run(
        [str(built_executable), "--detect-engines"],
        env=no_python_environment(tmp_path), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert [v["engine"] for v in json.loads(proc.stdout)] == known.stdout.split("\n")


@pytest.mark.parametrize("engine", ["claude-code", "codex"])
def test_the_cli_engine_preference_reaches_the_executable_through_the_command_the_plugin_builds(
    built_executable: Path, photos: Path, tmp_path: Path, engine: str
):
    """Card #423, Done-when 2 at the real boundary. With the engine
    preference set to a subscription CLI, the command the plugin builds
    carries `--backend claude-code` (or codex) and sets no key variable
    ahead of the executable, whatever LrPasswords holds; run through sh
    against dist/melampus with nothing on the PATH (so no `claude` or
    `codex` either), through the shell LrTasks.execute hands it to, the
    executable receives the name and answers with its own refusal for a CLI
    that is not installed, naming where to get it, exit 3, in the CLI log
    the plugin points a failed run at. Nothing runs, nothing is sent
    anywhere."""
    from melampus import providers

    cli = {"claude-code": providers.CLAUDE_CODE_CLI, "codex": providers.CODEX_CLI}[engine]
    plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
    env = no_python_environment(tmp_path)
    assert shutil.which(cli.program, path=env["PATH"]) is None

    command = _command_the_plugin_builds(
        plugin_dir, photos, photos / "results.json", tmp_path, engine=engine,
        stored_key="stored-in-lrpasswords-not-a-real-key-9b2d")
    assert f"--backend {as_the_shell_receives_it(engine)}" in command, command
    # The executable first: on Windows after the quote the whole line is
    # wrapped in for cmd.exe, on macOS at the very start.
    line = command[1:] if WINDOWS else command
    assert line.startswith(as_the_shell_receives_it(plugin_dir / built_executable.name)), (
        f"something is set ahead of the executable for an engine that needs no key:\n{command}")
    assert "MELAMPUS_" not in command and "not-a-real-key" not in command, command

    proc = run_as_lightroom_would(command, env=env, cwd=tmp_path,
                                  capture_output=True, text=True, timeout=600)

    tail = _cli_log_tail(tmp_path)
    assert proc.returncode == 3, f"exit {proc.returncode}: {proc.stderr[-2000:]}\n{tail}"
    assert f"{cli.title} is not installed" in tail, tail
    assert cli.install in tail, tail
    assert "invalid choice" not in tail, f"the executable does not accept {engine}:\n{tail}"


@pytest.mark.parametrize("stored_key", ["", "stored-in-lrpasswords-not-a-real-key-9b2d"])
def test_the_stored_key_reaches_the_executable_through_the_command_the_plugin_builds(
    built_executable: Path, photos: Path, tmp_path: Path, stored_key: str
):
    """Card #405, Done-when 2 at the real boundary. With openai picked and a
    key in LrPasswords, the command the plugin builds carries the key in the
    executable's environment, not its arguments; run through sh against
    dist/melampus with no key variable in the environment and no python on
    the path, the executable gets past its key check. `model.max_images = 0`
    in melampus.local.toml then refuses the run at the cost ceiling, exit 3,
    so nothing is sent anywhere. Without a stored key the same command stops
    one step earlier, on the executable's own "needs an API key"."""
    plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
    env = per_user_config(tmp_path, "[model]\nmax_images = 0\n")
    assert "MELAMPUS_OPENAI_KEY" not in env and "OPENAI_API_KEY" not in env

    command = _command_the_plugin_builds(
        plugin_dir, photos, photos / "results.json", tmp_path, engine="openai",
        stored_key=stored_key)
    assert f"--backend {as_the_shell_receives_it('openai')}" in command, command
    arguments = command.split(as_the_shell_receives_it(plugin_dir / built_executable.name), 1)[1]
    assert stored_key == "" or stored_key not in arguments, (
        f"the key is an argument of the executable:\n{command}")

    proc = run_as_lightroom_would(command, env=env, cwd=tmp_path,
                                  capture_output=True, text=True, timeout=600)

    tail = _cli_log_tail(tmp_path)
    assert proc.returncode == 3, f"exit {proc.returncode}: {proc.stderr[-2000:]}\n{tail}"
    if stored_key:
        assert "needs an API key" not in tail, f"the key did not reach the executable:\n{tail}"
        assert "exceed model.max_images" in tail, f"did not reach the cost ceiling:\n{tail}"
        assert stored_key not in tail, f"the executable printed the key:\n{tail}"
    else:
        assert "needs an API key" in tail, tail


def test_the_detection_the_plugin_runs_reaches_the_executable_and_fills_the_picker(
    built_executable: Path, tmp_path: Path
):
    """Card #405, Done-when 1 at the real boundary. The plugin's Analyze
    module, run under the mock SDK with `_PLUGIN.path` at a plugin folder
    that holds the executable, builds the one `--detect-engines` line and
    hands it to the shell LrTasks.execute hands it to on this host (sh
    against dist/melampus, cmd.exe against dist/melampus.exe); what the
    executable actually printed then goes through the plugin's own JSON
    decoder and `Rules.engineItems`: seven items in the executable's order
    (card #423: the owner's four, then the two subscription CLIs), ollama
    greyed with the download address from the executable's reason as its
    link, mlx and the CLIs as this machine decides (the CLIs' real verdicts
    are asked from this process too, the way test_binary.py asks Ollama's,
    so a developer's signed-in CLI decides nothing the test did not
    measure). No Ollama answers on a runner and
    nothing is sent anywhere; the executable's output on its own, with no
    python on the path, is test_binary.py's."""
    plugin_dir = _plugin_folder_holding(built_executable, tmp_path)

    listing = _plugin_under_the_mock(
        plugin_dir, tmp_path,
        "mock.state.onExecute = mock.runThroughTheShell\n"
        "local verdicts, problem = Analyze.detectEngines()\n"
        "assert(verdicts, problem)\n"
        "assert(#mock.state.executed == 1, 'detection ran ' .. #mock.state.executed .. ' commands')\n"
        "for _, item in ipairs(Rules.engineItems(verdicts, problem)) do\n"
        "  io.write(item.value, '\\t', tostring(item.enabled), '\\t', tostring(item.link), '\\n')\n"
        "end\n")

    items = [line.split("\t") for line in listing.splitlines()]
    engines = [b for b in providers.BACKEND_CHOICES if b != providers.SCRIPTED]
    assert [value for value, _, _ in items] == ["", *engines, providers.CLAUDE_CODE, providers.CODEX], listing
    enabled = {value: state == "true" for value, state, _ in items}
    links = {value: link for value, _, link in items}
    assert enabled[""] and enabled["openai"] and enabled["claude"], listing
    assert enabled["mlx"] is providers.on_apple_silicon(), listing
    assert not enabled["ollama"], f"ollama greyed by nothing; an Ollama server answered?\n{listing}"
    assert links["ollama"] == providers.OLLAMA_INSTALL, listing
    assert all(links[value] == "nil" for value in ("", "mlx", "openai", "claude")), listing
    for cli, verdict in ((providers.CLAUDE_CODE_CLI, REAL_CLAUDE_CODE_VERDICT()),
                         (providers.CODEX_CLI, REAL_CODEX_VERDICT())):
        assert enabled[cli.engine] is verdict.available, listing
        # Not installed, the install page is the link; signed in, or installed
        # and not signed in, the reason names no address.
        expected_link = cli.install if cli.install in verdict.reason else "nil"
        assert links[cli.engine] == expected_link, listing


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


def _model_commands_the_dialog_builds(
    plugin_dir: Path, tmp_path: Path, engine: str = "mlx"
) -> tuple[str, str, str, Path]:
    """The three shell commands the Settings dialog builds for `engine`'s
    model (card #408, #409) under the mock SDK with `_PLUGIN.path` at
    `plugin_dir` and TMPDIR at `tmp_path`: the `--model-status` line it runs
    at open, the `--download-model` line the Download button runs, stdout
    redirected to the progress file the poller reads, and the
    `--remove-model` line the Remove button runs, each as the mock recorded
    the dialog running it and each carrying the engine; and the mock's temp
    directory, under tmp_path, where the lines put their files."""
    listing = _plugin_under_the_mock(
        plugin_dir, tmp_path,
        "Analyze.modelStatus(os.getenv('MELAMPUS_ENGINE'))\n"
        "local command, err = Analyze.downloadCommand(os.getenv('MELAMPUS_ENGINE'))\n"
        "assert(command, err)\n"
        "local removed, removeErr = Analyze.removeModel(os.getenv('MELAMPUS_ENGINE'))\n"
        "assert(removed, removeErr)\n"
        "io.write(mock.state.executed[1] .. '\\n' .. command .. '\\n' .. mock.state.executed[2]"
        " .. '\\n' .. mock.state.tempDir .. '\\n')\n",
        MELAMPUS_ENGINE=engine)
    status, download, remove, temp = listing.splitlines()
    assert "--model-status" in status and "--download-model" in download and "--remove-model" in remove
    for command in (status, download, remove):
        assert f"--backend '{engine}'" in command, command
    return status, download, remove, Path(temp)


def _status_the_dialog_reads(command: str, env: dict[str, str], temp: Path) -> dict:
    proc = run_as_lightroom_would(command, env=env, cwd=temp.parent, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"exit {proc.returncode}: {_cli_log_tail(temp)}"
    return json.loads((temp / "melampus-model-status.json").read_text(encoding="utf-8"))


@needs_sh
def test_the_download_command_the_dialog_builds_fetches_the_model_and_the_status_flips_to_installed(
    built_executable: Path, fake_hub, hub_env: dict[str, str], tmp_path: Path
):
    """Card #408, Done-when 1 and 3 at the real boundary. The commands the
    dialog builds, run through sh against dist/melampus with no python on
    the path, HF_ENDPOINT at the fake hub and HF_HOME under tmp_path:
    `--model-status` reports the model absent with the fake's size, the
    download line fills the progress file the poller reads with protocol
    lines ending in `done <path>`, and the status then reports installed
    at that path. No real weights move."""
    plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
    env = per_user_config(tmp_path, f'[model]\nrepo = "{FAKE_REPO}"\n') | hub_env
    data_dir = per_user_data_dir(Path(env["HOME"]))
    status_command, download_command, _, temp = _model_commands_the_dialog_builds(plugin_dir, tmp_path)

    before = _status_the_dialog_reads(status_command, env, temp)
    assert before["repo"] == FAKE_REPO and before["installed"] is False and before["path"] is None
    assert before["bytes_total"] == FAKE_TOTAL and before["bytes_done"] == 0
    assert Path(before["cancel_path"]) == data_dir / "cache" / "download-cancel"

    proc = run_as_lightroom_would(download_command, env=env, cwd=tmp_path,
                                  capture_output=True, text=True, timeout=600)
    log = (temp / "melampus-download.log").read_text(encoding="utf-8")
    assert proc.returncode == 0, f"exit {proc.returncode}:\n{log[-3000:]}"
    lines = (temp / "melampus-download.progress").read_text(encoding="utf-8").splitlines()
    updates = [Update.parse(line) for line in lines]
    assert updates[0] == Update.progress(0, FAKE_TOTAL) and updates[-2] == Update.progress(FAKE_TOTAL, FAKE_TOTAL)
    assert updates[-1].state == "done"
    assert snapshot_files(Path(updates[-1].path)) == FAKE_FILES

    after = _status_the_dialog_reads(status_command, env, temp)
    assert after["installed"] is True and after["path"] == updates[-1].path
    assert after["bytes_done"] == after["bytes_total"] == FAKE_TOTAL


@needs_sh
def test_the_marker_the_dialog_writes_cancels_the_download_the_dialog_started(
    built_executable: Path, tmp_path: Path
):
    """Card #408, Done-when 2 at the real boundary: the path `--model-status`
    reports is the one `--download-model` watches. The download line runs
    against a throttled fake hub; once the progress file shows the first
    chunk, the marker is written where the status said (what Cancel does),
    and the executable ends the file with `cancelled`, exit 4, with the
    partial blob kept in HF_HOME."""
    big = fake_bytes(4 * DOWNLOAD_CHUNK_SIZE)
    hub = FakeHub(files={"config.json": FAKE_FILES["config.json"], "model.safetensors": big})
    hub.throttle = (64 * 1024, 0.002)
    with hub.serve():
        plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
        env = per_user_config(tmp_path, f'[model]\nrepo = "{FAKE_REPO}"\n') | {
            "HF_ENDPOINT": hub.endpoint, "HF_HOME": str(tmp_path / "hf")}
        status_command, download_command, _, temp = _model_commands_the_dialog_builds(plugin_dir, tmp_path)
        marker = Path(_status_the_dialog_reads(status_command, env, temp)["cancel_path"])
        assert marker.name == CANCEL_MARKER and not marker.exists()

        proc = subprocess.Popen(download_command, shell=True, env=env, cwd=tmp_path,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        progress = temp / "melampus-download.progress"
        import time
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            lines = progress.read_text(encoding="utf-8").splitlines() if progress.is_file() else []
            if lines and Update.parse(lines[-1]).bytes_done >= DOWNLOAD_CHUNK_SIZE:
                break
            time.sleep(0.05)
        else:
            proc.kill()
            pytest.fail("the progress file never showed a chunk")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        code = proc.wait(timeout=120)
        hub.throttle = None
        assert code == EXIT_CANCELLED, (code, (temp / "melampus-download.log").read_text()[-3000:])
        lines = progress.read_text(encoding="utf-8").splitlines()
        assert Update.parse(lines[-1]) == Update.cancelled()
        assert not marker.exists(), "the executable did not remove the marker on exit"
        blobs = tmp_path / "hf" / "hub" / FAKE_FOLDER / "blobs"
        (partial,) = blobs.glob("*.incomplete")
        assert DOWNLOAD_CHUNK_SIZE <= partial.stat().st_size < len(big), "the partial file was not kept"


@needs_sh
def test_the_download_command_the_dialog_builds_for_ollama_pulls_the_model_and_the_status_flips_to_installed(
    built_executable: Path, tmp_path: Path
):
    """Card #409, Done-when 1 and 2 at the real boundary. The commands the
    dialog builds for the ollama engine, run through sh against
    dist/melampus with no python on the path and `[model] ollama_url` in
    the per-user config pointing at the fake Ollama on loopback:
    `--model-status --backend ollama` reports the model absent with no
    size, the download line fills the progress file the poller reads with
    protocol lines ending in `done <model>`, the fake was asked to pull
    exactly that model, the status then reports installed with the size
    the fake lists, and `--remove-model` empties it again. No real model
    is pulled and nothing leaves loopback."""
    from conftest import FakeOllama
    from melampus.download import Update
    from test_download import FAKE_MODEL

    with FakeOllama(library={FAKE_MODEL: [3000, 1000]}).serve() as ollama:
        plugin_dir = _plugin_folder_holding(built_executable, tmp_path)
        env = per_user_config(
            tmp_path, f'[model]\nollama_url = "{ollama.endpoint}"\nollama_model = "{FAKE_MODEL}"\n')
        data_dir = per_user_data_dir(Path(env["HOME"]))
        status_command, download_command, remove_command, temp = _model_commands_the_dialog_builds(
            plugin_dir, tmp_path, engine="ollama")

        before = _status_the_dialog_reads(status_command, env, temp)
        assert before["repo"] == FAKE_MODEL and before["installed"] is False and before["path"] is None
        assert before["bytes_total"] is None and before["bytes_done"] == 0
        assert Path(before["cancel_path"]) == data_dir / "cache" / "download-cancel"

        proc = run_as_lightroom_would(download_command, env=env, cwd=tmp_path,
                                      capture_output=True, text=True, timeout=600)
        log = (temp / "melampus-download.log").read_text(encoding="utf-8")
        assert proc.returncode == 0, f"exit {proc.returncode}:\n{log[-3000:]}"
        lines = (temp / "melampus-download.progress").read_text(encoding="utf-8").splitlines()
        updates = [Update.parse(line) for line in lines]
        assert updates[0] == Update.progress(0, 3000) and updates[-2] == Update.progress(4000, 4000)
        assert updates[-1] == Update.done(FAKE_MODEL)
        assert [p["model"] for p in ollama.pulls] == [FAKE_MODEL]

        after = _status_the_dialog_reads(status_command, env, temp)
        assert after["installed"] is True and after["path"] == FAKE_MODEL
        assert after["bytes_done"] == after["bytes_total"] == 4000

        removed = run_as_lightroom_would(remove_command, env=env, cwd=tmp_path,
                                         capture_output=True, text=True, timeout=600)
        assert removed.returncode == 0, f"exit {removed.returncode}: {_cli_log_tail(temp)}"
        removal = (temp / "melampus-removed.txt").read_text(encoding="utf-8")
        assert removal.strip() == f"removed {FAKE_MODEL}" and ollama.deletes == [FAKE_MODEL]
        assert _status_the_dialog_reads(status_command, env, temp)["installed"] is False
        assert {path for _, path in ollama.requests} == {"/api/tags", "/api/pull", "/api/delete"}
