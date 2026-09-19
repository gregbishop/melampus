"""Run the Lua plugin test suites from pytest, so one command covers everything.

The plugin's decision logic lives in dependency-free Lua modules precisely so it
can be tested without Lightroom. Skips cleanly if no interpreter is installed;
Lightroom itself runs Lua 5.1, so this catches logic errors, not dialect ones.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import PHOTO

from test_binary import per_user_config

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin" / "Melampus.lrplugin"
TESTS = REPO / "plugin" / "tests"

pytestmark = pytest.mark.skipif(shutil.which("lua") is None, reason="lua not installed")


def run_lua(script: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["lua", "-e", f'package.path="{TESTS}/?.lua;{PLUGIN}/?.lua;"..package.path',
         str(script)],
        capture_output=True, text=True, cwd=TESTS, env=env,
    )


def test_write_rules():
    """CLAUDE.md §5.3 safety rules: no overwrites, dry run, idempotency."""
    proc = run_lua(TESTS / "test_rules.lua")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed" in proc.stdout, proc.stdout


def test_import_runs_against_a_mock_lightroom():
    """Executes the real MelampusImport.lua end to end against a mock SDK.

    Catches what unit tests could not: keywords failing to attach, ratings not
    written, overwrite protection, dry run writing nothing, idempotency, and
    file I/O inside a write gate.

    It does NOT reproduce every real SDK behaviour — see docs/plugin.md for
    what remains unverified.
    """
    proc = run_lua(TESTS / "test_import_integration.lua")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed" in proc.stdout, proc.stdout


def test_json_decoder():
    proc = run_lua(TESTS / "test_json.lua")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed" in proc.stdout, proc.stdout


def test_every_plugin_file_compiles():
    """luac -p on the whole plugin, so a syntax error never reaches Lightroom."""
    if shutil.which("luac") is None:
        pytest.skip("luac not installed")
    files = sorted(str(p) for p in PLUGIN.glob("*.lua"))
    assert files, "no plugin Lua files found"
    proc = subprocess.run(["luac", "-p", *files], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_the_command_the_plugin_builds_runs_the_executable_beside_it(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Card #401, Done-when 1 at the real boundary. The plugin's Analyze module,
    run under the mock SDK with `_PLUGIN.path` pointing at a plugin folder that
    holds the executable, builds one shell command; that exact command is then
    run through sh, as Lightroom's LrTasks.execute runs it on macOS, against
    dist/melampus with no python on the path. The scripted backend is selected
    the way a user selects any backend for the plugin: `[model] backend` in
    melampus.local.toml under the per-user data directory. The enriched results
    the plugin reads must land where the command said."""
    from melampus.plugin_results import PLUGIN_FIELDS

    plugin_dir = tmp_path / "Melampus.lrplugin"
    plugin_dir.mkdir()
    (plugin_dir / built_executable.name).symlink_to(built_executable)
    # Lightroom's previews folder: the committed frame, from conftest's fixture.
    previews = photos
    results = previews / "results.json"
    env = per_user_config(tmp_path, "[model]\nbackend = 'scripted'\n")

    script = tmp_path / "command.lua"
    script.write_text(
        "local mock = require('lrmock')\n"
        "mock.reset()\n"
        "mock.install(os.getenv('MELAMPUS_PLUGIN_DIR'))\n"
        "local Analyze = dofile(os.getenv('MELAMPUS_ANALYZE'))\n"
        "local ok, message = Analyze.run(os.getenv('MELAMPUS_PREVIEWS'),"
        " os.getenv('MELAMPUS_RESULTS'), 'wildlife')\n"
        "assert(ok, message)\n"
        "io.write(mock.state.executed[1])\n",
        encoding="utf-8",
    )
    built = run_lua(script, env=os.environ | {
        "MELAMPUS_PLUGIN_DIR": str(plugin_dir),
        "MELAMPUS_ANALYZE": str(PLUGIN / "MelampusAnalyze.lua"),
        "MELAMPUS_PREVIEWS": str(previews),
        "MELAMPUS_RESULTS": str(results),
        "TMPDIR": str(tmp_path),
    })
    assert built.returncode == 0, built.stdout + built.stderr
    command = built.stdout
    assert str(plugin_dir / built_executable.name) in command, command

    proc = subprocess.run(command, shell=True, env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)

    # The mock keeps its temp directory under TMPDIR, so the CLI log is here.
    log = next(tmp_path.rglob("melampus-cli.log"), None)
    assert proc.returncode == 0, (
        f"exit {proc.returncode}: {proc.stderr[-2000:]}\n"
        f"{log.read_text(encoding='utf-8')[-3000:] if log else 'no CLI log'}")
    rows = json.loads(results.read_text(encoding="utf-8"))
    assert [r["file"] for r in rows] == [PHOTO]
    # The scripted fake answers nothing, so the row has no identification and
    # therefore no burst_agreement (that is agreement between calls); every
    # other enrichment field is scored from the pixels and the capture times.
    expected = set(PLUGIN_FIELDS) - {"burst_agreement"}
    assert expected <= set(rows[0]), f"missing {expected - set(rows[0])}"
    assert 0 < rows[0]["quality"] <= 100, "quality was not scored on the pixels"


def test_the_mock_hands_its_temp_paths_to_sh_as_data(tmp_path: Path):
    """The mock SDK makes, lists and removes its temp directory through sh
    (`mktemp -d`, `mkdir -p`, `ls`, `rm -rf`), under TMPDIR. TMPDIR comes
    from the environment, not from the plugin, so whatever it holds must reach
    sh as one quoted argument: a space, a quote, a `$` and a backtick in it
    make a directory of that name, a sibling that an unquoted `rm -rf` of the
    first word would remove stays, and nothing in the name runs."""
    marker = tmp_path / "marker"
    base = tmp_path / "a b'c$HOME`touch $MARKER`"
    base.mkdir()
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
    assert (sibling / "canary").exists(), "cleanUp removed a sibling of TMPDIR"
    assert not marker.exists(), "a backtick in TMPDIR ran through sh"
