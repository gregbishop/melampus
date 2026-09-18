"""The shipped executable (card #399): the service and MLX in one file.

Done-when 1: given the build script runs on an Apple Silicon Mac, when it
finishes, then one executable exists that carries the service and MLX.
Done-when 2: given a machine with no Python on the path, when the executable
analyzes a fixture image with the fake backend, then it prints the same JSON
the CLI does today.
Done-when 3: given the repo's test command, when it runs, then the build and
this smoke test are part of it — `.venv/bin/python -m pytest -q --build-binary`
(the option lives in conftest.py; without it these tests use an existing
build, or skip and say how to get one).

Card #400 adds Windows, where MLX does not exist. Done-when 1: given the CI
workflow runs on a Windows runner, when it finishes, then a melampus.exe
exists that starts and analyzes a fixture image with the scripted backend.
Done-when 2: given melampus.exe, when the local MLX engine is requested, then
it says clearly that MLX needs Apple Silicon and names the engines that work
here. The build plan is checked on faked platforms everywhere; the executable
itself is checked on whichever platform is running the tests.

Nothing here downloads a model: the MLX check stops at the point where the
executable goes looking for weights.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from conftest import PHOTO

from melampus.config import load_config
from melampus.providers import on_apple_silicon

CONFTEST = Path(__file__).with_name("conftest.py")
# What `.venv/bin/melampus-id` runs, spelled so it works from any interpreter
# that has the package installed (CI has no root .venv).
VENV_CLI = [sys.executable, "-m", "melampus.cli"]


def _no_python_environment(tmp_path: Path) -> dict[str, str]:
    """An environment in which no python of any kind can be found."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    (tmp_path / "home").mkdir()
    env = {"PATH": str(empty), "HOME": str(tmp_path / "home")}
    if sys.platform == "win32":
        # A Windows process needs the system root to load system DLLs, and the
        # one-file executable unpacks itself into the temp directory. Neither
        # gives a python back.
        (tmp_path / "tmp").mkdir()
        env |= {
            "SYSTEMROOT": os.environ["SYSTEMROOT"],
            "USERPROFILE": env["HOME"],
            "TEMP": str(tmp_path / "tmp"),
            "TMP": str(tmp_path / "tmp"),
        }
    for name in ("python", "python3", "uv"):
        assert shutil.which(name, path=env["PATH"]) is None, f"{name} is on the test PATH"
    return env


def _analyze(command: list[str], photos: Path, workdir: Path, *, env: dict | None) -> list:
    """Run one invocation against the scripted backend; return what --json-out wrote."""
    workdir.mkdir(exist_ok=True)
    out = workdir / "results.json"
    proc = subprocess.run(
        [*command, str(photos), "--backend", "scripted",
         "--cache", str(workdir / "cache.jsonl"), "--json-out", str(out)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"{command[0]} failed:\n{proc.stderr[-3000:]}"
    return json.loads(out.read_text(encoding="utf-8"))


def test_repo_root_is_the_bundle_when_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path):
    """Inside the executable the package lives in PyInstaller's unpack directory,
    not under service/ in a checkout; `_repo_root()` is that directory when
    frozen and the checkout otherwise. What is resolved against it is the
    business of the tests on load_config's defaults."""
    from melampus import config

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert config._repo_root() == tmp_path
    monkeypatch.delattr(sys, "_MEIPASS")
    assert config._repo_root() == repo


def _fake_pyinstaller(monkeypatch: pytest.MonkeyPatch, run) -> None:
    """A PyInstaller whose `__main__.run` is `run`, whether or not the real one
    is installed."""
    package = types.ModuleType("PyInstaller")
    package.__main__ = types.ModuleType("PyInstaller.__main__")
    package.__main__.run = run
    monkeypatch.setitem(sys.modules, "PyInstaller", package)
    monkeypatch.setitem(sys.modules, "PyInstaller.__main__", package.__main__)


@pytest.fixture()
def build(build_script: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.ModuleType:
    """The build script (conftest loads it once), writing under tmp_path
    instead of the checkout's dist/ and build/, on a machine that counts as
    Apple Silicon."""
    monkeypatch.setattr(build_script, "DIST", tmp_path / "dist")
    monkeypatch.setattr(build_script, "WORK", tmp_path / "build" / "pyinstaller")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    return build_script


def test_build_says_pyinstaller_is_missing_and_where_the_install_is_documented(
    build, monkeypatch: pytest.MonkeyPatch, capsys
):
    """The install command is spelled once, in readme.md; the script points
    there rather than carrying a copy the docs gate cannot see drift."""
    monkeypatch.setitem(sys.modules, "PyInstaller", None)
    assert build.main() == 3
    message = capsys.readouterr().err
    assert "PyInstaller is not installed" in message
    assert "readme.md" in message and "Building the executable" in message
    assert "uv sync" not in message
    assert "uv sync" not in build.__doc__


def test_build_bundles_the_service_the_prompts_and_the_mlx_runtime(
    build, monkeypatch: pytest.MonkeyPatch, repo: Path
):
    """Done-when 1's ingredients, as the arguments handed to PyInstaller: one
    file, the service on the path, prompts/ at the bundle's top level, all of
    mlx (its native library and Metal shaders sit beside the module) and every
    submodule of the packages that import model code by name at run time —
    written to dist/melampus, no suffix."""
    calls: list[list[str]] = []

    def run(arguments: list[str]) -> None:
        calls.append(arguments)
        build.DIST.mkdir()
        (build.DIST / build.NAME).write_text("the executable", encoding="utf-8")

    _fake_pyinstaller(monkeypatch, run)
    assert build.main() == 0
    assert build.executable_path() == build.DIST / "melampus"
    (arguments,) = calls
    pairs = set(zip(arguments, arguments[1:]))
    assert "--onefile" in arguments
    assert ("--paths", str(repo / "service")) in pairs
    assert ("--add-data", f"{repo / 'prompts'}:prompts") in pairs
    assert ("--collect-all", "mlx") in pairs
    for package in ("mlx_vlm", "mlx_lm", "transformers"):
        assert ("--collect-submodules", package) in pairs
    entry = Path(arguments[-1])
    assert "from melampus.cli import main" in entry.read_text(encoding="utf-8")
    assert entry.is_relative_to(build.WORK)


def test_build_fails_when_pyinstaller_writes_nothing(build, monkeypatch: pytest.MonkeyPatch, capsys):
    _fake_pyinstaller(monkeypatch, lambda arguments: None)
    assert build.main() == 1
    assert "does not exist" in capsys.readouterr().err


# Shaped like the real script: conftest.py imports it for executable_path()
# and runs it as a program for the build.
FAKE_BUILD_SCRIPT = """\
import sys
from pathlib import Path

DIST = Path(__file__).resolve().parents[1] / "dist"


def executable_path():
    return DIST / "melampus"


if __name__ == "__main__":
    DIST.mkdir(exist_ok=True)
    executable_path().write_text("built by tools/build_binary.py", encoding="utf-8")
    raise SystemExit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
"""

SMOKE_TEST = """\
def test_smoke(built_executable):
    assert built_executable.read_text(encoding="utf-8") == "built by tools/build_binary.py"
"""


def _checkout(pytester: pytest.Pytester, build_script: str = FAKE_BUILD_SCRIPT) -> None:
    """A checkout laid out like this one, tools/build_binary.py and
    service/tests/conftest.py, where the build is a script that writes
    dist/melampus and one smoke test that uses it."""
    tools, tests = pytester.mkdir("tools"), pytester.mkdir("service").joinpath("tests")
    tests.mkdir()
    (tools / "build_binary.py").write_text(build_script, encoding="utf-8")
    shutil.copy(CONFTEST, tests / "conftest.py")
    (tests / "test_smoke.py").write_text(SMOKE_TEST, encoding="utf-8")


def test_the_test_command_builds_the_executable_before_the_smoke_tests(pytester: pytest.Pytester):
    """Done-when 3: given the repo's test command, when it runs, then the
    build and the executable smoke test are part of it. With --build-binary
    the fixture runs tools/build_binary.py and the smoke test gets its output."""
    _checkout(pytester)
    pytester.runpytest("service/tests", "--build-binary").assert_outcomes(passed=1)


def test_a_failed_build_fails_the_test_command(pytester: pytest.Pytester):
    _checkout(pytester, build_script=FAKE_BUILD_SCRIPT.replace("else 0", "else 1"))
    result = pytester.runpytest("service/tests", "--build-binary")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*CalledProcessError*build_binary.py*"])


def test_without_a_build_the_smoke_tests_skip_and_say_how_to_get_one(pytester: pytest.Pytester):
    _checkout(pytester)
    result = pytester.runpytest("service/tests", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*no executable at dist/melampus; build one with*--build-binary*"])


def _frozen(monkeypatch: pytest.MonkeyPatch, bundle: Path, executable: Path) -> None:
    """What PyInstaller's bootloader sets before the package runs: the unpack
    directory, and the executable itself as sys.executable."""
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))


def test_frozen_prompts_come_from_the_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The build script puts prompts/ at the top level of the unpack directory."""
    bundle, executable = tmp_path / "unpack", tmp_path / "dist" / "melampus"
    _frozen(monkeypatch, bundle, executable)
    assert load_config(use_local=False).run.prompts_dir == bundle / "prompts"


def test_frozen_user_data_sits_beside_the_executable_not_in_the_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The unpack directory is deleted when the process exits, so a cache
    written there is thrown away and a melampus.local.toml there is never read.
    User data lives beside the executable, as it lives beside the code in a
    checkout."""
    tmp_path = tmp_path.resolve()
    bundle, executable = tmp_path / "unpack", tmp_path / "dist" / "melampus"
    executable.parent.mkdir()
    (executable.parent / "melampus.local.toml").write_text('[run]\nprofile = "sport"\n', encoding="utf-8")
    _frozen(monkeypatch, bundle, executable)
    config = load_config()
    cache = executable.parent / ".melampus_cache"
    assert config.run.cache_path == cache / "identifications.jsonl"
    assert config.occurrence.cache_path == cache / "occurrence.json"
    assert config.escalation.cache_path == cache / "escalations.jsonl"
    assert config.run.profile == "sport", "melampus.local.toml beside the executable was not read"


def test_build_plan_on_windows_names_the_exe_and_leaves_mlx_out(
    build_script: types.ModuleType, monkeypatch: pytest.MonkeyPatch, repo: Path
):
    """Card #400: the same build script, run on a Windows runner, must write
    dist/melampus.exe and not ask PyInstaller to collect mlx (there is no such
    package there, and PyInstaller refuses to collect a package it cannot
    find)."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    assert build_script.executable_path() == repo / "dist" / "melampus.exe"
    arguments = build_script.pyinstaller_arguments(Path("entry.py"))
    assert "--collect-all" not in arguments
    assert not any(a.startswith("mlx") for a in arguments), arguments
    assert "--collect-submodules" not in arguments
    assert ("--add-data", f"{repo / 'prompts'}:prompts") in set(zip(arguments, arguments[1:]))


def _request_mlx(executable: Path, photos: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Ask the executable for the mlx backend, offline and with no weights."""
    env = _no_python_environment(tmp_path)
    env |= {"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    return subprocess.run(
        [str(executable), str(photos), "--backend", "mlx",
         "--cache", str(tmp_path / "cache.jsonl")],
        env=env, capture_output=True, text=True, timeout=600,
    )


@pytest.mark.skipif(not on_apple_silicon(), reason="MLX exists only on Apple Silicon")
def test_executable_carries_the_service_and_mlx(built_executable: Path, photos: Path, tmp_path: Path):
    """Done-when 1 (#399). Asked for the mlx backend with the HuggingFace cache
    empty and offline, the executable must get as far as looking for weights —
    which means mlx, mlx_vlm and transformers all import inside the bundle — and
    stop there. An executable that does not carry MLX dies on an import error first."""
    proc = _request_mlx(built_executable, photos, tmp_path)
    tail = proc.stderr[-3000:]
    assert proc.returncode != 0, "loaded a model with no weights and no network?"
    for missing in ("ModuleNotFoundError", "ImportError"):
        assert missing not in tail, f"the executable does not carry MLX:\n{tail}"
    assert "LocalEntryNotFoundError" in tail, f"did not get as far as looking for weights:\n{tail}"


@pytest.mark.skipif(on_apple_silicon(), reason="this machine can run MLX")
def test_executable_refuses_mlx_off_apple_silicon_and_names_what_works(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Done-when 2 (#400). The Windows executable asked for mlx says MLX needs
    Apple Silicon and names the backends that do work here, instead of dying on
    a missing module."""
    proc = _request_mlx(built_executable, photos, tmp_path)
    tail = proc.stderr[-3000:]
    assert proc.returncode != 0, "ran the mlx backend with no MLX?"
    assert "Apple Silicon" in tail, tail
    for works_here in ("anthropic", "openai", "scripted"):
        assert works_here in tail, f"{works_here!r} is not named as working here:\n{tail}"
    for missing in ("ModuleNotFoundError", "ImportError"):
        assert missing not in tail, f"the refusal came from a missing module, not the CLI:\n{tail}"


def test_executable_prints_the_same_json_as_the_cli_with_no_python_on_the_path(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Done-when 2 (#399) and Done-when 1 (#400), on whichever platform built
    the executable. Same folder, same fake backend, same JSON — from the
    executable alone, in an environment where no python exists."""
    expected = _analyze(VENV_CLI, photos, tmp_path / "venv", env=None)
    actual = _analyze(
        [str(built_executable)], photos, tmp_path / "binary",
        env=_no_python_environment(tmp_path),
    )
    assert [r["file"] for r in expected] == [PHOTO], "the CLI did not analyze the photo"
    assert actual == expected
