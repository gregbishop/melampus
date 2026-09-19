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

Nothing here downloads a model: the MLX check stops at the point where the
executable goes looking for weights.
"""

from __future__ import annotations

import importlib.util
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

from melampus.backend import ScriptedBackend
from melampus.config import load_config
from melampus.identify import Identifier

CONFTEST = Path(__file__).with_name("conftest.py")
# What `.venv/bin/melampus-id` runs, spelled so it works from any interpreter
# that has the package installed (CI has no root .venv).
VENV_CLI = [sys.executable, "-m", "melampus.cli"]


def _no_python_environment(tmp_path: Path) -> dict[str, str]:
    """An environment in which no python of any kind can be found."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    (tmp_path / "home").mkdir()
    return {"PATH": str(empty), "HOME": str(tmp_path / "home")}


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


def _build_script(repo: Path) -> types.ModuleType:
    """tools/build_binary.py imported as a module: tools/ is not a package."""
    spec = importlib.util.spec_from_file_location("build_binary", repo / "tools" / "build_binary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_pyinstaller(monkeypatch: pytest.MonkeyPatch, run) -> None:
    """A PyInstaller whose `__main__.run` is `run`, whether or not the real one
    is installed."""
    package = types.ModuleType("PyInstaller")
    package.__main__ = types.ModuleType("PyInstaller.__main__")
    package.__main__.run = run
    monkeypatch.setitem(sys.modules, "PyInstaller", package)
    monkeypatch.setitem(sys.modules, "PyInstaller.__main__", package.__main__)


@pytest.fixture()
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path) -> types.ModuleType:
    """The build script, writing under tmp_path instead of the checkout's
    dist/ and build/, on a machine that counts as Apple Silicon."""
    script = _build_script(repo)
    monkeypatch.setattr(script, "DIST", tmp_path / "dist")
    monkeypatch.setattr(script, "WORK", tmp_path / "build" / "pyinstaller")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    return script


def test_build_refuses_anything_but_apple_silicon(build, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert build.main() == 2
    assert "Apple Silicon" in capsys.readouterr().err


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
    submodule of the packages that import model code by name at run time."""
    calls: list[list[str]] = []

    def run(arguments: list[str]) -> None:
        calls.append(arguments)
        build.DIST.mkdir()
        (build.DIST / build.NAME).write_text("the executable", encoding="utf-8")

    _fake_pyinstaller(monkeypatch, run)
    assert build.main() == 0
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


FAKE_BUILD_SCRIPT = """\
import sys
from pathlib import Path

dist = Path(__file__).resolve().parents[1] / "dist"
dist.mkdir(exist_ok=True)
(dist / "melampus").write_text("built by tools/build_binary.py", encoding="utf-8")
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
    monkeypatch.delenv("MELAMPUS_LOCAL_CONFIG", raising=False)
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


def test_melampus_local_config_names_the_local_file_in_a_checkout_and_in_the_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The CLI reads the checkout's melampus.local.toml and the executable
    reads dist/'s, so a smoke test that compares the two would otherwise be
    comparing a developer's settings with the defaults. With
    MELAMPUS_LOCAL_CONFIG set, both read the file it names and neither reads
    its own."""
    tmp_path = tmp_path.resolve()
    named = tmp_path / "synthetic.toml"
    named.write_text('[run]\nprofile = "sport"\n', encoding="utf-8")
    monkeypatch.setenv("MELAMPUS_LOCAL_CONFIG", str(named))
    assert load_config().run.profile == "sport", "the checkout did not read the named file"

    bundle, executable = tmp_path / "unpack", tmp_path / "dist" / "melampus"
    executable.parent.mkdir()
    (executable.parent / "melampus.local.toml").write_text("[run]\nmax_retries = 5\n", encoding="utf-8")
    _frozen(monkeypatch, bundle, executable)
    config = load_config()
    assert config.run.profile == "sport", "the executable did not read the named file"
    assert config.run.max_retries == 1, "the executable read melampus.local.toml beside itself as well"
    assert load_config(use_local=False).run.profile == "wildlife", "use_local=False must still skip it"


def test_executable_carries_the_service_and_mlx(built_executable: Path, photos: Path, tmp_path: Path):
    """Done-when 1. Asked for the mlx backend with the HuggingFace cache empty
    and offline, the executable must get as far as looking for weights — which
    means mlx, mlx_vlm and transformers all import inside the bundle — and stop
    there. An executable that does not carry MLX dies on an import error first."""
    env = _no_python_environment(tmp_path)
    env |= {"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    proc = subprocess.run(
        [str(built_executable), str(photos), "--backend", "mlx",
         "--cache", str(tmp_path / "cache.jsonl")],
        env=env, capture_output=True, text=True, timeout=600,
    )
    tail = proc.stderr[-3000:]
    assert proc.returncode != 0, "loaded a model with no weights and no network?"
    for missing in ("ModuleNotFoundError", "ImportError"):
        assert missing not in tail, f"the executable does not carry MLX:\n{tail}"
    assert "LocalEntryNotFoundError" in tail, f"did not get as far as looking for weights:\n{tail}"


# Settings that reach the JSON — max_tokens through run_fingerprint, max_retries
# through retries — at values that are not the defaults, so the comparison also
# shows both processes read this file and not their own.
SYNTHETIC_LOCAL_CONFIG = "[model]\nmax_tokens = 700\n\n[run]\nmax_retries = 2\n"


def test_executable_prints_the_same_json_as_the_cli_with_no_python_on_the_path(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Done-when 2. Same folder, same fake backend, same JSON — from the
    executable alone, in an environment where no python exists.

    Both read one synthetic melampus.local.toml, not the checkout's and
    dist/'s: a developer's own settings would otherwise decide whether the
    two agree, since they change the fingerprint and the retry count."""
    local = tmp_path / "melampus.local.toml"
    local.write_text(SYNTHETIC_LOCAL_CONFIG, encoding="utf-8")
    isolated = {"MELAMPUS_LOCAL_CONFIG": str(local)}
    expected = _analyze(VENV_CLI, photos, tmp_path / "venv", env=os.environ | isolated)
    actual = _analyze(
        [str(built_executable)], photos, tmp_path / "binary",
        env=_no_python_environment(tmp_path) | isolated,
    )
    reference = Identifier(ScriptedBackend([]), load_config(local, use_local=False)).identify(photos / PHOTO)
    assert [r["file"] for r in expected] == [PHOTO], "the CLI did not analyze the photo"
    assert (expected[0]["run_fingerprint"], expected[0]["retries"]) == (
        reference.run_fingerprint, reference.retries
    ), "the CLI did not read the synthetic melampus.local.toml"
    assert actual == expected
