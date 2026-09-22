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

Card #436 puts the plugin's enrichment pass inside the same executable.
Done-when 2: given the executable, when it runs with --plugin-out and
--backend scripted on the committed fixture, then the enriched JSON is written.
Done-when 3: given the occurrence cache and config, when the executable runs,
then they resolve under the per-user data directory, never the unpack directory.

Card #434: the executable carries the cloud SDKs on every platform. Done-when 2:
given the executable with no API key set, when `--backend claude` or
`--backend openai` runs, then it reaches the key check and says the key is
missing, not that the SDK is missing.

Card #403 names the engines mlx, ollama, openai, claude. The executable must
accept every one of them; `--backend ollama` with no server answering at the
configured address is refused as not running (card #406), naming the address
and the install, with the backends that do work here, exit 3.

Card #404: `--detect-engines` from the executable prints, as JSON, which of
the four can run on this machine and why or why not. The executable's mlx and
ollama verdicts are checked against the same two questions asked from the
test process (is this Apple Silicon; does a server answer at OLLAMA_URL over
loopback), so a developer's running Ollama decides nothing the test did not
measure too.

Card #407: `--download-model` from the executable fetches the model with
progress on stdout, from a fake hub on loopback, on every platform's build.

Card #420: `--backend command` with a program that is not installed is
refused before any image is read, naming the command, exit 3. Card #421:
`--backend claude-code` with no `claude` on PATH, likewise, naming where to
install it; card #422: `--backend codex` with no `codex`, the same.

Nothing here downloads a model: the MLX check stops at the point where the
executable goes looking for weights, and the download test's host is the fake
in conftest.py, so no real weights are ever fetched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest
from conftest import (
    FAKE_REPO,
    PHOTO,
    VENV_CLI,
    assert_download_completed,
    closed_port,
    fake_platform,
)

from melampus import config, providers
from melampus.backend import ScriptedBackend
from melampus.config import load_config
from melampus.identify import Identifier
from melampus.providers import on_apple_silicon

CONFTEST = Path(__file__).with_name("conftest.py")


def no_python_environment(tmp_path: Path) -> dict[str, str]:
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


def _analyze(command: list[str], photos: Path, workdir: Path, *, env: dict | None) -> tuple[list, list]:
    """Run one invocation against the scripted backend; return what --json-out
    and --plugin-out wrote (card #436: the enrichment runs in the same process)."""
    workdir.mkdir(exist_ok=True)
    out = workdir / "results.json"
    enriched = workdir / "plugin_results.json"
    proc = subprocess.run(
        [*command, str(photos), "--backend", "scripted",
         "--cache", str(workdir / "cache.jsonl"), "--json-out", str(out),
         "--plugin-out", str(enriched)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"{command[0]} failed:\n{proc.stderr[-3000:]}"
    return (json.loads(out.read_text(encoding="utf-8")),
            json.loads(enriched.read_text(encoding="utf-8")))


def test_repo_root_is_the_bundle_when_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path):
    """Inside the executable the package lives in PyInstaller's unpack directory,
    not under service/ in a checkout; `_repo_root()` is that directory when
    frozen and the checkout otherwise. What is resolved against it is the
    business of the tests on load_config's defaults."""
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
    fake_platform(monkeypatch, "darwin", "arm64")
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


def per_user_data_dir(home: Path) -> Path:
    """Where config._data_root() lands for the executable under this HOME, in a
    bare environment (no LOCALAPPDATA, no XDG_DATA_HOME)."""
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Melampus"
    if sys.platform == "win32":
        return home / "AppData" / "Local" / "Melampus"
    return home / ".local" / "share" / "Melampus"


def per_user_config(tmp_path: Path, toml: str) -> dict[str, str]:
    """An environment with no python and a fresh HOME whose per-user data
    directory holds `toml` as melampus.local.toml: the way a user configures
    the executable, and the only way the tests do."""
    env = no_python_environment(tmp_path)
    data_dir = per_user_data_dir(Path(env["HOME"]))
    data_dir.mkdir(parents=True)
    (data_dir / "melampus.local.toml").write_text(toml, encoding="utf-8")
    return env


def test_frozen_config_and_caches_live_in_the_per_user_data_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path
):
    """Card #436, Done-when 3. The unpack directory is temporary: a config put
    there is lost at the next launch and a cache written there is thrown away.
    Inside the executable melampus.local.toml and the caches resolve under the
    per-user data directory instead: ~/Library/Application Support/Melampus on
    macOS, %LOCALAPPDATA%\\Melampus on Windows (falling back to the profile's
    AppData\\Local when the variable is unset, as it is in a bare environment),
    $XDG_DATA_HOME/Melampus or ~/.local/share/Melampus elsewhere. In a checkout
    nothing moves: the repo root, as before."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "unpack"), raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert config._data_root() == home / "Library" / "Application Support" / "Melampus"
    monkeypatch.setattr(sys, "platform", "win32")
    assert config._data_root() == home / "AppData" / "Local" / "Melampus"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert config._data_root() == tmp_path / "local" / "Melampus"
    monkeypatch.setattr(sys, "platform", "linux")
    assert config._data_root() == home / ".local" / "share" / "Melampus"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert config._data_root() == tmp_path / "xdg" / "Melampus"
    for platform_name in ("darwin", "win32", "linux"):
        monkeypatch.setattr(sys, "platform", platform_name)
        assert not config._data_root().is_relative_to(tmp_path / "unpack")
        assert config.cache_file("x") == config._data_root() / "cache" / "x"

    monkeypatch.delattr(sys, "_MEIPASS")
    assert config._data_root() == repo
    assert config.cache_file("x") == repo / ".melampus_cache" / "x"


def test_frozen_user_data_lives_in_the_per_user_directory_not_in_the_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The frozen resolution above is what load_config's defaults use: the
    unpack directory is deleted when the process exits, so a cache written
    there is thrown away and a melampus.local.toml there is never read. Inside
    the executable the caches and the local config resolve under this
    platform's per-user data directory; prompts ship in the bundle and stay
    there. Only user data moves."""
    tmp_path = tmp_path.resolve()
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    data = per_user_data_dir(home)
    data.mkdir(parents=True)
    (data / "melampus.local.toml").write_text('[run]\nprofile = "sport"\n', encoding="utf-8")
    bundle, executable = tmp_path / "unpack", tmp_path / "dist" / "melampus"
    _frozen(monkeypatch, bundle, executable)
    settings = load_config()
    cache = data / "cache"
    assert settings.run.cache_path == cache / "identifications.jsonl"
    assert settings.occurrence.cache_path == cache / "occurrence.json"
    assert settings.escalation.cache_path == cache / "escalations.jsonl"
    assert settings.run.prompts_dir == bundle / "prompts"
    assert settings.run.profile == "sport", "melampus.local.toml under the per-user directory was not read"


def test_build_plan_on_windows_names_the_exe_and_leaves_mlx_out(
    build_script: types.ModuleType, monkeypatch: pytest.MonkeyPatch, repo: Path
):
    """Card #400: the same build script, run on a Windows runner, must write
    dist/melampus.exe and not ask PyInstaller to collect mlx (there is no such
    package there, and PyInstaller refuses to collect a package it cannot
    find)."""
    fake_platform(monkeypatch, "win32", "AMD64")
    assert build_script.executable_path() == repo / "dist" / "melampus.exe"
    arguments = build_script.pyinstaller_arguments(Path("entry.py"))
    assert "--collect-all" not in arguments
    assert not any(a.startswith("mlx") for a in arguments), arguments
    assert "--collect-submodules" not in arguments
    assert ("--add-data", f"{repo / 'prompts'}:prompts") in set(zip(arguments, arguments[1:]))


# Settings that reach the JSON — max_tokens through run_fingerprint, max_retries
# through retries — at values that are not the defaults, so a comparison also
# shows the process read this file and not its own melampus.local.toml.
SYNTHETIC_CONFIG = "[model]\nmax_tokens = 700\n\n[run]\nmax_retries = 2\n"


def _synthetic_config(tmp_path: Path, text: str) -> list[str]:
    """`text` written as a config file of the test's own, and the arguments
    that make it a run's whole configuration, for the CLI and the executable
    alike: --config <file> --no-local-config, so the melampus.local.toml
    beside the data is not read."""
    file = tmp_path / "synthetic.toml"
    file.write_text(text, encoding="utf-8")
    return ["--config", str(file), "--no-local-config"]


def _fingerprint_and_retries(text: str, photos: Path) -> tuple[str, int]:
    """What a run's JSON carries after reading `text` and nothing else: the
    fingerprint and retry count of an in-process Identifier on those settings
    as overrides, with no file read at all."""
    config = load_config(use_local=False, **tomllib.loads(text))
    result = Identifier(ScriptedBackend([]), config).identify(photos / PHOTO)
    return result.run_fingerprint, result.retries


def test_no_local_config_makes_the_config_file_the_whole_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path, photos: Path
):
    """The CLI reads the checkout's melampus.local.toml and the executable
    reads the per-user data directory's, so a smoke test that compares the two
    would otherwise be comparing a developer's settings with the defaults.
    `--config <file> --no-local-config` makes that file the whole
    configuration: it is read, and the melampus.local.toml under the data root
    is not. Run frozen with a fresh HOME, so the data root is a per-user
    directory of this test's own; prompts/ sits in the bundle as the build
    lays it out."""
    from melampus.cli import main

    tmp_path = tmp_path.resolve()
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    data = per_user_data_dir(home)
    data.mkdir(parents=True)
    # Reaches the fingerprint if read; the synthetic file leaves it alone.
    (data / "melampus.local.toml").write_text("[image]\nmax_edge = 640\n", encoding="utf-8")
    bundle, executable = tmp_path / "unpack", tmp_path / "dist" / "melampus"
    bundle.mkdir()
    (bundle / "prompts").symlink_to(repo / "prompts")
    _frozen(monkeypatch, bundle, executable)
    out = tmp_path / "results.json"
    code = main([
        str(photos), "--backend", "scripted", "--cache", str(tmp_path / "cache.jsonl"),
        "--json-out", str(out), *_synthetic_config(tmp_path, SYNTHETIC_CONFIG),
    ])
    assert code == 0
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert (result["run_fingerprint"], result["retries"]) == _fingerprint_and_retries(
        SYNTHETIC_CONFIG, photos
    ), "the run did not read --config alone: melampus.local.toml under the data root was read too"


# The model the mlx smoke tests look for: a synthetic repo, named in a
# synthetic config file, so the lookup depends neither on the default
# model being cached nor on a developer's own model.repo.
SYNTHETIC_MODEL = "melampus-tests/synthetic-model"


MISSING_MODULE = ("ModuleNotFoundError", "ImportError")


def _assert_no_missing_module(tail: str, what: str, signs: tuple[str, ...] = MISSING_MODULE) -> None:
    """The executable's stderr shows none of `signs`: it did not die on a
    module the bundle lacks. `what` says what that would have meant."""
    for missing in signs:
        assert missing not in tail, f"{what}:\n{tail}"


def _request_backend(
    executable: Path, photos: Path, tmp_path: Path, backend: str, *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Ask the executable for `backend` on the photos, with `arguments` after
    them, in `env`: a no-python environment, by default a fresh one."""
    return subprocess.run(
        [str(executable), str(photos), "--backend", backend,
         "--cache", str(tmp_path / "cache.jsonl"), *arguments],
        env=env or no_python_environment(tmp_path), capture_output=True, text=True, timeout=600,
    )


def _request_mlx(
    executable: Path, photos: Path, tmp_path: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Ask the executable for the mlx backend, with the HuggingFace cache empty
    and offline, and a synthetic config file naming SYNTHETIC_MODEL as its
    whole configuration. `env` is the no-python environment to run in; by
    default a fresh one."""
    env = (env or no_python_environment(tmp_path)) | {"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    return _request_backend(
        executable, photos, tmp_path, "mlx",
        *_synthetic_config(tmp_path, f'[model]\nrepo = "{SYNTHETIC_MODEL}"\n'), env=env,
    )


def _look_for_weights(
    executable: Path, photos: Path, tmp_path: Path, env: dict[str, str] | None = None
) -> str:
    """Run `executable` with the mlx backend as _request_mlx does and return
    the end of its stderr. It must fail: there are no weights and no network."""
    proc = _request_mlx(executable, photos, tmp_path, env)
    assert proc.returncode != 0, "loaded a model with no weights and no network?"
    return proc.stderr[-3000:]


@pytest.mark.skipif(not on_apple_silicon(), reason="MLX exists only on Apple Silicon")
def test_executable_carries_the_service_and_mlx(built_executable: Path, photos: Path, tmp_path: Path):
    """Done-when 1 (#399). Asked for the mlx backend with the HuggingFace cache
    empty and offline, the executable must get as far as looking for weights —
    which means mlx, mlx_vlm and transformers all import inside the bundle — and
    stop there. An executable that does not carry MLX dies on an import error first."""
    tail = _look_for_weights(built_executable, photos, tmp_path)
    _assert_no_missing_module(tail, "the executable does not carry MLX")
    assert "LocalEntryNotFoundError" in tail, f"did not get as far as looking for weights:\n{tail}"
    assert SYNTHETIC_MODEL in tail, f"did not look for the model the synthetic config file names:\n{tail}"


@pytest.mark.skipif(not on_apple_silicon(), reason="MLX exists only on Apple Silicon")
def test_the_mlx_smoke_test_ignores_a_local_model_in_the_per_user_directory(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """A user's melampus.local.toml under the per-user data directory may
    point model.repo at a directory of weights on disk, and mlx_vlm takes an
    existing directory as it is: an empty, offline HuggingFace cache does not
    stop the executable loading them. The lookup reads its synthetic file, not
    the per-user one. The HOME is this test's own, so the developer's is never
    read or written."""
    weights = tmp_path / "weights"
    weights.mkdir()
    env = per_user_config(tmp_path, f'[model]\nrepo = "{weights}"\n')
    tail = _look_for_weights(built_executable, photos, tmp_path, env)
    assert str(weights) not in tail, f"read melampus.local.toml under the per-user directory:\n{tail}"
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
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in tail, f"{works_here!r} is not named as working here:\n{tail}"
    _assert_no_missing_module(tail, "the refusal came from a missing module, not the CLI")


@pytest.mark.parametrize(
    ("backend", "needs_a_key"),
    [("claude", "The Claude backend needs an API key. Set MELAMPUS_ANTHROPIC_KEY"),
     ("openai", "The OpenAI backend needs an API key. Set MELAMPUS_OPENAI_KEY")],
)
def test_executable_carries_the_cloud_sdks_and_asks_for_the_key(
    built_executable: Path, photos: Path, tmp_path: Path, backend: str, needs_a_key: str
):
    """Card #434, Done-when 2 (and 3: the same test runs on every platform's
    build). Given the executable with no API key set, when a cloud backend is
    requested, then it gets as far as the key check and says the key is
    missing: the SDK imported inside the bundle. An executable built without
    the cloud and openai extras stops one step earlier, on the CLI's install
    hint, which means nothing to a user who has no venv to install into. The
    environment carries no key variable, so nothing is sent anywhere."""
    proc = _request_backend(built_executable, photos, tmp_path, backend)
    tail = proc.stderr[-3000:]
    assert proc.returncode == 3, f"exit {proc.returncode}:\n{tail}"
    _assert_no_missing_module(
        tail, f"the executable does not carry the {backend} SDK", ("is not installed. Run:", *MISSING_MODULE)
    )
    assert needs_a_key in tail, f"did not reach the key check:\n{tail}"


def test_executable_refuses_ollama_when_no_server_answers(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Card #406, Done-when 2 in the frozen build: with no Ollama server at
    the address (a closed port on loopback, so a developer's running Ollama
    cannot answer), `--backend ollama` exits 3 on the not-running message,
    naming the address tried and where to install Ollama, and the backends
    that do work here."""
    port = closed_port()
    settings = tmp_path / "settings.toml"
    settings.write_text(f'[model]\nollama_url = "http://127.0.0.1:{port}"\n', encoding="utf-8")
    proc = _request_backend(built_executable, photos, tmp_path, "ollama", "--config", str(settings))
    tail = proc.stderr[-3000:]
    assert proc.returncode == 3, f"exit {proc.returncode}:\n{tail}"
    assert "invalid choice" not in tail, f"the executable does not accept ollama:\n{tail}"
    assert f"No Ollama server at http://127.0.0.1:{port}" in tail, tail
    assert "https://ollama.com/download" in tail, tail
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in tail, f"{works_here!r} is not named as working here:\n{tail}"


def test_executable_refuses_a_command_that_is_not_installed(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Card #420, Done-when 2 in the frozen build: `[model] backend =
    "command"` naming a program nothing on this machine is called, and the
    executable exits 3 on the not-installed message, naming the command and
    the backends that do work here, before any image is read."""
    settings = tmp_path / "settings.toml"
    settings.write_text(
        '[model]\nbackend = "command"\n'
        'command = ["melampus-no-such-command-420", "{image}", "{prompt}"]\n',
        encoding="utf-8",
    )
    proc = _request_backend(built_executable, photos, tmp_path, "command", "--config", str(settings))
    tail = proc.stderr[-3000:]
    assert proc.returncode == 3, f"exit {proc.returncode}:\n{tail}"
    assert "invalid choice" not in tail, f"the executable does not accept command:\n{tail}"
    assert "'melampus-no-such-command-420' is not installed or not on PATH" in tail, tail
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in tail, f"{works_here!r} is not named as working here:\n{tail}"


@pytest.mark.parametrize(
    "cli", [providers.CLAUDE_CODE_CLI, providers.CODEX_CLI], ids=lambda cli: cli.engine
)
def test_executable_refuses_a_cli_engine_that_is_not_installed(
    built_executable: Path, photos: Path, tmp_path: Path, cli: providers.CliEngine
):
    """Cards #421 and #422, Done-when 2 in the frozen build: `--backend
    <engine>` on a PATH with no `<program>`, and the executable exits 3 on
    the not-installed message, naming the CLI, where to install it and how
    to sign in, and the backends that do work here, before any image is
    read. One test per CliEngine: the refusal's words are its fields."""
    env = no_python_environment(tmp_path)
    assert shutil.which(cli.program, path=env["PATH"]) is None
    proc = _request_backend(built_executable, photos, tmp_path, cli.engine, env=env)
    tail = proc.stderr[-3000:]
    assert proc.returncode == 3, f"exit {proc.returncode}:\n{tail}"
    assert "invalid choice" not in tail, f"the executable does not accept {cli.engine}:\n{tail}"
    assert f"{cli.title} is not installed" in tail, tail
    assert cli.install in tail and cli.sign_in in tail, tail
    for works_here in ("claude", "openai", "scripted"):
        assert works_here in tail, f"{works_here!r} is not named as working here:\n{tail}"


def test_executable_prints_the_same_json_as_the_cli_with_no_python_on_the_path(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Done-when 2 (#399) and Done-when 1 (#400), on whichever platform built
    the executable. Same folder, same fake backend, same JSON — from the
    executable alone, in an environment where no python exists.

    Both read one synthetic config file, not the checkout's and the per-user
    directory's melampus.local.toml: a developer's own settings would otherwise
    decide whether the two agree, since they change the fingerprint and the
    retry count."""
    isolated = _synthetic_config(tmp_path, SYNTHETIC_CONFIG)
    expected, expected_enriched = _analyze([*VENV_CLI, *isolated], photos, tmp_path / "venv", env=None)
    actual, actual_enriched = _analyze(
        [str(built_executable), *isolated], photos, tmp_path / "binary",
        env=no_python_environment(tmp_path),
    )
    assert [r["file"] for r in expected] == [PHOTO], "the CLI did not analyze the photo"
    assert (expected[0]["run_fingerprint"], expected[0]["retries"]) == _fingerprint_and_retries(
        SYNTHETIC_CONFIG, photos
    ), "the CLI did not read the synthetic config file"
    assert actual == expected
    assert actual_enriched == expected_enriched


def test_executable_writes_the_enriched_results_and_reads_config_from_the_per_user_directory(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Card #436, Done-when 2 and 3. Given the executable, when it runs with
    --plugin-out and --backend scripted on the committed fixture, then the
    enriched JSON the plugin reads is written, with all six fields — from the
    executable alone, no python on the path. The identification it enriches
    comes from a cache named in melampus.local.toml under the per-user data
    directory of a fresh HOME, which is the proof that config resolves there and
    not in the temporary unpack directory. No default location is configured, so
    the range check is skipped and nothing touches the network."""
    from melampus.plugin_results import PLUGIN_FIELDS
    from test_plugin_results import HERON, seed

    seeded = tmp_path / "seeded.jsonl"
    seed(seeded, [photos / PHOTO], {PHOTO: HERON})
    env = per_user_config(tmp_path, f"[run]\ncache_path = '{seeded.as_posix()}'\n")
    out = tmp_path / "plugin_results.json"

    proc = subprocess.run(
        [str(built_executable), str(photos), "--backend", "scripted", "--report-only",
         "--plugin-out", str(out)],
        env=env, capture_output=True, text=True, timeout=600,
    )

    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "no default location configured" in proc.stderr, proc.stderr[-3000:]
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [r["file"] for r in rows] == [PHOTO]
    row = rows[0]
    assert set(PLUGIN_FIELDS) <= set(row), f"missing {set(PLUGIN_FIELDS) - set(row)}"
    assert row["identification"]["candidates"][0]["common_name"] == "Tricolored Heron", (
        "the executable did not read melampus.local.toml from the per-user directory")
    assert row["burst_agreement"] == 1.0 and row["range_flag"] is False
    assert row["encounter"] == 0 and row["encounter_frames"] == 1 and row["quality_rank"] == 0.0
    assert 0 < row["quality"] <= 100, "quality was not scored on the pixels"


def test_executable_detects_engines_as_json_with_no_python_on_the_path(
    built_executable: Path, tmp_path: Path
):
    """Card #404, from the executable alone: valid JSON on stdout, the four
    engines in the owner's order then claude-code and codex (cards #421,
    #422), exit 0, no folder needed. Each local verdict mirrors the machine
    running the suite, never a guess about it: mlx's is whether this is Apple
    Silicon, ollama's is whether a server answers at OLLAMA_URL, asked from
    this process over loopback (Done-when 4: a developer's running Ollama
    decides nothing the test did not measure too), with the install pointer
    when none does; the cloud engines are available and name their key
    variable; with no `claude` or `codex` on the PATH, both CLIs are not
    installed, with where to get them. Every verdict carries the title the
    plugin's picker shows (card #423), so the dialog holds no title table of
    its own, and, where going and installing it is the fix, the install page
    the picker links to (review round 9, finding 1), so the dialog scrapes no
    address out of prose."""
    proc = subprocess.run(
        [str(built_executable), "--detect-engines"],
        env=no_python_environment(tmp_path), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    verdicts = json.loads(proc.stdout)
    assert [v["engine"] for v in verdicts] == [
        "mlx", "ollama", "openai", "claude", "claude-code", "codex"]
    assert all(
        set(v) == {"engine", "title", "available", "reason", "install"} for v in verdicts), verdicts
    assert all(v["title"] for v in verdicts), "a verdict with no title for the picker"
    by_engine = {v["engine"]: v for v in verdicts}
    assert by_engine["claude-code"]["title"].startswith(providers.CLAUDE_CODE_CLI.title)
    assert by_engine["codex"]["title"].startswith(providers.CODEX_CLI.title)
    assert by_engine["claude-code"]["available"] is False, "a claude on the empty PATH?"
    assert "not installed" in by_engine["claude-code"]["reason"]
    assert providers.CLAUDE_CODE_INSTALL in by_engine["claude-code"]["reason"]
    assert by_engine["claude-code"]["install"] == providers.CLAUDE_CODE_INSTALL
    assert by_engine["codex"]["available"] is False, "a codex on the empty PATH?"
    assert "not installed" in by_engine["codex"]["reason"]
    assert providers.CODEX_INSTALL in by_engine["codex"]["reason"]
    assert by_engine["codex"]["install"] == providers.CODEX_INSTALL
    assert by_engine["mlx"]["available"] is on_apple_silicon()
    if not on_apple_silicon():
        assert by_engine["mlx"]["reason"] == "needs Apple Silicon"
    ollama_here = providers.ollama_answers()
    assert by_engine["ollama"]["available"] is ollama_here
    if not ollama_here:
        assert providers.OLLAMA_INSTALL in by_engine["ollama"]["reason"]
        assert by_engine["ollama"]["install"] == providers.OLLAMA_INSTALL
    for engine in ("openai", "claude"):
        assert by_engine[engine]["available"] is True
        assert "API key required" in by_engine[engine]["reason"]
        assert providers.KEY_VARIABLES[engine][0] in by_engine[engine]["reason"]
    # Nothing to go and install: the picker offers no link for these, whatever
    # address their reason happens to name (review round 9, finding 1).
    for engine in ("mlx", "openai", "claude"):
        assert by_engine[engine]["install"] == ""


def test_executable_downloads_the_model_from_the_hub_with_no_python_on_the_path(
    built_executable: Path, hub_env: dict[str, str], tmp_path: Path
):
    """Card #407, from the executable alone, on whichever platform built it:
    the hub library is in the bundle (on Windows only because pyproject names
    it), the protocol lines come out on stdout, exit 0 with `done <path>`, and
    the path under HF_HOME holds the fake host's files byte for byte."""
    env = no_python_environment(tmp_path) | hub_env
    proc = subprocess.run(
        [str(built_executable), "--download-model", "--model", FAKE_REPO],
        env=env, capture_output=True, text=True, timeout=600,
    )
    tail = proc.stderr[-3000:]
    _assert_no_missing_module(tail, "the executable does not carry the hub library")
    assert proc.returncode == 0, f"exit {proc.returncode}:\n{tail}"
    assert_download_completed(proc.stdout, hub_env)


def test_executable_reports_ollamas_model_absent_with_no_server_and_exits_0(
    built_executable: Path, tmp_path: Path
):
    """Card #409, from the executable alone: `--model-status --backend
    ollama` with nothing answering at the configured address (a closed
    port on loopback, so a developer's Ollama cannot answer) prints one
    JSON object saying the model is absent with the size unknown, exit 0,
    so the Settings dialog opens with Ollama down. Nothing is pulled."""
    port = closed_port()
    settings = tmp_path / "settings.toml"
    settings.write_text(f'[model]\nollama_url = "http://127.0.0.1:{port}"\n', encoding="utf-8")
    proc = subprocess.run(
        [str(built_executable), "--model-status", "--backend", "ollama", "--config", str(settings)],
        env=no_python_environment(tmp_path), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"exit {proc.returncode}:\n{proc.stderr[-3000:]}"
    status = json.loads(proc.stdout)
    assert status["repo"] == "qwen3-vl:8b-instruct" and status["installed"] is False
    assert status["bytes_total"] is None and status["bytes_done"] == 0 and status["path"] is None
    assert status["cancel_path"].endswith("download-cancel")
    assert "Traceback" not in proc.stderr
