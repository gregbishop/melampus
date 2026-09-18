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

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from melampus.config import load_config

# The frame test_quality.py leans on; any corpus JPEG would do.
FIXTURE = Path("fixtures") / "0A1A2829.jpg"
# What `.venv/bin/melampus-id` runs, spelled so it works from any interpreter
# that has the package installed (CI has no root .venv).
VENV_CLI = [sys.executable, "-m", "melampus.cli"]


@pytest.fixture()
def photos(tmp_path: Path, repo: Path) -> Path:
    fixture = repo / FIXTURE
    if not fixture.is_file():
        pytest.skip("corpus fixtures not present")
    folder = tmp_path / "photos"
    folder.mkdir()
    shutil.copy(fixture, folder / fixture.name)
    return folder


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


def test_executable_prints_the_same_json_as_the_cli_with_no_python_on_the_path(
    built_executable: Path, photos: Path, tmp_path: Path
):
    """Done-when 2. Same folder, same fake backend, same JSON — from the
    executable alone, in an environment where no python exists."""
    expected = _analyze(VENV_CLI, photos, tmp_path / "venv", env=None)
    actual = _analyze(
        [str(built_executable)], photos, tmp_path / "binary",
        env=_no_python_environment(tmp_path),
    )
    assert [r["file"] for r in expected] == [FIXTURE.name], "the CLI did not analyze the fixture"
    assert actual == expected
