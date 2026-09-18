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

REPO = Path(__file__).resolve().parents[2]
# The frame test_quality.py leans on; any corpus JPEG would do.
FIXTURE = REPO / "fixtures" / "0A1A2829.jpg"
# What `.venv/bin/melampus-id` runs, spelled so it works from any interpreter
# that has the package installed (CI has no root .venv).
VENV_CLI = [sys.executable, "-m", "melampus.cli"]



def test_frozen_defaults_come_from_the_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Inside the executable the package lives in PyInstaller's unpack directory,
    not under service/ in a checkout, so the prompts are found relative to that
    directory: the build script puts them at its top level as `prompts/`."""
    from melampus import config

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert config._repo_root() == tmp_path
    monkeypatch.delattr(sys, "_MEIPASS")
    assert config._repo_root() == REPO

