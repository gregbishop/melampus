"""Session-wide wiring for the shipped executable (card #399).

`--build-binary` makes the repo's own test command build the executable before
the binary smoke tests run, so the build is part of the test command without
costing every unit-test run the minutes a PyInstaller build takes. Without the
option the smoke tests use an existing build, or skip and say how to get one.

The build script is the one source of truth for where the executable lands
(`dist/melampus`, or `dist/melampus.exe` on Windows — card #400), so it is
loaded here rather than having its answer restated.

`photos` is the one-frame folder the scripted backend is run against, from
the executable and from the CLI alike.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

# pytester runs a pytest inside pytest: how test_binary.py proves what this
# file's option and fixture do without a real build.
pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO / "tools" / "build_binary.py"
# The frame test_quality.py leans on, downscaled to 1200 px and stripped of
# metadata so it can be committed: the corpus is gitignored and CI has none,
# and the smoke test must analyze the same image on every platform (card #400).
FIXTURE = Path(__file__).with_name("fixtures") / "0A1A2829.jpg"
PHOTO = FIXTURE.name


def _load_build_script() -> ModuleType:
    """tools/ is not a package; import the script by path, without running it."""
    spec = importlib.util.spec_from_file_location("build_binary", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_build_script()
EXECUTABLE: Path = BUILD.executable_path()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--build-binary",
        action="store_true",
        default=False,
        help=f"build {EXECUTABLE.relative_to(REPO)} with "
             f"{BUILD_SCRIPT.relative_to(REPO)} before the binary smoke tests "
             "(takes minutes; without it they use an existing build or skip)",
    )


@pytest.fixture(scope="session")
def repo() -> Path:
    """The checkout root, for tests that reach outside service/ (fixtures/,
    tools/, the docs)."""
    return REPO


@pytest.fixture(scope="session")
def build_script() -> ModuleType:
    return BUILD


@pytest.fixture()
def photos(tmp_path: Path) -> Path:
    """A folder holding one JPEG, PHOTO: a copy of the committed frame."""
    folder = tmp_path / "photos"
    folder.mkdir()
    shutil.copy(FIXTURE, folder / PHOTO)
    return folder


@pytest.fixture(scope="session")
def built_executable(request: pytest.FixtureRequest) -> Path:
    if request.config.getoption("--build-binary"):
        subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=REPO, check=True)
    if not EXECUTABLE.is_file():
        pytest.skip(
            f"no executable at {EXECUTABLE.relative_to(REPO)}; build one with "
            "`.venv/bin/python -m pytest -q --build-binary` or "
            f"`.venv/bin/python {BUILD_SCRIPT.relative_to(REPO)}`"
        )
    return EXECUTABLE
