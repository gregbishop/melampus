"""Build one `melampus` executable that carries the service (cards #399, #400).

    .venv/bin/python tools/build_binary.py            # macOS
    .venv\\Scripts\\python.exe tools\\build_binary.py   # Windows

Writes `dist/melampus` (`dist/melampus.exe` on Windows; git-ignored, alongside
the `build/` scratch tree). It is a PyInstaller one-file bundle: the Python
runtime, the service package and the prompts, all unpacked to a temporary
directory at launch. On Apple Silicon it also carries the MLX runtime with its
Metal library; model weights are not bundled and come from the HuggingFace
cache, as before. Everywhere else MLX does not exist, so the executable carries
everything but MLX and `--backend` selects a cloud provider (or scripted).

Needs the `build` extra, installed from the lockfile like everything else; the
command is in readme.md § Building the executable. The repo's test command
runs this and then the smoke tests: `.venv/bin/python -m pytest -q --build-binary`.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from melampus.providers import on_apple_silicon

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"
WORK = REPO / "build" / "pyinstaller"
NAME = "melampus"
# PyInstaller's switch for where it keeps its cache (bincache*/index.dat). By
# default that is one directory per user, `~/Library/Application Support/
# pyinstaller` on macOS, `%LOCALAPPDATA%\pyinstaller` on Windows, `~/.cache/
# pyinstaller` elsewhere (PyInstaller/configure.py, and the manual's "Supporting
# Multiple Operating Systems": "by default it uses a subdirectory of your home
# directory as its cache location"), shared by every checkout on the machine;
# two builds at once left index.dat half-written and the next build died
# reading it (card #440).
CONFIG_DIR_VARIABLE = "PYINSTALLER_CONFIG_DIR"

# Packages PyInstaller's static analysis cannot see the whole of: mlx loads its
# native library and Metal shaders from files beside the module; mlx_vlm, mlx_lm
# and transformers import model modules by name at run time. All of them exist
# only on Apple Silicon (the pyproject marker, `on_apple_silicon`), and
# PyInstaller refuses to collect a package it cannot find, so they are asked
# for only there.
COLLECT_ALL = ("mlx",)
COLLECT_SUBMODULES = ("mlx_vlm", "mlx_lm", "transformers")


def executable_path() -> Path:
    """Where PyInstaller puts the one-file build for this platform."""
    return DIST / (f"{NAME}.exe" if sys.platform == "win32" else NAME)


def config_dir(checkout: Path) -> Path:
    """Where this checkout's build keeps PyInstaller's cache: beside the work
    tree, under the git-ignored build/, so no two checkouts share one."""
    return checkout / "build" / "pyinstaller-config"


def _from_the_cache_index(error: BaseException) -> bool:
    """Whether PyInstaller failed reading its cache index: index.dat is Python
    source it evals (PyInstaller/utils/misc.py, load_py_data_struct), so a
    half-written one, from an interrupted or concurrent build, is a SyntaxError
    raised there. Any other SyntaxError is a module PyInstaller compiled."""
    return isinstance(error, SyntaxError) and any(
        frame.name == "load_py_data_struct" for frame in traceback.extract_tb(error.__traceback__)
    )


def pyinstaller_arguments(entry: Path) -> list[str]:
    arguments = [
        "--name", NAME,
        "--onefile",
        "--noconfirm",
        "--clean",
        "--distpath", str(DIST),
        "--workpath", str(WORK),
        "--specpath", str(WORK),
        "--paths", str(REPO / "service"),
        "--add-data", f"{REPO / 'prompts'}:prompts",
    ]
    if on_apple_silicon():
        for package in COLLECT_ALL:
            arguments += ["--collect-all", package]
        for package in COLLECT_SUBMODULES:
            arguments += ["--collect-submodules", package]
    arguments.append(str(entry))
    return arguments


def main() -> int:
    try:
        import PyInstaller.__main__
    except ImportError:
        print(
            "PyInstaller is not installed: install the build extra as "
            "readme.md § Building the executable says",
            file=sys.stderr,
        )
        return 3

    # cli.py uses relative imports, so it cannot be the entry script itself.
    WORK.mkdir(parents=True, exist_ok=True)
    entry = WORK / f"{NAME}_entry.py"
    entry.write_text("from melampus.cli import main\n\nraise SystemExit(main())\n", encoding="utf-8")

    # Set already, the caller's choice stands: CI, or a user who wants one
    # cache for every checkout, may point every build at the same directory.
    os.environ.setdefault(CONFIG_DIR_VARIABLE, str(config_dir(REPO)))
    try:
        PyInstaller.__main__.run(pyinstaller_arguments(entry))
    except SyntaxError as error:
        if not _from_the_cache_index(error):
            raise
        cache = os.environ[CONFIG_DIR_VARIABLE]
        print(
            f"PyInstaller's cache under {cache} is corrupt (index.dat: {error.msg}), "
            f"usually from a build that was interrupted; delete {cache} and re-run the build",
            file=sys.stderr,
        )
        return 2

    built = executable_path()
    if not built.is_file():
        print(f"build finished but {built} does not exist", file=sys.stderr)
        return 1
    print(f"built {built} ({built.stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
