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

import sys
from pathlib import Path

from melampus.providers import on_apple_silicon

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"
WORK = REPO / "build" / "pyinstaller"
NAME = "melampus"

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

    PyInstaller.__main__.run(pyinstaller_arguments(entry))

    built = executable_path()
    if not built.is_file():
        print(f"build finished but {built} does not exist", file=sys.stderr)
        return 1
    print(f"built {built} ({built.stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
