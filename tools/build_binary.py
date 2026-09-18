"""Build one `melampus` executable that carries the service and MLX (card #399).

    .venv/bin/python tools/build_binary.py

Writes `dist/melampus` (git-ignored, alongside the `build/` scratch tree). It is
a PyInstaller one-file bundle: the Python runtime, the service package, the
prompts, and the MLX runtime with its Metal library, all unpacked to a
temporary directory at launch. Model weights are not bundled — they come from
the HuggingFace cache, as before.

Needs the `build` extra, installed from the lockfile like everything else:
`VIRTUAL_ENV=.venv uv sync --project service --locked --extra dev --extra build --active`.
Apple Silicon only, like the runtime it packages. The repo's test command
runs this and then the smoke tests: `.venv/bin/python -m pytest -q --build-binary`.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"
WORK = REPO / "build" / "pyinstaller"
NAME = "melampus"

# Packages PyInstaller's static analysis cannot see the whole of: mlx loads its
# native library and Metal shaders from files beside the module; mlx_vlm, mlx_lm
# and transformers import model modules by name at run time.
COLLECT_ALL = ("mlx",)
COLLECT_SUBMODULES = ("mlx_vlm", "mlx_lm", "transformers")


def main() -> int:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("the executable carries MLX, which only exists on Apple Silicon", file=sys.stderr)
        return 2
    try:
        import PyInstaller.__main__
    except ImportError:
        print(
            "PyInstaller is not installed. Run:\n"
            "  VIRTUAL_ENV=.venv uv sync --project service --locked "
            "--extra dev --extra build --active",
            file=sys.stderr,
        )
        return 3

    # cli.py uses relative imports, so it cannot be the entry script itself.
    WORK.mkdir(parents=True, exist_ok=True)
    entry = WORK / f"{NAME}_entry.py"
    entry.write_text("from melampus.cli import main\n\nraise SystemExit(main())\n", encoding="utf-8")

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
    for package in COLLECT_ALL:
        arguments += ["--collect-all", package]
    for package in COLLECT_SUBMODULES:
        arguments += ["--collect-submodules", package]
    arguments.append(str(entry))

    PyInstaller.__main__.run(arguments)

    built = DIST / NAME
    if not built.is_file():
        print(f"build finished but {built} does not exist", file=sys.stderr)
        return 1
    print(f"built {built} ({built.stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
