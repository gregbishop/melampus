"""Package the plugin and its executable as the release zip (card #402).

    .venv/bin/python tools/package_plugin.py            # macOS
    .venv\\Scripts\\python.exe tools\\package_plugin.py   # Windows

Writes `dist/Melampus-macOS.zip` (or `dist/Melampus-Windows.zip`), which
unpacks to a single `Melampus.lrplugin/` folder: the plugin's files as they are
in `plugin/Melampus.lrplugin/`, with `dist/melampus` (`melampus.exe`) at its
root, where the plugin looks for it (card #401). A user adds that folder in
Lightroom's Plug-in Manager and is done. The executable's mode is kept in the
zip, so it is still executable after unzip or Archive Utility unpacks it.

This is the one place that knows the layout: the release workflow runs it after
the build, and the test suite checks its output. It needs the executable first
(readme.md § Building the executable) and refuses otherwise.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# tools/ is not a package; the build script beside this one is the source of
# truth for where the executable lands.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_binary  # noqa: E402

PLUGIN = build_binary.REPO / "plugin" / "Melampus.lrplugin"
FOLDER = PLUGIN.name
ZIP_NAMES = {"darwin": "Melampus-macOS.zip", "win32": "Melampus-Windows.zip"}
# A copy of either executable left in the checkout's plugin folder by a by-hand
# install (readme.md § Reviewing in Lightroom) is never what ships.
EXECUTABLE_NAMES = {build_binary.NAME, f"{build_binary.NAME}.exe"}


def executable_path() -> Path:
    return build_binary.executable_path()


def zip_path() -> Path:
    """The zip lands in dist/ beside the executable, named for the platform."""
    try:
        return executable_path().with_name(ZIP_NAMES[sys.platform])
    except KeyError:
        raise SystemExit(f"no release zip for {sys.platform}: the releases are macOS and Windows")


def plugin_files(plugin_dir: Path = PLUGIN) -> list[Path]:
    """The plugin's own files: everything under the folder that is not hidden
    (Finder's .DS_Store, an editor's cache) and not an executable copy."""
    return sorted(
        path for path in plugin_dir.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(plugin_dir).parts)
        and path.name not in EXECUTABLE_NAMES
    )


def package(executable: Path, target: Path, plugin_dir: Path = PLUGIN) -> list[str]:
    """Write `target`; return its entries in order. ZipFile.write keeps each
    file's mode in the entry, which is what unzip restores."""
    entries = [(path, f"{FOLDER}/{path.relative_to(plugin_dir).as_posix()}") for path in plugin_files(plugin_dir)]
    entries.append((executable, f"{FOLDER}/{executable.name}"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in entries:
            archive.write(path, name)
    return [name for _, name in entries]


def main() -> int:
    executable = executable_path()
    if not executable.is_file():
        print(
            f"no executable at {executable}: build it first, as readme.md "
            "§ Building the executable says",
            file=sys.stderr,
        )
        return 1
    target = zip_path()
    listing = package(executable, target)
    print(f"wrote {target} ({target.stat().st_size / 2**20:.0f} MB)")
    print(*listing, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
