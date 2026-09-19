"""The release zip (card #402): one zip per platform, unpacking to the plugin
folder with its executable inside.

Done-when 1: given a tag is pushed, when the release workflow runs, then a
GitHub release exists with Melampus-macOS.zip and Melampus-Windows.zip
attached, each holding the plugin and its executable. The workflow itself is
gated in test_docs.py; what it packages is tools/package_plugin.py, the one
place that knows the zip's layout, checked here on a fake executable and,
with --build-binary (or an existing dist/melampus), on the real one.
Done-when 2: given either zip, when unpacked and added in Plug-in Manager,
then the plugin loads and its settings open. Plug-in Manager needs Lightroom;
the loadable part is checked without it: Info.lua parses, and every file it
names, and every module those files require, is in the zip listing
(plugin/tests/test_info.lua). Opening it in Lightroom stays a manual check.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from test_binary import no_python_environment, per_user_data_dir
from test_lua_plugin import TESTS, assert_scripted_results, run_lua_suite

FOLDER = "Melampus.lrplugin"


def _fake_plugin(tmp_path: Path) -> Path:
    """A plugin folder as a developer's checkout leaves it: the Lua files, a
    Finder droppings file, a stale copy of each platform's executable from an
    earlier by-hand install, and a cache folder from an editor."""
    plugin = tmp_path / FOLDER
    plugin.mkdir()
    for name in ("Info.lua", "MelampusInit.lua", "MelampusRules.lua"):
        (plugin / name).write_text(f"-- {name}\n", encoding="utf-8")
    (plugin / ".DS_Store").write_bytes(b"\0")
    (plugin / "melampus").write_text("stale", encoding="utf-8")
    (plugin / "melampus.exe").write_text("stale", encoding="utf-8")
    (plugin / ".cache").mkdir()
    (plugin / ".cache" / "x").write_text("x", encoding="utf-8")
    return plugin


def _fake_executable(tmp_path: Path, name: str = "melampus") -> Path:
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)
    executable = dist / name
    executable.write_text("#!/bin/sh\necho built\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _mode(info: zipfile.ZipInfo) -> int:
    """The POSIX mode a zip entry carries (what unzip restores)."""
    return (info.external_attr >> 16) & 0o7777


def test_zip_holds_the_plugin_folder_its_files_and_the_executable(package_script, tmp_path: Path):
    """Done-when 1's layout, the same on both platforms: one top-level
    Melampus.lrplugin/ holding the plugin's files and the executable at its
    root, nothing hidden, no stale executable from the checkout. The mode the
    zip carries is a POSIX property (a file named melampus has no execute bit
    on Windows), so it is checked in the unzip test below, not here."""
    plugin = _fake_plugin(tmp_path)
    executable = _fake_executable(tmp_path)
    target = tmp_path / "out" / "Melampus-macOS.zip"

    listing = package_script.package(executable, target, plugin_dir=plugin)

    assert listing == [
        f"{FOLDER}/Info.lua", f"{FOLDER}/MelampusInit.lua",
        f"{FOLDER}/MelampusRules.lua", f"{FOLDER}/melampus",
    ]
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == listing
        assert archive.read(f"{FOLDER}/melampus") == executable.read_bytes(), "the stale copy was packaged"


def test_zip_is_named_for_the_platform_and_holds_that_platform_s_executable(
    package_script, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Melampus-macOS.zip carries melampus; Melampus-Windows.zip carries
    melampus.exe; there is no release for any other platform."""
    monkeypatch.setattr(package_script.build_binary, "DIST", tmp_path / "dist")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert package_script.zip_path() == tmp_path / "dist" / "Melampus-macOS.zip"
    assert package_script.build_binary.executable_path().name == "melampus"
    monkeypatch.setattr(sys, "platform", "win32")
    assert package_script.zip_path() == tmp_path / "dist" / "Melampus-Windows.zip"
    assert package_script.build_binary.executable_path().name == "melampus.exe"
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(SystemExit, match="no release zip for linux"):
        package_script.zip_path()


def test_packaging_refuses_when_the_executable_is_absent(
    package_script, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    """No executable, no zip: the script says which file it wanted and where the
    build is documented, and writes nothing."""
    monkeypatch.setattr(package_script.build_binary, "DIST", tmp_path / "dist")
    monkeypatch.setattr(package_script, "PLUGIN", _fake_plugin(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert package_script.main() == 1
    message = capsys.readouterr().err
    assert str(tmp_path / "dist" / "melampus") in message
    assert "readme.md" in message and "Building the executable" in message
    assert not (tmp_path / "dist").exists()


def test_main_writes_the_platform_zip_and_prints_its_listing(
    package_script, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    """What the workflow runs: no arguments, the zip lands in dist/ beside the
    executable and the listing is printed so the run log shows what shipped.
    The folder packaged is the one PLUGIN names at call time (here the fake,
    not the checkout), so the listing is exactly the fake folder's files."""
    monkeypatch.setattr(package_script.build_binary, "DIST", tmp_path / "dist")
    monkeypatch.setattr(package_script, "PLUGIN", _fake_plugin(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    _fake_executable(tmp_path, "melampus.exe")
    assert package_script.main() == 0
    out = capsys.readouterr().out
    target = tmp_path / "dist" / "Melampus-Windows.zip"
    assert str(target) in out
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == [
            f"{FOLDER}/Info.lua", f"{FOLDER}/MelampusInit.lua",
            f"{FOLDER}/MelampusRules.lua", f"{FOLDER}/melampus.exe",
        ], "main() packaged a folder other than the one PLUGIN names"
        for name in archive.namelist():
            assert name in out


def _unpack(archive: Path, into: Path) -> None:
    """Unpack the way a user does: unzip on macOS (which restores the mode
    bits), the zip module where there is no unzip (Windows has no mode bits)."""
    into.mkdir()
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", str(archive), "-d", str(into)], check=True)
    else:
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(into)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no execute bit")
def test_unzip_restores_the_execute_bit_and_one_plugin_folder(package_script, tmp_path: Path):
    """Given the zip, when unpacked with unzip, then exactly one folder
    appears, Melampus.lrplugin, and the executable inside it is executable:
    the zip entry carries the executable's mode and no other file's, and
    unzip restores it."""
    if shutil.which("unzip") is None:
        pytest.skip("unzip not installed")
    plugin = _fake_plugin(tmp_path)
    target = tmp_path / "Melampus-macOS.zip"
    package_script.package(_fake_executable(tmp_path), target, plugin_dir=plugin)
    with zipfile.ZipFile(target) as archive:
        assert _mode(archive.getinfo(f"{FOLDER}/melampus")) & 0o111 == 0o111
        assert _mode(archive.getinfo(f"{FOLDER}/Info.lua")) & 0o111 == 0

    unpacked = tmp_path / "unpacked"
    _unpack(target, unpacked)

    assert [p.name for p in unpacked.iterdir()] == [FOLDER]
    executable = unpacked / FOLDER / "melampus"
    mode = stat.S_IMODE(executable.stat().st_mode)
    assert mode & 0o111 == 0o111, f"mode {mode:o}: the execute bit did not survive the zip"
    assert stat.S_IMODE((unpacked / FOLDER / "Info.lua").stat().st_mode) & 0o111 == 0
    assert subprocess.run([str(executable)], capture_output=True, text=True).stdout == "built\n"


@pytest.mark.skipif(shutil.which("lua") is None, reason="lua not installed")
def test_info_lua_parses_and_names_only_files_in_the_zip(package_script, tmp_path: Path):
    """Done-when 2, the part that runs without Lightroom: from the real plugin
    folder's zip, Info.lua parses under a plain interpreter and every file it
    names (init, metadata provider, tagset, each menu item) and every module
    those files require is in the zip listing. What Plug-in Manager does with
    them is the manual check after the first tagged release."""
    target = tmp_path / "Melampus-macOS.zip"
    listing = package_script.package(_fake_executable(tmp_path), target)
    unpacked = tmp_path / "unpacked"
    _unpack(target, unpacked)

    run_lua_suite(TESTS / "test_info.lua", env=os.environ | {
        "MELAMPUS_PLUGIN_DIR": str(unpacked / FOLDER),
        "MELAMPUS_ZIP_LISTING": "\n".join(listing),
    })


def test_packaged_executable_runs_from_the_unpacked_plugin_folder(
    package_script, built_executable: Path, photos: Path, tmp_path: Path
):
    """Done-when 1 at the real boundary: the zip of the real dist/melampus,
    unpacked, holds an executable that runs from inside Melampus.lrplugin/
    with --backend scripted --plugin-out on the committed fixture, with no
    python on the path, and writes the enriched results. It writes nothing
    into the plugin folder: its cache lands under the per-user data directory,
    so the unpacked folder still lists exactly what the zip did."""
    target = tmp_path / package_script.zip_path().name
    listing = package_script.package(built_executable, target)
    print("zip listing:", *listing, sep="\n  ")
    unpacked = tmp_path / "unpacked"
    _unpack(target, unpacked)
    plugin = unpacked / FOLDER
    executable = plugin / built_executable.name
    assert executable.is_file()

    env = no_python_environment(tmp_path)
    data_dir = per_user_data_dir(Path(env["HOME"]))
    out = tmp_path / "plugin_results.json"
    proc = subprocess.run(
        [str(executable), str(photos), "--backend", "scripted", "--plugin-out", str(out)],
        env=env, cwd=plugin, capture_output=True, text=True, timeout=600,
    )

    assert proc.returncode == 0, proc.stderr[-3000:]
    assert_scripted_results(out)
    assert (data_dir / "cache").is_dir(), "the cache did not land under the per-user data directory"
    assert sorted(f"{FOLDER}/{p.relative_to(plugin).as_posix()}" for p in plugin.rglob("*") if p.is_file()) == listing, (
        "the executable wrote into the plugin folder")
