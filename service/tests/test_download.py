"""The model download command (card #407).

Done-when 1: given the download command, when it runs, then it fetches the
configured MLX model, reports bytes done and total on stdout as it goes, and
resumes if interrupted.
Done-when 2: given a cancel signal, when it arrives, then the download stops
cleanly and partial files are kept for resume.
Done-when 3: given the tests, when they run, then a local fake of the model
host stands in and nothing is fetched from the internet.

The progress protocol is the plugin's (card #408) contract, so it is tested in
both directions: the lines the command prints, and the parse the plugin will do.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import tomllib
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from conftest import (
    AGENT_HARNESS,
    CUT_IN_THE_SECOND_CHUNK,
    FAKE_COMMIT,
    FAKE_FILES,
    FAKE_FOLDER,
    FAKE_REPO,
    FAKE_TOTAL,
    VENV_CLI,
    FakeHub,
    Silent,
    assert_download_completed,
    closed_port,
    fake_bytes,
    loopback_server,
    snapshot_files,
)
from filelock import Timeout
from huggingface_hub import constants
from huggingface_hub.constants import DOWNLOAD_CHUNK_SIZE
from huggingface_hub.file_download import repo_folder_name

from melampus import download
from melampus.cli import main
from melampus.config import ModelConfig
from melampus.download import (
    CANCEL_MARKER,
    EXIT_CANCELLED,
    MODEL_FILE_PATTERNS,
    DownloadCancelled,
    DownloadError,
    Status,
    Update,
    _hub_client,
    _token_may_go,
    cancel_marker_path,
    cancel_on_signals,
    download_model,
    model_status,
    remove_model,
)


def _fetch(hub: FakeHub, cache: Path, repo: str = FAKE_REPO) -> tuple[Path, list[Update]]:
    updates: list[Update] = []
    path = download_model(repo, endpoint=hub.endpoint, cache_dir=cache, on_update=updates.append)
    return path, updates


def _incomplete(cache: Path) -> list[Path]:
    return sorted((cache / FAKE_FOLDER / "blobs").glob("*.incomplete"))


def _cli_sees_the_cache(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    """The cache under tmp_path as the entry point's own (`HF_HUB_CACHE`,
    what `_cache_paths` reads when no `cache_dir` is passed), never the
    real one."""
    monkeypatch.setattr(download.constants, "HF_HUB_CACHE", str(cache))


def _refused_through_the_cli(capsys, flag: str, naming: str, repo: str = FAKE_REPO) -> str:
    """The one contract every refusal keeps through the entry point (cli.py,
    docs/config.md): exit 3, the reason on stderr naming `naming`, nothing
    on stdout, never a traceback. Returns stderr for what else a test asks
    of the reason."""
    assert main([flag, "--no-local-config", "--model", repo]) == 3, flag
    out, err = capsys.readouterr()
    assert out == "" and naming in err and "Traceback" not in err, (flag, err)
    return err


# The sample lines both parsers are tested against, so the Lua one in the
# plugin (Rules.parseDownloadLine, plugin/tests/test_rules.lua) cannot drift
# from this one: `<input>\t<state>[\t<field>...]`, `rejected` for a non-update.
SAMPLE_LINES = Path(__file__).with_name("fixtures") / "download-lines.txt"


def sample_lines() -> list[tuple[str, list[str]]]:
    rows = []
    for row in SAMPLE_LINES.read_text(encoding="utf-8").splitlines():
        if row.startswith("#"):
            continue
        line, *expected = row.split("\t")
        rows.append((line, expected))
    assert len(rows) >= 10 and any(e == ["rejected"] for _, e in rows) and any(e[0] == "done" for _, e in rows)
    return rows


@pytest.mark.parametrize(("line", "expected"), sample_lines(), ids=lambda value: repr(value)[:40])
def test_progress_protocol_prints_and_parses_the_same_line(line: str, expected: list[str]):
    """One line per update, machine-readable and stable: `progress <done> <total>`
    while bytes arrive, `done <path>` once the model is complete, `cancelled`
    when a signal stopped it. A path may hold spaces, so it is the rest of the
    line. The plugin must be able to tell an update from any other line."""
    if expected == ["rejected"]:
        with pytest.raises(ValueError):
            Update.parse(line)
        return
    state, *fields = expected
    update = {
        "progress": lambda: Update.progress(int(fields[0]), int(fields[1])),
        "done": lambda: Update.done(fields[0]),
        "cancelled": lambda: Update.cancelled(),
    }[state]()
    assert update.line() == line
    assert Update.parse(line) == update
    assert Update.parse(line + "\n") == update, "a line read from a pipe keeps its newline"
    assert Update.parse(line + "\r\n") == update


def test_download_fetches_every_file_from_the_hub_and_reports_bytes_done_of_total(
    fake_hub: FakeHub, tmp_path: Path
):
    """Done-when 1: the model lands in the Hugging Face cache, laid out the way
    huggingface_hub lays it out (mlx-vlm reads it from there), and every update
    says how many bytes of the whole model are done: the total is known from
    the first update, bytes done never go backwards and end at the total."""
    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert snapshot_files(path) == FAKE_FILES
    assert path == tmp_path / "hub" / FAKE_FOLDER / "snapshots" / FAKE_COMMIT
    assert (path.parent.parent / "refs" / "main").read_text() == FAKE_COMMIT, "no ref for mlx-vlm to load offline"
    assert [u.state for u in updates] == ["progress"] * len(updates)
    assert updates[0] == Update.progress(0, FAKE_TOTAL), "the total is known before any byte arrives"
    counts = [u.bytes_done for u in updates]
    assert counts == sorted(counts) and counts[-1] == FAKE_TOTAL
    assert all(u.bytes_total == FAKE_TOTAL for u in updates)
    assert not _incomplete(tmp_path / "hub")


def test_download_goes_only_to_the_fake_host(fake_hub: FakeHub, tmp_path: Path):
    """Done-when 3: every request of the whole conversation lands on the fake
    on 127.0.0.1: the tree listing, then metadata and bytes per file. Nothing
    is left for the internet to answer."""
    _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.requests, "the fake saw nothing"
    for r in fake_hub.requests:
        assert r.path.startswith((f"/api/models/{FAKE_REPO}", f"/{FAKE_REPO}/resolve/")), (r.method, r.path)
    for name in FAKE_FILES:
        assert fake_hub.gets(name) == [None], f"{name} was not fetched whole, exactly once"


def test_download_lays_out_the_commit_it_planned_when_the_branch_moves_during_the_run(
    fake_hub: FakeHub, tmp_path: Path
):
    """Codex round 1, both reviewers (download.py:284). The plan resolves
    `main` to a commit and fetches and verifies that commit's files; then
    `snapshot_download` resolved `main` a second time, so a branch that moved
    in between had the hub library fetch another revision's files outside
    this module's progress, checksum and resume. Given a branch that moves
    once the plan is made (on the first update, before any byte), the
    snapshot laid out is the planned commit's, `refs/main` names it, no
    other snapshot appears, and the hub is asked what `main` is once."""
    cache = tmp_path / "hub"
    moved = "fedcba9876543210fedcba9876543210fedcba98"
    updates: list[Update] = []

    def move_the_branch(update: Update) -> None:
        updates.append(update)
        fake_hub.commit = moved

    path = download_model(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=cache, on_update=move_the_branch)

    assert path == cache / FAKE_FOLDER / "snapshots" / FAKE_COMMIT, "the snapshot is not the planned commit's"
    assert snapshot_files(path) == FAKE_FILES
    assert (cache / FAKE_FOLDER / "refs" / "main").read_text() == FAKE_COMMIT
    assert sorted(p.name for p in (cache / FAKE_FOLDER / "snapshots").iterdir()) == [FAKE_COMMIT]
    resolved = [r for r in fake_hub.requests if r.path.partition("?")[0] == f"/api/models/{FAKE_REPO}"]
    assert len(resolved) == 1, f"`main` was resolved {len(resolved)} times: {resolved}"
    assert updates[-1] == Update.progress(FAKE_TOTAL, FAKE_TOTAL)


def test_download_of_a_complete_model_fetches_nothing_and_says_it_is_complete(
    fake_hub: FakeHub, tmp_path: Path
):
    """A second run is the plugin's way to check: no bytes move, the one
    update says all of the total is done, and the same path comes back."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    fake_hub.requests.clear()

    again, updates = _fetch(fake_hub, tmp_path / "hub")

    assert again == path
    assert updates == [Update.progress(FAKE_TOTAL, FAKE_TOTAL)]
    assert not fake_hub.gets(), fake_hub.requests


def test_download_keeps_the_partial_file_when_the_connection_drops_and_resumes_it_next_run(
    fake_hub: FakeHub, tmp_path: Path
):
    """Done-when 1, resume. The host drops the connection after one chunk of
    the large file and then answers 503: the run fails, saying so, with the
    chunk kept in the cache's `.incomplete` blob. The next run asks the host
    for the rest (a Range request from the byte it has), and the file it
    finishes is byte-identical to the host's."""
    fake_hub.cut_after = CUT_IN_THE_SECOND_CHUNK

    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    assert "503" in str(failure.value) and "--download-model" in str(failure.value)
    (partial,) = _incomplete(tmp_path / "hub")
    assert partial.stat().st_size == DOWNLOAD_CHUNK_SIZE
    assert partial.read_bytes() == FAKE_FILES["model.safetensors"][:DOWNLOAD_CHUNK_SIZE]

    fake_hub.outage = False
    fake_hub.requests.clear()
    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.gets("model.safetensors") == [f"bytes={DOWNLOAD_CHUNK_SIZE}-"], "the rest was not asked for by Range"
    assert snapshot_files(path) == FAKE_FILES
    assert not _incomplete(tmp_path / "hub")
    assert updates[0] == Update.progress(DOWNLOAD_CHUNK_SIZE + len(FAKE_FILES["config.json"]), FAKE_TOTAL), (
        "the first update did not count what was already on disk")
    assert updates[-1] == Update.progress(FAKE_TOTAL, FAKE_TOTAL)


def test_download_takes_back_the_partial_when_the_host_ignores_the_range_and_never_passes_the_total(
    fake_hub: FakeHub, tmp_path: Path
):
    """Codex round 1 (download.py:239). A host that answers the Range request
    with 200 and the whole file (a CDN that ignores Range) makes the hub
    library's `http_get` truncate the partial and take the file from byte
    zero; its own counter rollback reaches only a bar it reuses across its
    retries, never the fresh one of this first call, so the partial's bytes,
    already counted from disk, were counted a second time and `progress`
    passed the total. Given a partial left by a cut run and a host that then
    ignores Range, the re-run asks for the rest, takes the partial's bytes
    back (one update steps back to what the other files hold), climbs to
    exactly the total, never past it, and the file is byte-identical."""
    fake_hub.cut_after = CUT_IN_THE_SECOND_CHUNK
    with pytest.raises(DownloadError):
        _fetch(fake_hub, tmp_path / "hub")
    (partial,) = _incomplete(tmp_path / "hub")
    kept = partial.stat().st_size
    assert kept == DOWNLOAD_CHUNK_SIZE

    fake_hub.outage, fake_hub.ignore_range = False, True
    fake_hub.requests.clear()
    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.gets("model.safetensors") == [f"bytes={kept}-"], "the rest was not asked for by Range"
    assert snapshot_files(path) == FAKE_FILES
    assert not _incomplete(tmp_path / "hub")
    config = len(FAKE_FILES["config.json"])
    assert updates[0] == Update.progress(kept + config, FAKE_TOTAL), "the first update did not count what was on disk"
    assert updates[1] == Update.progress(config, FAKE_TOTAL), "the partial's bytes were not taken back"
    assert max(u.bytes_done for u in updates) == FAKE_TOTAL == updates[-1].bytes_done, "progress passed the total"


def test_download_names_the_file_and_both_sizes_when_the_resumed_answer_is_the_wrong_size_and_keeps_the_partial(
    fake_hub: FakeHub, tmp_path: Path
):
    """Security (Codex round 4, download.py:508). The host drops the
    connection after one chunk and answers the hub library's own retry, a
    Range request, with a complete body that ends short of the file, so its
    size check fails. Its message names the file as the library formatted
    it, which on its retry is the tail of the URL; that wording is not
    repeated. The run fails naming the file, the bytes that arrived, the size
    the hub said and the re-run, with the partial kept, and the next run,
    from a host that serves the rest, finishes the file byte-identical."""
    big = len(FAKE_FILES["model.safetensors"])
    fake_hub.cut_after = CUT_IN_THE_SECOND_CHUNK
    fake_hub.short_resume = 100

    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    message = str(failure.value)
    assert fake_hub.gets("model.safetensors") == [None, f"bytes={DOWNLOAD_CHUNK_SIZE}-"], "the drop was not retried by Range"
    assert "model.safetensors" in message and "--download-model" in message, message
    assert f"{big - 100}" in message and f"{big}" in message, message
    assert "Consistency check" not in message and "(…)" not in message, "the library's own wording is repeated"
    (partial,) = _incomplete(tmp_path / "hub")
    assert partial.stat().st_size == big - 100, "the partial file was not kept"

    fake_hub.short_resume = 0
    fake_hub.requests.clear()
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.gets("model.safetensors") == [f"bytes={big - 100}-"], "the rest was not asked for by Range"
    assert snapshot_files(path) == FAKE_FILES
    assert not _incomplete(tmp_path / "hub")


# Two weight files with the same bytes: the hub names one etag for both, so
# the cache holds one blob that two pointers share.
SHARED_FILES = {
    "config.json": FAKE_FILES["config.json"],
    "model-00001.safetensors": FAKE_FILES["model.safetensors"],
    "model-00002.safetensors": FAKE_FILES["model.safetensors"],
}
SHARED_TOTAL = len(SHARED_FILES["config.json"]) + len(SHARED_FILES["model-00001.safetensors"])


@pytest.mark.parametrize("fake_hub", [SHARED_FILES], indirect=True, ids=["two files, one blob"])
def test_download_counts_a_blob_two_files_share_once_and_ends_at_the_total(fake_hub: FakeHub, tmp_path: Path):
    """Codex round 3 (download.py:386). Two files with identical bytes share
    one etag, so one blob, fetched once; the total counted every file, so a
    fresh run ended `done` with bytes_done below bytes_total. The total is
    the bytes that move, each blob once: the run climbs to exactly it, the
    shared blob's bytes are fetched once, and both files are in the snapshot."""
    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert snapshot_files(path) == SHARED_FILES
    assert updates[0] == Update.progress(0, SHARED_TOTAL), "the total counted the shared blob twice"
    assert updates[-1] == Update.progress(SHARED_TOTAL, SHARED_TOTAL), "the run ended below the total"
    assert fake_hub.gets("model-00001.safetensors") == [None] and fake_hub.gets("model-00002.safetensors") == []


@pytest.mark.parametrize("fake_hub", [SHARED_FILES], indirect=True, ids=["two files, one blob"])
def test_download_resumes_a_blob_two_files_share_counting_its_partial_once(fake_hub: FakeHub, tmp_path: Path):
    """The same, cut and resumed: the partial of the shared blob is counted
    once from disk, the rest is asked for by Range once, and the re-run ends
    at the total with both files in the snapshot."""
    fake_hub.cut_after = CUT_IN_THE_SECOND_CHUNK
    with pytest.raises(DownloadError):
        _fetch(fake_hub, tmp_path / "hub")
    (partial,) = _incomplete(tmp_path / "hub")
    kept = partial.stat().st_size

    fake_hub.outage = False
    fake_hub.requests.clear()
    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert snapshot_files(path) == SHARED_FILES
    assert updates[0] == Update.progress(kept + len(SHARED_FILES["config.json"]), SHARED_TOTAL), (
        "the partial was not counted exactly once")
    assert updates[-1] == Update.progress(SHARED_TOTAL, SHARED_TOTAL), "the re-run ended below the total"
    assert fake_hub.gets("model-00001.safetensors") == [f"bytes={kept}-"] and fake_hub.gets("model-00002.safetensors") == []
    assert not _incomplete(tmp_path / "hub")


@pytest.mark.parametrize("name", ["model.safetensors", "config.json"])
def test_download_rejects_bytes_that_do_not_match_the_hub_checksum_and_keeps_no_partial_of_them(
    fake_hub: FakeHub, tmp_path: Path, name: str
):
    """The hub names every file's checksum in its etag: the sha256 of an LFS
    file (the weights), git's blob sha1 of a regular one. A finished file whose
    bytes do not match (corrupted or substituted in transit, or stitched onto
    a bad partial from an earlier run) never becomes a blob: the run fails
    naming the file and the re-run, the partial is discarded rather than
    resumed forever, and the next run fetches the file whole and completes."""
    fake_hub.corrupt = {name}

    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    message = str(failure.value)
    assert message == (
        f"{name} did not match the checksum the hub gave for it; "
        "the partial file is discarded; re-run melampus-id --download-model to fetch it whole"
    ), "the exit-3 message must name the file, the discarded partial and the whole re-fetch, not a resume"
    assert not _incomplete(tmp_path / "hub"), "the bad partial was kept"
    blobs = tmp_path / "hub" / FAKE_FOLDER / "blobs"
    assert not (blobs / fake_hub.etags[name]).exists(), "the bad bytes became the blob"

    fake_hub.corrupt = set()
    fake_hub.requests.clear()
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.gets(name) == [None], "the next run resumed the discarded partial instead of fetching whole"
    assert snapshot_files(path) == FAKE_FILES


def _escape(tmp_path: Path) -> list[Path]:
    """Anything named `escape` the run left, inside or outside the cache."""
    return sorted(tmp_path.rglob("*escape*"))


@pytest.mark.parametrize("hostile", [
    pytest.param(lambda tmp: "../../../escape", id="relative traversal"),
    pytest.param(lambda tmp: str(tmp / "escape"), id="absolute path"),
    pytest.param(lambda tmp: "abc123", id="not a full digest"),
])
def test_download_rejects_an_etag_that_is_not_a_checksum_before_it_becomes_a_path(
    fake_hub: FakeHub, tmp_path: Path, hostile
):
    """Security (Codex round 1, download.py:203). The etag the hub names for
    a file became the blob's path and the lock's name unchecked, so a hub
    (or whatever answers for it) naming `../../../escape` wrote outside the
    cache; and an etag that is neither a sha256 nor a git blob sha1 passed
    `_verify` unchecked. Given a hub whose etag for a file is a path, or is
    not a full 64- or 40-hex digest, the run fails naming the file and the
    etag before any byte of it is asked for, and nothing is created outside
    the cache, nor under that name inside it."""
    etag = hostile(tmp_path)
    fake_hub.etags["config.json"] = etag

    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    assert "config.json" in str(failure.value) and etag in str(failure.value)
    assert sorted(tmp_path.iterdir()) == [tmp_path / "hub"], "the run wrote outside the cache"
    assert not _escape(tmp_path) and not list(tmp_path.rglob("abc123*")), "the etag became a path"
    assert not fake_hub.gets("config.json"), "bytes were fetched for a file whose etag is not a checksum"


def test_download_rejects_a_commit_that_is_not_a_hash_before_it_becomes_a_path(fake_hub: FakeHub, tmp_path: Path):
    """Security (Codex round 1, download.py:203), the other hub value that
    becomes a path: the commit `main` resolves to names the snapshot folder.
    Given a hub whose `main` points at `../../../escape`, the run fails
    naming it and nothing is created outside the cache."""
    fake_hub.commit = "../../../escape"

    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    assert "../../../escape" in str(failure.value)
    assert sorted(tmp_path.iterdir()) == [tmp_path / "hub"], "the run wrote outside the cache"
    assert not _escape(tmp_path), "the commit became a path"
    assert not [r for r in fake_hub.requests if "/resolve/" in r.path], "files were asked for at a commit that is not one"


def test_download_lays_out_the_snapshot_from_the_verified_blobs_and_asks_the_hub_nothing_more(
    fake_hub: FakeHub, tmp_path: Path
):
    """Security (Codex round 2, download.py:349). Once every blob was fetched
    and verified, `snapshot_download` laid out the pointers by asking the hub
    for each file's metadata a second time and trusting that answer: an etag
    it named became a blob path unchecked (`../../../escape` wrote outside the
    cache) and a blob it named that was not in the cache was fetched, outside
    `_verify` and the progress protocol. Given a hub whose answer to every
    HEAD after a file's first is a traversal etag, the snapshot is laid out
    from the blobs this module verified: the pointers hold the files byte for
    byte, `refs/main` names the commit, each file's metadata was asked for
    once and its bytes once, and nothing is written outside the cache."""
    fake_hub.later_etag = "../../../escape"

    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert path == tmp_path / "hub" / FAKE_FOLDER / "snapshots" / FAKE_COMMIT
    assert snapshot_files(path) == FAKE_FILES
    assert (path.parent.parent / "refs" / "main").read_text() == FAKE_COMMIT
    assert updates[-1] == Update.progress(FAKE_TOTAL, FAKE_TOTAL)
    assert sorted(tmp_path.iterdir()) == [tmp_path / "hub"], "the run wrote outside the cache"
    assert not _escape(tmp_path), "the second etag became a path"
    for name in FAKE_FILES:
        assert fake_hub.heads(name) == 1, f"{name}'s metadata was asked for again after the plan"
        assert fake_hub.gets(name) == [None], f"{name}'s bytes were fetched outside the verified path"
    assert not _incomplete(tmp_path / "hub")


def test_download_repairs_a_snapshot_file_that_is_a_short_copy_of_its_blob(fake_hub: FakeHub, tmp_path: Path):
    """Codex round 3 (download.py:359). Where symlinks are unavailable
    (Windows without developer mode) the hub library's pointer helper copies
    the blob into the snapshot instead, and a run cut during that copy (a
    cancel, a full disk) left a short file under the file's name; the next
    run skipped it because it existed and said `done` of a corrupt model.
    Given a snapshot file that is a regular file shorter than its blob, the
    re-run replaces it with the blob's bytes, moves no bytes, and says done."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    short = path / "model.safetensors"
    short.unlink()
    short.write_bytes(FAKE_FILES["model.safetensors"][:4096])
    fake_hub.requests.clear()

    again, updates = _fetch(fake_hub, tmp_path / "hub")

    assert again == path
    assert snapshot_files(path) == FAKE_FILES, "the short copy was kept"
    assert updates == [Update.progress(FAKE_TOTAL, FAKE_TOTAL)]
    assert not fake_hub.gets(), "bytes were fetched again"


def test_download_publishes_a_copied_snapshot_file_whole_or_not_at_all(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The interrupted-copy regression: with symlinks off (the hub library's
    own switch, HF_HUB_DISABLE_SYMLINKS) the pointer is a copy, and the disk
    fills halfway through the large file's. The run fails naming the re-run,
    nothing under the file's name is left in the snapshot, and the re-run
    publishes the copy whole: the snapshot holds exactly the model's files,
    byte for byte, as regular files, with no bytes fetched again."""
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS", True)
    real_copyfile, disk_full = shutil.copyfile, [True]

    def copy_until_the_disk_fills(src, dst, *args, **kwargs):
        if disk_full and Path(src).stat().st_size == len(FAKE_FILES["model.safetensors"]):
            disk_full.clear()
            Path(dst).write_bytes(Path(src).read_bytes()[:4096])
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copy_until_the_disk_fills)
    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    assert "No space left" in str(failure.value) and "--download-model" in str(failure.value)
    snapshot = tmp_path / "hub" / FAKE_FOLDER / "snapshots" / FAKE_COMMIT
    assert not (snapshot / "model.safetensors").exists(), "a short copy was published under the file's name"
    assert not disk_full, "the copy was never cut"

    fake_hub.requests.clear()
    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert path == snapshot
    assert snapshot_files(path) == FAKE_FILES, "the snapshot holds something other than the model's files"
    assert all(not (path / name).is_symlink() for name in FAKE_FILES), "symlinks were off"
    assert updates == [Update.progress(FAKE_TOTAL, FAKE_TOTAL)]
    assert not fake_hub.gets(), "bytes were fetched again"


# A repo holding a file named like another file's staging name: any name is
# a valid repo path. Served in each order, since the listing's order decided
# which of the two survived.
TWIN_FILES = {**FAKE_FILES, "config.json.incomplete": b'{"model_type": "twin"}\n'}


@pytest.mark.parametrize(
    "fake_hub", [TWIN_FILES, dict(reversed(TWIN_FILES.items()))], indirect=True,
    ids=["config.json listed first", "config.json.incomplete listed first"],
)
def test_download_lays_out_a_file_named_like_another_files_staging_name(fake_hub: FakeHub, tmp_path: Path):
    """Codex round 7 (download.py:452). Each pointer was staged beside its
    own as `<filename>.incomplete`, a name the repo may legitimately hold:
    listed before `config.json`, the repo's `config.json.incomplete` was laid
    out and then destroyed by `config.json`'s staging, the run saying `done`
    of a snapshot missing a file. Given a repo with both files, in either
    listing order, the snapshot holds exactly the files served, each with its
    own bytes, and nothing else."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    assert snapshot_files(path) == fake_hub.files, "a file of the repo is missing from the snapshot, or a stale one is in it"
    assert not _incomplete(tmp_path / "hub")


@pytest.mark.parametrize("fake_hub", [{**FAKE_FILES, "../../../escape": b"not a model file\n"}],
                         indirect=True, ids=["a listing naming a path"])
def test_download_rejects_a_filename_that_is_a_path_before_any_byte_of_it_is_asked_for(
    fake_hub: FakeHub, tmp_path: Path
):
    """The other hub value that becomes a path in the snapshot: a file's name
    from the tree listing, joined under `snapshots/<commit>/`. The hub
    library's own download refused one that traverses; now that the snapshot
    is laid out here, the plan refuses it, naming the file, before any byte is
    asked for, and nothing is written outside the cache."""
    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    assert "../../../escape" in str(failure.value)
    assert sorted(tmp_path.iterdir()) == [tmp_path / "hub"], "the run wrote outside the cache"
    assert not _escape(tmp_path), "the filename became a path"
    assert not fake_hub.gets(), (
        "bytes were fetched for a listing that names a path")


def test_download_sends_the_user_token_to_the_hub_and_never_to_the_host_serving_the_bytes(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The hub answers the metadata HEAD of every LFS file with a redirect to
    its CDN, a signed URL on another host. The user's token (`hf auth login`,
    or HF_TOKEN) belongs to the hub: it goes on the hub's requests and on no
    request to the other host, exactly as huggingface_hub's own download
    strips it when the location's host is not the endpoint's."""
    monkeypatch.setenv("HF_TOKEN", "synthetic-token")
    with FakeHub().serve() as cdn:
        fake_hub.bytes_host = cdn.endpoint
        path, _ = _fetch(fake_hub, tmp_path / "hub")

    assert snapshot_files(path) == FAKE_FILES
    assert all(r.authorization == "Bearer synthetic-token" for r in fake_hub.requests), fake_hub.requests
    assert [name for name in FAKE_FILES if cdn.gets(name)] == list(FAKE_FILES), "the bytes did not come from the CDN"
    assert all(r.authorization is None for r in cdn.requests), "the token left the hub"


def test_download_sends_the_user_token_to_the_hub_and_never_to_a_host_the_listing_pages_on_to(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Security (Codex round 2, download.py:244). The tree listing is paged:
    huggingface_hub follows the `Link: rel="next"` URL the hub names, with
    the same headers, wherever it points. The token's origin rule covered a
    file's bytes only, so a hub naming another origin as the next page (another
    host, or an `http://` downgrade of its own) had the user's token sent
    there. Given a listing whose next page is on another loopback host, the
    listing is followed there (it lists the rest, here nothing more) and
    every request to it goes without the token, while every request to the
    hub carries it; the model completes."""
    monkeypatch.setenv("HF_TOKEN", "synthetic-token")
    with FakeHub(files={}).serve() as elsewhere:
        fake_hub.next_page = f"{elsewhere.endpoint}/api/models/{FAKE_REPO}/tree/{FAKE_COMMIT}?cursor=2"
        path, _ = _fetch(fake_hub, tmp_path / "hub")

    assert snapshot_files(path) == FAKE_FILES
    assert [r.path for r in elsewhere.requests] == [f"/api/models/{FAKE_REPO}/tree/{FAKE_COMMIT}?cursor=2"], (
        "the listing's next page was not followed")
    assert all(r.authorization is None for r in elsewhere.requests), "the token left the hub"
    assert all(r.authorization == "Bearer synthetic-token" for r in fake_hub.requests), fake_hub.requests


@pytest.mark.parametrize(("url", "endpoint", "trusted"), [
    ("http://127.0.0.1:8/fake-org/fake-model/resolve/abc/config.json", "http://127.0.0.1:8", True),
    ("http://localhost:8/x", "http://localhost:8", True),
    ("http://[::1]:8/x", "http://[::1]:8", True),
    ("https://huggingface.co/x/resolve/abc/config.json", "https://huggingface.co", True),
    ("http://huggingface.co/x/resolve/abc/config.json", "https://huggingface.co", False),
    ("https://cdn-lfs.hf.co/x", "https://huggingface.co", False),
    ("https://huggingface.co:8443/x", "https://huggingface.co", False),
    ("http://hub.example:8080/x/resolve/abc/config.json", "http://hub.example:8080", False),
    ("http://10.0.0.5/x", "http://10.0.0.5", False),
], ids=["fake hub", "localhost", "ipv6 loopback", "the hub itself", "https downgraded to http", "the CDN",
        "another port", "a remote http hub", "a remote http hub by address"])
def test_the_token_goes_only_to_the_endpoints_own_origin_over_https_or_loopback(url: str, endpoint: str, trusted: bool):
    """Security (Codex round 1, download.py:279). Whether the token goes with
    a file's bytes was decided on the host alone, so an `https://` hub
    naming an `http://` download URL on the same host would have had the
    token sent in cleartext. The decision is the whole origin, scheme
    included: a downgrade is another host, and gets no token. Security
    (Codex round 3, download.py:248): the hub's own origin was trusted
    whatever its scheme, so a configured `http://` hub on another machine
    received the token in cleartext. It goes only where cleartext cannot
    leave the machine or there is none: an `https://` hub, or a loopback
    host (127.0.0.1, ::1, localhost), the fake hub of these tests. The fake
    hub cannot serve two schemes on one host and port, so this is the
    decision on its own; the boundary is proven by the CDN test above."""
    assert _token_may_go(url, endpoint) is trusted


@pytest.mark.parametrize(("endpoint", "carried"), [
    ("http://127.0.0.1:8", True), ("https://hub.example", True), ("http://hub.example:8080", False),
], ids=["loopback http", "remote https", "remote http"])
def test_the_hub_client_strips_the_token_from_a_request_to_a_remote_http_hub(endpoint: str, carried: bool):
    """The decision wired into the client every hub request goes through:
    the request the library makes to the endpoint itself, with the user's
    token, keeps it on a loopback or https hub and loses it on a remote
    http one."""
    client = _hub_client(endpoint)
    request = httpx.Request("GET", f"{endpoint}/api/models/{FAKE_REPO}", headers={"authorization": "Bearer synthetic"})
    for hook in client.event_hooks["request"]:
        hook(request)
    assert ("authorization" in request.headers) is carried, dict(request.headers)


def test_download_of_a_repo_the_hub_does_not_have_names_the_setting_to_fix(
    fake_hub: FakeHub, tmp_path: Path
):
    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub", repo="fake-org/no-such-model")
    message = str(failure.value)
    assert "fake-org/no-such-model" in message
    assert "[model] repo" in message and "--model" in message


@pytest.mark.parametrize("repo", ["https://huggingface.co/fake-org/fake-model", "fake-org/fake-model/extra"],
                         ids=["a pasted hub URL", "a path under the repo"])
def test_download_of_a_repo_id_that_is_not_one_names_the_setting_to_fix_before_asking_the_hub(
    fake_hub: FakeHub, tmp_path: Path, repo: str
):
    """Codex round 3 (download.py:382). A repo id that is not one (a pasted
    hub URL, a path under the repo) failed the hub library's own validation
    before the error handler was reached, so the command printed a traceback
    and exited 1, not 3. Given such an id, the run fails as the missing-repo
    case does: the message names the id and the setting to fix, and the hub
    is asked nothing."""
    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub", repo=repo)

    message = str(failure.value)
    assert repo in message and "[model] repo" in message and "--model" in message
    assert not fake_hub.requests, "the hub was asked about an id that is not a repo's"


def test_download_of_a_gated_repo_names_the_access_to_request_not_a_missing_repo(
    fake_hub: FakeHub, tmp_path: Path
):
    """A gated repo is on the hub; what is missing is the user's access to
    it: accepting its terms on the hub, signed in with a token. huggingface_hub
    raises GatedRepoError, a RepositoryNotFoundError, so the message must not
    send the user to the repo setting, which is right."""
    fake_hub.gated = True
    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")
    message = str(failure.value)
    assert FAKE_REPO in message and "gated" in message and "access" in message
    assert "hf auth login" in message and "HF_TOKEN" in message
    assert "[model] repo" not in message and "--model" not in message


def test_download_with_no_host_answering_names_the_network(tmp_path: Path):
    port = closed_port()
    with pytest.raises(DownloadError) as failure:
        download_model(FAKE_REPO, endpoint=f"http://127.0.0.1:{port}", cache_dir=tmp_path / "hub",
                       on_update=lambda update: None)
    message = str(failure.value)
    assert f"http://127.0.0.1:{port}" in message and "network" in message


def test_download_gives_up_when_the_hub_accepts_and_never_answers(monkeypatch, tmp_path: Path):
    """Security (Codex round 1, download.py:194). The hub library's repo info
    and tree listing name no timeout and its httpx client is built with
    none, so a hub that accepts the connection and never answers held the
    command forever. Given a listener that accepts and never speaks, the run
    fails naming the network within the library's own metadata timeout,
    HF_HUB_ETAG_TIMEOUT (shortened here), rather than never. The run is on a
    thread so the failure is a bound not met, never a hung suite."""
    monkeypatch.setattr(constants, "HF_HUB_ETAG_TIMEOUT", 0.3)
    outcome: list[BaseException] = []

    with loopback_server(Silent, ThreadingHTTPServer) as server:
        endpoint = f"http://127.0.0.1:{server.server_port}"

        def run() -> None:
            try:
                download_model(FAKE_REPO, endpoint=endpoint, cache_dir=tmp_path / "hub", on_update=lambda u: None)
            except BaseException as exc:  # noqa: BLE001 - whatever it raised is the evidence
                outcome.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "the download hung on a hub that accepts and never answers"

    (failure,) = outcome
    assert isinstance(failure, DownloadError), failure
    assert endpoint in str(failure) and "network" in str(failure)


def declared_dependency(package: str) -> list[str]:
    """The entries of service/pyproject.toml's core dependencies (the ones
    the lockfile installs on every platform) that name `package`."""
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    return [d for d in pyproject["project"]["dependencies"] if d.startswith(package)]


def test_the_hub_library_is_a_dependency_on_every_platform_pinned_to_the_reviewed_version():
    """download.py imports huggingface_hub directly, and on Windows nothing
    else brings it (mlx-vlm is Apple Silicon only), so the executable built
    there carries the command only if service/pyproject.toml names it, for
    every platform, in the core dependencies the lockfile installs. Security
    (Codex round 1, pyproject.toml:28): a new dependency is pinned exactly,
    as pyinstaller is, so an install without the lockfile cannot pull a
    version nobody reviewed; what download.py leans on (`http_get`'s resume,
    `resolve_revision`, the client factory) was read at 1.26.0."""
    (declared,) = declared_dependency("huggingface_hub")
    assert declared == "huggingface_hub==1.26.0", f"not the exact reviewed version: {declared}"


@pytest.mark.parametrize("package", ["huggingface_hub", "filelock"])
def test_the_libraries_the_download_imports_are_dependencies_on_every_platform(package: str):
    """download.py imports huggingface_hub directly, and on Windows nothing
    else brings it (mlx-vlm is Apple Silicon only), so the executable built
    there carries the command only if service/pyproject.toml names it, for
    every platform, in the core dependencies the lockfile installs. The same
    for filelock (card #408: `--remove-model` catches its Timeout): it is in
    the environment today as the hub library's own dependency, and a direct
    import of a package only a dependency brings breaks the day that
    dependency drops it, so it is declared, not borrowed."""
    declared = declared_dependency(package)
    assert len(declared) == 1, f"{package} is not declared in service/pyproject.toml: {declared}"
    assert ";" not in declared[0], f"platform-restricted: {declared[0]}"


def test_filelock_is_pinned_to_the_reviewed_version():
    """Security (round 2, pyproject.toml:32): a new dependency is pinned
    exactly, as pyinstaller is and as the hub library is on the base branch
    (its own security round), so an install without the lockfile cannot pull
    a version nobody reviewed. `--remove-model` leans on filelock's Timeout
    being what WeakFileLock raises; that was read at 3.32.2, the version the
    lockfile resolves."""
    (declared,) = declared_dependency("filelock")
    assert declared == "filelock==3.32.2", f"not the exact reviewed version: {declared}"


# --- the command: exit codes and signals (unit) -----------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill cannot send SIGINT to this process on Windows")
def test_a_signal_inside_cancel_on_signals_raises_cancelled_and_the_handler_is_restored():
    """Done-when 2's mechanism: within the context a SIGINT or SIGTERM becomes
    DownloadCancelled wherever the download is; outside it the process's own
    handlers are back."""
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    for number in (signal.SIGINT, signal.SIGTERM):
        with pytest.raises(DownloadCancelled) as cancelled:
            with cancel_on_signals():
                os.kill(os.getpid(), number)
        assert str(cancelled.value) == signal.Signals(number).name
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_download_model_flag_needs_no_folder_and_passes_the_configured_repo(monkeypatch, capsys, tmp_path):
    """Exit 0 with the `done <path>` line once the model is complete. The repo
    is [model] repo, or --model. `--no-local-config` keeps a developer's own
    `[model] repo` in melampus.local.toml out of the assertion (load_config)."""
    asked = []

    def fake_download(repo, *, on_update, **_):
        asked.append(repo)
        on_update(Update.progress(1, 2))
        return tmp_path / "snapshots" / "abc"

    monkeypatch.setattr(download, "download_model", fake_download)
    assert main(["--download-model", "--no-local-config"]) == 0
    assert main(["--download-model", "--no-local-config", "--model", "fake-org/other"]) == 0
    out = capsys.readouterr().out
    assert asked == [ModelConfig().repo, "fake-org/other"]
    assert out.splitlines() == ["progress 1 2", f"done {tmp_path / 'snapshots' / 'abc'}"] * 2


def test_download_model_flag_exits_4_with_the_cancelled_line_when_a_signal_stops_it(monkeypatch, capsys):
    def fake_download(repo, *, on_update, **_):
        on_update(Update.progress(5, 9))
        raise DownloadCancelled("SIGINT")

    monkeypatch.setattr(download, "download_model", fake_download)
    assert main(["--download-model", "--no-local-config"]) == EXIT_CANCELLED == 4
    assert capsys.readouterr().out.splitlines() == ["progress 5 9", "cancelled"]


def test_download_model_flag_exits_3_with_the_fix_on_stderr_when_it_fails(monkeypatch, capsys):
    def fake_download(repo, *, on_update, **_):
        raise DownloadError("could not reach the hub: check the network")

    monkeypatch.setattr(download, "download_model", fake_download)
    assert main(["--download-model", "--no-local-config"]) == 3
    out, err = capsys.readouterr()
    assert out == ""
    assert "could not reach the hub: check the network" in err


TELEMETRY_SWITCHES = ("HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY", "DO_NOT_TRACK")


def _without(*names: str) -> dict[str, str]:
    """This process's environment less `names`: what a user who has set
    nothing runs the command in."""
    return {k: v for k, v in os.environ.items() if k not in names}


def _setting_at_import(name: str) -> list[str]:
    """What `os.environ[name]` and the hub library's constant of that name
    hold in a fresh interpreter that imported melampus.download with `name`
    (and the telemetry switches) unset."""
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import melampus.download; from huggingface_hub import constants; "
         f"import os; print(os.environ[{name!r}], constants.{name})"],
        env=_without(name, *TELEMETRY_SWITCHES), capture_output=True, text=True, check=True,
    )
    return proc.stdout.split()


def test_importing_the_download_module_disables_the_xet_transfer_as_the_readme_requires():
    """readme.md § Install: HF_HUB_DISABLE_XET=1 is not optional on some
    networks (docs/troubleshooting.md). The command sets it itself, before
    the hub library reads it, so the user need not know."""
    assert _setting_at_import("HF_HUB_DISABLE_XET") == ["1", "True"]


def test_importing_the_download_module_disables_the_hub_library_telemetry():
    """Security (Codex round 2, download.py:27). Rule 11: runtime code sends
    no telemetry. With HF_HUB_DISABLE_TELEMETRY unset the hub library fetches
    the hub's registry of AI coding agents and names the agent it runs under,
    and the torch version, in the User-Agent of every request. The command
    sets the switch itself, before the library reads it, as it does the Xet
    one; a user who set it, or DO_NOT_TRACK, keeps their own value."""
    assert _setting_at_import("HF_HUB_DISABLE_TELEMETRY") == ["1", "True"]


# --- the command, driven the way the plugin will (acceptance) --------------


def _cli(args: list[str], env: dict[str, str], env_base: dict[str, str] = os.environ, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*VENV_CLI, *args], env={**env_base, **env},
                          capture_output=True, text=True, timeout=300, **kwargs)


def test_cli_downloads_the_model_reporting_progress_and_exits_0_on_done(hub_env: dict[str, str]):
    """Done-when 1 through the entry point: every stdout line is a protocol
    line, progress climbs to the total, the last line is `done <path>` and
    that path holds the model, byte for byte."""
    proc = _cli(["--download-model", "--model", FAKE_REPO], hub_env)

    assert proc.returncode == 0, proc.stderr[-3000:]
    assert_download_completed(proc.stdout, hub_env)


def test_cli_asks_the_hub_which_agent_it_runs_under_never_and_names_none_in_its_requests(
    fake_hub: FakeHub, hub_env: dict[str, str]
):
    """Security (Codex round 2, download.py:27), at the boundary. Given a
    user who has set no telemetry switch and runs the command under an AI
    coding agent the hub's registry names, the hub is never asked for that
    registry (`/api/agent-harnesses`) and no request's User-Agent names the
    agent or the packages installed; the download completes."""
    proc = _cli(["--download-model", "--model", FAKE_REPO], {**hub_env, "AGENT_HARNESS": "1"},
                env_base=_without(*TELEMETRY_SWITCHES))

    assert proc.returncode == 0, proc.stderr[-3000:]
    assert_download_completed(proc.stdout, hub_env)
    assert not [r for r in fake_hub.requests if r.path.startswith("/api/agent-harnesses")], "the registry was fetched"
    for r in fake_hub.requests:
        assert r.user_agent and "agent/" not in r.user_agent and "torch/" not in r.user_agent, (r.path, r.user_agent)
        assert AGENT_HARNESS not in r.user_agent


def test_cli_exits_3_naming_the_fix_when_the_repo_is_not_on_the_hub(hub_env: dict[str, str]):
    proc = _cli(["--download-model", "--model", "fake-org/no-such-model"], hub_env)
    assert proc.returncode == 3, proc.stderr[-3000:]
    assert proc.stdout == "", "an error must not be spoken in the protocol"
    assert "fake-org/no-such-model" in proc.stderr and "--model" in proc.stderr


def test_cli_exits_3_naming_the_fix_when_the_repo_id_is_a_pasted_url(hub_env: dict[str, str]):
    """Codex round 3 (download.py:382), at the boundary: exit 3 with the
    fix on stderr and nothing in the protocol, not a traceback and exit 1."""
    proc = _cli(["--download-model", "--model", "https://huggingface.co/fake-org/fake-model"], hub_env)
    assert proc.returncode == 3, proc.stderr[-3000:]
    assert proc.stdout == "", "an error must not be spoken in the protocol"
    assert "Traceback" not in proc.stderr, proc.stderr[-3000:]
    assert "https://huggingface.co/fake-org/fake-model" in proc.stderr
    assert "[model] repo" in proc.stderr and "--model" in proc.stderr


# The query of the CDN's signed URL, in the order the real CDN's carry it:
# the signature last, where the hub library's forty-character tail of a URL
# lands.
SIGNED_QUERY = "Expires=1700000000&Signature=synthetic-not-a-secret"


def _cli_with_the_bytes_on_a_signed_cdn(
    fake_hub: FakeHub, hub_env: dict[str, str], **cdn_knobs
) -> tuple[FakeHub, subprocess.CompletedProcess[str]]:
    """`--download-model` with the hub redirecting every LFS file's bytes to
    a second fake host at a signed URL (SIGNED_QUERY), `cdn_knobs` set on
    that host: the host, for its requests, and the run."""
    with FakeHub().serve() as cdn:
        fake_hub.bytes_host = cdn.endpoint
        fake_hub.cdn_query = SIGNED_QUERY
        for knob, value in cdn_knobs.items():
            setattr(cdn, knob, value)
        return cdn, _cli(["--download-model", "--model", FAKE_REPO], hub_env)


def _assert_no_signature_on(stderr: str) -> None:
    assert "synthetic-not-a-secret" not in stderr, "the signature is on stderr"
    assert "Signature=" not in stderr and "Expires=" not in stderr, "the signed query is on stderr"


def test_cli_names_the_cdn_url_on_stderr_without_its_signed_query_when_the_bytes_fail(
    fake_hub: FakeHub, hub_env: dict[str, str]
):
    """Security (Codex round 3, download.py:340). The hub redirects an LFS
    file's bytes to its CDN at a signed URL: a query holding the signature
    and its expiry, a credential for that file. The hub library logs the
    full URL on every retry, and its exception carries it, so a drop of the
    connection put the signature on stderr twice. Given a CDN whose
    connection drops after one chunk and then answers 503, stderr names the
    file's path on the CDN and neither the signature's value nor its
    parameter, and the run fails naming the re-run, exit 3."""
    cdn, proc = _cli_with_the_bytes_on_a_signed_cdn(fake_hub, hub_env, cut_after=CUT_IN_THE_SECOND_CHUNK)

    assert proc.returncode == 3, proc.stderr[-3000:]
    assert cdn.gets("model.safetensors") == [None, f"bytes={DOWNLOAD_CHUNK_SIZE}-"], "the drop was not retried by Range"
    assert "503" in proc.stderr and "--download-model" in proc.stderr
    assert f"{cdn.endpoint}/{FAKE_REPO}/resolve/{FAKE_COMMIT}/model.safetensors" in proc.stderr, "the URL's path is not named"
    _assert_no_signature_on(proc.stderr)


def test_cli_names_the_file_on_stderr_and_never_the_tail_of_its_signed_url_when_the_resumed_bytes_are_the_wrong_size(
    fake_hub: FakeHub, hub_env: dict[str, str]
):
    """Security (Codex round 4, download.py:508). The hub library's
    `http_get` retries a dropped connection without the file's name, so its
    size check names the file by the last forty characters of its URL: for
    an LFS file, the tail of the CDN's signed query, which no URL rule on
    the message catches. Given a CDN whose connection drops after one chunk
    and whose answer to the retry's Range request is complete but short of
    the file, the failure line on stderr names the file and the sizes, and
    stderr carries neither the signature's value nor its parameter nor the
    library's `(…)` tail; the run fails naming the re-run, exit 3."""
    cdn, proc = _cli_with_the_bytes_on_a_signed_cdn(fake_hub, hub_env, cut_after=CUT_IN_THE_SECOND_CHUNK, short_resume=100)

    assert proc.returncode == 3, proc.stderr[-3000:]
    assert cdn.gets("model.safetensors") == [None, f"bytes={DOWNLOAD_CHUNK_SIZE}-"], "the drop was not retried by Range"
    _assert_no_signature_on(proc.stderr)
    assert "(…)" not in proc.stderr, "the library's tail of the URL is on stderr"
    failure = proc.stderr.strip().splitlines()[-1]
    big = len(FAKE_FILES["model.safetensors"])
    assert "model.safetensors" in failure and f"{big - 100}" in failure and f"{big}" in failure, failure
    assert "--download-model" in failure, failure


def _interrupt(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGINT)


@pytest.mark.parametrize(
    "fake_hub",
    [{"config.json": FAKE_FILES["config.json"], "model.safetensors": fake_bytes(4 * DOWNLOAD_CHUNK_SIZE)}],
    indirect=True, ids=["four-chunk model"],
)
def test_cli_cancelled_by_a_signal_keeps_the_partial_file_and_the_next_run_resumes_it(
    fake_hub: FakeHub, hub_env: dict[str, str]
):
    """Done-when 2. A four-chunk file served slowly; once the first chunk is on
    disk (the second progress line) the signal arrives: the command prints
    `cancelled`, exits 4, and the chunk stays in the cache's .incomplete blob.
    Run again at full speed, the host is asked for the rest by Range and the
    file finishes byte-identical."""
    big = fake_hub.files["model.safetensors"]
    fake_hub.throttle = (64 * 1024, 0.002)
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
    proc = subprocess.Popen([*VENV_CLI, "--download-model", "--model", FAKE_REPO], env={**os.environ, **hub_env},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **flags)
    lines = []
    for line in proc.stdout:
        lines.append(Update.parse(line))
        if lines[-1].state == "progress" and lines[-1].bytes_done >= DOWNLOAD_CHUNK_SIZE:
            _interrupt(proc)
            break
    rest = proc.stdout.read()
    stderr = proc.stderr.read()
    code = proc.wait(timeout=60)
    fake_hub.throttle = None
    assert code == EXIT_CANCELLED, (code, stderr[-3000:])
    assert rest.splitlines() == ["cancelled"], rest
    assert "Traceback" not in stderr, stderr[-3000:]
    (partial,) = _incomplete(Path(hub_env["HF_HOME"]) / "hub")
    kept = partial.stat().st_size
    assert DOWNLOAD_CHUNK_SIZE <= kept < len(big), "the partial file was not kept"
    assert partial.read_bytes() == big[:kept]

    fake_hub.requests.clear()
    proc = _cli(["--download-model", "--model", FAKE_REPO], hub_env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert fake_hub.gets("model.safetensors") == [f"bytes={kept}-"], "the rest was not asked for by Range"
    updates = [Update.parse(line) for line in proc.stdout.splitlines()]
    assert updates[0].bytes_done == kept + len(FAKE_FILES["config.json"])
    assert snapshot_files(Path(updates[-1].path))["model.safetensors"] == big
    assert not _incomplete(Path(hub_env["HF_HOME"]) / "hub")


# --- the cooperative cancel: a marker file (card #408) ----------------------
#
# The Lightroom plugin cannot signal the executable (LrTasks.execute returns
# only the exit code), so a download also stops when the cancel marker
# appears, checked between chunks, and ends exactly as the signal path does:
# `cancelled`, exit 4, the partial file kept for the next run to resume.


def _marker(tmp_path: Path) -> Path:
    """The cancel marker under a data folder of the test's own, where
    `cancel_marker_path` would put it under the per-user data directory."""
    return tmp_path / "data" / "download-cancel"


@pytest.mark.parametrize(
    "fake_hub",
    [{"config.json": FAKE_FILES["config.json"], "model.safetensors": fake_bytes(4 * DOWNLOAD_CHUNK_SIZE)}],
    indirect=True, ids=["four-chunk model"],
)
def test_download_stops_when_the_cancel_marker_appears_and_the_next_run_resumes(
    fake_hub: FakeHub, tmp_path: Path
):
    """Done-when 2 (#408) at the library: the marker is written once the first
    chunk is on disk; the download raises DownloadCancelled (the same
    exception a signal raises, so the entry point prints `cancelled` and
    exits 4 through one path), the chunk stays in the cache, the marker is
    gone on exit, and the re-run asks the host for the rest by Range."""
    fake_hub.throttle = (64 * 1024, 0.002)
    marker = _marker(tmp_path)
    big = fake_hub.files["model.safetensors"]
    seen: list[Update] = []

    def cancel_after_a_chunk(update: Update) -> None:
        seen.append(update)
        if update.bytes_done >= DOWNLOAD_CHUNK_SIZE and not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    with pytest.raises(DownloadCancelled) as cancelled:
        download_model(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=tmp_path / "hub",
                       on_update=cancel_after_a_chunk, cancel_marker=marker)
    assert CANCEL_MARKER in str(cancelled.value)
    assert not marker.exists(), "the marker was not removed on exit"
    (partial,) = _incomplete(tmp_path / "hub")
    kept = partial.stat().st_size
    assert DOWNLOAD_CHUNK_SIZE <= kept < len(big), "the partial file was not kept"
    assert partial.read_bytes() == big[:kept]
    assert seen[-1].bytes_done < len(big) + len(FAKE_FILES["config.json"]), "the download did not stop"

    fake_hub.throttle = None
    fake_hub.requests.clear()
    path, updates = _fetch(fake_hub, tmp_path / "hub")
    assert fake_hub.gets("model.safetensors") == [f"bytes={kept}-"], "the rest was not asked for by Range"
    assert snapshot_files(path)["model.safetensors"] == big
    assert not _incomplete(tmp_path / "hub")


def test_a_stale_cancel_marker_is_removed_when_a_download_starts(fake_hub: FakeHub, tmp_path: Path):
    """A marker left by an earlier click must not cancel the next download
    before it begins: it is removed on start, and the run completes."""
    marker = _marker(tmp_path)
    marker.parent.mkdir(parents=True)
    marker.touch()

    path = download_model(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=tmp_path / "hub",
                          on_update=lambda update: None, cancel_marker=marker)

    assert snapshot_files(path) == FAKE_FILES
    assert not marker.exists()


def test_a_cancel_marker_that_cannot_be_removed_on_start_is_a_download_error_naming_it(
    fake_hub: FakeHub, tmp_path: Path
):
    """Codex review 5, code finding 1 (Claude review 12, security finding 2;
    download.py:583). A folder at the marker's path (or any marker the OS
    refuses to remove) made the start's unlink raise its OSError straight
    out of download_model: a traceback, exit 1, where docs/config.md names
    exit 3 and a reason. Left in place such a marker would cancel the next
    download at its first chunk, so the start refuses instead: the failure
    is a DownloadError naming the path and what to do, raised before the
    hub is asked anything."""
    marker = _marker(tmp_path)
    marker.mkdir(parents=True)

    with pytest.raises(DownloadError) as failure:
        download_model(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=tmp_path / "hub",
                       on_update=lambda update: None, cancel_marker=marker)

    message = str(failure.value)
    assert str(marker) in message and "by hand" in message, message
    assert not fake_hub.requests, "the hub was asked with a marker that cannot be removed in place"
    assert marker.is_dir()


@contextlib.contextmanager
def _handling_an_exception_of_the_callers_own():
    try:
        raise ValueError("the caller's own, being handled while it downloads")
    except ValueError:
        yield


@pytest.mark.parametrize("called", [
    pytest.param(contextlib.nullcontext, id="plainly"),
    pytest.param(_handling_an_exception_of_the_callers_own, id="from inside the caller's handler"),
])
def test_a_cancel_marker_that_cannot_be_removed_on_exit_is_a_download_error_naming_it_with_the_model_complete(
    fake_hub: FakeHub, tmp_path: Path, called
):
    """Codex review 5, code finding 1 (Claude review 12, security finding 2;
    download.py:628). The exit's unlink in the `finally` raised the same
    bare OSError: a folder that appeared at the marker's path as the last
    chunk was counted (nothing looked for it after) left the model complete
    and the command in a traceback. The failure names the marker and what
    to do; the snapshot is laid out and the status reads installed.

    From inside the caller's handler (Claude review 13, code finding 1;
    download.py:635): the `finally` decided "something is in flight" from
    `sys.exc_info()`, which is the exception any enclosing `except` is
    handling, not this `try`'s: called from inside a caller's handler, the
    download completing with a folder at the marker's path returned the
    snapshot and reported nothing, and the next run refused on start with
    no warning of why. Whether the caller is handling an exception of its
    own must not change the outcome: the same DownloadError naming the
    marker, the model complete."""
    marker = _marker(tmp_path)
    marker.parent.mkdir(parents=True)

    def a_folder_at_the_marker_once_complete(update: Update) -> None:
        if update.bytes_done == update.bytes_total:
            marker.mkdir(exist_ok=True)

    with pytest.raises(DownloadError) as failure, called():
        download_model(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=tmp_path / "hub",
                       on_update=a_folder_at_the_marker_once_complete, cancel_marker=marker)

    message = str(failure.value)
    assert str(marker) in message and "by hand" in message, message
    assert marker.is_dir()
    status = _status(fake_hub, tmp_path / "hub")
    assert status.installed is True and status.bytes_done == FAKE_TOTAL


def test_a_cancel_marker_that_cannot_be_removed_on_exit_does_not_mask_the_cancellation_in_flight(
    fake_hub: FakeHub, tmp_path: Path
):
    """Codex review 5, code finding 1 (Claude review 12, security finding 2;
    download.py:628). A folder appearing at the marker's path mid-download
    is the marker appearing: the run is cancelled at the next chunk, and the
    `finally`'s unlink, refused by the folder, must not replace that
    DownloadCancelled (exit 4, `cancelled`, the partial kept) with its own
    failure."""
    marker = _marker(tmp_path)
    marker.parent.mkdir(parents=True)

    def a_folder_at_the_marker_after_the_first_chunk(update: Update) -> None:
        if update.bytes_done:
            marker.mkdir(exist_ok=True)

    with pytest.raises(DownloadCancelled) as cancelled:
        download_model(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=tmp_path / "hub",
                       on_update=a_folder_at_the_marker_after_the_first_chunk, cancel_marker=marker)

    assert CANCEL_MARKER in str(cancelled.value)
    assert marker.is_dir()
    assert _status(fake_hub, tmp_path / "hub").installed is False


def test_download_model_flag_exits_3_naming_the_marker_it_cannot_remove_with_nothing_on_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
):
    """Codex review 5, code finding 1 (Claude review 12, security finding 2),
    through the entry point, as the finding measured it: a folder at the
    path `--model-status` reports as `cancel_path`, then `--download-model`.
    Exit 3 with the reason on stderr naming the marker, no traceback, nothing
    on stdout, the hub (a closed port here) never asked."""
    marker = _marker(tmp_path)
    marker.mkdir(parents=True)
    monkeypatch.setattr(download, "cancel_marker_path", lambda: marker)
    monkeypatch.setattr(constants, "ENDPOINT", f"http://127.0.0.1:{closed_port()}")

    _refused_through_the_cli(capsys, "--download-model", str(marker))


def test_the_download_watches_the_documented_marker_by_default(monkeypatch, tmp_path: Path, fake_hub: FakeHub):
    """`--download-model` passes no marker: the download watches the path
    `--model-status` reports (the one docs/config.md documents), which is
    what the plugin writes to."""
    marker = _marker(tmp_path)
    monkeypatch.setattr(download, "cancel_marker_path", lambda: marker)
    marker.parent.mkdir(parents=True)
    marker.touch()

    _fetch(fake_hub, tmp_path / "hub")

    assert not marker.exists(), "the download did not use the documented marker"


# --- the model's status and removal, for the Settings button (card #408) ---
#
# Done-when 1 and 3 of card #408: the dialog must know, without downloading,
# whether the model is present, how big it is and what it is called, and it
# must be able to remove it. `--model-status` and `--remove-model` are the
# executable's answers; both are proven against the fake hub and a cache
# under tmp_path, never the real one.


def _status(hub: FakeHub, cache: Path) -> Status:
    return model_status(FAKE_REPO, endpoint=hub.endpoint, cache_dir=cache)


def test_status_of_an_absent_model_reports_not_installed_with_the_size_from_the_hub(
    fake_hub: FakeHub, tmp_path: Path
):
    """The button's title needs the name and the size before any byte moves:
    the size comes from the hub's file listing, present-ness from the cache."""
    status = _status(fake_hub, tmp_path / "hub")

    assert status == Status(FAKE_REPO, installed=False, bytes_total=FAKE_TOTAL, bytes_done=0,
                            path=None, cancel_path=str(cancel_marker_path()))
    assert not [r for r in fake_hub.requests if r.method == "GET" and "/resolve/" in r.path], "status moved bytes"


def test_status_of_a_partial_download_counts_the_bytes_already_in_the_cache(fake_hub: FakeHub, tmp_path: Path):
    fake_hub.cut_after = DOWNLOAD_CHUNK_SIZE + 4096
    with pytest.raises(DownloadError):
        _fetch(fake_hub, tmp_path / "hub")

    status = _status(fake_hub, tmp_path / "hub")

    assert status.installed is False and status.path is None
    assert status.bytes_done == DOWNLOAD_CHUNK_SIZE + len(FAKE_FILES["config.json"])
    assert status.bytes_total == FAKE_TOTAL


def test_status_of_an_installed_model_reports_it_with_its_path(fake_hub: FakeHub, tmp_path: Path):
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    status = _status(fake_hub, tmp_path / "hub")

    assert status.installed is True
    assert status.path == str(path)
    assert status.bytes_done == status.bytes_total == FAKE_TOTAL


@pytest.mark.parametrize("left", [
    pytest.param(lambda pointer: pointer.unlink(), id="a file never laid out"),
    pytest.param(lambda pointer: (pointer.unlink(), pointer.write_bytes(FAKE_FILES[pointer.name][:1000])),
                 id="a short copy in place of a file"),
])
def test_status_of_a_snapshot_missing_a_file_the_hub_lists_reports_not_installed(
    fake_hub: FakeHub, tmp_path: Path, left
):
    """Codex review 3, finding 1 (download.py:659). A download stopped while
    the snapshot was being laid out (this module's, one pointer at a time
    once every blob is whole; or the hub library's own, which mlx-vlm's load
    runs, a file at a time as each completes) leaves `refs/main` naming a
    snapshot that holds some of the model, and the status said Installed of
    it, so Settings offered Remove where the model could not load. Installed
    means whole: every file the hub lists for the repo that the model's load
    needs (mlx-vlm's own patterns, `download.MODEL_FILE_PATTERNS`) is in the
    snapshot, at the hub's size. The blobs stay counted, so the next
    download lays the snapshot out without fetching them again."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    left(path / "model.safetensors")

    status = _status(fake_hub, tmp_path / "hub")

    assert status.installed is False and status.path is None
    assert status.bytes_total == FAKE_TOTAL and status.bytes_done == FAKE_TOTAL


# What the hub writes into every repo beside the model: its listing names both,
# and mlx-vlm's load fetches neither.
HUB_EXTRAS = {".gitattributes": b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
              "README.md": b"# fake model\n"}


def _the_patterns_mlx_vlms_load_fetches() -> list[str] | None:
    """The allow patterns mlx-vlm's `get_model_path` hands the hub library's
    `snapshot_download`, which is what its `load` (calling `get_model_path`
    with none of its own) lays out: recorded from a call on a repo id, the
    fetch itself replaced. None where mlx-vlm does not import (Windows, an
    Intel Mac: it needs Apple Silicon), which is where download.py's copy
    of the list stands in."""
    try:
        from mlx_vlm import utils as mlx_vlm_utils
    except ImportError:
        return None
    recorded: list[list[str]] = []

    def record_instead_of_fetching(*, allow_patterns: list[str], **_) -> str:
        recorded.append(list(allow_patterns))
        return "/nowhere"

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(mlx_vlm_utils, "snapshot_download", record_instead_of_fetching)
        mlx_vlm_utils.get_model_path(FAKE_REPO)
    (patterns,) = recorded
    return patterns


def test_model_file_patterns_are_the_ones_mlx_vlms_load_hands_snapshot_download():
    """Rule 7 (Claude review 12, code finding 2; download.py:110-115).
    MODEL_FILE_PATTERNS is a copy of mlx-vlm's list, kept because mlx-vlm
    does not import off Apple Silicon, and mlx-vlm is not pinned: a release
    that adds or drops a pattern would make a model its load laid out read
    not installed again, with no test to say so. Where mlx-vlm imports (the
    Mac runner), the copy is what its `get_model_path` hands
    `snapshot_download`; elsewhere this is skipped, not passed."""
    patterns = _the_patterns_mlx_vlms_load_fetches()
    if patterns is None:
        pytest.skip("mlx-vlm does not import here: it needs Apple Silicon")

    assert list(MODEL_FILE_PATTERNS) == patterns


@pytest.mark.parametrize("fake_hub", [{**FAKE_FILES, **HUB_EXTRAS}], indirect=True, ids=["with the hub's extras"])
def test_status_of_a_snapshot_laid_out_by_mlx_vlms_own_load_reports_installed(fake_hub: FakeHub, tmp_path: Path):
    """Done-when 3 (Claude review 11, code finding 1; download.py:676). The
    executable's mlx engine loads the model through mlx-vlm's `load`, whose
    `get_model_path` runs the hub library's `snapshot_download` with
    mlx-vlm's own allow patterns (`*.json`, `*.safetensors`, `*.py`,
    `*.model`, `*.tiktoken`, `*.txt`, `*.jinja`), so the snapshot it lays
    out, complete and loadable, never holds the `.gitattributes` the hub
    writes into every repo nor the model card `README.md`, both of which
    the hub's listing names. Installed meant every listed file, so a model
    the first identification run fetched (the "download on first use" path
    readme.md names) read not installed, and Settings offered Download,
    with no Remove, for a model the engine had just used. Whole means what
    the model's load needs: the listed files that match mlx-vlm's patterns,
    each at the hub's size. The snapshot is laid out with the patterns
    recorded from mlx-vlm itself where it imports, download.py's copy of
    them only where it does not."""
    from huggingface_hub import snapshot_download

    patterns = _the_patterns_mlx_vlms_load_fetches() or list(MODEL_FILE_PATTERNS)
    laid_out = Path(snapshot_download(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=tmp_path / "hub",
                                      allow_patterns=patterns))
    assert snapshot_files(laid_out) == FAKE_FILES, "the load's snapshot does not hold the model files alone"

    status = _status(fake_hub, tmp_path / "hub")

    assert status.installed is True and status.path == str(laid_out)
    assert status.bytes_total == FAKE_TOTAL + sum(len(data) for data in HUB_EXTRAS.values())


def test_status_of_a_snapshot_missing_a_file_with_no_host_answering_says_what_the_cache_lays_out(
    fake_hub: FakeHub, tmp_path: Path
):
    """With the hub unreachable nothing on the machine names the files the
    repo should hold, so the status says what the cache lays out: the
    snapshot `main` names, whole as far as the cache knows. The network
    being down is not a reason to offer Download for a model that is there
    (and Download could fetch nothing then anyway)."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    (path / "model.safetensors").unlink()

    status = model_status(FAKE_REPO, endpoint=f"http://127.0.0.1:{closed_port()}", cache_dir=tmp_path / "hub")

    assert status.installed is True and status.path == str(path) and status.bytes_total is None


def test_status_of_an_installed_model_with_no_host_answering_says_the_size_is_unknown_and_the_rest(
    fake_hub: FakeHub, tmp_path: Path
):
    """The network being down is not a reason for Settings not to open: the
    status still says what the cache holds, with `bytes_total` null."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    status = model_status(FAKE_REPO, endpoint=f"http://127.0.0.1:{closed_port()}", cache_dir=tmp_path / "hub")

    assert status.bytes_total is None
    assert status.installed is True and status.path == str(path)
    assert status.bytes_done == FAKE_TOTAL


def test_status_of_an_absent_model_with_no_host_answering_never_fails(tmp_path: Path):
    """With nothing in the cache and nothing answering, the status is still
    an answer: absent, size unknown, and where to write to cancel."""
    status = model_status("fake-org/other", endpoint=f"http://127.0.0.1:{closed_port()}", cache_dir=tmp_path / "hub")

    assert status == Status("fake-org/other", installed=False, bytes_total=None, bytes_done=0,
                            path=None, cancel_path=str(cancel_marker_path()))


def test_status_gives_up_on_a_hub_that_accepts_the_connection_and_never_answers(monkeypatch, tmp_path: Path):
    """The Settings dialog runs the status as it opens and waits for the exit
    code, so a hub that takes the connection and then says nothing (a stalled
    network, a captive portal) must end the request, not hang the dialog: the
    listing is waited for at most STATUS_TIMEOUT, the hub library's own
    request timeout, and then the size is unknown as when nothing answers."""
    import threading
    import time
    from http.server import ThreadingHTTPServer

    from conftest import QuietHandler, loopback_server

    released = threading.Event()

    class Stalled(QuietHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            released.wait(timeout=8)

    assert download.STATUS_TIMEOUT == 10, "the default is the hub library's DEFAULT_REQUEST_TIMEOUT"
    monkeypatch.setattr(download, "STATUS_TIMEOUT", 0.5)
    try:
        with loopback_server(Stalled, ThreadingHTTPServer) as server:
            started = time.monotonic()
            status = model_status(FAKE_REPO, endpoint=f"http://127.0.0.1:{server.server_port}",
                                  cache_dir=tmp_path / "hub")
            waited = time.monotonic() - started
    finally:
        released.set()

    assert waited < 5, f"the status waited {waited:.1f}s on a hub that never answered"
    assert status == Status(FAKE_REPO, installed=False, bytes_total=None, bytes_done=0,
                            path=None, cancel_path=str(cancel_marker_path()))


# What a host at HF_ENDPOINT can answer 200 with that is not the hub's answer.
_NOT_A_HUBS_ANSWER = [
    ("not json", "text/html", b"<html><body>Sign in to the network</body></html>"),
    ("json of another shape", "application/json", b"[1, 2, 3]"),
    ("a json object without id", "application/json", b'{"error": "blocked"}'),
    ("a sibling without rfilename", "application/json",
     b'{"id": "fake-org/fake-model", "siblings": [{"size": 3}]}'),
    ("a size that is not a number", "application/json",
     b'{"id": "fake-org/fake-model", "siblings": [{"rfilename": "model.safetensors", "size": "big"}]}'),
    ("a name that is null", "application/json",
     b'{"id": "fake-org/fake-model", "siblings": [{"rfilename": null, "size": 3}]}'),
    ("a name that is a number", "application/json",
     b'{"id": "fake-org/fake-model", "siblings": [{"rfilename": 7, "size": 3}]}'),
    ("lastModified that is a number", "application/json", b'{"id": "fake-org/fake-model", "lastModified": 5}'),
    ("createdAt that is a number", "application/json", b'{"id": "fake-org/fake-model", "createdAt": 5}'),
    ("evalResults that is a list of numbers", "application/json",
     b'{"id": "fake-org/fake-model", "evalResults": [5]}'),
]


@pytest.mark.parametrize(("content_type", "body", "installed"), [
    *[(content_type, body, False) for _, content_type, body in _NOT_A_HUBS_ANSWER],
    *[(content_type, body, True) for name, content_type, body in _NOT_A_HUBS_ANSWER if name.startswith("a name")],
], ids=[*[name for name, _, _ in _NOT_A_HUBS_ANSWER],
        *[f"{name}, the model installed" for name, _, _ in _NOT_A_HUBS_ANSWER if name.startswith("a name")]])
def test_status_treats_a_hub_answering_200_with_something_else_as_unreachable(
    fake_hub: FakeHub, tmp_path: Path, content_type: str, body: bytes, installed: bool
):
    """Security (Claude review 9 of #16, download.py:648; Codex review 4,
    security finding 1 and Claude review 11, security finding 1,
    download.py:672-677; Codex review 5, security finding 1 and Claude
    review 12, security finding 1 and code finding 1, download.py:690). A
    host at HF_ENDPOINT that answers 200 with
    something that is not the hub's answer (a captive portal's sign-in page,
    a proxy's block page, a JSON of another shape) raised the library's
    decoding error out of the status uncaught: a traceback in
    melampus-cli.log naming source paths, and the dialog pointing at exit 1
    instead of the row. Closed for a page and a JSON list, it stayed open
    for a JSON object: one without `id` (a proxy's or a mirror's JSON error
    page, the commonest non-hub JSON answer) and one whose sibling has no
    `rfilename` raised the library's KeyError, not in the handler's tuple,
    and a sibling whose `size` is a string passed the handler and broke the
    sum of the sizes outside it. Closed for those, it stayed open for a
    sibling whose `rfilename` is null or a number, which the library takes
    unchecked and which `_holds`, run outside the handler, handed to the
    library's `filter_repo_objects`: a ValueError exactly when the model is
    installed (with the cache empty `_holds` never ran), so the row that
    should read Installed became the exit-1 note; and for `lastModified`,
    `createdAt` or `evalResults` of another shape, the library's own
    AttributeError parsing them, not in the tuple either. The status never
    fails for the network: whatever the hub's answer does wrong, such a hub
    is one that could not be reached, the size unknown and installed what
    the cache lays out; through the CLI that is exit 0, the JSON on stdout
    with `bytes_total` null, and no traceback on stderr."""
    from conftest import QuietHandler, loopback_server

    class Elsewhere(QuietHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body)

    cache = tmp_path / "hf" / "hub"
    path = str(_fetch(fake_hub, cache)[0]) if installed else None
    with loopback_server(Elsewhere) as server:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        status = model_status(FAKE_REPO, endpoint=endpoint, cache_dir=cache)
        proc = _cli(["--model-status", "--model", FAKE_REPO], {"HF_ENDPOINT": endpoint, "HF_HOME": str(tmp_path / "hf")})

    assert status == Status(FAKE_REPO, installed=installed, bytes_total=None, bytes_done=FAKE_TOTAL if installed else 0,
                            path=path, cancel_path=str(cancel_marker_path()))
    assert proc.returncode == 0 and "Traceback" not in proc.stderr, proc.stderr[-3000:]
    printed = json.loads(proc.stdout)
    assert printed["bytes_total"] is None and printed["installed"] is installed and printed["path"] == path


@pytest.mark.parametrize(("host", "carried"), [("127.0.0.1", True), ("hub.example", False)],
                         ids=["loopback http", "remote http"])
def test_status_sends_the_user_token_to_a_loopback_hub_and_never_to_a_remote_http_hub(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, carried: bool
):
    """Security (Codex review 1 of #16, download.py:645). The status asked the
    hub through the hub library's default client, not `_hub_client`, which
    `download_model` installs for every request the library makes: so
    `--model-status`, run on its own as the Settings dialog runs it, sent the
    user's token (`hf auth login`, or HF_TOKEN) to the hub whatever its origin,
    in cleartext to an `http://` hub on another machine. The status goes
    through the same client, so the token's rule is one rule: carried to a
    loopback hub, stripped from a request to a remote http one. The remote
    name resolves to the fake hub here, in this process only, so the request
    it records is the one that would have left the machine."""
    import socket

    from huggingface_hub import set_client_factory
    from huggingface_hub.utils._http import default_client_factory

    monkeypatch.setenv("HF_TOKEN", "synthetic-token")
    # The client `--model-status` starts with: the library's own, before any
    # download in this process installed the protected one.
    set_client_factory(default_client_factory)
    resolve = socket.getaddrinfo
    monkeypatch.setattr(socket, "getaddrinfo", lambda name, *rest, **kw: resolve(
        "127.0.0.1" if name == "hub.example" else name, *rest, **kw))
    port = fake_hub.endpoint.rpartition(":")[2]

    status = model_status(FAKE_REPO, endpoint=f"http://{host}:{port}", cache_dir=tmp_path / "hub")

    assert status.bytes_total == FAKE_TOTAL, "the hub was not asked"
    assert fake_hub.requests, "no request reached the hub"
    expected = "Bearer synthetic-token" if carried else None
    assert all(r.authorization == expected for r in fake_hub.requests), fake_hub.requests


def test_the_cancel_marker_lives_under_the_per_user_data_directory_beside_the_caches():
    """The plugin writes this file to cancel (docs/config.md § Downloading the
    model); it is named once here and the status carries it, so the plugin
    never derives the per-user directory itself."""
    from melampus import config

    marker = cancel_marker_path()
    assert marker.name == CANCEL_MARKER == "download-cancel"
    assert marker == config.cache_file(CANCEL_MARKER)
    assert marker.is_relative_to(config._data_root())


def test_remove_deletes_the_installed_model_from_the_cache(fake_hub: FakeHub, tmp_path: Path):
    """Done-when 3 (#408), Remove: the repo's whole cache folder goes, through
    the hub library's own deletion, and the status reads absent again."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    removed = remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")

    assert removed == tmp_path / "hub" / FAKE_FOLDER
    assert not removed.exists() and not path.exists()
    status = _status(fake_hub, tmp_path / "hub")
    assert status.installed is False and status.bytes_done == 0 and status.path is None


FAKE_FORK = "fake-org/fake-fork"


@pytest.mark.parametrize("removed", [FAKE_REPO, FAKE_FORK], ids=["the model", "its fork"])
def test_remove_deletes_the_named_model_and_no_other_repo_at_the_same_commit(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, removed: str
):
    """Security (Codex review 1 of #16, download.py:675). The removal named
    the repo's revisions to the hub library's `delete_revisions`, which
    searches the whole cache by commit hash and takes the first repo found
    at it: a fork (or a mirror) cached at the same commit could be the one
    deleted, the model asked for staying installed. Given the model and a
    fork of it at the same commit in one cache, whichever of the two is
    removed, that repo's folder goes and the other's snapshot stays. The
    scan yields its repos in a frozenset's order, which varies by process,
    so the case is pinned: the other repo comes first."""
    from dataclasses import replace

    kept = FAKE_FORK if removed == FAKE_REPO else FAKE_REPO
    paths = {FAKE_REPO: _fetch(fake_hub, tmp_path / "hub")[0]}
    with FakeHub(repo=FAKE_FORK).serve() as fork:
        paths[FAKE_FORK] = _fetch(fork, tmp_path / "hub", repo=FAKE_FORK)[0]
    assert paths[FAKE_REPO].name == paths[FAKE_FORK].name == FAKE_COMMIT, "the two repos do not share the commit"
    scan = download.scan_cache_dir

    def other_first(cache: Path):
        info = scan(cache)
        return replace(info, repos=tuple(sorted(info.repos, key=lambda r: r.repo_id != kept)))

    monkeypatch.setattr(download, "scan_cache_dir", other_first)

    gone = remove_model(removed, cache_dir=tmp_path / "hub")

    assert gone == tmp_path / "hub" / repo_folder_name(repo_id=removed, repo_type="model")
    assert not gone.exists() and not paths[removed].exists(), f"{removed} is still installed"
    assert paths[kept].exists() and snapshot_files(paths[kept]) == FAKE_FILES, f"{kept} was removed instead"


def test_remove_with_nothing_installed_says_so(fake_hub: FakeHub, tmp_path: Path):
    with pytest.raises(DownloadError) as failure:
        remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")
    assert FAKE_REPO in str(failure.value) and "nothing to remove" in str(failure.value)


def _refused_removal_leaving_the_aside_folder(
    fake_hub: FakeHub, cache: Path
) -> tuple[Path, Path, Path, DownloadError]:
    """The model fetched, then its removal refused by a snapshot folder it
    cannot delete (read-only, restored once the removal has returned):
    the repo's folder is gone from the cache under its own name and what
    could not be deleted sits under the aside name. Returns the snapshot,
    the repo's folder, the aside folder (`download._incomplete`, the one
    name for the mark) and the refusal."""
    path, _ = _fetch(fake_hub, cache)
    folder = cache / FAKE_FOLDER
    aside = download._incomplete(folder)
    os.chmod(path, 0o500)
    try:
        with pytest.raises(DownloadError) as failure:
            remove_model(FAKE_REPO, cache_dir=cache)
    finally:
        for snapshot in (path, aside / "snapshots" / path.name):
            if snapshot.is_dir():
                os.chmod(snapshot, 0o700)
    return path, folder, aside, failure.value


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="a read-only folder does not stop a deletion here")
def test_remove_refused_by_a_folder_it_cannot_delete_leaves_the_model_gone_from_the_cache_and_set_aside(
    fake_hub: FakeHub, tmp_path: Path
):
    """Done-when 3 (Codex review 2, finding 1; Claude review 9, code finding 1;
    Claude review 10, code finding 1; download.py:696). The hub library's
    deletion strategy is one rmtree, which deletes what it can and stops at
    the first entry it cannot; the library catches the PermissionError, logs
    it and returns. The removal then said "the folder is still there" of a
    model whose blobs and refs were gone: the row stayed Installed, a second
    removal said "nothing to remove" with the folder still in the cache, and
    the next download fetched everything again. The invariant: after a
    refused removal the model is whole, or gone from the cache's view, never
    both. Here it is gone: the repo's folder was set aside within the cache
    before the deletion, so what could not be deleted sits under the aside
    name, the message names it for the owner to delete by hand, the status
    reads absent, a second removal has nothing to remove, and once the aside
    folder is deleted by hand a download installs the model whole again."""
    path, folder, aside, failure = _refused_removal_leaving_the_aside_folder(fake_hub, tmp_path / "hub")

    assert FAKE_REPO in str(failure) and str(aside) in str(failure)
    assert "by hand" in str(failure), failure
    assert not folder.exists(), "the repo's folder is still in the cache under its own name"
    assert aside.is_dir(), "what could not be deleted is not where the message says"
    status = _status(fake_hub, tmp_path / "hub")
    assert status.installed is False and status.bytes_done == 0 and status.path is None
    with pytest.raises(DownloadError, match="nothing to remove"):
        remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")
    shutil.rmtree(aside)
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    assert snapshot_files(path) == FAKE_FILES and _status(fake_hub, tmp_path / "hub").installed is True


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="a read-only folder does not stop a deletion here")
def test_remove_refuses_while_an_earlier_refused_removal_left_its_set_aside_folder_naming_it_and_the_model_untouched(
    fake_hub: FakeHub, tmp_path: Path
):
    """Done-when 3 (Claude review 11, code finding 2; download.py:740-746).
    The set-aside name is one fixed name, and a refused removal leaves the
    folder under it for the owner to delete by hand. An owner who instead
    downloads the model again (the row reads Download) and later clicks
    Remove had the rename onto that folder refused with the OS's errno line
    (`[Errno 66] Directory not empty`, on Windows `[WinError 183]` even for
    an empty one), the same on every Remove after, saying neither the
    reason nor what to do. Given the aside folder of an earlier refused
    removal, the removal is refused before anything moves: the message
    names the folder as left by an earlier refused removal, says to delete
    it by hand, and the model is untouched; once it is gone, Remove goes
    ahead."""
    path, folder, aside, _ = _refused_removal_leaving_the_aside_folder(fake_hub, tmp_path / "hub")
    assert aside.is_dir() and not folder.exists()
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    assert _status(fake_hub, tmp_path / "hub").installed is True

    with pytest.raises(DownloadError) as failure:
        remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")

    message = str(failure.value)
    assert FAKE_REPO in message and str(aside) in message, message
    assert "earlier" in message and "refused" in message and "by hand" in message, message
    assert "untouched" in message and "Errno" not in message, message
    assert folder.is_dir() and aside.is_dir() and snapshot_files(path) == FAKE_FILES
    assert _status(fake_hub, tmp_path / "hub").installed is True
    shutil.rmtree(aside)
    assert remove_model(FAKE_REPO, cache_dir=tmp_path / "hub") == folder
    assert not folder.exists() and not aside.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="a symlink to a folder needs a privilege here")
def test_remove_refuses_a_repo_folder_that_is_a_link_with_exit_3_leaving_the_link_and_its_target(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """Security (Claude review 10, security finding 1; download.py:696). A
    repo folder that is a symbolic link (one big model moved to another disk
    and linked back, a layout the hub library's scan accepts and the status
    reports installed) made the removal end in a 26-line traceback, exit 1:
    rmtree refuses a link with a plain OSError, the one deletion failure the
    library's strategy does not swallow, and the CLI mapped DownloadError
    alone. The refusal is right, never delete through a link; its shape is
    a DownloadError naming the link and where to remove the model instead,
    exit 3 from the CLI, nothing on stdout, no traceback on stderr, and the
    link and its target untouched."""
    elsewhere = tmp_path / "elsewhere"
    path, _ = _fetch(fake_hub, elsewhere)
    cache = tmp_path / "hub"
    cache.mkdir()
    link = cache / FAKE_FOLDER
    link.symlink_to(elsewhere / FAKE_FOLDER, target_is_directory=True)
    assert _status(fake_hub, cache).installed is True, "the scan does not accept the link"

    with pytest.raises(DownloadError) as failure:
        remove_model(FAKE_REPO, cache_dir=cache)

    assert FAKE_REPO in str(failure.value) and str(link) in str(failure.value) and "link" in str(failure.value)
    assert link.is_symlink() and snapshot_files(path) == FAKE_FILES, "the link or its target was touched"

    _cli_sees_the_cache(monkeypatch, cache)
    _refused_through_the_cli(capsys, "--remove-model", str(link))
    assert link.is_symlink() and snapshot_files(path) == FAKE_FILES, "the link or its target was touched"


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="an unreadable file does not stop a probe here")
def test_remove_refuses_with_the_reason_when_a_lock_it_probes_cannot_be_opened(fake_hub: FakeHub, tmp_path: Path):
    """Security (Claude review 10, security finding 1). The probe for a
    running download opens each of the repo's lock files; one it cannot open
    raised the OSError through the CLI as a traceback. It is a DownloadError
    naming the lock, exit 3, and the model stays."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    lock = _lock_dir(tmp_path / "hub") / "abc.lock"
    lock.touch()
    os.chmod(lock, 0)
    try:
        with pytest.raises(DownloadError) as failure:
            remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")
    finally:
        os.chmod(lock, 0o600)

    assert FAKE_REPO in str(failure.value) and str(lock) in str(failure.value)
    assert path.exists() and snapshot_files(path) == FAKE_FILES, "the model was removed"


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="an unsearchable folder does not stop a scan here")
def test_status_and_remove_refuse_with_the_reason_when_another_repo_in_the_cache_cannot_be_scanned(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """Security (Claude review 13, security finding 1; download.py:655-665).
    The hub library's scan of the cache walks every repo folder in it, other
    tools' models included; one the process cannot search raised its
    PermissionError through `--model-status` and `--remove-model` as a
    traceback, exit 1, naming the build machine's source paths, where the
    contract is exit 3 with a reason. It is a DownloadError naming the
    cache and the folder the OS named, from both, and through the CLI exit
    3 with the folder on stderr, nothing on stdout, no traceback; the model
    stays."""
    cache = tmp_path / "hub"
    path, _ = _fetch(fake_hub, cache)
    other = cache / "models--other--repo"
    other.mkdir()
    os.chmod(other, 0)
    try:
        for ask in (lambda: _status(fake_hub, cache), lambda: remove_model(FAKE_REPO, cache_dir=cache)):
            with pytest.raises(DownloadError) as failure:
                ask()
            message = str(failure.value)
            assert str(cache) in message and str(other) in message and "permissions" in message, message
        _cli_sees_the_cache(monkeypatch, cache)
        for flag in ("--model-status", "--remove-model"):
            _refused_through_the_cli(capsys, flag, str(other))
    finally:
        os.chmod(other, 0o700)
    assert snapshot_files(path) == FAKE_FILES, "the model was removed"


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="an unsearchable folder does not stop a stat here")
def test_status_and_remove_refuse_with_the_reason_when_the_cache_itself_cannot_be_stat_ed(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """Security (Codex review 7, code finding 1; Claude review 14, security
    finding 1; download.py:664). The cache's own `is_dir` guard ran before
    the bound round 13 put around the scan: a cache whose parent the
    process cannot search (`HF_HOME` at mode 000, or the folder `--cache`
    names under one) raised its PermissionError from `Path.is_dir` through
    `--model-status` and `--remove-model` as a traceback, exit 1. It is the
    one DownloadError every read of the cache says, naming the cache, from
    both, and through the CLI exit 3 with the cache on stderr, nothing on
    stdout, no traceback; the model stays."""
    cache = tmp_path / "hf" / "hub"
    path, _ = _fetch(fake_hub, cache)
    os.chmod(cache.parent, 0)
    try:
        for ask in (lambda: _status(fake_hub, cache), lambda: remove_model(FAKE_REPO, cache_dir=cache)):
            with pytest.raises(DownloadError) as failure:
                ask()
            message = str(failure.value)
            assert str(cache) in message and "permissions" in message, message
        _cli_sees_the_cache(monkeypatch, cache)
        for flag in ("--model-status", "--remove-model"):
            _refused_through_the_cli(capsys, flag, str(cache))
    finally:
        os.chmod(cache.parent, 0o700)
    assert snapshot_files(path) == FAKE_FILES, "the model was removed"


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="an unsearchable folder does not stop a count here")
def test_status_refuses_with_the_reason_when_the_repo_blobs_cannot_be_counted(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """Security (Claude review 13, security finding 1; download.py:668-671).
    The status counts the repo's bytes by listing and stat-ing its `blobs`
    folder; one the process cannot search raised the PermissionError as a
    traceback, exit 1. It is a DownloadError naming the cache and the
    folder, exit 3 through the CLI with the folder on stderr, nothing on
    stdout, no traceback. The snapshot's files are copies of the blobs
    here, the layout the hub library makes where it cannot link, so the
    scan (which stats the snapshot's files) reads them and the count is
    what meets the folder."""
    cache = tmp_path / "hub"
    path, _ = _fetch(fake_hub, cache)
    for file in path.iterdir():
        blob = file.resolve()
        file.unlink()
        shutil.copyfile(blob, file)
    blobs = cache / FAKE_FOLDER / "blobs"
    os.chmod(blobs, 0)
    try:
        with pytest.raises(DownloadError) as failure:
            _status(fake_hub, cache)
        message = str(failure.value)
        assert str(cache) in message and str(blobs) in message and "permissions" in message, message
        _cli_sees_the_cache(monkeypatch, cache)
        _refused_through_the_cli(capsys, "--model-status", str(blobs))
    finally:
        os.chmod(blobs, 0o700)


def _lock_dir(cache: Path) -> Path:
    """The cache's locks folder for the fake repo, where a running download
    holds the hub library's per-file lock on the blob it is appending to (the
    one `_fetch` takes)."""
    lock_dir = cache / ".locks" / FAKE_FOLDER
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir


def test_remove_refuses_while_a_download_holds_the_lock(fake_hub: FakeHub, tmp_path: Path):
    """Removing the model out from under a running download is refused, exit
    3 from the CLI, and the model stays."""
    from huggingface_hub.utils import WeakFileLock

    path, _ = _fetch(fake_hub, tmp_path / "hub")
    with WeakFileLock(_lock_dir(tmp_path / "hub") / "abc.lock"):
        with pytest.raises(DownloadError) as failure:
            remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")
    assert "running" in str(failure.value) and FAKE_REPO in str(failure.value)
    assert path.exists(), "the model was removed under a running download"


def test_remove_goes_ahead_once_the_download_has_released_the_lock(fake_hub: FakeHub, tmp_path: Path):
    """A lock file left behind by a download that finished is not a running
    download: the removal takes the lock itself and goes ahead."""
    from huggingface_hub.utils import WeakFileLock

    _fetch(fake_hub, tmp_path / "hub")
    with WeakFileLock(_lock_dir(tmp_path / "hub") / "abc.lock"):
        pass

    assert remove_model(FAKE_REPO, cache_dir=tmp_path / "hub").exists() is False, "the lock outlived its holder"


def test_remove_holds_the_locks_a_download_takes_through_the_rename_and_the_deletion(
    fake_hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Codex review 8, code finding 1 (download.py:765). The probe for a
    running download took each of the repo's locks and released it at once,
    so a download starting after the probe and before the rename took its
    lock (the one `_fetch` takes on the blob it appends to) and appended
    to a blob the removal then set aside and deleted: the model removed
    under a running download, its partial file lost, where the contract
    is the refusal. The removal holds every lock the download would take
    from the probe through the rename and the deletion: a download
    starting at either moment finds its lock held (as the probe finds a
    download's: `_fetch`'s own timeout refuses it, or it waits), and the
    lock is free once the removal has returned."""
    from huggingface_hub.utils import WeakFileLock

    path, _ = _fetch(fake_hub, tmp_path / "hub")
    lock = _lock_dir(tmp_path / "hub") / "abc.lock"
    lock.touch()
    a_download_got_the_lock: dict[str, bool] = {}

    def a_download_starts(at: str) -> None:
        try:
            with WeakFileLock(lock, timeout=0.2):
                a_download_got_the_lock[at] = True
        except Timeout:
            a_download_got_the_lock[at] = False

    rename, execute = Path.rename, download.DeleteCacheStrategy.execute

    def rename_as_a_download_starts(self: Path, target: Path) -> Path:
        a_download_starts("at the rename")
        return rename(self, target)

    def execute_as_a_download_starts(self) -> None:
        a_download_starts("at the deletion")
        execute(self)

    monkeypatch.setattr(Path, "rename", rename_as_a_download_starts)
    monkeypatch.setattr(download.DeleteCacheStrategy, "execute", execute_as_a_download_starts)
    removed = remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")
    a_download_starts("after the removal")

    assert removed == path.parents[1] and not removed.exists()
    assert a_download_got_the_lock == {"at the rename": False, "at the deletion": False, "after the removal": True}


def test_model_status_flag_needs_no_folder_and_prints_one_json_object_for_the_configured_repo(
    monkeypatch, capsys
):
    """Like --download-model: no folder, [model] repo (or --model), and the
    status as one JSON object on stdout, exit 0."""
    asked = []
    monkeypatch.setattr(download, "model_status", lambda repo: asked.append(repo) or Status(
        repo, installed=False, bytes_total=None, bytes_done=0, path=None, cancel_path="/data/download-cancel"))

    assert main(["--model-status", "--no-local-config"]) == 0

    assert asked == [ModelConfig().repo]
    (status_line,) = capsys.readouterr().out.splitlines()
    assert json.loads(status_line) == {
        "repo": ModelConfig().repo, "installed": False, "bytes_total": None,
        "bytes_done": 0, "path": None, "cancel_path": "/data/download-cancel"}


def test_remove_model_flag_takes_model_and_prints_removed_with_the_path(monkeypatch, capsys, tmp_path):
    asked = []
    monkeypatch.setattr(download, "remove_model", lambda repo: asked.append(repo) or tmp_path / "gone")

    assert main(["--remove-model", "--no-local-config", "--model", "fake-org/other"]) == 0

    assert asked == ["fake-org/other"]
    assert capsys.readouterr().out.splitlines() == [f"removed {tmp_path / 'gone'}"]


def test_remove_model_flag_exits_3_with_the_refusal_on_stderr_and_nothing_on_stdout(monkeypatch, capsys):
    def refuse(repo):
        raise DownloadError("a download of x is running; cancel it first")

    monkeypatch.setattr(download, "remove_model", refuse)

    assert main(["--remove-model", "--no-local-config"]) == 3

    out, err = capsys.readouterr()
    assert out == "" and "cancel it first" in err


@pytest.mark.parametrize("flag", ["--download-model", "--model-status", "--remove-model"])
def test_model_flags_exit_3_with_the_config_key_on_stderr_when_the_repo_is_not_a_repo_id(capsys, flag: str):
    """Security (Claude review 9 of #16, download.py:643 and :675). A `[model]
    repo` (or `--model`) that is not a repo id (a URL, a path) is refused by
    the hub library's own check before the hub is asked anything; the
    download mapped that to exit 3 naming the config key, the status and the
    removal let it out as a 19-line traceback, exit 1. All three refuse the
    same way: exit 3, the key on stderr, nothing on stdout."""
    err = _refused_through_the_cli(capsys, flag, "[model] repo", repo="not a repo id/x/y")
    assert "not a repo id/x/y" in err


def test_cli_reports_absent_then_installed_then_removed_against_the_fake_hub(
    fake_hub: FakeHub, hub_env: dict[str, str]
):
    """Done-when 1 and 3 (#408) through the entry point, in the order the
    Settings dialog will see them: absent with the size, installed with the
    path after `--download-model`, absent again after `--remove-model`."""
    before = _cli(["--model-status", "--model", FAKE_REPO], hub_env)
    assert before.returncode == 0, before.stderr[-3000:]
    status = json.loads(before.stdout)
    assert status["repo"] == FAKE_REPO and status["installed"] is False
    assert status["bytes_total"] == FAKE_TOTAL and status["bytes_done"] == 0 and status["path"] is None
    assert status["cancel_path"].endswith(CANCEL_MARKER)

    downloaded = _cli(["--download-model", "--model", FAKE_REPO], hub_env)
    assert downloaded.returncode == 0, downloaded.stderr[-3000:]
    snapshot = Update.parse(downloaded.stdout.splitlines()[-1]).path

    after = _cli(["--model-status", "--model", FAKE_REPO], hub_env)
    assert after.returncode == 0, after.stderr[-3000:]
    status = json.loads(after.stdout)
    assert status["installed"] is True and status["path"] == snapshot
    assert status["bytes_done"] == status["bytes_total"] == FAKE_TOTAL

    removed = _cli(["--remove-model", "--model", FAKE_REPO], hub_env)
    assert removed.returncode == 0, removed.stderr[-3000:]
    assert removed.stdout.startswith("removed ") and not Path(snapshot).exists()
    assert json.loads(_cli(["--model-status", "--model", FAKE_REPO], hub_env).stdout)["installed"] is False
