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

import errno
import os
import shutil
import signal
import subprocess
import sys
import threading
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
from huggingface_hub import constants
from huggingface_hub.constants import DOWNLOAD_CHUNK_SIZE

from melampus import download
from melampus.cli import main
from melampus.config import ModelConfig
from melampus.download import (
    EXIT_CANCELLED,
    DownloadCancelled,
    DownloadError,
    Update,
    _hub_client,
    _token_may_go,
    cancel_on_signals,
    download_model,
)


def _fetch(hub: FakeHub, cache: Path, repo: str = FAKE_REPO) -> tuple[Path, list[Update]]:
    updates: list[Update] = []
    path = download_model(repo, endpoint=hub.endpoint, cache_dir=cache, on_update=updates.append)
    return path, updates


def _incomplete(cache: Path) -> list[Path]:
    return sorted((cache / FAKE_FOLDER / "blobs").glob("*.incomplete"))


@pytest.mark.parametrize(
    ("update", "line"),
    [
        (Update.progress(0, 18_300_000_000), "progress 0 18300000000"),
        (Update.progress(4_096, 4_096), "progress 4096 4096"),
        (Update.done("/hf/hub/models--x--y/snapshots/abc"), "done /hf/hub/models--x--y/snapshots/abc"),
        (Update.done("C:\\Users\\me\\AppData\\Local\\hf hub\\snapshots\\abc"),
         "done C:\\Users\\me\\AppData\\Local\\hf hub\\snapshots\\abc"),
        (Update.cancelled(), "cancelled"),
    ],
)
def test_progress_protocol_prints_and_parses_the_same_line(update: Update, line: str):
    """One line per update, machine-readable and stable: `progress <done> <total>`
    while bytes arrive, `done <path>` once the model is complete, `cancelled`
    when a signal stopped it. A path may hold spaces, so it is the rest of the
    line."""
    assert update.line() == line
    assert Update.parse(line) == update
    assert Update.parse(line + "\n") == update, "a line read from a pipe keeps its newline"


@pytest.mark.parametrize("line", [
    "", "progress", "progress 1", "progress one two", "progress 1 2 3",
    "done", "cancelled now", "Downloading bytes: 100%", "engine: mlx",
])
def test_progress_protocol_rejects_what_is_not_an_update(line: str):
    """The plugin must be able to tell an update from any other line."""
    with pytest.raises(ValueError):
        Update.parse(line)


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


def test_the_hub_library_is_a_dependency_on_every_platform_pinned_to_the_reviewed_version():
    """download.py imports huggingface_hub directly, and on Windows nothing
    else brings it (mlx-vlm is Apple Silicon only), so the executable built
    there carries the command only if service/pyproject.toml names it, for
    every platform, in the core dependencies the lockfile installs. Security
    (Codex round 1, pyproject.toml:28): a new dependency is pinned exactly,
    as pyinstaller is, so an install without the lockfile cannot pull a
    version nobody reviewed; what download.py leans on (`http_get`'s resume,
    `resolve_revision`, the client factory) was read at 1.26.0."""
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    (declared,) = [d for d in pyproject["project"]["dependencies"] if d.startswith("huggingface_hub")]
    assert declared == "huggingface_hub==1.26.0", f"not the exact reviewed version: {declared}"


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
