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

from pathlib import Path

import pytest
from conftest import FAKE_COMMIT, FAKE_FILES, FAKE_REPO, FakeHub, closed_port

from melampus.download import DownloadError, Update, download_model

CHUNK = 10 * 1024 * 1024  # huggingface_hub's download chunk: what a cut leaves on disk


def _fetch(hub: FakeHub, cache: Path, repo: str = FAKE_REPO) -> tuple[Path, list[Update]]:
    updates: list[Update] = []
    path = download_model(repo, endpoint=hub.endpoint, cache_dir=cache, on_update=updates.append)
    return path, updates


def _snapshot_files(path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(path.iterdir())}


def _incomplete(cache: Path) -> list[Path]:
    return sorted((cache / f"models--{FAKE_REPO.replace('/', '--')}" / "blobs").glob("*.incomplete"))


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
    total = sum(len(data) for data in FAKE_FILES.values())

    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert _snapshot_files(path) == FAKE_FILES
    assert path == tmp_path / "hub" / f"models--{FAKE_REPO.replace('/', '--')}" / "snapshots" / FAKE_COMMIT
    assert (path.parent.parent / "refs" / "main").read_text() == FAKE_COMMIT, "no ref for mlx-vlm to load offline"
    assert [u.state for u in updates] == ["progress"] * len(updates)
    assert updates[0] == Update.progress(0, total), "the total is known before any byte arrives"
    counts = [u.bytes_done for u in updates]
    assert counts == sorted(counts) and counts[-1] == total
    assert all(u.bytes_total == total for u in updates)
    assert not _incomplete(tmp_path / "hub")


def test_download_goes_only_to_the_fake_host(fake_hub: FakeHub, tmp_path: Path):
    """Done-when 3: every request of the whole conversation lands on the fake
    on 127.0.0.1: the tree listing, then metadata and bytes per file. Nothing
    is left for the internet to answer."""
    _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.requests, "the fake saw nothing"
    for method, path, _ in fake_hub.requests:
        assert path.startswith((f"/api/models/{FAKE_REPO}", f"/{FAKE_REPO}/resolve/")), (method, path)
    for name in FAKE_FILES:
        assert fake_hub.gets(name) == [None], f"{name} was not fetched whole, exactly once"


def test_download_of_a_complete_model_fetches_nothing_and_says_it_is_complete(
    fake_hub: FakeHub, tmp_path: Path
):
    """A second run is the plugin's way to check: no bytes move, the one
    update says all of the total is done, and the same path comes back."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    fake_hub.requests.clear()

    again, updates = _fetch(fake_hub, tmp_path / "hub")

    assert again == path
    total = sum(len(data) for data in FAKE_FILES.values())
    assert updates == [Update.progress(total, total)]
    assert not [r for r in fake_hub.requests if r[0] == "GET" and "/resolve/" in r[1]], fake_hub.requests


def test_download_keeps_the_partial_file_when_the_connection_drops_and_resumes_it_next_run(
    fake_hub: FakeHub, tmp_path: Path
):
    """Done-when 1, resume. The host drops the connection after one chunk of
    the large file and then answers 503: the run fails, saying so, with the
    chunk kept in the cache's `.incomplete` blob. The next run asks the host
    for the rest (a Range request from the byte it has), and the file it
    finishes is byte-identical to the host's."""
    fake_hub.cut_after = CHUNK + 4096

    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub")

    assert "503" in str(failure.value) and "--download-model" in str(failure.value)
    (partial,) = _incomplete(tmp_path / "hub")
    assert partial.stat().st_size == CHUNK
    assert partial.read_bytes() == FAKE_FILES["model.safetensors"][:CHUNK]

    fake_hub.outage = False
    fake_hub.requests.clear()
    path, updates = _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.gets("model.safetensors") == [f"bytes={CHUNK}-"], "the rest was not asked for by Range"
    assert _snapshot_files(path) == FAKE_FILES
    assert not _incomplete(tmp_path / "hub")
    total = sum(len(data) for data in FAKE_FILES.values())
    assert updates[0] == Update.progress(CHUNK + len(FAKE_FILES["config.json"]), total), (
        "the first update did not count what was already on disk")
    assert updates[-1] == Update.progress(total, total)


def test_download_of_a_repo_the_hub_does_not_have_names_the_setting_to_fix(
    fake_hub: FakeHub, tmp_path: Path
):
    with pytest.raises(DownloadError) as failure:
        _fetch(fake_hub, tmp_path / "hub", repo="fake-org/no-such-model")
    message = str(failure.value)
    assert "fake-org/no-such-model" in message
    assert "[model] repo" in message and "--model" in message


def test_download_with_no_host_answering_names_the_network(tmp_path: Path):
    port = closed_port()
    with pytest.raises(DownloadError) as failure:
        download_model(FAKE_REPO, endpoint=f"http://127.0.0.1:{port}", cache_dir=tmp_path / "hub",
                       on_update=lambda update: None)
    message = str(failure.value)
    assert f"http://127.0.0.1:{port}" in message and "network" in message
