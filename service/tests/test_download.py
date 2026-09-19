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

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FAKE_COMMIT, FAKE_FILES, FAKE_REPO, FakeHub, FakeOllama

from melampus import download
from melampus.cli import main
from melampus.download import (
    CANCEL_MARKER,
    EXIT_CANCELLED,
    DownloadCancelled,
    DownloadError,
    Status,
    Update,
    cancel_marker_path,
    cancel_on_signals,
    download_model,
    model_status,
    ollama_status,
    pull_model,
    remove_model,
    remove_ollama_model,
)

# What `.venv/bin/melampus-id` runs, from any interpreter that has the package.
VENV_CLI = [sys.executable, "-m", "melampus.cli"]

CHUNK = 10 * 1024 * 1024  # huggingface_hub's download chunk: what a cut leaves on disk


def _fetch(hub: FakeHub, cache: Path, repo: str = FAKE_REPO) -> tuple[Path, list[Update]]:
    updates: list[Update] = []
    path = download_model(repo, endpoint=hub.endpoint, cache_dir=cache, on_update=updates.append)
    return path, updates


def _snapshot_files(path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(path.iterdir())}


def _incomplete(cache: Path) -> list[Path]:
    return sorted((cache / f"models--{FAKE_REPO.replace('/', '--')}" / "blobs").glob("*.incomplete"))


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
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(DownloadError) as failure:
        download_model(FAKE_REPO, endpoint=f"http://127.0.0.1:{port}", cache_dir=tmp_path / "hub",
                       on_update=lambda update: None)
    message = str(failure.value)
    assert f"http://127.0.0.1:{port}" in message and "network" in message


def test_a_download_that_names_no_hub_reaches_no_real_one(tmp_path: Path):
    """conftest's guard: with no endpoint named, the download goes to a
    closed loopback port, not huggingface.co, and the cache is under
    tmp_path. A test that misses the fake by mistake fails here instead of
    fetching weights (docs/brief.md § hard rules)."""
    from huggingface_hub import constants

    with pytest.raises(DownloadError) as failure:
        download_model(FAKE_REPO, on_update=lambda update: None, cancel_marker=tmp_path / "download-cancel")
    message = str(failure.value)
    assert "http://127.0.0.1:" in message and "huggingface.co" not in message, message
    assert Path(constants.HF_HUB_CACHE).is_relative_to(tmp_path)
    assert os.environ["HF_HOME"].startswith(str(tmp_path)) and os.environ["HF_ENDPOINT"].startswith("http://127.0.0.1:")


def test_the_hub_library_is_a_dependency_on_every_platform():
    """download.py imports huggingface_hub directly, and on Windows nothing
    else brings it (mlx-vlm is Apple Silicon only), so the executable built
    there carries the command only if service/pyproject.toml names it, for
    every platform, in the core dependencies the lockfile installs."""
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    (declared,) = [d for d in pyproject["project"]["dependencies"] if d.startswith("huggingface_hub")]
    assert ";" not in declared, f"platform-restricted: {declared}"


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
    is [model] repo, or --model."""
    asked = []

    def fake_download(repo, *, on_update, **_):
        asked.append(repo)
        on_update(Update.progress(1, 2))
        return tmp_path / "snapshots" / "abc"

    monkeypatch.setattr(download, "download_model", fake_download)
    assert main(["--download-model"]) == 0
    assert main(["--download-model", "--model", "fake-org/other"]) == 0
    out = capsys.readouterr().out
    assert asked == ["mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit", "fake-org/other"]
    assert out.splitlines() == ["progress 1 2", f"done {tmp_path / 'snapshots' / 'abc'}"] * 2


def test_download_model_flag_exits_4_with_the_cancelled_line_when_a_signal_stops_it(monkeypatch, capsys):
    def fake_download(repo, *, on_update, **_):
        on_update(Update.progress(5, 9))
        raise DownloadCancelled("SIGINT")

    monkeypatch.setattr(download, "download_model", fake_download)
    assert main(["--download-model"]) == EXIT_CANCELLED == 4
    assert capsys.readouterr().out.splitlines() == ["progress 5 9", "cancelled"]


def test_download_model_flag_exits_3_with_the_fix_on_stderr_when_it_fails(monkeypatch, capsys):
    def fake_download(repo, *, on_update, **_):
        raise DownloadError("could not reach the hub: check the network")

    monkeypatch.setattr(download, "download_model", fake_download)
    assert main(["--download-model"]) == 3
    out, err = capsys.readouterr()
    assert out == ""
    assert "could not reach the hub: check the network" in err


def test_importing_the_download_module_disables_the_xet_transfer_as_the_readme_requires():
    """readme.md § Install: HF_HUB_DISABLE_XET=1 is not optional on some
    networks (docs/troubleshooting.md). The command sets it itself, before
    the hub library reads it, so the user need not know."""
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_XET"}
    proc = subprocess.run(
        [sys.executable, "-c",
         "import melampus.download; from huggingface_hub import constants; "
         "import os; print(os.environ['HF_HUB_DISABLE_XET'], constants.HF_HUB_DISABLE_XET)"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert proc.stdout.split() == ["1", "True"]


# --- the command, driven the way the plugin will (acceptance) --------------


def _cli(args: list[str], env: dict[str, str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*VENV_CLI, *args], env={**os.environ, **env},
                          capture_output=True, text=True, timeout=300, **kwargs)


def test_cli_downloads_the_model_reporting_progress_and_exits_0_on_done(
    fake_hub: FakeHub, hub_env: dict[str, str], tmp_path: Path
):
    """Done-when 1 through the entry point: every stdout line is a protocol
    line, progress climbs to the total, the last line is `done <path>` and
    that path holds the model, byte for byte."""
    proc = _cli(["--download-model", "--model", FAKE_REPO], hub_env)

    assert proc.returncode == 0, proc.stderr[-3000:]
    updates = [Update.parse(line) for line in proc.stdout.splitlines()]
    total = sum(len(data) for data in FAKE_FILES.values())
    assert updates[0] == Update.progress(0, total)
    assert updates[-2] == Update.progress(total, total)
    assert updates[-1].state == "done"
    assert _snapshot_files(Path(updates[-1].path)) == FAKE_FILES
    assert Path(updates[-1].path).is_relative_to(hub_env["HF_HOME"]), "the model went outside HF_HOME"


def test_cli_exits_3_naming_the_fix_when_the_repo_is_not_on_the_hub(fake_hub: FakeHub, hub_env: dict[str, str]):
    proc = _cli(["--download-model", "--model", "fake-org/no-such-model"], hub_env)
    assert proc.returncode == 3, proc.stderr[-3000:]
    assert proc.stdout == "", "an error must not be spoken in the protocol"
    assert "fake-org/no-such-model" in proc.stderr and "--model" in proc.stderr


def _interrupt(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGINT)


def test_cli_cancelled_by_a_signal_keeps_the_partial_file_and_the_next_run_resumes_it(
    hub_env: dict[str, str], tmp_path: Path
):
    """Done-when 2. A 40 MiB file served slowly; once the first chunk is on
    disk (the second progress line) the signal arrives: the command prints
    `cancelled`, exits 4, and the chunk stays in the cache's .incomplete blob.
    Run again at full speed, the host is asked for the rest by Range and the
    file finishes byte-identical."""
    big = bytes(range(256)) * (40 * 4096)
    hub = FakeHub(files={"config.json": FAKE_FILES["config.json"], "model.safetensors": big})
    hub.throttle = (64 * 1024, 0.002)
    hub.start()
    env = {**os.environ, **hub_env, "HF_ENDPOINT": hub.endpoint}
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
    try:
        proc = subprocess.Popen([*VENV_CLI, "--download-model", "--model", FAKE_REPO], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **flags)
        lines = []
        for line in proc.stdout:
            lines.append(Update.parse(line))
            if lines[-1].state == "progress" and lines[-1].bytes_done >= CHUNK:
                _interrupt(proc)
                break
        rest = proc.stdout.read()
        stderr = proc.stderr.read()
        code = proc.wait(timeout=60)
        hub.throttle = None
        assert code == EXIT_CANCELLED, (code, stderr[-3000:])
        assert rest.splitlines() == ["cancelled"], rest
        assert "Traceback" not in stderr, stderr[-3000:]
        (partial,) = _incomplete(Path(hub_env["HF_HOME"]) / "hub")
        kept = partial.stat().st_size
        assert CHUNK <= kept < len(big), "the partial file was not kept"
        assert partial.read_bytes() == big[:kept]

        hub.requests.clear()
        proc = _cli(["--download-model", "--model", FAKE_REPO], {**hub_env, "HF_ENDPOINT": hub.endpoint})
        assert proc.returncode == 0, proc.stderr[-3000:]
        assert hub.gets("model.safetensors") == [f"bytes={kept}-"], "the rest was not asked for by Range"
        updates = [Update.parse(line) for line in proc.stdout.splitlines()]
        assert updates[0].bytes_done == kept + len(FAKE_FILES["config.json"])
        assert _snapshot_files(Path(updates[-1].path))["model.safetensors"] == big
        assert not _incomplete(Path(hub_env["HF_HOME"]) / "hub")
    finally:
        hub.stop()


# --- the cooperative cancel: a marker file (card #408) ----------------------
#
# The Lightroom plugin cannot signal the executable (LrTasks.execute returns
# only the exit code), so a download also stops when the cancel marker
# appears, checked between chunks, and ends exactly as the signal path does:
# `cancelled`, exit 4, the partial file kept for the next run to resume.


def _slow_hub() -> FakeHub:
    big = bytes(range(256)) * (40 * 4096)
    hub = FakeHub(files={"config.json": FAKE_FILES["config.json"], "model.safetensors": big})
    hub.throttle = (64 * 1024, 0.002)
    hub.start()
    return hub


def test_download_stops_when_the_cancel_marker_appears_and_the_next_run_resumes(tmp_path: Path):
    """Done-when 2 (#408) at the library: the marker is written once the first
    chunk is on disk; the download raises DownloadCancelled (the same
    exception a signal raises, so the entry point prints `cancelled` and
    exits 4 through one path), the chunk stays in the cache, the marker is
    gone on exit, and the re-run asks the host for the rest by Range."""
    hub = _slow_hub()
    marker = tmp_path / "data" / "download-cancel"
    big = hub.files["model.safetensors"]
    seen: list[Update] = []

    def cancel_after_a_chunk(update: Update) -> None:
        seen.append(update)
        if update.bytes_done >= CHUNK and not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    try:
        with pytest.raises(DownloadCancelled) as cancelled:
            download_model(FAKE_REPO, endpoint=hub.endpoint, cache_dir=tmp_path / "hub",
                           on_update=cancel_after_a_chunk, cancel_marker=marker)
        assert CANCEL_MARKER in str(cancelled.value)
        assert not marker.exists(), "the marker was not removed on exit"
        (partial,) = _incomplete(tmp_path / "hub")
        kept = partial.stat().st_size
        assert CHUNK <= kept < len(big), "the partial file was not kept"
        assert partial.read_bytes() == big[:kept]
        assert seen[-1].bytes_done < len(big) + len(FAKE_FILES["config.json"]), "the download did not stop"

        hub.throttle = None
        hub.requests.clear()
        path, updates = _fetch(hub, tmp_path / "hub")
        assert hub.gets("model.safetensors") == [f"bytes={kept}-"], "the rest was not asked for by Range"
        assert _snapshot_files(path)["model.safetensors"] == big
        assert not _incomplete(tmp_path / "hub")
    finally:
        hub.stop()


def test_a_stale_cancel_marker_is_removed_when_a_download_starts(fake_hub: FakeHub, tmp_path: Path):
    """A marker left by an earlier click must not cancel the next download
    before it begins: it is removed on start, and the run completes."""
    marker = tmp_path / "data" / "download-cancel"
    marker.parent.mkdir(parents=True)
    marker.touch()

    path = download_model(FAKE_REPO, endpoint=fake_hub.endpoint, cache_dir=tmp_path / "hub",
                          on_update=lambda update: None, cancel_marker=marker)

    assert _snapshot_files(path) == FAKE_FILES
    assert not marker.exists()


def test_the_download_watches_the_documented_marker_by_default(monkeypatch, tmp_path: Path, fake_hub: FakeHub):
    """`--download-model` passes no marker: the download watches the path
    `--model-status` reports (the one docs/config.md documents), which is
    what the plugin writes to."""
    marker = tmp_path / "data" / "download-cancel"
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


def _status(hub: FakeHub | None, cache: Path, repo: str = FAKE_REPO) -> Status:
    return model_status(repo, endpoint=hub.endpoint if hub else None, cache_dir=cache)


def test_status_of_an_absent_model_reports_not_installed_with_the_size_from_the_hub(
    fake_hub: FakeHub, tmp_path: Path
):
    """The button's title needs the name and the size before any byte moves:
    the size comes from the hub's file listing, present-ness from the cache."""
    total = sum(len(data) for data in FAKE_FILES.values())

    status = _status(fake_hub, tmp_path / "hub")

    assert status == Status(FAKE_REPO, installed=False, bytes_total=total, bytes_done=0,
                            path=None, cancel_path=str(cancel_marker_path()))
    assert not [r for r in fake_hub.requests if r[0] == "GET" and "/resolve/" in r[1]], "status moved bytes"


def test_status_of_a_partial_download_counts_the_bytes_already_in_the_cache(fake_hub: FakeHub, tmp_path: Path):
    fake_hub.cut_after = CHUNK + 4096
    with pytest.raises(DownloadError):
        _fetch(fake_hub, tmp_path / "hub")

    status = _status(fake_hub, tmp_path / "hub")

    assert status.installed is False and status.path is None
    assert status.bytes_done == CHUNK + len(FAKE_FILES["config.json"])
    assert status.bytes_total == sum(len(data) for data in FAKE_FILES.values())


def test_status_of_an_installed_model_reports_it_with_its_path(fake_hub: FakeHub, tmp_path: Path):
    path, _ = _fetch(fake_hub, tmp_path / "hub")
    total = sum(len(data) for data in FAKE_FILES.values())

    status = _status(fake_hub, tmp_path / "hub")

    assert status.installed is True
    assert status.path == str(path)
    assert status.bytes_done == status.bytes_total == total


def test_status_with_no_host_answering_says_the_size_is_unknown_and_never_fails(
    fake_hub: FakeHub, tmp_path: Path
):
    """The network being down is not a reason for Settings not to open: the
    status still says what the cache holds, with `bytes_total` null."""
    import socket

    path, _ = _fetch(fake_hub, tmp_path / "hub")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    status = model_status(FAKE_REPO, endpoint=f"http://127.0.0.1:{port}", cache_dir=tmp_path / "hub")

    assert status.bytes_total is None
    assert status.installed is True and status.path == str(path)
    assert status.bytes_done == sum(len(data) for data in FAKE_FILES.values())
    absent = model_status("fake-org/other", endpoint=f"http://127.0.0.1:{port}", cache_dir=tmp_path / "hub")
    assert absent == Status("fake-org/other", installed=False, bytes_total=None, bytes_done=0,
                            path=None, cancel_path=str(cancel_marker_path()))


def test_the_cancel_marker_lives_under_the_per_user_data_directory_beside_the_caches():
    """The plugin writes this file to cancel (docs/config.md § Downloading the
    model); it is named once here and the status carries it, so the plugin
    never derives the per-user directory itself."""
    from melampus import config

    marker = cancel_marker_path()
    assert marker.name == CANCEL_MARKER == "download-cancel"
    assert marker == config._cache(CANCEL_MARKER)
    assert marker.is_relative_to(config._data_root())


def test_remove_deletes_the_installed_model_from_the_cache(fake_hub: FakeHub, tmp_path: Path):
    """Done-when 3 (#408), Remove: the repo's whole cache folder goes, through
    the hub library's own deletion, and the status reads absent again."""
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    removed = remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")

    assert removed == tmp_path / "hub" / f"models--{FAKE_REPO.replace('/', '--')}"
    assert not removed.exists() and not path.exists()
    status = _status(fake_hub, tmp_path / "hub")
    assert status.installed is False and status.bytes_done == 0 and status.path is None


def test_remove_with_nothing_installed_says_so(fake_hub: FakeHub, tmp_path: Path):
    with pytest.raises(DownloadError) as failure:
        remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")
    assert FAKE_REPO in str(failure.value) and "nothing to remove" in str(failure.value)


def test_remove_refuses_while_a_download_holds_the_lock(fake_hub: FakeHub, tmp_path: Path):
    """A running download holds the hub library's per-file lock on the blob it
    is appending to (the one `_fetch` takes); removing the model out from
    under it is refused, exit 3 from the CLI, and the model stays."""
    from huggingface_hub.utils import WeakFileLock

    path, _ = _fetch(fake_hub, tmp_path / "hub")
    lock_dir = tmp_path / "hub" / ".locks" / f"models--{FAKE_REPO.replace('/', '--')}"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(lock_dir / "abc.lock"):
        with pytest.raises(DownloadError) as failure:
            remove_model(FAKE_REPO, cache_dir=tmp_path / "hub")
    assert "running" in str(failure.value) and FAKE_REPO in str(failure.value)
    assert path.exists(), "the model was removed under a running download"
    assert remove_model(FAKE_REPO, cache_dir=tmp_path / "hub").exists() is False, "the lock outlived its holder"


def test_model_status_and_remove_model_flags_follow_the_download_flag(monkeypatch, capsys, tmp_path):
    """Both need no folder and take [model] repo or --model; status prints one
    JSON object, remove prints `removed <path>`; a refusal is exit 3 with the
    message on stderr and nothing on stdout."""
    seen = []
    monkeypatch.setattr(download, "model_status", lambda repo: seen.append(("status", repo)) or Status(
        repo, installed=False, bytes_total=None, bytes_done=0, path=None, cancel_path="/data/download-cancel"))
    monkeypatch.setattr(download, "remove_model", lambda repo: seen.append(("remove", repo)) or tmp_path / "gone")

    assert main(["--model-status"]) == 0
    assert main(["--remove-model", "--model", "fake-org/other"]) == 0
    out, err = capsys.readouterr()
    assert seen == [("status", "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"), ("remove", "fake-org/other")]
    status_line, removed_line = out.splitlines()
    assert json.loads(status_line) == {
        "repo": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit", "installed": False, "bytes_total": None,
        "bytes_done": 0, "path": None, "cancel_path": "/data/download-cancel"}
    assert removed_line == f"removed {tmp_path / 'gone'}"

    def refuse(repo):
        raise DownloadError("a download of x is running; cancel it first")

    monkeypatch.setattr(download, "remove_model", refuse)
    assert main(["--remove-model"]) == 3
    out, err = capsys.readouterr()
    assert out == "" and "cancel it first" in err


def test_cli_reports_absent_then_installed_then_removed_against_the_fake_hub(
    fake_hub: FakeHub, hub_env: dict[str, str]
):
    """Done-when 1 and 3 (#408) through the entry point, in the order the
    Settings dialog will see them: absent with the size, installed with the
    path after `--download-model`, absent again after `--remove-model`."""
    total = sum(len(data) for data in FAKE_FILES.values())

    before = _cli(["--model-status", "--model", FAKE_REPO], hub_env)
    assert before.returncode == 0, before.stderr[-3000:]
    status = json.loads(before.stdout)
    assert status["repo"] == FAKE_REPO and status["installed"] is False
    assert status["bytes_total"] == total and status["bytes_done"] == 0 and status["path"] is None
    assert status["cancel_path"].endswith(CANCEL_MARKER)

    downloaded = _cli(["--download-model", "--model", FAKE_REPO], hub_env)
    assert downloaded.returncode == 0, downloaded.stderr[-3000:]
    snapshot = Update.parse(downloaded.stdout.splitlines()[-1]).path

    after = _cli(["--model-status", "--model", FAKE_REPO], hub_env)
    assert after.returncode == 0, after.stderr[-3000:]
    status = json.loads(after.stdout)
    assert status["installed"] is True and status["path"] == snapshot
    assert status["bytes_done"] == status["bytes_total"] == total

    removed = _cli(["--remove-model", "--model", FAKE_REPO], hub_env)
    assert removed.returncode == 0, removed.stderr[-3000:]
    assert removed.stdout.startswith("removed ") and not Path(snapshot).exists()
    assert json.loads(_cli(["--model-status", "--model", FAKE_REPO], hub_env).stdout)["installed"] is False


# --- the same button for Ollama, through its pull (card #409) -------------
#
# Done-when 1: given ollama is picked and the model is absent, when Download
# is clicked, then the plugin asks Ollama to pull it and shows Ollama's
# progress. Done-when 2: given the tests, when they run, then a fake Ollama
# serves the pull progress. One protocol: Ollama's pull stream (its
# docs/api.md § Pull a Model, newline-delimited JSON objects) is mapped to
# the same `progress` / `done` / `cancelled` lines the plugin already parses.

FAKE_MODEL = "fake-org/fake-vision:1b"


def _stream(*objects: dict) -> list[bytes]:
    """The pull stream as Ollama writes it: one JSON object per line."""
    return [json.dumps(o).encode("utf-8") + b"\n" for o in objects]


def test_pull_stream_maps_each_layer_line_to_a_progress_line_summing_across_layers():
    """docs/api.md § Pull a Model: after `pulling manifest`, one object per
    layer as it downloads with `digest`, `total` and `completed`
    (`completed` may be missing until any of it is done), layers one after
    another. The protocol's `progress` is the whole pull: the sum of every
    layer's completed over the sum of every layer's total seen so far. The
    statuses around the layers (manifest, verifying, writing, removing) say
    nothing new and print nothing; `success` is `done <model>`."""
    from melampus.download import pull_updates

    updates = list(pull_updates(FAKE_MODEL, _stream(
        {"status": "pulling manifest"},
        {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100},
        {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100, "completed": 40},
        {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100, "completed": 100},
        {"status": "pulling bbb", "digest": "sha256:bbb", "total": 50, "completed": 10},
        {"status": "pulling bbb", "digest": "sha256:bbb", "total": 50, "completed": 50},
        {"status": "verifying sha256 digest"},
        {"status": "writing manifest"},
        {"status": "removing any unused layers"},
        {"status": "success"},
    )))

    assert updates == [
        Update.progress(0, 100), Update.progress(40, 100), Update.progress(100, 100),
        Update.progress(110, 150), Update.progress(150, 150),
        Update.done(FAKE_MODEL),
    ]


def test_pull_stream_of_a_model_already_there_is_one_complete_line_per_layer_then_done():
    """A layer Ollama already holds is reported once with completed equal to
    total (server/download.go: a blob on disk answers with its size for
    both), so a re-pull of a complete model prints its layers at 100% and
    `done`, the way a complete MLX model prints one equal `progress` line."""
    from melampus.download import pull_updates

    updates = list(pull_updates(FAKE_MODEL, _stream(
        {"status": "pulling manifest"},
        {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100, "completed": 100},
        {"status": "pulling bbb", "digest": "sha256:bbb", "total": 50, "completed": 50},
        {"status": "success"},
    )))

    assert updates == [Update.progress(100, 100), Update.progress(150, 150), Update.done(FAKE_MODEL)]


def test_pull_stream_error_line_for_an_unknown_model_names_the_model_and_the_setting():
    """An error mid-stream is a JSON object with `error` (server/routes.go
    streamResponse); an unknown model's is `pull model manifest: file does
    not exist` (a 404 from the library is os.ErrNotExist). It ends the pull
    as a failure naming the model and `[model] ollama_model`, with Ollama's
    own words, and nothing after it is read."""
    from melampus.download import pull_updates

    lines = _stream(
        {"status": "pulling manifest"},
        {"error": "pull model manifest: file does not exist"},
        {"status": "success"},
    )
    seen = []
    with pytest.raises(DownloadError) as failure:
        for update in pull_updates(FAKE_MODEL, lines):
            seen.append(update)
    message = str(failure.value)
    assert FAKE_MODEL in message and "[model] ollama_model" in message
    assert "file does not exist" in message
    assert seen == []


def test_pull_stream_error_line_mid_download_keeps_the_progress_so_far_and_says_to_re_run():
    """Any other error mid-stream (the library unreachable, say) fails the
    same way, with Ollama's words and the re-run hint: Ollama keeps the
    layers it has and the next pull resumes them (docs/api.md § Pull a
    Model)."""
    from melampus.download import pull_updates

    seen = []
    with pytest.raises(DownloadError) as failure:
        for update in pull_updates(FAKE_MODEL, _stream(
            {"status": "pulling manifest"},
            {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100, "completed": 40},
            {"error": "max retries exceeded: connection reset"},
        )):
            seen.append(update)
    assert seen == [Update.progress(40, 100)]
    assert "connection reset" in str(failure.value) and "--download-model" in str(failure.value)


def test_pull_stream_that_is_not_json_is_a_failure_not_a_traceback():
    from melampus.download import pull_updates

    with pytest.raises(DownloadError) as failure:
        list(pull_updates(FAKE_MODEL, [b"<html>proxy error</html>\n"]))
    assert "not JSON" in str(failure.value)


def test_pull_stream_ending_without_success_is_a_failure():
    """The connection dropped before `success`: not done, and said so."""
    from melampus.download import pull_updates

    with pytest.raises(DownloadError) as failure:
        list(pull_updates(FAKE_MODEL, _stream(
            {"status": "pulling manifest"},
            {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100, "completed": 40},
        )))
    assert "ended" in str(failure.value) and "--download-model" in str(failure.value)


# The pull against a fake Ollama on 127.0.0.1 (Done-when 2): conftest's
# FakeOllama grows the pull endpoint, streaming the layers of a model in its
# `library` and keeping what a cut-off pull had, the way Ollama does.


@pytest.fixture()
def fake_ollama() -> FakeOllama:
    ollama = FakeOllama(library={FAKE_MODEL: [3000, 1000]})
    ollama.start()
    try:
        yield ollama
    finally:
        ollama.stop()


def _pull(ollama: FakeOllama, model: str = FAKE_MODEL, **kwargs) -> list[Update]:
    """The progress updates of a pull, and, as the CLI prints it, `done
    <model>` last from what pull_model returns."""
    updates: list[Update] = []
    pulled = pull_model(model, ollama.endpoint, on_update=updates.append, **kwargs)
    return [*updates, Update.done(pulled)]


def test_pull_asks_ollama_to_pull_the_model_and_reports_its_progress_then_done(fake_ollama: FakeOllama):
    """Done-when 1 at the library: POST /api/pull with the model (docs/api.md
    § Pull a Model), the stream's layer lines become progress lines that
    climb to the whole model, `done <model>` ends it, and the fake then
    holds the model. Every request went to the fake's pull endpoint."""
    updates = _pull(fake_ollama)

    assert [p["model"] for p in fake_ollama.pulls] == [FAKE_MODEL]
    assert {path for _, path in fake_ollama.requests} == {"/api/pull"}
    assert updates[0] == Update.progress(0, 3000), "the first layer's total, before any byte"
    assert updates[-2] == Update.progress(4000, 4000)
    assert updates[-1] == Update.done(FAKE_MODEL)
    counts = [u.bytes_done for u in updates[:-1]]
    assert counts == sorted(counts)
    assert fake_ollama.models == {FAKE_MODEL: 4000}


def test_pull_of_a_model_ollama_has_no_name_for_names_the_setting(fake_ollama: FakeOllama):
    with pytest.raises(DownloadError) as failure:
        _pull(fake_ollama, model="fake-org/no-such-model:1b")
    message = str(failure.value)
    assert "fake-org/no-such-model:1b" in message and "[model] ollama_model" in message


def test_pull_with_no_ollama_answering_uses_the_backends_not_running_message(tmp_path: Path):
    """The same words the ollama backend uses for the same condition, so the
    dialog's message matches what a run would say."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(DownloadError) as failure:
        pull_model(FAKE_MODEL, f"http://127.0.0.1:{port}", on_update=lambda update: None,
                   cancel_marker=tmp_path / "download-cancel")
    message = str(failure.value)
    assert f"no Ollama server answering at http://127.0.0.1:{port}" in message
    assert "start Ollama, or set [model] ollama_url" in message


def test_pull_refused_before_any_content_carries_ollamas_words(fake_ollama: FakeOllama):
    """An error before the stream starts is an HTTP status with the same
    {"error": ...} object (server/routes.go: a bad name is 400): the message
    carries Ollama's words, not a traceback."""
    with pytest.raises(DownloadError) as failure:
        _pull(fake_ollama, model="")
    assert "invalid model name" in str(failure.value)


# The cooperative cancel, for Ollama: the marker and the signals end the
# pull the way they end the MLX download, exit 4 and `cancelled` through
# one path; Ollama keeps the layers it has and the next pull resumes them.


def _slow_ollama() -> FakeOllama:
    ollama = FakeOllama(library={FAKE_MODEL: [40 * 4096 * 256]})  # one 40 MiB layer
    ollama.throttle = (64 * 1024, 0.002)
    ollama.start()
    return ollama


def test_pull_stops_when_the_cancel_marker_appears_and_the_next_pull_resumes(tmp_path: Path):
    """Done-when 1's Cancel: the marker is written once the first megabyte is
    reported; the pull raises DownloadCancelled (the exception a signal
    raises, so the entry point prints `cancelled` and exits 4 through one
    path) with the stream closed and the marker gone. Ollama resumes a
    cancelled pull by itself (docs/api.md § Pull a Model), so the proof is
    the second pull: it is asked of the fake, its first line starts at what
    the first pull had kept, and it completes."""
    ollama = _slow_ollama()
    marker = tmp_path / "data" / "download-cancel"
    size = ollama.library[FAKE_MODEL][0]
    seen: list[Update] = []

    def cancel_after_a_megabyte(update: Update) -> None:
        seen.append(update)
        if update.bytes_done >= 1024 * 1024 and not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    try:
        with pytest.raises(DownloadCancelled) as cancelled:
            pull_model(FAKE_MODEL, ollama.endpoint, on_update=cancel_after_a_megabyte, cancel_marker=marker)
        assert CANCEL_MARKER in str(cancelled.value)
        assert not marker.exists(), "the marker was not removed on exit"
        assert 1024 * 1024 <= seen[-1].bytes_done < size, "the pull did not stop"
        assert FAKE_MODEL not in ollama.models, "the fake finished the pull after the stream closed"

        ollama.throttle = None
        updates = _pull(ollama, cancel_marker=marker)
        assert [p["model"] for p in ollama.pulls] == [FAKE_MODEL, FAKE_MODEL], "the second pull was not asked for"
        assert updates[0].bytes_done >= seen[-1].bytes_done, "the second pull did not start from what was kept"
        assert updates[-2] == Update.progress(size, size) and updates[-1] == Update.done(FAKE_MODEL)
        assert ollama.models == {FAKE_MODEL: size}
    finally:
        ollama.stop()


def test_a_stale_cancel_marker_is_removed_when_a_pull_starts(fake_ollama: FakeOllama, tmp_path: Path):
    marker = tmp_path / "data" / "download-cancel"
    marker.parent.mkdir(parents=True)
    marker.touch()

    updates = _pull(fake_ollama, cancel_marker=marker)

    assert updates[-1] == Update.done(FAKE_MODEL)
    assert not marker.exists()


def test_the_pull_watches_the_documented_marker_by_default(monkeypatch, tmp_path: Path, fake_ollama: FakeOllama):
    marker = tmp_path / "data" / "download-cancel"
    monkeypatch.setattr(download, "cancel_marker_path", lambda: marker)
    marker.parent.mkdir(parents=True)
    marker.touch()

    _pull(fake_ollama)

    assert not marker.exists(), "the pull did not use the documented marker"


# The model's status and removal in Ollama, for the same Settings row.


def _ollama_status(ollama: FakeOllama | None, model: str = FAKE_MODEL, port: int | None = None) -> Status:
    return ollama_status(model, ollama.endpoint if ollama else f"http://127.0.0.1:{port}")


def test_status_of_a_model_ollama_does_not_hold_reports_absent_with_no_size(fake_ollama: FakeOllama):
    """Present-ness from the list endpoint (docs/api.md § List Local Models:
    GET /api/tags, `models` each with `name` and `size`). A model not held
    has no size to give: the docs list sizes for local models only, so
    bytes_total is null and the button says the size is unknown."""
    status = _ollama_status(fake_ollama)

    assert status == Status(FAKE_MODEL, installed=False, bytes_total=None, bytes_done=0,
                            path=None, cancel_path=str(cancel_marker_path()))
    assert {path for _, path in fake_ollama.requests} == {"/api/tags"}


def test_status_of_a_pulled_model_reports_installed_with_its_size_and_name(fake_ollama: FakeOllama):
    """Installed, with the size the list gives for both totals, and the
    model's name as the path: it lives in Ollama under that name, which is
    what `done` printed."""
    _pull(fake_ollama)

    status = _ollama_status(fake_ollama)

    assert status.installed is True
    assert status.bytes_total == status.bytes_done == 4000
    assert status.path == FAKE_MODEL


def test_status_matches_a_name_without_a_tag_to_ollamas_latest(fake_ollama: FakeOllama):
    """docs/api.md § Model names: the tag is optional and defaults to
    `latest`, and the list names the model with it."""
    fake_ollama.library["fake-org/plain"] = [10]
    _pull(fake_ollama, model="fake-org/plain")
    assert "fake-org/plain:latest" in {m["name"] for m in fake_ollama.tags()}

    assert _ollama_status(fake_ollama, model="fake-org/plain").installed is True
    assert _ollama_status(fake_ollama, model="fake-org/plain:latest").installed is True
    assert _ollama_status(fake_ollama, model="fake-org/plain:1b").installed is False


def test_status_with_no_ollama_answering_says_absent_and_never_fails():
    """Settings must open with Ollama down: absent, size unknown, exit 0."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    status = _ollama_status(None, port=port)

    assert status == Status(FAKE_MODEL, installed=False, bytes_total=None, bytes_done=0,
                            path=None, cancel_path=str(cancel_marker_path()))


def test_remove_deletes_the_pulled_model_from_ollama(fake_ollama: FakeOllama):
    """Remove: the delete endpoint (docs/api.md § Delete a Model: DELETE
    /api/delete with the model's name, 200 when gone). The status reads
    absent again, and the removal is what the fake saw."""
    _pull(fake_ollama)

    removed = remove_ollama_model(FAKE_MODEL, fake_ollama.endpoint)

    assert removed == FAKE_MODEL
    assert fake_ollama.deletes == [FAKE_MODEL]
    assert fake_ollama.models == {}
    assert _ollama_status(fake_ollama).installed is False


def test_remove_of_a_model_ollama_does_not_hold_says_so(fake_ollama: FakeOllama):
    """404 with Ollama's not-found error (§ Delete a Model): nothing to
    remove, named."""
    with pytest.raises(DownloadError) as failure:
        remove_ollama_model(FAKE_MODEL, fake_ollama.endpoint)
    assert FAKE_MODEL in str(failure.value) and "nothing to remove" in str(failure.value)


def test_remove_with_no_ollama_answering_uses_the_not_running_message():
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(DownloadError) as failure:
        remove_ollama_model(FAKE_MODEL, f"http://127.0.0.1:{port}")
    assert f"no Ollama server answering at http://127.0.0.1:{port}" in str(failure.value)

