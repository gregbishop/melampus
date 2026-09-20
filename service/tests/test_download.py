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

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FAKE_COMMIT, FAKE_FILES, FAKE_REPO, FakeHub, closed_port

from melampus import download
from melampus.cli import main
from melampus.download import (
    EXIT_CANCELLED,
    DownloadCancelled,
    DownloadError,
    Update,
    cancel_on_signals,
    download_model,
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
    assert name in message and "checksum" in message and "--download-model" in message
    assert not _incomplete(tmp_path / "hub"), "the bad partial was kept"
    blobs = tmp_path / "hub" / f"models--{FAKE_REPO.replace('/', '--')}" / "blobs"
    assert not (blobs / fake_hub.etags[name]).exists(), "the bad bytes became the blob"

    fake_hub.corrupt = set()
    fake_hub.requests.clear()
    path, _ = _fetch(fake_hub, tmp_path / "hub")

    assert fake_hub.gets(name) == [None], "the next run resumed the discarded partial instead of fetching whole"
    assert _snapshot_files(path) == FAKE_FILES


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

    assert _snapshot_files(path) == FAKE_FILES
    assert all(a == "Bearer synthetic-token" for a in fake_hub.authorizations), fake_hub.authorizations
    assert [name for name in FAKE_FILES if cdn.gets(name)] == list(FAKE_FILES), "the bytes did not come from the CDN"
    assert cdn.authorizations == [None] * len(cdn.authorizations), "the token left the hub"


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
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
    with hub.serve():
        env = {**os.environ, **hub_env, "HF_ENDPOINT": hub.endpoint}
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
