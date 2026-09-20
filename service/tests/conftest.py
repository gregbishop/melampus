"""Session-wide wiring for the shipped executable (card #399).

`--build-binary` makes the repo's own test command build the executable before
the binary smoke tests run, so the build is part of the test command without
costing every unit-test run the minutes a PyInstaller build takes. Without the
option the smoke tests use an existing build, or skip and say how to get one.

The build script is the one source of truth for where the executable lands
(`dist/melampus`, or `dist/melampus.exe` on Windows — card #400), so it is
loaded here rather than having its answer restated. The packaging script
(card #402) is loaded the same way for the tests on the release zip.

`photos` is the one-frame folder the scripted backend is run against, from
the executable and from the CLI alike.

`loopback_server` is the one fake-server plumbing for tests at a real HTTP
boundary (GBIF's occurrence search, Ollama's version endpoint): a handler
speaking the real protocol, served on 127.0.0.1 at an ephemeral port. Every
handler derives from `QuietHandler`, which keeps http.server's request log
out of pytest's output, and `recording_handler` is the one that answers 200
to anything and remembers what it was asked, for the assertion "this server
never heard from the client".

`fake_platform` is the one way the suite fakes the machine `on_apple_silicon`
reads (sys.platform and platform.machine(), together), whether the caller is
a providers test or the build plan.

`fake_hub` is the model host the download command (card #407) is proven
against, from the executable and from the CLI alike, and `hub_env` points a
child process at it.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub.file_download import repo_folder_name

from melampus.download import Update

# pytester runs a pytest inside pytest: how test_binary.py proves what this
# file's option and fixture do without a real build.
pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
BUILD_SCRIPT = TOOLS / "build_binary.py"
PACKAGE_SCRIPT = TOOLS / "package_plugin.py"
# The frame test_quality.py leans on, downscaled to 1200 px and stripped of
# metadata so it can be committed: the corpus is gitignored and CI has none,
# and the smoke test must analyze the same image on every platform (card #400).
FIXTURE = Path(__file__).with_name("fixtures") / "0A1A2829.jpg"
PHOTO = FIXTURE.name


def _load_tool(script: Path) -> ModuleType:
    """tools/ is not a package; import the script by path, without running it."""
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_tool(BUILD_SCRIPT)
EXECUTABLE: Path = BUILD.executable_path()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--build-binary",
        action="store_true",
        default=False,
        help=f"build {EXECUTABLE.relative_to(REPO)} with "
             f"{BUILD_SCRIPT.relative_to(REPO)} before the binary smoke tests "
             "(takes minutes; without it they use an existing build or skip)",
    )


@pytest.fixture(scope="session")
def repo() -> Path:
    """The checkout root, for tests that reach outside service/ (fixtures/,
    tools/, the docs)."""
    return REPO


@pytest.fixture(scope="session")
def build_script() -> ModuleType:
    return BUILD


@pytest.fixture(scope="session")
def package_script() -> ModuleType:
    """tools/package_plugin.py (card #402): the one place that knows the
    release zip's layout."""
    return _load_tool(PACKAGE_SCRIPT)


@pytest.fixture()
def photos(tmp_path: Path) -> Path:
    """A folder holding one JPEG, PHOTO: a copy of the committed frame."""
    folder = tmp_path / "photos"
    folder.mkdir()
    shutil.copy(FIXTURE, folder / PHOTO)
    return folder


def fake_platform(monkeypatch: pytest.MonkeyPatch, platform_name: str, machine: str) -> None:
    """The machine as `on_apple_silicon` sees it: sys.platform and
    platform.machine(), faked together, the only way the suite fakes them."""
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(platform, "machine", lambda: machine)


class QuietHandler(BaseHTTPRequestHandler):
    """The base of every fake server's handler: http.server logs each request
    to stderr, and a test's output is its assertions, not the fake's log."""

    def log_message(self, *_args) -> None:
        return None


def recording_handler(seen: list[str]) -> type[QuietHandler]:
    """A handler that answers 200 `{}` to any GET or POST and appends the path
    it was asked to `seen`: the server a test stands up to prove the client
    under test never reached it (a proxy, a redirect's destination)."""

    class Recording(QuietHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.do_GET()

    return Recording


def closed_port() -> int:
    """A loopback port nothing listens on: where a test puts the Ollama address
    so a developer's running server cannot answer for it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def loopback_server(
    handler: type[BaseHTTPRequestHandler], server_class: type[HTTPServer] = HTTPServer
) -> Iterator[HTTPServer]:
    """An HTTP server on 127.0.0.1 at an ephemeral port, serving `handler` on
    a daemon thread until exit, then stopped and joined. The caller points the
    client under test at `server.server_port`; nothing leaves the machine.
    `server_class` is ThreadingHTTPServer (a daemon thread per request) for a
    client that keeps HTTP/1.1 connections open: a single-threaded server
    would sit on an idle one instead of taking the next."""
    server = server_class(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def built_executable(request: pytest.FixtureRequest) -> Path:
    if request.config.getoption("--build-binary"):
        subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=REPO, check=True)
    if not EXECUTABLE.is_file():
        pytest.skip(
            f"no executable at {EXECUTABLE.relative_to(REPO)}; build one with "
            "`.venv/bin/python -m pytest -q --build-binary` or "
            f"`.venv/bin/python {BUILD_SCRIPT.relative_to(REPO)}`"
        )
    return EXECUTABLE


# --- the fake model host (card #407) ---------------------------------------
#
# The model download must never touch the internet in a test (docs/brief.md
# § hard rules: weights are fetched only by the owner running a command). This
# is a Hugging Face hub on 127.0.0.1 speaking exactly what huggingface_hub asks
# of the real one for a model download, read from its source: the tree listing
# (`GET /api/models/<repo>/tree/<revision>`), and the resolve endpoint's HEAD
# (ETag, X-Repo-Commit, Content-Length) and GET (bytes, honouring Range with a
# 206 and Content-Range), plus the repo info `snapshot_download` resolves the
# commit from (`GET /api/models/<repo>`). Two knobs drive the resume tests: `cut_after` drops
# the connection once that many bytes of a file have been sent and starts an
# outage (503 until `outage` is cleared); `throttle` slows the bytes so a cancel
# can land mid-file. `bytes_host` is the real hub's CDN: the resolve HEAD
# answers 302 to that host, as huggingface.co does for every LFS file, so the
# bytes are fetched from a host that is not the hub. `corrupt` names files
# served with their first byte flipped while the etag stays the true one. The
# etags are the real hub's: the sha256 of an LFS file (the weights), git's blob
# sha1 of a regular file. Every request is kept on `requests`, and the
# Authorization header each one carried on `authorizations`.

FAKE_REPO = "fake-org/fake-model"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
# One file larger than huggingface_hub's 10 MiB download chunk, so a cut leaves
# a whole chunk on disk to resume from, and a small one.
FAKE_FILES = {
    "config.json": b'{"model_type": "fake"}\n',
    "model.safetensors": bytes(range(256)) * (12 * 4096),  # 12 MiB, deterministic
}
FAKE_TOTAL = sum(len(data) for data in FAKE_FILES.values())
# The repo's folder in the cache, named the way the code under test names it.
FAKE_FOLDER = repo_folder_name(repo_id=FAKE_REPO, repo_type="model")




def snapshot_files(path: Path) -> dict[str, bytes]:
    """What a snapshot folder holds, by name: compared with FAKE_FILES."""
    return {p.name: p.read_bytes() for p in sorted(path.iterdir())}


def assert_download_completed(stdout: str, hub_env: dict[str, str]) -> None:
    """What a finished `--download-model` run says on stdout, from the CLI and
    the executable alike: every line is a protocol line, the total is known
    from the first, progress reaches it, the last line is `done <path>`, and
    that path, under HF_HOME, holds the fake host's files byte for byte."""
    updates = [Update.parse(line) for line in stdout.splitlines()]
    assert updates[0] == Update.progress(0, FAKE_TOTAL)
    assert updates[-2] == Update.progress(FAKE_TOTAL, FAKE_TOTAL)
    assert updates[-1].state == "done"
    snapshot = Path(updates[-1].path)
    assert snapshot.is_relative_to(hub_env["HF_HOME"]), "the model went outside HF_HOME"
    assert snapshot_files(snapshot) == FAKE_FILES


class FakeHub:
    """The hub's state and its handler; `serve` puts it on loopback."""

    def __init__(self, files: dict[str, bytes] = FAKE_FILES, repo: str = FAKE_REPO) -> None:
        self.files, self.repo = files, repo
        self.endpoint = ""  # set while `serve` runs
        self.requests: list[tuple[str, str, str | None]] = []
        self.cut_after: int | None = None
        self.outage = False
        self.throttle: tuple[int, float] | None = None  # (bytes per write, seconds between)
        self.bytes_host: str | None = None
        self.corrupt: set[str] = set()
        self.authorizations: list[str | None] = []
        hub = self
        etags = self.etags = {
            name: hashlib.sha256(data).hexdigest() if name.endswith(".safetensors")
            else hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            for name, data in files.items()
        }

        class Handler(QuietHandler):
            protocol_version = "HTTP/1.1"

            def _json(self, code: int, payload, headers: dict[str, str] = {}) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def _resolve(self) -> str | None:
                """The file a /<repo>/resolve/<revision>/<file> path names, else None."""
                prefix = f"/{hub.repo}/resolve/"
                if not self.path.startswith(prefix):
                    return None
                _, _, name = self.path[len(prefix):].partition("/")
                return name if name in hub.files else None

            def _unknown(self) -> None:
                self._json(404, {"error": "Repository not found"}, {"X-Error-Code": "RepoNotFound"})

            def do_GET(self):  # noqa: N802 - http.server's name
                hub.requests.append(("GET", self.path, self.headers.get("Range")))
                hub.authorizations.append(self.headers.get("Authorization"))
                path = self.path.partition("?")[0]
                if path.startswith("/api/models/"):
                    if path.startswith(f"/api/models/{hub.repo}/tree/"):
                        self._json(200, [
                            {"type": "file", "path": name, "size": len(data),
                             "oid": hashlib.sha1(data).hexdigest()}
                            for name, data in hub.files.items()
                        ])
                    elif path.removeprefix(f"/api/models/{hub.repo}") in ("", "/revision/main"):
                        self._json(200, {"id": hub.repo, "sha": FAKE_COMMIT,
                                         "siblings": [{"rfilename": name} for name in hub.files]})
                    else:
                        self._unknown()
                    return
                name = self._resolve()
                if name is None:
                    self._unknown()
                    return
                if hub.outage:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = hub.files[name]
                if name in hub.corrupt:
                    data = bytes([data[0] ^ 0xFF]) + data[1:]
                start = 0
                if self.headers.get("Range"):
                    start = int(self.headers["Range"].removeprefix("bytes=").partition("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                end = len(data)
                if hub.cut_after is not None and hub.cut_after < end:
                    end, hub.cut_after, hub.outage = hub.cut_after, None, True
                    self.close_connection = True
                if hub.throttle:
                    step, pause = hub.throttle
                    for offset in range(start, end, step):
                        self.wfile.write(data[offset:min(offset + step, end)])
                        self.wfile.flush()
                        time.sleep(pause)
                else:
                    self.wfile.write(data[start:end])
                if self.close_connection:
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)

            def do_HEAD(self):  # noqa: N802 - http.server's name
                hub.requests.append(("HEAD", self.path, None))
                hub.authorizations.append(self.headers.get("Authorization"))
                name = self._resolve()
                if name is None:
                    self._unknown()
                    return
                if hub.bytes_host:
                    self.send_response(302)
                    self.send_header("Location", f"{hub.bytes_host}{self.path}")
                    self.send_header("X-Linked-Etag", f'"{etags[name]}"')
                    self.send_header("X-Linked-Size", str(len(hub.files[name])))
                else:
                    self.send_response(200)
                    self.send_header("ETag", f'"{etags[name]}"')
                self.send_header("X-Repo-Commit", FAKE_COMMIT)
                self.send_header("Content-Length", str(len(hub.files[name])))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

        self.handler = Handler

    @contextlib.contextmanager
    def serve(self) -> Iterator[FakeHub]:
        """Serve this hub on loopback for the block; `endpoint` is its URL.
        Threaded, because huggingface_hub keeps HTTP/1.1 connections open."""
        with loopback_server(self.handler, ThreadingHTTPServer) as server:
            self.endpoint = f"http://127.0.0.1:{server.server_port}"
            yield self

    def gets(self, name: str) -> list[str | None]:
        """The Range header of every GET for `name`'s bytes, at any revision,
        in order (None: no Range)."""
        return [rng for method, path, rng in self.requests
                if method == "GET" and path.startswith(f"/{self.repo}/resolve/") and path.endswith(f"/{name}")]


@pytest.fixture()
def fake_hub() -> Iterator[FakeHub]:
    with FakeHub().serve() as hub:
        yield hub


@pytest.fixture()
def hub_env(fake_hub: FakeHub, tmp_path: Path) -> dict[str, str]:
    """The environment that points huggingface_hub at the fake and at a cache
    under tmp_path, so the real cache is never touched and nothing can reach
    the internet (Done-when 3 of card #407)."""
    return {"HF_ENDPOINT": fake_hub.endpoint, "HF_HOME": str(tmp_path / "hf")}
