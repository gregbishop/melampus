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
never heard from the client". `Silent` accepts and never answers, for the
deadline tests.

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
import socketserver
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest
from huggingface_hub.constants import DOWNLOAD_CHUNK_SIZE
from huggingface_hub.file_download import REGEX_COMMIT_HASH, repo_folder_name

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


@pytest.fixture(autouse=True)
def no_real_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test, whatever it exercises, is pointed away from the real
    Hugging Face hub and the real cache (docs/brief.md § hard rules: no
    model downloads; docs/plugin.md: weights are fetched only by the owner
    running a command). A test that reaches the hub by mistake, say a red
    test against a dispatch not yet written, then fails on a closed
    loopback port instead of fetching 18 GB into ~/.cache. The environment
    covers child processes; the constants cover this process, since the
    hub library reads the environment once at import. The tests that mean
    to reach a hub pass the fake's endpoint explicitly or through hub_env."""
    from huggingface_hub import constants

    closed = f"http://127.0.0.1:{closed_port()}"
    home = tmp_path / "no-real-hub"
    monkeypatch.setenv("HF_ENDPOINT", closed)
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setattr(constants, "ENDPOINT", closed)
    # The file URL template bakes the endpoint in at import; hf_hub_url swaps
    # an explicit endpoint in only where the template starts with ENDPOINT.
    monkeypatch.setattr(constants, "HUGGINGFACE_CO_URL_TEMPLATE",
                        closed + "/{repo_id}/resolve/{revision}/{filename}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(home / "hub"))


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


from melampus import providers

#: The real Claude Code and Codex detection, kept for the tests that run
#: them against a fake `claude` or `codex` on PATH (test_providers'
#: _fake_claude and _fake_codex); every other test gets the stubs below.
REAL_CLAUDE_CODE_VERDICT = providers.claude_code_verdict
REAL_CODEX_VERDICT = providers.codex_verdict


@pytest.fixture(autouse=True)
def no_ambient_subscription_cli(monkeypatch):
    """A developer's installed Claude Code or Codex must not decide what any
    test asserts, nor be run by one: detection (cards #421, #422) reports
    them not installed without looking. A test that wants one puts a fake
    `claude` or `codex` on PATH and restores REAL_CLAUDE_CODE_VERDICT or
    REAL_CODEX_VERDICT."""
    monkeypatch.setattr(
        providers, "claude_code_verdict",
        lambda program=None: providers.EngineVerdict(
            providers.CLAUDE_CODE, providers.CLAUDE_CODE_CLI.title, False,
            "Claude Code is not installed (kept out of the tests)"),
    )
    monkeypatch.setattr(
        providers, "codex_verdict",
        lambda program=None: providers.EngineVerdict(
            providers.CODEX, providers.CODEX_CLI.title, False,
            "Codex CLI is not installed (kept out of the tests)"),
    )


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
    """A handler that answers 200 `{}` to any GET, POST or DELETE and appends
    the path it was asked to `seen`: the server a test stands up to prove the
    client under test never reached it (a proxy, a redirect's destination)."""

    class Recording(QuietHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.do_GET()

        do_DELETE = do_POST  # noqa: N815 - http.server's name

    return Recording


def trickle(wfile, data: bytes) -> None:
    """`data` one byte every hundred milliseconds: each byte within the
    socket timeout, the whole (two seconds for twenty bytes) well past a
    sub-second deadline. The one trickle for every listener that holds a
    call for as long as it likes (the probe's headers, a frame's body, the
    list's and the delete's reply, a line of the pull's stream). Once the
    client hangs up, the next write raises; a caller's suppress(OSError)
    ends the trickle there."""
    for byte in data:
        time.sleep(0.1)
        wfile.write(bytes([byte]))


def redirecting_handler(elsewhere: str) -> type[QuietHandler]:
    """A handler that answers 302 to any GET, POST or DELETE with a Location
    at `elsewhere` (a server's root) plus the path it was asked: the squatter
    a test stands up at the address to prove the client under test never
    follows a redirect off it, `elsewhere` being a recording server's."""

    class Redirecting(QuietHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(302)
            self.send_header("Location", f"{elsewhere}{self.path}")
            self.end_headers()

        do_POST = do_DELETE = do_GET  # noqa: N815 - http.server's names

    return Redirecting


@contextlib.contextmanager
def proxy_in_the_environment(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> Iterator[HTTPServer]:
    """A proxy on loopback recording every request it is asked to `seen`,
    named by `http_proxy` for the block with no bypass list in the way: what
    urlopen's default opener would send a request through. That opener is
    built once, reading the proxy variables then, so it is started fresh
    for the environment set here. A client that stays at its address
    leaves `seen` empty."""
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(urllib.request, "_opener", None)
    with loopback_server(recording_handler(seen)) as proxy:
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_port}")
        yield proxy


class Silent(socketserver.BaseRequestHandler):
    """A listener that accepts the TCP connection and never speaks: a TLS
    handshake against it waits for a ServerHello that never comes, an HTTP
    request for a status line that never comes. Served threaded, it does not
    hold `loopback_server`'s shutdown while a client is still waiting."""

    def handle(self) -> None:
        with contextlib.suppress(OSError):
            self.request.recv(65536)
            self.request.recv(65536)


class BadStatusLine(socketserver.BaseRequestHandler):
    """A listener that answers whatever it is asked with a status line that
    is not HTTP, carrying an escape sequence and a carriage return."""

    def handle(self) -> None:
        with contextlib.suppress(OSError):
            self.request.recv(65536)
            self.request.sendall(b"\x1b[31mHTTP/9.9 OK\r\x07fake log line\r\n\r\n")


def resetting_handler(answer: bytes) -> type[socketserver.BaseRequestHandler]:
    """A listener that answers whatever it is asked with `answer` and then
    resets the connection (SO_LINGER at zero makes the close an RST, not a
    FIN, so the client's next read is a ConnectionResetError, the raw
    socket error urllib lets through unwrapped): Ollama killed, or a
    listener hanging up, while the reply is being read. The socket is
    closed here, ahead of the server's own shutdown, so the reset is what
    the client sees, not the orderly close."""

    class Resetting(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            with contextlib.suppress(OSError):
                self.request.recv(65536)
                self.request.sendall(answer)
                self.request.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self.request.close()

    return Resetting


class TricklingPull(QuietHandler):
    """A listener answering a pull's stream (POST /api/pull) with two whole
    lines, each after a pause within the deadline the tests give and the
    two together past it (a deadline on the exchange alone would end the
    stream before the second), then a line trickled a byte every tenth of
    a second: each byte within the socket timeout, the whole well past the
    deadline. The shape that held the pull, and the cancel marker read
    between lines, for as long as it liked (security review round 4)."""

    PAUSE = 0.3
    WHOLE = b'{"status": "pulling manifest"}\n'
    TRICKLED = b'{"status": "success"}\n'

    def do_POST(self) -> None:  # noqa: N802 - http.server's name
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with contextlib.suppress(OSError):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for _ in range(2):
                time.sleep(self.PAUSE)
                self.wfile.write(self.WHOLE)
            trickle(self.wfile, self.TRICKLED)


# What `.venv/bin/melampus-id` runs, spelled so it works from any interpreter
# that has the package installed (CI has no root .venv).
VENV_CLI = [sys.executable, "-m", "melampus.cli"]


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
# 206 and Content-Range), plus the repo info the library's `resolve_revision`
# resolves the commit from (`GET /api/models/<repo>`). The knobs that drive the resume tests: `cut_after` drops
# the connection once that many bytes of a file have been sent and starts an
# outage (503 until `outage` is cleared); `throttle` slows the bytes so a cancel
# can land mid-file; `ignore_range` answers a Range request with 200 and the
# whole file, as a CDN that ignores Range does; `short_resume` answers a Range
# request with a body that ends that many bytes before the file's end,
# Content-Length agreeing, a host whose resumed answer is complete and the
# wrong size (a cut then starts no outage, so the retry's Range request gets
# that answer). `bytes_host` is the real hub's CDN: the resolve HEAD
# answers 302 to that host, as huggingface.co does for every LFS file, so the
# bytes are fetched from a host that is not the hub; `cdn_query` is the query
# that Location carries, as the real CDN's signed URLs do. `corrupt` names files
# served with their first byte flipped while the etag stays the true one.
# `gated` makes the repo one the user has no access to: it is listed, but
# every resolve answers 403 with X-Error-Code GatedRepo, as huggingface.co
# does until the user has accepted the repo's terms with their token. `commit`
# is what `main` points at: a test moves the branch mid-run by setting it. The
# etags are the real hub's: the sha256 of an LFS file (the weights), git's blob
# sha1 of a regular file; `later_etag` is what every HEAD after a file's first
# answers instead, a hub that changes its story once the run has planned.
# `next_page` is a URL the tree listing names in its `Link: rel="next"`
# header, as the real hub paginates a long listing and huggingface_hub
# follows. `/api/agent-harnesses` is the hub's registry of AI coding agents,
# which huggingface_hub fetches to name the agent it runs under in its
# User-Agent unless telemetry is off: the fake's names one, AGENT_HARNESS,
# matched by the environment variable of that name. Every request is kept on
# `requests`, one HubRequest each: method, path, and the Range, Authorization
# and User-Agent headers it carried.

FAKE_REPO = "fake-org/fake-model"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
AGENT_HARNESS = "fake-agent"


def fake_bytes(size: int) -> bytes:
    """`size` deterministic bytes: a model file's stand-in."""
    return (bytes(range(256)) * (size // 256 + 1))[:size]


# One file larger than the chunk huggingface_hub's http_get iterates by (what
# a cut leaves on disk, so there is a whole chunk to resume from), and a
# small one.
FAKE_FILES = {
    "config.json": b'{"model_type": "fake"}\n',
    "model.safetensors": fake_bytes(DOWNLOAD_CHUNK_SIZE * 6 // 5),
}
FAKE_TOTAL = sum(len(data) for data in FAKE_FILES.values())
# Where a resume test cuts the large file: in its second chunk, so exactly
# one whole chunk (DOWNLOAD_CHUNK_SIZE bytes) is on disk for the next run to
# ask for the rest from.
CUT_IN_THE_SECOND_CHUNK = DOWNLOAD_CHUNK_SIZE + 4096
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


class HubRequest(NamedTuple):
    """One request the fake hub answered: what came in, as the test reads it."""

    method: str
    path: str
    range: str | None
    authorization: str | None
    user_agent: str | None


class FakeHub:
    """The hub's state and its handler; `serve` puts it on loopback."""

    def __init__(self, files: dict[str, bytes] = FAKE_FILES, repo: str = FAKE_REPO) -> None:
        self.files, self.repo = files, repo
        self.endpoint = ""  # set while `serve` runs
        self.requests: list[HubRequest] = []
        self.cut_after: int | None = None
        self.outage = False
        self.ignore_range = False
        self.short_resume = 0  # bytes a Range answer stops short of the file's end, Content-Length agreeing
        self.throttle: tuple[int, float] | None = None  # (bytes per write, seconds between)
        self.bytes_host: str | None = None
        self.cdn_query: str | None = None  # `Signature=...&Expires=...` on the LFS redirect's Location
        self.corrupt: set[str] = set()
        self.gated = False
        self.commit = FAKE_COMMIT  # what `main` points at; a test moves the branch by setting it
        self.later_etag: str | None = None  # the etag of every HEAD after a file's first
        self.next_page: str | None = None  # the tree listing's `Link: rel="next"` URL
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
                """The file a /<repo>/resolve/<revision>/<file> path names, else
                None; `commit` is what the revision resolves to, as the real
                hub answers X-Repo-Commit: the commit hash itself, or what the
                branch points at now."""
                prefix = f"/{hub.repo}/resolve/"
                path = self.path.partition("?")[0]
                if not path.startswith(prefix):
                    return None
                revision, _, name = path[len(prefix):].partition("/")
                self.commit = revision if REGEX_COMMIT_HASH.match(revision) else hub.commit
                return name if name in hub.files else None

            def _record(self) -> None:
                hub.requests.append(HubRequest(self.command, self.path, self.headers.get("Range"),
                                               self.headers.get("Authorization"), self.headers.get("User-Agent")))

            def _unknown(self) -> None:
                self._json(404, {"error": "Repository not found"}, {"X-Error-Code": "RepoNotFound"})

            def _resolvable(self) -> str | None:
                """The file a resolve path names once the repo's gate is
                passed; None with the error already sent."""
                name = self._resolve()
                if name is None:
                    self._unknown()
                elif hub.gated:
                    self._json(403, {"error": "Access to this repo is restricted"},
                               {"X-Error-Code": "GatedRepo"})
                    return None
                return name

            def do_GET(self):  # noqa: N802 - http.server's name
                self._record()
                path = self.path.partition("?")[0]
                if path == "/api/agent-harnesses":
                    self._json(200, {"standardEnvVars": ["AI_AGENT"],
                                     "harnesses": {AGENT_HARNESS: {"envVars": {"AGENT_HARNESS": "*"}}}})
                    return
                if path.startswith("/api/models/"):
                    if path.startswith(f"/api/models/{hub.repo}/tree/"):
                        self._json(200, [
                            {"type": "file", "path": name, "size": len(data),
                             "oid": hashlib.sha1(data).hexdigest()}
                            for name, data in hub.files.items()
                        ], {"Link": f'<{hub.next_page}>; rel="next"'} if hub.next_page else {})
                    elif path.removeprefix(f"/api/models/{hub.repo}") in ("", "/revision/main"):
                        # The real hub names the files only, and their sizes
                        # too when asked with blobs=true (files_metadata).
                        with_sizes = "blobs=true" in self.path.partition("?")[2].lower()
                        self._json(200, {"id": hub.repo, "sha": hub.commit, "siblings": [
                            {"rfilename": name, "size": len(data)} if with_sizes else {"rfilename": name}
                            for name, data in hub.files.items()]})
                    else:
                        self._unknown()
                    return
                name = self._resolvable()
                if name is None:
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
                if self.headers.get("Range") and not hub.ignore_range:
                    start = int(self.headers["Range"].removeprefix("bytes=").partition("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                end = len(data) - (hub.short_resume if start else 0)
                self.send_header("Content-Length", str(end - start))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if hub.cut_after is not None and hub.cut_after < end:
                    end, hub.cut_after, hub.outage = hub.cut_after, None, not hub.short_resume
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
                self._record()
                name = self._resolvable()
                if name is None:
                    return
                etag = etags[name]
                if hub.later_etag is not None and hub.heads(name) > 1:
                    etag = hub.later_etag
                if hub.bytes_host:
                    self.send_response(302)
                    self.send_header("Location", f"{hub.bytes_host}{self.path}" + (f"?{hub.cdn_query}" if hub.cdn_query else ""))
                    self.send_header("X-Linked-Etag", f'"{etag}"')
                    self.send_header("X-Linked-Size", str(len(hub.files[name])))
                else:
                    self.send_response(200)
                    self.send_header("ETag", f'"{etag}"')
                self.send_header("X-Repo-Commit", self.commit)
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

    def _resolves(self, method: str, name: str | None) -> list[HubRequest]:
        """Every `method` request for `name`'s bytes or metadata (any file's
        when `name` is None), at any revision, in order."""
        return [r for r in self.requests if r.method == method
                and r.path.startswith(f"/{self.repo}/resolve/")
                and (name is None or r.path.partition("?")[0].endswith(f"/{name}"))]

    def gets(self, name: str | None = None) -> list[str | None]:
        """The Range header of every GET for `name`'s bytes, at any revision,
        in order (None: no Range); with no name, of every GET for any file's
        bytes, so `not hub.gets()` says no bytes were fetched."""
        return [r.range for r in self._resolves("GET", name)]

    def heads(self, name: str) -> int:
        """How many times `name`'s metadata was asked for, at any revision."""
        return len(self._resolves("HEAD", name))


@pytest.fixture()
def fake_hub(request: pytest.FixtureRequest) -> Iterator[FakeHub]:
    """The hub, serving FAKE_FILES, or the files a test names by parametrizing
    this fixture indirectly (`indirect=True`), so `hub_env` points at that
    one hub."""
    with FakeHub(files=getattr(request, "param", FAKE_FILES)).serve() as hub:
        yield hub


@pytest.fixture()
def hub_env(fake_hub: FakeHub, tmp_path: Path) -> dict[str, str]:
    """The environment that points huggingface_hub at the fake and at a cache
    under tmp_path, so the real cache is never touched and nothing can reach
    the internet (Done-when 3 of card #407)."""
    return {"HF_ENDPOINT": fake_hub.endpoint, "HF_HOME": str(tmp_path / "hf")}


# --- the fake Ollama (card #406, #409) -----------------------------------
#
# An HTTP server on 127.0.0.1 speaking the endpoints of Ollama's docs/api.md
# that the service uses: § Version (`GET /api/version`, the probe),
# § Generate a chat completion (`POST /api/chat`, one response object with
# `stream` false), and for the Download button (card #409) § Pull a Model
# (`POST /api/pull`, a stream of JSON objects one per line: `pulling
# manifest`, then per layer `pulling <digest>` with `digest`, `total` and
# `completed`, then the verifying, writing and removing statuses, then
# `success`; an error is an object with `error`, mid-stream once content
# has started, else the HTTP status with the same object). No model, no
# weights, no network beyond loopback.
#
# `status` is what the version probe answers, `delay` holds that answer,
# `replies` are the texts the chat answers with in order (a 404 with Ollama's
# not-found error once they run out); every chat request's JSON body lands on
# `chats`. `library` is what can be pulled, name -> layer sizes; `models` is
# what is held, name -> size, filled by a pull, listed by § List Local Models
# (`GET /api/tags`) and emptied by § Delete a Model (`DELETE /api/delete`,
# 200, or 404 with the not-found error); `pulls` keeps every pull's body,
# `deletes` every deletion's name and `requests` every request. `throttle` (bytes per line, seconds
# between) slows a pull so a cancel can land mid-stream, and a pull the
# client cut off keeps what each layer had, so the next pull of the same
# model starts there (docs/api.md § Pull a Model: "Cancelled pulls are
# resumed from where they left off").


def ollama_chat_reply(model: str, text: str) -> dict:
    """The final response object POST /api/chat answers with when `stream` is
    false (Ollama's docs/api.md § Generate a chat completion): the text is
    `message.content`, the counts `prompt_eval_count` and `eval_count`, the
    other fields as the docs show them. The one shape every fake Ollama
    answers with: FakeOllama at the HTTP boundary, and test_providers.py's
    fake at the `urlopen` edge."""
    return {
        "model": model,
        "created_at": "2026-09-18T00:00:00Z",
        "message": {"role": "assistant", "content": text},
        "done_reason": "stop",
        "done": True,
        "total_duration": 1668506709,
        "prompt_eval_count": 26,
        "eval_count": 83,
    }


class FakeOllama:
    """The fake Ollama's state and its handler; `serve` puts it on loopback.
    `prefix` mounts the endpoints under a path, the way a reverse proxy
    does; any other path is Ollama's own 404."""

    def __init__(
        self, *, status: int = 200, delay: float = 0.0, replies: list[str] = (), prefix: str = "",
        library: dict[str, list[int]] | None = None,
    ) -> None:
        self.chats: list[dict] = []
        self.pulls: list[dict] = []
        self.deletes: list[str] = []
        self.requests: list[tuple[str, str]] = []
        self.library = dict(library or {})
        self.models: dict[str, int] = {}
        self.partial: dict[tuple[str, str], int] = {}
        self.throttle: tuple[int, float] | None = None
        self.endpoint = ""  # set while `serve` runs
        self.server_port = 0
        self.release = threading.Event()
        pending = list(replies)
        ollama = self

        class Handler(QuietHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                ollama.requests.append(("GET", self.path))
                if self.path == f"{prefix}/api/tags":
                    self._answer(200, {"models": ollama.tags()})
                    return
                if self.path != f"{prefix}/api/version":
                    self._answer(404, {"error": "404 page not found"})
                    return
                if delay:
                    ollama.release.wait(delay)
                self._answer(status, {"version": "0.0.0-fake"})

            def do_DELETE(self):  # noqa: N802 - http.server's name
                ollama.requests.append(("DELETE", self.path))
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != f"{prefix}/api/delete":
                    self._answer(404, {"error": "404 page not found"})
                    return
                name = body.get("model") or ""
                ollama.deletes.append(name)
                held = name if name in ollama.models else (name.removesuffix(":latest")
                                                             if name.endswith(":latest") else None)
                if held in ollama.models:
                    del ollama.models[held]
                    self._answer(200, {})
                else:
                    self._answer(404, {"error": f"model '{name}' not found"})

            def do_POST(self):  # noqa: N802 - http.server's name
                ollama.requests.append(("POST", self.path))
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == f"{prefix}/api/pull":
                    self._pull(body)
                    return
                if self.path != f"{prefix}/api/chat":
                    self._answer(404, {"error": "404 page not found"})
                    return
                ollama.chats.append(body)
                if not pending:
                    self._answer(404, {"error": f"model '{body.get('model')}' not found"})
                    return
                self._answer(200, ollama_chat_reply(body["model"], pending.pop(0)))

            def _pull(self, body: dict) -> None:
                ollama.pulls.append(body)
                name = body.get("model") or ""
                if not name:
                    self._answer(400, {"error": "invalid model name"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    self._line({"status": "pulling manifest"})
                    if name not in ollama.library:
                        self._line({"error": "pull model manifest: file does not exist"})
                        return
                    for index, size in enumerate(ollama.library[name]):
                        digest = "sha256:" + hashlib.sha256(f"{name}:{index}".encode()).hexdigest()
                        line = {"status": f"pulling {digest[7:19]}", "digest": digest, "total": size}
                        done = ollama.partial.get((name, digest), 0)
                        if done == 0 and name not in ollama.models:
                            self._line(line)
                        step, pause = ollama.throttle or (size, 0.0)
                        while done < size:
                            done = min(done + step, size)
                            ollama.partial[(name, digest)] = done
                            self._line({**line, "completed": done})
                            time.sleep(pause)
                        if done == size and name in ollama.models:
                            self._line({**line, "completed": done})
                    for status_ in ("verifying sha256 digest", "writing manifest", "removing any unused layers"):
                        self._line({"status": status_})
                    ollama.models[name] = sum(ollama.library[name])
                    self._line({"status": "success"})
                except (BrokenPipeError, ConnectionResetError):
                    # The client closed the stream: the pull stops with what
                    # each layer had kept, for the next pull to start from.
                    self.close_connection = True

            def _line(self, item: dict) -> None:
                self.wfile.write(json.dumps(item).encode("utf-8") + b"\n")
                self.wfile.flush()

            def _answer(self, code: int, payload: dict) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))

        self.handler = Handler

    def tags(self) -> list[dict]:
        """The models held, as § List Local Models lists them: `name` and
        `model` with the tag (`latest` when the pull named none, § Model
        names), `size`, `digest`, `modified_at`, `details`."""
        return [
            {
                "name": name if ":" in name else f"{name}:latest",
                "model": name if ":" in name else f"{name}:latest",
                "modified_at": "2026-09-18T00:00:00Z", "size": size,
                "digest": hashlib.sha256(name.encode()).hexdigest(),
                "details": {"parent_model": "", "format": "gguf", "family": "fake",
                            "families": ["fake"], "parameter_size": "1B", "quantization_level": "Q4_0"},
            }
            for name, size in self.models.items()
        ]

    @contextlib.contextmanager
    def serve(self) -> Iterator[FakeOllama]:
        """Serve this Ollama on loopback for the block; `endpoint` is its URL.
        Threaded, as the hub is, so a held probe cannot block the next
        request; a delayed answer is released when the block ends."""
        with loopback_server(self.handler, ThreadingHTTPServer) as server:
            self.server_port = server.server_port
            self.endpoint = f"http://127.0.0.1:{self.server_port}"
            try:
                yield self
            finally:
                self.release.set()
