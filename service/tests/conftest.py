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

`fake_hub` is the model host the download command (card #407) is proven
against, from the executable and from the CLI alike, and `hub_env` points a
child process at it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

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

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
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
# can land mid-file. Every request is kept on `requests`.

FAKE_REPO = "fake-org/fake-model"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
# One file larger than huggingface_hub's 10 MiB download chunk, so a cut leaves
# a whole chunk on disk to resume from, and a small one.
FAKE_FILES = {
    "config.json": b'{"model_type": "fake"}\n',
    "model.safetensors": bytes(range(256)) * (12 * 4096),  # 12 MiB, deterministic
}


class FakeHub(threading.Thread):
    def __init__(self, files: dict[str, bytes] = FAKE_FILES, repo: str = FAKE_REPO) -> None:
        super().__init__(daemon=True)
        self.files, self.repo = files, repo
        self.requests: list[tuple[str, str, str | None]] = []
        self.cut_after: int | None = None
        self.outage = False
        self.throttle: tuple[int, float] | None = None  # (bytes per write, seconds between)
        hub = self
        etags = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                return None

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
                name = self._resolve()
                if name is None:
                    self._unknown()
                    return
                self.send_response(200)
                self.send_header("ETag", f'"{etags[name]}"')
                self.send_header("X-Repo-Commit", FAKE_COMMIT)
                self.send_header("Content-Length", str(len(hub.files[name])))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

        # Threaded: huggingface_hub keeps HTTP/1.1 connections open, and a
        # single-threaded server would sit on an idle one instead of taking
        # the next.
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def run(self) -> None:
        self.server.serve_forever(poll_interval=0.05)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def gets(self, name: str) -> list[str | None]:
        """The Range header of every GET for `name`'s bytes, at any revision,
        in order (None: no Range)."""
        return [rng for method, path, rng in self.requests
                if method == "GET" and path.startswith(f"/{self.repo}/resolve/") and path.endswith(f"/{name}")]


@pytest.fixture()
def fake_hub() -> FakeHub:
    hub = FakeHub()
    hub.start()
    try:
        yield hub
    finally:
        hub.stop()


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


class FakeOllama(threading.Thread):
    def __init__(
        self, *, status: int = 200, delay: float = 0.0, replies: list[str] = (),
        library: dict[str, list[int]] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.chats: list[dict] = []
        self.pulls: list[dict] = []
        self.deletes: list[str] = []
        self.requests: list[tuple[str, str]] = []
        self.library = dict(library or {})
        self.models: dict[str, int] = {}
        self.partial: dict[tuple[str, str], int] = {}
        self.throttle: tuple[int, float] | None = None
        self.release = threading.Event()
        pending = list(replies)
        ollama = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                ollama.requests.append(("GET", self.path))
                if self.path == "/api/tags":
                    self._answer(200, {"models": ollama.tags()})
                    return
                assert self.path == "/api/version", self.path
                if delay:
                    ollama.release.wait(delay)
                self._answer(status, {"version": "0.0.0-fake"})

            def do_DELETE(self):  # noqa: N802 - http.server's name
                ollama.requests.append(("DELETE", self.path))
                assert self.path == "/api/delete", self.path
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
                if self.path == "/api/pull":
                    self._pull(body)
                    return
                assert self.path == "/api/chat", self.path
                ollama.chats.append(body)
                if not pending:
                    self._answer(404, {"error": f"model '{body.get('model')}' not found"})
                    return
                self._answer(200, {
                    "model": body["model"], "created_at": "2026-09-18T00:00:00Z",
                    "message": {"role": "assistant", "content": pending.pop(0)},
                    "done_reason": "stop", "done": True, "total_duration": 1668506709,
                    "prompt_eval_count": 26, "eval_count": 83,
                })

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

            def log_message(self, *_):
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server_port = self.server.server_port
        self.endpoint = f"http://127.0.0.1:{self.server_port}"

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

    def run(self) -> None:
        self.server.serve_forever(poll_interval=0.05)

    def stop(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
