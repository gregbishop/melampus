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
"""

from __future__ import annotations

import contextlib
import importlib.util
import platform
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
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


@contextlib.contextmanager
def loopback_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[HTTPServer]:
    """An HTTP server on 127.0.0.1 at an ephemeral port, serving `handler` on
    a daemon thread until exit, then stopped and joined. The caller points the
    client under test at `server.server_port`; nothing leaves the machine."""
    server = HTTPServer(("127.0.0.1", 0), handler)
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
