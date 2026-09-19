"""Fetch the MLX model into the Hugging Face cache, with progress the plugin can parse (card #407).

The progress protocol is the plugin's contract (card #408): one line per update
on stdout, defined once here in `Update` and documented in docs/config.md
§ Downloading the model. Errors go to stderr, never into the protocol.

The same protocol carries an Ollama pull (card #409): `pull_updates` maps the
stream of Ollama's pull endpoint onto it, so the plugin's parser, poller and
row serve both engines with no second protocol.

Why the bytes are fetched here and not by `snapshot_download`: huggingface_hub
1.26 downloads each file to a process-unique temporary file, from byte zero,
and deletes it when the run fails, so nothing survives an interrupted run for
the next one to resume (its Range resume exists only inside one process's
retries). Done-when 1 and 2 need the partial file kept and resumed. So the
hub library does everything but the bytes: the commit and file list from its
API, each file's etag, size and URL from its metadata call, the Range request
and consistency check in its `http_get`, the per-file lock, and the cache
layout, pointers and `refs/main` from its `snapshot_download` once every
blob is complete. What this module adds is the one thing it lacks: the
bytes go to the cache's `<etag>.incomplete` blob, appended to across runs.

`HF_HUB_DISABLE_XET=1` is set before the library is imported, as readme.md
§ Install requires (the Xet transfer stalls on some networks,
docs/troubleshooting.md); the bytes here go over plain HTTP regardless.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json  # noqa: E402 - after the environment the hub reads at import
import signal  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Iterable, Iterator  # noqa: E402

import httpx  # noqa: E402
from filelock import Timeout  # noqa: E402
from huggingface_hub import (  # noqa: E402
    HfApi,
    constants,
    get_hf_file_metadata,
    hf_hub_url,
    scan_cache_dir,
    snapshot_download,
)
from huggingface_hub.errors import RepositoryNotFoundError  # noqa: E402
from huggingface_hub.file_download import http_get, repo_folder_name  # noqa: E402
from huggingface_hub.hf_api import RepoFile  # noqa: E402
from huggingface_hub.utils import WeakFileLock, build_hf_headers  # noqa: E402

from .backend import OllamaBackend, ollama_not_running  # noqa: E402
from .config import _cache  # noqa: E402

PROGRESS = "progress"
DONE = "done"
CANCELLED = "cancelled"

# Exit codes of `melampus-id --download-model`: 0 complete, 3 failed (the
# message on stderr names the fix), and this one when a signal cancelled it.
EXIT_CANCELLED = 4

# The file whose appearance cancels a running download (card #408): the
# Lightroom plugin cannot signal the executable, so it writes this instead.
# Under the per-user data directory beside the caches; `--model-status`
# carries the path so the plugin never derives the directory itself.
CANCEL_MARKER = "download-cancel"


def cancel_marker_path() -> Path:
    return _cache(CANCEL_MARKER)


RERUN = "re-run melampus-id --download-model; it resumes where it stopped"


@dataclass(frozen=True)
class Update:
    """One stdout line of the download protocol.

    `progress <bytes_done> <bytes_total>` while bytes arrive, `done <path>` once
    the model is complete (the path is the rest of the line: it may hold
    spaces), `cancelled` when a signal stopped the download.
    """

    state: str
    bytes_done: int = 0
    bytes_total: int = 0
    path: str = ""

    @classmethod
    def progress(cls, bytes_done: int, bytes_total: int) -> Update:
        return cls(PROGRESS, bytes_done, bytes_total)

    @classmethod
    def done(cls, path: str) -> Update:
        return cls(DONE, path=path)

    @classmethod
    def cancelled(cls) -> Update:
        return cls(CANCELLED)

    def line(self) -> str:
        if self.state == PROGRESS:
            return f"{PROGRESS} {self.bytes_done} {self.bytes_total}"
        if self.state == DONE:
            return f"{DONE} {self.path}"
        return CANCELLED

    @classmethod
    def parse(cls, line: str) -> Update:
        """The plugin's side of the protocol. Raises ValueError for any other line."""
        word, _, rest = line.rstrip("\r\n").partition(" ")
        if word == PROGRESS:
            fields = rest.split(" ")
            if len(fields) == 2 and all(f.isdigit() for f in fields):
                return cls.progress(int(fields[0]), int(fields[1]))
        elif word == DONE and rest:
            return cls.done(rest)
        elif word == CANCELLED and not rest:
            return cls.cancelled()
        raise ValueError(f"not a download update: {line!r}")


@dataclass(frozen=True)
class Status:
    """What `--model-status` prints, one JSON object (card #408): the repo,
    whether its snapshot is in the cache and where, the bytes the cache holds
    (complete files and partials alike), the whole model's size from the hub
    or None when the hub cannot be reached, and where to write to cancel."""

    repo: str
    installed: bool
    bytes_total: int | None
    bytes_done: int
    path: str | None
    cancel_path: str

    def json(self) -> str:
        return json.dumps(asdict(self))


class DownloadError(Exception):
    """The download failed; the message names what to fix."""


class DownloadCancelled(Exception):
    """A signal, or the cancel marker, asked the download to stop; partial
    files are kept for resume."""


@contextmanager
def cancel_on_signals() -> Iterator[None]:
    """For the entry point that owns the process: SIGINT (Ctrl+C), SIGTERM and,
    on Windows, Ctrl+Break raise DownloadCancelled wherever the download is,
    including inside a blocking read, so it stops within the current chunk.
    The previous handlers come back on exit."""

    def raise_cancelled(signum: int, frame) -> None:
        raise DownloadCancelled(signal.Signals(signum).name)

    previous = {}
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            previous[number] = signal.signal(number, raise_cancelled)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@dataclass
class _Blob:
    """One file of the model as the hub describes it: where its bytes are and
    where they land in the cache."""

    filename: str
    url: str
    etag: str
    size: int
    path: Path  # blobs/<etag>; the partial file is `<etag>.incomplete` beside it

    @property
    def partial(self) -> Path:
        return self.path.with_name(f"{self.path.name}.incomplete")

    def on_disk(self) -> int:
        """Bytes of this file already in the cache: all of it, or the partial's."""
        if self.path.exists():
            return self.size
        return self.partial.stat().st_size if self.partial.exists() else 0


class _Progress:
    """The bytes-done counter across files, phrased as the tqdm huggingface_hub's
    `http_get` expects so it needs no tqdm at all: it is constructed per file
    with `initial` and `total` (already accounted for here), `update(n)` is
    called per chunk written, and a negative `update` takes back a resume the
    server ignored. Every update becomes one protocol line, and between
    chunks the cancel marker is looked for: its appearance raises
    DownloadCancelled exactly as a signal does (card #408)."""

    def __init__(self, total: int, on_update: Callable[[Update], None], cancel_marker: Path) -> None:
        self.done, self.total, self.on_update, self.cancel_marker = 0, total, on_update, cancel_marker

    def advance(self, n: int) -> None:
        # Looked for before the chunk is counted: `http_get` calls update
        # before it writes the chunk, so raising here loses only the chunk in
        # hand, and what was reported done is what is on disk.
        if self.cancel_marker.exists():
            raise DownloadCancelled(CANCEL_MARKER)
        self.done += n
        self.on_update(Update.progress(self.done, self.total))

    def tqdm_class(self) -> type:
        progress = self

        class ChunkCounter:
            def __init__(self, **_ignored) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc) -> None:
                pass

            def update(self, n: int | float | None = 1) -> None:
                progress.advance(int(n or 0))

        return ChunkCounter


def _files(api: HfApi, repo: str, revision: str | None = None) -> list[RepoFile]:
    """Every file of the repo at `revision` (the hub's default when None), from
    the tree listing: names and sizes."""
    return [entry for entry in api.list_repo_tree(repo, recursive=True, revision=revision)
            if isinstance(entry, RepoFile)]


def _plan(repo: str, endpoint: str | None, storage: Path) -> tuple[str, list[_Blob]]:
    """The commit `main` points at and every file of the repo at it, as the
    hub describes them: the tree listing for the names, the metadata call for
    each file's etag (its blob name in the cache), size and URL."""
    api = HfApi(endpoint=endpoint)
    commit = api.repo_info(repo).sha
    blobs = []
    for entry in _files(api, repo, commit):
        url = hf_hub_url(repo, entry.path, revision=commit, endpoint=endpoint)
        meta = get_hf_file_metadata(url, endpoint=endpoint)
        if meta.etag is None or meta.size is None:
            raise DownloadError(f"the hub gave no etag or size for {entry.path}; {RERUN}")
        blobs.append(_Blob(entry.path, meta.location, meta.etag, meta.size, storage / "blobs" / meta.etag))
    return commit, blobs


def _fetch(blob: _Blob, progress: _Progress, headers: dict[str, str], lock_dir: Path) -> None:
    """Append the rest of one file to its `.incomplete` blob and, once the
    size checks out, make it the blob. huggingface_hub's own `http_get` asks
    for the rest by Range and verifies the size; the lock is the one it takes
    for the same blob, so two runs cannot append to the same file."""
    blob.partial.parent.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(lock_dir / f"{blob.etag}.lock", timeout=5):
        with blob.partial.open("ab") as partial:
            http_get(
                blob.url, partial,
                resume_size=partial.tell(), headers=headers, expected_size=blob.size,
                displayed_filename=blob.filename, tqdm_class=progress.tqdm_class(),
            )
        blob.partial.replace(blob.path)


def download_model(
    repo: str,
    *,
    on_update: Callable[[Update], None],
    endpoint: str | None = None,
    cache_dir: Path | None = None,
    cancel_marker: Path | None = None,
) -> Path:
    """Fetch every file of `repo` into the Hugging Face cache (`HF_HOME`, or
    `cache_dir`) from the hub at `HF_ENDPOINT` (or `endpoint`), resuming any
    partial file left by an earlier run, and return the snapshot folder.

    `on_update` gets one Update per chunk received, the first before any byte
    moves so the total is known at once. Raises DownloadError with the fix in
    the message; a DownloadCancelled raised from `cancel_on_signals`, or here
    when `cancel_marker` (the documented path by default) appears between
    chunks, passes through with the partial file kept. A stale marker is
    removed on start, and the marker on exit, whatever the outcome.
    """
    endpoint = endpoint or constants.ENDPOINT
    cache = Path(cache_dir or constants.HF_HUB_CACHE)
    folder = repo_folder_name(repo_id=repo, repo_type="model")
    marker = cancel_marker or cancel_marker_path()
    marker.unlink(missing_ok=True)
    try:
        commit, blobs = _plan(repo, endpoint, cache / folder)
        progress = _Progress(sum(b.size for b in blobs), on_update, marker)
        progress.advance(sum(b.on_disk() for b in blobs))
        headers = build_hf_headers()
        for blob in blobs:
            if not blob.path.exists():
                _fetch(blob, progress, headers, cache / ".locks" / folder)
        # Every blob is complete: the hub library lays out the snapshot, the
        # pointers and refs/main exactly as mlx-vlm will look for them, and
        # moves no bytes because every file is already in the cache.
        return Path(snapshot_download(repo, cache_dir=cache, endpoint=endpoint))
    except RepositoryNotFoundError as exc:
        raise DownloadError(
            f"the hub at {endpoint} has no model repo named {repo}: "
            f"check [model] repo in config, or --model ({exc})"
        ) from exc
    except httpx.TransportError as exc:
        raise DownloadError(
            f"could not reach the hub at {endpoint} ({type(exc).__name__}: {exc}): "
            f"check the network, then {RERUN}"
        ) from exc
    except (OSError, httpx.HTTPError) as exc:
        raise DownloadError(f"download of {repo} from {endpoint} failed: {exc}; {RERUN}") from exc
    finally:
        marker.unlink(missing_ok=True)


def _cached(repo: str, cache: Path):
    """The hub library's scan of the cache and its view of `repo` in it: the
    CachedRepoInfo when a snapshot is laid out, else None. A repo with only
    partial blobs has no snapshots folder, which the scan reports as a
    warning, not a repo."""
    if not cache.is_dir():
        return None, None
    info = scan_cache_dir(cache)
    for cached in info.repos:
        if cached.repo_id == repo and cached.repo_type == "model":
            return info, cached
    return info, None


def _bytes_in_cache(storage: Path) -> int:
    """Every byte of the repo the cache holds: complete blobs and the
    `.incomplete` partial alike, which is what the next run starts from."""
    blobs = storage / "blobs"
    return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file()) if blobs.is_dir() else 0


def model_status(repo: str, *, endpoint: str | None = None, cache_dir: Path | None = None) -> Status:
    """Whether `repo` is in the cache, its size on disk, and its whole size from
    the hub. The cache is read without the network; the hub is asked once
    for the file listing and, when it cannot answer, `bytes_total` is None:
    the status never fails for the network being down."""
    endpoint = endpoint or constants.ENDPOINT
    cache = Path(cache_dir or constants.HF_HUB_CACHE)
    storage = cache / repo_folder_name(repo_id=repo, repo_type="model")
    _, cached = _cached(repo, cache)
    main = next((r for r in cached.revisions if "main" in r.refs), None) if cached else None
    installed, path = main is not None, str(main.snapshot_path) if main else None
    try:
        total: int | None = sum(f.size or 0 for f in _files(HfApi(endpoint=endpoint), repo))
    except (RepositoryNotFoundError, httpx.HTTPError, OSError):
        total = None
    return Status(repo, installed, total, _bytes_in_cache(storage), path, str(cancel_marker_path()))


def _download_running(lock_dir: Path) -> bool:
    """Whether another process is appending to one of the repo's blobs: it
    holds the per-file lock `_fetch` takes, under the cache's `.locks`."""
    for lock in lock_dir.glob("*.lock") if lock_dir.is_dir() else ():
        try:
            with WeakFileLock(lock, timeout=0.1):
                pass
        except Timeout:
            return True
    return False


def remove_model(repo: str, *, cache_dir: Path | None = None) -> Path:
    """Delete `repo` from the cache through the hub library's own deletion
    (every revision, so the whole repo folder goes) and return that folder.
    Raises DownloadError when nothing is installed or a download of it is
    running."""
    cache = Path(cache_dir or constants.HF_HUB_CACHE)
    folder = repo_folder_name(repo_id=repo, repo_type="model")
    info, cached = _cached(repo, cache)
    if cached is None:
        raise DownloadError(f"{repo} is not in the cache at {cache}: nothing to remove")
    if _download_running(cache / ".locks" / folder):
        raise DownloadError(f"a download of {repo} is running; cancel it first, then remove")
    info.delete_revisions(*(r.commit_hash for r in cached.revisions)).execute()
    return cached.repo_path


# --- the same button for Ollama, through its pull (card #409) -------------
#
# Ollama holds its own models; the plugin's Download button asks it to pull
# one, and the stream it answers with (docs/api.md § Pull a Model) is mapped
# onto the protocol above, line for line, so nothing downstream knows which
# engine is fetching.


OLLAMA_PULL = "/api/pull"
OLLAMA_TAGS = "/api/tags"


def _pull_error(model: str, error: object) -> DownloadError:
    """Ollama's error, in a message naming the fix: the model and
    `[model] ollama_model` when the library has no such model (`pull model
    manifest: file does not exist`, its 404 as os.ErrNotExist in
    server/images.go), else the re-run hint, since Ollama keeps the layers it
    has and resumes them."""
    if "file does not exist" in str(error):
        return DownloadError(
            f"Ollama has no model named {model} ({error}): check [model] ollama_model "
            "is a tag from ollama.com/library"
        )
    return DownloadError(f"Ollama could not pull {model}: {error}; {RERUN}")


def pull_updates(model: str, lines: Iterable[bytes | str]) -> Iterator[Update]:
    """Ollama's pull stream as protocol updates. `lines` are the response's
    lines, one JSON object each (docs/api.md § Streaming responses; the
    server delimits them with newlines).

    The stream (§ Pull a Model): `{"status": "pulling manifest"}`, then one
    object per layer as it downloads, `{"status": "pulling <digest>",
    "digest", "total", "completed"}`, where `completed` may be missing until
    any of the layer is done and the layers come one after another (a layer
    Ollama already holds is reported once, complete); then the verifying,
    writing-manifest and removing-unused-layers statuses; then `{"status":
    "success"}`. Each layer line becomes `progress <sum of completed> <sum
    of total>` over every layer seen so far; `success` becomes `done
    <model>`; the other statuses print nothing.

    An error is an object with `error` (server/routes.go streamResponse): it
    raises DownloadError with Ollama's words, naming the model and
    `[model] ollama_model` when the library has no such model (`pull model
    manifest: file does not exist`, its 404), else the re-run hint, since
    Ollama keeps the layers it has and resumes them. A stream that ends
    before `success` is a failure too.
    """
    layers: dict[str, tuple[int, int]] = {}
    for raw in lines:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not text.strip():
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DownloadError(f"Ollama's pull reply was not JSON: {text[:120]!r}") from exc
        if not isinstance(item, dict):
            raise DownloadError(f"Ollama's pull reply was not JSON: {text[:120]!r}")
        error = item.get("error")
        if error:
            raise _pull_error(model, error)
        status = str(item.get("status") or "")
        if status == "success":
            yield Update.done(model)
            return
        if status.startswith("pulling ") and item.get("digest") and "total" in item:
            layers[str(item["digest"])] = (int(item["total"]), int(item.get("completed") or 0))
            yield Update.progress(sum(c for _, c in layers.values()), sum(t for t, _ in layers.values()))
    raise DownloadError(f"Ollama's pull of {model} ended before it reported success; {RERUN}")


def _lines_until_cancelled(response, marker: Path) -> Iterator[bytes]:
    """The response's lines, looking for the cancel marker before each one
    is handed on: its appearance raises DownloadCancelled exactly as a
    signal does, and leaving the `with` around the response closes the
    stream, which is how Ollama learns to stop (the request's context is
    cancelled; it keeps the layers it has)."""
    for line in response:
        if marker.exists():
            raise DownloadCancelled(CANCEL_MARKER)
        yield line


def pull_model(
    model: str,
    url: str,
    *,
    on_update: Callable[[Update], None],
    cancel_marker: Path | None = None,
    timeout: float = 180.0,
    opener: Callable | None = None,
) -> str:
    """Ask the Ollama server at `url` to pull `model` (its docs/api.md § Pull
    a Model: POST /api/pull with the model's name, the stream of JSON
    objects mapped by `pull_updates`), handing `on_update` each protocol
    update, and return the model's name: what `done` prints, and what
    `--model-status` reports as the path, since the model lives in Ollama
    under that name. The stream is read with `timeout` per line.

    Raises DownloadError with the fix in the message: the backend's own
    not-running words when nothing answers at `url`, Ollama's words for a
    refusal before the stream starts (an HTTP status with its {"error"}
    object) or an error line within it. DownloadCancelled from a signal, or
    from `cancel_marker` (the documented path by default) appearing between
    lines, passes through with the stream closed; a stale marker is removed
    on start and the marker on exit, as the MLX download does.
    """
    marker = cancel_marker or cancel_marker_path()
    marker.unlink(missing_ok=True)
    request = urllib.request.Request(
        f"{url.rstrip('/')}{OLLAMA_PULL}",
        data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            for update in pull_updates(model, _lines_until_cancelled(response, marker)):
                on_update(update)
    except urllib.error.HTTPError as exc:
        raise _pull_error(model, OllamaBackend._error_text(exc)) from exc
    except urllib.error.URLError as exc:
        raise DownloadError(ollama_not_running(url, exc.reason)) from exc
    except (OSError, TimeoutError) as exc:
        raise DownloadError(f"the pull of {model} from {url} failed: {exc}; {RERUN}") from exc
    finally:
        marker.unlink(missing_ok=True)
    return model


def _ollama_request(url: str, path: str, body: dict | None = None, *, method: str = "POST",
                    timeout: float = 10.0) -> dict:
    """One JSON answer from the Ollama server at `url`: `body` sent as JSON
    when given, the reply decoded. Raises DownloadError with the backend's
    not-running words when nothing answers, Ollama's words for an HTTP
    error."""
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"Ollama answered {exc.code}: {OllamaBackend._error_text(exc)}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(ollama_not_running(url, exc.reason)) from exc
    except (OSError, TimeoutError) as exc:
        raise DownloadError(ollama_not_running(url, exc)) from exc
    try:
        return json.loads(raw or b"{}")
    except json.JSONDecodeError as exc:
        raise DownloadError(f"Ollama's reply from {url}{path} was not JSON: {raw[:120]!r}") from exc


def _held(model: str, url: str) -> dict | None:
    """The list entry for `model` in the Ollama at `url` (docs/api.md § List
    Local Models: GET /api/tags, `models` each with `name` and `size`), or
    None when it is not held. A name without a tag is `<name>:latest`
    there (§ Model names: the tag defaults to `latest`)."""
    names = {model, model if ":" in model else f"{model}:latest"}
    for entry in _ollama_request(url, OLLAMA_TAGS, method="GET").get("models") or []:
        if isinstance(entry, dict) and (entry.get("name") in names or entry.get("model") in names):
            return entry
    return None


def ollama_status(model: str, url: str) -> Status:
    """Whether the Ollama at `url` holds `model`, from its list endpoint:
    installed with the size it reports for both totals and the model's
    name as the path (where it lives: in Ollama, under that name, what
    `done` printed); else absent with `bytes_total` None, since the docs
    give sizes for held models only. Never fails: with no server answering
    the model is reported absent, size unknown, so Settings opens."""
    try:
        entry = _held(model, url)
    except DownloadError:
        entry = None
    if entry is None:
        return Status(model, False, None, 0, None, str(cancel_marker_path()))
    size = int(entry.get("size") or 0)
    return Status(model, True, size, size, str(entry.get("name") or model), str(cancel_marker_path()))
