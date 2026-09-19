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
hub library does everything but the bytes: the commit `main` points at
(resolved once, and recorded in the cache's `refs/main`, by its
`resolve_revision`) and the file list at it from its API, each file's etag,
size and URL from its metadata call, the Range request and consistency check
in its `http_get`, the per-file lock, and the pointer of each file in the
snapshot folder of that commit from its `_create_symlink`, made here from the
verified blobs alone once every one is complete (its `snapshot_download`
would ask the hub for every file's metadata again and trust that answer).
What this module adds is the one thing it lacks: the bytes go to the cache's
`<etag>.incomplete` blob, appended to across runs.

Two settings the library reads at import are set before it is imported,
unless the user set them: `HF_HUB_DISABLE_XET=1`, as readme.md § Install
requires (the Xet transfer stalls on some networks, docs/troubleshooting.md;
the bytes here go over plain HTTP regardless), and `HF_HUB_DISABLE_TELEMETRY=1`,
because runtime code sends no telemetry: with it unset the library fetches the
hub's registry of AI coding agents and names the agent it runs under, and the
torch version, in the User-Agent of every request.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import hashlib  # noqa: E402 - after the environment the hub reads at import
import ipaddress  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Iterable, Iterator  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import httpx  # noqa: E402
from filelock import Timeout  # noqa: E402
from huggingface_hub import (  # noqa: E402
    DeleteCacheStrategy,
    HfApi,
    constants,
    get_hf_file_metadata,
    hf_hub_url,
    scan_cache_dir,
    set_client_factory,
)
from huggingface_hub._local_folder import _validate_relative_filename  # noqa: E402 - the library's own check
from huggingface_hub.errors import (  # noqa: E402
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
    RevisionResolutionError,
)
from huggingface_hub.file_download import (  # noqa: E402
    REGEX_COMMIT_HASH,
    REGEX_SHA256,
    _create_symlink,
    _get_pointer_path,
    http_get,
    repo_folder_name,
)
from huggingface_hub.hf_api import RepoFile  # noqa: E402
from huggingface_hub.utils import WeakFileLock, build_hf_headers, filter_repo_objects  # noqa: E402
from huggingface_hub.utils import logging as hub_logging  # noqa: E402 - the library's own logger, where its warnings go
from huggingface_hub.utils._http import default_client_factory  # noqa: E402 - the library's own client, not a copy of it

from .backend import OllamaBackend, ollama_not_running  # noqa: E402
from .config import cache_file  # noqa: E402

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
    return cache_file(CANCEL_MARKER)


# How long `--model-status` waits for the hub's file listing: the hub
# library's own request timeout. The Settings dialog runs the status as it
# opens and waits for the exit code, so a connection the hub takes and never
# answers must end here, not hang the dialog.
STATUS_TIMEOUT: float = constants.DEFAULT_REQUEST_TIMEOUT

# The files a model's load needs: mlx-vlm's own allow patterns
# (mlx_vlm.utils.get_model_path, mlx-vlm 0.6.8), which its `load` hands the
# hub library's snapshot_download, so the snapshot it lays out holds these
# and not the rest of what the hub lists (`.gitattributes`, the model card).
# Named here, once, because mlx-vlm is not importable off Apple Silicon.
MODEL_FILE_PATTERNS = ("*.json", "*.safetensors", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jinja")

RERUN = "re-run melampus-id --download-model; it resumes where it stopped"
# The refusal when the repo's lock (REPO_LOCK) is held, said once for the
# download and the removal: whichever of the lock's three takers holds it,
# the message names all three, since the probe cannot tell them apart.
HELD = "another run holds {repo}: a download, an identification run loading it or a removal of it is running; " \
       "wait for it to finish"
NOT_A_HUB = "whatever answers there is not a Hugging Face hub; check HF_ENDPOINT"

# The one lock the download, the load and the removal of a repo all take,
# in the repo's `.locks` folder beside the hub library's per-blob locks:
# `download_model` holds it for the run, from before the hub is asked until
# the model is laid out, the model's load (`load_lock`) from its start
# until the model is in memory, and `remove_model` from its probe through
# the rename and the deletion. The per-blob locks alone left a race: a
# blob the download had not reached yet had no lock file for the removal's
# probe to find, so a download starting between the probe and the rename
# made that lock, opened its partial file and lost its bytes to the
# deletion. The name is no etag (an etag is hex), so it cannot collide
# with a blob's.
REPO_LOCK = "repo.lock"

# How long a run waits at a lock another run holds, the repo's or a blob's,
# before refusing.
LOCK_TIMEOUT: float = 5

# How long the removal's probe waits at each lock it tries, the repo's and
# every blob's: it asks whether another run holds one, it does not wait for
# that run to finish.
PROBE_TIMEOUT: float = 0.1

# The largest size, of a file or of the model, the status takes from the
# hub's listing: the largest integer a double carries exactly, so the
# plugin's JSON decoder (MelampusJson.lua, `tonumber`) reads it as the hub
# gave it; a real file is far below it (nine petabytes). Above it, or with
# a total above it, the listing is not a hub's: `json.dumps` cannot print
# an integer of more than 4300 digits (Python's int-to-str limit, which its
# JSON decoder shares, so two sizes of 4300 digits pass and their sum does
# not), and the plugin reads a shorter one as `inf` or as a rounded number.
MAX_SIZE = 2**53

# The query string of any URL in a piece of text: an LFS file's bytes come
# from the CDN at a signed URL, whose query is the signature and its expiry,
# a credential for that file. Nothing this command prints carries it.
_URL_QUERY = re.compile(r"(https?://[^\s'\"<>?]*)\?[^\s'\"<>]*")


def _without_query(text: str) -> str:
    """`text` with every URL in it cut at its `?`: the path stays, so the
    message still names the file, the signed query does not."""
    return _URL_QUERY.sub(r"\1", text)


def _redact(record: logging.LogRecord) -> bool:
    """A filter for the hub library's stderr handler: its retry warning
    names the full URL it is downloading from, signed query included."""
    record.msg, record.args = _without_query(record.getMessage()), ()
    return True


@contextmanager
def _hub_warnings_redacted() -> Iterator[None]:
    """For the command's lifetime, what the hub library's own logger prints
    (its handler on stderr) goes through `_redact`."""
    handlers = list(hub_logging.get_logger().handlers)
    for handler in handlers:
        handler.addFilter(_redact)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(_redact)


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
    """The download failed; the message names what to fix. A URL in it is
    named by its path alone: the hub library's exceptions carry the URL they
    failed at, an LFS file's being the CDN's signed one."""

    def __init__(self, message: str) -> None:
        super().__init__(_without_query(message))


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


def _incomplete(path: Path) -> Path:
    """The cache's mark for "not yet whole": `<name>.incomplete` beside the
    path, for a blob's partial file and a snapshot's staging folder alike."""
    return path.with_name(f"{path.name}.incomplete")


@dataclass
class _Blob:
    """One file of the model as the hub describes it: where its bytes are and
    where they land in the cache."""

    filename: str
    url: str
    etag: str
    size: int
    path: Path  # blobs/<etag>; the partial file is `<etag>.incomplete` beside it
    pointer: Path  # snapshots/<commit>/<filename>, pointing at the blob once complete

    @property
    def partial(self) -> Path:
        return _incomplete(self.path)

    def on_disk(self) -> int:
        """Bytes of this file already in the cache: all of it, or the partial's."""
        if self.path.exists():
            return self.size
        return self.partial.stat().st_size if self.partial.exists() else 0

    def laid_out(self) -> bool:
        """Whether the snapshot's pointer serves this blob: a symlink to it,
        or, where symlinks are unavailable and the hub library copied the
        blob instead, a copy of its size. A copy cut short is not, so a run
        that skipped it because it existed said `done` of a corrupt model."""
        if self.pointer.is_symlink():
            return self.pointer.resolve() == self.path.resolve()
        return self.pointer.is_file() and self.pointer.stat().st_size == self.size


class _Progress:
    """The bytes-done counter across files, phrased as the tqdm huggingface_hub's
    `http_get` expects so it needs no tqdm at all: it is constructed per file
    with `initial` and `total`, and `update(n)` is called per chunk written.
    Every update becomes one protocol line, and between chunks the cancel
    marker is looked for: its appearance raises DownloadCancelled exactly as
    a signal does (card #408).

    `initial` is what `http_get` keeps of the partial file, already counted
    here from disk: all of it, or none when the host answered the Range
    request with 200 and the whole file, so `http_get` truncated the partial
    and starts over. Its own rollback of that resume reaches only a bar it
    reuses across its retries, never the fresh one of this call, so the
    counter takes back here what `initial` says is gone."""

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

    def tqdm_class(self, resumed: int) -> type:
        """The counter class for one file, `resumed` bytes of it counted from disk."""
        progress = self

        class ChunkCounter:
            def __init__(self, initial: int = 0, **_ignored) -> None:
                if initial != resumed:
                    progress.advance(initial - resumed)

            def __enter__(self):
                return self

            def __exit__(self, *_exc) -> None:
                pass

            def update(self, n: int | float | None = 1) -> None:
                progress.advance(int(n or 0))

        return ChunkCounter


def _hub_client(endpoint: str) -> httpx.Client:
    """The hub library's own httpx client, with two rules on every request it
    sends, wherever in the library the request is made, one hook each.

    `bound_by_the_metadata_timeout`: every request without a timeout is
    bounded by the library's own metadata timeout, HF_HUB_ETAG_TIMEOUT. Its
    repo info and tree listing (each page) name none, and httpx reads none
    as wait forever, so a hub that accepts the connection and never answers
    held the command for good. Requests that name a timeout (the bytes, the
    metadata HEAD) keep theirs.

    `token_only_to_the_hub`: the user's token goes only to the hub's own
    origin, and only where it cannot cross the wire in cleartext. The
    library sends the token on every request it makes with the user's
    headers: the bytes of an LFS file, which the hub redirects to its CDN
    (a signed URL on another host), and each further page of the tree
    listing, at whatever URL the hub's `Link: rel="next"` names. Any
    request whose origin is not the endpoint's, another host or an
    `http://` downgrade of the hub's own, goes without it; so does every
    request to an `http://` hub that is not on loopback (`_token_may_go`)."""
    client = default_client_factory()

    def bound_by_the_metadata_timeout(request: httpx.Request) -> None:
        timeout = request.extensions.get("timeout") or {}
        request.extensions["timeout"] = {
            phase: constants.HF_HUB_ETAG_TIMEOUT if timeout.get(phase) is None else timeout[phase]
            for phase in ("connect", "read", "write", "pool")
        }

    def token_only_to_the_hub(request: httpx.Request) -> None:
        if not _token_may_go(str(request.url), endpoint):
            request.headers.pop("authorization", None)

    client.event_hooks["request"].extend([bound_by_the_metadata_timeout, token_only_to_the_hub])
    return client


def _plan(repo: str, endpoint: str | None, cache: Path, storage: Path) -> tuple[str, list[_Blob]]:
    """The commit `main` points at and every file of the repo at it, as the
    hub describes them: the tree listing for the names, the metadata call for
    each file's etag (its blob name in the cache), size and URL. `main` is
    resolved once, by the hub library's own `resolve_revision`, which also
    writes the cache's `refs/main`; everything after, the snapshot included,
    is at that commit, so a branch that moves during the run changes nothing.
    `storage` is the repo's own folder under `cache`, where its blobs and
    snapshots go."""
    api = HfApi(endpoint=endpoint)
    commit = api.resolve_revision(repo, cache_dir=cache).resolved
    # The commit names the snapshot folder, the filename the pointer under
    # it, and the etag the blob and its lock: a hub's answer becomes a path
    # only in the one shape each has, checked here before any byte is asked
    # for. The filename checks are the hub library's own (its download made
    # them): a name that is absolute, a drive or UNC path, or traverses is
    # refused, and the pointer must land under the snapshot folder.
    if not REGEX_COMMIT_HASH.match(commit):
        raise DownloadError(f"the hub at {endpoint} says main of {repo} is {commit!r}, not a commit hash; {NOT_A_HUB}")
    blobs = []
    for entry in api.list_repo_tree(repo, recursive=True, revision=commit):
        if not isinstance(entry, RepoFile):
            continue
        try:
            _validate_relative_filename(entry.path)
            pointer = _get_pointer_path(str(storage), commit, os.path.join(*entry.path.split("/")))
        except ValueError as exc:
            raise DownloadError(
                f"the hub at {endpoint} lists {entry.path!r} in {repo}, which is a path, not a file name; {NOT_A_HUB}"
            ) from exc
        url = hf_hub_url(repo, entry.path, revision=commit, endpoint=endpoint)
        meta = get_hf_file_metadata(url, endpoint=endpoint)
        if meta.etag is None or meta.size is None:
            raise DownloadError(f"the hub gave no etag or size for {entry.path}; {RERUN}")
        if not (REGEX_SHA256.match(meta.etag) or REGEX_COMMIT_HASH.match(meta.etag)):
            raise DownloadError(
                f"the hub at {endpoint} gave {entry.path} the etag {meta.etag!r}, "
                f"not a sha256 or git blob checksum; {NOT_A_HUB}"
            )
        blobs.append(_Blob(entry.path, meta.location, meta.etag, meta.size,
                           storage / "blobs" / meta.etag, Path(pointer)))
    return commit, blobs


def _verify(blob: _Blob) -> None:
    """The finished bytes must match the checksum the hub named in the etag:
    the sha256 of an LFS file (the weights), git's blob sha1 of a regular
    file (40 hex, the shape REGEX_COMMIT_HASH matches); `_plan` admits no
    other etag. Bytes that do not match never become the blob, and the partial
    is discarded so the next run fetches the file whole instead of resuming
    it forever."""
    if REGEX_SHA256.match(blob.etag):
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(b"blob %d\0" % blob.size, usedforsecurity=False)
    with blob.partial.open("rb") as done:
        for chunk in iter(lambda: done.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != blob.etag:
        blob.partial.unlink()
        raise DownloadError(
            f"{blob.filename} did not match the checksum the hub gave for it; "
            "the partial file is discarded; re-run melampus-id --download-model to fetch it whole"
        )


def _token_may_go(url: str, endpoint: str) -> bool:
    """Whether the user's token may go on a request: it goes to the hub
    itself, the whole origin, scheme and host (an `http://` URL on an
    `https://` hub's host is another origin), and that origin keeps it out
    of cleartext on the wire: `https://`, or a loopback host (127.0.0.1,
    ::1, localhost), where plain HTTP never leaves the machine. A hub
    configured as `http://` on another machine gets no token."""
    ours, theirs = urlparse(url), urlparse(endpoint)
    if (ours.scheme, ours.netloc) != (theirs.scheme, theirs.netloc):
        return False
    if theirs.scheme == "https":
        return True
    try:
        return ipaddress.ip_address(theirs.hostname or "").is_loopback
    except ValueError:
        return theirs.hostname == "localhost"


def _fetch(blob: _Blob, progress: _Progress, headers: dict[str, str], lock_dir: Path) -> None:
    """Append the rest of one file to its `.incomplete` blob and, once the
    size and the checksum check out, make it the blob. huggingface_hub's own
    `http_get` asks for the rest by Range and verifies the size; the lock is
    the one it takes for the same blob, so two runs cannot append to the same
    file.

    The OSError of its size check, or of a write, is reported by the file's
    name and the two sizes, never by its own text: `http_get` retries a
    dropped connection without the file's name, so on the retry its size
    check names the file by the last forty characters of the URL, an LFS
    file's being the tail of the CDN's signed query, which no URL rule on
    the message catches. The hub's HTTP errors, OSErrors too in the hub
    library, keep their own handling in `download_model`."""
    blob.partial.parent.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(lock_dir / f"{blob.etag}.lock", timeout=LOCK_TIMEOUT):
        with blob.partial.open("ab") as partial:
            resumed = partial.tell()
            try:
                http_get(
                    blob.url, partial,
                    resume_size=resumed, headers=headers, expected_size=blob.size,
                    displayed_filename=blob.filename, tqdm_class=progress.tqdm_class(resumed),
                )
            except httpx.HTTPError:
                raise
            except OSError as exc:
                why = exc.strerror or f"{partial.tell()} bytes arrived where the hub said {blob.size}"
                raise DownloadError(f"{blob.filename}: {why}; the partial file is kept; {RERUN}") from exc
        _verify(blob)
        blob.partial.replace(blob.path)


def _lay_out(storage: Path, commit: str, blobs: list[_Blob]) -> Path:
    """The snapshot folder of the planned commit: `snapshots/<commit>/<filename>`
    pointing at each verified blob, made by the hub library's own pointer
    helper exactly as its download lays them out, so mlx-vlm finds them
    (`refs/main`, which `resolve_revision` wrote, names the commit). Nothing
    is asked of the hub: its `snapshot_download` asked for every file's
    metadata a second time and took that answer for the blob's name, unchecked
    and, when no such blob existed, fetched it unverified.

    Each pointer is made under a staging folder and renamed into place:
    where symlinks are unavailable the helper copies the blob, byte by byte,
    and a copy cut short by a cancel or a full disk must never sit under the
    file's name for the next run to take as complete. The staging folder is
    `snapshots/<commit>.incomplete/`, beside the snapshot folder and at its
    depth, so the relative link the helper makes holds once renamed, and it
    is no path of the repo's: a staging name beside the pointer's own was
    one (`config.json.incomplete` is a valid repo filename) and, laid out
    first, was destroyed by the other file's staging. The folder is removed
    once every pointer is in place. A pointer already serving its blob is
    kept; one that does not (a short copy) is replaced."""
    snapshot = storage / "snapshots" / commit
    staging = _incomplete(snapshot)
    for blob in blobs:
        if blob.laid_out():
            continue
        staged = staging / blob.pointer.relative_to(snapshot)
        for folder in (staged.parent, blob.pointer.parent):
            folder.mkdir(parents=True, exist_ok=True)
        _create_symlink(str(blob.path), str(staged), new_blob=False)
        staged.replace(blob.pointer)
    shutil.rmtree(staging, ignore_errors=True)
    return snapshot


def _hub_at(endpoint: str | None) -> str:
    """The hub every request of this process goes to, said once for the
    download and the status: `endpoint`, or `HF_ENDPOINT`, and the protected
    client (`_hub_client`, which keeps the user's token on that hub alone)
    installed as the client the hub library makes every request with."""
    endpoint = endpoint or constants.ENDPOINT
    set_client_factory(lambda: _hub_client(endpoint))
    return endpoint


def _cache_paths(repo: str, cache_dir: Path | None) -> tuple[Path, Path, Path]:
    """Where `repo` lives in the Hugging Face cache (`HF_HOME`'s, or
    `cache_dir`), said once for the download, the status and the removal:
    the cache, the repo's storage folder in it (the hub library's own
    layout, `models--org--name`), and the `.locks` folder holding the
    per-file locks `_fetch` takes and `_repo_held` tries. The hub
    library's own check of the id (`namespace/name`, no URL, no path under
    it) runs here, before the hub is asked anything, so the three refuse an
    id that is not one the same way: a DownloadError naming the config key."""
    cache = Path(cache_dir or constants.HF_HUB_CACHE)
    try:
        folder = repo_folder_name(repo_id=repo, repo_type="model")
    except HFValidationError as exc:
        raise DownloadError(
            f"{repo} is not a model repo id (the hub's form is namespace/name, not a URL or a path): "
            f"check [model] repo in config, or --model ({exc})"
        ) from exc
    return cache, cache / folder, cache / ".locks" / folder


def _repo_lock(lock_dir: Path, timeout: float | None) -> AbstractContextManager:
    """The repo's lock (REPO_LOCK) in its `.locks` folder, made if absent,
    waited for at most `timeout` seconds (None: without bound): the one
    `download_model`, `remove_model` and the model's load (`load_lock`)
    take, said once so all take the same file."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    return WeakFileLock(lock_dir / REPO_LOCK, timeout=timeout)


def load_lock(repo: str) -> AbstractContextManager:
    """The repo's lock for the model's load (`MLXBackend._ensure_loaded`):
    mlx-vlm's `load` runs the hub library's `snapshot_download` for what
    the cache (`HF_HUB_CACHE`, the one `_cache_paths` reads) does not hold,
    under the library's per-blob locks alone, so a removal's probe could
    find nothing held for a blob the load had not reached and delete the
    load's files under it. Held for the whole load, the lock makes a
    removal in that time refuse as running, and a load starting under a
    removal wait for it, without bound as the library's own download
    waits at a blob's lock, then fetch from nothing. A `repo` that is a
    folder on disk (weights mlx-vlm's `get_model_path` loads as they are,
    fetching nothing) is in no cache: nothing to lock, and no id for the
    cache to refuse. Said here, beside the lock's other two takers, so the
    backend names one thing of the cache and none of its layout."""
    if Path(repo).exists():
        return nullcontext()
    _, _, locks = _cache_paths(repo, None)
    return _repo_lock(locks, None)


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
    removed on start, and the marker on exit, whatever the outcome; one that
    cannot be removed (a folder at its path) is a DownloadError naming it,
    on start before the hub is asked, on exit only when nothing else is in
    flight: a cancellation or a failure already raised is the outcome, and
    the next run names the marker. The repo's lock (REPO_LOCK, the one
    `remove_model` takes too) is held from before the hub is asked until
    the snapshot is laid out, so no removal of the repo starts under the
    run; a run that finds it held, by a removal, by the model's load or by
    another download of the repo, waits LOCK_TIMEOUT and is then a
    DownloadError saying so. No URL's
    query string reaches the message or the hub library's warnings: an LFS
    file's is the CDN's signature for it.
    """
    endpoint = _hub_at(endpoint)
    marker = cancel_marker or cancel_marker_path()
    _remove_marker(marker)
    completed = False
    with _hub_warnings_redacted():
        try:
            # The folders are the repo's in the cache and beside it, in `.locks`;
            # an id that is not a repo id is refused here, before the hub is asked.
            cache, storage, locks = _cache_paths(repo, cache_dir)
            # The repo's lock is held for the whole run, so a removal cannot
            # set the folder aside between the plan and a blob, or between blobs.
            with _repo_lock(locks, LOCK_TIMEOUT):
                commit, blobs = _plan(repo, endpoint, cache, storage)
                # Files with the same bytes share one etag, so one blob in the cache:
                # its bytes move once and count once, in the total and from disk.
                distinct: dict[str, _Blob] = {}
                for blob in blobs:
                    distinct.setdefault(blob.etag, blob)
                progress = _Progress(sum(b.size for b in distinct.values()), on_update, marker)
                progress.advance(sum(b.on_disk() for b in distinct.values()))
                # The user's token is for the hub: `_hub_client` keeps it off any
                # request to another origin, an LFS file's bytes from the CDN included.
                headers = build_hf_headers()
                for blob in distinct.values():
                    if not blob.path.exists():
                        _fetch(blob, progress, headers, locks)
                # Every blob is complete and verified: the snapshot of the planned
                # commit points at those blobs and nothing else.
                path = _lay_out(storage, commit, blobs)
            completed = True
        except GatedRepoError as exc:
            # A GatedRepoError is a RepositoryNotFoundError, but the repo exists:
            # what is missing is the user's access to it.
            raise DownloadError(
                f"the model repo {repo} on the hub at {endpoint} is gated: request access to it "
                f"on the hub, sign in with `hf auth login` (or set HF_TOKEN), then {RERUN} ({exc})"
            ) from exc
        except RepositoryNotFoundError as exc:
            raise DownloadError(
                f"the hub at {endpoint} has no model repo named {repo}: "
                f"check [model] repo in config, or --model ({exc})"
            ) from exc
        except (httpx.TransportError, RevisionResolutionError) as exc:
            # RevisionResolutionError: the hub could not be reached to resolve
            # `main` and the cache has no refs/main to fall back on.
            raise DownloadError(
                f"could not reach the hub at {endpoint} ({type(exc).__name__}: {exc}): "
                f"check the network, then {RERUN}"
            ) from exc
        except Timeout as exc:
            # The repo's lock, or a blob's: a download of this model (another
            # run of this command), a load of it (the hub library's own
            # download for mlx-vlm's load) or a removal of it holds it.
            raise DownloadError(f"{HELD.format(repo=repo)}, then {RERUN} ({exc})") from exc
        except (OSError, httpx.HTTPError) as exc:
            raise DownloadError(f"download of {repo} from {endpoint} failed: {exc}; {RERUN}") from exc
        finally:
            # A cancellation or failure already in flight is the outcome; a
            # marker that then cannot be removed is the next run's to name.
            # Whether one is in flight is this try's own (`completed`), not
            # sys.exc_info(), which is whatever the caller is handling.
            try:
                _remove_marker(marker)
            except DownloadError:
                if completed:
                    raise
    return path


def _remove_marker(marker: Path) -> None:
    """Remove the cancel marker, or raise DownloadError naming it: a marker
    the OS refuses to remove (a folder at its path, say) would, left in
    place, cancel the next download at its first chunk, so the start refuses
    instead and the owner is told where it is."""
    try:
        marker.unlink(missing_ok=True)
    except OSError as exc:
        raise DownloadError(
            f"the cancel marker at {marker} could not be removed ({exc}): remove it by hand, then {RERUN}"
        ) from exc


def _cached(repo: str, cache: Path):
    """The hub library's view of `repo` in its scan of the cache: the
    CachedRepoInfo when a snapshot is laid out, else None. A repo with only
    partial blobs has no snapshots folder, which the scan reports as a
    warning, not a repo. Every read of the cache here, the guard's stat of
    the cache itself included (a cache whose parent the process cannot
    search raises from `is_dir`), is bounded by the one refusal
    `_cache_unreadable` says."""
    try:
        if not cache.is_dir():
            return None
        repos = scan_cache_dir(cache).repos
    except OSError as exc:
        raise _cache_unreadable(cache, exc) from exc
    for cached in repos:
        if cached.repo_id == repo and cached.repo_type == "model":
            return cached
    return None


def _bytes_in_cache(storage: Path) -> int:
    """Every byte of the repo the cache holds: complete blobs and the
    `.incomplete` partial alike, which is what the next run starts from."""
    blobs = storage / "blobs"
    try:
        return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file()) if blobs.is_dir() else 0
    except OSError as exc:
        raise _cache_unreadable(storage.parent, exc) from exc


def _cache_unreadable(cache: Path, exc: OSError) -> DownloadError:
    """The DownloadError for a cache the process cannot read (a folder in it,
    another tool's model included, that it cannot search or list): the scan
    the status and the removal share, and the status's byte count, name the
    cache and the path the OS named, and the CLI maps it to exit 3."""
    return DownloadError(f"could not read the model cache at {cache} ({exc}): check that folder's permissions")


def model_status(repo: str, *, endpoint: str | None = None, cache_dir: Path | None = None) -> Status:
    """Whether `repo` is installed, its size on disk, and its whole size from
    the hub. The cache is read without the network; the hub is asked once
    for the file listing (the repo's info with file metadata, the one listing
    call that takes a timeout: STATUS_TIMEOUT) and, when it cannot answer,
    does not within that time, or answers with something that is not a hub's
    answer (a captive portal's page, a proxy's block page, a JSON error page
    or a listing of another shape: a file's name that is not a string, a
    size that is not a non-negative integer of at most MAX_SIZE (a string,
    a bool, a negative number, a fraction, `1e309`, a float, infinite,
    which the plugin's JSON decoder rejects, or an integer above MAX_SIZE,
    no file's and not one the plugin's decoder or `json.dumps` carries
    exactly, a total above it included), a date or an evaluation result
    the library cannot read, a repo id or a name missing, a page that is
    not JSON at all), `bytes_total` is None and the listing is as if the
    hub had not answered: the status never fails for the network. Whatever the answer
    does wrong is caught as one class, every exception the library's
    request, its constructor or the reading of the listing raises, since
    what the library's parsing can raise is the library's to enumerate, not
    this module's. It raises DownloadError for a `repo` that is
    not a repo id. The hub is asked through `_hub_client`, as the download
    asks it: the user's token goes only where that client lets it go.

    Installed means whole: the snapshot `main` names in the cache holds
    every file the hub lists that the model's load needs (those matching
    MODEL_FILE_PATTERNS, mlx-vlm's own; the hub's `.gitattributes` and the
    model card are listed but not needed, and mlx-vlm's load never fetches
    them), each at the hub's size. A download stopped while the snapshot
    was being laid out (this module's, or the hub library's own, which
    mlx-vlm's load runs) leaves `refs/main` naming a snapshot with some of
    the files, which mlx-vlm cannot load; the blobs stay counted in
    `bytes_done`, so the next download lays the rest out without fetching
    them again. When the hub cannot answer, nothing on the
    machine names the files the repo should hold, and installed is what the
    cache lays out: the snapshot `main` names, whole as far as the cache
    knows (the network being down is no reason to offer Download for a
    model that is there). `path` is the installed snapshot, else None."""
    endpoint = _hub_at(endpoint)
    cache, storage, _ = _cache_paths(repo, cache_dir)
    cached = _cached(repo, cache)
    main = next((r for r in cached.revisions if "main" in r.refs), None) if cached else None
    try:
        info = HfApi(endpoint=endpoint).model_info(repo, files_metadata=True, timeout=STATUS_TIMEOUT)
        listed: dict[str, int | None] = {f.rfilename: f.size for f in info.siblings or []}
        if not all(isinstance(name, str) for name in listed):
            raise TypeError("a file name in the listing is not a string")
        # A bool is an int in Python; 1e309 is a float, inf, which the plugin's
        # JSON decoder rejects as `Infinity`; above MAX_SIZE, a size or the
        # total is one `json.dumps` or the plugin's decoder cannot carry.
        if not all(size is None or (type(size) is int and 0 <= size <= MAX_SIZE) for size in listed.values()):
            raise TypeError("a file size in the listing is not a non-negative integer, or is above MAX_SIZE")
        total: int | None = sum(size or 0 for size in listed.values())
        if total > MAX_SIZE:
            raise TypeError("the sizes in the listing add up to more than MAX_SIZE")
        installed = main is not None and _holds(main, listed)
    except Exception:  # noqa: BLE001 - the hub's answer, whatever it did wrong, is as if it had not answered
        total, installed = None, main is not None
    path = str(main.snapshot_path) if installed else None
    return Status(repo, installed, total, _bytes_in_cache(storage), path, str(cancel_marker_path()))


def _holds(revision, listed: dict[str, int | None]) -> bool:
    """Whether the cached revision's snapshot holds every file `listed` (the
    hub's file names, at the repo's root as the hub spells them, to their
    sizes; a size the hub did not give asks the name alone) that the
    model's load needs (MODEL_FILE_PATTERNS, matched as the hub library's
    snapshot_download matches them), each at the hub's size. The scan's
    `size_on_disk` is the blob's size, a pointer's target or a copy's own,
    so a copy cut short is not the file."""
    on_disk = {f.file_path.relative_to(revision.snapshot_path).as_posix(): f.size_on_disk for f in revision.files}
    needed = filter_repo_objects(listed, allow_patterns=list(MODEL_FILE_PATTERNS))
    return all(name in on_disk and (listed[name] is None or on_disk[name] == listed[name]) for name in needed)


def _repo_held(lock_dir: Path, held: ExitStack) -> bool:
    """Whether another run holds the repo: the repo's lock (`download_model`
    for its run, the model's load (`load_lock`) from its start until the
    model is in memory, whose `snapshot_download` fetches what the cache
    lacks, or another removal from its probe through the deletion) or the
    per-file lock on the blob a download is appending to (`_fetch`), under
    the cache's `.locks`. The probe cannot tell the holders apart, so the
    refusal (HELD) names all three.
    Each lock this takes it keeps, on `held`, until the caller leaves that
    stack: released at once, a download starting after the probe took its
    lock and appended to a blob the removal then set aside and deleted.
    The repo's lock is taken first, whether or not its file exists, and
    then every blob's that does: a blob a download has not reached yet has
    no lock file, so the blob locks alone let a download start between the
    probe and the rename. Held through the rename and the deletion, a
    download starting in between refuses (`download_model`'s own timeout
    on the repo's lock, the one it takes first) or waits and starts from
    nothing once the removal is done."""
    locks = [_repo_lock(lock_dir, PROBE_TIMEOUT)]
    locks += [WeakFileLock(lock, timeout=PROBE_TIMEOUT) for lock in lock_dir.glob("*.lock") if lock.name != REPO_LOCK]
    for lock in locks:
        try:
            held.enter_context(lock)
        except Timeout:
            return True
    return False


def remove_model(repo: str, *, cache_dir: Path | None = None) -> Path:
    """Delete `repo` from the cache, all or nothing from the cache's point of
    view, and return the folder it was in. The repo's folder is first set
    aside within the cache as `<folder>.incomplete` (`_incomplete`, the
    module's mark for "not whole"), which the scan no longer lists as the
    repo, then deleted through the hub library's own deletion strategy, for
    this repo alone: every revision of it goes, so the whole folder does.
    The library's `delete_revisions` is not used: it searches the whole
    cache by commit hash and takes the first repo found at one, which can be
    a fork cached at the same commit, leaving `repo` installed.

    Raises DownloadError when nothing is installed, the repo's folder is a
    symbolic link (a model laid out on another disk and linked into the
    cache, which the scan accepts: nothing is deleted through a link, and
    rmtree would refuse it with the one OSError the library's deletion does
    not catch; the model is removed where the link points), another run
    holds it (a download, an identification run loading it or another
    removal; the message names all three, since the probe cannot tell them
    apart), the set-aside name is taken by the folder an earlier
    refused removal left (the message named it then for the owner to
    delete by hand, and names it again: the rename onto it would fail with
    the OS's errno line, on every removal after, and say neither why nor
    what to do; the model is untouched), the folder cannot be set aside
    (Windows refuses while another program holds a file in it open; the
    model is then untouched),
    or the set-aside folder is still there after the deletion: the library's
    deletion is one rmtree, which deletes what it can and stops at the first
    entry it cannot, and the library catches its PermissionError, logs it
    and returns, so the folder is the one signal it leaves. Deleted under
    the repo's own name, that left the model half there, listed by nothing
    and loadable by nothing; set aside, the model is gone from the cache
    (the status reads absent, a second removal has nothing to remove), and
    the message names the folder for the owner to delete by hand. Any other
    OSError of the scan (a folder in the shared cache the process cannot
    search, another tool's model included), the lock probe or the deletion
    (a lock file that cannot be opened) is a DownloadError naming it too, as
    `download_model` bounds its own: the CLI maps DownloadError to exit 3
    and lets nothing else out."""
    cache, _, locks = _cache_paths(repo, cache_dir)
    cached = _cached(repo, cache)
    if cached is None:
        raise DownloadError(f"{repo} is not in the cache at {cache}: nothing to remove")
    if cached.repo_path.is_symlink():
        raise DownloadError(f"could not remove {repo} from {cache}: {cached.repo_path} is a link, not a folder; "
                            f"remove the model where the link points ({cached.repo_path.resolve()})")
    aside = _incomplete(cached.repo_path)
    if aside.exists():
        raise DownloadError(f"could not remove {repo} from {cache}: {aside} is still there, left by an earlier "
                            f"removal that was refused; delete that folder by hand, then remove again; "
                            "the model is untouched")
    # The locks the probe takes are held until the deletion is done: no
    # download or load can start on this model between the probe and the end.
    with ExitStack() as held:
        try:
            if _repo_held(locks, held):
                raise DownloadError(f"{HELD.format(repo=repo)} (cancel a download first), then remove")
            cached.repo_path.rename(aside)
        except OSError as exc:
            raise DownloadError(f"could not remove {repo} from {cache}: {exc}; the model is untouched") from exc
        left = (f"could not remove {repo} from {cache} whole: the cache no longer lists it, and what could not be "
                f"deleted is set aside at {aside}; check that folder's permissions (on Windows, that no other "
                "program holds a file in it open) and delete it by hand")
        try:
            DeleteCacheStrategy(expected_freed_size=cached.size_on_disk, blobs=frozenset(), refs=frozenset(),
                                repos=frozenset({aside}), snapshots=frozenset()).execute()
        except OSError as exc:
            raise DownloadError(f"{left} ({exc})") from exc
        if aside.exists():
            raise DownloadError(left)
    return cached.repo_path


# --- the same button for Ollama, through its pull (card #409) -------------
#
# Ollama holds its own models; the plugin's Download button asks it to pull
# one, and the stream it answers with (docs/api.md § Pull a Model) is mapped
# onto the protocol above, line for line, so nothing downstream knows which
# engine is fetching.


OLLAMA_PULL = "/api/pull"


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
