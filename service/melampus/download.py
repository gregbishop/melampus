"""Fetch the MLX model into the Hugging Face cache, with progress the plugin can parse (card #407).

The progress protocol is the plugin's contract (card #408): one line per update
on stdout, defined once here in `Update` and documented in docs/config.md
§ Downloading the model. Errors go to stderr, never into the protocol.

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

`HF_HUB_DISABLE_XET=1` is set before the library is imported, as readme.md
§ Install requires (the Xet transfer stalls on some networks,
docs/troubleshooting.md); the bytes here go over plain HTTP regardless.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import hashlib  # noqa: E402 - after the environment the hub reads at import
import signal  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Iterator  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import httpx  # noqa: E402
from huggingface_hub import (  # noqa: E402
    HfApi,
    constants,
    get_hf_file_metadata,
    hf_hub_url,
    set_client_factory,
)
from huggingface_hub._local_folder import _validate_relative_filename  # noqa: E402 - the library's own check
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionResolutionError  # noqa: E402
from huggingface_hub.file_download import (  # noqa: E402
    REGEX_COMMIT_HASH,
    REGEX_SHA256,
    _create_symlink,
    _get_pointer_path,
    http_get,
    repo_folder_name,
)
from huggingface_hub.hf_api import RepoFile  # noqa: E402
from huggingface_hub.utils import WeakFileLock, build_hf_headers  # noqa: E402
from huggingface_hub.utils._http import default_client_factory  # noqa: E402 - the library's own client, not a copy of it

PROGRESS = "progress"
DONE = "done"
CANCELLED = "cancelled"

# Exit codes of `melampus-id --download-model`: 0 complete, 3 failed (the
# message on stderr names the fix), and this one when a signal cancelled it.
EXIT_CANCELLED = 4

RERUN = "re-run melampus-id --download-model; it resumes where it stopped"
NOT_A_HUB = "whatever answers there is not a Hugging Face hub; check HF_ENDPOINT"


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


class DownloadError(Exception):
    """The download failed; the message names what to fix."""


class DownloadCancelled(Exception):
    """A signal asked the download to stop; partial files are kept for resume."""


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
    pointer: Path  # snapshots/<commit>/<filename>, pointing at the blob once complete

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
    with `initial` and `total`, and `update(n)` is called per chunk written.
    Every update becomes one protocol line.

    `initial` is what `http_get` keeps of the partial file, already counted
    here from disk: all of it, or none when the host answered the Range
    request with 200 and the whole file, so `http_get` truncated the partial
    and starts over. Its own rollback of that resume reaches only a bar it
    reuses across its retries, never the fresh one of this call, so the
    counter takes back here what `initial` says is gone."""

    def __init__(self, total: int, on_update: Callable[[Update], None]) -> None:
        self.done, self.total, self.on_update = 0, total, on_update

    def advance(self, n: int) -> None:
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


def _bounded_client() -> httpx.Client:
    """The hub library's own httpx client, with every request it sends
    without a timeout bounded by its own metadata timeout,
    HF_HUB_ETAG_TIMEOUT. Its repo info and tree listing (each page) name
    none, and httpx reads none as wait forever, so a hub that accepts the
    connection and never answers held the command for good. Requests that
    name a timeout (the bytes, the metadata HEAD) keep theirs."""
    client = default_client_factory()

    def bound(request: httpx.Request) -> None:
        timeout = request.extensions.get("timeout") or {}
        request.extensions["timeout"] = {
            phase: constants.HF_HUB_ETAG_TIMEOUT if timeout.get(phase) is None else timeout[phase]
            for phase in ("connect", "read", "write", "pool")
        }

    client.event_hooks["request"].append(bound)
    return client


def _plan(repo: str, endpoint: str | None, cache: Path) -> tuple[str, list[_Blob]]:
    """The commit `main` points at and every file of the repo at it, as the
    hub describes them: the tree listing for the names, the metadata call for
    each file's etag (its blob name in the cache), size and URL. `main` is
    resolved once, by the hub library's own `resolve_revision`, which also
    writes the cache's `refs/main`; everything after, the snapshot included,
    is at that commit, so a branch that moves during the run changes nothing."""
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
    storage = cache / repo_folder_name(repo_id=repo, repo_type="model")
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


def _same_origin(url: str, endpoint: str) -> bool:
    """Whether a file's bytes are served by the hub itself, so the user's
    token may go with the request: the whole origin, scheme and host, must
    match. An `http://` URL on an `https://` hub's host is another origin,
    or the token would go in cleartext."""
    ours, theirs = urlparse(url), urlparse(endpoint)
    return (ours.scheme, ours.netloc) == (theirs.scheme, theirs.netloc)


def _fetch(blob: _Blob, progress: _Progress, headers: dict[str, str], lock_dir: Path) -> None:
    """Append the rest of one file to its `.incomplete` blob and, once the
    size and the checksum check out, make it the blob. huggingface_hub's own
    `http_get` asks for the rest by Range and verifies the size; the lock is
    the one it takes for the same blob, so two runs cannot append to the same
    file."""
    blob.partial.parent.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(lock_dir / f"{blob.etag}.lock", timeout=5):
        with blob.partial.open("ab") as partial:
            resumed = partial.tell()
            http_get(
                blob.url, partial,
                resume_size=resumed, headers=headers, expected_size=blob.size,
                displayed_filename=blob.filename, tqdm_class=progress.tqdm_class(resumed),
            )
        _verify(blob)
        blob.partial.replace(blob.path)


def _lay_out(storage: Path, commit: str, blobs: list[_Blob]) -> Path:
    """The snapshot folder of the planned commit: `snapshots/<commit>/<filename>`
    pointing at each verified blob, made by the hub library's own pointer
    helper exactly as its download lays them out, so mlx-vlm finds them
    (`refs/main`, which `resolve_revision` wrote, names the commit). Nothing
    is asked of the hub: its `snapshot_download` asked for every file's
    metadata a second time and took that answer for the blob's name, unchecked
    and, when no such blob existed, fetched it unverified."""
    for blob in blobs:
        blob.pointer.parent.mkdir(parents=True, exist_ok=True)
        if not blob.pointer.exists():
            _create_symlink(str(blob.path), str(blob.pointer), new_blob=False)
    return storage / "snapshots" / commit


def download_model(
    repo: str,
    *,
    on_update: Callable[[Update], None],
    endpoint: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Fetch every file of `repo` into the Hugging Face cache (`HF_HOME`, or
    `cache_dir`) from the hub at `HF_ENDPOINT` (or `endpoint`), resuming any
    partial file left by an earlier run, and return the snapshot folder.

    `on_update` gets one Update per chunk received, the first before any byte
    moves so the total is known at once. Raises DownloadError with the fix in
    the message; a DownloadCancelled raised from `cancel_on_signals` passes
    through with the partial file kept.
    """
    endpoint = endpoint or constants.ENDPOINT
    cache = Path(cache_dir or constants.HF_HUB_CACHE)
    folder = repo_folder_name(repo_id=repo, repo_type="model")
    set_client_factory(_bounded_client)
    try:
        commit, blobs = _plan(repo, endpoint, cache)
        progress = _Progress(sum(b.size for b in blobs), on_update)
        progress.advance(sum(b.on_disk() for b in blobs))
        headers = build_hf_headers()
        # The user's token is for the hub. An LFS file's bytes come from the
        # hub's CDN (a signed URL on another host): no token goes there, as
        # huggingface_hub's own download strips it when the host differs,
        # nor to a plain-HTTP URL on the hub's own host.
        no_token = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        for blob in blobs:
            if not blob.path.exists():
                _fetch(blob, progress, headers if _same_origin(blob.url, endpoint) else no_token,
                       cache / ".locks" / folder)
        # Every blob is complete and verified: the snapshot of the planned
        # commit points at those blobs and nothing else.
        return _lay_out(cache / folder, commit, blobs)
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
    except (OSError, httpx.HTTPError) as exc:
        raise DownloadError(f"download of {repo} from {endpoint} failed: {exc}; {RERUN}") from exc
