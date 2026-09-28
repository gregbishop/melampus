"""Image handling — and the enforcement point for the no-metadata-leak rule.

Only pixels may reach the model. Filenames, keywords, EXIF and XMP must not. Rather
than trusting call sites to remember that, `staged_pixels` re-encodes the image to a
temporary file with a fixed neutral name and strips all metadata on the way out. The
model backend is only ever handed that staged path, so a leak would require actively
bypassing this module.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageOps

from .config import _bundle, cache_file
from .providers import BackendUnavailable

# Fixed name for every staged file: carries zero information about the original.
NEUTRAL_NAME = "image.jpg"

#: Where staged folders are made, under melampus's own directory: the one
#: config.cache_file names, so it is the same place in a checkout and inside
#: the executable. Not $TMPDIR, and that is the point: the staged folder is
#: also the working directory of a CLI engine's run
#: (backend.CommandBackend.complete) and the one place the Codex template's
#: permission profile leaves readable (providers.CODEX_COMMAND), and
#: `tempfile` falls back to /tmp whenever $TMPDIR is unset — ordinary on
#: Linux, in a container and under a cleared environment — so a folder placed
#: by $TMPDIR alone would be inside the grant on exactly the machines nobody
#: set it on (security review round 12). Whether this directory is outside
#: the grant is not settled by the name, though: where it lands is
#: `_data_root()`'s answer, and `staging_root` below is what checks it.
#: Nothing else is stored here: each frame's folder is removed when its
#: staging ends.
STAGING_ROOT = "staging"

#: The shared temp directories the Codex template's `:minimal` grant covers
#: whole and writable, at codex-cli 0.155.1 (security review rounds 9 and 12,
#: measured with `codex sandbox -P` under the profile in
#: providers.CODEX_COMMAND, no model call). The set lives here, in code,
#: because `staging_root` enforces it and the suite never runs the real
#: Codex; the prose that records the measurement is the `#:` block above
#: CODEX_COMMAND and docs/config.md § Codex CLI, pinned by test_docs.py.
MINIMAL_GRANTED_TEMP = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp")


def _same_directory(left: Path, right: Path) -> bool:
    """Whether two paths are the same directory on disk, by the filesystem's
    own identity (device and inode) rather than by spelling. Missing is not
    the same: nothing can be inside a directory that is not there."""
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _granted_containing(root: Path) -> str | None:
    """Which of MINIMAL_GRANTED_TEMP `root` sits inside, or None.

    Spelling does not settle it. A Mac's boot volume is case-insensitive, so
    /private/TMP and /private/tmp are one directory — `os.path.samefile` says
    so — while `Path.resolve` keeps whatever case it was handed, and
    /private/TMP is a real directory rather than a symlink, so nothing
    normalises it away: measured here, a root at
    /private/TMP/melampus-x/.melampus_cache/staging resolves to itself and
    `is_relative_to("/private/tmp")` is False, which let a root inside the
    grant through the check (Codex security review round 10, S1). So the
    ancestors are compared by identity as well.

    By identity on an *ancestor*, because the staging root usually does not
    exist when it is checked: `staged_pixels` asks for it before the mkdir
    that creates it, and `samefile` on a missing path only raises. The walk
    goes up from the root until a directory that exists is found; every
    ancestor above one that exists, exists too, so the whole chain is
    checked once the first real directory is reached. `is_relative_to`
    still runs first, for a granted directory this machine does not have at
    all — /var/tmp is absent on some Linux images, and `mkdir(parents=True)`
    would make it — where identity has nothing to compare against.
    """
    ancestors = (root, *root.parents)
    for granted in MINIMAL_GRANTED_TEMP:
        target = Path(granted)
        if root.is_relative_to(granted) or any(
            _same_directory(ancestor, target) for ancestor in ancestors
        ):
            return granted
    return None


def _refuse_inside_grant(root: Path, where: str, harm: str, fix: str) -> None:
    """Refuse `root`, resolved, when `_granted_containing` places it in the grant.

    The one refusal both of `staging_root`'s roots get: it names the root and
    the granted directory, says what a run could do there (`harm`) and what
    moves the root out (`fix`, finished by the granted directories' names).
    """
    granted = _granted_containing(root)
    if granted is not None:
        raise BackendUnavailable(
            f"{where} {root}, which is inside {granted}: one of the shared temp "
            "directories a CLI engine's permission profile grants whole and writable "
            f"(providers.CODEX_COMMAND), so {harm}. {fix} outside "
            f"{', '.join(MINIMAL_GRANTED_TEMP)} and run again."
        )


def staging_root() -> Path:
    """The directory staged folders are made in, resolved and checked first.

    `cache_file` is melampus's own directory, but *where* that directory
    lands is not melampus's choice: it is the checkout root in a checkout
    and the per-user data directory inside the executable
    ($XDG_DATA_HOME/Melampus on Linux, the variable's to set). Neither is
    guaranteed to sit outside the directories `:minimal` grants, and both
    were measured inside them — a checkout under /tmp, and a frozen run with
    $XDG_DATA_HOME pointed there — with a sibling frame's staged file read
    and the staged image itself overwritten and read back by the run
    analysing it (Codex security review round 9 on PR #17). So the root is
    checked before it is used rather than assumed, and it is checked
    resolved: /tmp is a symlink to /private/tmp on a Mac, and a root reached
    through a link of its own is where the link leads, not where it is
    spelled. Resolved is still only a path, though, and a path is not what
    the filesystem thinks: `_granted_containing` is what judges the root,
    and it compares the directories themselves.

    A root inside the grant is refused, not worked around. Staging elsewhere
    would mean picking a directory melampus can prove is outside the grant,
    and the two directories it has — the checkout root and the per-user data
    directory — are exactly the two that can be inside it; a third, guessed,
    would also move the user's data somewhere they never configured. The
    refusal names the root, the granted directory it sits in, and the one
    thing that fixes it.

    It is refused as `providers.BackendUnavailable`, the refusal this machine
    cannot run as configured already has: an uninstalled command, a signed-out
    CLI, an Ollama with no server. That is what makes it stop the run
    (`providers.BATCH_FATAL`, exit 3) rather than being recorded as one more
    frame's error — the root is the same on every frame, so a per-frame error
    would be written once per photograph with the fix scrolling past above the
    table (Codex review round 10, C1).

    Inside the executable there is a second root to judge, the unpack
    directory (`config._bundle`, `sys._MEIPASS`): the code and the prompts
    are there, `PromptLibrary.render` reads a prompt file from it for every
    frame, after the routing run as well as before it, and escalation's lazy
    `import anthropic` loads a native module from it into this process. The
    bootloader puts it in $TMPDIR at every launch, and in /tmp when that is
    unset, and a command under the Codex profile overwrote a prompt file in
    a folder under /tmp (security review round 1 on PR #28). The staging
    root's check never sees it — in the executable that root is the per-user
    data directory — so it is judged here too, before any frame is staged,
    whatever the engine; in a checkout the code sits under the checkout
    root, which the staging root's check already covers.
    """
    root = cache_file(STAGING_ROOT).resolve()
    _refuse_inside_grant(
        root, "melampus would stage images in",
        "the run analysing one frame could read the frames staged beside it and "
        "overwrite the image it was given",
        "That directory follows the root melampus keeps its data under — the checkout "
        "root in a checkout, $XDG_DATA_HOME/Melampus or the platform's per-user data "
        "directory inside the executable — so put that root",
    )
    bundle = _bundle()
    if bundle is not None:
        _refuse_inside_grant(
            bundle.resolve(), "melampus is running from",
            "the run analysing one frame could rewrite the prompts melampus reads from "
            "there for the next, or the code it loads",
            "The executable unpacks itself into $TMPDIR, and into /tmp when that is unset, "
            "so set $TMPDIR to a directory",
        )
    return root


def content_hash(path: Path) -> str:
    """SHA-256 of the file bytes. Cache key, so re-runs skip completed work."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def staged_pixels(path: Path, max_edge: int, quality: int = 92) -> Iterator[Path]:
    """Yield a path to a metadata-free, bounded-size copy of the image.

    Bounding the long edge matters for speed: full-resolution frames cost far more
    vision tokens without improving identification. Re-encoding through a fresh
    Image object drops EXIF, XMP and IPTC.
    """
    root = staging_root()
    with Image.open(path) as source:
        # Honour EXIF orientation before discarding EXIF, or subjects arrive rotated.
        oriented = ImageOps.exif_transpose(source)
        rgb = oriented.convert("RGB")
        if max_edge > 0 and max(rgb.size) > max_edge:
            scale = max_edge / max(rgb.size)
            rgb = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                Image.LANCZOS,
            )
        # Rebuild from raw bytes: carries pixels across and nothing else.
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="melampus-", dir=root) as tmp:
        staged = Path(tmp) / NEUTRAL_NAME
        clean.save(staged, format="JPEG", quality=quality)
        yield staged
