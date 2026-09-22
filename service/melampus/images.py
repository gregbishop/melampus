"""Image handling — and the enforcement point for the no-metadata-leak rule.

Only pixels may reach the model. Filenames, keywords, EXIF and XMP must not. Rather
than trusting call sites to remember that, `staged_pixels` re-encodes the image to a
temporary file with a fixed neutral name and strips all metadata on the way out. The
model backend is only ever handed that staged path, so a leak would require actively
bypassing this module.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageOps

from .config import cache_file

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
    spelled.

    A root inside the grant is refused, not worked around. Staging elsewhere
    would mean picking a directory melampus can prove is outside the grant,
    and the two directories it has — the checkout root and the per-user data
    directory — are exactly the two that can be inside it; a third, guessed,
    would also move the user's data somewhere they never configured. The
    refusal names the root, the granted directory it sits in, and the one
    thing that fixes it.
    """
    root = cache_file(STAGING_ROOT).resolve()
    granted = next((d for d in MINIMAL_GRANTED_TEMP if root.is_relative_to(d)), None)
    if granted is not None:
        raise RuntimeError(
            f"melampus would stage images in {root}, which is inside {granted}: one of "
            "the shared temp directories a CLI engine's permission profile grants whole "
            "and writable (providers.CODEX_COMMAND), so the run analysing one frame "
            "could read the frames staged beside it and overwrite the image it was "
            "given. That directory follows the root melampus keeps its data under — the "
            "checkout root in a checkout, $XDG_DATA_HOME/Melampus or the platform's "
            "per-user data directory inside the executable — so put that root outside "
            f"{', '.join(MINIMAL_GRANTED_TEMP)} and run again."
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
