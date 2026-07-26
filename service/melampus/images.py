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

# Fixed name for every staged file: carries zero information about the original.
NEUTRAL_NAME = "image.jpg"


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

    with tempfile.TemporaryDirectory(prefix="melampus-") as tmp:
        staged = Path(tmp) / NEUTRAL_NAME
        clean.save(staged, format="JPEG", quality=quality)
        yield staged
