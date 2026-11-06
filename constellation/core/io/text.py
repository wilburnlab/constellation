"""Opening text files that may be gzip-compressed.

One function, because seven modules each carried a private copy that
decided by the ``.gz`` suffix. This one decides by the gzip magic bytes,
so a compressed file without the suffix reads, and an uncompressed one
that happens to end in ``.gz`` does too. BGZF (``bgzip``) output is a
valid gzip stream and reads the same way.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

__all__ = ["GZIP_MAGIC", "is_gzip", "open_text"]

#: The first two bytes of every gzip member (RFC 1952).
GZIP_MAGIC = b"\x1f\x8b"


def is_gzip(path: str | Path) -> bool:
    """True when ``path`` starts with the gzip magic bytes."""
    with open(path, "rb") as fh:
        return fh.read(2) == GZIP_MAGIC


def open_text(path: str | Path, *, encoding: str = "utf-8") -> IO[str]:
    """Open ``path`` for reading text, decompressing gzip transparently."""
    p = Path(path)
    if is_gzip(p):
        return gzip.open(p, "rt", encoding=encoding)
    return p.open("r", encoding=encoding)
