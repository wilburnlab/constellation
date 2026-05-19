"""Protocol adapter wiring for the NIST .msp library-view reader.

Self-registers ``_MspLibraryReader`` against ``LIBRARY_READERS`` so
``massspec.library.io.load_library("foo.msp")`` resolves by suffix
without callers thinking about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from constellation.massspec.library.io import register_reader
from constellation.massspec.library.library import Library

from constellation.massspec.io.msp._read import read_msp_library


class _MspLibraryReader:
    """LibraryReader Protocol adapter for ``.msp`` files."""

    extension: ClassVar[str] = ".msp"
    format_name: ClassVar[str] = "nist.msp"

    def read(self, path: Path, **opts: Any) -> Library:
        return read_msp_library(path, **opts)


register_reader(_MspLibraryReader())


__all__ = ["_MspLibraryReader"]
