"""Protocol adapter wiring for the MaxQuant ``combined/txt/`` reader.

Self-registers ``_MaxQuantSearchReader`` against ``SEARCH_READERS`` under
the format name ``"maxquant"``.

**Format-name addressing only.** ``extension`` is the empty string, NOT
``"/"``: a MaxQuant export is a *directory*, and ``ParquetDirReader``
already claims the ``"/"`` suffix — registering a second ``"/"`` reader
would make ``load_search(some_dir)`` ambiguous. An empty extension never
auto-resolves, so this reader is reachable only via the explicit
``load_search(path, format="maxquant")``. (Both a MaxQuant ``txt/`` dir
and a Constellation ParquetDir are "directories"; only the caller knows
which.) There is no MaxQuant *writer* — persistence is the native
``ParquetDirWriter`` (which round-trips ``Search.psms``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from constellation.massspec.search.io import register_reader
from constellation.massspec.search.search import Search

from constellation.massspec.io.maxquant._read import read_maxquant_search


class _MaxQuantSearchReader:
    """SearchReader Protocol adapter for MaxQuant ``combined/txt/`` exports."""

    extension: ClassVar[str] = ""  # never auto-resolves; use format="maxquant"
    format_name: ClassVar[str] = "maxquant"

    def read(self, path: Path, **opts: Any) -> Search:
        return read_maxquant_search(path, **opts)


register_reader(_MaxQuantSearchReader())


__all__ = ["_MaxQuantSearchReader"]
