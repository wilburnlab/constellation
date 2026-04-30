"""Protocol adapters that register the encyclopedia reader/writer.

The shared ``read_encyclopedia`` / ``write_encyclopedia`` core produces
``(Library, Quant?, Search?)`` tuples; six tiny adapter classes plug
into the Library / Quant / Search Reader+Writer registries so callers
using ``load_library`` / ``load_quant`` / ``load_search`` (or their
save counterparts) on a ``.dlib`` / ``.elib`` file resolve to the same
backend.

Each adapter is registered twice: once with ``extension = ".dlib"``
and once with ``.elib"`` (different ``format_name`` keys to satisfy
the registry's no-duplicate rule). Both routes call the same shared
core; the file extension is purely a naming convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from constellation.massspec.io.encyclopedia._read import read_encyclopedia
from constellation.massspec.io.encyclopedia._write import write_encyclopedia
from constellation.massspec.library.io import (
    register_reader as register_library_reader,
    register_writer as register_library_writer,
)
from constellation.massspec.library.library import Library
from constellation.massspec.quant.io import (
    register_reader as register_quant_reader,
    register_writer as register_quant_writer,
)
from constellation.massspec.quant.quant import Quant
from constellation.massspec.search.io import (
    register_reader as register_search_reader,
    register_writer as register_search_writer,
)
from constellation.massspec.search.search import Search


# ──────────────────────────────────────────────────────────────────────
# Library Reader/Writer adapters
# ──────────────────────────────────────────────────────────────────────


class _DlibLibraryReader:
    extension: ClassVar[str] = ".dlib"
    format_name: ClassVar[str] = "encyclopedia.dlib"

    def read(self, path: Path, **opts: Any) -> Library:
        return read_encyclopedia(path, **opts).library


class _ElibLibraryReader:
    extension: ClassVar[str] = ".elib"
    format_name: ClassVar[str] = "encyclopedia.elib"

    def read(self, path: Path, **opts: Any) -> Library:
        return read_encyclopedia(path, **opts).library


class _DlibLibraryWriter:
    extension: ClassVar[str] = ".dlib"
    format_name: ClassVar[str] = "encyclopedia.dlib"
    lossy: ClassVar[bool] = True  # terminal-mod placement collapses (see _modseq.py)

    def write(
        self,
        library: Library,
        path: Path,
        *,
        quant: Quant | None = None,
        search: Search | None = None,
        overwrite: bool = False,
        **opts: Any,
    ) -> None:
        write_encyclopedia(path, library, quant=quant, search=search, overwrite=overwrite)


class _ElibLibraryWriter:
    extension: ClassVar[str] = ".elib"
    format_name: ClassVar[str] = "encyclopedia.elib"
    lossy: ClassVar[bool] = True

    def write(
        self,
        library: Library,
        path: Path,
        *,
        quant: Quant | None = None,
        search: Search | None = None,
        overwrite: bool = False,
        **opts: Any,
    ) -> None:
        write_encyclopedia(path, library, quant=quant, search=search, overwrite=overwrite)


# ──────────────────────────────────────────────────────────────────────
# Quant Reader/Writer adapters
# ──────────────────────────────────────────────────────────────────────


class _DlibQuantReader:
    extension: ClassVar[str] = ".dlib"
    format_name: ClassVar[str] = "encyclopedia.dlib"

    def read(self, path: Path, **opts: Any) -> Quant:
        result = read_encyclopedia(path, **opts)
        if result.quant is None:
            raise ValueError(
                f"{path} carries no sample-specific quant data; "
                "use load_library() if you only want the predicted library"
            )
        return result.quant


class _ElibQuantReader:
    extension: ClassVar[str] = ".elib"
    format_name: ClassVar[str] = "encyclopedia.elib"

    def read(self, path: Path, **opts: Any) -> Quant:
        result = read_encyclopedia(path, **opts)
        if result.quant is None:
            raise ValueError(
                f"{path} carries no sample-specific quant data; "
                "use load_library() if you only want the predicted library"
            )
        return result.quant


class _DlibQuantWriter:
    extension: ClassVar[str] = ".dlib"
    format_name: ClassVar[str] = "encyclopedia.dlib"
    lossy: ClassVar[bool] = True

    def write(
        self,
        quant: Quant,
        path: Path,
        *,
        library: Library,
        search: Search | None = None,
        overwrite: bool = False,
        **opts: Any,
    ) -> None:
        write_encyclopedia(path, library, quant=quant, search=search, overwrite=overwrite)


class _ElibQuantWriter:
    extension: ClassVar[str] = ".elib"
    format_name: ClassVar[str] = "encyclopedia.elib"
    lossy: ClassVar[bool] = True

    def write(
        self,
        quant: Quant,
        path: Path,
        *,
        library: Library,
        search: Search | None = None,
        overwrite: bool = False,
        **opts: Any,
    ) -> None:
        write_encyclopedia(path, library, quant=quant, search=search, overwrite=overwrite)


# ──────────────────────────────────────────────────────────────────────
# Search Reader/Writer adapters
# ──────────────────────────────────────────────────────────────────────


class _DlibSearchReader:
    extension: ClassVar[str] = ".dlib"
    format_name: ClassVar[str] = "encyclopedia.dlib"

    def read(self, path: Path, **opts: Any) -> Search:
        result = read_encyclopedia(path, **opts)
        if result.search is None:
            raise ValueError(
                f"{path} carries no peptidescores / proteinscores; "
                "no Search projection available"
            )
        return result.search


class _ElibSearchReader:
    extension: ClassVar[str] = ".elib"
    format_name: ClassVar[str] = "encyclopedia.elib"

    def read(self, path: Path, **opts: Any) -> Search:
        result = read_encyclopedia(path, **opts)
        if result.search is None:
            raise ValueError(
                f"{path} carries no peptidescores / proteinscores; "
                "no Search projection available"
            )
        return result.search


class _DlibSearchWriter:
    extension: ClassVar[str] = ".dlib"
    format_name: ClassVar[str] = "encyclopedia.dlib"
    lossy: ClassVar[bool] = True

    def write(
        self,
        search: Search,
        path: Path,
        *,
        library: Library,
        quant: Quant | None = None,
        overwrite: bool = False,
        **opts: Any,
    ) -> None:
        write_encyclopedia(path, library, quant=quant, search=search, overwrite=overwrite)


class _ElibSearchWriter:
    extension: ClassVar[str] = ".elib"
    format_name: ClassVar[str] = "encyclopedia.elib"
    lossy: ClassVar[bool] = True

    def write(
        self,
        search: Search,
        path: Path,
        *,
        library: Library,
        quant: Quant | None = None,
        overwrite: bool = False,
        **opts: Any,
    ) -> None:
        write_encyclopedia(path, library, quant=quant, search=search, overwrite=overwrite)


# ──────────────────────────────────────────────────────────────────────
# Self-register on import
# ──────────────────────────────────────────────────────────────────────


register_library_reader(_DlibLibraryReader())
register_library_reader(_ElibLibraryReader())
register_library_writer(_DlibLibraryWriter())
register_library_writer(_ElibLibraryWriter())

register_quant_reader(_DlibQuantReader())
register_quant_reader(_ElibQuantReader())
register_quant_writer(_DlibQuantWriter())
register_quant_writer(_ElibQuantWriter())

register_search_reader(_DlibSearchReader())
register_search_reader(_ElibSearchReader())
register_search_writer(_DlibSearchWriter())
register_search_writer(_ElibSearchWriter())


__all__: list[str] = []
