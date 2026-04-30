"""``Search`` Reader/Writer Protocols + native Parquet round-trip.

Mirrors :mod:`massspec.library.io` and :mod:`massspec.quant.io`. New
search-engine adapters (encyclopedia .dlib/.elib via
``massspec.io.encyclopedia``, MSFragger TSV, etc.) register themselves
in ``SEARCH_READERS`` / ``SEARCH_WRITERS`` so ``load_search`` /
``save_search`` resolve by suffix or by explicit ``format`` name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.acquisitions import ACQUISITION_TABLE, Acquisitions
from constellation.massspec.search.search import Search


# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class SearchWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, search: Search, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class SearchReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> Search: ...


SEARCH_WRITERS: dict[str, SearchWriter] = {}
SEARCH_READERS: dict[str, SearchReader] = {}


def register_writer(writer: SearchWriter) -> None:
    if writer.format_name in SEARCH_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    SEARCH_WRITERS[writer.format_name] = writer


def register_reader(reader: SearchReader) -> None:
    if reader.format_name in SEARCH_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    SEARCH_READERS[reader.format_name] = reader


def _resolve_writer(format: str | None, path: Path) -> SearchWriter:
    if format is not None:
        if format not in SEARCH_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return SEARCH_WRITERS[format]
    suffix = path.suffix.lower() or ("/" if path.is_dir() else "")
    matches = [w for w in SEARCH_WRITERS.values() if w.extension == suffix]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; specify format=. "
            f"registered: {sorted(SEARCH_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}: "
            f"{[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> SearchReader:
    if format is not None:
        if format not in SEARCH_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return SEARCH_READERS[format]
    suffix = path.suffix.lower() or ("/" if path.is_dir() else "")
    matches = [r for r in SEARCH_READERS.values() if r.extension == suffix]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; specify format=. "
            f"registered: {sorted(SEARCH_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}: "
            f"{[r.format_name for r in matches]}"
        )
    return matches[0]


def save_search(
    search: Search, path: str | Path, *, format: str | None = None, **opts: Any
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(search, p, **opts)


def load_search(
    path: str | Path, *, format: str | None = None, **opts: Any
) -> Search:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir — native, lossless
# ──────────────────────────────────────────────────────────────────────


_PARQUET_TABLES = ("peptide_scores", "protein_scores")


class ParquetDirWriter:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, search: Search, path: Path, **opts: Any) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(search.acquisitions.table, path / "acquisitions.parquet")
        for name in _PARQUET_TABLES:
            pq.write_table(getattr(search, name), path / f"{name}.parquet")
        manifest = {
            "format": self.format_name,
            "tables": ["acquisitions", *_PARQUET_TABLES],
            "metadata": search.metadata_extras,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Search:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        acq_table = pq.read_table(path / "acquisitions.parquet")
        if acq_table.schema.metadata is None:
            acq_table = acq_table.replace_schema_metadata(
                ACQUISITION_TABLE.metadata
            )
        acquisitions = Acquisitions(acq_table)

        tables: dict[str, pa.Table] = {
            name: pq.read_table(path / f"{name}.parquet")
            for name in _PARQUET_TABLES
        }
        return Search(
            acquisitions=acquisitions,
            peptide_scores=tables["peptide_scores"],
            protein_scores=tables["protein_scores"],
            metadata_extras=dict(manifest.get("metadata", {})),
        )


register_writer(ParquetDirWriter())
register_reader(ParquetDirReader())


__all__ = [
    "SearchWriter",
    "SearchReader",
    "SEARCH_WRITERS",
    "SEARCH_READERS",
    "register_writer",
    "register_reader",
    "save_search",
    "load_search",
    "ParquetDirWriter",
    "ParquetDirReader",
]
