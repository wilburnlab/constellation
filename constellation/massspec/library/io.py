"""``Library`` Reader/Writer Protocols + native Parquet round-trip.

The Reader/Writer pair is the extension point for new library formats
(.dlib for EncyclopeDIA, BiblioSpec .blib, NIST .msp, ...). Each format
adapter registers itself by name; ``Library.save(path, format=...)``
dispatches by name (and falls back to suffix-matching when one writer
owns the extension).

This session ships only ``ParquetDirReader`` / ``ParquetDirWriter`` —
the lossless native form. ``DlibReader`` / ``DlibWriter`` are stubbed
with ``NotImplementedError`` so the registry slot exists; full
implementation lands with the EncyclopeDIA reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.library.library import Library

# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class LibraryWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, library: Library, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class LibraryReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> Library: ...


LIBRARY_WRITERS: dict[str, LibraryWriter] = {}
LIBRARY_READERS: dict[str, LibraryReader] = {}


def register_writer(writer: LibraryWriter) -> None:
    if writer.format_name in LIBRARY_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    LIBRARY_WRITERS[writer.format_name] = writer


def register_reader(reader: LibraryReader) -> None:
    if reader.format_name in LIBRARY_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    LIBRARY_READERS[reader.format_name] = reader


def _resolve_writer(format: str | None, path: Path) -> LibraryWriter:
    if format is not None:
        if format not in LIBRARY_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return LIBRARY_WRITERS[format]
    suffix = path.suffix.lower() or ("/" if path.is_dir() else "")
    matches = [w for w in LIBRARY_WRITERS.values() if w.extension == suffix]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(LIBRARY_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> LibraryReader:
    if format is not None:
        if format not in LIBRARY_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return LIBRARY_READERS[format]
    suffix = path.suffix.lower() or ("/" if path.is_dir() else "")
    matches = [r for r in LIBRARY_READERS.values() if r.extension == suffix]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(LIBRARY_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[r.format_name for r in matches]}"
        )
    return matches[0]


def save_library(
    library: Library, path: str | Path, *, format: str | None = None, **opts: Any
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(library, p, **opts)


def load_library(
    path: str | Path, *, format: str | None = None, **opts: Any
) -> Library:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir — native, lossless
# ──────────────────────────────────────────────────────────────────────


_PARQUET_TABLES = (
    "proteins",
    "peptides",
    "precursors",
    "fragments",
    "protein_peptide",
)


# Sentinel embedded into manifest.json in place of a pa.Table value —
# the reader recognises this shape and rehydrates the table from the
# sidecar parquet file referenced by ``__pa_table__``.
_TABLE_REF_KEY = "__pa_table__"


def _safe_sidecar_name(key: str) -> str:
    """Map a metadata_extras key (e.g. ``x.msp.precursor_comments``)
    to a filesystem-safe parquet filename stem."""
    return "_metadata_" + "".join(c if c.isalnum() else "_" for c in key)


def _encode_metadata_for_manifest(
    metadata: dict[str, Any], path: Path
) -> dict[str, Any]:
    """Replace ``pa.Table`` values with a sidecar reference, writing
    each table to ``path/_metadata_<key>.parquet``. All other values
    must be JSON-serialisable as today.
    """
    encoded: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, pa.Table):
            sidecar = _safe_sidecar_name(key) + ".parquet"
            pq.write_table(value, path / sidecar)
            encoded[key] = {_TABLE_REF_KEY: sidecar}
        else:
            encoded[key] = value
    return encoded


def _decode_metadata_from_manifest(
    encoded: dict[str, Any], path: Path
) -> dict[str, Any]:
    """Inverse of ``_encode_metadata_for_manifest``."""
    decoded: dict[str, Any] = {}
    for key, value in encoded.items():
        if isinstance(value, dict) and _TABLE_REF_KEY in value:
            decoded[key] = pq.read_table(path / value[_TABLE_REF_KEY])
        else:
            decoded[key] = value
    return decoded


class ParquetDirWriter:
    """Write a ``Library`` to a directory of one Parquet file per table.

    Library-level metadata round-trips through ``manifest.json``. Values
    in ``metadata_extras`` that are ``pa.Table`` instances are written
    out as sidecar parquet files (``_metadata_<key>.parquet``) and the
    manifest carries a reference object in their place; this lets
    readers (e.g. NIST .msp) attach per-precursor side-tables that
    survive a ParquetDir round-trip.
    """

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, library: Library, path: Path, **opts: Any) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for name in _PARQUET_TABLES:
            pq.write_table(getattr(library, name), path / f"{name}.parquet")
        manifest = {
            "format": self.format_name,
            "tables": list(_PARQUET_TABLES),
            "metadata": _encode_metadata_for_manifest(
                library.metadata_extras, path
            ),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Library:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        tables: dict[str, pa.Table] = {
            name: pq.read_table(path / f"{name}.parquet")
            for name in _PARQUET_TABLES
        }
        return Library(
            proteins=tables["proteins"],
            peptides=tables["peptides"],
            precursors=tables["precursors"],
            fragments=tables["fragments"],
            protein_peptide=tables["protein_peptide"],
            metadata_extras=_decode_metadata_from_manifest(
                manifest.get("metadata", {}), path
            ),
        )


register_writer(ParquetDirWriter())
register_reader(ParquetDirReader())


__all__ = [
    "LibraryWriter",
    "LibraryReader",
    "LIBRARY_WRITERS",
    "LIBRARY_READERS",
    "register_writer",
    "register_reader",
    "save_library",
    "load_library",
    "ParquetDirWriter",
    "ParquetDirReader",
]
