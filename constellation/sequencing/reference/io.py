"""``GenomeReference`` Reader/Writer Protocols + native ParquetDir round-trip.

Mirrors the shape of :mod:`massspec.library.io` — Reader/Writer Protocols
with ``extension`` / ``format_name`` / ``lossy`` ClassVars, suffix-or-format
dispatch, ParquetDir as the native lossless form. External-format adapters
(FASTA, .2bit, etc.) self-register when their reader modules are imported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.reference.reference import GenomeReference

# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class GenomeReferenceWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, reference: GenomeReference, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class GenomeReferenceReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> GenomeReference: ...


GENOME_REFERENCE_WRITERS: dict[str, GenomeReferenceWriter] = {}
GENOME_REFERENCE_READERS: dict[str, GenomeReferenceReader] = {}


def register_writer(writer: GenomeReferenceWriter) -> None:
    if writer.format_name in GENOME_REFERENCE_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    GENOME_REFERENCE_WRITERS[writer.format_name] = writer


def register_reader(reader: GenomeReferenceReader) -> None:
    if reader.format_name in GENOME_REFERENCE_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    GENOME_REFERENCE_READERS[reader.format_name] = reader


def _resolve_writer(format: str | None, path: Path) -> GenomeReferenceWriter:
    if format is not None:
        if format not in GENOME_REFERENCE_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return GENOME_REFERENCE_WRITERS[format]
    # No file extension → treat as a directory destination (the most
    # common case for this container, since the native form is
    # ParquetDir). Only fall back to a literal extension match when the
    # path actually has one.
    suffix = path.suffix.lower() or "/"
    matches = [w for w in GENOME_REFERENCE_WRITERS.values() if w.extension == suffix]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(GENOME_REFERENCE_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> GenomeReferenceReader:
    if format is not None:
        if format not in GENOME_REFERENCE_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return GENOME_REFERENCE_READERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [r for r in GENOME_REFERENCE_READERS.values() if r.extension == suffix]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(GENOME_REFERENCE_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[r.format_name for r in matches]}"
        )
    return matches[0]


def save_genome_reference(
    reference: GenomeReference,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(reference, p, **opts)


def load_genome_reference(
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> GenomeReference:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir native lossless form
# ──────────────────────────────────────────────────────────────────────


_PARQUET_TABLES = ("contigs", "sequences")


class ParquetDirWriter:
    """Write a ``GenomeReference`` to a directory of one Parquet per table.

    Container-level metadata round-trips through ``manifest.json``.
    """

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, reference: GenomeReference, path: Path, **opts: Any) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for name in _PARQUET_TABLES:
            table = getattr(reference, name)
            kwargs: dict[str, Any] = {}
            if name == "sequences":
                # One contig per row group. Lets the viz reader (and any
                # other consumer) skip whole-genome materialization when
                # only one contig's sequence is needed — at default
                # row_group_size mouse/human references pack every
                # contig's bytes into a single row group, so a
                # `contig_id == X` filter still pays the full decode.
                kwargs["row_group_size"] = 1
            pq.write_table(table, path / f"{name}.parquet", **kwargs)
        manifest = {
            "format": self.format_name,
            "tables": list(_PARQUET_TABLES),
            "container": "GenomeReference",
            "metadata": reference.metadata_extras,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    """Read a ``GenomeReference`` from a ParquetDir bundle."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> GenomeReference:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        tables: dict[str, pa.Table] = {
            name: pq.read_table(path / f"{name}.parquet") for name in _PARQUET_TABLES
        }
        return GenomeReference(
            contigs=tables["contigs"],
            sequences=tables["sequences"],
            metadata_extras=dict(manifest.get("metadata", {})),
        )


register_writer(ParquetDirWriter())
register_reader(ParquetDirReader())


__all__ = [
    "GenomeReferenceReader",
    "GenomeReferenceWriter",
    "GENOME_REFERENCE_READERS",
    "GENOME_REFERENCE_WRITERS",
    "register_reader",
    "register_writer",
    "save_genome_reference",
    "load_genome_reference",
    "ParquetDirReader",
    "ParquetDirWriter",
]
