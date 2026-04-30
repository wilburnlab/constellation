"""``Reference`` Reader/Writer Protocols + native ParquetDir round-trip.

Mirrors the shape of :mod:`massspec.library.io` — Reader/Writer
Protocols with ``extension`` / ``format_name`` / ``lossy`` ClassVars,
suffix-or-format dispatch, ParquetDir as the native lossless form.
External-format adapters (FASTA + GFF, .2bit, etc.) self-register
when their reader modules are imported.

Status: STUB. Protocols + registry final shape; Reader/Writer
implementations raise ``NotImplementedError`` pending Phase 1
(Foundation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from constellation.sequencing.reference.reference import Reference


# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class ReferenceWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, reference: Reference, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class ReferenceReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> Reference: ...


REFERENCE_WRITERS: dict[str, ReferenceWriter] = {}
REFERENCE_READERS: dict[str, ReferenceReader] = {}


def register_writer(writer: ReferenceWriter) -> None:
    if writer.format_name in REFERENCE_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    REFERENCE_WRITERS[writer.format_name] = writer


def register_reader(reader: ReferenceReader) -> None:
    if reader.format_name in REFERENCE_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    REFERENCE_READERS[reader.format_name] = reader


# ──────────────────────────────────────────────────────────────────────
# User-facing entry points
# ──────────────────────────────────────────────────────────────────────


_PHASE = "Phase 1 (Foundation)"


def save_reference(
    reference: Reference,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    raise NotImplementedError(f"save_reference pending {_PHASE}")


def load_reference(
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> Reference:
    raise NotImplementedError(f"load_reference pending {_PHASE}")


# ──────────────────────────────────────────────────────────────────────
# ParquetDir native lossless form
# ──────────────────────────────────────────────────────────────────────


class ParquetDirReader:
    """Native lossless Reference reader.

    Reads a directory containing one parquet file per table
    (``contigs.parquet``, ``sequences.parquet``, ``features.parquet``)
    plus a ``manifest.json`` with metadata extras.
    """

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Reference:
        raise NotImplementedError(f"ParquetDirReader.read pending {_PHASE}")


class ParquetDirWriter:
    """Native lossless Reference writer."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, reference: Reference, path: Path, **opts: Any) -> None:
        raise NotImplementedError(f"ParquetDirWriter.write pending {_PHASE}")


# Register the native form so the registry has a default at import
# time. External-format adapters (FASTA+GFF, etc.) will register from
# their own modules when those land in Phase 1.
register_reader(ParquetDirReader())
register_writer(ParquetDirWriter())


__all__ = [
    "ReferenceReader",
    "ReferenceWriter",
    "REFERENCE_READERS",
    "REFERENCE_WRITERS",
    "register_reader",
    "register_writer",
    "save_reference",
    "load_reference",
    "ParquetDirReader",
    "ParquetDirWriter",
]
