"""``Alignments`` Reader/Writer Protocols + native ParquetDir + BAM.

ParquetDir round-trip for processed / cached alignments; BAM via
:mod:`sequencing.io.sam_bam` is the authoritative on-disk form for
externally-produced alignments.

Status: STUB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from constellation.sequencing.alignments.alignments import Alignments


@runtime_checkable
class AlignmentsWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, alignments: Alignments, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class AlignmentsReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> Alignments: ...


ALIGNMENTS_WRITERS: dict[str, AlignmentsWriter] = {}
ALIGNMENTS_READERS: dict[str, AlignmentsReader] = {}


def register_writer(writer: AlignmentsWriter) -> None:
    if writer.format_name in ALIGNMENTS_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    ALIGNMENTS_WRITERS[writer.format_name] = writer


def register_reader(reader: AlignmentsReader) -> None:
    if reader.format_name in ALIGNMENTS_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    ALIGNMENTS_READERS[reader.format_name] = reader


_PHASE = "Phase 2 (readers/sam_bam) for BAM; Phase 1 for ParquetDir"


def save_alignments(
    alignments: Alignments,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    raise NotImplementedError(f"save_alignments pending {_PHASE}")


def load_alignments(
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> Alignments:
    raise NotImplementedError(f"load_alignments pending {_PHASE}")


class ParquetDirReader:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Alignments:
        raise NotImplementedError(f"ParquetDirReader.read pending {_PHASE}")


class ParquetDirWriter:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, alignments: Alignments, path: Path, **opts: Any) -> None:
        raise NotImplementedError(f"ParquetDirWriter.write pending {_PHASE}")


register_reader(ParquetDirReader())
register_writer(ParquetDirWriter())


__all__ = [
    "AlignmentsReader",
    "AlignmentsWriter",
    "ALIGNMENTS_READERS",
    "ALIGNMENTS_WRITERS",
    "register_reader",
    "register_writer",
    "save_alignments",
    "load_alignments",
    "ParquetDirReader",
    "ParquetDirWriter",
]
