"""``Assembly`` Reader/Writer Protocols + native ParquetDir round-trip.

Mirrors :mod:`sequencing.reference.io` and
:mod:`massspec.library.io`. Status: STUB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from constellation.sequencing.assembly.assembly import Assembly


@runtime_checkable
class AssemblyWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, assembly: Assembly, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class AssemblyReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> Assembly: ...


ASSEMBLY_WRITERS: dict[str, AssemblyWriter] = {}
ASSEMBLY_READERS: dict[str, AssemblyReader] = {}


def register_writer(writer: AssemblyWriter) -> None:
    if writer.format_name in ASSEMBLY_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    ASSEMBLY_WRITERS[writer.format_name] = writer


def register_reader(reader: AssemblyReader) -> None:
    if reader.format_name in ASSEMBLY_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    ASSEMBLY_READERS[reader.format_name] = reader


_PHASE = "Phase 7 (assembly/{hifiasm, stats})"


def save_assembly(
    assembly: Assembly,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    raise NotImplementedError(f"save_assembly pending {_PHASE}")


def load_assembly(
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> Assembly:
    raise NotImplementedError(f"load_assembly pending {_PHASE}")


class ParquetDirReader:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Assembly:
        raise NotImplementedError(f"ParquetDirReader.read pending {_PHASE}")


class ParquetDirWriter:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, assembly: Assembly, path: Path, **opts: Any) -> None:
        raise NotImplementedError(f"ParquetDirWriter.write pending {_PHASE}")


register_reader(ParquetDirReader())
register_writer(ParquetDirWriter())


__all__ = [
    "AssemblyReader",
    "AssemblyWriter",
    "ASSEMBLY_READERS",
    "ASSEMBLY_WRITERS",
    "register_reader",
    "register_writer",
    "save_assembly",
    "load_assembly",
    "ParquetDirReader",
    "ParquetDirWriter",
]
