"""``Assembly`` Reader/Writer Protocols + native ParquetDir round-trip.

Mirrors :mod:`sequencing.reference.io` and :mod:`massspec.library.io` —
Reader/Writer Protocols with ``extension`` / ``format_name`` / ``lossy``
ClassVars, suffix-or-format dispatch, ParquetDir as the native lossless
form. The ``scaffolds`` table is optional: written only when present,
and a missing ``scaffolds.parquet`` reads back as ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow.parquet as pq

from constellation.sequencing.assembly.assembly import Assembly


# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


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


def _resolve_writer(format: str | None, path: Path) -> AssemblyWriter:
    if format is not None:
        if format not in ASSEMBLY_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return ASSEMBLY_WRITERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [w for w in ASSEMBLY_WRITERS.values() if w.extension == suffix]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(ASSEMBLY_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> AssemblyReader:
    if format is not None:
        if format not in ASSEMBLY_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return ASSEMBLY_READERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [r for r in ASSEMBLY_READERS.values() if r.extension == suffix]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(ASSEMBLY_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[r.format_name for r in matches]}"
        )
    return matches[0]


def save_assembly(
    assembly: Assembly,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(assembly, p, **opts)


def load_assembly(
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> Assembly:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir native lossless form
# ──────────────────────────────────────────────────────────────────────


class ParquetDirWriter:
    """Write an ``Assembly`` to a directory of one Parquet per table.

    ``scaffolds`` is written only when present; container-level metadata
    round-trips through ``manifest.json``.
    """

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, assembly: Assembly, path: Path, **opts: Any) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(assembly.contigs, path / "contigs.parquet")
        # One contig per row group so a `contig_id == X` filter can skip
        # whole-genome decode (same rationale as GenomeReference).
        pq.write_table(assembly.sequences, path / "sequences.parquet", row_group_size=1)
        pq.write_table(assembly.stats, path / "stats.parquet")
        tables = ["contigs", "sequences", "stats"]
        if assembly.scaffolds is not None:
            pq.write_table(assembly.scaffolds, path / "scaffolds.parquet")
            tables.append("scaffolds")
        manifest = {
            "format": self.format_name,
            "tables": tables,
            "container": "Assembly",
            "has_scaffolds": assembly.scaffolds is not None,
            "metadata": assembly.metadata_extras,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    """Read an ``Assembly`` from a ParquetDir bundle."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Assembly:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        contigs = pq.read_table(path / "contigs.parquet")
        sequences = pq.read_table(path / "sequences.parquet")
        stats = pq.read_table(path / "stats.parquet")
        scaffolds_path = path / "scaffolds.parquet"
        scaffolds = pq.read_table(scaffolds_path) if scaffolds_path.exists() else None
        return Assembly(
            contigs=contigs,
            sequences=sequences,
            scaffolds=scaffolds,
            stats=stats,
            metadata_extras=dict(manifest.get("metadata", {})),
        )


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
