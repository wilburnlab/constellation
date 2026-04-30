"""``Quant`` Reader/Writer Protocols + native Parquet round-trip.

Mirrors ``massspec.library.io``: native ParquetDir round-trip ships now;
.elib (EncyclopeDIA empirical-library quants) is stubbed pending the
EncyclopeDIA reader port.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.acquisitions import ACQUISITION_TABLE, Acquisitions
from constellation.massspec.quant.quant import Quant

# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class QuantWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, quant: Quant, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class QuantReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> Quant: ...


QUANT_WRITERS: dict[str, QuantWriter] = {}
QUANT_READERS: dict[str, QuantReader] = {}


def register_writer(writer: QuantWriter) -> None:
    if writer.format_name in QUANT_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    QUANT_WRITERS[writer.format_name] = writer


def register_reader(reader: QuantReader) -> None:
    if reader.format_name in QUANT_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    QUANT_READERS[reader.format_name] = reader


def _resolve_writer(format: str | None, path: Path) -> QuantWriter:
    if format is not None:
        if format not in QUANT_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return QUANT_WRITERS[format]
    suffix = path.suffix.lower() or ("/" if path.is_dir() else "")
    matches = [w for w in QUANT_WRITERS.values() if w.extension == suffix]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; specify format=. "
            f"registered: {sorted(QUANT_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}: "
            f"{[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> QuantReader:
    if format is not None:
        if format not in QUANT_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return QUANT_READERS[format]
    suffix = path.suffix.lower() or ("/" if path.is_dir() else "")
    matches = [r for r in QUANT_READERS.values() if r.extension == suffix]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; specify format=. "
            f"registered: {sorted(QUANT_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}: "
            f"{[r.format_name for r in matches]}"
        )
    return matches[0]


def save_quant(
    quant: Quant, path: str | Path, *, format: str | None = None, **opts: Any
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(quant, p, **opts)


def load_quant(
    path: str | Path, *, format: str | None = None, **opts: Any
) -> Quant:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir — native, lossless
# ──────────────────────────────────────────────────────────────────────


_PARQUET_TABLES = (
    "protein_quant",
    "peptide_quant",
    "precursor_quant",
    "transmission_protein_peptide",
    "transmission_peptide_precursor",
)


class ParquetDirWriter:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, quant: Quant, path: Path, **opts: Any) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(quant.acquisitions.table, path / "acquisitions.parquet")
        for name in _PARQUET_TABLES:
            pq.write_table(getattr(quant, name), path / f"{name}.parquet")
        manifest = {
            "format": self.format_name,
            "tables": ["acquisitions", *_PARQUET_TABLES],
            "metadata": quant.metadata_extras,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Quant:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        acq_table = pq.read_table(path / "acquisitions.parquet")
        # Preserve schema metadata if a previous version omitted it.
        if acq_table.schema.metadata is None:
            acq_table = acq_table.replace_schema_metadata(
                ACQUISITION_TABLE.metadata
            )
        acquisitions = Acquisitions(acq_table)

        tables: dict[str, pa.Table] = {
            name: pq.read_table(path / f"{name}.parquet")
            for name in _PARQUET_TABLES
        }
        return Quant(
            acquisitions=acquisitions,
            protein_quant=tables["protein_quant"],
            peptide_quant=tables["peptide_quant"],
            precursor_quant=tables["precursor_quant"],
            transmission_protein_peptide=tables["transmission_protein_peptide"],
            transmission_peptide_precursor=tables["transmission_peptide_precursor"],
            metadata_extras=dict(manifest.get("metadata", {})),
        )


register_writer(ParquetDirWriter())
register_reader(ParquetDirReader())


__all__ = [
    "QuantWriter",
    "QuantReader",
    "QUANT_WRITERS",
    "QUANT_READERS",
    "register_writer",
    "register_reader",
    "save_quant",
    "load_quant",
    "ParquetDirWriter",
    "ParquetDirReader",
]
