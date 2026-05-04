"""``Alignments`` Reader/Writer Protocols + native ParquetDir + BAM.

ParquetDir round-trip for processed / cached alignments; BAM via
:mod:`sequencing.io.sam_bam` is the authoritative on-disk form for
externally-produced alignments.

``load_alignments`` materialises the full table into memory — fine for
ParquetDir-as-cache use cases (small alignment subsets, test fixtures),
NOT for opening the pipeline's ``alignments/`` partition at full scale.
For pipeline-scale access, callers open ``pa.dataset.dataset(alignments/)``
directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow.parquet as pq

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


def _resolve_writer(format: str | None, path: Path) -> AlignmentsWriter:
    if format is not None:
        if format not in ALIGNMENTS_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return ALIGNMENTS_WRITERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [w for w in ALIGNMENTS_WRITERS.values() if w.extension == suffix]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(ALIGNMENTS_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> AlignmentsReader:
    if format is not None:
        if format not in ALIGNMENTS_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return ALIGNMENTS_READERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [r for r in ALIGNMENTS_READERS.values() if r.extension == suffix]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(ALIGNMENTS_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[r.format_name for r in matches]}"
        )
    return matches[0]


def save_alignments(
    alignments: Alignments,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(alignments, p, **opts)


def load_alignments(
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> Alignments:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir native lossless form
# ──────────────────────────────────────────────────────────────────────


_PARQUET_TABLES = ("alignments", "tags")


class ParquetDirWriter:
    """Write an ``Alignments`` to a directory of one Parquet per table."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, alignments: Alignments, path: Path, **opts: Any) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(alignments.alignments, path / "alignments.parquet")
        pq.write_table(alignments.tags, path / "tags.parquet")
        manifest: dict[str, Any] = {
            "format": self.format_name,
            "tables": list(_PARQUET_TABLES),
            "container": "Alignments",
            "metadata": alignments.metadata_extras,
        }
        if alignments.acquisitions is not None:
            pq.write_table(
                alignments.acquisitions.table, path / "acquisitions.parquet"
            )
            manifest["has_acquisitions"] = True
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    """Read an ``Alignments`` from a ParquetDir bundle."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Alignments:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        alignments_table = pq.read_table(path / "alignments.parquet")
        tags_table = pq.read_table(path / "tags.parquet")
        acquisitions = None
        if (path / "acquisitions.parquet").exists():
            from constellation.sequencing.acquisitions import Acquisitions

            acquisitions = Acquisitions(pq.read_table(path / "acquisitions.parquet"))
        return Alignments(
            alignments=alignments_table,
            tags=tags_table,
            acquisitions=acquisitions,
            metadata_extras=dict(manifest.get("metadata", {})),
        )


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
