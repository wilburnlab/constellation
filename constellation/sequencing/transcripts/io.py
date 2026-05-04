"""``TranscriptReference`` Reader/Writer Protocols + native ParquetDir
round-trip.

Mirrors the shape of :mod:`sequencing.reference.io`. External-format
adapters (transcript FASTA) self-register from their reader modules
under :mod:`sequencing.readers`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow.parquet as pq

from constellation.sequencing.transcripts.transcripts import TranscriptReference

# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class TranscriptReferenceWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(
        self, transcripts: TranscriptReference, path: Path, **opts: Any
    ) -> None: ...


@runtime_checkable
class TranscriptReferenceReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> TranscriptReference: ...


TRANSCRIPT_REFERENCE_WRITERS: dict[str, TranscriptReferenceWriter] = {}
TRANSCRIPT_REFERENCE_READERS: dict[str, TranscriptReferenceReader] = {}


def register_writer(writer: TranscriptReferenceWriter) -> None:
    if writer.format_name in TRANSCRIPT_REFERENCE_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    TRANSCRIPT_REFERENCE_WRITERS[writer.format_name] = writer


def register_reader(reader: TranscriptReferenceReader) -> None:
    if reader.format_name in TRANSCRIPT_REFERENCE_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    TRANSCRIPT_REFERENCE_READERS[reader.format_name] = reader


def _resolve_writer(format: str | None, path: Path) -> TranscriptReferenceWriter:
    if format is not None:
        if format not in TRANSCRIPT_REFERENCE_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return TRANSCRIPT_REFERENCE_WRITERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [
        w for w in TRANSCRIPT_REFERENCE_WRITERS.values() if w.extension == suffix
    ]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; specify format= explicitly. "
            f"registered: {sorted(TRANSCRIPT_REFERENCE_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}; specify format= "
            f"explicitly: {[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> TranscriptReferenceReader:
    if format is not None:
        if format not in TRANSCRIPT_REFERENCE_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return TRANSCRIPT_REFERENCE_READERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [
        r for r in TRANSCRIPT_REFERENCE_READERS.values() if r.extension == suffix
    ]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; specify format= explicitly. "
            f"registered: {sorted(TRANSCRIPT_REFERENCE_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}; specify format= "
            f"explicitly: {[r.format_name for r in matches]}"
        )
    return matches[0]


def save_transcript_reference(
    transcripts: TranscriptReference,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(transcripts, p, **opts)


def load_transcript_reference(
    path: str | Path, *, format: str | None = None, **opts: Any
) -> TranscriptReference:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir native lossless form
# ──────────────────────────────────────────────────────────────────────


class ParquetDirWriter:
    """Write a ``TranscriptReference`` to a directory containing a
    single Parquet file plus a ``manifest.json``."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(
        self, transcripts: TranscriptReference, path: Path, **opts: Any
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(transcripts.transcripts, path / "transcripts.parquet")
        manifest = {
            "format": self.format_name,
            "tables": ["transcripts"],
            "container": "TranscriptReference",
            "metadata": transcripts.metadata_extras,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    """Read a ``TranscriptReference`` from a ParquetDir bundle."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> TranscriptReference:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        transcripts = pq.read_table(path / "transcripts.parquet")
        return TranscriptReference(
            transcripts=transcripts,
            metadata_extras=dict(manifest.get("metadata", {})),
        )


register_writer(ParquetDirWriter())
register_reader(ParquetDirReader())


__all__ = [
    "TranscriptReferenceReader",
    "TranscriptReferenceWriter",
    "TRANSCRIPT_REFERENCE_READERS",
    "TRANSCRIPT_REFERENCE_WRITERS",
    "register_reader",
    "register_writer",
    "save_transcript_reference",
    "load_transcript_reference",
    "ParquetDirReader",
    "ParquetDirWriter",
]
