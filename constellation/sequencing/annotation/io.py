"""``Annotation`` Reader/Writer Protocols + native ParquetDir round-trip.

Mirrors the shape of :mod:`sequencing.reference.io`. External-format
adapters (GFF3, BED) self-register from their reader modules under
:mod:`sequencing.readers`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pyarrow.parquet as pq

from constellation.sequencing.annotation.annotation import Annotation

# ──────────────────────────────────────────────────────────────────────
# Protocols + registries
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class AnnotationWriter(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]
    lossy: ClassVar[bool]

    def write(self, annotation: Annotation, path: Path, **opts: Any) -> None: ...


@runtime_checkable
class AnnotationReader(Protocol):
    extension: ClassVar[str]
    format_name: ClassVar[str]

    def read(self, path: Path, **opts: Any) -> Annotation: ...


ANNOTATION_WRITERS: dict[str, AnnotationWriter] = {}
ANNOTATION_READERS: dict[str, AnnotationReader] = {}


def register_writer(writer: AnnotationWriter) -> None:
    if writer.format_name in ANNOTATION_WRITERS:
        raise ValueError(f"writer already registered: {writer.format_name!r}")
    ANNOTATION_WRITERS[writer.format_name] = writer


def register_reader(reader: AnnotationReader) -> None:
    if reader.format_name in ANNOTATION_READERS:
        raise ValueError(f"reader already registered: {reader.format_name!r}")
    ANNOTATION_READERS[reader.format_name] = reader


def _resolve_writer(format: str | None, path: Path) -> AnnotationWriter:
    if format is not None:
        if format not in ANNOTATION_WRITERS:
            raise KeyError(f"no writer registered for format {format!r}")
        return ANNOTATION_WRITERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [w for w in ANNOTATION_WRITERS.values() if w.extension == suffix]
    if not matches:
        raise KeyError(
            f"no writer for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(ANNOTATION_WRITERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple writers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[w.format_name for w in matches]}"
        )
    return matches[0]


def _resolve_reader(format: str | None, path: Path) -> AnnotationReader:
    if format is not None:
        if format not in ANNOTATION_READERS:
            raise KeyError(f"no reader registered for format {format!r}")
        return ANNOTATION_READERS[format]
    suffix = path.suffix.lower() or "/"
    matches = [r for r in ANNOTATION_READERS.values() if r.extension == suffix]
    if not matches:
        raise KeyError(
            f"no reader for path {path!s}; "
            f"specify format= explicitly. registered: {sorted(ANNOTATION_READERS)}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"multiple readers claim suffix {suffix!r}; "
            f"specify format= explicitly: {[r.format_name for r in matches]}"
        )
    return matches[0]


def save_annotation(
    annotation: Annotation,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    p = Path(path)
    writer = _resolve_writer(format, p)
    writer.write(annotation, p, **opts)


def load_annotation(
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> Annotation:
    p = Path(path)
    reader = _resolve_reader(format, p)
    return reader.read(p, **opts)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir native lossless form
# ──────────────────────────────────────────────────────────────────────


class ParquetDirWriter:
    """Write an ``Annotation`` to a directory containing a single
    Parquet file plus a ``manifest.json``."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"
    lossy: ClassVar[bool] = False

    def write(self, annotation: Annotation, path: Path, **opts: Any) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(annotation.features, path / "features.parquet")
        manifest = {
            "format": self.format_name,
            "tables": ["features"],
            "container": "Annotation",
            "metadata": annotation.metadata_extras,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


class ParquetDirReader:
    """Read an ``Annotation`` from a ParquetDir bundle."""

    extension: ClassVar[str] = "/"
    format_name: ClassVar[str] = "parquet_dir"

    def read(self, path: Path, **opts: Any) -> Annotation:
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        features = pq.read_table(path / "features.parquet")
        return Annotation(
            features=features,
            metadata_extras=dict(manifest.get("metadata", {})),
        )


register_writer(ParquetDirWriter())
register_reader(ParquetDirReader())


__all__ = [
    "AnnotationReader",
    "AnnotationWriter",
    "ANNOTATION_READERS",
    "ANNOTATION_WRITERS",
    "register_reader",
    "register_writer",
    "save_annotation",
    "load_annotation",
    "ParquetDirReader",
    "ParquetDirWriter",
]
