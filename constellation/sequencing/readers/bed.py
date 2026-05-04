"""BED interval reader → ``Annotation`` (FEATURE_TABLE).

BED is the simplest of the genomic-interval formats — 3 to 12 tab-
separated columns, no nested attributes, 0-based half-open coordinates
matching constellation's internal convention. We hand-roll the parser
because (a) it's small, (b) the no-pandas invariant rules out the
pyranges/bedtools-py path, (c) BED columns are positional (no header)
so a Python parser is simpler than Arrow CSV-with-options.

Public surface:

    read_bed(path) -> Annotation
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.schemas.reference import FEATURE_TABLE


def _open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_bed(
    path: str | Path,
    *,
    feature_type: str = "interval",
    contig_name_to_id: dict[str, int] | None = None,
) -> Annotation:
    """Parse a BED (3+ columns) into an ``Annotation``.

    Column mapping:

        col 1 (chrom)    → resolved against ``contig_name_to_id`` (or
                           assigned in encounter order)
        col 2 (start)    → ``start`` (already 0-based half-open)
        col 3 (end)      → ``end``
        col 4 (name)     → ``name``
        col 5 (score)    → ``score``
        col 6 (strand)   → ``strand``

    Columns 7-12 (thickStart/thickEnd/itemRgb/blockCount/blockSizes/
    blockStarts) are stashed into ``attributes_json`` so callers who
    care about exonic block structure can recover it without us
    bloating the canonical FEATURE_TABLE schema.

    ``feature_type`` is the SO/biotype string emitted on every row
    (default ``'interval'``). Override per-call when the input has a
    consistent biotype (``feature_type='enhancer'`` for ENCODE eRNA
    BEDs, etc.).

    BED files have no parent-child relationships; ``parent_id`` is
    always null.
    """
    p = Path(path)

    contig_map: dict[str, int] = (
        dict(contig_name_to_id) if contig_name_to_id else {}
    )
    next_contig_id = max(contig_map.values(), default=-1) + 1

    feature_ids: list[int] = []
    contig_ids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    strands: list[str] = []
    types: list[str] = []
    names: list[str | None] = []
    scores: list[float | None] = []
    attrs_json: list[str | None] = []

    with _open_text(p) as fh:
        next_id = 0
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(("#", "track", "browser")):
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            chrom = cols[0]
            try:
                start_i = int(cols[1])
                end_i = int(cols[2])
            except ValueError:
                continue
            if chrom not in contig_map:
                contig_map[chrom] = next_contig_id
                next_contig_id += 1
            cid = contig_map[chrom]

            name = cols[3] if len(cols) > 3 and cols[3] != "." else None
            score: float | None
            if len(cols) > 4 and cols[4] not in ("", "."):
                try:
                    score = float(cols[4])
                except ValueError:
                    score = None
            else:
                score = None
            strand = cols[5] if len(cols) > 5 and cols[5] in ("+", "-") else "."

            extra: dict[str, Any] = {}
            if len(cols) > 6 and cols[6] not in ("", "."):
                extra["thickStart"] = cols[6]
            if len(cols) > 7 and cols[7] not in ("", "."):
                extra["thickEnd"] = cols[7]
            if len(cols) > 8 and cols[8] not in ("", "."):
                extra["itemRgb"] = cols[8]
            if len(cols) > 9 and cols[9] not in ("", "."):
                extra["blockCount"] = cols[9]
            if len(cols) > 10 and cols[10] not in ("", "."):
                extra["blockSizes"] = cols[10]
            if len(cols) > 11 and cols[11] not in ("", "."):
                extra["blockStarts"] = cols[11]

            feature_ids.append(next_id)
            next_id += 1
            contig_ids.append(cid)
            starts.append(start_i)
            ends.append(end_i)
            strands.append(strand)
            types.append(feature_type)
            names.append(name)
            scores.append(score)
            import json as _json

            attrs_json.append(_json.dumps(extra) if extra else None)

    if not feature_ids:
        table = FEATURE_TABLE.empty_table()
    else:
        table = pa.table(
            {
                "feature_id": pa.array(feature_ids, type=pa.int64()),
                "contig_id": pa.array(contig_ids, type=pa.int64()),
                "start": pa.array(starts, type=pa.int64()),
                "end": pa.array(ends, type=pa.int64()),
                "strand": pa.array(strands, type=pa.string()),
                "type": pa.array(types, type=pa.string()),
                "name": pa.array(names, type=pa.string()),
                "parent_id": pa.array([None] * len(feature_ids), type=pa.int64()),
                "source": pa.array([None] * len(feature_ids), type=pa.string()),
                "score": pa.array(scores, type=pa.float32()),
                "phase": pa.array([None] * len(feature_ids), type=pa.int32()),
                "attributes_json": pa.array(attrs_json, type=pa.string()),
            }
        )

    metadata: dict[str, Any] = {
        "source_path": str(p),
        "contig_name_to_id": dict(contig_map),
        "feature_type_default": feature_type,
    }
    return Annotation(features=table, metadata_extras=metadata)


def iter_bed_lines(path: str | Path) -> Iterator[str]:
    """Yield non-comment, non-track BED data lines."""
    p = Path(path)
    with _open_text(p) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith(("#", "track", "browser")):
                continue
            yield line


__all__ = ["read_bed", "iter_bed_lines"]
