"""Long-read mapping — minimap2 (mappy) the workhorse.

Maps many reads against a reference. Two backends:

    - ``mappy`` (Python bindings): in-process, no subprocess overhead,
      best for small-to-medium read sets and Jupyter exploration.
    - ``minimap2`` subprocess: required for full feature coverage
      (some flags only land on the CLI), better for batch pipelines.

Both produce ``ALIGNMENT_TABLE``-shaped Arrow output. Splice-aware
mapping for cDNA → genome uses preset ``'splice'`` (mappy) /
``-ax splice`` (subprocess).

Status: STUB. Pending Phase 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pyarrow as pa


_PHASE = "Phase 4 (align/map + thirdparty/minimap2)"


def map_reads(
    reads: pa.Table,             # READ_TABLE
    reference: Path,             # FASTA
    *,
    backend: Literal["mappy", "minimap2"] = "mappy",
    preset: Literal["map-ont", "splice", "asm5", "asm10", "asm20"] = "map-ont",
    threads: int = 8,
    extra_args: tuple[str, ...] = (),
) -> pa.Table:                   # ALIGNMENT_TABLE
    """Map a READ_TABLE against a reference FASTA → ALIGNMENT_TABLE.

    ``preset`` selects the minimap2 preset:
        map-ont    long-read genomic DNA (default for nanopore)
        splice     splice-aware cDNA → genome mapping
        asm5/10/20 assembly-vs-assembly comparisons

    Note: caller assigns ``acquisition_id`` / ``alignment_id`` —
    map_reads doesn't know which acquisition the reads came from.
    """
    raise NotImplementedError(f"map_reads pending {_PHASE}")


def map_assembly(
    query_fasta: Path,
    target_fasta: Path,
    *,
    backend: Literal["mappy", "minimap2"] = "minimap2",
    preset: Literal["asm5", "asm10", "asm20"] = "asm5",
    threads: int = 8,
) -> pa.Table:
    """Map one assembly against another (RagTag-style scaffold-prep,
    cross-genome comparison).
    """
    raise NotImplementedError(f"map_assembly pending {_PHASE}")


__all__ = [
    "map_reads",
    "map_assembly",
]
