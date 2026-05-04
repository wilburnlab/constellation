"""BAM cross-tier adapter — single-shot decode into ``Alignments``.

The container-shaped entry point sits on top of the format-level
decoder in :mod:`sequencing.readers.sam_bam`. ``read_bam`` materialises
the full table in memory; appropriate for tests and small Jupyter use
but not the pipeline at 30–200M alignments — production paths use the
chunked decoder + fused worker in :mod:`sequencing.quant.genome_count`.

Self-registers ``_BamAlignmentsReader`` with ``ALIGNMENTS_READERS`` on
import so ``load_alignments(path.bam)`` resolves cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from constellation.sequencing.alignments.alignments import Alignments
from constellation.sequencing.alignments.io import (
    register_reader as register_alignments_reader,
)
from constellation.sequencing.readers.sam_bam import read_bam_alignments
from constellation.sequencing.reference.reference import GenomeReference


def read_bam(
    path: Path,
    *,
    genome: GenomeReference | None = None,
    acquisition_id: int = 0,
    tags_to_keep: tuple[str, ...] = (),
    threads: int = 1,
) -> Alignments:
    """Decode a BAM file into an ``Alignments`` container.

    When ``genome`` is provided, ``Alignments.validate_against`` is
    invoked so any rogue ``ref_name`` (BAM produced against a different
    reference) trips before downstream code sees the data. Without a
    genome, validation is limited to PK/FK closure on the alignments
    themselves.

    In-memory decode — at 30–200M alignments this OOMs. Pipeline paths
    use the chunked decoder in
    :mod:`sequencing.readers.sam_bam.read_bam_alignments_chunk`.
    """
    alignments_table, tags_table = read_bam_alignments(
        Path(path),
        acquisition_id=acquisition_id,
        tags_to_keep=tags_to_keep,
        threads=threads,
    )
    container = Alignments(alignments=alignments_table, tags=tags_table)
    if genome is not None:
        container.validate_against(genome)
    return container


def write_bam(path: Path, alignments: Alignments, **opts: Any) -> None:
    """Encode an Alignments container back to BAM.

    Deferred — no current consumer requires constellation-produced
    BAMs (we always emit ParquetDir for cached forms; minimap2 produces
    BAM directly). Lift when a use case appears.
    """
    raise NotImplementedError("write_bam not yet implemented")


# ──────────────────────────────────────────────────────────────────────
# Container-Reader adapters (slot the shared decode into each registry)
# ──────────────────────────────────────────────────────────────────────


class _BamAlignmentsReader:
    """Slots BAM into the ALIGNMENTS_READERS registry."""

    extension: ClassVar[str] = ".bam"
    format_name: ClassVar[str] = "bam"

    def read(self, path: Path, **opts: Any) -> Alignments:
        return read_bam(path, **opts)


# Register on import so consumers see ``.bam`` resolution by default.
register_alignments_reader(_BamAlignmentsReader())


__all__ = ["read_bam", "write_bam"]
