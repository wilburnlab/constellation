"""Alignments container — per-acquisition mapped reads.

``Alignments`` is the empirical, per-acquisition record of reads
mapped to a ``Reference``. FK closure: ``read_id`` into a ``Reads``
table, ``acquisition_id`` into ``Acquisitions``, ``ref_name`` into
``Reference.contigs.name``. Polished, scaffolded, and external-ref
workflows all produce Alignments in the same shape.

The on-disk authoritative form for alignments is BAM (the readers /
writers in :mod:`sequencing.io.sam_bam` go between BAM and the
``ALIGNMENT_TABLE`` + ``ALIGNMENT_TAG_TABLE`` pair). ParquetDir is the
native lossless Constellation form for cached / processed alignments.
"""

from __future__ import annotations

from constellation.sequencing.alignments.alignments import Alignments
from constellation.sequencing.alignments.io import (
    ALIGNMENTS_READERS,
    ALIGNMENTS_WRITERS,
    AlignmentsReader,
    AlignmentsWriter,
    load_alignments,
    register_reader,
    register_writer,
    save_alignments,
)

__all__ = [
    "Alignments",
    "AlignmentsReader",
    "AlignmentsWriter",
    "ALIGNMENTS_READERS",
    "ALIGNMENTS_WRITERS",
    "register_reader",
    "register_writer",
    "save_alignments",
    "load_alignments",
]
