"""``GenomeReference`` container — contigs + sequences.

The genome-level reference is the sequencing analog to
``massspec.library.Library``: PK uniqueness, FK closure across contigs
and sequences, and a ParquetDir lossless native form. Two origins flow
into this shape:

    External reference   FASTA → GenomeReference (paired with an
                         Annotation built from companion GFF3)
    De novo assembly     ``sequencing.assembly.Assembly.to_genome_reference()``
                         lifts a finalized assembly into the same shape
                         so downstream alignment / quant code is uniform
                         across both workflows

Annotation features (genes, exons, repeats) live in
:class:`sequencing.annotation.Annotation`, which references this
container's contigs by id. Spliced transcript records live in
:class:`sequencing.transcripts.TranscriptReference`.
"""

from __future__ import annotations

from constellation.sequencing.reference.io import (
    GENOME_REFERENCE_READERS,
    GENOME_REFERENCE_WRITERS,
    GenomeReferenceReader,
    GenomeReferenceWriter,
    load_genome_reference,
    register_reader,
    register_writer,
    save_genome_reference,
)
from constellation.sequencing.reference.reference import GenomeReference

__all__ = [
    "GenomeReference",
    "GenomeReferenceReader",
    "GenomeReferenceWriter",
    "GENOME_REFERENCE_READERS",
    "GENOME_REFERENCE_WRITERS",
    "register_reader",
    "register_writer",
    "save_genome_reference",
    "load_genome_reference",
]
