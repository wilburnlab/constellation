"""Sequencing raw-data readers — ``RawReader`` subclasses, self-register.

Importing this package triggers ``@register_reader`` on each submodule
so suffix-based lookup via :func:`core.io.readers.find_reader` finds
the right class. Modality tag is ``"nanopore"`` for POD5 / FASTX /
SAM-BAM / PAF inputs.

Modules:

    pod5         POD5 raw signal → RAW_SIGNAL_TABLE
    fastx        FASTA / FASTQ (gzip-aware) → READ_TABLE; also
                 ``read_fasta_genome`` / ``read_fasta_transcriptome``
                 / ``read_fasta_genetic_tools`` for reference-side use
    sam_bam      SAM / BAM → READ_TABLE + ALIGNMENT_TABLE
    paf          PAF (minimap2) → ALIGNMENT_TABLE (no separate PAF schema)
    gff          GFF3 → ``read_gff3`` / ``write_gff3`` (Annotation)
                 plus stubbed RawReader subclasses for the read-ingest
                 path
    bed          BED intervals → ``read_bed`` (Annotation)
"""

from __future__ import annotations

# Submodule imports trigger @register_reader decorators + define the
# reference-side read_fasta_* / read_gff3 / read_bed helpers.
from constellation.sequencing.readers import (  # noqa: F401
    bed,
    fastx,
    gff,
    paf,
    pod5,
    sam_bam,
)

__all__ = [
    "bed",
    "fastx",
    "gff",
    "paf",
    "pod5",
    "sam_bam",
]
