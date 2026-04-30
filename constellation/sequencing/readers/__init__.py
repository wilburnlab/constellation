"""Sequencing raw-data readers — ``RawReader`` subclasses, self-register.

Importing this package triggers ``@register_reader`` on each submodule
so suffix-based lookup via :func:`core.io.readers.find_reader` finds
the right class. Modality tag is ``"nanopore"`` for POD5 / FASTX /
SAM-BAM / PAF inputs.

Modules:

    pod5         POD5 raw signal → RAW_SIGNAL_TABLE
    fastx        FASTA / FASTQ (gzip-aware) → READ_TABLE
    sam_bam      SAM / BAM → READ_TABLE + ALIGNMENT_TABLE
    paf          PAF (minimap2) → ALIGNMENT_TABLE (no separate PAF schema)
    gff          GFF3 / GTF → FEATURE_TABLE
"""

from __future__ import annotations

# Submodule imports trigger @register_reader decorators
from constellation.sequencing.readers import (  # noqa: F401
    fastx,
    gff,
    paf,
    pod5,
    sam_bam,
)

__all__ = [
    "fastx",
    "gff",
    "paf",
    "pod5",
    "sam_bam",
]
