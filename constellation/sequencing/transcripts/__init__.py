"""``TranscriptReference`` — spliced transcript records.

Sibling to :class:`sequencing.reference.GenomeReference` and
:class:`sequencing.annotation.Annotation`. Two origins:

    Direct FASTA import      A standalone transcript FASTA →
                             ``read_fasta_transcriptome``
    Derived from annotation  ``TranscriptReference.from_annotation``
                             splices contig sequences against
                             ``mRNA``/``exon`` features

A TranscriptReference is sample-agnostic; per-transcript counts ride
under ``feature_origin='transcript_id'`` rows in ``FEATURE_QUANT``.
"""

from __future__ import annotations

from constellation.sequencing.transcripts.io import (
    TRANSCRIPT_REFERENCE_READERS,
    TRANSCRIPT_REFERENCE_WRITERS,
    TranscriptReferenceReader,
    TranscriptReferenceWriter,
    load_transcript_reference,
    register_reader,
    register_writer,
    save_transcript_reference,
)
from constellation.sequencing.transcripts.transcripts import TranscriptReference

__all__ = [
    "TranscriptReference",
    "TranscriptReferenceReader",
    "TranscriptReferenceWriter",
    "TRANSCRIPT_REFERENCE_READERS",
    "TRANSCRIPT_REFERENCE_WRITERS",
    "register_reader",
    "register_writer",
    "save_transcript_reference",
    "load_transcript_reference",
]
