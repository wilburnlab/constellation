"""Sequencing-domain Arrow schemas.

Each schema self-registers with :mod:`core.io.schemas` on submodule
import (so ``from constellation.sequencing import schemas`` is enough
to populate the registry — no explicit registration step needed).

Generic shapes (``RaggedTrace1D``, ``Trace1D``, ``PeakTable``) live in
:mod:`core.io.schemas`; sequencing-specific shapes live here. The split
follows the architecture invariant: a schema earns a slot in core only
if its column layout is universal across ≥2 modalities.

Notable instances:

    READ_TABLE              basecalled-read records (FASTA/FASTQ-shaped)
    ALIGNMENT_TABLE         BAM-canonical alignment columns
    ALIGNMENT_TAG_TABLE     long-form for the rare-tag tail
    ALIGNMENT_BLOCK_TABLE   per-CIGAR-block exon view (Phase 1)
    INTRON_TABLE            denormalised intron view: one row per
                            observed (donor, acceptor) pair with
                            cluster-membership info (Phase 1)
    READ_FINGERPRINT_TABLE  per-read canonical junction hash (Phase 1)
    RAW_SIGNAL_TABLE        POD5 raw signal — uses core.ragged_trace_1d
    CONTIG_TABLE            reference / assembly contigs
    FEATURE_TABLE           GFF-shaped features (genes, exons, repeats)
    SEQUENCE_TABLE          per-contig sequence bytes
    TRANSCRIPT_TABLE        spliced transcript records (id, gene_id?,
                            sequence) — the TranscriptReference shape
    GENETIC_TOOL_TABLE      common cloning/engineering parts (AbR,
                            FPs, tags, promoters, vector backbones —
                            the cRAP analog for genomics)
    ASSEMBLY_CONTIG_TABLE   de novo contigs with polishing history
    SCAFFOLD_TABLE          RagTag-style scaffolds over contigs
    ASSEMBLY_STATS          N50, BUSCO completeness, GC content
    FEATURE_QUANT           per-feature counts/coverage/TPM keyed on
                            (feature_id, sample_id)
    COVERAGE_TABLE          per-position coverage tracks
    READ_SEGMENT_TABLE      one row per located segment (adapter,
                            polyA, barcode, transcript) — chimeras
                            emerge as multiple rows per read
    READ_DEMUX_TABLE        derived: one row per resolved transcript
                            window with sample_id assigned
    TRANSCRIPT_CLUSTER_TABLE  consensus transcripts with predicted ORF
"""

from __future__ import annotations

from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
    ALIGNMENT_TAG_TABLE,
    INTRON_TABLE,
    READ_FINGERPRINT_TABLE,
    cigar_to_ops,
)
from constellation.sequencing.schemas.assembly import (
    ASSEMBLY_CONTIG_TABLE,
    ASSEMBLY_STATS,
    SCAFFOLD_TABLE,
)
from constellation.sequencing.schemas.quant import (
    COVERAGE_TABLE,
    FEATURE_ORIGIN_VOCAB,
    FEATURE_QUANT,
)
from constellation.sequencing.schemas.reads import READ_TABLE
from constellation.sequencing.schemas.reference import (
    CONTIG_TABLE,
    FEATURE_TABLE,
    GENETIC_TOOL_CATEGORIES,
    GENETIC_TOOL_SEQUENCE_TYPES,
    GENETIC_TOOL_TABLE,
    SEQUENCE_TABLE,
    TRANSCRIPT_TABLE,
)
from constellation.sequencing.schemas.signal import RAW_SIGNAL_TABLE
from constellation.sequencing.schemas.transcriptome import (
    READ_DEMUX_TABLE,
    READ_SEGMENT_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)

__all__ = [
    "READ_TABLE",
    "ALIGNMENT_TABLE",
    "ALIGNMENT_TAG_TABLE",
    "ALIGNMENT_BLOCK_TABLE",
    "INTRON_TABLE",
    "READ_FINGERPRINT_TABLE",
    "RAW_SIGNAL_TABLE",
    "CONTIG_TABLE",
    "FEATURE_TABLE",
    "SEQUENCE_TABLE",
    "TRANSCRIPT_TABLE",
    "GENETIC_TOOL_TABLE",
    "GENETIC_TOOL_CATEGORIES",
    "GENETIC_TOOL_SEQUENCE_TYPES",
    "ASSEMBLY_CONTIG_TABLE",
    "SCAFFOLD_TABLE",
    "ASSEMBLY_STATS",
    "FEATURE_QUANT",
    "FEATURE_ORIGIN_VOCAB",
    "COVERAGE_TABLE",
    "READ_SEGMENT_TABLE",
    "READ_DEMUX_TABLE",
    "TRANSCRIPT_CLUSTER_TABLE",
    "cigar_to_ops",
]
