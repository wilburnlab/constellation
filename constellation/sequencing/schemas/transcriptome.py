"""Transcriptome-pipeline schemas — segments, demux, clusters.

Three tables track the cDNA / direct-RNA workflow from raw reads to
consensus transcripts:

    READ_SEGMENT_TABLE
        One row per *located segment* within a read — adapter, polyA
        anchor, barcode, transcript, UMI. Chimeras / concatemers (very
        common in nanopore) emerge as multiple segment-runs per read
        without forcing a chimera-aware schema upstream of demux.

    READ_DEMUX_TABLE
        Derived from READ_SEGMENT_TABLE by chimera resolution: one row
        per resolved transcript window with sample_id assigned. This
        is what clustering / consensus / ORF prediction consumes.

    TRANSCRIPT_CLUSTER_TABLE
        Output of clustering — both Phase 2 genome-guided
        (fingerprint-keyed, splicing-topology-resolved) and Phase 3
        de novo (kmer-minimizer + abundance-weighted greedy
        set-cover) populate the same row shape. The ``mode``
        discriminator says which produced a row; mode-specific
        columns are nullable on the other mode's rows.

    CLUSTER_MEMBERSHIP_TABLE
        Long-form (cluster_id, read_id, role, score-columns) — one
        row per (cluster, member-read) edge. Role-tagged so callers
        can distinguish the representative read, ordinary members,
        Layer-0-deduplicated duplicates, and drift-filtered reads
        retained for cross-validation.

These schemas are sequencing-specific (no ≥2-modality bar to clear) so
they live here rather than in core. All four self-register so cast /
metadata helpers in core.io.schemas work uniformly.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# ──────────────────────────────────────────────────────────────────────
# Per-read segmentation (output of `transcriptome.demux.locate_segments`)
# ──────────────────────────────────────────────────────────────────────


READ_SEGMENT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        # 0-indexed, ordered 5' → 3' along the read
        pa.field("segment_index", pa.int32(), nullable=False),
        # 'adapter_5p' | 'adapter_3p' | 'polyA' | 'barcode' |
        # 'transcript' | 'umi' | 'unknown'
        pa.field("segment_kind", pa.string(), nullable=False),
        # 0-based half-open offsets into the read sequence
        pa.field("start", pa.int32(), nullable=False),
        pa.field("end", pa.int32(), nullable=False),
        # Edlib edit-distance score; low = good match. -1 = unscored
        # (e.g. for transcript / umi segments that aren't aligned
        # against a reference)
        pa.field("score", pa.int32(), nullable=True),
        # Best-vs-second-best score gap. Populated on barcode rows
        # (barcode panel ranks ≥2 candidates by edit distance — the gap
        # between best and second-best disambiguates close-tie matches);
        # null on adapter / polyA / transcript rows that don't have a
        # competing alternative. Calibration input for the future
        # ProbabilisticScorer's barcode prior.
        pa.field("score_delta", pa.int32(), nullable=True),
        # FK into the barcode registry; null when segment_kind != 'barcode'
        pa.field("barcode_id", pa.int64(), nullable=True),
        # '+' = read as-stored matched the construct; '-' = the read
        # was reverse-complemented to match
        pa.field("orientation", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"ReadSegmentTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Resolved per-transcript records (output of `transcriptome.demux.resolve_demux`)
# ──────────────────────────────────────────────────────────────────────


READ_DEMUX_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        # Multiple rows for a single read_id when the read is a
        # concatemer; segment_index distinguishes them
        pa.field("transcript_segment_index", pa.int32(), nullable=False),
        # Nullable: the read may have no resolvable sample assignment
        # (no barcode found, or barcode below acceptance threshold).
        # We still emit a row so downstream code can audit the un-
        # assigned reads; null sample_id is the explicit "Source_None"
        # case in NanoporeAnalysis's per-read output.
        pa.field("sample_id", pa.int64(), nullable=True),
        # 0-based half-open offsets into the read for the transcript window
        pa.field("transcript_start", pa.int32(), nullable=False),
        pa.field("transcript_end", pa.int32(), nullable=False),
        # Aggregated demux confidence (0..1)
        pa.field("score", pa.float32(), nullable=True),
        # True if the parent read was a concatemer / chimera with ≥2
        # transcript segments
        pa.field("is_chimera", pa.bool_(), nullable=False),
        # Read-level classification (NanoporeAnalysis parity values:
        # 'Complete', "3' Only", "5' Only", 'Missing Barcode',
        # 'Unknown', 'Complex'; later sessions add 'Palindromic',
        # 'Truncated', 'TerminalDropout' as artifact detection lands).
        # Stored as string for stable parquet round-trips and to mirror
        # NanoporeAnalysis's `cDNA status` column literally; the Python-
        # side `ReadStatus(StrEnum)` in transcriptome.classify gives type
        # safety where it matters.
        pa.field("status", pa.string(), nullable=False),
        # True when the resolved transcript window is shorter than the
        # min_transcript_len threshold — broken out as a separate column
        # rather than munged onto status (NanoporeAnalysis appended a
        # " Fragment" suffix to the string).
        pa.field("is_fragment", pa.bool_(), nullable=False),
        # Artifact tag (Session 3 fills with 'palindromic' / 'chimera' /
        # 'terminal_dropout_5p' / 'terminal_dropout_3p'; populated 'none'
        # in Session 1 so the column is real from day one).
        pa.field("artifact", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"ReadDemuxTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Consensus clusters (output of `transcriptome.cluster_genome` (Phase 2)
# and `transcriptome.cluster_denovo` (Phase 3))
# ──────────────────────────────────────────────────────────────────────


# Mode-agnostic cluster row. Phase 2 (genome-guided) populates the
# splicing-topology columns (contig_id, strand, span_start/end,
# fingerprint_hash) and leaves consensus / ORF columns null unless
# `--build-consensus` is set. Phase 3 (de novo) populates
# identity_threshold + consensus_sequence + ORF fields and leaves the
# splicing-topology columns null. The ``mode`` enum is the row-level
# discriminator; downstream consumers filter on it.
TRANSCRIPT_CLUSTER_TABLE: pa.Schema = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        # Reference read for the cluster — most-abundant unique trimmed
        # transcript window (Layer-0 derep), tiebreak by mean
        # dorado_quality. Always populated.
        pa.field("representative_read_id", pa.string(), nullable=False),
        pa.field("n_reads", pa.int32(), nullable=False),
        # mmseqs-style sequence identity threshold the cluster was
        # formed at (e.g. 0.85). Phase 3 only.
        pa.field("identity_threshold", pa.float32(), nullable=True),
        # Consensus nucleotide sequence (post-error-correction). Phase 3
        # always populates; Phase 2 populates only when
        # `--build-consensus` is set.
        pa.field("consensus_sequence", pa.string(), nullable=True),
        # Best ORF detected on the consensus; null if no ORF passes
        # length / start-codon filters, or when consensus_sequence is
        # null. Predicted protein in IUPAC one-letter.
        pa.field("predicted_protein", pa.string(), nullable=True),
        pa.field("orf_start", pa.int32(), nullable=True),
        pa.field("orf_end", pa.int32(), nullable=True),
        # '+' | '-' relative to the consensus_sequence
        pa.field("orf_strand", pa.string(), nullable=True),
        # 1 (Standard) | 2 (Vertebrate Mito) | ... — NCBI transl_table
        pa.field("codon_table", pa.int32(), nullable=True),
        # 'genome-guided' | 'de-novo'
        pa.field("mode", pa.string(), nullable=False),
        # Splicing-topology columns — Phase 2 (genome-guided) only.
        pa.field("contig_id", pa.int64(), nullable=True),
        pa.field("strand", pa.string(), nullable=True),
        # Span on the genome coord frame; first block.ref_start ..
        # last block.ref_end across the cluster's representative read.
        pa.field("span_start", pa.int64(), nullable=True),
        pa.field("span_end", pa.int64(), nullable=True),
        # The cluster key in genome-guided mode. FK→READ_FINGERPRINT_TABLE.
        pa.field("fingerprint_hash", pa.uint64(), nullable=True),
        # Layer-0 dedup depth — number of distinct trimmed transcript
        # windows observed across the cluster's members. 1 means the
        # cluster collapses to a single unique sequence; >1 reflects
        # within-cluster substitution noise. Always populated.
        pa.field("n_unique_sequences", pa.int32(), nullable=False),
        # Populated only when `--per-sample-clusters` opts into
        # per-sample cluster assignment. Null = cluster spans samples.
        pa.field("sample_id", pa.int64(), nullable=True),
    ],
    metadata={b"schema_name": b"TranscriptClusterTable"},
)


# Long-form (cluster_id, read_id) edges with role + multi-parameter
# fit-to-cluster diagnostics. Drift_5p_bp / drift_3p_bp are signed
# bp distances from the cluster's median 5'/3' span (strand-aware:
# on +, 5' = span_start; on -, 5' = span_end). match_rate / indel_rate
# are derived from cs:long-populated n_match/n_mismatch/n_insert/
# n_delete in ALIGNMENT_BLOCK_TABLE — null when cs:long was not
# captured upstream. Combined, downstream consumers can build any
# weighted goodness-of-fit function (e.g. score = 0.4*match_rate +
# 0.3*(1 - drift_5p/max_5p) + 0.3*(1 - drift_3p/max_3p)) without
# recomputing from raw alignments.
CLUSTER_MEMBERSHIP_TABLE: pa.Schema = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        pa.field("read_id", pa.string(), nullable=False),
        # 'representative' | 'member' | 'duplicate' | 'drift_filtered'
        # 'duplicate' = Layer-0-deduplicated to the same trimmed window
        # as another member; 'drift_filtered' = retained but dropped
        # from cluster summary statistics by the 5'/3' drift filter.
        pa.field("role", pa.string(), nullable=False),
        # Signed bp distance from cluster median 5'/3' span; null on
        # representative / duplicate / singleton rows where there's no
        # comparison cohort.
        pa.field("drift_5p_bp", pa.int32(), nullable=True),
        pa.field("drift_3p_bp", pa.int32(), nullable=True),
        # n_match / (n_match + n_mismatch); null when cs:long absent.
        pa.field("match_rate", pa.float32(), nullable=True),
        # (n_insert + n_delete) / (n_match + n_mismatch + n_insert +
        # n_delete); null when cs:long absent.
        pa.field("indel_rate", pa.float32(), nullable=True),
        # n_match + n_mismatch + n_insert + n_delete summed across the
        # primary alignment's blocks. Always populated (CIGAR-derived).
        pa.field("n_aligned_bp", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"ClusterMembershipTable"},
)


register_schema("ReadSegmentTable", READ_SEGMENT_TABLE)
register_schema("ReadDemuxTable", READ_DEMUX_TABLE)
register_schema("TranscriptClusterTable", TRANSCRIPT_CLUSTER_TABLE)
register_schema("ClusterMembershipTable", CLUSTER_MEMBERSHIP_TABLE)


__all__ = [
    "READ_SEGMENT_TABLE",
    "READ_DEMUX_TABLE",
    "TRANSCRIPT_CLUSTER_TABLE",
    "CLUSTER_MEMBERSHIP_TABLE",
]
