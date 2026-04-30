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
        Output of mmseqs-style kmer clustering with abundance-weighted
        consensus building. One row per cluster with the consensus
        sequence and predicted ORF / protein.

These schemas are sequencing-specific (no ≥2-modality bar to clear) so
they live here rather than in core. All three self-register so cast /
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
        pa.field("sample_id", pa.int64(), nullable=False),
        # 0-based half-open offsets into the read for the transcript window
        pa.field("transcript_start", pa.int32(), nullable=False),
        pa.field("transcript_end", pa.int32(), nullable=False),
        # Aggregated demux confidence (0..1)
        pa.field("score", pa.float32(), nullable=True),
        # True if the parent read was a concatemer / chimera with ≥2
        # transcript segments
        pa.field("is_chimera", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_name": b"ReadDemuxTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Consensus clusters (output of `transcriptome.cluster` + `consensus`)
# ──────────────────────────────────────────────────────────────────────


TRANSCRIPT_CLUSTER_TABLE: pa.Schema = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        # Reference / centroid read for the cluster (highest-quality member)
        pa.field("representative_read_id", pa.string(), nullable=False),
        pa.field("n_reads", pa.int32(), nullable=False),
        # mmseqs-style sequence identity threshold the cluster was
        # formed at (e.g. 0.85)
        pa.field("identity_threshold", pa.float32(), nullable=True),
        # Consensus nucleotide sequence (post-error-correction)
        pa.field("consensus_sequence", pa.string(), nullable=False),
        # Best ORF detected on the consensus; null if no ORF passes
        # length / start-codon filters. Predicted protein in IUPAC
        # one-letter
        pa.field("predicted_protein", pa.string(), nullable=True),
        pa.field("orf_start", pa.int32(), nullable=True),
        pa.field("orf_end", pa.int32(), nullable=True),
        # '+' | '-' relative to the consensus_sequence
        pa.field("orf_strand", pa.string(), nullable=True),
        # 1 (Standard) | 2 (Vertebrate Mito) | ... — NCBI transl_table
        pa.field("codon_table", pa.int32(), nullable=True),
    ],
    metadata={b"schema_name": b"TranscriptClusterTable"},
)


register_schema("ReadSegmentTable", READ_SEGMENT_TABLE)
register_schema("ReadDemuxTable", READ_DEMUX_TABLE)
register_schema("TranscriptClusterTable", TRANSCRIPT_CLUSTER_TABLE)


__all__ = [
    "READ_SEGMENT_TABLE",
    "READ_DEMUX_TABLE",
    "TRANSCRIPT_CLUSTER_TABLE",
]
