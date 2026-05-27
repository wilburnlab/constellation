"""Consensus transcript construction from clusters.

Takes the cluster membership from :func:`cluster_reads` plus the
underlying read sequences, builds a per-cluster multiple-sequence
alignment, and emits ``TRANSCRIPT_CLUSTER_TABLE`` rows with the
consensus sequence + best ORF.

Two-stage:

    1. MSA per cluster — minimap2 (asm5 preset) against the
       highest-quality member as a centroid, or all-vs-all for small
       clusters; collapse with abundance-weighted majority vote.
    2. Polish the consensus — apply abundance-weighted pile-up
       correction. The high-error rate of nanopore is what makes this
       cost-effective: each position has many independent observations
       and the per-position majority vote (with abundance weighting)
       has much lower error than any single read.

The output consensus sequences feed ORF prediction (re-run on the
consensus, not on individual reads), which gives the predicted
proteins that bridge into massspec spectral libraries.

Status: STUB. Pending Phase 6.
"""

from __future__ import annotations

import pyarrow as pa


_PHASE = "Phase 6 (transcriptome/consensus)"


def build_consensus(
    cluster_membership: pa.Table,    # output of cluster_reads
    reads: pa.Table,                 # READ_TABLE joined to READ_DEMUX_TABLE
    *,
    min_cluster_size: int = 3,
    codon_table: int = 1,            # NCBI transl_table for ORF call
) -> pa.Table:                       # TRANSCRIPT_CLUSTER_TABLE
    """Build per-cluster consensus transcripts + predict ORFs.

    Returns ``TRANSCRIPT_CLUSTER_TABLE``-shaped rows with:

        cluster_id              from cluster_membership
        representative_read_id  highest-quality member of the cluster
        n_reads                 cluster size
        consensus_sequence      MSA + abundance-weighted majority vote
        predicted_protein       longest ORF on the consensus
                                (codon_table per arg)
        orf_start/end/strand    coords on consensus_sequence
    """
    raise NotImplementedError(f"build_consensus pending {_PHASE}")


__all__ = ["build_consensus"]
