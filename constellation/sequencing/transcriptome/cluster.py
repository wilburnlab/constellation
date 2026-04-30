"""mmseqs-style kmer clustering with abundance-weighted consensus.

The lab's working clustering is naive: predict ORFs per read, group
by predicted protein. Empirically only ~1-3% of reads cluster — the
~1-2% per-base error rate creates so many singleton predictions that
genuine isoforms get fragmented across many singletons.

Phase 6 ships an alpha approach the lab has tested:

    1. mmseqs-style kmer prefiltering — cluster reads by shared
       k-mers using mmseqs2 (already conda-installed).
    2. Abundance-sort the kmer-clusters — high-abundance signals are
       more likely to be real transcripts (vs. error-degraded variants).
    3. Use high-abundance reads as cluster guides; align the lower-
       abundance variants against them; build per-position consensus
       with abundance weighting (pile-up consensus).
    4. The consensus sequence becomes the cluster representative;
       ``TRANSCRIPT_CLUSTER_TABLE.consensus_sequence`` carries it
       forward.

Goal: ≥10× the per-read clustering rate.

Status: STUB. Pending Phase 6.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa


_PHASE = "Phase 6 (transcriptome/cluster)"


def cluster_reads(
    demuxed_reads: pa.Table,         # READ_DEMUX_TABLE joined to READ_TABLE
    *,
    work_dir: Path,
    kmer_size: int = 15,
    identity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    threads: int = 8,
) -> pa.Table:                       # cluster-id-per-read mapping
    """Cluster reads via mmseqs kmer prefiltering + abundance-weighted
    consensus.

    Returns a long-format Arrow table with columns
    ``(read_id, cluster_id, score)`` — one row per cluster-read
    membership. Singleton reads (clusters of size 1) are dropped if
    ``min_cluster_size`` ≥ 2.

    ``work_dir`` is where mmseqs writes its index + alignments;
    callers are responsible for cleanup.
    """
    raise NotImplementedError(f"cluster_reads pending {_PHASE}")


__all__ = ["cluster_reads"]
