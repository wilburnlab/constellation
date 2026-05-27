"""Read-similarity network → reference-free transcriptome structure.

Eventual goal: build a ``core.graph.Network`` over reads (or clusters)
where edges are sequence-similarity scores, and recover transcript /
gene / allele / exon structure as graph properties without needing a
reference genome:

    cluster (transcript)
      └─ multiple haplotypes within a cluster (alleles)
          └─ exon shared across multiple transcripts (M:N edges)

Reference-free transcriptome assembly is a hard problem; this module
documents the direction the lab plans to take (network-based
deconvolution) without committing to a specific algorithm yet.

Status: scaffold. The Network shape and the cluster-to-network
projection are what's already in :mod:`core.graph.network` (see
``massspec.library.Library.to_network`` for the worked example);
the actual graph-construction + traversal logic for transcripts is
Phase 6 / 12 work.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.graph.network import Network


_PHASE = "Phase 6 (transcriptome/network)"


def build_read_network(
    cluster_membership: pa.Table,
    cluster_table: pa.Table,         # TRANSCRIPT_CLUSTER_TABLE
    *,
    edge_threshold: float = 0.7,
) -> Network:
    """Build a ``Network[ClusterId, EdgeT]`` over consensus clusters.

    Nodes are clusters (representative consensus sequences); edges
    connect clusters whose pairwise similarity exceeds
    ``edge_threshold``. Connected components in the resulting graph
    are candidate gene-level groupings; cycle structure within a
    component is suggestive of allelic / isoform variation.
    """
    raise NotImplementedError(f"build_read_network pending {_PHASE}")


__all__ = ["build_read_network"]
