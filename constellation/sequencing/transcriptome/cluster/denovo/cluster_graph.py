"""Connected-components clustering over the verified-similarity graph.

At ~1% per-base error over ~1 kb reads, almost every read is a distinct
unique sequence (exact-derep abundance ≈ 1), so the roadmap's
abundance-anchored *radius-1* greedy set-cover under-clusters — a
centroid only claims its direct neighbours, leaving the rest of a true
transcript's reads stranded as separate clusters. Grouping by
**connected components** of the edit-distance-verified graph collapses
all minor variants of one transcript together: two reads are joined only
through a chain of ≥ identity-threshold edges, and distinct transcripts
(> 2% divergent) share no edge, so they stay separate.

Each component's centroid is its most-supported, longest member
(abundance desc → length desc → uniq asc); it becomes the consensus
coordinate frame. Members reach the centroid through the component (not
necessarily a direct edge), so the consensus stage aligns each member to
the centroid — reusing the cached verify CIGAR for direct edges and
re-aligning only the component-path members.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _scc


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """Connected-component assignment over unique sequences.

    ``cluster_of`` maps ``uniq_id`` → dense ``cluster_id``; ``centroid_uniq``
    maps ``cluster_id`` → its centroid ``uniq_id``.
    """

    cluster_of: np.ndarray  # int64 (U,)
    centroid_uniq: np.ndarray  # int64 (C,)


def connected_components(
    n_uniq: int,
    abundance: np.ndarray,
    seq_len: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
) -> ComponentResult:
    """Connected components of the verified graph + per-component centroid."""
    if n_uniq == 0:
        return ComponentResult(np.empty(0, np.int64), np.empty(0, np.int64))
    if edge_a.shape[0] == 0:
        labels = np.arange(n_uniq, dtype=np.int64)
    else:
        data = np.ones(edge_a.shape[0], dtype=np.int8)
        graph = coo_matrix(
            (data, (edge_a.astype(np.int64), edge_b.astype(np.int64))),
            shape=(n_uniq, n_uniq),
        )
        _, labels = _scc(graph, directed=False)
        labels = labels.astype(np.int64)

    n_comp = int(labels.max()) + 1 if n_uniq else 0

    # Per-component centroid: highest-priority member by (abundance desc,
    # seq_len desc, uniq asc). Sort uniqs into that priority order, then the
    # first occurrence of each component label in that order is its centroid
    # (np.unique returns the first index per sorted label — fully vectorized).
    uniq_ids = np.arange(n_uniq, dtype=np.int64)
    order = np.lexsort(
        (uniq_ids, -seq_len.astype(np.int64), -abundance.astype(np.int64))
    )
    lab_ordered = labels[order]
    centroid_uniq = np.full(n_comp, -1, dtype=np.int64)
    uniq_labels, first_idx = np.unique(lab_ordered, return_index=True)
    centroid_uniq[uniq_labels] = order[first_idx]
    return ComponentResult(cluster_of=labels, centroid_uniq=centroid_uniq)


def greedy_set_cover(
    n_uniq: int,
    abundance: np.ndarray,
    seq_len: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
) -> ComponentResult:
    """Abundance-ordered **radius-1** greedy set cover over the same graph.

    Walk the uniques in ``(abundance desc, length desc, uniq asc)`` order —
    the same priority :func:`connected_components` uses for its centroid — and
    let each unclaimed unique claim itself plus its *unclaimed direct
    neighbours*. Because a claimant never claims a neighbour's neighbour, a
    group cannot chain: on the ORF-level fixture the largest greedy group is
    2,889 uniques against 5,743 for components over the identical edge set, at
    the same purity.

    This is the right grouping wherever abundance is a real prior. At the
    whole-read level it is not — ~1% per-base error over ~1 kb makes almost
    every read a distinct unique with abundance ≈ 1, so radius-1 greedy
    under-clusters badly and :func:`connected_components` is used instead. At
    the *ORF* level hubs do exist (30% of reads share an exact ORF), which is
    what this exists for.
    """
    if n_uniq == 0:
        return ComponentResult(np.empty(0, np.int64), np.empty(0, np.int64))

    uniq_ids = np.arange(n_uniq, dtype=np.int64)
    order = np.lexsort(
        (uniq_ids, -seq_len.astype(np.int64), -abundance.astype(np.int64))
    )

    if edge_a.shape[0]:
        # Symmetrised CSR: each claimant needs its neighbours in both
        # directions, and the verified edge list stores each pair once.
        a = np.concatenate([edge_a, edge_b]).astype(np.int64)
        b = np.concatenate([edge_b, edge_a]).astype(np.int64)
        csr = coo_matrix(
            (np.ones(a.shape[0], dtype=np.int8), (a, b)), shape=(n_uniq, n_uniq)
        ).tocsr()
        indptr, indices = csr.indptr, csr.indices
    else:
        indptr = np.zeros(n_uniq + 1, dtype=np.int64)
        indices = np.empty(0, dtype=np.int64)

    labels = np.full(n_uniq, -1, dtype=np.int64)
    centroids: list[int] = []
    for u in order:
        u = int(u)
        if labels[u] >= 0:
            continue
        cid = len(centroids)
        labels[u] = cid
        nbrs = indices[indptr[u] : indptr[u + 1]]
        if nbrs.shape[0]:
            free = nbrs[labels[nbrs] < 0]
            labels[free] = cid
        centroids.append(u)
    return ComponentResult(
        cluster_of=labels, centroid_uniq=np.asarray(centroids, dtype=np.int64)
    )


__all__ = ["connected_components", "greedy_set_cover", "ComponentResult"]
