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


def _edge_rows(n: int, src: np.ndarray, dst: np.ndarray, edge: np.ndarray):
    """CSR holding ``edge + 1``, rows sorted; a repeated pair comes back summed."""
    rows = coo_matrix((edge + 1, (src, dst)), shape=(n, n)).tocsr()
    rows.sum_duplicates()
    return rows


def guarded_set_cover(
    n: int,
    order: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_off_5p: np.ndarray,
    edge_off_3p: np.ndarray,
    *,
    tol_5p: int,
    tol_3p: int,
    origin: np.ndarray | None = None,
    edge_identical: np.ndarray | None = None,
) -> ComponentResult:
    """Radius-1 greedy set cover whose merge guards hold per **group**.

    A merge is decided pair by pair, but :func:`greedy_set_cover` groups a
    *star* around its claimant, and a guard every edge of a star passes says
    nothing about two of its leaves. Two failures follow, both verified:

    - a hub ``B`` mergeable with ``A`` and with ``C`` joins ``A`` to ``C``
      through itself even when they are the two children of one M-step split
      (edges ``(B, A)``, ``(B, C)`` with ``B`` best supported collapse all
      three) — the split/merge limit cycle the pairwise kin test was meant to
      stop, reached by one hop;
    - two members each within the end tolerance of the hub can lie twice the
      tolerance from each other.

    So the claimant carries the group's state and tests each neighbour against
    it. Nodes are walked in ``order`` (a permutation of ``range(n)``, best
    first); an unclaimed node becomes a claimant and considers its unclaimed
    *direct* neighbours, best first.

    **Extent guard.** ``edge_off_5p`` / ``edge_off_3p`` are signed nt: how far
    ``b`` reaches beyond ``a`` at that end (negative: ``a`` beyond ``b``).
    Read from the claimant, which sits at 0, a neighbour is accepted only if
    the accepted offsets — claimant included — still span at most ``tol_5p``
    and ``tol_3p`` with it added.

    **Kin guard** (``origin`` given; ``-1`` = no recorded split). A neighbour
    whose origin is already in the group is refused unless it and every member
    holding that origin are byte-identical to the claimant
    (``edge_identical``; the claimant is identical to itself). Reads cannot
    tell byte-identical templates apart, so merging them moves nothing.

    A refused neighbour stays unclaimed: it becomes a claimant at its own turn
    or is claimed by a later one. Self-edges are ignored and of a repeated
    pair the first is used. Where no guard can bind the result is
    :func:`greedy_set_cover`'s for the same order, array for array.
    """
    n = int(n)
    order = np.asarray(order, dtype=np.int64)
    if order.shape != (n,):
        raise ValueError(f"order must have shape ({n},), got {order.shape}")
    if tol_5p < 0 or tol_3p < 0:
        raise ValueError(f"tolerances must be >= 0, got {tol_5p} / {tol_3p}")
    edge_a = np.asarray(edge_a, dtype=np.int64)
    edge_b = np.asarray(edge_b, dtype=np.int64)
    off_5p = np.asarray(edge_off_5p, dtype=np.int64)
    off_3p = np.asarray(edge_off_3p, dtype=np.int64)
    per_edge = [edge_b, off_5p, off_3p]
    if edge_identical is not None:
        twin = np.asarray(edge_identical, dtype=bool)
        per_edge.append(twin)
    else:
        twin = np.zeros(edge_a.shape[0], dtype=bool)
    if edge_a.ndim != 1 or any(x.shape != edge_a.shape for x in per_edge):
        raise ValueError("edge arrays must be 1-D and of one length")
    if edge_a.size and (
        min(edge_a.min(), edge_b.min()) < 0 or max(edge_a.max(), edge_b.max()) >= n
    ):
        raise ValueError(f"edge endpoints must lie in [0, {n})")
    if origin is not None:
        origin = np.asarray(origin, dtype=np.int64)
        if origin.shape != (n,):
            raise ValueError(f"origin must have shape ({n},), got {origin.shape}")
    if n == 0:
        return ComponentResult(np.empty(0, np.int64), np.empty(0, np.int64))
    if order.min() < 0 or order.max() >= n or np.bincount(order, minlength=n).min() < 1:
        raise ValueError("order must be a permutation of range(n)")

    # Work in rank space. When a node's turn comes every better-ranked node is
    # already claimed — it claimed itself or was claimed — so a claimant only
    # ever looks at WORSE-ranked neighbours: each pair is stored once, under
    # its better-ranked endpoint, with the offsets signed as that endpoint
    # reads them. Relabelled by rank, a sorted row is the claim-priority order.
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    rank_a, rank_b = rank[edge_a], rank[edge_b]
    edge = np.flatnonzero(rank_a != rank_b)
    rank_a, rank_b = rank_a[edge], rank_b[edge]
    a_claims = rank_a < rank_b
    src = np.where(a_claims, rank_a, rank_b)
    dst = np.where(a_claims, rank_b, rank_a)
    sign = np.ones(edge_a.shape[0], dtype=np.int64)
    sign[edge[~a_claims]] = -1
    del rank_a, rank_b, a_claims

    # Each entry carries the input edge it came from. scipy SUMS the data of a
    # repeated pair, which for an edge index is meaningless — so a repeat is
    # detected by the entry count and the pairs cut to their first occurrence
    # before the matrix is built again. An edge table that stores each pair
    # once never pays for this.
    adj = _edge_rows(n, src, dst, edge)
    if adj.nnz < edge.shape[0]:
        # src * n + dst fits int64 for any n below 3.03e9.
        _, first = np.unique(src * n + dst, return_index=True)
        first.sort()
        adj = _edge_rows(n, src[first], dst[first], edge[first])
    del src, dst
    indptr = adj.indptr.astype(np.int64)
    nbr = adj.indices
    edge = adj.data - 1
    off_5p = (sign * off_5p)[edge]
    off_3p = (sign * off_3p)[edge]
    twin = twin[edge]
    del adj, edge, sign

    kin = None
    n_kin = 0
    if origin is not None:
        # Origins are arbitrary ids (a template id is ``round << 40 | row``);
        # dense codes are what lets (row, origin) pack into one int64 below.
        by_rank = origin[order]
        known = by_rank >= 0
        uniq, code = np.unique(by_rank[known], return_inverse=True)
        n_kin = int(uniq.shape[0])
        kin = np.full(n, -1, dtype=np.int64)
        kin[known] = code

    # Which rows could a guard bind on at all? Judged once, over each row's
    # WHOLE neighbour list: a guard that cannot bind on the list cannot bind
    # on any subset of it, in any order. Those rows claim as greedy_set_cover
    # does, and the per-neighbour walk is paid only where a refusal is possible.
    rows = np.flatnonzero(indptr[1:] > indptr[:-1])
    starts = indptr[rows]
    guarded = np.zeros(n, dtype=bool)
    if rows.shape[0]:
        for off, tol in ((off_5p, tol_5p), (off_3p, tol_3p)):
            span = np.maximum(np.maximum.reduceat(off, starts), 0) - np.minimum(
                np.minimum.reduceat(off, starts), 0
            )
            guarded[rows[span > tol]] = True
        if n_kin:
            row_of = np.repeat(rows, indptr[rows + 1] - starts)
            k = kin[nbr]
            guarded[row_of[(k >= 0) & (k == kin[row_of])]] = True
            key = np.sort(row_of[k >= 0] * n_kin + k[k >= 0])
            guarded[key[1:][key[1:] == key[:-1]] // n_kin] = True
            del row_of, k, key

    claimant = np.full(n, -1, dtype=np.int64)  # rank -> its claimant's rank
    for r, lo, hi, may_refuse in zip(
        rows.tolist(),
        starts.tolist(),
        indptr[rows + 1].tolist(),
        guarded[rows].tolist(),
    ):
        if claimant[r] >= 0:
            continue
        claimant[r] = r
        free = np.flatnonzero(claimant[nbr[lo:hi]] < 0) + lo
        if not may_refuse:
            claimant[nbr[free]] = r
            continue

        # origin -> "every member holding it is byte-identical to the
        # claimant", which is all a later twin of that origin needs to know.
        held: dict[int, bool] = {}
        cand = nbr[free]
        if kin is not None:
            if kin[r] >= 0:
                held[int(kin[r])] = True
            cand_kin = kin[cand].tolist()
        else:
            cand_kin = [-1] * cand.shape[0]
        lo5 = hi5 = lo3 = hi3 = 0
        taken: list[int] = []
        for m, m5, m3, k_m, same in zip(
            cand.tolist(),
            off_5p[free].tolist(),
            off_3p[free].tolist(),
            cand_kin,
            twin[free].tolist(),
        ):
            if max(hi5, m5) - min(lo5, m5) > tol_5p:
                continue
            if max(hi3, m3) - min(lo3, m3) > tol_3p:
                continue
            if k_m >= 0:
                seen = held.get(k_m)
                if seen is not None and not (seen and same):
                    continue
                held[k_m] = same if seen is None else True
            lo5, hi5 = min(lo5, m5), max(hi5, m5)
            lo3, hi3 = min(lo3, m3), max(hi3, m3)
            taken.append(m)
        if taken:
            claimant[np.asarray(taken, dtype=np.int64)] = r

    # A node nobody claimed is its own claimant, and groups are numbered in
    # claimant order — the ids greedy_set_cover assigns while it walks.
    unclaimed = claimant < 0
    claimant[unclaimed] = np.flatnonzero(unclaimed)
    is_claimant = claimant == np.arange(n, dtype=np.int64)
    group_of_rank = np.cumsum(is_claimant) - 1
    cluster_of = np.empty(n, dtype=np.int64)
    cluster_of[order] = group_of_rank[claimant]
    return ComponentResult(
        cluster_of=cluster_of, centroid_uniq=order[is_claimant].astype(np.int64)
    )


__all__ = [
    "connected_components",
    "greedy_set_cover",
    "guarded_set_cover",
    "ComponentResult",
]
