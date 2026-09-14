"""Covariance over candidate columns: which of them earn a node, and how
reads partition across the ones that do.

The organising principle is that **a column earns a node only by co-varying
with another column**. That is what kills the homopolymers with no special
case: a systematic 22% indel at a 5-G run passes candidacy easily and then
covaries with nothing, so it never becomes a node. It has a stated cost — a
lone real frameshift with no covarying partner dies the same way (see the
module guide and the plan's "cost of pure covariance").

Written against plain arrays in ``(candidate index, read index)`` space only:
no :class:`ConsensusResult`, no ``MemberSpec``, no ORFs, no edlib. The clique
and assignment behaviour is then provable on hand-built inputs rather than
asserted through an end-to-end fixture.

Three things here are **not** what the spec assumed, each because it was
measured:

1. **Coverage columns never earn an edge from another coverage column.** A
   read's coverage is an interval, so "uncovered at u" and "uncovered at v"
   are both implied by "this read ended before min(u, v)": the two sets are
   *nested*, which is maximal association, not the independence §3f assumed.
   A 300-read degradation ramp tests at p = 9e-44 between columns 200 apart.
   No threshold separates a ramp from a start mode on that statistic, so the
   statistic is not used. Coverage evidence comes from the exact-identity
   clique instead — see 2 — and from cross-route edges with allelic columns,
   where the confound does not apply.
2. **A collapsed coverage representative is tested as a degenerate 2×2 on
   its delimiting boundary mass.** Run collapse is exact, so a representative
   standing for ``run_len ≥ 2`` columns is already a clique; what has to be
   judged is whether the boundary is real, and the evidence for "these reads
   start here" is the reads that start *there*, not the reads that happen to
   be absent downstream.
3. **Marginals are not free from the PWM.** §3d says both "only ``n_uv``
   needs the pass" and "test each pair on the co-covered 2×2". The PWM's
   marginals are over all reads covering each column separately, so using
   them inflates significance in a depth- and overlap-dependent way — and
   they differ most in exactly the coverage-variable case that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo.orfem.columns import (
    ROUTE_ALLELIC,
    ROUTE_COVERAGE,
    CandidateSet,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import (
    benjamini_hochberg,
)


#: Read-state alphabet. ``MAJOR`` is the implicit default and never stored.
MAJOR = 0
#: A base or gap allele ``a ∈ [0, 4]`` is stored as ``a + 1``.
UNCOVERED = 6  # observed *as* a state, at coverage columns only
UNOBSERVED = -1  # missing data: contributes exactly 0 to every score

@dataclass(frozen=True, slots=True)
class ReadStates:
    """Sparse per-read state over ``V`` candidate columns.

    Only departures from ``MAJOR`` are stored, and only where the read
    *observes* the column: covered, or — at a coverage column — anywhere at
    all, since "this read is not here" is the signal that route selected for.
    """

    ptr: np.ndarray  # (M+1,) int64
    v: np.ndarray  # (nnz,) int32 — candidate index
    state: np.ndarray  # (nnz,) int8
    obs_lo: np.ndarray  # (M,) int32 — first candidate index the read covers
    obs_hi: np.ndarray  # (M,) int32 — one past the last
    weight: np.ndarray  # (M,) float64
    is_coverage: np.ndarray  # (V,) bool — uncovered is a STATE at these
    n_members: int
    n_candidates: int

    @property
    def total_weight(self) -> float:
        return float(self.weight.sum())


@dataclass(frozen=True, slots=True)
class CovarianceGraph:
    """Significant covariance edges and the columns they retain."""

    edges: np.ndarray  # (E, 2) int32 — u ≤ v; u == v is a collapsed-run self-edge
    p_value: np.ndarray  # (E,) float64
    keep: np.ndarray  # (V,) bool
    q_cut: float
    n_pairs_seen: int  # pairs with any co-occurrence at all
    n_tested: int  # size of the BH family
    n_significant: int
    n_reads_capped: int  # excluded from the pass by max_state_per_read
    sum_k2: int  # Sigma k_i^2 actually spent by the co-occurrence pass


def read_states(events, cand: CandidateSet, weight: np.ndarray) -> ReadStates:
    """Project :class:`AlleleEvents` onto the candidate set's state alphabet.

    An uncovered entry becomes a *state* at a **pure**-coverage column and
    *missing data* everywhere else. That split is the direct fix for the 28.1%
    of round-6 nodes (1.24M reads) that differed from their major only at
    uncovered positions: there, missing data was a symbol everywhere, so a
    read that said nothing manufactured a species. Here it can only speak at a
    column selected *because* coverage varies there.

    "Pure" is load-bearing, and was measured. A column admitted by both routes
    already carries the coverage fact through the collapsed pure-coverage
    representative of the same block, so letting it speak again makes a read
    outside a block report "I am not here" once per column of every other
    block. On a 12 kb six-gene fused template that is 85 states per read
    instead of ~21 — past ``max_state_per_read`` for **every** read, which
    silently emptied the whole co-occurrence pass and left the within-block
    allelic splits undetected. The redundancy, not the cap, was the bug.
    """
    v_cols = np.asarray(cand.columns, dtype=np.int64)
    route = np.asarray(cand.route)
    is_cov = ((route & ROUTE_COVERAGE) != 0) & ((route & ROUTE_ALLELIC) == 0)
    m = int(events.n_members)

    lo = np.searchsorted(v_cols, events.span_lo, side="left").astype(np.int32)
    hi = np.searchsorted(v_cols, events.span_hi, side="left").astype(np.int32)

    rows_v: list[np.ndarray] = []
    rows_s: list[np.ndarray] = []
    counts = np.zeros(m, dtype=np.int64)
    for i in range(m):
        a, b = events.nm_ptr[i], events.nm_ptr[i + 1]
        nm_v = events.nm_v[a:b]
        nm_s = (events.nm_allele[a:b] + 1).astype(np.int8)
        c, d = events.uc_ptr[i], events.uc_ptr[i + 1]
        uc_v = events.uc_v[c:d]
        uc_v = uc_v[is_cov[uc_v]]  # pure-coverage columns only
        vv = np.concatenate([nm_v, uc_v])
        ss = np.concatenate([nm_s, np.full(uc_v.shape[0], UNCOVERED, np.int8)])
        order = np.argsort(vv, kind="stable")
        rows_v.append(vv[order].astype(np.int32))
        rows_s.append(ss[order])
        counts[i] = vv.shape[0]

    ptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ReadStates(
        ptr=ptr,
        v=(
            np.concatenate(rows_v) if rows_v else np.zeros(0, np.int32)
        ),
        state=(np.concatenate(rows_s) if rows_s else np.zeros(0, np.int8)),
        obs_lo=lo,
        obs_hi=hi,
        weight=np.asarray(weight, dtype=np.float64),
        is_coverage=is_cov,
        n_members=m,
        n_candidates=int(v_cols.shape[0]),
    )


def _cooccurrence(
    states: ReadStates,
    active: np.ndarray,
    max_state_per_read: int,
    pure_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """``(n11, deg_w, n_capped, budget)`` over reads within the state cap.

    ``n11`` is ``Aᵀ·diag(w)·A`` over the sparse non-major indicator, which
    touches only pairs that actually co-occur and never the ``V²`` that do
    not. The pure-coverage × pure-coverage block is **never computed**, not
    merely discarded: those pairs are not testable (see the module docstring),
    and on a degradation ramp — where every candidate is a pure-coverage
    column — skipping it takes the co-occurrence budget from 257,273 to 0 on
    the 300-read synthetic fixture, rather than spending it to test nothing.

    ``max_state_per_read`` is a diagnostic escape hatch, not a design element.
    A read excluded here is still assigned; it just does not vote on which
    columns are linked. It exists because the budget is ``Σ kᵢ²`` and a read
    outside a wide coverage block contributes one state per column of it.
    """
    from scipy import sparse

    s_ptr, w = states.ptr, states.weight
    n_v = states.n_candidates
    k = (s_ptr[1:] - s_ptr[:-1]).astype(np.int64)
    ok = active & (k <= max_state_per_read)
    n_capped = int((active & ~ok).sum())

    rows = np.flatnonzero(ok)
    n11 = np.zeros((n_v, n_v), dtype=np.float64)
    deg_w = np.zeros(n_v, dtype=np.float64)
    if rows.size == 0 or n_v == 0:
        return n11, deg_w, n_capped, 0

    gather = np.concatenate([np.arange(s_ptr[i], s_ptr[i + 1]) for i in rows])
    if gather.size == 0:
        return n11, deg_w, n_capped, 0
    col = states.v[gather].astype(np.int64)
    row = np.repeat(np.arange(rows.shape[0]), k[rows])
    ww = np.repeat(w[rows], k[rows])
    deg_w = np.bincount(col, weights=ww, minlength=n_v)

    shape = (rows.shape[0], n_v)
    a_ind = sparse.csr_matrix((np.ones(col.shape[0]), (row, col)), shape=shape)
    a_w = sparse.csr_matrix((ww, (row, col)), shape=shape)

    oth = np.flatnonzero(~pure_cov)
    pcv = np.flatnonzero(pure_cov)
    k_oth = np.bincount(row, weights=(~pure_cov)[col], minlength=rows.shape[0])
    k_pc = k[rows] - k_oth
    budget = int((k_oth**2).sum() + (k_pc * k_oth).sum())
    if oth.size:
        blk = (a_ind[:, oth].T @ a_w[:, oth]).toarray()
        n11[np.ix_(oth, oth)] = blk
        if pcv.size:
            cross = (a_ind[:, pcv].T @ a_w[:, oth]).toarray()
            n11[np.ix_(pcv, oth)] = cross
            n11[np.ix_(oth, pcv)] = cross.T
    np.fill_diagonal(n11, 0.0)
    return n11, deg_w, n_capped, budget


def _observation_tables(
    states: ReadStates, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(D, A)``.

    ``D[u, v]`` (u ≤ v) is the weight of reads covering both allelic columns
    ``u`` and ``v``; ``D[v, v]`` is the weight covering ``v`` alone.
    ``A[u, v]`` is the weight of reads **non-major at u** that cover ``v``.

    Both are dominance counts over the reads' covered candidate intervals, not
    PWM marginals. The 2-D histogram is ``(V+1)²`` — 2 MB at the 512-column
    cap — so a prefix-in-``a`` / suffix-in-``b`` cumsum turns every pair query
    into one lookup.
    """
    n_v = states.n_candidates
    rows = np.flatnonzero(active)
    lo = np.clip(states.obs_lo[rows], 0, n_v).astype(np.int64)
    hi = np.clip(states.obs_hi[rows], 0, n_v).astype(np.int64)
    w = states.weight[rows]

    h = np.bincount(
        lo * (n_v + 1) + hi, weights=w, minlength=(n_v + 1) * (n_v + 1)
    ).reshape(n_v + 1, n_v + 1)
    c = np.cumsum(h, axis=0)  # Σ_{a ≤ u}
    r = np.cumsum(c[:, ::-1], axis=1)[:, ::-1]  # Σ_{b ≥ ·}
    d = r[:n_v, 1:]  # D[u, v] = Σ_{a ≤ u} Σ_{b ≥ v+1}

    # A: one difference array per column u over its non-major reads' spans,
    # accumulated as a single pair of bincounts over (u, span endpoint).
    s_ptr = states.ptr
    counts = (s_ptr[1:] - s_ptr[:-1])[rows]
    gather = np.concatenate(
        [np.arange(s_ptr[i], s_ptr[i + 1]) for i in rows]
    ).astype(np.int64) if rows.size and counts.sum() else np.zeros(0, np.int64)
    flat_u = states.v[gather].astype(np.int64)
    width = n_v + 2
    keys_lo = flat_u * width + np.repeat(lo, counts)
    keys_hi = flat_u * width + np.repeat(hi, counts)
    rep_w = np.repeat(w, counts)
    diff = np.bincount(
        keys_lo, weights=rep_w, minlength=n_v * width
    ) - np.bincount(keys_hi, weights=rep_w, minlength=n_v * width)
    a_tab = np.cumsum(diff.reshape(n_v, width), axis=1)[:, :n_v]
    return d, a_tab


def _pair_pvalues(
    n11: np.ndarray,
    n1x: np.ndarray,
    nx1: np.ndarray,
    n: np.ndarray,
    *,
    min_expected: float = 5.0,
    max_exact: int = 20_000,
) -> np.ndarray:
    """Two-sided p-values for many 2×2 tables: chi-square where its
    approximation holds, exact hypergeometric where it does not.

    An earlier draft used chi-square everywhere and declared any table with an
    expected cell below 5 *untested*, "the conservative direction". Measured,
    that is not conservative, it is disabling: the expected co-occurrence cell
    is ``k²/n``, so requiring it to reach 5 demands ``k ≥ √(5n)`` — 51 reads of
    520, 316 of 20,000. It rejects precisely the low-frequency linked
    proteoform the pipeline exists to preserve, and only those.

    The fix is not to test those tables with the wrong distribution but to
    test them with the right one. The split also pays for itself: the exact
    test costs ~51 µs per pair, so at the 512-column cap testing every pair
    exactly is ~6.7 s **per template**, while chi-square is a vectorized
    microsecond. Small-cell tables are the sparse co-occurring ones — bounded
    by the co-occurrence budget, 35 of 36 on the synthetic fused fixture —
    so the exact path stays small by construction, with ``max_exact`` as a
    backstop that keeps the most extreme tables and leaves the rest untested.
    """
    e11 = n1x * nx1 / n
    e10 = n1x * (n - nx1) / n
    e01 = (n - n1x) * nx1 / n
    e00 = (n - n1x) * (n - nx1) / n
    valid = (n > 0) & (n1x > 0) & (nx1 > 0) & (n1x < n) & (nx1 < n)
    big = valid & (
        (e11 >= min_expected)
        & (e10 >= min_expected)
        & (e01 >= min_expected)
        & (e00 >= min_expected)
    )
    out = np.ones(n.shape[0], dtype=np.float64)
    if big.any():
        from scipy import stats

        n10 = n1x[big] - n11[big]
        n01 = nx1[big] - n11[big]
        n00 = n[big] - n1x[big] - nx1[big] + n11[big]
        num = (
            np.maximum(
                np.abs(n11[big] * n00 - n10 * n01) - n[big] / 2.0, 0.0
            )
            ** 2
        )
        den = n1x[big] * (n[big] - n1x[big]) * nx1[big] * (n[big] - nx1[big])
        out[big] = stats.chi2.sf(n[big] * num / den, 1)

    small = np.flatnonzero(valid & ~big)
    if small.size > max_exact:
        extreme = np.abs(n11[small] - e11[small])
        small = small[np.argsort(-extreme, kind="stable")[:max_exact]]
    if small.size:
        out[small] = _fisher_sf(
            n11[small], n1x[small], nx1[small], n[small]
        )
    return out


def _fisher_sf(
    n11: np.ndarray, n1x: np.ndarray, nx1: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """Two-sided Fisher exact p-values for many 2×2 tables at once.

    ``scipy.stats.hypergeom.sf`` **is** the one-sided Fisher test and it is a
    vectorized ufunc, so this never becomes the scalar
    ``scipy.stats.fisher_exact`` in a hot row loop — pitfall #1 in the module
    guide, and at 10k pairs per template it would dominate the M-step.

    The test is **two-sided**, because depletion is evidence too. Measured on
    the synthetic fused template: gene A's columns and gene B's columns encode
    the same read partition from opposite sides, so a read non-major at one is
    never non-major at the other and their co-occurrence is exactly zero. A
    one-sided test finds no edge, the two column sets become two separate
    signatures, and the state-tuple product then re-splits the template along
    a partition it had already made — emitting a spurious node for every
    disagreement between two descriptions of the same thing.

    Weights are read multiplicities, so they are rounded to integers here.
    """
    from scipy import stats

    nn = np.rint(n).astype(np.int64)
    a = np.rint(n1x).astype(np.int64)
    b = np.rint(nx1).astype(np.int64)
    k = np.rint(n11).astype(np.int64)
    ok = (nn > 0) & (a > 0) & (b > 0) & (a <= nn) & (b <= nn)
    out = np.ones(nn.shape[0], dtype=np.float64)
    if ok.any():
        upper = stats.hypergeom.sf(k[ok] - 1, nn[ok], a[ok], b[ok])
        lower = stats.hypergeom.cdf(k[ok], nn[ok], a[ok], b[ok])
        out[ok] = np.clip(2.0 * np.minimum(upper, lower), 0.0, 1.0)
    return out


def _mode_sf(mass: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Is this boundary a *mode*, or what a locally uniform spread would give?

    A collapsed coverage representative does not pose a question about
    association — run collapse already proved its columns are identical. The
    question is whether the boundary delimiting it is real, and the thing that
    separates an alternative transcription start from a degradation ramp is
    not mass and not fraction (measured, a ramp's boundaries carry 10% of the
    template each once endpoints are clustered) but *concentration*: a ramp
    scatters endpoints evenly across its support, a mode piles them at one
    position. So the null is "endpoints are uniform within ±``local_window``"
    and the statistic is a Poisson upper tail against that local density.
    """
    from scipy import stats

    lam = np.maximum(np.asarray(expected, dtype=np.float64), 1e-9)
    return stats.poisson.sf(np.rint(mass).astype(np.int64) - 1, lam)


def covariance_graph(
    states: ReadStates,
    cand: CandidateSet,
    *,
    q_edge: float = 0.01,
    min_n11: float = 3.0,
    min_mode_mass: float = 3.0,
    max_state_per_read: int = 64,
) -> CovarianceGraph:
    """Test every co-occurring candidate pair and retain what earns an edge."""
    n_v = states.n_candidates
    route = np.asarray(cand.route)
    is_all = (route & ROUTE_ALLELIC).astype(bool)
    # A dual-route column speaks only through its allele here; its coverage
    # half is already carried by the block's collapsed pure-coverage
    # representative. `is_cov` therefore means "uncovered is a state", which
    # is what both the observation counts and `read_states` use.
    pure_cov = ((route & ROUTE_COVERAGE) != 0) & ~is_all
    is_cov = pure_cov
    active = np.ones(states.n_members, dtype=bool)

    n11, deg_w, n_capped, sum_k2 = _cooccurrence(
        states, active, max_state_per_read, pure_cov
    )
    d_tab, a_tab = _observation_tables(states, active)
    total_w = float(states.weight[active].sum())

    uu, vv = np.triu_indices(n_v, 1)
    # "Observes" means covered at an allelic column and unconditional at a
    # coverage column, so the co-observed set is an intersection of the two.
    k11 = n11[uu, vv]
    n1x = np.where(is_cov[vv], deg_w[uu], a_tab[uu, vv])
    nx1 = np.where(is_cov[uu], deg_w[vv], a_tab[vv, uu])
    n_obs = np.where(
        is_cov[uu] & is_cov[vv],
        total_w,
        np.where(
            is_cov[uu],
            d_tab[vv, vv],
            np.where(is_cov[vv], d_tab[uu, uu], d_tab[uu, vv]),
        ),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        expect = np.where(n_obs > 0, n1x * nx1 / np.maximum(n_obs, 1e-12), 0.0)
    # Either tail is evidence, so a pair qualifies on observed co-occurrence
    # OR on having enough expected co-occurrence for its absence to mean
    # something. A pair of pure-coverage columns is excluded outright, not
    # thresholded: coverage is an interval, so nested uncovered sets make the
    # test significant whether or not anything latent is there.
    seen = (k11 >= min_n11) | (expect >= min_n11)
    seen &= ~(pure_cov[uu] & pure_cov[vv])
    uu, vv = uu[seen], vv[seen]
    n_pairs_seen = int(seen.sum())

    if uu.size:
        p_pair = _pair_pvalues(k11[seen], n1x[seen], nx1[seen], n_obs[seen])
        tested_p = np.ones(p_pair.shape[0], dtype=bool)
    else:
        p_pair = np.zeros(0)
        tested_p = np.zeros(0, dtype=bool)

    # Collapsed coverage runs: the representative already stands for a clique
    # of perfectly correlated columns, so what is judged is whether the
    # boundary delimiting it is real. k is the DELIMITING BOUNDARY MASS, not
    # the cumulative uncovered mass — a ramp accumulates the latter out of
    # boundaries carrying one read each.
    self_v = np.flatnonzero(pure_cov & (np.asarray(cand.run_len) >= 2))
    if self_v.size:
        k = np.asarray(cand.boundary_mass, dtype=np.float64)[self_v]
        p_self = _mode_sf(k, np.asarray(cand.boundary_expected)[self_v])
        tested_s = k >= min_mode_mass
    else:
        p_self = np.zeros(0)
        tested_s = np.zeros(0, dtype=bool)

    pvals = np.concatenate([p_pair, p_self])
    tested = np.concatenate([tested_p, tested_s])
    eu = np.concatenate([uu, self_v]).astype(np.int32)
    ev = np.concatenate([vv, self_v]).astype(np.int32)

    keep = np.zeros(n_v, dtype=bool)
    t_idx = np.flatnonzero(tested)
    if t_idx.size == 0:
        return CovarianceGraph(
            edges=np.zeros((0, 2), np.int32),
            p_value=np.zeros(0),
            keep=keep,
            q_cut=-1.0,
            n_pairs_seen=n_pairs_seen,
            n_tested=0,
            n_significant=0,
            n_reads_capped=n_capped,
            sum_k2=sum_k2,
        )
    p = pvals[t_idx]
    sig, q_cut = benjamini_hochberg(p, q_edge)
    hit = t_idx[sig]
    keep[eu[hit]] = True
    keep[ev[hit]] = True
    return CovarianceGraph(
        edges=np.stack([eu[hit], ev[hit]], axis=1),
        p_value=p[sig],
        keep=keep,
        q_cut=q_cut,
        n_pairs_seen=n_pairs_seen,
        n_tested=int(t_idx.size),
        n_significant=int(hit.size),
        n_reads_capped=n_capped,
        sum_k2=sum_k2,
    )


def quasi_cliques(graph: CovarianceGraph, *, gamma: float = 0.6) -> list[np.ndarray]:
    """Partition each connected component into density-``γ`` quasi-cliques.

    Seed at the highest-degree node ordered by ``(degree desc, Σ −log10 p
    desc, index asc)``; grow by adding any ``v`` with
    ``|N(v) ∩ S| ≥ ⌈γ|S|⌉``, restarting the scan on each addition; emit;
    remove; repeat.

    Components alone are not enough — one spurious edge between two real
    splits merges them, which is the failure §3f names. Maximal cliques are
    the wrong target in the other direction: a single *missing* edge inside a
    real k-column split yields two overlapping maximal cliques where the
    answer is one signature, leaving a set-cover problem with no objective.
    Bron–Kerbosch is also exponential exactly on the fused-template hairballs
    where it would be needed. It is kept as a **test oracle** on small random
    graphs instead.

    The ordering is part of the contract: greedy is order-dependent, so a
    deterministic key is what makes the result reproducible across runs and
    worker counts.
    """
    n_v = graph.keep.shape[0]
    pair = graph.edges[graph.edges[:, 0] != graph.edges[:, 1]]
    adj = [set() for _ in range(n_v)]
    strength = np.zeros(n_v, dtype=np.float64)
    for (u, v), p in zip(pair, graph.p_value[graph.edges[:, 0] != graph.edges[:, 1]]):
        adj[u].add(int(v))
        adj[v].add(int(u))
        s = -np.log10(max(float(p), 1e-300))
        strength[u] += s
        strength[v] += s

    out: list[np.ndarray] = []
    for comp in _components(adj, graph.keep):
        remaining = set(comp)
        while remaining:
            order = sorted(
                remaining,
                key=lambda x: (-len(adj[x] & remaining), -strength[x], x),
            )
            # Seed with an EDGE, not a node. At |S| = 1 the growth rule admits
            # any neighbour (⌈γ·1⌉ = 1), so a node-seeded greedy will happily
            # pick a spurious bridge and then stall at size 2 — which splits
            # the two real cliques it was supposed to keep apart. Choosing the
            # neighbour with the most common neighbours starts inside a dense
            # region instead: a bridge edge has zero triangles, a clique edge
            # has |C| − 2.
            head = order[0]
            nbrs = adj[head] & remaining
            s_set = {head}
            if nbrs:
                s_set.add(
                    min(nbrs, key=lambda y: (-len(adj[head] & adj[y] & remaining), y))
                )
            grew = True
            while grew:
                grew = False
                for x in order:
                    if x in s_set:
                        continue
                    need = int(np.ceil(gamma * len(s_set)))
                    if len(adj[x] & s_set) >= need:
                        s_set.add(x)
                        grew = True
                        break
            out.append(np.array(sorted(s_set), dtype=np.int32))
            remaining -= s_set
    return out


def _components(adj: list[set], keep: np.ndarray) -> list[list[int]]:
    """Connected components over the retained nodes (iterative DFS)."""
    seen = np.zeros(keep.shape[0], dtype=bool)
    comps: list[list[int]] = []
    for start in np.flatnonzero(keep):
        if seen[start]:
            continue
        stack, comp = [int(start)], []
        seen[start] = True
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if keep[y] and not seen[y]:
                    seen[y] = True
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


@dataclass(frozen=True, slots=True)
class Assignment:
    """Reads partitioned across one signature's patterns."""

    columns: np.ndarray  # (S,) candidate indices
    patterns: np.ndarray  # (K, S) int8 states
    mass: np.ndarray  # (K,) float64
    labels: np.ndarray  # (M,) int32 — pattern index per read
    margin: np.ndarray  # (M,) float64 — best minus second-best log-likelihood


def dense_states(states: ReadStates, columns: np.ndarray) -> np.ndarray:
    """``(M, S) int8`` states at ``columns``: ``MAJOR``, a state, or
    ``UNOBSERVED``."""
    cols = np.asarray(columns, dtype=np.int64)
    s = cols.shape[0]
    pos = {int(c): j for j, c in enumerate(cols)}
    x = np.full((states.n_members, s), MAJOR, dtype=np.int8)
    covered = (cols[None, :] >= states.obs_lo[:, None]) & (
        cols[None, :] < states.obs_hi[:, None]
    )
    x[~covered & ~states.is_coverage[cols][None, :]] = UNOBSERVED
    for i in range(states.n_members):
        a, b = states.ptr[i], states.ptr[i + 1]
        for v, st in zip(states.v[a:b], states.state[a:b]):
            j = pos.get(int(v))
            if j is not None:
                x[i, j] = st
    return x


def resolve_signature(
    states: ReadStates,
    columns: np.ndarray,
    *,
    eps: np.ndarray,
    a_min: float = 3.0,
    max_patterns: int = 8,
    n_rounds: int = 3,
) -> Assignment:
    """Assign every read to one pattern of ``columns`` by likelihood.

    For read ``r`` and pattern ``P_j``, summed over the columns ``r``
    **observes**::

        LL(r, j) = log π_j + Σ_c [ 1{a_r[c] == P_j[c]}·log(1−ε_c)
                                 + 1{a_r[c] != P_j[c]}·log ε_c ]

    An unobserved column contributes exactly **0** — never a symbol, never a
    pseudo-allele.

    Since ``Σ_observed log(1−ε_c)`` does not depend on the pattern, the argmax
    reduces to an ε-weighted Hamming distance over observed columns minus a
    log-prior, with ``w_c = log((1−ε_c)/ε_c)``. That reduction is the physical
    content: a substitution column (ε = 0.003) is worth 5.8 nats and a 22%
    homopolymer column 1.27, so the scheme discounts the unreliable column
    with no special case. It also means **"matches nothing" is not a case** —
    exact-row grouping has one, a metric does not. A read observing none of
    the signature scores ``log π_j`` against every pattern and lands on the
    prior with ``margin ≈ 0``, which reports the ambiguity: a read that *said
    nothing* placed on the most probable node is not the same as modal
    absorption folding in a read that *disagreed*.

    Patterns are seeded from distinct observed state vectors over reads
    observing all of ``columns`` with mass ≥ ``a_min``, then refined by
    ``n_rounds`` of (assign → re-estimate each pattern as its members'
    weighted per-column majority). This is not ``np.unique(A, axis=0)`` in
    disguise: the key never contains unobserved, it is over a denoised clique
    rather than every selected column, assignment is by likelihood rather than
    byte identity, and patterns are re-estimated from their members rather
    than frozen at an observed row.
    """
    cols = np.asarray(columns, dtype=np.int64)
    x = dense_states(states, cols)
    w = states.weight
    eps = np.clip(np.asarray(eps, dtype=np.float64), 1e-9, 0.5 - 1e-9)
    wc = np.log((1.0 - eps) / eps)

    observed = x != UNOBSERVED
    # Seed over the columns anyone observes. A signature column that NO read
    # observes must not collapse the seeding to a single all-major pattern —
    # an unobserved column contributes 0 to every score, so it can only ever
    # be major, and it must not decide how many patterns exist.
    usable = observed.any(axis=0)
    patterns = np.full((1, cols.shape[0]), MAJOR, dtype=np.int8)
    prior = np.array([max(float(w.sum()), 1.0)])
    if usable.any():
        full = np.flatnonzero(observed[:, usable].all(axis=1))
        if full.size:
            keys, inv = np.unique(x[np.ix_(full, np.flatnonzero(usable))],
                                  axis=0, return_inverse=True)
            mass = np.bincount(inv, weights=w[full], minlength=keys.shape[0])
            order = np.argsort(-mass, kind="stable")
            order = order[mass[order] >= a_min][:max_patterns]
            if order.size:
                patterns = np.full((order.size, cols.shape[0]), MAJOR, dtype=np.int8)
                patterns[:, usable] = keys[order]
                prior = mass[order]

    labels = np.zeros(states.n_members, dtype=np.int32)
    margin = np.zeros(states.n_members, dtype=np.float64)
    for _ in range(max(1, n_rounds)):
        ll = np.empty((states.n_members, patterns.shape[0]), dtype=np.float64)
        log_prior = np.log(np.maximum(prior, 1e-12))
        for j in range(patterns.shape[0]):
            mism = observed & (x != patterns[j][None, :])
            ll[:, j] = log_prior[j] - mism.astype(np.float64) @ wc
        labels = np.argmax(ll, axis=1).astype(np.int32)
        if patterns.shape[0] > 1:
            part = np.partition(ll, -2, axis=1)
            margin = part[:, -1] - part[:, -2]
        prior = np.bincount(labels, weights=w, minlength=patterns.shape[0])
        patterns = _reestimate(x, observed, labels, w, patterns)
    return Assignment(
        columns=cols, patterns=patterns, mass=prior, labels=labels, margin=margin
    )


def _reestimate(x, observed, labels, w, patterns):
    """Weighted per-column majority over each pattern's own members."""
    out = patterns.copy()
    n_states = int(max(int(x.max()) + 1, UNCOVERED + 1))
    for j in range(patterns.shape[0]):
        rows = np.flatnonzero(labels == j)
        if rows.size == 0:
            continue
        for s in range(patterns.shape[1]):
            sel = rows[observed[rows, s]]
            if sel.size == 0:
                continue
            votes = np.bincount(
                x[sel, s].astype(np.int64), weights=w[sel], minlength=n_states
            )
            out[j, s] = np.int8(np.argmax(votes))
    return out


__all__ = [
    "MAJOR",
    "UNCOVERED",
    "UNOBSERVED",
    "Assignment",
    "CovarianceGraph",
    "ReadStates",
    "covariance_graph",
    "dense_states",
    "quasi_cliques",
    "read_states",
    "resolve_signature",
]
