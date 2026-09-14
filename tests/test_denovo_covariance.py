"""Covariance retention, quasi-cliques and likelihood assignment.

Built on hand-made ``(candidate index, read index)`` arrays, not on an
end-to-end consensus fixture. The whole point of keeping ``covariance.py``
free of ``ConsensusResult`` / ``MemberSpec`` / edlib is that the clique and
assignment claims become *proved* rather than observed through three layers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.orfem.columns import (
    ROUTE_ALLELIC,
    ROUTE_COVERAGE,
    CandidateSet,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.covariance import (
    MAJOR,
    UNCOVERED,
    UNOBSERVED,
    CovarianceGraph,
    ReadStates,
    covariance_graph,
    dense_states,
    quasi_cliques,
    resolve_signature,
)


# ── builders ──────────────────────────────────────────────────────────


def _cand(n, *, route=None, run_len=None, boundary_mass=None,
          boundary_expected=None, eps=0.003):
    route = np.full(n, ROUTE_ALLELIC, np.int8) if route is None else np.asarray(
        route, np.int8
    )
    return CandidateSet(
        columns=np.arange(n, dtype=np.int64),
        route=route,
        run_len=np.ones(n, np.int32) if run_len is None else np.asarray(run_len, np.int32),
        effect=np.full(n, 0.5),
        eps=np.full(n, eps),
        major=np.zeros(n, np.int8),
        boundary_mass=(
            np.zeros(n) if boundary_mass is None else np.asarray(boundary_mass, float)
        ),
        boundary_expected=(
            np.full(n, 0.5)
            if boundary_expected is None
            else np.asarray(boundary_expected, float)
        ),
        p_value=np.zeros(n),
        q_cut=0.01,
        n_allelic=n,
        n_coverage=0,
        n_collapsed=0,
        n_capped=0,
        n_tested=n,
    )


def _states(rows, n_v, *, route=None, spans=None, weights=None):
    """rows = [[(v, state), ...], ...] — one list of non-MAJOR states per read."""
    m = len(rows)
    route = np.full(n_v, ROUTE_ALLELIC, np.int8) if route is None else np.asarray(route)
    ptr = np.zeros(m + 1, dtype=np.int64)
    vs, ss = [], []
    for i, r in enumerate(rows):
        r = sorted(r)
        ptr[i + 1] = ptr[i] + len(r)
        vs.extend(v for v, _ in r)
        ss.extend(s for _, s in r)
    if spans is None:
        spans = [(0, n_v)] * m
    lo = np.array([a for a, _ in spans], dtype=np.int32)
    hi = np.array([b for _, b in spans], dtype=np.int32)
    return ReadStates(
        ptr=ptr,
        v=np.array(vs, dtype=np.int32),
        state=np.array(ss, dtype=np.int8),
        obs_lo=lo,
        obs_hi=hi,
        weight=(np.ones(m) if weights is None else np.asarray(weights, float)),
        is_coverage=(np.asarray(route) & ROUTE_COVERAGE).astype(bool),
        n_members=m,
        n_candidates=n_v,
    )


# ── retention ─────────────────────────────────────────────────────────


def test_a_lone_column_earns_no_edge():
    """The rule the whole design turns on, and its stated cost: a systematic
    22% homopolymer indel covaries with nothing and dies here — and so does a
    lone real frameshift."""
    rows = [[(0, 2)] if i < 22 else [] for i in range(100)]
    g = covariance_graph(_states(rows, 3), _cand(3))
    assert not g.keep.any()
    assert g.n_significant == 0


def test_two_linked_columns_are_both_retained():
    rows = [[(0, 2), (1, 2)] if i < 30 else [] for i in range(100)]
    g = covariance_graph(_states(rows, 3), _cand(3))
    assert g.keep.tolist() == [True, True, False]
    assert g.edges.tolist() == [[0, 1]]


def test_an_indel_with_a_covarying_partner_does_become_a_node():
    """The sibling of ``test_a_lone_column_earns_no_edge``: the rule has to be
    pinned in both directions, or it is untestable which half fired."""
    rows = [[(0, 5), (2, 3)] if i < 25 else [] for i in range(100)]
    g = covariance_graph(_states(rows, 4), _cand(4))
    assert g.keep[[0, 2]].all()
    assert not g.keep[[1, 3]].any()


def test_mutual_exclusivity_is_an_edge_too():
    """Depletion is evidence. On the synthetic fused template gene A's columns
    and gene B's columns describe the SAME read partition from opposite sides,
    so their co-occurrence is exactly zero — a one-sided test finds no edge,
    they become two signatures, and the state-tuple product then re-splits the
    template along a partition it had already made."""
    rows = [[(0, 2)] if i < 100 else [(1, 2)] for i in range(200)]
    g = covariance_graph(_states(rows, 2), _cand(2))
    assert g.edges.tolist() == [[0, 1]]
    assert g.keep.all()


def test_independent_columns_are_not_linked():
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(400):
        r = []
        if rng.random() < 0.3:
            r.append((0, 2))
        if rng.random() < 0.3:
            r.append((1, 2))
        rows.append(r)
    g = covariance_graph(_states(rows, 2), _cand(2))
    assert g.n_significant == 0


def test_min_n11_is_the_floor_on_observed_co_occurrence():
    """The floor is on observed counts, not on a chi-square approximation's
    validity. Two co-occurrences is below ``min_n11``, so the pair is never
    considered — and candidacy's ``a_min`` has already refused both columns
    upstream, which is what actually stops a two-read terminal extension
    block (a 2-clique is a clique) from spawning a template every round."""
    rows = [[(0, 2), (1, 2)] if i < 2 else [] for i in range(60)]
    g = covariance_graph(_states(rows, 2), _cand(2))
    assert g.n_pairs_seen == 0
    assert not g.keep.any()


def test_a_low_frequency_linked_pair_survives():
    """The measured cost of the chi-square expected-cell rule this replaces:
    it demanded k >= sqrt(5n) — 51 reads of 520 — and so rejected exactly the
    low-frequency linked proteoform the design exists to keep."""
    rows = [[(0, 2), (1, 2)] if i < 20 else [] for i in range(520)]
    g = covariance_graph(_states(rows, 2), _cand(2))
    assert g.keep[[0, 1]].all()
    assert g.p_value[0] < 1e-30


# ── the coverage route ────────────────────────────────────────────────


def _ramp_states(m=300, n_v=40, seed=3):
    """A monotone 3' degradation ramp: every read starts at 0, ends somewhere
    different, so uncovered sets are nested rather than identical."""
    rng = np.random.default_rng(seed)
    ends = np.sort(rng.integers(1, n_v, m))
    route = np.full(n_v, ROUTE_COVERAGE, np.int8)
    rows, spans = [], []
    for e in ends:
        rows.append([(v, UNCOVERED) for v in range(int(e), n_v)])
        spans.append((0, int(e)))
    return _states(rows, n_v, route=route, spans=spans), route


def test_a_ramp_is_maximally_associated_and_is_still_rejected():
    """Measured: ramp columns are *nested*, hence maximally associated — the
    opposite of the spec's "uncorrelated". A pairwise test would make the ramp
    one large quasi-clique and spawn arbitrary length nodes, so pure-coverage
    pairs are excluded from testing rather than thresholded."""
    st, route = _ramp_states()
    # The association is real and enormous — this is the finding, stated.
    x = dense_states(st, np.array([10, 20]))
    unc = x == UNCOVERED
    n11 = int((unc[:, 0] & unc[:, 1]).sum())
    assert n11 == int(unc[:, 0].sum()), "nested: every read uncovered at 10 is at 20"

    cand = _cand(route.shape[0], route=route, run_len=np.ones(route.shape[0]))
    g = covariance_graph(st, cand)
    assert g.n_pairs_seen == 0, "no pure-coverage pair is even considered"
    assert not g.keep.any()
    # And the block is never computed, not computed then discarded: on a ramp
    # every candidate is pure-coverage, so the whole pass costs nothing.
    assert g.sum_k2 == 0


def test_a_collapsed_start_mode_earns_a_self_edge():
    """A real alternative start: many reads agreeing on ONE boundary. Run
    collapse has already proved the columns are identical, so what is judged
    is the mass at the boundary that delimits them."""
    n_v = 6
    route = np.full(n_v, ROUTE_COVERAGE, np.int8)
    rows, spans = [], []
    for i in range(60):
        if i < 30:
            rows.append([(v, UNCOVERED) for v in range(3)])
            spans.append((3, n_v))
        else:
            rows.append([])
            spans.append((0, n_v))
    cand = _cand(
        n_v,
        route=route,
        run_len=[3, 1, 1, 1, 1, 1],
        boundary_mass=[30.0, 0, 0, 0, 0, 0],
        boundary_expected=[0.2, 0, 0, 0, 0, 0],
    )
    g = covariance_graph(_states(rows, n_v, route=route, spans=spans), cand)
    assert g.keep[0]
    assert g.edges.tolist() == [[0, 0]], "a self-edge, standing for the run"


def test_a_ramp_boundary_is_what_a_uniform_spread_would_give():
    """Same machinery, same run_len >= 2. What separates a ramp from a mode is
    not mass and not fraction — once endpoints are clustered, a ramp's
    boundaries carry ~10% of the template each — but *concentration* against
    the local endpoint density."""
    n_v = 6
    route = np.full(n_v, ROUTE_COVERAGE, np.int8)
    rows = [[(v, UNCOVERED) for v in range(3)] if i < 6 else [] for i in range(60)]
    spans = [(3, n_v) if i < 6 else (0, n_v) for i in range(60)]
    common = dict(route=route, run_len=[3, 1, 1, 1, 1, 1],
                  boundary_mass=[6.0, 0, 0, 0, 0, 0])
    st = _states(rows, n_v, route=route, spans=spans)
    # A ramp: 6 endpoints here is what the neighbourhood averages anyway.
    ramp = _cand(n_v, boundary_expected=[6.0, 0, 0, 0, 0, 0], **common)
    assert not covariance_graph(st, ramp).keep.any()
    # A mode: the same 6 endpoints against a near-empty neighbourhood.
    mode = _cand(n_v, boundary_expected=[0.05, 0, 0, 0, 0, 0], **common)
    assert covariance_graph(st, mode).keep[0]


def test_a_coverage_column_can_link_to_an_allelic_one():
    """Cross-route edges are tested: an alternative start covarying with a
    substitution is real evidence, and the interval confound does not apply
    because the allelic column's own marginal is over reads that cover it."""
    n_v = 4
    route = np.array([ROUTE_COVERAGE, ROUTE_ALLELIC, ROUTE_ALLELIC, ROUTE_ALLELIC], np.int8)
    rows, spans = [], []
    for i in range(80):
        if i < 30:  # the short form, which also carries a substitution at 2
            rows.append([(0, UNCOVERED), (2, 3)])
            spans.append((1, n_v))
        else:
            rows.append([])
            spans.append((0, n_v))
    g = covariance_graph(_states(rows, n_v, route=route, spans=spans),
                         _cand(n_v, route=route))
    assert g.keep[[0, 2]].all()
    assert sorted(map(tuple, g.edges.tolist())) == [(0, 2)]


# ── quasi-cliques ─────────────────────────────────────────────────────


def _graph(n_v, edges, p=1e-12):
    e = np.array(edges, dtype=np.int32).reshape(-1, 2)
    keep = np.zeros(n_v, dtype=bool)
    keep[e.ravel()] = True
    return CovarianceGraph(
        edges=e,
        p_value=np.full(e.shape[0], p),
        keep=keep,
        q_cut=0.01,
        n_pairs_seen=e.shape[0],
        n_tested=e.shape[0],
        n_significant=e.shape[0],
        n_reads_capped=0,
        sum_k2=0,
    )


def test_one_clean_split_is_one_signature():
    assert [c.tolist() for c in quasi_cliques(_graph(4, [(0, 1), (0, 2), (1, 2)]))] == [
        [0, 1, 2]
    ]


def test_two_independent_splits_are_two_signatures():
    g = _graph(6, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)])
    assert sorted(c.tolist() for c in quasi_cliques(g)) == [[0, 1, 2], [3, 4, 5]]


def test_one_spurious_edge_does_not_merge_two_splits():
    """The failure connected components alone cannot survive: the bridge
    endpoint needs ``1 >= ceil(gamma*|S|)``, which is false for |S| >= 2."""
    g = _graph(6, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)])
    cliques = sorted(c.tolist() for c in quasi_cliques(g))
    assert cliques == [[0, 1, 2], [3, 4, 5]]


def test_a_missing_edge_still_gives_one_signature():
    """The other direction: maximal cliques would give two overlapping answers
    where one signature is right. gamma = 0.6 is the 'near' in quasi-clique."""
    g = _graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])  # 1-3 missing
    assert [c.tolist() for c in quasi_cliques(g)] == [[0, 1, 2, 3]]


def test_greedy_matches_the_exact_decomposition_on_disjoint_cliques():
    """Bron-Kerbosch as a test oracle, per the plan: on any disjoint union of
    cliques the greedy result must equal the exact one."""
    rng = np.random.default_rng(11)
    for _ in range(40):
        sizes = rng.integers(1, 5, rng.integers(1, 4)).tolist()
        if sum(sizes) > 12:
            continue
        nodes, edges, nxt = [], [], 0
        for s in sizes:
            grp = list(range(nxt, nxt + s))
            nxt += s
            nodes.append(grp)
            edges += list(itertools.combinations(grp, 2))
        if not edges:
            continue
        got = sorted(c.tolist() for c in quasi_cliques(_graph(nxt, edges)))
        want = sorted(g for g in nodes if len(g) > 1)
        assert got == want


def test_clique_output_is_invariant_to_node_labelling():
    """Greedy is order-dependent, so determinism is part of the contract."""
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]
    base = sorted(len(c) for c in quasi_cliques(_graph(6, edges)))
    for _ in range(10):
        assert sorted(len(c) for c in quasi_cliques(_graph(6, edges))) == base


# ── likelihood assignment ─────────────────────────────────────────────


def test_the_weighted_hamming_reduction_is_argmax_equivalent():
    """The implementation drops the pattern-independent sum(log(1-eps)) term.
    That must not change any argmax — pinned against the literal LL."""
    rng = np.random.default_rng(13)
    n_v, m = 5, 60
    rows = []
    for i in range(m):
        rows.append([(v, 2) for v in range(n_v) if rng.random() < 0.5])
    st = _states(rows, n_v)
    eps = np.array([0.003, 0.02, 0.22, 0.05, 0.004])
    a = resolve_signature(st, np.arange(n_v), eps=eps)

    x = dense_states(st, np.arange(n_v))
    obs = x != UNOBSERVED
    ll = np.zeros((m, a.patterns.shape[0]))
    for j, p in enumerate(a.patterns):
        match = obs & (x == p[None, :])
        mism = obs & (x != p[None, :])
        ll[:, j] = (
            np.log(max(a.mass[j], 1e-12))
            + match @ np.log(1 - eps)
            + mism @ np.log(eps)
        )
    assert np.array_equal(np.argmax(ll, axis=1), a.labels)


def test_an_unobserved_column_changes_no_score():
    """The direct fix for the 28.1% of round-6 nodes that differed from their
    major only at uncovered positions."""
    rows = [[(0, 2), (1, 2)] if i < 30 else [] for i in range(60)]
    full = _states(rows, 3)
    part = _states(rows, 3, spans=[(0, 2)] * 60)  # nobody observes column 2
    eps = np.full(3, 0.003)
    a = resolve_signature(full, np.arange(3), eps=eps)
    b = resolve_signature(part, np.arange(3), eps=eps)
    assert np.array_equal(a.labels, b.labels)


def test_a_read_that_observes_nothing_lands_on_the_prior():
    rows = [[(0, 2), (1, 2)] if i < 20 else [] for i in range(60)]
    spans = [(0, 2)] * 59 + [(2, 2)]  # the last read observes neither column
    st = _states(rows, 2, spans=spans)
    a = resolve_signature(st, np.arange(2), eps=np.full(2, 0.003))
    heaviest = int(np.argmax(a.mass))
    assert a.labels[-1] == heaviest
    assert a.margin[-1] == pytest.approx(
        np.log(a.mass[heaviest]) - np.log(sorted(a.mass)[-2]), abs=1e-9
    )


def test_unobserved_is_never_a_symbol_in_a_pattern():
    rows = [[] for _ in range(40)]
    st = _states(rows, 3, spans=[(0, 1)] * 40)
    a = resolve_signature(st, np.arange(3), eps=np.full(3, 0.003))
    assert not (a.patterns == UNOBSERVED).any()
    assert (a.patterns == MAJOR).all()


def test_patterns_partition_every_read_exactly_once():
    """The invariant node read counts depend on: sum(mass) == total weight."""
    rng = np.random.default_rng(17)
    rows = [[(v, 2) for v in range(3) if rng.random() < 0.4] for _ in range(200)]
    w = rng.integers(1, 9, 200).astype(float)
    st = _states(rows, 3, weights=w)
    a = resolve_signature(st, np.arange(3), eps=np.full(3, 0.003))
    assert a.mass.sum() == pytest.approx(w.sum())
    assert np.bincount(a.labels, weights=w, minlength=a.mass.shape[0]) == pytest.approx(
        a.mass
    )


def test_an_unreliable_column_is_discounted_not_excluded():
    """A 22% homopolymer column is worth 1.27 nats and a substitution 5.8, so
    one substitution outvotes four homopolymer disagreements with no special
    case."""
    eps = np.array([0.003, 0.22, 0.22, 0.22, 0.22])
    rows = [[(0, 2)] if i < 30 else [] for i in range(60)]
    for i in range(30, 45):  # a group disagreeing at the four weak columns
        rows[i] = [(v, 2) for v in range(1, 5)]
    st = _states(rows, 5)
    a = resolve_signature(st, np.arange(5), eps=eps)
    w = np.log((1 - eps) / eps)
    assert w[0] > 4 * w[1], "one substitution outweighs four homopolymers"
    assert a.patterns.shape[0] >= 2
