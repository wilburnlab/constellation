"""``guarded_set_cover`` — merge guards that hold per group, not per pair.

Merging is decided pairwise; grouping is a star around a claimant. The
load-bearing tests are the two cases a pairwise guard cannot see — a hub that
joins two split siblings through itself, and two members each inside the end
tolerance of the hub yet twice the tolerance from each other — and the one
that keeps the guards free where they cannot bind: array-for-array equality
with ``greedy_set_cover``.

Randomized cases are held to ``_by_the_book``, the specification transcribed
with dicts and whole member lists (no rank space, no running min/max, no
row-level shortcut), so both branches of the implementation's walk answer to
the same oracle, and to ``_assert_guards_hold``, which checks the claim in the
function's name on the groups themselves.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
    ComponentResult,
    greedy_set_cover,
    guarded_set_cover,
)

A, B, C = 0, 1, 2
#: B is mergeable with A and with C; A and C share no edge.
HUB = [(B, A, 0, 0), (B, C, 0, 0)]
#: Template ids are ``round << 40 | row``, so origins are not small integers.
PARENT, OTHER_PARENT = (3 << 40) | 17, (3 << 40) | 4


def _cover(
    n, order, edges, *, tol_5p=30, tol_3p=30, origin=None, identical=None
) -> ComponentResult:
    """Run on edges written as ``(a, b, off_5p, off_3p)`` rows."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 4)
    return guarded_set_cover(
        n,
        np.asarray(order, dtype=np.int64),
        e[:, 0],
        e[:, 1],
        e[:, 2],
        e[:, 3],
        tol_5p=tol_5p,
        tol_3p=tol_3p,
        origin=None if origin is None else np.asarray(origin, dtype=np.int64),
        edge_identical=None if identical is None else np.asarray(identical, bool),
    )


def _groups(res: ComponentResult) -> list[list[int]]:
    out: dict[int, list[int]] = {}
    for node, cid in enumerate(res.cluster_of.tolist()):
        out.setdefault(cid, []).append(node)
    return sorted(out.values())


def _span(offsets) -> int:
    return max(offsets) - min(offsets)


def _first_edges(a, b, o5, o3, identical):
    """``(claimant, neighbour) -> (off_5p, off_3p, identical)``, first copy wins."""
    edge: dict[tuple[int, int], tuple[int, int, bool]] = {}
    for e in range(len(a)):
        u, v = int(a[e]), int(b[e])
        if u == v or (u, v) in edge:
            continue
        same = bool(identical[e]) if identical is not None else False
        edge[(u, v)] = (int(o5[e]), int(o3[e]), same)
        edge[(v, u)] = (-int(o5[e]), -int(o3[e]), same)
    return edge


def _by_the_book(
    n, order, a, b, o5, o3, *, tol_5p, tol_3p, origin=None, identical=None
):
    """The specification, transcribed: one neighbour at a time, whole lists."""
    edge = _first_edges(a, b, o5, o3, identical)
    rank = {int(u): i for i, u in enumerate(order)}
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for u, v in edge:
        nbrs[u].append(v)
    label = [-1] * n
    centroids: list[int] = []
    for c in (int(u) for u in order):
        if label[c] >= 0:
            continue
        label[c] = len(centroids)
        centroids.append(c)
        members = [(c, 0, 0, True)]
        for m in sorted(nbrs[c], key=rank.__getitem__):
            if label[m] >= 0:
                continue
            m5, m3, same = edge[(c, m)]
            if _span([x[1] for x in members] + [m5]) > tol_5p:
                continue
            if _span([x[2] for x in members] + [m3]) > tol_3p:
                continue
            if origin is not None and origin[m] >= 0:
                clash = [x for x in members if origin[x[0]] == origin[m]]
                if clash and not (same and all(x[3] for x in clash)):
                    continue
            members.append((m, m5, m3, same))
            label[m] = label[c]
    return np.asarray(label, dtype=np.int64), np.asarray(centroids, dtype=np.int64)


def _assert_guards_hold(
    res, a, b, o5, o3, *, tol_5p, tol_3p, origin=None, identical=None
):
    """Every group is a star around its centroid and passes both guards whole."""
    edge = _first_edges(a, b, o5, o3, identical)
    n_groups = res.centroid_uniq.shape[0]
    assert res.cluster_of[res.centroid_uniq].tolist() == list(range(n_groups))
    assert set(res.cluster_of.tolist()) == set(range(n_groups))
    for cid, c in enumerate(res.centroid_uniq.tolist()):
        seen = {c: (0, 0, True)}
        for m in np.flatnonzero(res.cluster_of == cid).tolist():
            if m != c:
                assert (c, m) in edge, "a member must be the claimant's neighbour"
                seen[m] = edge[(c, m)]
        assert _span([v[0] for v in seen.values()]) <= tol_5p
        assert _span([v[1] for v in seen.values()]) <= tol_3p
        if origin is None:
            continue
        holders: dict[int, list[int]] = {}
        for m in seen:
            if origin[m] >= 0:
                holders.setdefault(int(origin[m]), []).append(m)
        for kin in holders.values():
            assert len(kin) == 1 or all(seen[m][2] for m in kin)


def _family_graph(rng, n, n_edges, *, spread, n_twin_classes=None):
    """A random graph whose offsets are mutually consistent.

    Each node has a 5' and a 3' position and an edge's offset is ``b``'s minus
    ``a``'s, as extents measured on real templates would be. Nodes of one twin
    class share both positions, and an edge inside a class is byte-identical.
    Each pair appears once and never as a self-edge.
    """
    twin_class = (
        np.arange(n) if n_twin_classes is None else rng.integers(0, n_twin_classes, n)
    )
    n_classes = int(twin_class.max()) + 1
    pos_5p = rng.integers(0, spread + 1, n_classes)[twin_class]
    pos_3p = rng.integers(0, spread + 1, n_classes)[twin_class]
    a = rng.integers(0, n, n_edges)
    b = rng.integers(0, n, n_edges)
    pair = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    a, b = pair // n, pair % n
    a, b = a[a != b], b[a != b]
    swap = rng.random(a.shape[0]) < 0.5  # the claimant is not always ``a``
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    return (
        a,
        b,
        pos_5p[b] - pos_5p[a],
        pos_3p[b] - pos_3p[a],
        twin_class[a] == twin_class[b],
    )


def _greedy_order(abundance, seq_len):
    """The priority order ``greedy_set_cover`` derives for itself."""
    ids = np.arange(abundance.shape[0], dtype=np.int64)
    return np.lexsort((ids, -seq_len.astype(np.int64), -abundance.astype(np.int64)))


# ── the hub ───────────────────────────────────────────────────────────


def test_the_unguarded_cover_joins_split_siblings_through_a_hub():
    """The defect, pinned on the shipped function: B best, all three collapse."""
    reads = np.array([5, 9, 1], dtype=np.int64)
    res = greedy_set_cover(
        3, reads, np.full(3, 900), np.array([B, B]), np.array([A, C])
    )
    assert _groups(res) == [[A, B, C]]


@pytest.mark.parametrize("order, joined, alone", [([B, A, C], A, C), ([B, C, A], C, A)])
def test_a_hub_keeps_two_split_siblings_apart(order, joined, alone):
    """A and C came from one split; exactly one joins B — the better ranked."""
    res = _cover(3, order, HUB, origin=[PARENT, OTHER_PARENT, PARENT])
    assert _groups(res) == sorted([sorted([B, joined]), [alone]])
    assert res.centroid_uniq.tolist() == [B, alone]


def test_the_same_hub_without_origins_merges_all_three():
    res = _cover(3, [B, A, C], HUB, origin=None)
    assert _groups(res) == [[A, B, C]]
    assert res.centroid_uniq.tolist() == [B]


def test_templates_with_no_recorded_split_are_nobodys_kin():
    """``-1`` is the absence of an origin, not an origin two nodes can share."""
    res = _cover(3, [B, A, C], HUB, origin=[-1, OTHER_PARENT, -1])
    assert _groups(res) == [[A, B, C]]


def test_a_claimant_refuses_its_own_sibling():
    res = _cover(2, [A, B], [(A, B, 0, 0)], origin=[PARENT, PARENT])
    assert _groups(res) == [[A], [B]]


# ── byte-identical twins ──────────────────────────────────────────────


def test_byte_identical_siblings_merge_despite_sharing_an_origin():
    """Reads cannot tell them apart, so merging them moves nothing."""
    res = _cover(
        3, [B, A, C], HUB, origin=[PARENT, OTHER_PARENT, PARENT], identical=[1, 1]
    )
    assert _groups(res) == [[A, B, C]]


@pytest.mark.parametrize("identical", [[True, False], [False, True]])
def test_one_byte_identical_edge_does_not_exempt_the_pair(identical):
    """Both siblings must equal the claimant for them to equal each other."""
    res = _cover(
        3, [B, A, C], HUB, origin=[PARENT, OTHER_PARENT, PARENT], identical=identical
    )
    assert _groups(res) == [[A, B], [C]]


def test_a_claimant_takes_its_own_sibling_when_byte_identical():
    res = _cover(2, [A, B], [(A, B, 0, 0)], origin=[PARENT, PARENT], identical=[1])
    assert _groups(res) == [[A, B]]


def test_a_third_twin_joins_two_already_merged():
    """The exemption is a property of the whole kin set, not of its first pair."""
    edges = [(0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0)]
    origin = [OTHER_PARENT, PARENT, PARENT, PARENT]
    res = _cover(4, [0, 1, 2, 3], edges, origin=origin, identical=[1, 1, 1])
    assert _groups(res) == [[0, 1, 2, 3]]
    res = _cover(4, [0, 1, 2, 3], edges, origin=origin, identical=[1, 1, 0])
    assert _groups(res) == [[0, 1, 2], [3]]


# ── extent ────────────────────────────────────────────────────────────


def _at(end, offset):
    return (offset, 0) if end == "5p" else (0, offset)


@pytest.mark.parametrize("end", ["5p", "3p"])
def test_two_members_on_the_same_side_of_the_hub_merge(end):
    edges = [(0, 1, *_at(end, 25)), (0, 2, *_at(end, 25))]
    assert _groups(_cover(3, [0, 1, 2], edges)) == [[0, 1, 2]]


@pytest.mark.parametrize("end", ["5p", "3p"])
def test_two_members_on_opposite_sides_of_the_hub_do_not_both_join(end):
    """Each is 25 nt from the hub and passes alone; they are 50 nt apart."""
    edges = [(0, 1, *_at(end, 25)), (0, 2, *_at(end, -25))]
    res = _cover(3, [0, 1, 2], edges)
    assert _groups(res) == [[0, 1], [2]]
    assert _groups(_cover(3, [0, 2, 1], edges)) == [[0, 2], [1]]


@pytest.mark.parametrize("end", ["5p", "3p"])
def test_a_lone_member_is_measured_against_the_claimant(end):
    """One neighbour has nothing else to disagree with; the claimant sits at 0."""
    for offset in (31, -31):
        assert _groups(_cover(2, [0, 1], [(0, 1, *_at(end, offset))])) == [[0], [1]]
    for offset in (30, -30):
        assert _groups(_cover(2, [0, 1], [(0, 1, *_at(end, offset))])) == [[0, 1]]


def test_each_end_answers_to_its_own_tolerance():
    edges = [(0, 1, 25, 25), (0, 2, -25, -25)]
    assert _groups(_cover(3, [0, 1, 2], edges, tol_5p=50, tol_3p=50)) == [[0, 1, 2]]
    assert _groups(_cover(3, [0, 1, 2], edges, tol_5p=50, tol_3p=49)) == [[0, 1], [2]]
    assert _groups(_cover(3, [0, 1, 2], edges, tol_5p=49, tol_3p=50)) == [[0, 1], [2]]


def test_offsets_are_read_from_the_claimant_whichever_endpoint_it_is():
    """``(1, 0, -25)`` says the hub reaches 25 short of 1: 1 sits at +25."""
    same_side = [(1, 0, -25, 0), (0, 2, 25, 0)]
    assert _groups(_cover(3, [0, 1, 2], same_side)) == [[0, 1, 2]]
    opposite = [(1, 0, 25, 0), (0, 2, 25, 0)]
    assert _groups(_cover(3, [0, 1, 2], opposite)) == [[0, 1], [2]]


def test_a_refusal_leaves_the_group_as_it_was():
    """2 would stretch the group to 35 and is refused; 3 then fits at exactly 30."""
    edges = [(0, 1, 20, 0), (0, 2, -15, 0), (0, 3, -10, 0)]
    assert _groups(_cover(4, [0, 1, 2, 3], edges)) == [[0, 1, 3], [2]]


# ── radius 1 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("order", list(itertools.permutations([A, B, C])))
def test_a_chain_merges_only_through_its_middle(order):
    """A~B~C with no A~C edge: only B is a direct neighbour of both ends."""
    res = _cover(3, list(order), [(A, B, 0, 0), (B, C, 0, 0)])
    assert (res.cluster_of[A] == res.cluster_of[C]) == (order[0] == B)


def test_the_middle_of_a_chain_still_answers_to_the_guards():
    chain = [(A, B, -25, 0), (B, C, -25, 0)]  # A at +25 of B, C at -25
    assert _groups(_cover(3, [B, A, C], chain)) == [[A, B], [C]]
    kin = [PARENT, OTHER_PARENT, PARENT]
    flush = [(A, B, 0, 0), (B, C, 0, 0)]
    assert _groups(_cover(3, [B, C, A], flush, origin=kin)) == [[A], [B, C]]


def test_a_refused_neighbour_claims_for_itself_at_its_turn():
    edges = [(0, 1, 25, 0), (0, 2, -25, 0), (2, 3, 0, 0)]
    res = _cover(4, [0, 1, 2, 3], edges)
    assert _groups(res) == [[0, 1], [2, 3]]
    assert res.centroid_uniq.tolist() == [0, 2]


def test_a_refused_neighbour_can_be_claimed_by_a_later_claimant():
    edges = [(0, 1, 25, 0), (0, 2, -25, 0), (3, 2, 0, 0)]
    res = _cover(4, [0, 1, 3, 2], edges)
    assert _groups(res) == [[0, 1], [2, 3]]
    assert res.centroid_uniq.tolist() == [0, 3]


# ── against greedy_set_cover and the specification ───────────────────


@pytest.mark.parametrize("origins", ["disabled", "none recorded", "all distinct"])
def test_it_equals_greedy_set_cover_when_no_guard_can_bind(origins):
    rng = np.random.default_rng(20260927)
    n = 400
    a, b, o5, o3, _ = _family_graph(rng, n, 900, spread=30)
    abundance = rng.integers(1, 6, n)  # ties, so length and id break them
    seq_len = rng.integers(300, 3000, n)
    origin = {
        "disabled": None,
        "none recorded": np.full(n, -1, dtype=np.int64),
        "all distinct": (5 << 40) | rng.permutation(n).astype(np.int64),
    }[origins]

    greedy = greedy_set_cover(n, abundance, seq_len, a, b)
    res = guarded_set_cover(
        n,
        _greedy_order(abundance, seq_len),
        a,
        b,
        o5,
        o3,
        tol_5p=30,
        tol_3p=30,
        origin=origin,
    )

    assert greedy.centroid_uniq.shape[0] < n  # the graph does group something
    np.testing.assert_array_equal(res.cluster_of, greedy.cluster_of)
    np.testing.assert_array_equal(res.centroid_uniq, greedy.centroid_uniq)
    assert res.cluster_of.dtype == np.int64 and res.centroid_uniq.dtype == np.int64


def _binding_case(seed):
    """Extents wider than the tolerance and origins shared within a row."""
    rng = np.random.default_rng(seed)
    n = 240
    a, b, o5, o3, same = _family_graph(rng, n, 1100, spread=45, n_twin_classes=24)
    origin = np.where(
        rng.random(n) < 0.2, -1, (2 << 40) | rng.integers(0, 30, n)
    ).astype(np.int64)
    return n, rng.permutation(n), a, b, o5, o3, origin, same


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kin_guard", [False, True])
def test_it_follows_the_specification_where_the_guards_bind(seed, kin_guard):
    n, order, a, b, o5, o3, origin, same = _binding_case(seed)
    if not kin_guard:
        origin = same = None
    tol = dict(tol_5p=30, tol_3p=30)
    res = guarded_set_cover(
        n, order, a, b, o5, o3, origin=origin, edge_identical=same, **tol
    )

    label, centroids = _by_the_book(
        n, order, a, b, o5, o3, origin=origin, identical=same, **tol
    )
    np.testing.assert_array_equal(res.cluster_of, label)
    np.testing.assert_array_equal(res.centroid_uniq, centroids)
    _assert_guards_hold(res, a, b, o5, o3, origin=origin, identical=same, **tol)

    # The case has teeth: unguarded, the same walk groups differently.
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    free = greedy_set_cover(n, n - inv, np.ones(n, dtype=np.int64), a, b)
    assert not np.array_equal(free.cluster_of, res.cluster_of)


def test_the_twin_exemption_is_exercised_by_the_randomized_cases():
    """Otherwise the oracle comparison says nothing about byte-identical kin."""
    merged_kin = 0
    for seed in range(6):
        n, order, a, b, o5, o3, origin, same = _binding_case(seed)
        res = guarded_set_cover(
            n,
            order,
            a,
            b,
            o5,
            o3,
            tol_5p=30,
            tol_3p=30,
            origin=origin,
            edge_identical=same,
        )
        for cid in range(res.centroid_uniq.shape[0]):
            kin = origin[res.cluster_of == cid]
            kin = kin[kin >= 0]
            merged_kin += kin.shape[0] - np.unique(kin).shape[0]
    assert merged_kin > 0


# ── what the input may look like ──────────────────────────────────────


def test_the_result_does_not_depend_on_edge_order_or_orientation():
    n, order, a, b, o5, o3, origin, same = _binding_case(11)
    kw = dict(tol_5p=30, tol_3p=30, origin=origin)
    base = guarded_set_cover(n, order, a, b, o5, o3, edge_identical=same, **kw)

    rng = np.random.default_rng(12)
    for _ in range(5):
        by = rng.permutation(a.shape[0])
        flip = rng.random(a.shape[0]) < 0.5
        res = guarded_set_cover(
            n,
            order,
            np.where(flip, b, a)[by],
            np.where(flip, a, b)[by],
            np.where(flip, -o5, o5)[by],
            np.where(flip, -o3, o3)[by],
            edge_identical=same[by],
            **kw,
        )
        np.testing.assert_array_equal(res.cluster_of, base.cluster_of)
        np.testing.assert_array_equal(res.centroid_uniq, base.centroid_uniq)


def test_self_edges_and_repeated_pairs_change_nothing():
    """The repeats contradict their originals, so using one would show."""
    n, order, a, b, o5, o3, origin, same = _binding_case(21)
    kw = dict(tol_5p=30, tol_3p=30, origin=origin)
    base = guarded_set_cover(n, order, a, b, o5, o3, edge_identical=same, **kw)

    rng = np.random.default_rng(22)
    again = rng.choice(a.shape[0], 300, replace=False)
    flip = rng.random(again.shape[0]) < 0.5
    loops = rng.integers(0, n, 50)
    res = guarded_set_cover(
        n,
        order,
        np.concatenate([a, np.where(flip, b[again], a[again]), loops]),
        np.concatenate([b, np.where(flip, a[again], b[again]), loops]),
        np.concatenate([o5, o5[again] + 100, np.full(50, 500)]),
        np.concatenate([o3, o3[again] - 100, np.full(50, -500)]),
        edge_identical=np.concatenate([same, ~same[again], np.ones(50, bool)]),
        **kw,
    )

    np.testing.assert_array_equal(res.cluster_of, base.cluster_of)
    np.testing.assert_array_equal(res.centroid_uniq, base.centroid_uniq)


def test_of_a_repeated_pair_the_first_copy_is_the_one_used():
    assert _groups(_cover(2, [0, 1], [(0, 1, 0, 0), (0, 1, 100, 0)])) == [[0, 1]]
    assert _groups(_cover(2, [0, 1], [(0, 1, 100, 0), (0, 1, 0, 0)])) == [[0], [1]]
    # The same pair written from its other end is still the same pair.
    assert _groups(_cover(2, [0, 1], [(1, 0, 100, 0), (0, 1, 0, 0)])) == [[0], [1]]


def test_a_graph_of_self_edges_alone_is_a_graph_with_no_edges():
    res = _cover(3, [2, 0, 1], [(0, 0, 0, 0), (2, 2, 0, 0)])
    assert res.centroid_uniq.tolist() == [2, 0, 1]


def test_the_edge_table_s_int32_columns_are_accepted():
    res = guarded_set_cover(
        3,
        np.array([B, A, C], dtype=np.int32),
        np.array([B, B], dtype=np.int32),
        np.array([A, C], dtype=np.int32),
        np.array([25, -25], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        tol_5p=30,
        tol_3p=30,
    )
    assert _groups(res) == [[A, B], [C]]


def test_no_nodes_gives_empty_arrays():
    res = _cover(0, [], [])
    assert res.cluster_of.shape == (0,) and res.cluster_of.dtype == np.int64
    assert res.centroid_uniq.shape == (0,) and res.centroid_uniq.dtype == np.int64


def test_no_edges_gives_singletons_in_claim_order():
    res = _cover(3, [2, 0, 1], [], origin=[PARENT, PARENT, PARENT])
    assert res.centroid_uniq.tolist() == [2, 0, 1]
    assert res.cluster_of.tolist() == [1, 2, 0]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(order=[0, 1, 1]), "permutation"),
        (dict(order=[0, 1, 3]), "permutation"),
        (dict(order=[0, 1]), "order must have shape"),
        (dict(edges=[(0, 3, 0, 0)]), "edge endpoints"),
        (dict(edges=[(-1, 2, 0, 0)]), "edge endpoints"),
        (dict(tol_5p=-1), "tolerances"),
        (dict(origin=[PARENT, PARENT]), "origin must have shape"),
        (dict(identical=[True, False]), "one length"),
    ],
)
def test_malformed_input_is_refused_rather_than_grouped(kwargs, match):
    """A bad ``order`` would otherwise come back as a plausible partition."""
    args = dict(n=3, order=[0, 1, 2], edges=[(0, 1, 0, 0)]) | kwargs
    with pytest.raises(ValueError, match=match):
        _cover(**args)
