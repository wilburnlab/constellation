"""The template graph's pair kernel, against known geometry.

Every case builds a pair with a known relation — a truncation, a stagger, an
unrelated end, a skipped segment — and asserts the kernel reports *that*
geometry in nucleotides: which sequence reaches beyond the other at each end,
by how much, and what the span they share costs.

The cases that matter are the ones a local alignment gets wrong. An unrelated
first exon must never come out as an exact pair; a 5' extension on the shorter
template must be an overhang and not a run of edits; two ends that differ must
not both be read as overhang, which would make them free. Several of those are
statements about *every* pair rather than one, so they are checked over seeded
random trials, against edlib itself as the oracle for what the shared span
costs.

None of this needs minimap2.
"""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

edlib = pytest.importorskip("edlib")

from constellation.core.io.schemas import get_schema
from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
    CLUSTER_EDGE_TABLE,
    EDGE_FEATURE_FIELDS,
    GRAPH_KERNEL_VERSION,
    PRODUCED_RELATIONS,
    RELATION_NAMES,
    TEMPLATE_EDGE_TABLE,
    TRUNCATION_NAMES,
    GraphParams,
    PairMeasurement,
    Relation,
    _choose_location,
    edit_budget,
    effective_tolerance,
    relate_pair,
)

P = GraphParams()

_OTHER_BASE = bytes.maketrans(b"ACGT", b"CGTA")
_COMPLEMENT = bytes.maketrans(b"ACGT", b"TGCA")


def rnd(n: int, rng: random.Random) -> bytes:
    return bytes(rng.choices(b"ACGT", k=n))


def substituted(seq: bytes, *positions: int) -> bytes:
    """``seq`` with a different base at each of ``positions``."""
    out = bytearray(seq)
    for p in positions:
        out[p] = _OTHER_BASE[seq[p]]
    return bytes(out)


def without(seq: bytes, at: int, n: int) -> bytes:
    """``seq`` with the ``n`` bases from ``at`` removed."""
    return seq[:at] + seq[at + n :]


def overhangs(m: PairMeasurement) -> tuple[int, int, int, int]:
    """``(src 5', src 3', dst 5', dst 3')``."""
    return (
        m.src_overhang_5p,
        m.src_overhang_3p,
        m.dst_overhang_5p,
        m.dst_overhang_3p,
    )


def shared_span_distance(src: bytes, dst: bytes, m: PairMeasurement) -> int:
    """edlib's global distance of the two cores ``m``'s overhangs define."""
    cs = src[m.src_overhang_5p : len(src) - m.src_overhang_3p]
    cd = dst[m.dst_overhang_5p : len(dst) - m.dst_overhang_3p]
    return edlib.align(cs, cd, mode="NW", task="distance")["editDistance"]


# ──────────────────────────────────────────────────────────────────────
# Parameters, budget, tolerance
# ──────────────────────────────────────────────────────────────────────


def test_the_edit_budget_is_integer_arithmetic():
    """``(1 - 0.93) * 100`` is 6.999… as a float and would truncate to 6."""
    assert edit_budget(100, GraphParams(identity_floor=0.93)) == 7
    assert edit_budget(1400, GraphParams(identity_floor=0.99)) == 14
    assert int((1.0 - 0.93) * 100) == 6  # the form this replaces


def test_the_edit_budget_never_falls_below_its_floor():
    for length in (0, 1, 37, 150, 299, 300):
        assert edit_budget(length, P) == P.min_budget
    assert edit_budget(150, GraphParams(min_budget=5)) == 5
    assert edit_budget(1000, GraphParams(identity_floor=1.0)) == P.min_budget


def test_a_short_template_gets_a_shorter_tolerance():
    assert effective_tolerance(150, P) == (15, 15)
    assert effective_tolerance(300, P) == (30, 30)
    assert effective_tolerance(5000, P) == (30, 30)
    assert effective_tolerance(200, GraphParams(tol_5p=10, tol_3p=50)) == (10, 20)


def test_a_fixed_tolerance_would_call_a_fifth_of_a_short_template_equivalent():
    """30 nt is within the configured tolerance and is a fifth of 150 nt."""
    a = rnd(150, random.Random(1))
    m = relate_pair(a[30:], a, 30, P)
    assert m.relation is Relation.CONTAINED
    assert m.truncation == "5p"
    assert overhangs(m) == (0, 0, 30, 0)
    assert m.n_edits == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"identity_floor": 0.0},
        {"identity_floor": 1.5},
        {"identity_floor": -0.1},
        {"tol_5p": -1},
        {"tol_3p": -1},
        {"tol_5p": float("inf")},
        {"tol_3p": 30.0},
        {"tol_5p": True},
        {"min_budget": -1},
        {"anchor_len": 0},
        {"short_tol_div": 0},
        {"kmer": 32},
        {"kmer": 0},
        {"identity_floor": True},
        {"identity_floor": "0.99"},
        # Accepted once, and each of them answers instead of failing: with
        # internal_indel_min <= 0 every measured pair is an internal variant.
        {"internal_indel_min": 0},
        {"internal_indel_min": -5},
        {"hint_band": -1},
        {"window": 0},
        {"window": 2.5},
        {"bucket_cap": 1},
        {"min_shared": 0},
        {"diag_band": -1},
        {"max_rows": 0},
    ],
)
def test_parameters_that_cannot_mean_anything_are_refused(kwargs):
    with pytest.raises(ValueError):
        GraphParams(**kwargs)


def test_the_boundary_parameter_values_are_accepted():
    GraphParams(identity_floor=1.0, tol_5p=0, tol_3p=0, anchor_len=1, kmer=31)
    GraphParams(min_budget=0, hint_band=0, diag_band=0, bucket_cap=2, kmer=1)
    GraphParams(internal_indel_min=1, window=1, probes_per_seq=1, min_shared=1)


def test_every_count_is_checked():
    """All of them, by name — a field added later is checked or this fails."""
    import dataclasses

    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import _LEAST

    counts = [f.name for f in dataclasses.fields(P) if f.name != "identity_floor"]
    assert len(counts) == 17
    assert set(_LEAST) <= set(counts)
    for name in counts:
        least = _LEAST.get(name, 1)
        GraphParams(**{name: max(least, getattr(P, name))})
        for bad in (least - 1, float(getattr(P, name)), True, None, "3"):
            with pytest.raises(ValueError, match=name):
                GraphParams(**{name: bad})


def test_the_cache_stamp_leaves_out_how_the_work_was_cut_up():
    """...and nothing else. `max_rows` is a backstop, but when it binds
    buckets are demoted to the anchor fallback and the edges change, so it is
    not a matter of how the work was cut up."""
    stamp = P.semantic()
    assert "chunk_rows" not in stamp
    assert set(stamp) == set(GraphParams.__dataclass_fields__) - {"chunk_rows"}
    assert GraphParams(chunk_rows=1).semantic() == stamp
    assert GraphParams(max_rows=2).semantic() != stamp
    assert GraphParams(tol_5p=29).semantic() != stamp
    assert stamp["identity_floor"] == 0.99 and stamp["bucket_cap"] == 20_480


def test_relation_names_follow_the_enum():
    assert RELATION_NAMES == {
        0: "equivalent",
        1: "contained",
        2: "no_placement",
        3: "divergent_5p",
        4: "divergent_3p",
        5: "staggered",
        6: "internal_variant",
        7: "below_floor",
    }
    assert PRODUCED_RELATIONS == (Relation.EQUIVALENT, Relation.CONTAINED)
    assert TRUNCATION_NAMES == ("5p", "3p", "both")
    assert GRAPH_KERNEL_VERSION == 1


# ──────────────────────────────────────────────────────────────────────
# The tie rule
# ──────────────────────────────────────────────────────────────────────


def test_the_longest_of_tied_ends_wins():
    """edlib reports a last-base mismatch as both ends; the shorter one
    reads the mismatch as a free 1-nt overhang."""
    assert _choose_location([(0, 1998), (0, 1999)], 0, 8) == (0, 1999)


def test_a_placement_near_the_hint_beats_a_longer_one_elsewhere():
    copies = [(300, 419), (306, 426), (420, 539)]
    assert _choose_location(copies, 420, 8) == (420, 539)


def test_tied_spans_near_the_hint_go_to_the_nearest_then_the_smallest_start():
    assert _choose_location([(414, 533), (420, 539), (426, 545)], 420, 8) == (
        420,
        539,
    )
    assert _choose_location([(418, 537), (422, 541)], 420, 8) == (418, 537)


def test_with_nothing_near_the_hint_the_longest_overall_wins():
    assert _choose_location([(100, 219), (200, 320)], 0, 8) == (200, 320)
    assert _choose_location([(200, 319), (100, 219)], 0, 8) == (100, 219)


def test_without_a_hint_the_longest_wins_and_ties_go_to_the_smallest_start():
    assert _choose_location([(306, 425), (300, 419), (312, 432)], None, 8) == (
        312,
        432,
    )
    assert _choose_location([(306, 425), (300, 419)], None, 8) == (300, 419)


def test_degenerate_placements_are_never_chosen():
    assert _choose_location([(None, -1)], 0, 8) is None
    assert _choose_location([(5, 4)], 0, 8) is None
    assert _choose_location([], 0, 8) is None
    assert _choose_location([(None, -1), (3, 9)], 0, 8) == (3, 9)


# ──────────────────────────────────────────────────────────────────────
# Acceptance cases
# ──────────────────────────────────────────────────────────────────────


def test_exact_twins_are_equivalent():
    a = rnd(2000, random.Random(2))
    m = relate_pair(a, bytes(bytearray(a)), 0, P)
    assert m.relation is Relation.EQUIVALENT
    assert m.truncation is None
    assert m.identity == 1.0
    assert overhangs(m) == (0, 0, 0, 0)
    assert (m.aligned_len, m.n_edits, m.edit_distance_placed) == (2000, 0, 0)


def test_a_3p_truncation_is_contained():
    a = rnd(2000, random.Random(3))
    m = relate_pair(a[:1900], a, 0, P)
    assert m.relation is Relation.CONTAINED
    assert m.truncation == "3p"
    assert overhangs(m) == (0, 0, 0, 100)
    assert (m.aligned_len, m.n_edits) == (1900, 0)


def test_a_truncation_at_both_ends_names_both():
    a = rnd(2000, random.Random(4))
    m = relate_pair(a[150:1800], a, 150, P)
    assert (m.relation, m.truncation) == (Relation.CONTAINED, "both")
    assert overhangs(m) == (0, 0, 150, 200)


@pytest.mark.parametrize("diag", [20, None, 0])
def test_a_5p_difference_inside_the_tolerance_is_equivalent(diag):
    """The relation does not depend on the hint being right, or present."""
    a = rnd(2000, random.Random(5))
    m = relate_pair(a[20:], a, diag, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (0, 0, 20, 0)
    assert m.n_edits == 0


@pytest.mark.parametrize("diag", [20, None, 900])
def test_a_stagger_inside_the_tolerance_is_equivalent(diag):
    """With or without a hint, and with one that is simply wrong."""
    rng = random.Random(6)
    core = rnd(2000, rng)
    a = rnd(20, rng) + core
    b = core + rnd(15, rng)
    m = relate_pair(b, a, diag, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (0, 15, 20, 0)
    assert (m.n_edits, m.aligned_len) == (0, 2000)
    # Placing the whole of b inside a charges the 15 nt a does not have.
    assert m.edit_distance_placed == 15


def test_a_stagger_beyond_the_tolerance_is_not_produced():
    rng = random.Random(7)
    core = rnd(2000, rng)
    a = rnd(80, rng) + core
    b = core + rnd(70, rng)
    m = relate_pair(b, a, 80, P)
    assert m.relation not in PRODUCED_RELATIONS
    assert m.relation is Relation.STAGGERED
    assert overhangs(m) == (0, 70, 80, 0)


_FIRST_EXONS = [10, 15, 20, 25, 30, 40, 60, 90, 130]


@pytest.mark.parametrize("body", [2500, 4900, 10000])
def test_an_alternative_first_exon_is_never_an_exact_edge(body):
    """The Akap4 pair: one body behind two unrelated first exons, of equal
    and of unequal length. Short ends may be ``equivalent``, but only with
    their edits on the record — and the test counts how often it found one,
    since a kernel that placed nothing would pass every assertion below."""
    rng = random.Random(body)
    n_placed = n_produced = n_uneven_produced = 0
    for end_a in _FIRST_EXONS:
        for end_b in _FIRST_EXONS:
            for _ in range(2):
                shared = rnd(body, rng)
                a = rnd(end_a, rng) + shared
                b = rnd(end_b, rng) + shared
                src, dst = (a, b) if len(a) <= len(b) else (b, a)
                m = relate_pair(src, dst, len(dst) - len(src), P)
                n_placed += m.relation is not Relation.NO_PLACEMENT
                if m.relation not in PRODUCED_RELATIONS:
                    continue
                n_produced += 1
                n_uneven_produced += end_a != end_b
                assert min(end_a, end_b) < 40, (end_a, end_b)
                assert m.n_edits > 0, (end_a, end_b)
                assert m.n_edits == shared_span_distance(src, dst, m)
                assert not is_exact_edge(m)
    n_trials = 2 * len(_FIRST_EXONS) ** 2
    assert n_placed >= n_trials // 2, n_placed
    assert n_produced >= 20, n_produced
    assert n_uneven_produced >= 10, n_uneven_produced


def is_exact_edge(m) -> bool:
    return m.relation in PRODUCED_RELATIONS and m.n_edits == 0


@pytest.mark.parametrize("length", [40, 60, 100, 140, 150, 299])
def test_one_substitution_never_makes_a_short_pair_divergent(length):
    """Below 150 nt the tolerance (a tenth of the template) is shorter than
    ``anchor_len``. Against a 15-nt anchor one substitution 6-14 nt from an
    end was 7-15 columns of "divergence", past the tolerance: 55% of the
    positions of a 40-nt template wrote no edge."""
    a = rnd(length, random.Random(length))
    container = (
        rnd(200, random.Random(-length)) + a + rnd(200, random.Random(length + 1))
    )
    for at in range(length):
        twin = relate_pair(substituted(a, at), a, 0, P)
        assert twin.relation is Relation.EQUIVALENT, at
        assert twin.n_edits == 1, at
        nested = relate_pair(substituted(a, at), container, 200, P)
        assert nested.relation is Relation.CONTAINED, at
        assert (nested.truncation, nested.n_edits) == ("both", 1), at


def test_the_anchor_is_never_longer_than_the_tolerance():
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        _anchor_runs,
    )

    assert _anchor_runs((30, 30), P) == (15, 15)
    assert _anchor_runs((10, 10), P) == (10, 10)
    assert _anchor_runs((30, 4), P) == (15, 4)
    assert _anchor_runs((0, 0), P) == (1, 1)


def test_an_alternative_end_on_a_short_template_is_still_not_an_edge():
    """The shorter anchor must not let an unrelated end through."""
    rng = random.Random(65)
    seen = []
    for length in (60, 100, 140):
        for end in (8, 12, 20):
            for _ in range(20):
                body = rnd(length, rng)
                a, b = rnd(end, rng) + body, rnd(end, rng) + body
                m = relate_pair(min(a, b), max(a, b), 0, P)
                seen.append(m.relation)
                assert not is_exact_edge(m)
                if m.relation in PRODUCED_RELATIONS:
                    assert 0 < m.n_edits <= edit_budget(len(a), P)
    assert len(seen) == 180


def test_an_alternative_end_is_below_floor_when_short_and_divergent_when_long():
    """A pair over the edit budget is never traced, so it has no ``div``."""
    rng = random.Random(8)
    for body, divergent in ((2500, False), (4900, True)):
        shared = rnd(body, rng)
        a = rnd(60, rng) + shared
        b = rnd(60, rng) + shared
        m = relate_pair(a, b, 0, P)
        if divergent:
            assert m.relation is Relation.DIVERGENT_5P
            assert m.div_5p > 30 and m.n_edits <= edit_budget(len(a), P)
        else:
            assert m.relation not in (*PRODUCED_RELATIONS, Relation.DIVERGENT_5P)
            assert m.n_edits > edit_budget(len(a), P)
            assert (m.div_5p, m.div_3p) == (0, 0)
            assert (m.n_mismatch, m.n_insert, m.n_delete) == (0, 0, 0)


def test_one_internal_substitution_is_one_edit():
    a = rnd(2000, random.Random(9))
    m = relate_pair(substituted(a, 1000), a, 0, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (0, 0, 0, 0)
    assert (m.n_edits, m.n_mismatch, m.n_insert, m.n_delete) == (1, 1, 0, 0)
    assert (m.div_5p, m.div_3p) == (0, 0)
    assert m.identity == pytest.approx(1999 / 2000)


@pytest.mark.parametrize("position", [0, -1, 8, -9])
def test_a_single_terminal_difference_is_charged(position):
    """Not read as an overhang, which would make it free."""
    a = rnd(2000, random.Random(10))
    m = relate_pair(substituted(a, position), a, 0, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (0, 0, 0, 0)
    assert m.n_edits == 1
    assert m.aligned_len == 2000


def test_short_twins_are_equivalent():
    a = rnd(150, random.Random(11))
    m = relate_pair(a, bytes(bytearray(a)), 0, P)
    assert m.relation is Relation.EQUIVALENT
    assert (m.n_edits, m.aligned_len) == (0, 150)


def test_a_short_pair_one_substitution_apart_is_equivalent():
    a = rnd(150, random.Random(12))
    m = relate_pair(substituted(a, 70), a, 0, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (0, 0, 0, 0)
    assert m.n_edits == 1


def test_a_short_pair_offset_by_ten_is_equivalent():
    a = rnd(150, random.Random(13))
    m = relate_pair(a[:140], a[10:], -10, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (10, 0, 0, 10)
    assert (m.n_edits, m.aligned_len) == (0, 130)


@pytest.mark.parametrize("diag", [0, None])
def test_an_antisense_pair_has_no_placement(diag):
    a = rnd(2000, random.Random(14))
    m = relate_pair(a.translate(_COMPLEMENT)[::-1], a, diag, P)
    assert m.relation is Relation.NO_PLACEMENT
    assert m.truncation is None
    assert overhangs(m) == (0, 0, 0, 0)
    assert (m.aligned_len, m.n_edits, m.edit_distance_placed) == (0, 0, 0)
    assert m.identity == 0.0


def test_a_three_nt_skip_is_three_edits():
    a = rnd(3000, random.Random(15))
    m = relate_pair(without(a, 1500, 3), a, 0, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (0, 0, 0, 0)
    assert (m.n_edits, m.core_len_delta) == (3, 3)
    assert (m.n_mismatch, m.n_insert, m.n_delete) == (0, 0, 3)
    assert m.aligned_len == 3000


@pytest.mark.parametrize("skipped, length", [(12, 3000), (27, 3000), (90, 10000)])
def test_a_skipped_segment_is_an_internal_variant(skipped, length):
    """A skipped exon is inside the identity floor and is not the same
    transcript."""
    a = rnd(length, random.Random(16 + skipped))
    m = relate_pair(without(a, length // 2, skipped), a, 0, P)
    assert m.relation is Relation.INTERNAL_VARIANT
    assert overhangs(m) == (0, 0, 0, 0)
    assert (m.n_edits, m.core_len_delta) == (skipped, skipped)
    assert m.n_edits <= edit_budget(length - skipped, P)


def test_a_stagger_with_internal_substitutions_keeps_both():
    rng = random.Random(17)
    core = rnd(1500, rng)
    src = rnd(25, rng) + substituted(core, 15, 700, 1484)
    dst = core + rnd(30, rng)
    m = relate_pair(src, dst, -25, P)
    assert m.relation is Relation.EQUIVALENT
    assert overhangs(m) == (25, 0, 0, 30)
    assert (m.n_edits, m.n_mismatch, m.aligned_len) == (3, 3, 1500)
    assert m.edit_distance_placed == 28


def test_terminal_divergence_counts_toward_extent():
    """A difference within ``anchor_len`` of an end is sequence the two do
    not share there, which is what an overhang is."""
    a = rnd(2000, random.Random(31))
    clean = relate_pair(a[25:], a, 25, P)
    assert (clean.relation, clean.div_5p) == (Relation.EQUIVALENT, 0)

    m = relate_pair(substituted(a[25:], 10), a, 25, P)
    assert (m.dst_overhang_5p, m.div_5p, m.n_edits) == (25, 11, 1)
    assert (m.relation, m.truncation) == (Relation.CONTAINED, "5p")


def test_terminal_divergence_counts_toward_a_stagger():
    """25 nt apart at one end and 30 at the other is inside the tolerance at
    both. A difference near ONE end takes that end past it, and the sequence
    that starts later there is cut short: contained. Near BOTH, each reaches
    past the other: staggered."""
    rng = random.Random(32)
    core = rnd(2000, rng)
    x, y = rnd(25, rng), rnd(30, rng)
    clean = relate_pair(x + core, core + y, -25, P)
    assert clean.relation is Relation.EQUIVALENT

    one = relate_pair(x + substituted(core, 10), core + y, -25, P)
    assert (one.src_overhang_5p, one.div_5p, one.n_edits) == (25, 11, 1)
    assert (one.dst_overhang_3p, one.div_3p) == (30, 0)
    assert (one.relation, one.truncation) == (Relation.CONTAINED, "5p")
    assert one.src_is_container, "it is dst that starts 25 nt later"

    both = relate_pair(x + substituted(core, 10, 1989), core + y, -25, P)
    assert (both.div_5p, both.div_3p, both.n_edits) == (11, 11, 2)
    assert both.relation is Relation.STAGGERED
    assert not both.src_is_container


def test_which_sequence_is_contained_does_not_depend_on_which_is_src():
    """The pair the extent rule used to read two ways. Equal lengths, 28 nt
    apart at both ends, one substitution 5 nt into the shared span: whichever
    is handed over as ``src``, it is ``a`` that is cut short at the 5' end."""
    rng = random.Random(61)
    body = rnd(1256, rng)
    a = substituted(body[28:], 5)  # starts 28 nt later, ends 28 nt later
    b = body[:-28]
    assert len(a) == len(b) == 1228

    ab = relate_pair(a, b, 28, P)
    assert overhangs(ab) == (0, 28, 28, 0)
    assert (ab.div_5p, ab.n_edits) == (6, 1)
    assert (ab.relation, ab.truncation) == (Relation.CONTAINED, "5p")
    assert not ab.src_is_container

    ba = relate_pair(b, a, -28, P)
    assert overhangs(ba) == (28, 0, 0, 28)
    assert (ba.div_5p, ba.n_edits) == (6, 1)
    assert (ba.relation, ba.truncation) == (Relation.CONTAINED, "5p")
    assert ba.src_is_container


def test_a_contained_sequence_can_be_the_longer_by_a_few_nt():
    """``src`` is the shorter because an infix pass needs one, not because
    the relation does. Here the shorter reaches 25 nt past the other at the
    5' end, with a difference 10 nt inside the junction: 36 nt of 5' end the
    longer does not share. The longer reaches 30 nt past at the 3' end,
    which is inside the tolerance. So it is the LONGER that is cut short."""
    rng = random.Random(62)
    core = rnd(2000, rng)
    short = rnd(25, rng) + substituted(core, 10)
    long = core + rnd(30, rng)
    assert len(long) - len(short) == 5
    m = relate_pair(short, long, -25, P)
    assert overhangs(m) == (25, 0, 0, 30)
    assert (m.relation, m.truncation) == (Relation.CONTAINED, "5p")
    assert m.src_is_container


def test_an_alternative_last_exon_is_divergent_at_3p():
    rng = random.Random(33)
    body = rnd(4900, rng)
    m = relate_pair(body + rnd(60, rng), body + rnd(60, rng), 0, P)
    assert m.relation is Relation.DIVERGENT_3P
    assert m.div_3p > 30 and m.div_5p == 0


def test_divergence_at_both_ends_is_counted_at_5p():
    rng = random.Random(34)
    body = rnd(10000, rng)
    a = rnd(40, rng) + body + rnd(40, rng)
    b = rnd(40, rng) + body + rnd(40, rng)
    m = relate_pair(a, b, 0, P)
    assert m.div_5p > 30 and m.div_3p > 30
    assert m.relation is Relation.DIVERGENT_5P


def test_a_skipped_segment_is_an_internal_variant_when_it_was_traced():
    """The length rule reads an alignment. A pair over the budget has none —
    its cores are what two infix placements left — so there the budget
    answers first, and the same skip is named on the template long enough to
    afford it."""
    a = rnd(3000, random.Random(35))
    m = relate_pair(without(a, 1500, 40), a, 0, P)
    assert m.n_edits == 40 > edit_budget(2960, P)
    assert m.relation is Relation.BELOW_FLOOR

    a = rnd(5000, random.Random(35))
    m = relate_pair(without(a, 2500, 40), a, 0, P)
    assert m.n_edits == 40 <= edit_budget(4960, P)
    assert (m.n_delete, m.core_len_delta) == (40, 40)
    assert m.relation is Relation.INTERNAL_VARIANT


def test_uneven_ends_over_the_budget_are_not_called_an_internal_variant():
    """Unrelated first exons on a body too short to trace them. The two
    passes read such ends unevenly and the cores come out different lengths;
    that is not a segment one of them skipped."""
    rng = random.Random(63)
    seen = []
    for length in (600, 1000, 1400):
        for end in (40, 80, 130):
            for _ in range(12):
                body = rnd(length, rng)
                m = relate_pair(rnd(end, rng) + body, rnd(end, rng) + body, 0, P)
                seen.append(m.relation)
    assert Relation.INTERNAL_VARIANT not in seen
    assert not set(seen) & set(PRODUCED_RELATIONS)
    assert seen.count(Relation.BELOW_FLOOR) + seen.count(Relation.NO_PLACEMENT) == len(
        seen
    )


def test_a_pair_under_the_identity_floor_is_below_floor():
    a = rnd(1000, random.Random(36))
    at = range(30, 1000, 65)
    assert len(at) == 15 > edit_budget(1000, P)
    m = relate_pair(substituted(a, *at), a, 0, P)
    assert m.relation is Relation.BELOW_FLOOR
    assert overhangs(m) == (0, 0, 0, 0)
    assert (m.n_edits, m.aligned_len) == (15, 1000)
    assert m.identity == pytest.approx(0.985)
    # Over the budget nothing is traced, so the edits have no split.
    assert (m.n_mismatch, m.n_insert, m.n_delete) == (0, 0, 0)


def test_a_pair_exactly_at_the_budget_is_kept():
    a = rnd(1000, random.Random(37))
    at = range(30, 1000, 100)
    assert len(at) == 10 == edit_budget(1000, P)
    m = relate_pair(substituted(a, *at), a, 0, P)
    assert m.relation is Relation.EQUIVALENT
    assert (m.n_edits, m.n_mismatch) == (10, 10)


def test_an_unresolved_n_costs_one_edit():
    """``N`` is an ordinary character: it matches another N and nothing else."""
    a = rnd(1500, random.Random(38))
    masked = a[:700] + b"N" + a[701:]
    assert relate_pair(masked, a, 0, P).n_edits == 1
    assert relate_pair(masked, bytes(bytearray(masked)), 0, P).n_edits == 0


def test_a_numpy_hint_does_not_ride_into_the_measurement():
    """The builder's diagonals are array elements; an edge's columns are not."""
    a = rnd(500, random.Random(40))
    for seq in (a[20:], substituted(a[20:], 200)):
        m = relate_pair(seq, a, np.int64(20), P)
        assert m.dst_overhang_5p == 20
        assert all(type(v) is int for v in overhangs(m))


def test_src_must_be_the_shorter():
    a = rnd(200, random.Random(18))
    with pytest.raises(ValueError, match="shorter"):
        relate_pair(a, a[:199], 0, P)


# ──────────────────────────────────────────────────────────────────────
# Properties, over seeded trials
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hinted", [True, False])
def test_a_5p_extension_is_an_overhang_not_a_run_of_edits(hinted):
    """The pass-2 counterexample: differences at dst[1] and dst[3] sit hard
    against the junction, so placing src in dst alone cannot tell its 5'
    extension from edits."""
    rng = random.Random(19)
    for _ in range(300):
        x = rng.randint(3, 10)
        core = rnd(1300, rng)
        src = rnd(x, rng) + core
        dst = substituted(core, 1, 3) + rnd(40, rng)
        m = relate_pair(src, dst, -x if hinted else None, P)
        assert m.relation in PRODUCED_RELATIONS
        assert m.src_overhang_5p != 0
        assert abs(m.src_overhang_5p - x) <= 2
        assert m.dst_overhang_5p == 0
        assert m.n_edits == shared_span_distance(src, dst, m)


def test_the_3p_end_does_not_change_what_the_5p_end_measures():
    """Whether dst runs one base past src at 3' says nothing about 5'."""
    rng = random.Random(20)
    for _ in range(100):
        body = rnd(3000, rng)
        src = rnd(12, rng) + body
        flush = rnd(14, rng) + body
        longer = flush + rnd(1, rng)
        m0 = relate_pair(src, flush, 2, P)
        m1 = relate_pair(src, longer, 2, P)
        assert (m0.src_overhang_5p, m0.dst_overhang_5p, m0.n_edits) == (
            m1.src_overhang_5p,
            m1.dst_overhang_5p,
            m1.n_edits,
        )
        assert (m0.div_5p, m0.relation) == (m1.div_5p, m1.relation)
        assert (m0.dst_overhang_3p, m1.dst_overhang_3p) == (0, 1)
        assert m0.n_edits > 0


def _two_passes(src: bytes, dst: bytes, diag: int) -> tuple[int, int, int, int] | None:
    """The overhangs the two infix passes report, BEFORE any fold — the
    kernel's first two steps restated, so the fold can be checked alone.
    ``None`` when the pair is exact or has no placement."""
    tol5, tol3 = effective_tolerance(len(src), P)
    k1 = min(edit_budget(len(src), P) + tol5 + tol3, len(src) - 1)
    if src in dst:
        return None
    one = edlib.align(src, dst, mode="HW", task="locations", k=k1)
    if one["editDistance"] < 0:
        return None
    start, end = _choose_location(one["locations"], max(diag, 0), P.hint_band)
    two = edlib.align(
        dst[start : end + 1], src, mode="HW", task="locations", k=one["editDistance"]
    )
    start2, end2 = _choose_location(two["locations"], max(-diag, 0), P.hint_band)
    return start2, len(src) - 1 - end2, start, len(dst) - 1 - end


def test_no_measurement_overhangs_both_ways_at_one_end():
    """What both sequences overhang at one end is sequence they hold opposite
    each other: it goes back into the alignment and is charged."""
    rng = random.Random(21)
    n_placed = n_folded = n_tied = 0
    for _ in range(500):
        body = rnd(rng.choice([400, 1300, 2600]), rng)
        k = rng.randint(0, 3)
        a = (
            rnd(rng.randint(0, 20), rng)
            + substituted(body, *rng.sample(range(len(body)), k))
            + rnd(rng.randint(0, 20), rng)
        )
        b = rnd(rng.randint(0, 20), rng) + body + rnd(rng.randint(0, 20), rng)
        src, dst = (a, b) if len(a) <= len(b) else (b, a)
        diag = rng.randint(-20, 20)
        m = relate_pair(src, dst, diag, P)
        if m.relation is Relation.NO_PLACEMENT:
            continue
        n_placed += 1
        assert not (m.src_overhang_5p and m.dst_overhang_5p)
        assert not (m.src_overhang_3p and m.dst_overhang_3p)
        assert m.n_edits == shared_span_distance(src, dst, m)
        assert m.core_len_delta == (
            len(dst) - m.dst_overhang_5p - m.dst_overhang_3p
        ) - (len(src) - m.src_overhang_5p - m.src_overhang_3p)

        # A tied pair is measured in the order of the sequences, whichever
        # was passed first, and read back from the caller's side.
        tied = len(src) == len(dst) and src > dst
        n_tied += tied
        raw = _two_passes(dst, src, -diag) if tied else _two_passes(src, dst, diag)
        if raw is None:
            continue
        s5, s3, d5, d3 = raw
        m5, m3 = min(s5, d5), min(s3, d3)
        n_folded += bool(m5 or m3)
        folded = (s5 - m5, s3 - m3, d5 - m5, d3 - m3)
        assert overhangs(m) == (folded[2:] + folded[:2] if tied else folded)
    assert n_placed >= 450
    assert n_folded >= 25  # otherwise this test says nothing about the fold
    assert n_tied >= 3  # nor this about a tie


def test_randomised_staggers_are_recovered_exactly():
    rng = random.Random(22)
    for _ in range(500):
        core = rnd(rng.choice([400, 1300, 2600]), rng)
        # At each end one of the two, chosen at random, runs 0-30 nt further.
        ext = {(who, end): b"" for who in "ab" for end in ("5p", "3p")}
        for end in ("5p", "3p"):
            ext[rng.choice("ab"), end] = rnd(rng.randint(0, 30), rng)
        n_subs = rng.randint(0, 2)
        edited = substituted(core, *rng.sample(range(12, len(core) - 12), n_subs))
        a = ext["a", "5p"] + edited + ext["a", "3p"]
        b = ext["b", "5p"] + core + ext["b", "3p"]
        if len(a) <= len(b):
            src, dst, s, d = a, b, "a", "b"
        else:
            src, dst, s, d = b, a, "b", "a"
        want = (
            len(ext[s, "5p"]),
            len(ext[s, "3p"]),
            len(ext[d, "5p"]),
            len(ext[d, "3p"]),
        )
        m = relate_pair(src, dst, want[2] - want[0], P)
        assert overhangs(m) == want
        assert m.n_edits <= n_subs
        assert m.n_edits == n_subs  # two substitutions cannot cost fewer than two
        assert m.aligned_len == len(core)
        # Every overhang is inside the tolerance, so only a substitution
        # within anchor_len of an end (div > 0) can take a pair past it.
        if m.div_5p == m.div_3p == 0:
            assert m.relation is Relation.EQUIVALENT
        else:
            assert m.relation in (*PRODUCED_RELATIONS, Relation.STAGGERED)


def test_a_tandem_repeat_is_placed_at_the_hinted_copy():
    """One placement per copy comes back, all at the same distance. The last
    base becomes G, not C: the repeat unit starts with C, so a C there is
    also one edit from the PREVIOUS copy read one base longer."""
    rng = random.Random(23)
    repeat = b"CAGGCT" * 40
    dst = rnd(300, rng) + repeat + rnd(200, rng)
    src = repeat[-120:-1] + b"G"
    true_start = 300 + len(repeat) - 120

    m = relate_pair(src, dst, true_start, P)
    assert m.dst_overhang_5p == true_start
    assert m.dst_overhang_3p == 200
    assert (m.n_edits, m.src_overhang_5p, m.src_overhang_3p) == (1, 0, 0)

    # The hint is what decides: without it the first copy is as good as any.
    assert relate_pair(src, dst, None, P).dst_overhang_5p == 300


def test_a_longer_placement_one_copy_away_does_not_win():
    """Longest-first alone takes the wrong copy. Here the unit is longer than
    the hint band and the last base becomes the unit's first, so the previous
    copy, read one base further, is the longest placement there is."""
    rng = random.Random(29)
    unit = b"CAGGCTTACGGA"
    repeat = unit * 20
    dst = rnd(300, rng) + repeat + rnd(200, rng)
    src = repeat[-120:-1] + unit[:1]
    true_start = 300 + len(repeat) - 120

    found = edlib.align(src, dst, mode="HW", task="locations", k=27)["locations"]
    longest = max(end - start for start, end in found)
    assert (true_start - len(unit), true_start - len(unit) + longest) in found
    assert (true_start, true_start + longest) not in found

    m = relate_pair(src, dst, true_start, P)
    assert m.dst_overhang_5p == true_start
    assert (m.n_edits, m.src_overhang_5p, m.src_overhang_3p) == (1, 0, 0)


def test_an_exact_repeat_copy_is_placed_at_the_occurrence_nearest_the_hint():
    rng = random.Random(24)
    repeat = b"CAGGCT" * 40
    dst = rnd(300, rng) + repeat + rnd(200, rng)
    src = repeat[:60]
    assert relate_pair(src, dst, 420, P).dst_overhang_5p == 420
    assert relate_pair(src, dst, 422, P).dst_overhang_5p == 420
    assert relate_pair(src, dst, 424, P).dst_overhang_5p == 426
    assert relate_pair(src, dst, 5000, P).dst_overhang_5p == 480
    assert relate_pair(src, dst, -7, P).dst_overhang_5p == 300
    assert relate_pair(src, dst, None, P).dst_overhang_5p == 300


def test_swapping_equal_length_sequences_mirrors_the_measurement():
    """Length ties are the caller's to break, so neither order may matter."""
    rng = random.Random(25)
    core = rnd(1500, rng)
    a = rnd(11, rng) + without(substituted(core, 400), 900, 1)
    b = core + rnd(10, rng)
    assert len(a) == len(b)

    ab = relate_pair(a, b, -11, P)
    ba = relate_pair(b, a, 11, P)
    assert ab.relation is ba.relation is Relation.EQUIVALENT
    assert overhangs(ab) == (11, 0, 0, 10)
    assert overhangs(ba) == (0, 10, 11, 0)
    assert ab.n_edits == ba.n_edits == 2
    assert ab.aligned_len == ba.aligned_len == 1500
    assert ab.identity == ba.identity
    assert (ab.div_5p, ab.div_3p) == (ba.div_5p, ba.div_3p) == (0, 0)
    # a lacks one base of b: a deletion seen from a, an insertion from b.
    assert (ab.n_mismatch, ab.n_insert, ab.n_delete) == (1, 0, 1)
    assert (ba.n_mismatch, ba.n_insert, ba.n_delete) == (1, 1, 0)
    assert ab.core_len_delta == -ba.core_len_delta == 1


def test_a_tied_pair_reads_the_same_from_either_side():
    """Equal lengths, 0-44 nt apart, with differences planted within 12 nt
    of the ends of the shared span — where placing ``a`` in ``b`` and ``b``
    in ``a`` can disagree by an edit, and where ``div`` takes an end past
    the tolerance. Every field mirrors, for every relation."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        _mirrored,
    )

    rng = random.Random(64)
    seen: dict[Relation, int] = {}
    n_turned = 0
    for _ in range(1500):
        shift = rng.randint(0, 44)
        body = rnd(rng.choice([300, 700, 1300]) + shift, rng)
        shared = len(body) - shift
        near = [rng.randrange(0, 12) for _ in range(rng.randint(0, 2))]
        near += [shared - 1 - rng.randrange(0, 12) for _ in range(rng.randint(0, 2))]
        a = substituted(body[shift:], *set(near)) + rnd(shift, rng)
        b = body
        assert len(a) == len(b)
        ab = relate_pair(a, b, shift, P)
        ba = relate_pair(b, a, -shift, P)
        assert ab == _mirrored(ba)
        assert ba == _mirrored(ab)
        seen[ab.relation] = seen.get(ab.relation, 0) + 1
        n_turned += ab.src_is_container or ba.src_is_container
        if ab.relation is Relation.CONTAINED:
            assert ab.src_is_container != ba.src_is_container
            assert ab.truncation == ba.truncation
    assert seen[Relation.EQUIVALENT] >= 300
    assert seen[Relation.CONTAINED] >= 50
    assert seen[Relation.STAGGERED] >= 100
    assert n_turned >= 50


def test_the_kernel_hands_edlib_bytes(monkeypatch):
    """A numpy slice is accepted by edlib and converted per element, at
    several times the cost of the alignment itself."""
    real = edlib.align
    modes = []

    def spy(query, target, **kwargs):
        assert isinstance(query, bytes), type(query)
        assert isinstance(target, bytes), type(target)
        modes.append(kwargs["mode"])
        return real(query, target, **kwargs)

    monkeypatch.setattr(edlib, "align", spy)
    rng = random.Random(26)
    core = rnd(1400, rng)
    src = rnd(10, rng) + without(core, 700, 1)
    m = relate_pair(src, core + rnd(20, rng), -10, P)
    assert modes == ["HW", "HW", "NW"]
    assert (m.relation, m.n_edits) == (Relation.EQUIVALENT, 1)


def test_the_first_pass_never_lets_every_base_be_an_edit(monkeypatch):
    """At k >= len(query) edlib may answer with an all-insertion placement,
    which has no start. Only reachable with a tolerance this loose."""
    real = edlib.align
    asked = []

    def spy(query, target, **kwargs):
        asked.append((len(query), kwargs["k"]))
        return real(query, target, **kwargs)

    monkeypatch.setattr(edlib, "align", spy)
    rng = random.Random(39)
    loose = GraphParams(short_tol_div=1)
    assert effective_tolerance(40, loose) == (30, 30)
    relate_pair(rnd(40, rng), rnd(500, rng), None, loose)
    assert asked[0] == (40, 39)


def test_an_exact_pair_costs_no_alignment(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("edlib called for a pair that is a substring")

    monkeypatch.setattr(edlib, "align", refuse)
    a = rnd(1400, random.Random(27))
    assert relate_pair(a[100:1300], a, 100, P).relation is Relation.CONTAINED
    assert relate_pair(a[100:1300], a, 0, P).dst_overhang_5p == 100
    assert relate_pair(a[100:1300], a, None, P).dst_overhang_5p == 100


# ──────────────────────────────────────────────────────────────────────
# Edge tables
# ──────────────────────────────────────────────────────────────────────

_DICT = pa.dictionary(pa.int8(), pa.string())

_SCHEMAS = [
    ("EmTemplateEdgeTable", TEMPLATE_EDGE_TABLE),
    ("EmClusterEdgeTable", CLUSTER_EDGE_TABLE),
]


def test_the_feature_columns_are_shared_and_in_order():
    names = [f.name for f in EDGE_FEATURE_FIELDS]
    assert names == [
        "relation",
        "truncation",
        "src_len",
        "dst_len",
        "src_overhang_5p",
        "src_overhang_3p",
        "dst_overhang_5p",
        "dst_overhang_3p",
        "delta_5p",
        "delta_3p",
        "aligned_len",
        "n_edits",
        "core_len_delta",
        "n_mismatch",
        "n_insert",
        "n_delete",
        "identity",
        "edit_distance_placed",
        "div_5p",
        "div_3p",
        "n_shared_probes",
        "candidate_overflow",
        "same_split_origin",
        "src_n_reads",
        "dst_n_reads",
        "mergeable",
    ]
    assert [f.name for f in EDGE_FEATURE_FIELDS if f.nullable] == ["truncation"]
    assert "n_indel_runs" not in names
    assert TEMPLATE_EDGE_TABLE.names == [
        "node_round",
        "src_template_id",
        "dst_template_id",
        "src_row",
        "dst_row",
        *names,
    ]
    assert CLUSTER_EDGE_TABLE.names == ["src_cluster_id", "dst_cluster_id", *names]
    for _, schema in _SCHEMAS:
        assert [schema.field(n) for n in names] == EDGE_FEATURE_FIELDS


@pytest.mark.parametrize("name, schema", _SCHEMAS)
def test_an_edge_schema_is_registered(name, schema):
    assert get_schema(name) is schema
    assert schema.metadata == {b"schema_name": name.encode()}
    assert all(not schema.field(i).nullable for i in range(len(schema) - 26))


@pytest.mark.parametrize("name, schema", _SCHEMAS)
def test_an_empty_edge_table_round_trips_through_parquet(name, schema, tmp_path):
    path = tmp_path / "edges.parquet"
    pq.write_table(schema.empty_table(), path)
    back = pq.read_table(path)
    assert back.num_rows == 0
    assert back.schema.equals(schema, check_metadata=True)
    assert back.schema.field("relation").type == _DICT
    assert back.schema.field("truncation").type == _DICT


@pytest.mark.parametrize("name, schema", _SCHEMAS)
def test_a_measured_edge_round_trips_through_parquet(name, schema, tmp_path):
    a = rnd(2000, random.Random(28))
    src, dst = substituted(a[:1900], 900), a
    m = relate_pair(src, dst, 0, P)
    assert m.relation is Relation.CONTAINED

    features = {
        "relation": RELATION_NAMES[m.relation],
        "truncation": m.truncation,
        "src_len": len(src),
        "dst_len": len(dst),
        "src_overhang_5p": m.src_overhang_5p,
        "src_overhang_3p": m.src_overhang_3p,
        "dst_overhang_5p": m.dst_overhang_5p,
        "dst_overhang_3p": m.dst_overhang_3p,
        "delta_5p": m.dst_overhang_5p - m.src_overhang_5p,
        "delta_3p": m.dst_overhang_3p - m.src_overhang_3p,
        "aligned_len": m.aligned_len,
        "n_edits": m.n_edits,
        "core_len_delta": m.core_len_delta,
        "n_mismatch": m.n_mismatch,
        "n_insert": m.n_insert,
        "n_delete": m.n_delete,
        "identity": m.identity,
        "edit_distance_placed": m.edit_distance_placed,
        "div_5p": m.div_5p,
        "div_3p": m.div_3p,
        "n_shared_probes": 9,
        "candidate_overflow": False,
        "same_split_origin": True,
        "src_n_reads": 40,
        "dst_n_reads": 3,
        "mergeable": False,
    }
    row = {n: 0 for n in schema.names if n not in features} | features
    twin = row | {"relation": "equivalent", "truncation": None}
    table = pa.Table.from_pylist([row, twin], schema=schema)

    path = tmp_path / "edges.parquet"
    pq.write_table(table, path)
    back = pq.read_table(path)
    assert back.schema.equals(schema, check_metadata=True)
    assert back.to_pylist() == [row, twin]
    assert back["relation"].to_pylist() == ["contained", "equivalent"]
    assert back["truncation"].to_pylist() == ["3p", None]
    assert row["delta_3p"] == 100 and row["n_edits"] == 1
