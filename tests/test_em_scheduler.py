"""The E-step assignment rule: the pool, and the two ranking regimes.

Written against plain arrays, with no minimap2 on ``$PATH`` and no PAF — the
operating point of this rule is the thing most worth being able to prove from
hand-built inputs.

Two cases carry the design:

* ``test_round1_absorbs_a_singleton_into_a_better_replicated_template`` — the
  reason round 1 does not rank on score. Under argmax-AS a read wins its own
  seed by ~72 points at 1.2 kb, so self-capture is total and permanent; under
  ORF-replication ranking the same read joins the group it belongs to.
* ``test_abundance_escape_needs_both_a_near_tie_and_a_support_ratio`` — the
  reason a real low-abundance variant is not absorbed by a popular neighbour.
"""

from __future__ import annotations

import numpy as np
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.scheduler import (
    UNASSIGNED,
    admit_candidates,
    rank_likelihood,
    rank_round1,
    shortlist_for_likelihood,
)


def _ptr(*sizes: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


# ── the pool ──────────────────────────────────────────────────────────


def test_identity_gate_admits_at_the_floor_and_rejects_below():
    n_match = np.array([970, 969, 1000])
    aln_len = np.array([1000, 1000, 1000])
    admitted, n_admitted = admit_candidates(n_match, aln_len, _ptr(3), p_floor=0.97)
    assert admitted.tolist() == [True, False, True]
    assert n_admitted.tolist() == [2]


def test_zero_length_alignment_is_not_evidence():
    admitted, _ = admit_candidates(np.array([0]), np.array([0]), _ptr(1))
    assert admitted.tolist() == [False]


def test_a_read_with_no_admitted_hit_is_unassigned():
    """The floor's whole point: a bad best match is not a match."""
    admitted, n_admitted = admit_candidates(
        np.array([900, 910]), np.array([1000, 1000]), _ptr(2), p_floor=0.97
    )
    assert n_admitted.tolist() == [0]
    winner = rank_round1(
        admitted,
        np.array([0, 1]),
        _ptr(2),
        orf_replication=np.array([50, 80]),
        seed_read_quality=np.array([30.0, 30.0]),
    )
    assert winner.tolist() == [UNASSIGNED]


# ── round 1 ───────────────────────────────────────────────────────────


def test_round1_ranks_on_orf_replication():
    admitted = np.array([True, True, True])
    winner = rank_round1(
        admitted,
        np.array([0, 1, 2]),
        _ptr(3),
        orf_replication=np.array([1, 200, 30]),
        seed_read_quality=np.array([40.0, 10.0, 40.0]),
    )
    assert winner.tolist() == [1], "replication outranks quality"


def test_round1_breaks_replication_ties_on_seed_quality():
    admitted = np.array([True, True])
    winner = rank_round1(
        admitted,
        np.array([0, 1]),
        _ptr(2),
        orf_replication=np.array([25, 25]),
        seed_read_quality=np.array([18.0, 31.0]),
    )
    assert winner.tolist() == [1]


def test_round1_absorbs_a_singleton_into_a_better_replicated_template():
    """Self-capture, and why round 1 cannot rank on score.

    Hit 0 is the read's own seed template: a perfect self-match (replication
    1, since nothing else carried that ORF). Hit 1 is its gene's real
    template at 98% identity, carrying 500 reads. Any score-based rule returns
    hit 0 and the read never joins anything. Replication ranking returns
    hit 1, which is the whole convergence mechanism of round 1.
    """
    n_match = np.array([1200, 1176])
    aln_len = np.array([1200, 1200])
    admitted, _ = admit_candidates(n_match, aln_len, _ptr(2), p_floor=0.97)
    assert admitted.tolist() == [True, True]

    winner = rank_round1(
        admitted,
        np.array([0, 1]),
        _ptr(2),
        orf_replication=np.array([1, 500]),
        seed_read_quality=np.array([35.0, 35.0]),
    )
    assert winner.tolist() == [1]


def test_round1_keeps_a_genuinely_divergent_read_on_itself():
    """A true singleton >3% from everything has only itself admitted."""
    n_match = np.array([1200, 1100])  # 1.000 and 0.917
    aln_len = np.array([1200, 1200])
    admitted, n_admitted = admit_candidates(n_match, aln_len, _ptr(2), p_floor=0.97)
    assert n_admitted.tolist() == [1]
    winner = rank_round1(
        admitted,
        np.array([0, 1]),
        _ptr(2),
        orf_replication=np.array([1, 900]),
        seed_read_quality=np.array([35.0, 35.0]),
    )
    assert winner.tolist() == [0], "nothing within the floor; it stays on itself"


# ── round 2+ ──────────────────────────────────────────────────────────


def test_argmax_loglik_wins_by_default():
    admitted = np.array([True, True, True])
    winner, delta = rank_likelihood(
        admitted,
        np.array([0, 1, 2]),
        _ptr(3),
        logl=np.array([-30.0, -12.0, -50.0]),
        support=np.array([10.0, 10.0, 10.0]),
        read_len=np.full(3, 1000.0),
        template_len=np.array([1000.0, 1000.0, 1000.0]),
    )
    assert winner.tolist() == [1]
    assert delta.tolist() == pytest.approx([18.0, 0.0, 38.0])


def test_abundance_escape_needs_both_a_near_tie_and_a_support_ratio():
    """Neither condition alone is enough — that is what protects a variant."""
    idx = np.array([0, 1])
    common = dict(
        admitted=np.array([True, True]),
        template_idx=idx,
        group_ptr=_ptr(2),
        read_len=np.full(2, 1000.0),
        template_len=np.array([1000.0, 1000.0]),
        delta_logl=5.0,
        support_ratio=20.0,
    )

    # Near-tie AND 50x support -> the abundant one takes it.
    winner, _ = rank_likelihood(
        logl=np.array([-10.0, -13.0]), support=np.array([10.0, 500.0]), **common
    )
    assert winner.tolist() == [1]

    # Near-tie but only 5x support -> argmax holds.
    winner, _ = rank_likelihood(
        logl=np.array([-10.0, -13.0]), support=np.array([10.0, 50.0]), **common
    )
    assert winner.tolist() == [0]

    # 50x support but OUTSIDE the window -> argmax holds. A real variant with
    # a couple of clean discriminating positions cannot be absorbed.
    winner, _ = rank_likelihood(
        logl=np.array([-10.0, -22.0]), support=np.array([10.0, 500.0]), **common
    )
    assert winner.tolist() == [0]


def test_a_low_abundance_variant_survives_a_dominant_neighbour():
    """Two discriminating substitutions (~11.6 nats) beat any support ratio."""
    winner, _ = rank_likelihood(
        np.array([True, True]),
        np.array([0, 1]),
        _ptr(2),
        logl=np.array([-8.0, -19.6]),
        support=np.array([3.0, 100_000.0]),
        read_len=np.full(2, 1200.0),
        template_len=np.array([1200.0, 1200.0]),
    )
    assert winner.tolist() == [0]


def test_length_tie_break_prefers_the_length_matched_template():
    """AS is read-anchored, so a contained read ties; template coverage breaks it."""
    winner, _ = rank_likelihood(
        np.array([True, True]),
        np.array([0, 1]),
        _ptr(2),
        logl=np.array([-10.0, -10.0]),
        support=np.array([5.0, 5.0]),
        read_len=np.full(2, 1200.0),
        template_len=np.array([5700.0, 1250.0]),  # the 5.7 kb container vs a match
    )
    assert winner.tolist() == [1]


def test_unadmitted_hits_never_win_however_good_their_likelihood():
    winner, _ = rank_likelihood(
        np.array([False, True]),
        np.array([0, 1]),
        _ptr(2),
        logl=np.array([-1.0, -900.0]),
        support=np.array([1000.0, 1.0]),
        read_len=np.full(2, 1000.0),
        template_len=np.array([1000.0, 1000.0]),
    )
    assert winner.tolist() == [1]


def test_multiple_groups_are_independent():
    admitted = np.array([True, True, True, True, True])
    winner, _ = rank_likelihood(
        admitted,
        np.array([0, 1, 2, 3, 4]),
        _ptr(2, 3),
        logl=np.array([-5.0, -9.0, -40.0, -3.0, -60.0]),
        support=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        read_len=np.full(5, 1000.0),
        template_len=np.full(5, 1000.0),
    )
    assert winner.tolist() == [0, 3]


# ── the AS shortlist ──────────────────────────────────────────────────


def test_shortlist_window_scales_as_sqrt_of_span():
    """z=2 reproduces the hand-tuned band_abs=40 at the median 1.2 kb read."""
    for span, expected in ((300, 21), (1200, 41), (2500, 60)):
        width = 2.0 * 6.0 * np.sqrt(span * 0.01 * 0.99)
        assert width == pytest.approx(expected, abs=1.0)


def test_shortlist_keeps_near_ties_and_drops_clear_losers():
    admitted = np.array([True, True, True])
    keep = shortlist_for_likelihood(
        admitted,
        np.array([2400, 2380, 2000]),
        _ptr(3),
        span=np.full(3, 1200.0),
    )
    # window at 1.2 kb is ~41: 20 behind is in, 400 behind is out.
    assert keep.tolist() == [True, True, False]


def test_shortlist_never_includes_an_unadmitted_hit():
    keep = shortlist_for_likelihood(
        np.array([True, False]),
        np.array([2400, 2400]),
        _ptr(2),
        span=np.full(2, 1200.0),
    )
    assert keep.tolist() == [True, False]
