"""Shared statistical primitives in `denovo/variants.py`.

These are factored out of two byte-identical inline copies, so the first
requirement is that the extraction changed nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.variants import (
    ErrorModel,
    benjamini_hochberg,
    estimate_error_rates,
    homopolymer_runs,
    survival,
)


def _inline_bh(pvals: np.ndarray, q: float) -> float:
    """The pre-refactor inline block, verbatim, as the oracle."""
    m = pvals.shape[0]
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked <= (np.arange(1, m + 1) / m) * q
    return ranked[bh].max() if bh.any() else -1.0


# ── Benjamini-Hochberg ────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(25))
def test_bh_reproduces_the_inline_cutoff_exactly(seed):
    """Not 'equivalently' — bit-identical. The cutoff feeds a `<=` comparison
    that decides every variant call, so a float that differs in the last place
    is a behaviour change."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 200))
    # Mix uniform noise with a few genuinely small p-values.
    pvals = rng.random(n)
    pvals[: max(1, n // 10)] *= 1e-6
    for q in (0.001, 0.01, 0.05, 0.2):
        _sig, cut = benjamini_hochberg(pvals, q)
        assert cut == _inline_bh(pvals, q)


def test_bh_mask_matches_its_own_cutoff():
    pvals = np.array([1e-9, 1e-4, 0.02, 0.4, 0.9])
    sig, cut = benjamini_hochberg(pvals, 0.05)
    assert np.array_equal(sig, pvals <= cut)


def test_bh_empty_and_nothing_significant():
    sig, cut = benjamini_hochberg(np.empty(0), 0.05)
    assert sig.shape == (0,) and cut == -1.0
    sig, cut = benjamini_hochberg(np.array([0.9, 0.95, 0.99]), 0.01)
    assert not sig.any() and cut == -1.0


def test_bh_all_zero_p_values_are_all_significant():
    sig, cut = benjamini_hochberg(np.zeros(10), 0.01)
    assert sig.all() and cut == 0.0


def test_bh_is_less_conservative_than_bonferroni():
    """Sanity on the direction: BH rejects at least as much as Bonferroni."""
    rng = np.random.default_rng(7)
    pvals = rng.random(100) * 0.02
    sig, _cut = benjamini_hochberg(pvals, 0.05)
    assert sig.sum() >= (pvals <= 0.05 / pvals.size).sum()


# ── the survival function and its depth penalty ───────────────────────


def test_survival_binomial_and_betabinomial_agree_at_rho_zero():
    a = np.array([5, 10, 2])
    n = np.array([100, 1000, 20])
    eps = np.array([0.01, 0.01, 0.01])
    assert np.allclose(survival(a, n, eps), survival(a, n, eps, 0.0))


def test_overdispersion_is_a_depth_penalty():
    """A point binomial over-calls small minors once depth is large; rho caps
    the effective depth. The effect is in the TAIL — at the null's own mean
    the extra variance spreads mass both ways and the two agree.

    n=1500, eps=0.01: mean 15, binomial sd ~3.9. rho=0.01 inflates the
    variance by 1+(n-1)rho = 16, so sd ~15.4 and a count of 30 goes from
    ~4 sd out to ~1 sd out.
    """
    a, n, eps = np.array([30]), np.array([1500]), np.array([0.01])
    binom_p = survival(a, n, eps, 0.0)[0]
    betabinom_p = survival(a, n, eps, 0.01)[0]
    assert binom_p < 1e-3, "a point binomial calls this highly significant"
    assert betabinom_p > 0.05, "rho makes it unremarkable"


def test_a_real_minority_survives_the_depth_penalty():
    """rho must not be so strong that a genuine 20% minor at depth 1,500 stops
    being significant — that would defeat the point of the M-step."""
    a, n, eps = np.array([300]), np.array([1500]), np.array([0.01])
    assert survival(a, n, eps, 0.01)[0] < 1e-6


# ── homopolymer runs ──────────────────────────────────────────────────


def test_homopolymer_runs_labels_each_position_with_its_run_length():
    codes = np.array([0, 0, 0, 1, 2, 2], dtype=np.int8)
    assert homopolymer_runs(codes).tolist() == [3, 3, 3, 1, 2, 2]
    assert homopolymer_runs(np.empty(0, dtype=np.int8)).tolist() == []


# ── the error model cannot return zero, and can reach 22% ─────────────


def test_a_class_with_no_observed_minor_does_not_get_eps_zero():
    """A live defect on the components path under --error-model empirical.

    A plain m/t returns exactly 0 for a well-exposed class that happened to
    show no minor allele, and binom.sf(a-1, n, 0.0) is 0 for every a >= 1 —
    so every column in that class becomes maximally significant and the FDR
    gate stops gating entirely. Not a precision issue; a failure mode.
    """
    stats = {(0, 0): (0.0, 100_000.0), (2, 0): (50.0, 10_000.0)}
    fitted = estimate_error_rates(stats)
    assert fitted.eps_sub > 0.0

    a, n = np.array([1]), np.array([500])
    assert survival(a, n, np.array([0.0]))[0] == 0.0, "the bug, for contrast"
    assert survival(a, n, np.array([fitted.eps_sub]))[0] > 1e-4


def test_error_model_ceiling_is_above_the_measured_homopolymer_rate():
    """hp_max sat at 0.15 while the H3f3b 5-G run measures 22%, so the clamp
    was capping the empirical fit rather than bounding the prior."""
    assert ErrorModel().hp_max > 0.22


def test_a_22_percent_homopolymer_rate_is_representable():
    high = ErrorModel(eps_hp0=0.22, hp_ref_len=5, hp_slope=0.0)
    assert high.epsilon_homopolymer(5) == pytest.approx(0.22)


def test_jeffreys_leaves_the_shipped_priors_alone():
    """The estimator change must not move a well-exposed fit: 300/100000
    stays ~0.003, not drift to something else."""
    stats = {(0, 0): (300.0, 100_000.0), (2, 0): (50.0, 10_000.0)}
    fitted = estimate_error_rates(stats)
    assert fitted.eps_sub == pytest.approx(0.003, abs=1e-4)
    assert fitted.eps_indel == pytest.approx(0.005, abs=1e-4)
