"""Tests for core.stats.losses."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.stats import losses


# ──────────────────────────────────────────────────────────────────────
# Normalization helpers
# ──────────────────────────────────────────────────────────────────────


def test_l1_normalize_sums_to_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = losses.l1_normalize(x)
    assert torch.allclose(out.sum(), torch.tensor(1.0))


def test_l1_normalize_batched():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = losses.l1_normalize(x)
    assert torch.allclose(out.sum(dim=-1), torch.tensor([1.0, 1.0]))


def test_l1_normalize_with_pseudocount():
    x = torch.zeros(4)
    out = losses.l1_normalize(x, pseudocount=1.0)
    # Uniform after smoothing all-zeros
    assert torch.allclose(out, torch.full((4,), 0.25))


def test_l2_normalize_unit_norm():
    x = torch.tensor([3.0, 4.0])
    out = losses.l2_normalize(x)
    assert torch.allclose(torch.linalg.norm(out), torch.tensor(1.0))


def test_l2_normalize_batched():
    x = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    out = losses.l2_normalize(x, batch_dim=True)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.tensor([1.0, 1.0]))


# ──────────────────────────────────────────────────────────────────────
# kld
# ──────────────────────────────────────────────────────────────────────


def test_kld_identical_inputs_zero():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = losses.kld(p, p, pseudocount=0.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-10)


def test_kld_against_scipy_rel_entr():
    """KL(target || pred) after l1-normalization — cross-check vs scipy."""
    from scipy.special import rel_entr

    target = torch.tensor([0.4, 0.3, 0.2, 0.1])
    pred = torch.tensor([0.25, 0.25, 0.25, 0.25])
    # Already normalized; pseudocount=0 lets us compare cleanly
    out = losses.kld(pred, target, pseudocount=0.0, reduce="sum").item()
    expected = float(rel_entr(target.numpy(), pred.numpy()).sum())
    assert math.isclose(out, expected, rel_tol=1e-6)


def test_kld_mask_excludes_negative_sentinel():
    """Positions where target == -1 are masked out and don't contribute."""
    pred = torch.tensor([0.5, 0.5, 0.0])
    target_with_sentinel = torch.tensor([0.5, 0.5, -1.0])
    target_clean = torch.tensor([0.5, 0.5])
    pred_clean = torch.tensor([0.5, 0.5])

    out_masked = losses.kld(pred, target_with_sentinel, pseudocount=0.0).item()
    out_clean = losses.kld(pred_clean, target_clean, pseudocount=0.0).item()
    assert math.isclose(out_masked, out_clean, abs_tol=1e-9)


def test_kld_reduce_modes():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    none = losses.kld(pred, target, reduce="none")
    assert none.shape == (2,)
    assert torch.allclose(losses.kld(pred, target, reduce="mean"), none.mean())
    assert torch.allclose(losses.kld(pred, target, reduce="sum"), none.sum())


def test_kld_invalid_reduce():
    with pytest.raises(ValueError, match="reduce"):
        losses.kld(torch.tensor([1.0]), torch.tensor([1.0]), reduce="bogus")


# ──────────────────────────────────────────────────────────────────────
# spectral_angle
# ──────────────────────────────────────────────────────────────────────


def test_spectral_angle_identical_returns_one():
    # float64: arccos near 1 amplifies float32 roundoff into ~3e-4 error
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    out = losses.spectral_angle(x, x).item()
    assert math.isclose(out, 1.0, abs_tol=1e-5)


def test_spectral_angle_orthogonal_returns_zero():
    p = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    q = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    out = losses.spectral_angle(p, q).item()
    assert math.isclose(out, 0.0, abs_tol=1e-5)


def test_spectral_angle_mask_sentinel():
    """target == -1 positions excluded — adding a -1 to both shouldn't
    move the score relative to the un-masked version."""
    p = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    t = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float64)
    p_pad = torch.tensor([1.0, 2.0, 3.0, 999.0], dtype=torch.float64)
    t_pad = torch.tensor([2.0, 1.0, 4.0, -1.0], dtype=torch.float64)
    a = losses.spectral_angle(p, t).item()
    b = losses.spectral_angle(p_pad, t_pad).item()
    assert math.isclose(a, b, abs_tol=1e-5)


def test_spectral_angle_batched():
    p = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    t = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    out = losses.spectral_angle(p, t, batch_dim=True)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# spectral_entropy_loss
# ──────────────────────────────────────────────────────────────────────


def test_spectral_entropy_loss_runs():
    # Just exercise the formula end-to-end; numerical equivalence to KLD
    # at signal_logit→+∞ involves sigmoid saturation that isn't exact.
    pred = torch.tensor([0.4, 0.3, 0.2, 0.1])
    target = torch.tensor([0.25, 0.25, 0.25, 0.25])
    signal_logit = torch.tensor(0.0)
    out = losses.spectral_entropy_loss(pred, target, signal_logit)
    assert torch.is_tensor(out)
    assert torch.isfinite(out)


def test_spectral_entropy_loss_high_signal_approaches_kld():
    """At signal_logit large, signal_frac → 1; the loss should track
    kld(pred, target) to within sigmoid saturation tolerance."""
    pred = torch.tensor([0.4, 0.3, 0.2, 0.1])
    target = torch.tensor([0.25, 0.25, 0.25, 0.25])
    high = torch.tensor(20.0)  # sigmoid ≈ 1 - 2e-9
    out = losses.spectral_entropy_loss(pred, target, high).item()
    kl = losses.kld(pred, target, pseudocount=0.0, reduce="sum").item()
    # Loose tolerance — exact equivalence requires sigmoid → 1 exactly.
    assert math.isclose(out, kl, abs_tol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# cosine_similarity / normalized_dot
# ──────────────────────────────────────────────────────────────────────


def test_cosine_identical_returns_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert math.isclose(losses.cosine_similarity(x, x).item(), 1.0, abs_tol=1e-9)


def test_cosine_orthogonal_returns_zero():
    p = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    q = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    assert math.isclose(losses.cosine_similarity(p, q).item(), 0.0, abs_tol=1e-9)


def test_cosine_matches_spectral_angle_relation():
    """spectral_angle = 1 - 2·arccos(cosine)/π, exactly, on a 1-D pair."""
    p = torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=torch.float64)
    t = torch.tensor([2.0, 1.0, 4.0, 1.5], dtype=torch.float64)
    cos = losses.cosine_similarity(p, t).item()
    sa = losses.spectral_angle(p, t).item()
    assert math.isclose(sa, 1.0 - 2.0 * math.acos(cos) / math.pi, abs_tol=1e-9)


def test_normalized_dot_is_cosine_alias():
    p = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    t = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    assert torch.allclose(
        losses.normalized_dot(p, t), losses.cosine_similarity(p, t)
    )


def test_cosine_batched_shape_and_values():
    p = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    t = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    out = losses.cosine_similarity(p, t)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.ones(2, dtype=torch.float64), atol=1e-9)


def test_cosine_mask_sentinel():
    """target == -1 positions excluded — padding both with a (999, -1)
    channel must not move the score."""
    p = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    t = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float64)
    p_pad = torch.tensor([1.0, 2.0, 3.0, 999.0], dtype=torch.float64)
    t_pad = torch.tensor([2.0, 1.0, 4.0, -1.0], dtype=torch.float64)
    a = losses.cosine_similarity(p, t).item()
    b = losses.cosine_similarity(p_pad, t_pad).item()
    assert math.isclose(a, b, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# pearson_correlation
# ──────────────────────────────────────────────────────────────────────


def test_pearson_positive_linear_returns_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    y = 2.0 * x + 5.0  # all positive, no -1 sentinel
    assert math.isclose(losses.pearson_correlation(x, y).item(), 1.0, abs_tol=1e-9)


def test_pearson_anticorrelation_returns_minus_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    y = -2.0 * x + 20.0  # [18,16,14,12] — all positive
    assert math.isclose(losses.pearson_correlation(x, y).item(), -1.0, abs_tol=1e-9)


def test_pearson_batched_shape():
    p = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    t = torch.tensor([[2.0, 4.0, 6.0], [6.0, 4.0, 2.0]], dtype=torch.float64)
    out = losses.pearson_correlation(p, t)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.ones(2, dtype=torch.float64), atol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# spectral_entropy_similarity
# ──────────────────────────────────────────────────────────────────────


def test_spectral_entropy_similarity_identical_returns_one():
    p = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    assert math.isclose(
        losses.spectral_entropy_similarity(p, p).item(), 1.0, abs_tol=1e-9
    )


def test_spectral_entropy_similarity_disjoint_returns_zero():
    p = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    out = losses.spectral_entropy_similarity(p, q, pseudocount=0.0).item()
    assert math.isclose(out, 0.0, abs_tol=1e-9)


def test_spectral_entropy_similarity_symmetric():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    a = losses.spectral_entropy_similarity(p, q).item()
    b = losses.spectral_entropy_similarity(q, p).item()
    assert math.isclose(a, b, abs_tol=1e-12)
    assert 0.0 <= a <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Scale-free similarities and one masking rule (review of 2026-11-06)
# ──────────────────────────────────────────────────────────────────────


def test_cosine_and_pearson_are_scale_invariant_at_small_magnitude():
    """The norm used to be floored at sqrt(eps) on the *squared* norm, so
    identical [3e-5, 4e-5] vectors scored cosine 0.025."""
    x = torch.tensor([3e-5, 4e-5], dtype=torch.float64)
    assert math.isclose(losses.cosine_similarity(x, x).item(), 1.0, abs_tol=1e-12)
    trace = torch.tensor([1.0, 1.001, 1.002, 1.001, 1.0], dtype=torch.float64) * 1e-3
    assert math.isclose(losses.pearson_correlation(trace, trace).item(), 1.0, abs_tol=1e-9)
    p = torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=torch.float64)
    t = torch.tensor([2.0, 1.0, 4.0, 1.5], dtype=torch.float64)
    for f in (losses.cosine_similarity, losses.pearson_correlation, losses.spectral_entropy_similarity):
        base = f(p, t).item()
        for scale in (1e-6, 1e6):
            assert math.isclose(f(p * scale, t).item(), base, abs_tol=1e-9), f.__name__
            assert math.isclose(f(p, t * scale).item(), base, abs_tol=1e-9), f.__name__
    base = losses.kld(p, t, reduce="sum").item()
    assert math.isclose(losses.kld(p * 1e-6, t * 1e6, reduce="sum").item(), base, abs_tol=1e-9)


def test_a_vector_with_nothing_in_it_scores_zero():
    """An empty observed scan scored entropy similarity 0.80 against a
    prediction, better than a scan with one wrong peak (0.008)."""
    pred = torch.tensor([0.5, 0.3, 0.2, 0.0], dtype=torch.float64)
    empty = torch.zeros(4, dtype=torch.float64)
    wrong = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    assert losses.spectral_entropy_similarity(pred, empty).item() == 0.0
    assert losses.spectral_entropy_similarity(pred, wrong).item() >= 0.0
    assert losses.cosine_similarity(pred, empty).item() == 0.0
    assert losses.pearson_correlation(pred, torch.full((4,), 3.0, dtype=torch.float64)).item() == 0.0
    assert math.isnan(losses.kld(pred, empty, reduce="sum").item())


def test_spectral_entropy_loss_keeps_input_unit_pseudocount():
    """The training loss keeps Cartographer's raw-unit pseudocount."""
    pred = torch.tensor([0.4, 0.3, 0.2, 0.1])
    target = torch.tensor([0.25, 0.35, 0.25, 0.15])
    logit = torch.tensor(0.3)
    pc = 0.05
    signal = target * torch.sigmoid(logit)
    pm, sm = pred + pc, signal + pc
    pn, sn = pm / pm.sum(), sm / sm.sum()
    signal_kl = (sn * (sn / pn).log()).sum()
    inter = target - signal + pc
    inter_n = inter / inter.sum()
    entropy = -(inter_n * inter_n.log()).sum()
    frac = (signal.sum() + 1e-7) / (target.sum() + 1e-7)
    want = frac * signal_kl + (1 - frac) * entropy
    got = losses.spectral_entropy_loss(pred, target, logit, pseudocount=pc)
    assert torch.allclose(got, want, atol=1e-6)


def test_spectral_angle_refuses_a_batch_without_batch_dim():
    p = torch.ones(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="batch_dim"):
        losses.spectral_angle(p, p)


def test_a_negative_or_nan_channel_is_dropped_alike_by_every_comparison():
    """One rule: a channel negative or NaN in either vector is left out."""
    from constellation.massspec.spectra.similarity import compare_spectra

    q = torch.tensor([4.0, 2.0, 3.0, 1.0], dtype=torch.float64)
    r = torch.tensor([3.0, 1.0, 4.0, 2.0], dtype=torch.float64)
    keep = [0, 1, 3]
    for method in ("cosine", "dot", "pearson", "spectral_angle", "spectral_entropy", "kld"):
        want = compare_spectra(q[keep], r[keep], method=method).item()
        for bad_q, bad_r in ((-1.0, None), (None, -1.0), (float("nan"), None), (None, float("nan"))):
            qq, rr = q.clone(), r.clone()
            if bad_q is not None:
                qq[2] = bad_q
            if bad_r is not None:
                rr[2] = bad_r
            got = compare_spectra(qq, rr, method=method).item()
            assert math.isclose(got, want, abs_tol=1e-12), (method, bad_q, bad_r)
    counts = torch.tensor([40.0, 20.0, 30.0, 10.0], dtype=torch.float64)
    want = compare_spectra(counts[keep], r[keep], method="multinomial_deviance", as_counts=True).item()
    counts[2] = -1.0
    got = compare_spectra(counts, r, method="multinomial_deviance", as_counts=True).item()
    assert math.isclose(got, want, abs_tol=1e-12)
