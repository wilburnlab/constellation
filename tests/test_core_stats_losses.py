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
