"""Tests for `core.stats.variational` — Gaussian guides + MC-ELBO + fit_vb."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.optim import AdamOptimizer
from constellation.core.stats import (
    LowRankGuide,
    MeanFieldGuide,
    NormalDistribution,
    elbo,
    fit_vb,
)

_LOG_2PI = math.log(2.0 * math.pi)


# ──────────────────────────────────────────────────────────────────────
# Subset resolution
# ──────────────────────────────────────────────────────────────────────


def test_subset_all_and_glob() -> None:
    m = NormalDistribution()
    assert MeanFieldGuide(m).k == 2  # mu + log_sigma
    assert MeanFieldGuide(m, subset=["mu"]).k == 1
    assert MeanFieldGuide(m, subset=["log_*"]).k == 1  # log_sigma via glob


def test_unmatched_subset_raises() -> None:
    with pytest.raises(KeyError):
        MeanFieldGuide(NormalDistribution(), subset=["does_not_exist"])


# ──────────────────────────────────────────────────────────────────────
# MeanFieldGuide mechanics
# ──────────────────────────────────────────────────────────────────────


def test_rsample_shape() -> None:
    g = MeanFieldGuide(NormalDistribution())
    assert g.rsample(16).shape == (16, g.k)


def test_entropy_closed_form() -> None:
    g = MeanFieldGuide(NormalDistribution(), init_scale=0.5)
    expected = 0.5 * g.k * (1.0 + _LOG_2PI) + float(g.log_scale.detach().sum())
    assert float(g.entropy().detach()) == pytest.approx(expected, rel=1e-12)


def test_loc_inits_at_model_params() -> None:
    m = NormalDistribution(mu=1.5, sigma=2.0)
    g = MeanFieldGuide(m, subset=["mu"])
    assert float(g.loc.detach()) == pytest.approx(1.5)


def test_to_model_writes_loc_leaves_others() -> None:
    m = NormalDistribution(mu=0.0, sigma=1.0)
    g = MeanFieldGuide(m, subset=["mu"])
    with torch.no_grad():
        g.loc_param.fill_(3.14)
    g.to_model(m)
    assert float(m.mu.detach()) == pytest.approx(3.14)
    assert float(m.sigma.detach()) == pytest.approx(1.0)  # log_sigma untouched


# ──────────────────────────────────────────────────────────────────────
# ELBO
# ──────────────────────────────────────────────────────────────────────


def test_elbo_scalar_with_grad_to_guide() -> None:
    torch.manual_seed(0)
    m = NormalDistribution(mu=0.0, sigma=1.0)
    data = torch.randn(50, dtype=torch.float64)
    g = MeanFieldGuide(m, subset=["mu"])
    val = elbo(m, g, data, n_samples=8)
    assert val.dim() == 0
    val.backward()
    assert g.loc_param.grad is not None
    assert g.log_scale.grad is not None
    # The model's own params must NOT accumulate grad (it's a functional copy).
    assert m.mu.grad is None


def test_vb_recovers_gaussian_mean_posterior() -> None:
    """Gaussian likelihood + flat prior: the mean-field posterior over the
    mean is exactly N(sample_mean, sigma^2/N). VB must recover both the
    location and the scale (the latter = the Laplace/√N curvature)."""
    torch.manual_seed(1)
    sigma, n = 1.0, 400
    data = 2.0 + sigma * torch.randn(n, dtype=torch.float64)
    sample_mean = float(data.mean())

    m = NormalDistribution(mu=0.0, sigma=sigma)  # sigma fixed at truth
    g = MeanFieldGuide(m, subset=["mu"], init_scale=0.2)
    opt = AdamOptimizer(g, lr=0.02)
    fit_vb(m, g, data, optimizer=opt, n_samples=128, max_iter=4000, tol=1e-12)

    assert float(g.loc.detach()) == pytest.approx(sample_mean, abs=0.02)
    assert float(g.scale.detach()) == pytest.approx(sigma / math.sqrt(n), rel=0.3)


def test_log_prior_shrinks_posterior_mean() -> None:
    torch.manual_seed(2)
    sigma, n = 1.0, 50
    data = 2.0 + sigma * torch.randn(n, dtype=torch.float64)
    sample_mean = float(data.mean())

    m = NormalDistribution(mu=0.0, sigma=sigma)

    def log_prior(params: dict[str, torch.Tensor]) -> torch.Tensor:
        # N(0, 0.3^2) prior on mu (stored == natural for a location param)
        return -0.5 * (params["mu"] / 0.3) ** 2

    g = MeanFieldGuide(m, subset=["mu"], init_scale=0.2)
    opt = AdamOptimizer(g, lr=0.02)
    fit_vb(
        m, g, data, optimizer=opt, log_prior=log_prior, n_samples=64,
        max_iter=3000, tol=1e-12,
    )
    post = float(g.loc.detach())
    assert 0.0 < post < sample_mean  # pulled toward the prior at 0


# ──────────────────────────────────────────────────────────────────────
# LowRankGuide
# ──────────────────────────────────────────────────────────────────────


def test_lowrank_rsample_shape() -> None:
    g = LowRankGuide(NormalDistribution(), rank=2)
    assert g.rsample(10).shape == (10, g.k)


def test_lowrank_entropy_reduces_to_diagonal_when_factor_zero() -> None:
    g = LowRankGuide(NormalDistribution(), rank=2, init_scale=0.5)
    with torch.no_grad():
        g.factor.zero_()
    expected = 0.5 * g.k * (1.0 + _LOG_2PI) + float(g.log_diag.detach().sum())
    assert float(g.entropy().detach()) == pytest.approx(expected, rel=1e-8)


def test_lowrank_entropy_increases_with_correlation() -> None:
    # Adding a non-trivial factor strictly increases the Gaussian entropy.
    g = LowRankGuide(NormalDistribution(), rank=1, init_scale=0.5)
    with torch.no_grad():
        g.factor.zero_()
    base = float(g.entropy().detach())
    with torch.no_grad():
        g.factor.fill_(0.3)
    assert float(g.entropy().detach()) > base
