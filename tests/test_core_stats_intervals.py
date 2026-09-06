"""Tests for `core.stats.intervals` — credible / Laplace / profile intervals."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.optim import LBFGSOptimizer
from constellation.core.stats import (
    NormalDistribution,
    credible_interval,
    laplace_cov,
    profile_interval,
)


# ──────────────────────────────────────────────────────────────────────
# credible_interval
# ──────────────────────────────────────────────────────────────────────


def test_credible_interval_quantiles() -> None:
    samples = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    lo, hi = credible_interval(samples, level=0.95)
    assert float(lo) == pytest.approx(0.025, abs=1e-3)
    assert float(hi) == pytest.approx(0.975, abs=1e-3)


def test_credible_interval_along_dim() -> None:
    torch.manual_seed(0)
    s = torch.randn(20000, 3, dtype=torch.float64)
    lo, hi = credible_interval(s, level=0.90, dim=0)
    assert lo.shape == (3,) and hi.shape == (3,)
    assert float(lo.mean()) == pytest.approx(-1.645, abs=0.1)
    assert float(hi.mean()) == pytest.approx(1.645, abs=0.1)


def test_credible_interval_rejects_bad_level() -> None:
    with pytest.raises(ValueError):
        credible_interval(torch.zeros(10), level=1.5)


# ──────────────────────────────────────────────────────────────────────
# laplace_cov
# ──────────────────────────────────────────────────────────────────────


def test_laplace_cov_gaussian_mean() -> None:
    """Hessian of -loglik w.r.t. the mean is N/sigma^2, so the Laplace
    covariance is sigma^2/N."""
    torch.manual_seed(0)
    sigma, n = 1.0, 200
    data = sigma * torch.randn(n, dtype=torch.float64)
    m = NormalDistribution(mu=float(data.mean()), sigma=sigma)  # mu at MAP
    cov = laplace_cov(m, data, subset=["mu"])
    assert cov.shape == (1, 1)
    assert float(cov[0, 0]) == pytest.approx(sigma**2 / n, rel=0.05)


def test_laplace_cov_full_is_diagonal_for_normal() -> None:
    # For a Normal, mu and log_sigma are orthogonal at the MLE → off-diagonal ≈ 0.
    torch.manual_seed(1)
    data = torch.randn(300, dtype=torch.float64)
    m = NormalDistribution(mu=0.0, sigma=1.0)
    m.fit(data, optimizer=LBFGSOptimizer(m, max_iter=50), max_iter=50, tol=1e-12)
    cov = laplace_cov(m, data)
    assert cov.shape == (2, 2)
    assert float(cov[0, 1].abs()) < 1e-3


# ──────────────────────────────────────────────────────────────────────
# profile_interval
# ──────────────────────────────────────────────────────────────────────


def test_profile_interval_gaussian_mean() -> None:
    torch.manual_seed(3)
    sigma_true, n = 1.0, 200
    data = 5.0 + sigma_true * torch.randn(n, dtype=torch.float64)
    mean = float(data.mean())
    s = float(data.std(unbiased=True))

    m = NormalDistribution(mu=0.0, sigma=1.0)
    m.fit(data, optimizer=LBFGSOptimizer(m, max_iter=50), max_iter=80, tol=1e-12)

    lo, hi = profile_interval(m, data, "mu", level=0.95)
    assert lo < mean < hi
    half = (hi - lo) / 2.0
    wald = 1.96 * s / math.sqrt(n)
    assert half == pytest.approx(wald, rel=0.3)
    # Model is restored to its MAP after profiling.
    assert float(m.mu.detach()) == pytest.approx(mean, abs=0.05)


def test_profile_interval_rejects_nonscalar() -> None:
    from torch import nn

    from constellation.core.stats import Parametric

    class Vec(Parametric):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(3, dtype=torch.float64))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (self.w * x).sum()

        def parameters_dict(self) -> dict[str, torch.Tensor]:
            return {"w": self.w.detach()}

    with pytest.raises(ValueError, match="scalar"):
        profile_interval(Vec(), torch.ones(3, dtype=torch.float64), "w")
