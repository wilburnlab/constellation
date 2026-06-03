"""Tests for the Dirichlet-multinomial distribution + log-prob free function."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from constellation.core.stats.distributions import (
    DirichletMultinomial,
    Multinomial,
    dirichlet_multinomial_log_prob,
)


def test_logpmf_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    a = np.array([2.0, 3.0, 5.0])
    x = np.array([4, 1, 5])
    got = dirichlet_multinomial_log_prob(
        torch.tensor(x, dtype=torch.float64), torch.tensor(a)
    ).item()
    exp = float(sp.dirichlet_multinomial.logpmf(x, a, int(x.sum())))
    assert abs(got - exp) < 1e-9


def test_reduces_to_multinomial_as_alpha0_large():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    x = torch.tensor([10.0, 6.0, 4.0])
    dm = DirichletMultinomial(concentration=1e7 * p).log_prob(x).item()
    mn = Multinomial(logits=torch.log(p)).log_prob(x).item()
    assert abs(dm - mn) < 1e-3


def test_properties_recover_p_and_alpha0():
    p = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    dm = DirichletMultinomial(concentration=20.0 * p)
    assert abs(dm.alpha0.item() - 20.0) < 1e-9
    assert torch.allclose(dm.probs, p, atol=1e-12)
    assert dm.K == 3


def test_batched_log_prob_shape():
    a = torch.tensor([2.0, 3.0, 5.0])
    x = torch.tensor([[4.0, 1.0, 5.0], [2.0, 2.0, 6.0]])
    assert dirichlet_multinomial_log_prob(x, a).shape == (2,)


def test_overdispersion_widens_variance():
    """DirMult (finite α₀) draws have larger proportion variance than the
    multinomial (α₀→∞) at the same n, p — the overdispersion."""
    rng = np.random.default_rng(0)
    p = np.array([0.4, 0.35, 0.25])
    N, R = 200, 4000
    a = 15.0 * p  # α₀ = 15 (strong overdispersion)
    dm = np.array([rng.multinomial(N, rng.dirichlet(a)) for _ in range(R)]) / N
    mn = np.array([rng.multinomial(N, p) for _ in range(R)]) / N
    # variance of p̂_0 should be markedly larger under the Dirichlet-multinomial
    assert dm[:, 0].var() > 2.0 * mn[:, 0].var()


def test_log_prob_normalizes_over_count_simplex():
    """ΣP(x)=1 over all count vectors summing to n (small n, K)."""
    a = torch.tensor([1.5, 2.0, 1.0])
    n = 5
    total = 0.0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            x = torch.tensor([i, j, n - i - j], dtype=torch.float64)
            total += float(torch.exp(dirichlet_multinomial_log_prob(x, a)))
    assert abs(total - 1.0) < 1e-9
