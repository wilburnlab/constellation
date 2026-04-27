"""Tests for core.stats.distributions — log_prob / cdf cross-checks
against scipy, and parameter-recovery smoke tests via the LBFGS shim."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats as ss
import torch

from constellation.core.stats import (
    Beta,
    Dirichlet,
    Gamma,
    GeneralizedNormal,
    LogNormal,
    Multinomial,
    NormalDistribution,
    Poisson,
    StudentT,
)

from conftest import LBFGSAdapter


# ──────────────────────────────────────────────────────────────────────
# NormalDistribution
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.5, 0.7), (-2.0, 3.2)])
def test_normal_log_prob_vs_scipy(mu, sigma):
    xs = torch.linspace(-5.0, 5.0, 31, dtype=torch.float64)
    d = NormalDistribution(mu=mu, sigma=sigma)
    out = d.log_prob(xs).detach().numpy()
    expected = ss.norm.logpdf(xs.numpy(), loc=mu, scale=sigma)
    np.testing.assert_allclose(out, expected, atol=1e-10)


@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.5, 0.7), (-2.0, 3.2)])
def test_normal_cdf_vs_scipy(mu, sigma):
    xs = torch.linspace(-5.0, 5.0, 31, dtype=torch.float64)
    d = NormalDistribution(mu=mu, sigma=sigma)
    out = d.cdf(xs).detach().numpy()
    expected = ss.norm.cdf(xs.numpy(), loc=mu, scale=sigma)
    np.testing.assert_allclose(out, expected, atol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# StudentT
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mu,sigma,nu",
    [(0.0, 1.0, 4.0), (0.5, 2.0, 10.0), (-1.0, 0.5, 1.5)],
)
def test_studentt_log_prob_vs_scipy(mu, sigma, nu):
    xs = torch.linspace(-5.0, 5.0, 31, dtype=torch.float64)
    d = StudentT(mu=mu, sigma=sigma, nu=nu)
    out = d.log_prob(xs).detach().numpy()
    expected = ss.t.logpdf(xs.numpy(), df=nu, loc=mu, scale=sigma)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_studentt_cdf_pinned():
    """v1: torch lacks betainc; CDF raises pending a real use case."""
    d = StudentT(mu=0.0, sigma=1.0, nu=4.0)
    with pytest.raises(NotImplementedError, match="betainc"):
        d.cdf(torch.tensor(0.5, dtype=torch.float64))


def test_studentt_recovers_scipy_fit():
    """The Cartographer error-model port preview: fit StudentT and
    confirm parameters match scipy.stats.t.fit() within rtol=0.05."""
    rng = np.random.default_rng(0)
    samples = ss.t(df=4.0, loc=0.0, scale=1.0).rvs(2000, random_state=rng)
    samples_t = torch.from_numpy(samples).to(torch.float64)

    model = StudentT(mu=0.0, sigma=1.0, nu=2.0)
    opt = LBFGSAdapter(model, max_iter=30)
    model.fit(samples_t, optimizer=opt, max_iter=50)

    nu_torch, sigma_torch = model.nu.item(), model.sigma.item()
    nu_scipy, _, sigma_scipy = ss.t.fit(samples, floc=0.0)
    assert math.isclose(nu_torch, nu_scipy, rel_tol=0.15)
    assert math.isclose(sigma_torch, sigma_scipy, rel_tol=0.05)


# ──────────────────────────────────────────────────────────────────────
# GeneralizedNormal
# ──────────────────────────────────────────────────────────────────────


def test_generalized_normal_reduces_to_normal_at_beta2():
    """GN(μ, α, β=2) with α=σ·√2 matches Normal(μ, σ)."""
    mu, sigma = 0.0, 1.0
    alpha = sigma * math.sqrt(2.0)
    xs = torch.linspace(-3.0, 3.0, 21, dtype=torch.float64)
    gn = GeneralizedNormal(mu=mu, alpha=alpha, beta=2.0)
    norm = NormalDistribution(mu=mu, sigma=sigma)
    np.testing.assert_allclose(
        gn.log_prob(xs).detach().numpy(),
        norm.log_prob(xs).detach().numpy(),
        atol=1e-10,
    )


def test_generalized_normal_cdf_at_zero():
    """Symmetric distribution: CDF(μ) = 0.5."""
    gn = GeneralizedNormal(mu=2.0, alpha=1.5, beta=1.5)
    assert math.isclose(gn.cdf(torch.tensor(2.0)).item(), 0.5, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# Beta
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 5.0), (0.5, 0.5)])
def test_beta_log_prob_vs_scipy(alpha, beta):
    xs = torch.linspace(0.05, 0.95, 21, dtype=torch.float64)
    d = Beta(alpha=alpha, beta=beta)
    out = d.log_prob(xs).detach().numpy()
    expected = ss.beta.logpdf(xs.numpy(), alpha, beta)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_beta_cdf_pinned():
    """v1: torch lacks betainc; CDF raises pending a real use case."""
    d = Beta(alpha=2.0, beta=5.0)
    with pytest.raises(NotImplementedError, match="betainc"):
        d.cdf(torch.tensor(0.5, dtype=torch.float64))


# ──────────────────────────────────────────────────────────────────────
# Gamma
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (0.5, 2.0), (5.0, 0.5)])
def test_gamma_log_prob_vs_scipy(alpha, beta):
    xs = torch.linspace(0.1, 10.0, 21, dtype=torch.float64)
    d = Gamma(alpha=alpha, beta=beta)
    out = d.log_prob(xs).detach().numpy()
    # scipy uses scale = 1/rate
    expected = ss.gamma.logpdf(xs.numpy(), alpha, scale=1.0 / beta)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_gamma_cdf_vs_scipy():
    xs = torch.linspace(0.0, 10.0, 21, dtype=torch.float64)
    d = Gamma(alpha=2.0, beta=1.0)
    out = d.cdf(xs).detach().numpy()
    expected = ss.gamma.cdf(xs.numpy(), 2.0, scale=1.0)
    np.testing.assert_allclose(out, expected, atol=1e-8)


# ──────────────────────────────────────────────────────────────────────
# LogNormal
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.0, 0.5)])
def test_lognormal_log_prob_vs_scipy(mu, sigma):
    xs = torch.linspace(0.1, 5.0, 21, dtype=torch.float64)
    d = LogNormal(mu=mu, sigma=sigma)
    out = d.log_prob(xs).detach().numpy()
    expected = ss.lognorm.logpdf(xs.numpy(), s=sigma, scale=math.exp(mu))
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_lognormal_jacobian_relation():
    """LogNormal.log_prob(exp(z)) == Normal.log_prob(z) - z"""
    z = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64)
    ln = LogNormal(mu=0.0, sigma=1.0)
    n = NormalDistribution(mu=0.0, sigma=1.0)
    np.testing.assert_allclose(
        ln.log_prob(z.exp()).detach().numpy(),
        (n.log_prob(z) - z).detach().numpy(),
        atol=1e-10,
    )


# ──────────────────────────────────────────────────────────────────────
# Poisson
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rate", [0.5, 2.0, 10.0])
def test_poisson_log_prob_vs_scipy(rate):
    ks = torch.arange(0, 20, dtype=torch.float64)
    d = Poisson(rate=rate)
    out = d.log_prob(ks).detach().numpy()
    expected = ss.poisson.logpmf(ks.numpy(), rate)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_poisson_log_prob_at_zero():
    """log P(k=0) = -rate."""
    d = Poisson(rate=3.5)
    assert math.isclose(d.log_prob(torch.tensor(0.0)).item(), -3.5, abs_tol=1e-12)


def test_poisson_cdf_vs_scipy():
    ks = torch.arange(0, 15, dtype=torch.float64)
    d = Poisson(rate=4.0)
    out = d.cdf(ks).detach().numpy()
    expected = ss.poisson.cdf(ks.numpy(), 4.0)
    np.testing.assert_allclose(out, expected, atol=1e-8)


# ──────────────────────────────────────────────────────────────────────
# Multinomial
# ──────────────────────────────────────────────────────────────────────


def test_multinomial_log_prob_vs_scipy():
    logits = torch.tensor([0.0, 1.0, -0.5], dtype=torch.float64)
    d = Multinomial(logits=logits, K=3)
    probs = torch.softmax(logits, dim=-1).numpy()
    rng = np.random.default_rng(1)
    counts = rng.multinomial(20, probs, size=10)
    counts_t = torch.from_numpy(counts).to(torch.float64)
    out = d.log_prob(counts_t).detach().numpy()
    expected = ss.multinomial.logpmf(counts, n=20, p=probs)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_multinomial_K2_matches_binomial():
    logits = torch.tensor([0.0, 0.5], dtype=torch.float64)
    d = Multinomial(logits=logits, K=2)
    probs = torch.softmax(logits, dim=-1)
    p = probs[1].item()
    n = 10
    ks = torch.arange(0, n + 1, dtype=torch.float64)
    counts = torch.stack([n - ks, ks], dim=-1)
    out = d.log_prob(counts).detach().numpy()
    expected = ss.binom.logpmf(ks.numpy(), n, p)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_multinomial_cdf_raises():
    d = Multinomial(K=3)
    with pytest.raises(NotImplementedError):
        d.cdf(torch.tensor([1.0, 2.0, 3.0]))


# ──────────────────────────────────────────────────────────────────────
# Dirichlet
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("K", [2, 3, 5])
def test_dirichlet_log_prob_vs_scipy(K):
    rng = np.random.default_rng(K)
    alpha = rng.uniform(0.5, 3.0, size=K)
    d = Dirichlet(alpha=torch.from_numpy(alpha), K=K)
    samples = rng.dirichlet(alpha, size=10)
    samples_t = torch.from_numpy(samples).to(torch.float64)
    out = d.log_prob(samples_t).detach().numpy()
    expected = np.array([ss.dirichlet.logpdf(s, alpha) for s in samples])
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_dirichlet_cdf_raises():
    d = Dirichlet(K=3)
    with pytest.raises(NotImplementedError):
        d.cdf(torch.tensor([0.3, 0.4, 0.3]))
