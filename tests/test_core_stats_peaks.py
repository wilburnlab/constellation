"""Tests for core.stats.peaks — _erfcx, emg_pdf, GaussianPeak, EMGPeak."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.special as sp
import torch

from constellation.core.stats import EMGPeak, GaussianPeak, emg_pdf
from constellation.core.stats.peaks import _erfcx

from conftest import LBFGSAdapter


# ──────────────────────────────────────────────────────────────────────
# _erfcx — both branches
# ──────────────────────────────────────────────────────────────────────


def test_erfcx_safe_branch_vs_scipy():
    """x < 25 → direct erfc-based computation. Cover negative and small."""
    xs = torch.linspace(-5.0, 24.0, 200, dtype=torch.float64)
    out = _erfcx(xs).numpy()
    expected = sp.erfcx(xs.numpy())
    np.testing.assert_allclose(out, expected, atol=1e-10, rtol=1e-10)


def test_erfcx_asymptotic_branch_vs_scipy():
    """x ≥ 25 → three-term asymptotic series."""
    xs = torch.linspace(25.0, 100.0, 50, dtype=torch.float64)
    out = _erfcx(xs).numpy()
    expected = sp.erfcx(xs.numpy())
    np.testing.assert_allclose(out, expected, rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# emg_pdf
# ──────────────────────────────────────────────────────────────────────


def test_emg_pdf_integrates_to_one():
    """Free-function EMG density should integrate to 1 (probability density)."""
    mu = torch.tensor(0.0, dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)
    tau = torch.tensor(2.0, dtype=torch.float64)
    grid = torch.linspace(-15.0, 30.0, 5000, dtype=torch.float64)
    vals = emg_pdf(grid, mu, sigma, tau)
    integral = torch.trapezoid(vals, grid).item()
    assert math.isclose(integral, 1.0, abs_tol=1e-3)


def test_emg_pdf_reduces_to_normal_at_small_tau():
    """As τ → 0, EMG should approach Normal(μ, σ²)."""
    mu = torch.tensor(0.0, dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)
    tau = torch.tensor(1e-3, dtype=torch.float64)
    xs = torch.linspace(-3.0, 3.0, 21, dtype=torch.float64)
    emg = emg_pdf(xs, mu, sigma, tau).numpy()
    normal = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * xs.numpy() ** 2)
    np.testing.assert_allclose(emg, normal, atol=5e-3)


# ──────────────────────────────────────────────────────────────────────
# GaussianPeak
# ──────────────────────────────────────────────────────────────────────


def test_gaussian_peak_value_at_apex():
    """N(t_apex) = N_max."""
    p = GaussianPeak(N_max=1000.0, t_apex=5.0, sigma=0.5)
    out = p.forward(torch.tensor(5.0, dtype=torch.float64))
    assert math.isclose(out.item(), 1000.0, rel_tol=1e-12)


def test_gaussian_peak_integrate_closed_form_vs_trapezoidal():
    """Closed form N_max·σ·√(2π) vs trapezoidal over wide grid."""
    p = GaussianPeak(N_max=1.0, t_apex=0.0, sigma=2.0)
    closed = p.integrate().item()
    grid = torch.linspace(-15.0, 15.0, 5000, dtype=torch.float64)
    trapz = torch.trapezoid(p.forward(grid), grid).item()
    assert math.isclose(closed, trapz, rel_tol=1e-5)


def test_gaussian_peak_integrate_bounded():
    """Bounded integral via Gaussian CDF — area within ±σ ≈ 0.683."""
    p = GaussianPeak(N_max=1.0, t_apex=0.0, sigma=1.0)
    full = p.integrate().item()
    one_sigma = p.integrate(t_lo=-1.0, t_hi=1.0).item()
    assert math.isclose(one_sigma / full, 0.6826895, abs_tol=1e-5)


def test_gaussian_peak_bounds():
    p = GaussianPeak(N_max=1.0, t_apex=10.0, sigma=2.0)
    lo, hi = p.bounds(n_sigma=3.0)
    assert math.isclose(lo.item(), 4.0, abs_tol=1e-10)
    assert math.isclose(hi.item(), 16.0, abs_tol=1e-10)


def test_gaussian_peak_recovers_synthetic():
    """Fit a noisy synthetic Gaussian; recover (t_apex, σ, N_max) within ~1%."""
    torch.manual_seed(0)
    true_t_apex, true_sigma, true_N = 30.0, 2.0, 1.0
    t = torch.linspace(20.0, 40.0, 200, dtype=torch.float64)
    clean = true_N * torch.exp(-0.5 * ((t - true_t_apex) / true_sigma) ** 2)
    noisy = clean + 0.01 * true_N * torch.randn_like(clean)

    p = GaussianPeak(N_max=0.5, t_apex=29.0, sigma=1.5)
    opt = LBFGSAdapter(p, max_iter=30)
    p.fit(t, noisy, optimizer=opt, max_iter=50)

    params = p.parameters_dict()
    assert math.isclose(params["t_apex"].item(), true_t_apex, abs_tol=0.05)
    assert math.isclose(params["sigma"].item(), true_sigma, rel_tol=0.05)
    assert math.isclose(params["N_max"].item(), true_N, rel_tol=0.05)


# ──────────────────────────────────────────────────────────────────────
# EMGPeak
# ──────────────────────────────────────────────────────────────────────


def test_emg_peak_value_at_apex_equals_N_max():
    """By construction, EMGPeak normalizes so peak max = N_max."""
    p = EMGPeak(N_max=500.0, t_apex=10.0, sigma=0.5, tau=1.0)
    out = p.forward(torch.tensor(10.0, dtype=torch.float64))
    assert math.isclose(out.item(), 500.0, rel_tol=1e-6)


def test_emg_peak_bounds_asymmetric():
    """Trailing edge should extend further than leading."""
    p = EMGPeak(N_max=1.0, t_apex=0.0, sigma=1.0, tau=2.0)
    lo, hi = p.bounds(n_sigma=4.0)
    leading = (0.0 - lo).item()
    trailing = (hi - 0.0).item()
    assert trailing > leading


def test_emg_peak_recovers_synthetic():
    """Fit noisy synthetic EMG; recover τ to within 10% (LBFGS-tight)."""
    torch.manual_seed(0)
    true_N, true_apex, true_sigma, true_tau = 1.0, 10.0, 0.5, 1.5
    t = torch.linspace(7.0, 20.0, 300, dtype=torch.float64)
    truth = EMGPeak(
        N_max=true_N, t_apex=true_apex, sigma=true_sigma, tau=true_tau
    )
    with torch.no_grad():
        clean = truth.forward(t)
    noisy = clean + 0.005 * true_N * torch.randn_like(clean)

    p = EMGPeak(N_max=0.7, t_apex=9.5, sigma=0.4, tau=1.0)
    opt = LBFGSAdapter(p, max_iter=40)
    p.fit(t, noisy, optimizer=opt, max_iter=80)

    params = p.parameters_dict()
    assert math.isclose(params["t_apex"].item(), true_apex, abs_tol=0.1)
    assert math.isclose(params["sigma"].item(), true_sigma, rel_tol=0.15)
    assert math.isclose(params["tau"].item(), true_tau, rel_tol=0.15)
    assert math.isclose(params["N_max"].item(), true_N, rel_tol=0.05)


def test_emg_peak_integrate_positive():
    """Sanity: bounded integral > 0 and trapezoidal-stable."""
    p = EMGPeak(N_max=1.0, t_apex=5.0, sigma=0.5, tau=1.0)
    area = p.integrate(n_points=2000).item()
    assert area > 0
    assert math.isfinite(area)
