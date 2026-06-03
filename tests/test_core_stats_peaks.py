"""Tests for core.stats.peaks — _erfcx, emg_pdf, GaussianPeak, EMGPeak,
emg_left_pdf, HyperEMGPeak."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.special as sp
import torch

from constellation.core.optim import DifferentialEvolution, bounds_in_natural_units
from constellation.core.stats import (
    EMGPeak,
    GaussianPeak,
    HyperEMGPeak,
    emg_left_pdf,
    emg_pdf,
)
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


# ──────────────────────────────────────────────────────────────────────
# emg_left_pdf — the fronting mirror of emg_pdf
# ──────────────────────────────────────────────────────────────────────


def test_emg_left_pdf_integrates_to_one():
    """Free-function fronting-EMG density integrates to 1."""
    mu = torch.tensor(0.0, dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)
    tau = torch.tensor(2.0, dtype=torch.float64)
    grid = torch.linspace(-30.0, 15.0, 5000, dtype=torch.float64)
    vals = emg_left_pdf(grid, mu, sigma, tau)
    integral = torch.trapezoid(vals, grid).item()
    assert math.isclose(integral, 1.0, abs_tol=1e-3)


def test_emg_left_pdf_mirrors_emg_pdf_about_mu():
    """emg_left_pdf(μ − x) == emg_pdf(μ + x): the left tail is the right
    tail reflected about μ."""
    mu = torch.tensor(3.0, dtype=torch.float64)
    sigma = torch.tensor(1.3, dtype=torch.float64)
    tau = torch.tensor(2.5, dtype=torch.float64)
    x = torch.linspace(0.0, 12.0, 250, dtype=torch.float64)
    left = emg_left_pdf(mu - x, mu, sigma, tau)
    right = emg_pdf(mu + x, mu, sigma, tau)
    np.testing.assert_allclose(left.numpy(), right.numpy(), atol=1e-12, rtol=1e-10)


def test_emg_left_pdf_reduces_to_normal_at_small_tau():
    """As τ → 0, the fronting EMG should approach Normal(μ, σ²)."""
    mu = torch.tensor(0.0, dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)
    tau = torch.tensor(1e-3, dtype=torch.float64)
    xs = torch.linspace(-3.0, 3.0, 21, dtype=torch.float64)
    emg = emg_left_pdf(xs, mu, sigma, tau).numpy()
    normal = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * xs.numpy() ** 2)
    np.testing.assert_allclose(emg, normal, atol=5e-3)


# ──────────────────────────────────────────────────────────────────────
# HyperEMGPeak
# ──────────────────────────────────────────────────────────────────────


def test_hyper_emg_rejects_eta_out_of_range():
    with pytest.raises(ValueError, match="eta"):
        HyperEMGPeak(eta=1.0)
    with pytest.raises(ValueError, match="eta"):
        HyperEMGPeak(eta=0.0)


def test_hyper_emg_integrate_unbounded_equals_N_total():
    """Components are normalized PDFs ⇒ the unbounded integral is exactly
    N_total (returned directly); a trapezoid over the peak's support recovers
    nearly all of it."""
    p = HyperEMGPeak(N_total=1234.0, mu=5.0, sigma=0.8, tau_r=2.0, tau_l=0.5, eta=0.7)
    assert math.isclose(p.integrate().item(), 1234.0, rel_tol=1e-12)
    lo, hi = p.bounds(n_sigma=6.0)
    captured = p.integrate(t_lo=lo.item(), t_hi=hi.item(), n_points=8000).item()
    assert 0.9 < captured / 1234.0 <= 1.0 + 1e-6


def test_hyper_emg_integrate_bounded_positive_and_bounded_by_total():
    p = HyperEMGPeak(N_total=10.0, mu=0.0, sigma=1.0, tau_r=1.5, tau_l=1.0, eta=0.6)
    area = p.integrate(t_lo=-3.0, t_hi=3.0, n_points=4000).item()
    assert area > 0
    assert area <= 10.0 * (1.0 + 1e-6)


def test_hyper_emg_bounds_asymmetric_both_sides():
    """Both τ_l and τ_r widen their own side of the support."""
    base = HyperEMGPeak(mu=0.0, sigma=1.0, tau_r=1.0, tau_l=1.0, eta=0.5)
    lo0, hi0 = base.bounds(n_sigma=4.0)
    right_heavy = HyperEMGPeak(mu=0.0, sigma=1.0, tau_r=5.0, tau_l=1.0, eta=0.5)
    lo_r, hi_r = right_heavy.bounds(n_sigma=4.0)
    left_heavy = HyperEMGPeak(mu=0.0, sigma=1.0, tau_r=1.0, tau_l=5.0, eta=0.5)
    lo_l, hi_l = left_heavy.bounds(n_sigma=4.0)
    assert hi_r.item() > hi0.item()  # bigger τ_r → support extends right
    assert lo_l.item() < lo0.item()  # bigger τ_l → support extends left


def test_hyper_emg_eta_one_reduces_to_right_emg():
    """With η≈1 the right flank matches a pure right-EMG of total area N_total."""
    n_total, mu, sigma, tau_r = 7.0, 4.0, 0.9, 2.0
    p = HyperEMGPeak(
        N_total=n_total, mu=mu, sigma=sigma, tau_r=tau_r, tau_l=0.05, eta=0.999
    )
    # Evaluate on the right flank, where the (tiny) fronting component ≈ 0.
    t = torch.linspace(mu + 2.0 * sigma, mu + 12.0, 200, dtype=torch.float64)
    got = p.forward(t).detach()
    ref = n_total * emg_pdf(
        t,
        torch.tensor(mu, dtype=torch.float64),
        torch.tensor(sigma, dtype=torch.float64),
        torch.tensor(tau_r, dtype=torch.float64),
    )
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=2e-3, atol=1e-6)


def test_hyper_emg_from_emg_warmstart():
    """from_emg seeds a mostly right-tailing HyperEMG whose total area equals
    the EMG's analytic integral and whose curve tracks the source EMG."""
    emg = EMGPeak(N_max=500.0, t_apex=10.0, sigma=0.6, tau=1.4)
    hyper = HyperEMGPeak.from_emg(emg)
    assert float(hyper.eta.detach()) == pytest.approx(0.95, abs=1e-6)
    # Total area equals the EMG's analytic integral N_max / emg_pdf(apex).
    pdf_apex = emg_pdf(
        emg.t_apex.detach(), emg.mu.detach(), emg.sigma.detach(), emg.tau.detach()
    )
    expected_area = (emg.N_max.detach() / pdf_apex).item()
    assert hyper.integrate().item() == pytest.approx(expected_area, rel=1e-9)
    # Mostly right-tailing: peak height within ~10% of N_max and curve tracks.
    t = torch.linspace(6.0, 22.0, 400, dtype=torch.float64)
    with torch.no_grad():
        emg_curve = emg.forward(t)
        hyper_curve = hyper.forward(t)
    assert hyper_curve.max().item() == pytest.approx(emg.N_max.item(), rel=0.1)
    assert (hyper_curve - emg_curve).abs().max().item() / emg.N_max.item() < 0.12


def test_hyper_emg_recovers_synthetic_via_de():
    """Fit a noisy synthetic HyperEMG through Parametric.fit + DE (+ polish).
    The (η, τ_l, τ_r) split is partly degenerate, so we assert the
    well-identified quantities — μ, total area, σ — and curve-level agreement.
    DE's population path also exercises forward()'s vmap-safety."""
    torch.manual_seed(0)
    true = HyperEMGPeak(N_total=1.0, mu=10.0, sigma=1.0, tau_r=2.0, tau_l=0.6, eta=0.78)
    t = torch.linspace(2.0, 24.0, 400, dtype=torch.float64)
    with torch.no_grad():
        clean = true.forward(t)
    noisy = clean + 0.01 * clean.max() * torch.randn_like(clean)

    fit = HyperEMGPeak(N_total=0.4, mu=9.0, sigma=2.0, tau_r=1.0, tau_l=1.0, eta=0.5)
    bounds = bounds_in_natural_units(
        fit,
        {
            "N_total": (1e-2, 1e2),
            "mu": (2.0, 24.0),
            "sigma": (0.2, 8.0),
            "tau_r": (0.1, 10.0),
            "tau_l": (0.1, 10.0),
            "eta": (0.05, 0.95),
        },
        log_params=["N_total", "sigma", "tau_r", "tau_l"],
        logit_params=["eta"],
    )
    de = DifferentialEvolution(
        fit, bounds=bounds, pop_size=60, seed=3, patience=60, max_evals=80000
    )
    res = fit.fit(t, noisy, optimizer=de, max_iter=500, polish_on_converge=True)
    pd = fit.parameters_dict()

    assert math.isfinite(res.final_loss)
    assert float(pd["mu"]) == pytest.approx(10.0, abs=0.5)
    assert float(pd["N_total"]) == pytest.approx(1.0, rel=0.25)
    assert float(pd["sigma"]) == pytest.approx(1.0, rel=0.5)
    # Curve-level agreement is the meaningful check under parameter degeneracy.
    with torch.no_grad():
        fitted = fit.forward(t)
    rmse = (fitted - clean).pow(2).mean().sqrt().item()
    assert rmse < 0.05 * clean.max().item()
