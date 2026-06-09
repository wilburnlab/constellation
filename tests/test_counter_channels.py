"""Tests for Counter channel likelihoods, IIT conversion, and calibration."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.massspec.counter import channels as ch
from constellation.massspec.counter.calibration import GlobalCalibration
from constellation.massspec.counter.iit import accumulated_count


# ──────────────────────────────────────────────────────────────────────
# channels — student_t / intensity / censored
# ──────────────────────────────────────────────────────────────────────


def test_student_t_matches_scipy() -> None:
    ss = pytest.importorskip("scipy.stats")
    x = torch.linspace(-4, 4, 11, dtype=torch.float64)
    val = ch.student_t_log_prob(
        x, torch.tensor(0.3), torch.tensor(1.7), torch.tensor(5.0)
    )
    ref = ss.t.logpdf(x.numpy(), df=5.0, loc=0.3, scale=1.7)
    assert float((val - torch.tensor(ref)).abs().max()) < 1e-6


def test_censored_term_direction_and_grad() -> None:
    lam = torch.tensor([0.2, 5.0, 100.0], dtype=torch.float64, requires_grad=True)
    lp = ch.censored_log_prob(lam, floor_count=4.0)
    # No predicted ions → no penalty; many predicted → heavy penalty.
    assert float(lp[0].detach()) == pytest.approx(0.0, abs=1e-2)
    assert float(lp[2].detach()) < -20.0
    assert lp[0] > lp[1] > lp[2]
    lp.sum().backward()
    assert lam.grad is not None and torch.isfinite(lam.grad).all()


def test_poisson_count_log_prob_matches_scipy() -> None:
    ss = pytest.importorskip("scipy.stats")
    n = torch.tensor([1.0, 2.0, 5.0, 12.0], dtype=torch.float64)
    lam = torch.tensor([0.5, 3.0, 5.0, 10.0], dtype=torch.float64)
    val = ch.poisson_count_log_prob(n, lam)
    ref = ss.poisson.logpmf(n.numpy(), mu=lam.numpy())
    assert float((val - torch.tensor(ref)).abs().max()) < 1e-9
    # A log-PMF is <= 0, and lam->0 is heavily penalised for an observed count
    # (n>=1) -- so the detected-channel term cannot reward N->0 (no collapse).
    assert float(val.max()) <= 0.0
    spike = ch.poisson_count_log_prob(torch.tensor([1.0]), torch.tensor([1e-8]))
    assert float(spike) < -10.0


def test_detect_censor_partition_complementary() -> None:
    """`censored_log_prob` (count < floor) and the detection mass (count >=
    floor) must partition the SAME Poisson over the SAME floor and sum to 1 --
    the off-by-one regression guard (old code used floor+1)."""
    lam = torch.tensor([0.3, 1.0, 4.0, 20.0], dtype=torch.float64)
    # floor=1: P(count < 1) = P(count = 0) = e^{-lam} (NOT e^{-lam}(1+lam)).
    p_below_1 = ch.censored_log_prob(lam, floor_count=1.0).exp()
    assert torch.allclose(p_below_1, torch.exp(-lam), atol=1e-9)
    for floor in (1.0, 4.0):
        below = ch.censored_log_prob(lam, floor_count=floor).exp()
        above = 1.0 - torch.special.gammaincc(
            torch.tensor(floor, dtype=torch.float64), lam
        )
        assert torch.allclose(below + above, torch.ones_like(lam), atol=1e-9)


def test_intensity_variance_scales_with_iit_and_resolution() -> None:
    # Var ∝ 1/(τ·ρ_R): doubling τ halves the variance (wider τ → less shot noise).
    args = dict(
        intensity_obs=torch.tensor([10.0], dtype=torch.float64),
        intensity_pred=torch.tensor([10.0], dtype=torch.float64),
        sum_intensity_pred=torch.tensor([20.0], dtype=torch.float64),
        gain_z=torch.tensor([30.0], dtype=torch.float64),
        p_k=torch.tensor([0.4], dtype=torch.float64),
        nu_intensity=torch.tensor(50.0, dtype=torch.float64),
        rho_r=torch.tensor([1.0], dtype=torch.float64),
    )
    lp_lo_iit = ch.intensity_log_prob(iit=torch.tensor([10.0]), **args)
    lp_hi_iit = ch.intensity_log_prob(iit=torch.tensor([40.0]), **args)
    # At the mean (resid 0) a smaller variance → taller density → higher log-prob.
    assert float(lp_hi_iit) > float(lp_lo_iit)


# ──────────────────────────────────────────────────────────────────────
# iit — accumulated count
# ──────────────────────────────────────────────────────────────────────


def test_accumulated_count_formula() -> None:
    # N = I·τ/α  [ions]
    n = accumulated_count(
        torch.tensor(10.0), torch.tensor(20.0), torch.tensor(5.0)
    )
    assert float(n) == pytest.approx(40.0)


# ──────────────────────────────────────────────────────────────────────
# calibration
# ──────────────────────────────────────────────────────────────────────


def test_gain_models() -> None:
    z = torch.tensor([2.0, 3.0], dtype=torch.float64)
    lin = GlobalCalibration(alpha_model="linear", alpha0=1.0, alpha1=15.0)
    assert torch.allclose(
        lin.gain(z), torch.nn.functional.softplus(1.0 + 15.0 * z)
    )
    origin = GlobalCalibration(alpha_model="linear_origin", alpha1=10.0)
    assert torch.allclose(origin.gain(z), torch.nn.functional.softplus(10.0 * z))
    perz = GlobalCalibration(
        alpha_model="per_z", charges=(2, 3, 4), alpha0=0.0, alpha1=10.0
    )
    g = perz.gain(torch.tensor([2.0, 4.0]))
    assert g.shape == (2,)
    assert float(g[0].detach()) == pytest.approx(20.0, rel=1e-6)  # exp(log(0+10*2))


def test_mz_center_da_to_ppm() -> None:
    # d_mz_1 = 0.001 Da at charge 2, m/z 500 → (0.001/2)/500*1e6 = 1.0 ppm.
    cal = GlobalCalibration(
        n_isotopes=3, mz_offset_ppm=0.5, d_mz_da=[0.0, 0.001, 0.0]
    )
    center = cal.mz_center_ppm(
        torch.tensor([500.0]), torch.tensor([2.0]), torch.tensor([1])
    )
    assert float(center.detach()) == pytest.approx(0.5 + 1.0, abs=1e-6)
    # isotope 0 → only the global offset
    c0 = cal.mz_center_ppm(
        torch.tensor([500.0]), torch.tensor([2.0]), torch.tensor([0])
    )
    assert float(c0.detach()) == pytest.approx(0.5, abs=1e-9)


def test_mz_center_pins_monoisotopic_dmz() -> None:
    # d_mz_0 is the reference, pinned to 0: the isotope-0 center is `mz_offset`
    # only, regardless of a nonzero d_mz_da[0] (which would otherwise be
    # degenerate with mz_offset). Isotope-1 still uses its d_mz_da[1].
    cal = GlobalCalibration(
        n_isotopes=3, mz_offset_ppm=0.5, d_mz_da=[0.7, 0.001, 0.0]
    )
    c0 = cal.mz_center_ppm(
        torch.tensor([500.0]), torch.tensor([2.0]), torch.tensor([0])
    )
    assert float(c0.detach()) == pytest.approx(0.5, abs=1e-9)  # d_mz_0 ignored
    c1 = cal.mz_center_ppm(
        torch.tensor([500.0]), torch.tensor([2.0]), torch.tensor([1])
    )
    assert float(c1.detach()) == pytest.approx(0.5 + 1.0, abs=1e-6)  # +1 ppm


def test_alpha_mz_nu_mz_and_rho_defaults() -> None:
    cal = GlobalCalibration(alpha_mz=0.6, nu_mz=7.8)
    assert float(cal.alpha_mz.detach()) == pytest.approx(0.6, rel=1e-6)
    assert float(cal.nu_mz.detach()) == pytest.approx(7.8, rel=1e-6)
    # No R_ref/mz_ref → ρ_R = 1 (no resolution scaling).
    assert float(cal.rho_R(torch.tensor([700.0]))) == pytest.approx(1.0)
    # With R(m/z) set, ρ_R = (mz_ref/mz)^(ρ/2).
    cal2 = GlobalCalibration(rho=0.5, r_ref=60000.0, mz_ref=200.0)
    rr = cal2.rho_R(torch.tensor([800.0]))
    assert float(rr.detach()) == pytest.approx((200.0 / 800.0) ** 0.25, rel=1e-6)


def test_unknown_alpha_model_raises() -> None:
    with pytest.raises(ValueError):
        GlobalCalibration(alpha_model="quadratic")  # type: ignore[arg-type]
