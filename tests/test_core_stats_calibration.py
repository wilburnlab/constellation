"""Tests for core.stats.calibration — Sigmoidal, Hill, LogLinear."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.stats import Hill, LogLinear, Sigmoidal

from conftest import LBFGSAdapter


# ──────────────────────────────────────────────────────────────────────
# Sigmoidal (4PL)
# ──────────────────────────────────────────────────────────────────────


def test_sigmoidal_at_x50_is_midpoint():
    """y(x50) = (top + bottom) / 2, regardless of slope."""
    s = Sigmoidal(bottom=10.0, top=90.0, x50=5.0, slope=2.0)
    y = s.forward(torch.tensor(5.0, dtype=torch.float64)).item()
    assert math.isclose(y, 50.0, abs_tol=1e-9)


def test_sigmoidal_asymptotic_limits():
    s = Sigmoidal(bottom=10.0, top=90.0, x50=0.0, slope=1.0)
    y_low = s.forward(torch.tensor(-100.0, dtype=torch.float64)).item()
    y_high = s.forward(torch.tensor(100.0, dtype=torch.float64)).item()
    assert math.isclose(y_low, 10.0, abs_tol=1e-9)
    assert math.isclose(y_high, 90.0, abs_tol=1e-9)


def test_sigmoidal_recovers_synthetic_4pl():
    torch.manual_seed(0)
    true = Sigmoidal(bottom=5.0, top=95.0, x50=2.5, slope=1.5)
    x = torch.linspace(-5.0, 10.0, 100, dtype=torch.float64)
    with torch.no_grad():
        y = true.forward(x) + 0.5 * torch.randn_like(x)

    model = Sigmoidal(bottom=0.0, top=100.0, x50=0.0, slope=1.0)
    opt = LBFGSAdapter(model, max_iter=30)
    model.fit(x, y, optimizer=opt, max_iter=80)

    params = model.parameters_dict()
    assert math.isclose(params["bottom"].item(), 5.0, abs_tol=0.5)
    assert math.isclose(params["top"].item(), 95.0, abs_tol=0.5)
    assert math.isclose(params["x50"].item(), 2.5, abs_tol=0.1)
    assert math.isclose(params["slope"].item(), 1.5, rel_tol=0.1)


def test_sigmoidal_negative_slope_decreasing():
    """Negative slope → decreasing curve."""
    s = Sigmoidal(bottom=0.0, top=1.0, x50=0.0, slope=-1.0)
    y_left = s.forward(torch.tensor(-5.0, dtype=torch.float64)).item()
    y_right = s.forward(torch.tensor(5.0, dtype=torch.float64)).item()
    assert y_left > y_right


# ──────────────────────────────────────────────────────────────────────
# Hill
# ──────────────────────────────────────────────────────────────────────


def test_hill_at_K_is_half_vmax():
    """y(K) = Vmax / 2 regardless of n."""
    for n in (0.5, 1.0, 2.0, 4.0):
        h = Hill(Vmax=100.0, K=2.5, n=n)
        y = h.forward(torch.tensor(2.5, dtype=torch.float64)).item()
        assert math.isclose(y, 50.0, abs_tol=1e-9), f"n={n}"


def test_hill_n1_michaelis_menten():
    """At n=1 reduces to Michaelis-Menten: y = Vmax · x / (K + x)."""
    h = Hill(Vmax=10.0, K=2.0, n=1.0)
    xs = torch.tensor([0.5, 1.0, 2.0, 4.0, 10.0], dtype=torch.float64)
    out = h.forward(xs).detach().numpy()
    expected = 10.0 * xs.numpy() / (2.0 + xs.numpy())
    assert all(math.isclose(o, e, abs_tol=1e-9) for o, e in zip(out, expected))


def test_hill_zero_at_nonpositive_x():
    h = Hill(Vmax=1.0, K=1.0, n=2.0)
    out = h.forward(torch.tensor([-1.0, 0.0, 0.5], dtype=torch.float64))
    assert out[0].item() == 0.0
    assert out[1].item() == 0.0
    assert out[2].item() > 0.0


def test_hill_recovers_synthetic():
    torch.manual_seed(0)
    true = Hill(Vmax=5.0, K=2.0, n=2.5)
    x = torch.linspace(0.1, 20.0, 60, dtype=torch.float64)
    with torch.no_grad():
        y = true.forward(x) + 0.05 * torch.randn_like(x)

    model = Hill(Vmax=1.0, K=1.0, n=1.0)
    opt = LBFGSAdapter(model, max_iter=40)
    model.fit(x, y, optimizer=opt, max_iter=100)

    params = model.parameters_dict()
    assert math.isclose(params["Vmax"].item(), 5.0, rel_tol=0.05)
    assert math.isclose(params["K"].item(), 2.0, rel_tol=0.1)
    assert math.isclose(params["n"].item(), 2.5, rel_tol=0.15)


# ──────────────────────────────────────────────────────────────────────
# LogLinear
# ──────────────────────────────────────────────────────────────────────


def test_log_linear_value_at_unity():
    """y(1) = b (since log(1) = 0)."""
    f = LogLinear(a=2.5, b=7.0)
    y = f.forward(torch.tensor(1.0, dtype=torch.float64)).item()
    assert math.isclose(y, 7.0, abs_tol=1e-12)


def test_log_linear_recovers_closed_form():
    """5-point standard curve. Closed-form recovery via LBFGS."""
    torch.manual_seed(0)
    true_a, true_b = 2.0, 1.5
    x = torch.tensor([0.1, 1.0, 10.0, 100.0, 1000.0], dtype=torch.float64)
    y = true_a * torch.log(x) + true_b

    model = LogLinear(a=0.0, b=0.0)
    opt = LBFGSAdapter(model, max_iter=30)
    model.fit(x, y, optimizer=opt, max_iter=50)

    params = model.parameters_dict()
    assert math.isclose(params["a"].item(), true_a, abs_tol=1e-4)
    assert math.isclose(params["b"].item(), true_b, abs_tol=1e-4)


def test_log_linear_clamps_nonpositive():
    """x ≤ 0 is clamped to eps, so forward stays finite."""
    f = LogLinear(a=1.0, b=0.0)
    out = f.forward(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.isfinite(out).all()
