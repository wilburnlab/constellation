"""Monotone curves: piecewise-linear, rational-quadratic spline, extrapolation tails."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import LBFGSAdapter
from constellation.core.stats.monotone import (
    Extrapolation,
    LinearMap,
    MonotoneCurve,
    PiecewiseLinear,
    RationalQuadraticSpline,
    TailedCurve,
    interp,
    steffen_slopes,
)


def _f64(*v):
    return torch.tensor(v, dtype=torch.float64)


def test_interp_clamps_and_takes_any_values() -> None:
    xp, fp = _f64(0, 1, 2), _f64(5, -1, 3)
    assert interp(_f64(-1, 0.5, 1.5, 9), xp, fp).tolist() == [5.0, 2.0, 1.0, 3.0]


def test_piecewise_linear_averages_ties_so_staircases_invert() -> None:
    # A vertical run at x = 1 and a horizontal run at y = 2.
    curve = PiecewiseLinear(_f64(0, 1, 1, 2, 3), _f64(0, 1, 2, 2, 3))
    assert isinstance(curve, MonotoneCurve)
    assert curve.forward(_f64(1.0)).item() == 1.5
    assert curve.inverse(_f64(2.0)).item() == 2.0
    assert curve.forward(_f64(-5, 10)).tolist() == [0.0, 3.0]
    assert curve.derivative(_f64(0.5, 2.5, 9)).tolist() == [1.5, 1.0, 0.0]
    assert curve.domain == (0.0, 3.0) and curve.image == (0.0, 3.0)
    with pytest.raises(ValueError):
        PiecewiseLinear(_f64(0, 1, 2), _f64(0, 2, 1))


def test_linear_map() -> None:
    line = LinearMap(2.0, 1.0, (0.0, 4.0))
    assert line.forward(_f64(3.0)).item() == 7.0 and line.inverse(_f64(7.0)).item() == 3.0
    assert line.image == (1.0, 9.0)
    with pytest.raises(ValueError):
        LinearMap(0.0, 1.0, (0.0, 1.0))


def test_steffen_slopes_keep_linear_data_and_never_go_negative() -> None:
    x = _f64(0, 1, 3, 4, 7)
    assert torch.allclose(steffen_slopes(x, 2 * x + 1), torch.full((5,), 2.0, dtype=torch.float64))
    y = _f64(0, 0, 5, 5.1, 9)
    d = steffen_slopes(x, y)
    assert bool((d >= 0).all()) and d[1] == 0.0
    assert bool((steffen_slopes(x, y, min_fraction=0.01) > 0).all())


@pytest.fixture
def spline() -> RationalQuadraticSpline:
    x, y = _f64(0, 1, 2, 4, 7, 10), _f64(1, 1.5, 4, 4.5, 9, 30)
    return RationalQuadraticSpline.from_knots(x, y, steffen_slopes(x, y, min_fraction=1e-3))


def test_spline_passes_through_its_knots_with_their_slopes(spline) -> None:
    x, y = _f64(0, 1, 2, 4, 7, 10), _f64(1, 1.5, 4, 4.5, 9, 30)
    with torch.no_grad():
        assert torch.allclose(spline.forward(x), y, atol=1e-10)
        assert torch.allclose(spline.derivative(x), steffen_slopes(x, y, min_fraction=1e-3), rtol=1e-8)
    knots = spline.knots()
    assert knots.kind == "rational_quadratic" and torch.allclose(knots.x, x)


def test_spline_is_strictly_increasing_and_inverts(spline) -> None:
    q = torch.linspace(-3, 13, 2001, dtype=torch.float64)
    with torch.no_grad():
        y = spline.forward(q)
        assert bool((torch.diff(y) > 0).all())
        assert torch.allclose(spline.inverse(y), q, atol=1e-9)
        assert float(spline.derivative(q).min()) > 0


def test_spline_derivative_matches_autograd_and_finite_differences(spline) -> None:
    q = torch.linspace(0.05, 9.95, 37, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(spline.forward(q).sum(), q)
    with torch.no_grad():
        analytic = spline.derivative(q)
        h = 1e-6
        numeric = (spline.forward(q + h) - spline.forward(q - h)) / (2 * h)
    assert torch.allclose(grad, analytic, rtol=1e-8, atol=1e-10)
    assert torch.allclose(numeric, analytic, rtol=1e-5, atol=1e-7)


def test_spline_from_a_curve_follows_it() -> None:
    xs = torch.linspace(0, 100, 401, dtype=torch.float64)
    target = PiecewiseLinear(xs, 2 + 100 * ((xs + 10) / 110) ** 0.75)
    model = RationalQuadraticSpline.from_curve(target, 24)
    with torch.no_grad():
        q = torch.linspace(1, 99, 300, dtype=torch.float64)
        assert float((model.forward(q) - target.forward(q)).abs().max()) < 0.05


def test_a_welsch_fit_ignores_thirty_percent_outliers() -> None:
    torch.manual_seed(5)
    x = torch.rand(3000, dtype=torch.float64) * 10
    truth = lambda v: 3 + 2 * v + 0.15 * v * v  # noqa: E731
    y = truth(x) + 0.05 * torch.randn(3000, dtype=torch.float64)
    bad = torch.rand(3000) < 0.3
    y[bad] = torch.rand(int(bad.sum()), dtype=torch.float64) * 40
    model = RationalQuadraticSpline(0.0, 10.0, 8, y_lo=0.0, y_hi=40.0)
    for h in (4.0, 1.0, 0.2, 0.1):
        model.fit(
            x,
            y,
            optimizer=LBFGSAdapter(list(model.parameters()), max_iter=50),
            loss_fn=lambda pred, target, h=h: -torch.exp(-0.5 * ((target - pred) / h) ** 2).mean(),
            max_iter=20,
        )
    with torch.no_grad():
        q = torch.linspace(0.5, 9.5, 50, dtype=torch.float64)
        assert float((model.forward(q) - truth(q)).abs().max()) < 0.1


def test_parameters_dict_reports_natural_units(spline) -> None:
    params = spline.parameters_dict()
    assert set(params) == {"x_knots", "y_knots", "slopes"}
    assert params["y_knots"][0].item() == pytest.approx(1.0)


def _tails(policy, **kw) -> TailedCurve:
    curve = PiecewiseLinear(_f64(0, 5, 10), _f64(0, 5, 20))
    return TailedCurve.build(curve, policy, **kw)


def test_clamp_holds_the_end_values() -> None:
    tailed = _tails("clamp")
    assert tailed.forward(_f64(-3, 13)).tolist() == [0.0, 20.0]
    assert tailed.is_clamped(_f64(-3, 5, 13)).tolist() == [True, False, True]
    assert tailed.inverse(_f64(-1, 25)).tolist() == [0.0, 10.0]


def test_linear_tails_continue_the_end_secants() -> None:
    tailed = _tails(Extrapolation("linear", window=5.0))
    assert tailed.lower.slope == pytest.approx(1.0) and tailed.upper.slope == pytest.approx(3.0)
    assert tailed.forward(_f64(-2, 12)).tolist() == pytest.approx([-2.0, 26.0])
    assert tailed.inverse(_f64(-2, 26)).tolist() == pytest.approx([-2.0, 12.0])
    assert not bool(tailed.is_clamped(_f64(-100)).any())


def test_soft_tails_shrink_toward_the_prior_and_stop_at_the_margin() -> None:
    policy = Extrapolation("soft", margin=0.1, shrinkage=20.0, window=5.0)
    sparse = _tails(policy, x_support=_f64(9.5), prior_slope=2.0)
    dense = _tails(policy, x_support=torch.linspace(5, 10, 2000, dtype=torch.float64), prior_slope=2.0)
    none = _tails(policy, x_support=torch.empty(0, dtype=torch.float64), prior_slope=2.0)
    assert none.upper.slope == pytest.approx(2.0)  # no support: the prior
    assert dense.upper.slope == pytest.approx(3.0, rel=0.01)  # well supported: its own
    assert 2.0 < sparse.upper.slope < 3.0
    # The tail runs 10 % of the domain or of the image, whichever first,
    # and holds beyond, continuously.
    reach = dense.upper.reach
    assert reach == pytest.approx(min(1.0, 2.0 / dense.upper.slope))
    end = float(dense.forward(_f64(10 + reach)))
    assert float(dense.forward(_f64(10 + reach + 5))) == pytest.approx(end)
    assert float(dense.forward(_f64(10.0))) == pytest.approx(20.0)
    assert bool(dense.is_extrapolated(_f64(10.5))) and not bool(dense.is_clamped(_f64(10 + reach / 2)))
    y = dense.forward(_f64(10 + reach / 2))
    assert float(dense.inverse(y)) == pytest.approx(10 + reach / 2)


def test_a_zero_width_domain_gets_unbounded_tails() -> None:
    tailed = TailedCurve.build(LinearMap(1.0, 2.0, (3.0, 3.0)), "soft")
    assert math.isinf(tailed.upper.reach)
    assert float(tailed.forward(_f64(10.0))) == pytest.approx(12.0)
