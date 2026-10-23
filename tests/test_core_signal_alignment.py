"""Monotone alignment through modal density: greedy, global, refinement, tails, band."""

from __future__ import annotations

import pytest
import torch

from constellation.core.signal.alignment import (
    GlobalPathOptions,
    GreedyOptions,
    fit_monotone_alignment,
    quantile_band,
    theil_sen,
)
from constellation.core.stats.monotone import LinearMap, PiecewiseLinear, RationalQuadraticSpline


def _curve(x: torch.Tensor) -> torch.Tensor:
    return 2 + 100 * ((x + 5) / 105) ** 0.8


def _pairs(n: int, *, outliers: float, seed: int, warp=_curve, noise: float = 0.3):
    g = torch.Generator().manual_seed(seed)
    x = torch.distributions.Beta(1.6, 1.4).sample((n,)).double() * 100
    y = warp(x) + noise * torch.randn(n, generator=g, dtype=torch.float64)
    bad = torch.rand(n, generator=g) < outliers
    y[bad] = torch.rand(int(bad.sum()), generator=g, dtype=torch.float64) * 105
    return x, y


def _error(alignment, warp=_curve, lo: float = 5.0, hi: float = 95.0) -> torch.Tensor:
    q = torch.linspace(lo, hi, 300, dtype=torch.float64)
    with torch.no_grad():
        return (alignment.forward(q) - warp(q)).abs()


def test_quantile_band_is_encyclopedias_index_rule() -> None:
    r = torch.arange(101, dtype=torch.float64)
    assert quantile_band(r, 0.95) == (2.0, 98.0)  # floor(100·0.025), ceil(100·0.975)
    assert quantile_band(torch.tensor([float("nan"), 1.0, 2.0]), 0.95) == (1.0, 2.0)
    assert quantile_band(torch.tensor([float("nan")])) == (float("-inf"), float("inf"))


def test_theil_sen_ignores_a_quarter_of_wrong_points() -> None:
    x = torch.arange(40, dtype=torch.float64)
    y = 3 * x + 2
    y[::4] = 500.0
    slope, intercept = theil_sen(x, y)
    assert slope == pytest.approx(3.0) and intercept == pytest.approx(2.0)


@pytest.mark.parametrize("method", ["greedy", "global"])
def test_each_method_recovers_a_curved_warp_through_outliers(method: str) -> None:
    x, y = _pairs(8000, outliers=0.2, seed=11)
    alignment = fit_monotone_alignment(x, y, method=method, refine=None)
    error = _error(alignment)
    assert float(error.mean()) < 0.15 and float(error.max()) < 1.0
    lo, hi = alignment.band
    assert lo < 0 < hi


def test_greedy_is_a_walked_piecewise_linear_curve() -> None:
    x, y = _pairs(3000, outliers=0.1, seed=12)
    alignment = fit_monotone_alignment(x, y, method="greedy")
    assert alignment.method == "greedy" and "refine" not in alignment.diagnostics
    assert isinstance(alignment.curve.curve, PiecewiseLinear)
    assert alignment.diagnostics["resolution"] == GreedyOptions().resolution


def test_the_default_is_the_global_path_refined_by_the_modal_objective() -> None:
    x, y = _pairs(2000, outliers=0.1, seed=12)
    default = fit_monotone_alignment(x, y)
    assert default.method == "global" and default.diagnostics["refine"] == "modal"
    assert isinstance(default.curve.curve, RationalQuadraticSpline)
    unrefined = fit_monotone_alignment(x, y, refine=None)
    assert "refine" not in unrefined.diagnostics
    linear = fit_monotone_alignment(x, y, curve="linear")
    assert isinstance(linear.curve.curve, PiecewiseLinear) and "refine" not in linear.diagnostics


def test_the_global_curve_is_smooth_strictly_increasing_and_inverts() -> None:
    x, y = _pairs(6000, outliers=0.2, seed=13)
    alignment = fit_monotone_alignment(x, y, method="global", refine=None)
    assert isinstance(alignment.curve.curve, RationalQuadraticSpline)
    lo, hi = alignment.domain
    q = torch.linspace(lo, hi, 2000, dtype=torch.float64)
    with torch.no_grad():
        f = alignment.forward(q)
        assert bool((torch.diff(f) > 0).all())
        assert float((alignment.inverse(f) - q).abs().max()) < 1e-6


def test_the_global_fit_is_the_same_either_way_round() -> None:
    x, y = _pairs(6000, outliers=0.2, seed=14)
    forward = fit_monotone_alignment(x, y, method="global", refine=None)
    backward = fit_monotone_alignment(y, x, method="global", refine=None)
    q = torch.linspace(10, 90, 200, dtype=torch.float64)
    with torch.no_grad():
        assert float((backward.forward(forward.forward(q)) - q).abs().max()) < 0.1


def test_the_global_path_ignores_a_dense_off_curve_cluster() -> None:
    x, y = _pairs(6000, outliers=0.1, seed=15)
    g = torch.Generator().manual_seed(16)
    # Wrong identifications sharing an elution time: dense, but less so
    # than the true band around them.
    cx = 40 + 3 * torch.randn(250, generator=g, dtype=torch.float64)
    cy = _curve(cx) + 15 + 0.5 * torch.randn(250, generator=g, dtype=torch.float64)
    alignment = fit_monotone_alignment(torch.cat([x, cx]), torch.cat([y, cy]), method="global", refine=None)
    assert float(_error(alignment, lo=30, hi=50).max()) < 1.0


@pytest.mark.parametrize("refine", ["density", "modal"])
def test_refinement_keeps_the_fit_on_the_curve(refine: str) -> None:
    x, y = _pairs(3000, outliers=0.2, seed=17)
    base = fit_monotone_alignment(x, y, method="global", refine=None)
    refined = fit_monotone_alignment(x, y, method="global", refine=refine)
    assert refined.diagnostics["refine"] == refine
    assert float(_error(refined).mean()) < 1.5 * float(_error(base).mean()) + 0.02


def test_refine_needs_a_spline() -> None:
    x, y = _pairs(500, outliers=0.0, seed=18)
    with pytest.raises(ValueError):
        fit_monotone_alignment(x, y, method="greedy", curve="linear", refine="modal")


def test_few_points_give_a_theil_sen_line() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    alignment = fit_monotone_alignment(x, 2 * x + 1, method="global")
    assert isinstance(alignment.curve.curve, LinearMap)
    assert alignment.diagnostics["fallback"] == "theil_sen"
    one = fit_monotone_alignment(torch.tensor([3.0]), torch.tensor([10.0]))
    assert float(one.forward(torch.tensor([5.0]))) == pytest.approx(12.0)


def test_extrapolation_policies_beyond_the_domain() -> None:
    x, y = _pairs(4000, outliers=0.05, seed=19, warp=lambda v: 1.5 * v + 3)
    clamp = fit_monotone_alignment(x, y, method="global", refine=None, extrapolation="clamp")
    soft = fit_monotone_alignment(x, y, method="global", refine=None, extrapolation="soft")
    lo, hi = soft.domain
    beyond = torch.tensor([hi + 2.0], dtype=torch.float64)
    with torch.no_grad():
        assert float(clamp.forward(beyond)) == pytest.approx(float(clamp.forward(torch.tensor([hi]))))
        assert float(soft.forward(beyond)) == pytest.approx(1.5 * (hi + 2) + 3, abs=0.3)
    assert bool(soft.is_extrapolated(beyond)) and not bool(soft.is_extrapolated(torch.tensor([(lo + hi) / 2])))


def test_weights_and_non_finite_pairs() -> None:
    x, y = _pairs(3000, outliers=0.0, seed=20)
    x[:5] = float("nan")
    w = torch.ones_like(x)
    w[5:10] = 0.0  # dropped like the non-finite pairs
    alignment = fit_monotone_alignment(x, y, w, method="global", refine=None)
    assert alignment.n_points == 2990


def test_options_must_match_the_method() -> None:
    x, y = _pairs(500, outliers=0.0, seed=21)
    with pytest.raises(TypeError):
        fit_monotone_alignment(x, y, method="greedy", options=GlobalPathOptions())
    with pytest.raises(ValueError):
        fit_monotone_alignment(x, y, method="ridge")
