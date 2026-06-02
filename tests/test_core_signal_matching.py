"""Tests for the core.signal sorted-axis matching kernel."""

from __future__ import annotations

import pytest
import torch

from constellation.core.signal import (
    bounds_within_tolerance,
    nearest_within_tolerance,
    tolerance_window,
)
from constellation.core.stats.units import ppm_to_da


def _f64(*xs):
    return torch.tensor(list(xs), dtype=torch.float64)


# ──────────────────────────────────────────────────────────────────────
# tolerance_window — shared ppm/Da half-width
# ──────────────────────────────────────────────────────────────────────


def test_tolerance_window_ppm_matches_ppm_to_da():
    q = _f64(100.0, 500.0, 1000.0)
    assert torch.equal(tolerance_window(q, 20.0, "ppm"), ppm_to_da(20.0, q))


def test_tolerance_window_ppm_is_default_unit():
    q = _f64(100.0, 500.0, 1000.0)
    assert torch.equal(tolerance_window(q, 20.0), tolerance_window(q, 20.0, "ppm"))


def test_tolerance_window_da_is_constant():
    q = _f64(100.0, 500.0, 1000.0)
    assert torch.equal(tolerance_window(q, 0.01, "Da"), torch.full_like(q, 0.01))


def test_tolerance_window_rejects_bad_unit():
    with pytest.raises(ValueError, match="unit"):
        tolerance_window(_f64(100.0), 1.0, "foo")  # type: ignore[arg-type]


def test_nearest_ppm_match_and_reject():
    axis = _f64(100.0, 200.0, 300.0006, 400.0)  # 300.0006 ~ +2 ppm of 300.0
    idx, within, delta = nearest_within_tolerance(axis, _f64(300.0, 305.0), 10.0, "ppm")
    assert idx.tolist() == [2, 2]
    assert within.tolist() == [True, False]


def test_signed_delta_sign_is_query_minus_observed():
    axis = _f64(300.0006)
    _, _, delta = nearest_within_tolerance(axis, _f64(300.0), 10.0, "ppm")
    # query 300.0 below observed 300.0006 → negative
    assert delta[0].item() < 0
    assert abs(delta[0].item() + 0.0006) < 1e-9


def test_nearest_da_unit():
    axis = _f64(200.0, 200.02)
    _, within, _ = nearest_within_tolerance(axis, _f64(200.005), 0.01, "Da")
    assert within.tolist() == [True]
    _, within2, _ = nearest_within_tolerance(axis, _f64(200.5), 0.01, "Da")
    assert within2.tolist() == [False]


def test_neighbor_tie_and_boundary_clamp():
    axis = _f64(100.0, 102.0)
    # query equidistant → use_left (<=) picks left neighbor
    idx, _, _ = nearest_within_tolerance(axis, _f64(101.0), 50000.0, "ppm")
    assert idx.tolist() == [0]
    # query below min and above max clamp to ends
    idx2, within2, _ = nearest_within_tolerance(axis, _f64(50.0, 500.0), 100.0, "ppm")
    assert idx2.tolist() == [0, 1]
    assert within2.tolist() == [False, False]


def test_batched_rows_independent():
    axis = torch.tensor(
        [[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]], dtype=torch.float64
    )
    q = torch.tensor([[20.0001], [200.0]], dtype=torch.float64)
    idx, within, _ = nearest_within_tolerance(axis, q, 20.0, "ppm")
    assert idx.tolist() == [[1], [1]]
    assert within.tolist() == [[True], [True]]


def test_inf_padding_never_wins():
    axis = _f64(100.0, float("inf"), float("inf"))
    idx, within, _ = nearest_within_tolerance(axis, _f64(100.0001), 20.0, "ppm")
    assert idx.tolist() == [0]
    assert within.tolist() == [True]


def test_empty_axis():
    idx, within, delta = nearest_within_tolerance(_f64(), _f64(5.0), 10.0, "ppm")
    assert within.tolist() == [False]
    assert torch.isnan(delta).all()


def test_bounds_empty_single_multi():
    axis = _f64(100.0, 100.001, 100.002, 200.0)
    lo, hi = bounds_within_tolerance(axis, _f64(100.001, 500.0), 20.0, "ppm")
    # 100.001 ± 20ppm (~0.002) covers the first three; 500 covers none
    assert (hi[0] - lo[0]).item() == 3
    assert (hi[1] - lo[1]).item() == 0


def test_bounds_degenerate_two_peaks_in_window():
    axis = _f64(500.000, 500.004)
    lo, hi = bounds_within_tolerance(axis, _f64(500.0), 20.0, "ppm")  # 0.01 window
    assert (hi[0] - lo[0]).item() == 2


def test_bounds_empty_axis():
    lo, hi = bounds_within_tolerance(_f64(), _f64(5.0), 10.0, "ppm")
    assert lo.tolist() == [0] and hi.tolist() == [0]
