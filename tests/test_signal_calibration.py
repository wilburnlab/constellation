"""Tests for `constellation.core.signal.calibration`."""

from __future__ import annotations

import pytest
import torch

from constellation.core.signal.calibration import (
    find_peak_in_window,
    linear_axis_calibration,
)


# ──────────────────────────────────────────────────────────────────────
# find_peak_in_window
# ──────────────────────────────────────────────────────────────────────


def test_find_peak_in_window_locates_clear_peak() -> None:
    n = 512
    axis = torch.linspace(-1000.0, 1000.0, n, dtype=torch.float64)
    trace = torch.zeros(n, dtype=torch.float64)
    # Gaussian peak centred at axis = 250 Hz.
    peak_axis = 250.0
    sigma_axis = 5.0
    trace = trace + torch.exp(-((axis - peak_axis) ** 2) / (2.0 * sigma_axis**2))
    found = find_peak_in_window(trace, axis, expected_position=240.0, search_half_width=30.0)
    assert abs(found - peak_axis) < 5.0  # within one sample of axis spacing


def test_find_peak_in_window_picks_largest_in_window_not_global_max() -> None:
    n = 512
    axis = torch.linspace(0.0, 1000.0, n, dtype=torch.float64)
    trace = torch.zeros(n, dtype=torch.float64)
    # Bigger peak outside the search window; smaller peak inside.
    trace[100] = 10.0  # outside search window for expected_position=800
    trace[400] = 3.0   # inside  search window
    found = find_peak_in_window(
        trace, axis, expected_position=float(axis[400]), search_half_width=20.0
    )
    assert abs(found - float(axis[400])) < 5.0


def test_find_peak_in_window_raises_on_empty_window() -> None:
    axis = torch.linspace(0.0, 100.0, 128, dtype=torch.float64)
    trace = torch.zeros(128, dtype=torch.float64)
    with pytest.raises(ValueError, match="contains no axis points"):
        find_peak_in_window(
            trace, axis, expected_position=500.0, search_half_width=10.0
        )


# ──────────────────────────────────────────────────────────────────────
# linear_axis_calibration
# ──────────────────────────────────────────────────────────────────────


def test_linear_axis_calibration_identity_when_scale_one_observed_target_match() -> None:
    axis = torch.linspace(-10.0, 10.0, 21, dtype=torch.float64)
    out = linear_axis_calibration(axis, observed_position=0.0, target_position=0.0, scale=1.0)
    assert torch.allclose(out, axis)


def test_linear_axis_calibration_nmr_hz_to_ppm() -> None:
    """NMR special case: 600 MHz instrument, reference at observed Hz, target 0 ppm.

    The axis is chosen so the observed Hz value lands exactly on a grid
    point, allowing a direct equality check on the calibrated value at
    that index.
    """
    # n = 31, spacing 200 Hz over [-3000, 3000] → -2400 is exactly axis[3].
    n = 31
    axis_hz = torch.linspace(-3000.0, 3000.0, n, dtype=torch.float64)
    observed_hz = -2400.0
    obs_idx = int(torch.argmin(torch.abs(axis_hz - observed_hz)))
    assert float(axis_hz[obs_idx]) == observed_hz  # confirm exact grid hit

    out = linear_axis_calibration(
        axis_hz,
        observed_position=observed_hz,
        target_position=0.0,
        scale=600.0,  # MHz
    )
    # At the observed Hz position, the output should be exactly target ppm.
    assert abs(float(out[obs_idx]) - 0.0) < 1e-12
    # The output should be a linear function of the input: slope = 1/scale.
    diffs_in = torch.diff(axis_hz)
    diffs_out = torch.diff(out)
    expected_slope = 1.0 / 600.0
    assert torch.allclose(diffs_out / diffs_in, torch.full_like(diffs_in, expected_slope))


def test_linear_axis_calibration_raises_on_zero_scale() -> None:
    axis = torch.zeros(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-zero"):
        linear_axis_calibration(axis, observed_position=0.0, target_position=0.0, scale=0.0)
