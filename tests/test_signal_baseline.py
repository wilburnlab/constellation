"""Tests for `constellation.core.signal.baseline`.

Covers the four entry points:
    - detect_baseline_regions
    - polynomial
    - spline
    - arpls

Plus the regression test that motivated this port: arPLS recovers a
curved baseline accurately when only outer anchor regions are available,
where spline (the previous nmrwelder default) overshoots in the middle.
"""

from __future__ import annotations

import torch

from constellation.core.signal.baseline import (
    arpls,
    detect_baseline_regions,
    polynomial,
    spline,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ──────────────────────────────────────────────────────────────────────


def _two_gaussian_peaks(n: int, centers: tuple[int, int], height: float, sigma: float) -> torch.Tensor:
    """Two Gaussian peaks at given indices."""
    idx = torch.arange(n, dtype=torch.float64)
    peaks = torch.zeros(n, dtype=torch.float64)
    for c in centers:
        peaks = peaks + height * torch.exp(-((idx - c) ** 2) / (2.0 * sigma**2))
    return peaks


def _flat_spectrum(n: int = 512, seed: int = 0) -> torch.Tensor:
    """Two peaks on a flat zero baseline plus small noise."""
    torch.manual_seed(seed)
    peaks = _two_gaussian_peaks(n, centers=(n // 3, 2 * n // 3), height=5.0, sigma=10.0)
    noise = 0.02 * torch.randn(n, dtype=torch.float64)
    return peaks + noise


def _curved_spectrum(n: int = 512, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two peaks + a parabolic baseline + noise.

    Returns `(spectrum, true_baseline)`. Baseline has a positive maximum
    at the centre and falls to zero at the edges.
    """
    torch.manual_seed(seed)
    x = (torch.arange(n, dtype=torch.float64) - n / 2.0) / (n / 2.0)
    true_baseline = 1.0 - x**2  # 0 at edges, 1 at centre
    peaks = _two_gaussian_peaks(n, centers=(n // 3, 2 * n // 3), height=5.0, sigma=10.0)
    noise = 0.02 * torch.randn(n, dtype=torch.float64)
    return true_baseline + peaks + noise, true_baseline


# ──────────────────────────────────────────────────────────────────────
# detect_baseline_regions
# ──────────────────────────────────────────────────────────────────────


def test_detect_baseline_regions_returns_outer_intervals() -> None:
    spec = _flat_spectrum(n=512, seed=0)
    # Inject one tall peak in the middle to ensure the region detector
    # excludes it.
    spec = spec.clone()
    spec[256] = 10.0
    regions = detect_baseline_regions(spec, noise_factor=3.0, min_region_width=20)
    assert regions, "expected at least one baseline region for a flat-with-peak spectrum"
    # No region should contain the peak index 256.
    assert all(not (start <= 256 < end) for start, end in regions)


def test_detect_baseline_regions_min_width_excludes_short_runs() -> None:
    n = 256
    spec = torch.zeros(n, dtype=torch.float64)
    # Single point of noise above threshold — should not create a region of width 1.
    regions = detect_baseline_regions(spec, noise_factor=3.0, min_region_width=10)
    assert all((end - start) >= 10 for start, end in regions)


# ──────────────────────────────────────────────────────────────────────
# polynomial baseline
# ──────────────────────────────────────────────────────────────────────


def test_polynomial_on_flat_baseline_returns_near_zero_off_peaks() -> None:
    spec = _flat_spectrum(n=512, seed=0)
    corrected = polynomial(spec, regions=[(0, 100), (400, 512)], degree=2)
    # Off-peak regions should be close to zero post-correction.
    off_peak = torch.cat([corrected[:80], corrected[420:]])
    assert float(off_peak.abs().mean()) < 0.1


def test_polynomial_recovers_curved_baseline() -> None:
    n = 512
    spec, true_baseline = _curved_spectrum(n=n, seed=0)
    corrected = polynomial(spec, regions=[(0, 80), (n - 80, n)], degree=2)
    recovered_baseline = spec - corrected
    # In the off-peak outer regions, recovered baseline should match true.
    off_idx = torch.cat([torch.arange(80), torch.arange(n - 80, n)])
    err = float(
        torch.mean((recovered_baseline[off_idx] - true_baseline[off_idx]) ** 2)
    )
    assert err < 1e-2


# ──────────────────────────────────────────────────────────────────────
# spline baseline
# ──────────────────────────────────────────────────────────────────────


def test_spline_on_flat_baseline_returns_near_zero_off_peaks() -> None:
    spec = _flat_spectrum(n=512, seed=0)
    corrected = spline(spec, regions=[(0, 100), (400, 512)])
    off_peak = torch.cat([corrected[:80], corrected[420:]])
    assert float(off_peak.abs().mean()) < 0.1


def test_spline_preserves_shape_and_dtype() -> None:
    spec = _flat_spectrum(n=256, seed=1)
    corrected = spline(spec, regions=[(0, 50), (200, 256)])
    assert corrected.shape == spec.shape
    assert corrected.dtype == spec.dtype


# ──────────────────────────────────────────────────────────────────────
# arpls baseline
# ──────────────────────────────────────────────────────────────────────


def test_arpls_on_flat_baseline_returns_near_zero_off_peaks() -> None:
    spec = _flat_spectrum(n=512, seed=0)
    corrected = arpls(spec, lam=1e4)
    off_peak = torch.cat([corrected[:80], corrected[420:]])
    assert float(off_peak.abs().mean()) < 0.1


def test_arpls_recovers_curved_baseline_without_regions() -> None:
    n = 1024
    spec, true_baseline = _curved_spectrum(n=n, seed=2)
    corrected = arpls(spec, lam=1e5)
    recovered_baseline = spec - corrected
    err = float(torch.mean((recovered_baseline - true_baseline) ** 2))
    # arPLS without regions should recover the curved baseline to high accuracy.
    assert err < 5e-2


def test_arpls_preserves_shape_and_dtype() -> None:
    spec = _flat_spectrum(n=256, seed=3)
    corrected = arpls(spec, lam=1e4)
    assert corrected.shape == spec.shape
    assert corrected.dtype == spec.dtype


# ──────────────────────────────────────────────────────────────────────
# Regression: arPLS beats spline on outer-anchor curved-baseline case
# ──────────────────────────────────────────────────────────────────────


def test_arpls_beats_spline_on_outer_anchor_curved_baseline() -> None:
    """The NH-region overshoot regression motivating this port.

    With only outer-anchor regions available and a curved (parabolic)
    baseline plus dense peaks in the middle, the natural cubic spline's
    boundary conditions force zero second derivative at the endpoints,
    so the cubic interpolating the middle gap can't match the true
    curvature. arPLS, which uses no regions, recovers the curved baseline
    accurately. This test asserts arPLS beats spline by a large margin
    in mean-squared error over the dense middle region.
    """
    n = 1024
    spec, true_baseline = _curved_spectrum(n=n, seed=4)

    outer_regions = [(0, n // 8), (n - n // 8, n)]
    spline_corrected = spline(spec, regions=outer_regions)
    arpls_corrected = arpls(spec, lam=1e5)

    spline_baseline = spec - spline_corrected
    arpls_baseline = spec - arpls_corrected

    mid_slice = slice(n // 4, 3 * n // 4)
    spline_err = float(
        torch.mean((spline_baseline[mid_slice] - true_baseline[mid_slice]) ** 2)
    )
    arpls_err = float(
        torch.mean((arpls_baseline[mid_slice] - true_baseline[mid_slice]) ** 2)
    )

    assert arpls_err < spline_err / 3.0, (
        f"arpls should clearly beat spline on this fixture. "
        f"spline MSE={spline_err:.4f}, arpls MSE={arpls_err:.4f}"
    )
