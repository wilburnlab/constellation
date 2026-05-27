"""Tests for `constellation.core.signal.windows` and the `apodize` primitive."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.signal import apodize
from constellation.core.signal.windows import (
    blackman,
    hamming,
    hann,
    sine_bell,
    tukey,
)


# ──────────────────────────────────────────────────────────────────────
# Shapes and dtypes
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fn", [sine_bell, hann, hamming, blackman, tukey])
@pytest.mark.parametrize("n", [4, 17, 256])
def test_window_returns_expected_length(fn, n) -> None:
    out = fn(n)
    assert out.shape == (n,)
    assert out.dtype == torch.float64


# ──────────────────────────────────────────────────────────────────────
# sine_bell
# ──────────────────────────────────────────────────────────────────────


def test_sine_bell_default_zero_at_endpoints() -> None:
    w = sine_bell(64)
    assert float(w[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(w[-1]) == pytest.approx(0.0, abs=1e-12)


def test_sine_bell_default_peak_at_centre() -> None:
    n = 65  # odd → exact centre index
    w = sine_bell(n)
    centre_idx = n // 2
    assert float(w[centre_idx]) == pytest.approx(1.0, abs=1e-9)


def test_sine_bell_shifted_starts_at_one() -> None:
    """`off=0.5, end=1.0, power=1` starts at sin(π/2)=1 and decays to sin(π)=0."""
    w = sine_bell(64, off=0.5, end=1.0, power=1)
    assert float(w[0]) == pytest.approx(1.0, abs=1e-12)
    assert float(w[-1]) == pytest.approx(0.0, abs=1e-12)


def test_sine_bell_squared_matches_sine_bell_squared_value() -> None:
    w1 = sine_bell(64, power=1)
    w2 = sine_bell(64, power=2)
    assert torch.allclose(w2, w1**2)


# ──────────────────────────────────────────────────────────────────────
# hann / hamming / blackman
# ──────────────────────────────────────────────────────────────────────


def test_hann_endpoints_and_peak() -> None:
    w = hann(64)
    assert float(w[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(w[-1]) == pytest.approx(0.0, abs=1e-12)
    centre = float(w[32])
    # Hann peaks at 1.0 at the middle for odd lengths and is near 1.0 for even.
    assert centre > 0.99


def test_hamming_endpoints_nonzero() -> None:
    w = hamming(64)
    assert float(w[0]) == pytest.approx(0.08, abs=1e-9)
    assert float(w[-1]) == pytest.approx(0.08, abs=1e-9)


def test_blackman_zero_at_endpoints_and_symmetric() -> None:
    n = 65
    w = blackman(n)
    assert float(w[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(w[-1]) == pytest.approx(0.0, abs=1e-12)
    # Symmetry
    assert torch.allclose(w, w.flip(0), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# tukey
# ──────────────────────────────────────────────────────────────────────


def test_tukey_alpha_zero_is_rectangular() -> None:
    w = tukey(32, alpha=0.0)
    assert torch.allclose(w, torch.ones(32, dtype=torch.float64))


def test_tukey_alpha_one_matches_hann() -> None:
    n = 64
    assert torch.allclose(tukey(n, alpha=1.0), hann(n), atol=1e-12)


def test_tukey_flat_middle_at_moderate_alpha() -> None:
    n = 100
    alpha = 0.4
    w = tukey(n, alpha=alpha)
    # Middle 20 % of the window should be exactly 1.0 (well inside the flat region).
    flat_slice = slice(int(0.4 * n), int(0.6 * n))
    assert torch.allclose(w[flat_slice], torch.ones_like(w[flat_slice]), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# apodize
# ──────────────────────────────────────────────────────────────────────


def test_apodize_real_trace_real_window() -> None:
    trace = torch.arange(8, dtype=torch.float64) + 1.0  # [1, 2, ..., 8]
    window = 0.5 * torch.ones(8, dtype=torch.float64)
    out = apodize(trace, window)
    assert torch.allclose(out, trace * 0.5)


def test_apodize_complex_trace_real_window_preserves_complex_dtype() -> None:
    trace = torch.tensor(
        [1.0 + 1.0j, 2.0 + 0.0j, 0.0 + 3.0j, -1.0 - 1.0j], dtype=torch.complex128
    )
    window = torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.float64)
    out = apodize(trace, window)
    assert out.dtype == torch.complex128
    expected = torch.tensor(
        [1.0 + 1.0j, 1.0 + 0.0j, 0.0 + 0.75j, 0.0 + 0.0j], dtype=torch.complex128
    )
    assert torch.allclose(out, expected)


def test_apodize_preserves_trace_shape() -> None:
    n = 17
    trace = torch.randn(n, dtype=torch.float64)
    window = sine_bell(n)
    out = apodize(trace, window)
    assert out.shape == trace.shape
