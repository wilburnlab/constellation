"""Tests for `constellation.core.signal.phase.apply_polynomial_phase`.

Verifies the polynomial-phase generalization at orders 0, 1, 2:

- Order 0 (constant phase only) is a uniform multiplication by exp(i·c0).
- Order 1 matches the NMR-conventional ph0 + ph1 · (k − pivot)/N formula
  bit-for-bit when invoked with `coefficients=[ph0_rad, ph1_rad]`.
- Order 2 generalizes correctly: at the pivot, only c0 contributes; the
  quadratic term contributes c2·u² elsewhere.
"""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.signal.phase import apply_polynomial_phase


def _make_complex_spectrum(n: int = 16, seed: int = 0) -> torch.Tensor:
    """Random complex spectrum for round-trip tests."""
    torch.manual_seed(seed)
    return torch.randn(n, dtype=torch.complex128) + 1j * torch.randn(n, dtype=torch.complex128)


# ──────────────────────────────────────────────────────────────────────
# Order 0 — constant phase
# ──────────────────────────────────────────────────────────────────────


def test_order_zero_with_zero_coefficient_is_identity() -> None:
    spec = _make_complex_spectrum(n=32, seed=0)
    out = apply_polynomial_phase(spec, coefficients=[0.0])
    assert torch.allclose(out, spec, atol=1e-12)


def test_order_zero_with_pi_over_two_multiplies_by_i() -> None:
    spec = _make_complex_spectrum(n=32, seed=1)
    out = apply_polynomial_phase(spec, coefficients=[math.pi / 2.0])
    assert torch.allclose(out, 1j * spec, atol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# Order 1 — pivot semantics + NMR-style match
# ──────────────────────────────────────────────────────────────────────


def test_order_one_at_pivot_only_constant_term_contributes() -> None:
    n = 33  # odd so pivot = N // 2 is an exact integer index
    spec = _make_complex_spectrum(n=n, seed=2)
    pivot = n // 2
    ph0 = 0.7
    ph1 = 1.3
    out = apply_polynomial_phase(spec, coefficients=[ph0, ph1], pivot=pivot)
    # At index = pivot, u = 0, so angle = ph0. The value should be spec[pivot] * exp(i*ph0).
    expected_at_pivot = spec[pivot] * torch.exp(torch.tensor(1j * ph0, dtype=torch.complex128))
    assert abs(complex(out[pivot]) - complex(expected_at_pivot)) < 1e-12


def test_order_one_matches_nmr_conventional_formula() -> None:
    """``apply_polynomial_phase(spec, [ph0_rad, ph1_rad])`` should match
    the hardcoded NMR form ``spec * exp(i (ph0 + ph1·(k - pivot)/N))``."""
    n = 64
    spec = _make_complex_spectrum(n=n, seed=3)
    pivot = n // 2
    ph0_deg = 30.0
    ph1_deg = 90.0
    ph0 = math.radians(ph0_deg)
    ph1 = math.radians(ph1_deg)

    out_general = apply_polynomial_phase(spec, coefficients=[ph0, ph1], pivot=pivot)

    # Manual NMR-style computation.
    k = torch.arange(n, dtype=torch.float64)
    angle = ph0 + ph1 * (k - pivot) / n
    expected = spec * torch.exp(1j * angle).to(torch.complex128)

    assert torch.allclose(out_general, expected, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Order 2 — quadratic phase
# ──────────────────────────────────────────────────────────────────────


def test_order_two_zero_constant_and_linear_pure_quadratic() -> None:
    """With c0 = c1 = 0 and c2 != 0, angle[k] = c2 · u[k]^2."""
    n = 65
    spec = _make_complex_spectrum(n=n, seed=4)
    pivot = n // 2
    c2 = 0.4
    out = apply_polynomial_phase(spec, coefficients=[0.0, 0.0, c2], pivot=pivot)

    k = torch.arange(n, dtype=torch.float64)
    u = (k - pivot) / n
    angle = c2 * u**2
    expected = spec * torch.exp(1j * angle).to(torch.complex128)

    assert torch.allclose(out, expected, atol=1e-12)


def test_order_two_at_pivot_only_constant_contributes() -> None:
    """Pivot semantics generalize: at the pivot, only c0 contributes for any order."""
    n = 65
    spec = _make_complex_spectrum(n=n, seed=5)
    pivot = n // 2
    out = apply_polynomial_phase(spec, coefficients=[0.5, 2.0, 3.0], pivot=pivot)
    expected_at_pivot = spec[pivot] * torch.exp(torch.tensor(1j * 0.5, dtype=torch.complex128))
    assert abs(complex(out[pivot]) - complex(expected_at_pivot)) < 1e-12


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


def test_empty_coefficients_raises() -> None:
    spec = _make_complex_spectrum(n=8, seed=0)
    with pytest.raises(ValueError, match="at least one element"):
        apply_polynomial_phase(spec, coefficients=[])


def test_preserves_shape_and_dtype() -> None:
    spec = _make_complex_spectrum(n=16, seed=6)
    out = apply_polynomial_phase(spec, coefficients=[0.1, 0.2])
    assert out.shape == spec.shape
    assert out.dtype == spec.dtype
