"""Polynomial phase correction for complex 1D signals.

Applies a polynomial phase profile to a complex-valued 1D signal:

    angle[k] = Σ_p c_p · ((k − pivot) / N)^p
    corrected[k] = signal[k] · exp(i · angle[k])

The polynomial order is set by `len(coefficients)`. Constellation NMR's
0 + 1 order case (`ph0` + `ph1` in degrees) is the special case
`coefficients = [ph0_rad, ph1_rad]`; higher-order corrections (used in
radar pulse compression, beamforming, optics wavefront sensing, FT-MS
high-res calibration) drop in by extending the coefficient list. None of
those modalities are in constellation today; the general form is here
because writing it the wrong way around (`phase_correct(ph0, ph1)`) would
hide the fact that `ph0` and `ph1` are just consecutive coefficients of
the same polynomial.

Coefficients are in **radians**. Domain-specific wrappers handle unit
conversions at the wrapper boundary (e.g. NMR ports degrees ↔ radians in
``constellation.nmr.processing.phase.phase_correct``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def apply_polynomial_phase(
    spectrum: torch.Tensor,
    coefficients: Sequence[float] | torch.Tensor,
    pivot: int | None = None,
) -> torch.Tensor:
    """Apply a polynomial phase correction to a complex 1D signal.

    Computes ``corrected[k] = spectrum[k] · exp(i · angle[k])`` where

        angle[k] = Σ_p coefficients[p] · ((k − pivot) / N)^p

    is a polynomial of degree `len(coefficients) − 1` in the
    pivot-centred normalised index `(k − pivot) / N`. At `k = pivot` the
    argument is zero, so only `coefficients[0]` contributes — that's the
    defining property of the pivot.

    Parameters
    ----------
    spectrum : torch.Tensor
        Complex-valued 1D signal.
    coefficients : sequence of float or 1D torch.Tensor
        Polynomial coefficients in radians, ordered low-to-high
        (`coefficients[0]` is the constant term, `coefficients[1]` is
        the linear coefficient, etc.). NMR's `phase_correct(ph0, ph1)`
        in degrees becomes
        `coefficients = [math.radians(ph0), math.radians(ph1)]` here.
    pivot : int or None
        Index where only the constant term contributes (the centre of
        the first-order correction). Defaults to `N // 2`, which after
        `fftshift` corresponds to the carrier frequency for NMR.

    Returns
    -------
    torch.Tensor
        Phase-corrected complex spectrum, same shape and dtype as input.

    Raises
    ------
    ValueError
        If `coefficients` is empty.
    """
    if len(coefficients) == 0:
        raise ValueError("`coefficients` must have at least one element.")

    n = int(spectrum.shape[-1])
    if pivot is None:
        pivot = n // 2

    f64 = torch.float64
    coeffs = torch.as_tensor(coefficients, dtype=f64)

    k = torch.arange(n, dtype=f64)
    u = (k - pivot) / n  # pivot-centred normalised index

    # angle[k] = sum_p coeffs[p] * u[k]^p
    # u.unsqueeze(-1) has shape (N, 1); the power axis broadcasts.
    p_idx = torch.arange(coeffs.shape[-1], dtype=f64)
    powers = u.unsqueeze(-1) ** p_idx  # shape (N, P)
    angle = (powers * coeffs).sum(dim=-1)

    rotation = torch.exp(1j * angle).to(spectrum.dtype)
    return spectrum * rotation
