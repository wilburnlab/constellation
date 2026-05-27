"""NMR-conventional zero + first-order phase correction.

The actual math is the polynomial phase ramp implemented in
:func:`constellation.core.signal.phase.apply_polynomial_phase`. This
module is the NMR-fluent wrapper: ``ph0`` and ``ph1`` named for the
NMR convention (zero-order, first-order), expressed in **degrees** by
default (the spectrometer convention; pass ``unit='rad'`` to bypass
the conversion).

Mathematically: ``ph0`` and ``ph1`` are the first two coefficients of
the polynomial phase expansion ``angle[k] = Σ c_p · ((k − pivot) / N)^p``.
The thin wrapper just packs them into a coefficient list and forwards.

Higher-order phase (quadratic, cubic, ...) shows up in radar pulse
compression, optics wavefront sensing, and FT-MS calibration, but is
not commonly used in NMR. Callers who want higher-order corrections
should use :func:`constellation.core.signal.phase.apply_polynomial_phase`
directly.
"""

from __future__ import annotations

import math

import pyarrow as pa

from constellation.core.signal.phase import apply_polynomial_phase
from constellation.nmr.io.schemas import fid_to_complex, log_step, replace_fid_data


def phase_correct(
    table: pa.Table,
    ph0: float,
    ph1: float,
    *,
    pivot: int | None = None,
    unit: str = "deg",
) -> pa.Table:
    """Apply zero + first-order phase correction in the NMR convention.

    Applies ``rotation[k] = exp(i · (ph0 + ph1 · (k − pivot) / N))`` to
    the complex spectrum. Convention:

        - ``ph0``: constant phase added to every point. Source: receiver
          phase offset between excitation and acquisition.
        - ``ph1``: linear phase ramp across the spectrum. Source: finite
          dead time before acquisition starts, residual GRPDLY effects
          if not fully removed in :func:`fourier.fft`, frequency-dependent
          probe response.
        - ``pivot``: the spectral index at which only ``ph0`` contributes
          (the linear term is zero there). Defaults to ``N // 2``, the
          centre of an fftshifted spectrum — i.e., the carrier frequency.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying a complex spectrum (typically after
        :func:`fourier.fft`). The function does not enforce
        ``axis_domain`` matching ``"freq"`` — applying phase to a
        time-domain FID is technically well-defined (it's a complex
        rotation) and occasionally useful.
    ph0 : float
        Zero-order phase. Units controlled by ``unit``.
    ph1 : float
        First-order phase. Units controlled by ``unit``.
    pivot : int or None
        Index where only ``ph0`` contributes. Default ``N // 2``.
    unit : str
        ``'deg'`` (default — NMR convention) or ``'rad'``.

    Returns
    -------
    pa.Table
        Phase-corrected ``NMR_FID_TABLE``.

    Raises
    ------
    ValueError
        If ``unit`` is not ``'deg'`` or ``'rad'``.
    """
    if unit == "deg":
        ph0_rad = math.radians(ph0)
        ph1_rad = math.radians(ph1)
    elif unit == "rad":
        ph0_rad = ph0
        ph1_rad = ph1
    else:
        raise ValueError(f"unit must be 'deg' or 'rad'; got {unit!r}")

    spectrum = fid_to_complex(table)
    corrected = apply_polynomial_phase(
        spectrum, coefficients=[ph0_rad, ph1_rad], pivot=pivot
    )
    out = replace_fid_data(table, corrected)
    return log_step(
        out, "phase.phase_correct", ph0=ph0, ph1=ph1, pivot=pivot, unit=unit
    )


__all__ = ["phase_correct"]
