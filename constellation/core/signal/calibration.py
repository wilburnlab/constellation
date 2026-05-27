"""Linear axis calibration via a known reference point.

Two generic primitives:

    find_peak_in_window     argmax over a windowed slice of a 1D trace
    linear_axis_calibration affine x-axis transform pinned by one
                            known calibrant: (axis - observed) / scale
                            + target

Together they implement the "I have one known reference and a multiplicative
scale factor; align my x-axis to it" calibration pattern. NMR uses this for
Hz-to-ppm referencing (the scale factor is the spectrometer Larmor frequency);
HPLC for retention-time alignment against an internal standard; MS for
lock-mass calibration; CE for size-ladder calibration.

The modality-specific name and conventions live in the corresponding domain
wrapper — e.g. ``constellation.nmr.processing.reference.set_ppm_scale``.
"""

from __future__ import annotations

import torch


def find_peak_in_window(
    trace: torch.Tensor,
    axis: torch.Tensor,
    expected_position: float,
    search_half_width: float,
) -> float:
    """Locate the tallest point of a 1D trace within a search window.

    Builds a mask of `|axis - expected_position| <= search_half_width`,
    finds the argmax of `trace` within that mask, and returns the
    corresponding `axis` value.

    Parameters
    ----------
    trace : torch.Tensor
        Real-valued 1D tensor (e.g. absorption-mode spectrum, chromatogram).
    axis : torch.Tensor
        x-axis values, same length as `trace` (Hz / time / m/z / etc.).
    expected_position : float
        Approximate position of the calibrant on `axis`.
    search_half_width : float
        Half-width of the search window in the same units as `axis`.

    Returns
    -------
    float
        The `axis` value at the position of maximum `trace` inside the window.

    Raises
    ------
    ValueError
        If the search window contains no points (`expected_position` or
        `search_half_width` mis-specified relative to `axis`).
    """
    lo = expected_position - search_half_width
    hi = expected_position + search_half_width
    mask = (axis >= lo) & (axis <= hi)

    if not bool(mask.any()):
        raise ValueError(
            f"Search window [{lo:.6g}, {hi:.6g}] contains no axis points. "
            f"Check `expected_position` and `search_half_width` against the "
            f"axis range [{float(axis.min()):.6g}, {float(axis.max()):.6g}]."
        )

    masked_axis = axis[mask]
    masked_trace = trace[mask]
    local_idx = int(torch.argmax(masked_trace))
    return float(masked_axis[local_idx])


def linear_axis_calibration(
    axis: torch.Tensor,
    observed_position: float,
    target_position: float,
    scale: float,
) -> torch.Tensor:
    """Affine x-axis transform pinned by one known calibrant.

    Returns ``(axis - observed_position) / scale + target_position``.

    Interpretation: the calibrant is observed at position `observed_position`
    on the current axis, and is known to live at `target_position` on the
    desired axis. The transform rescales by `scale` (a unit-conversion or
    instrument-frequency factor) and shifts so the calibrant lands at the
    target.

    NMR special case (Hz → ppm referencing): `scale` is the spectrometer
    Larmor frequency in MHz, `observed_position` is the Hz position of the
    reference compound peak, `target_position` is its known ppm value
    (0.00 for TMS / DSS; 7.26 for CDCl₃; etc.).

    Parameters
    ----------
    axis : torch.Tensor
        Current 1D x-axis values.
    observed_position : float
        Position of the calibrant on the current axis (same units as `axis`).
    target_position : float
        Known position of the calibrant on the desired (output) axis.
    scale : float
        Multiplicative factor that converts axis units to the desired
        output units. For NMR Hz→ppm, this is the carrier frequency in MHz.

    Returns
    -------
    torch.Tensor
        Calibrated axis, same shape and dtype as `axis`.
    """
    if scale == 0.0:
        raise ValueError("`scale` must be non-zero.")
    return (axis - observed_position) / scale + target_position
