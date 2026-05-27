"""Heuristic 1D-trace primitives shared across modalities.

`core.signal` ships operations that act on a single 1D trace (intensity-vs-x)
and are not modality-specific: baseline correction, windowing, polynomial
phase correction, and linear axis calibration. Per the project-wide import
DAG, `signal` sits between `stats` and `structure`, importing only from
`torch` (and from `core.stats` when a future smoothing / peak-picking
helper wants a parametric peak shape — not used by anything shipped here).

Submodules:
    baseline      polynomial, spline, arpls (peak-aware penalized LS)
    calibration   find_peak_in_window, linear_axis_calibration
    windows       sine_bell + DSP standards (hann/hamming/blackman/tukey)
    phase         apply_polynomial_phase (arbitrary order; NMR's 0+1 is
                  the special case [c0, c1])

Top-level export:
    apodize(trace, window)  — the canonical name for the multiply-trace-by-
                              window operation. One line, but named at the
                              package root so the call site reads identically
                              across NMR apodization, future chromatography
                              prefiltering, and anywhere else windowing
                              applies.

Modality-specific wrappers (NMR-conventional Hz parametrization, ppm
referencing, degrees-vs-radians phase, etc.) live in the corresponding
domain module (`constellation.nmr.processing.*`) and compose these
primitives.
"""

from __future__ import annotations

import torch

from .baseline import arpls, detect_baseline_regions, polynomial, spline
from .calibration import find_peak_in_window, linear_axis_calibration
from .phase import apply_polynomial_phase
from .windows import blackman, hamming, hann, sine_bell, tukey


def apodize(trace: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Multiply a 1D trace by a window function.

    The window is cast to the trace's dtype before multiplying, so a
    real-valued window can be applied to a complex-valued trace
    (multiplies both real and imaginary parts identically).

    Parameters
    ----------
    trace : torch.Tensor
        1D signal (real or complex).
    window : torch.Tensor
        Real-valued window function, same length as `trace`.

    Returns
    -------
    torch.Tensor
        Apodized trace, same shape and dtype as `trace`.
    """
    return trace * window.to(trace.dtype)


__all__ = [
    "apodize",
    # baseline
    "arpls",
    "detect_baseline_regions",
    "polynomial",
    "spline",
    # calibration
    "find_peak_in_window",
    "linear_axis_calibration",
    # phase
    "apply_polynomial_phase",
    # windows
    "blackman",
    "hamming",
    "hann",
    "sine_bell",
    "tukey",
]
