"""Heuristic 1D-trace operations, sorted-axis matching and monotone alignment.

Residents: the tolerance-gated matching kernel (``matching``) shared by MS
chromatogram extraction, peak annotation, and search; M4 trace
decimation for drawing (``decimate``); monotone paths through the mode
of a gridded density (``modal_path``: EncyclopeDIA's greedy walk and a
globally optimal dynamic-programming path); and monotone alignment of
one axis onto another through their joint density (``alignment``), the
retention-time alignment's generic core; and the phase of one periodic
component of a signal (``phase``), the m/z axis's generic core.

Batched, masked trace operations (2026-11-11; rows of ``[..., N]`` with a
validity mask, irregular axes allowed): ``smooth`` (moving average,
Gaussian, an exact masked Savitzky–Golay), ``baseline`` (rolling quantile,
morphological opening, SNIP), ``noise`` (robust MADs), ``consensus``
(several traces made one), ``interpolate`` (brackets and per-row
interpolation), and ``peaks``, peak detection as interchangeable stages
composed by a ``PeakPicker`` (``docs/plans/peak-detection.md``).
Torch-native; imports ``core.stats`` for ppm/Da conversions and peak forms,
and ``core.optim`` for fits.
"""

from __future__ import annotations

from constellation.core.signal.baseline import opening, rolling_quantile, snip
from constellation.core.signal.consensus import collapse
from constellation.core.signal.interpolate import Bracket, bracket, interp_rows
from constellation.core.signal.noise import diff_mad, residual_mad, rolling_mad
from constellation.core.signal.smooth import gaussian, moving_average, savitzky_golay
from constellation.core.signal.alignment import (
    GlobalPathOptions,
    GreedyOptions,
    MonotoneAlignment,
    RefineOptions,
    fit_monotone_alignment,
    quantile_band,
    theil_sen,
)
from constellation.core.signal.decimate import decimate_trace
from constellation.core.signal.matching import (
    bounds_within_tolerance,
    nearest_within_tolerance,
    tolerance_window,
)
from constellation.core.signal.modal_path import PathPosterior, greedy_walk, monotone_path
from constellation.core.signal.phase import PhaseFold, WindowPhase

__all__ = [
    "decimate_trace",
    "nearest_within_tolerance",
    "bounds_within_tolerance",
    "tolerance_window",
    "greedy_walk",
    "monotone_path",
    "PathPosterior",
    "fit_monotone_alignment",
    "MonotoneAlignment",
    "GreedyOptions",
    "GlobalPathOptions",
    "RefineOptions",
    "quantile_band",
    "theil_sen",
    "PhaseFold",
    "WindowPhase",
    "moving_average",
    "gaussian",
    "savitzky_golay",
    "rolling_quantile",
    "opening",
    "snip",
    "diff_mad",
    "residual_mad",
    "rolling_mad",
    "collapse",
    "bracket",
    "Bracket",
    "interp_rows",
]
