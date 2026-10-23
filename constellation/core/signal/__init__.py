"""Heuristic 1D-trace operations, sorted-axis matching and monotone alignment.

Residents: the tolerance-gated matching kernel (``matching``) shared by MS
chromatogram extraction, peak annotation, and search; M4 trace
decimation for drawing (``decimate``); monotone paths through the mode
of a gridded density (``modal_path``: EncyclopeDIA's greedy walk and a
globally optimal dynamic-programming path); and monotone alignment of
one axis onto another through their joint density (``alignment``), the
retention-time alignment's generic core. Baseline
correction / smoothing / prominence peak-picking / calibration land here
as they are ported. Torch-native; imports ``core.stats`` for ppm/Da
conversions (and, later, ``Parametric`` peak fits).
"""

from __future__ import annotations

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
]
