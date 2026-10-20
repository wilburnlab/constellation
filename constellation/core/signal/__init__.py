"""Heuristic 1D-trace operations and sorted-axis matching.

Residents: the tolerance-gated matching kernel (``matching``) shared by MS
chromatogram extraction, peak annotation, and search, and M4 trace
decimation for drawing (``decimate``). Baseline
correction / smoothing / prominence peak-picking / calibration land here
as they are ported. Torch-native; imports ``core.stats`` for ppm/Da
conversions (and, later, ``Parametric`` peak fits).
"""

from __future__ import annotations

from constellation.core.signal.decimate import decimate_trace
from constellation.core.signal.matching import (
    bounds_within_tolerance,
    nearest_within_tolerance,
    tolerance_window,
)

__all__ = [
    "decimate_trace",
    "nearest_within_tolerance",
    "bounds_within_tolerance",
    "tolerance_window",
]
