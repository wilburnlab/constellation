"""Peak detection along a continuous axis: interchangeable stages and a
picker that composes them (``docs/plans/peak-detection.md``).

    from constellation.core.signal.peaks import PeakPicker, LocalMaxima, SavitzkyGolay
    picker = PeakPicker(detect=LocalMaxima(prominence=5.0), smooth=SavitzkyGolay(7, 2))
    peaks = picker(traces, x=times, valid=mask)        # Peaks [..., M]

A stage is named in its kind's registry (``DETECTORS``, ``SMOOTHERS``, ...),
built by ``make_stage(kind, name, **parameters)``, and recorded by its
``identity()``; a new method is a stage class registered with ``register``.
"""

from __future__ import annotations

from constellation.core.signal.peaks._types import Candidates, Peaks, Trace
from constellation.core.signal.peaks.boundaries import DetectorBounds, Fraction, Valley, WalkOut
from constellation.core.signal.peaks.detect import (
    LocalMaxima,
    MatchedFilter,
    RobustZ,
    local_maxima,
    prominences,
    select_by_distance,
    widths,
)
from constellation.core.signal.peaks.pipeline import PeakPicker
from constellation.core.signal.peaks.refine import NoRefine, Parabolic, ShapeFit
from constellation.core.signal.peaks.stages import (
    BASELINES,
    BOUNDARIES,
    DETECTORS,
    NOISE,
    REFINERS,
    REGISTRIES,
    SMOOTHERS,
    DiffMad,
    GaussianSmooth,
    MovingAverage,
    Opening,
    ResidualMad,
    RollingMad,
    RollingQuantile,
    SavitzkyGolay,
    Snip,
    Stage,
    make_stage,
    register,
    stage_from_identity,
)

__all__ = [
    "PeakPicker",
    "Peaks",
    "Trace",
    "Candidates",
    "Stage",
    "register",
    "make_stage",
    "stage_from_identity",
    "REGISTRIES",
    "SMOOTHERS",
    "BASELINES",
    "NOISE",
    "DETECTORS",
    "BOUNDARIES",
    "REFINERS",
    "MovingAverage",
    "GaussianSmooth",
    "SavitzkyGolay",
    "RollingQuantile",
    "Opening",
    "Snip",
    "DiffMad",
    "ResidualMad",
    "RollingMad",
    "LocalMaxima",
    "MatchedFilter",
    "RobustZ",
    "Valley",
    "DetectorBounds",
    "Fraction",
    "WalkOut",
    "NoRefine",
    "Parabolic",
    "ShapeFit",
    "local_maxima",
    "prominences",
    "widths",
    "select_by_distance",
]
