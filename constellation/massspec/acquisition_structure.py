"""What kind of acquisition a run is, read off its MS2 isolation windows.

Whether a run is DIA or DDA is a property of the data, not something a
user should have to assert: a DIA run cycles through a small, fixed set
of isolation windows and visits each one hundreds or thousands of times,
while a DDA run isolates a different precursor nearly every scan. The
number of MS2 scans per distinct window separates the two by orders of
magnitude (a 100-minute gas-phase-fraction run here: 80,213 scans over
49 windows; a DDA run on the same instrument: 179,533 scans over 134,180
windows).

The inference is used to label a ``.dia`` file's
``rawFileStructure.*`` metadata, which EncyclopeDIA reads to decide how
to search it, and to pick a default search mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

AcquisitionType = Literal["DIA", "DDA"]

#: A run is DIA when its MS2 scans revisit each isolation window at
#: least this many times on average. Real DIA is in the hundreds to
#: thousands; real DDA is between 1 and 2.
MIN_VISITS_PER_DIA_WINDOW = 10.0


@dataclass(frozen=True)
class AcquisitionStructure:
    """Summary of a run's scan structure."""

    acquisition_type: AcquisitionType
    #: DIA only: neighbouring windows overlap by about half their width,
    #: i.e. alternate cycles are offset (overlapped / staggered windows,
    #: which the search demultiplexes).
    is_staggered: bool
    n_ms1: int
    n_ms2: int
    #: Distinct ``(isolation_lower, isolation_upper)`` pairs among MS2 scans.
    n_windows: int


def isolation_windows(scan_metadata: pa.Table) -> np.ndarray:
    """Distinct MS2 isolation windows, as an ``(n, 2)`` float32 array sorted by lower bound.

    Bounds are compared at float32 precision: that is what a ``.dia``
    stores, and it keeps a window that the instrument reports with
    last-digit jitter from splitting in two.
    """
    ms2 = scan_metadata.filter(pc.equal(scan_metadata.column("level"), 2))
    lower = ms2.column("isolation_lower").to_numpy(zero_copy_only=False)
    upper = ms2.column("isolation_upper").to_numpy(zero_copy_only=False)
    keep = ~(np.isnan(lower) | np.isnan(upper))
    pairs = np.stack([lower[keep], upper[keep]], axis=1).astype(np.float32)
    return np.unique(pairs, axis=0) if pairs.size else pairs.reshape(0, 2)


def infer_acquisition_structure(scan_metadata: pa.Table) -> AcquisitionStructure:
    """Classify a run from a ``SCAN_METADATA_TABLE``-shaped table.

    Needs the ``level``, ``isolation_lower`` and ``isolation_upper``
    columns. Raises ``ValueError`` for a run with no MS2 scans or no
    usable isolation bounds, since neither label would mean anything.
    """
    level = scan_metadata.column("level")
    n_ms1 = int(pc.sum(pc.equal(level, 1)).as_py() or 0)
    n_ms2 = int(pc.sum(pc.equal(level, 2)).as_py() or 0)
    if n_ms2 == 0:
        raise ValueError("run has no MS2 scans; it is neither DIA nor DDA")
    windows = isolation_windows(scan_metadata)
    n_windows = int(windows.shape[0])
    if n_windows == 0:
        raise ValueError("no MS2 scan carries isolation bounds; cannot classify the run")

    is_dia = n_ms2 / n_windows >= MIN_VISITS_PER_DIA_WINDOW
    staggered = False
    if is_dia and n_windows > 1:
        # Overlap between each window and the next one up, as a fraction
        # of the narrower of the two. Contiguous schemes sit near 0 (a
        # small margin at most); staggered schemes sit near 0.5.
        lo, hi = windows[:, 0].astype(np.float64), windows[:, 1].astype(np.float64)
        overlap = np.clip(hi[:-1] - lo[1:], 0.0, None)
        narrower = np.minimum(hi[:-1] - lo[:-1], hi[1:] - lo[1:])
        frac = np.divide(overlap, narrower, out=np.zeros_like(overlap), where=narrower > 0)
        staggered = bool(np.median(frac) > 0.25)
    return AcquisitionStructure(
        acquisition_type="DIA" if is_dia else "DDA",
        is_staggered=staggered,
        n_ms1=n_ms1,
        n_ms2=n_ms2,
        n_windows=n_windows,
    )


__all__ = [
    "MIN_VISITS_PER_DIA_WINDOW",
    "AcquisitionStructure",
    "AcquisitionType",
    "infer_acquisition_structure",
    "isolation_windows",
]
