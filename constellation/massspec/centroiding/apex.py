"""The simplest peak list worth comparing with: the profile's local maxima.

Each maximum with a prominence of at least a quarter of its height, placed by a
parabola through the logs of the three points around it, with its area the
profile summed between the valleys either side. It reproduces the positions of
the ``apex`` row of the Stellar peak-picking benchmark, and the estimator behind
that benchmark's axis. It is there as a cheap baseline, not for precision: it
reads only three points, so a fit over all of them beats it (1.6-2x in MS2 at
30-300 ions on the benchmark's synthetic peaks), though its Gaussian assumption
costs under 1 mTh on Student-t peaks. Areas here differ from the benchmark's,
which divide a +-1 FWHM sum by a fitted peak shape's share of it; this needs no
peak shape, so it works at any scan rate.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import peak_prominences

#: A maximum at least this prominent, as a fraction of its height.
MIN_PROMINENCE = 0.25
#: The smallest area kept, in ions, for a scan whose injection time is known.
MIN_AREA_IONS = 0.5


def _log_parabola(y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Sub-grid offset of the maxima at ``k``, in grid points; linear where a
    neighbour is zero."""
    left, centre, right = y[k - 1], y[k], y[k + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.log(left) - 2 * np.log(centre) + np.log(right)
        by_log = 0.5 * (np.log(left) - np.log(right)) / denominator
        by_line = 0.5 * (left - right) / (left - 2 * centre + right)
    offset = np.where((left > 0) & (right > 0) & (denominator < 0), by_log, by_line)
    return np.clip(np.nan_to_num(offset), -1.0, 1.0)


def pick_apex(
    mz: np.ndarray,
    intensity: np.ndarray,
    iit_ms: float | None,
    *,
    min_prominence: float = MIN_PROMINENCE,
    min_area_ions: float = MIN_AREA_IONS,
) -> tuple[np.ndarray, np.ndarray]:
    """One profile scan's peaks: their m/z, and their areas in the profile's
    intensity units (a sum of points, Thermo's convention). With no injection
    time the smallest-area rule has no ion scale to apply to, and is skipped."""
    y = np.asarray(intensity, dtype=np.float64)
    x = np.asarray(mz, dtype=np.float64)
    if y.size < 3:
        return np.empty(0), np.empty(0)
    k = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    if k.size == 0:
        return np.empty(0), np.empty(0)
    k = k[peak_prominences(y, k)[0] >= min_prominence * y[k]]
    if k.size == 0:
        return np.empty(0), np.empty(0)
    step = (x[k + 1] - x[k - 1]) / 2.0
    positions = x[k] + _log_parabola(y, k) * step
    # Each peak owns the points up to the lowest point between it and the next.
    edges = np.empty(k.size + 1, dtype=np.int64)
    edges[0], edges[-1] = 0, y.size
    for i in range(k.size - 1):
        edges[i + 1] = k[i] + int(np.argmin(y[k[i] : k[i + 1] + 1]))
    running = np.concatenate(([0.0], np.cumsum(y)))
    areas = running[edges[1:]] - running[edges[:-1]]
    if iit_ms is not None and iit_ms > 0 and min_area_ions > 0:
        keep = areas * iit_ms / 1000.0 >= min_area_ions
        positions, areas = positions[keep], areas[keep]
    return positions, areas
