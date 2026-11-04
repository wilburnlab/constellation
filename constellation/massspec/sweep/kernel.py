"""Matching one group of scans against one vector of ions, two ways.

A group is scans of a block that share their queries (one isolation
window, or every MS1 scan), so they share one vector of ion m/z. Either
kernel gives every (scan, ion) cell's matches for a rule:

- :func:`match_cells` (ions → peaks) searches each ion in each scan's
  peaks with :mod:`core.signal`'s kernels. Its cost follows the number of
  cells, and every cell costs the same whether anything is there or not.
  The better choice when a scan holds more peaks than there are ions.
- :func:`match_peaks` (peaks → ions) looks every peak up in the ions
  sorted by m/z, so a cell with no peak near it costs nothing. The better
  choice when ions far outnumber a scan's peaks.

Both give the same matches, row for row. ``match_peaks`` takes every
(peak, ion) pair within a slightly widened window and then applies the
exact predicate of the rule it stands in for: for ``nearest``, the choice
:func:`~constellation.core.signal.nearest_within_tolerance` makes between
the peaks either side of the ion (the lower one on a tie), then its
``|q − a| <= half`` test; for ``all`` and ``sum``, the
``q − half <= a <= q + half`` that :func:`~constellation.core.signal.bounds_within_tolerance`
selects. A peak outside the widened window is farther from the ion than
the window, so it can be neither the match nor the reason another peak is
not; that is why the two agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from constellation.core.signal import (
    bounds_within_tolerance,
    nearest_within_tolerance,
    tolerance_window,
)
from constellation.massspec.sweep.rules import MatchRule
from constellation.massspec.sweep.schedule import ragged_arange

__all__ = ["GroupMatches", "match_cells", "match_peaks", "match_group"]

ToleranceUnit = Literal["ppm", "Da"]

#: How much wider than the tolerance the peak-major lookup reaches before
#: the exact test; far above rounding, far below any tolerance.
_WIDEN = 1e-6


@dataclass(frozen=True)
class GroupMatches:
    """Matches of one group, sorted by (row, ion, peak)."""

    row: np.ndarray  # position in the group's rows
    ion: np.ndarray  # position in the group's ion vector
    peak: np.ndarray  # the peak's index within its scan; -1 when aggregated
    n_peaks: np.ndarray  # peaks the match stands for
    mz: np.ndarray
    intensity: np.ndarray

    @classmethod
    def empty(cls) -> GroupMatches:
        i, f = np.empty(0, np.int64), np.empty(0, np.float64)
        return cls(i, i, i, i, f, f)

    @property
    def n(self) -> int:
        return int(self.row.size)


def _scan_points(offsets: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat point indices of ``rows``' scans, each point's row, and its index in its scan."""
    counts = offsets[rows + 1] - offsets[rows]
    flat = ragged_arange(offsets[rows], counts)
    row_of = np.repeat(np.arange(rows.size, dtype=np.int64), counts)
    return flat, row_of, flat - np.repeat(offsets[rows], counts)


def _aggregate(
    row: np.ndarray, ion: np.ndarray, peak: np.ndarray, mz: np.ndarray, intensity: np.ndarray
) -> GroupMatches:
    """One match per (row, ion) from its pairs, which arrive sorted: summed
    intensity, intensity-weighted m/z, summed in m/z order."""
    if row.size == 0:
        return GroupMatches.empty()
    starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (ion[1:] != ion[:-1])])
    total = np.add.reduceat(intensity, starts)
    weighted = np.add.reduceat(mz * intensity, starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.where(total > 0, weighted / total, np.add.reduceat(mz, starts) / np.diff(np.r_[starts, row.size]))
    return GroupMatches(
        row=row[starts],
        ion=ion[starts],
        peak=np.full(starts.size, -1, np.int64),
        n_peaks=np.diff(np.r_[starts, row.size]).astype(np.int64),
        mz=centre,
        intensity=total,
    )


def match_cells(
    peaks_mz: np.ndarray,
    peaks_intensity: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    ion_mz: np.ndarray,
    rule: MatchRule,
    tolerance: float,
    unit: ToleranceUnit,
    *,
    max_elements: int = 1 << 21,
) -> GroupMatches:
    """Ions → peaks: every ion searched in every scan of the group, in
    padded chunks of at most about ``max_elements`` cells and peaks."""
    if rows.size == 0 or ion_mz.size == 0:
        return GroupMatches.empty()
    counts = offsets[rows + 1] - offsets[rows]
    width = max(int(counts.max()), 1)
    n_ions = ion_mz.size
    ion_chunk = max(1, min(n_ions, max_elements // 2))
    parts: list[GroupMatches] = []
    for i0 in range(0, n_ions, ion_chunk):
        q_np = ion_mz[i0 : i0 + ion_chunk]
        m = q_np.size
        per = max(1, max_elements // (width + m))
        for r0 in range(0, rows.size, per):
            sub = rows[r0 : r0 + per]
            b = sub.size
            axis = np.full((b, width), np.inf)
            inten = np.zeros((b, width))
            flat, row_of, at = _scan_points(offsets, sub)
            axis[row_of, at] = peaks_mz[flat]
            inten[row_of, at] = peaks_intensity[flat]
            axis_t = torch.from_numpy(axis)
            q = torch.from_numpy(q_np).unsqueeze(0).expand(b, m)
            if rule.name == "nearest":
                best, within, _ = nearest_within_tolerance(axis_t, q, tolerance, unit)
                bi, mi = np.nonzero(within.numpy())
                pk = best.numpy()[bi, mi]
                parts.append(
                    GroupMatches(
                        row=bi.astype(np.int64) + r0,
                        ion=mi.astype(np.int64) + i0,
                        peak=pk.astype(np.int64),
                        n_peaks=np.ones(bi.size, np.int64),
                        mz=axis[bi, pk],
                        intensity=inten[bi, pk],
                    )
                )
                continue
            lo, hi = (x.numpy() for x in bounds_within_tolerance(axis_t, q, tolerance, unit))
            n = hi - lo
            bi, mi = np.nonzero(n > 0)
            k = n[bi, mi]
            row = np.repeat(bi.astype(np.int64), k)
            ion = np.repeat(mi.astype(np.int64), k)
            pk = ragged_arange(lo[bi, mi], k)
            keep = np.isfinite(axis[row, pk])
            row, ion, pk = row[keep], ion[keep], pk[keep]
            mz, it = axis[row, pk], inten[row, pk]
            if rule.name == "all":
                parts.append(GroupMatches(row + r0, ion + i0, pk, np.ones(row.size, np.int64), mz, it))
            else:
                agg = _aggregate(row, ion, pk, mz, it)
                parts.append(
                    GroupMatches(agg.row + r0, agg.ion + i0, agg.peak, agg.n_peaks, agg.mz, agg.intensity)
                )
    return _sorted(_concat(parts))


def _widened(a: np.ndarray, tolerance: float, unit: ToleranceUnit) -> tuple[np.ndarray, np.ndarray]:
    """The ion m/z range that could put each peak ``a`` within tolerance, a
    little wider than exact."""
    if unit == "ppm":
        t = tolerance * 1e-6 * (1.0 + _WIDEN) + 1e-12
        return a / (1.0 + t), a / max(1.0 - t, 1e-12)
    if unit == "Da":
        t = tolerance * (1.0 + _WIDEN) + 1e-9
        return a - t, a + t
    raise ValueError(f"unit must be 'ppm' or 'Da', got {unit!r}")


def match_peaks(
    peaks_mz: np.ndarray,
    peaks_intensity: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    ion_mz: np.ndarray,
    rule: MatchRule,
    tolerance: float,
    unit: ToleranceUnit,
) -> GroupMatches:
    """Peaks → ions: every peak of the group looked up in the ions sorted by
    m/z, then held to the rule's exact test."""
    if rows.size == 0 or ion_mz.size == 0:
        return GroupMatches.empty()
    order = np.argsort(ion_mz, kind="stable")
    sorted_mz = ion_mz[order]
    flat, row_of, at = _scan_points(offsets, rows)
    lo_w, hi_w = _widened(peaks_mz[flat], tolerance, unit)
    i0 = np.searchsorted(sorted_mz, lo_w, side="left")
    i1 = np.searchsorted(sorted_mz, hi_w, side="right")
    n = i1 - i0
    point = np.repeat(flat, n)
    row = np.repeat(row_of, n)
    pk = np.repeat(at, n)
    ion = order[ragged_arange(i0, n)]
    by = np.lexsort((pk, ion, row))
    point, row, pk, ion = point[by], row[by], pk[by], ion[by]
    q = ion_mz[ion]
    a = peaks_mz[point]
    half = tolerance_window(torch.from_numpy(q), tolerance, unit).numpy()
    if rule.name in ("all", "sum"):
        keep = (a >= q - half) & (a <= q + half)
        row, ion, pk, point = row[keep], ion[keep], pk[keep], point[keep]
        mz, it = peaks_mz[point], peaks_intensity[point]
        if rule.name == "all":
            return GroupMatches(row, ion, pk, np.ones(row.size, np.int64), mz, it)
        return _aggregate(row, ion, pk, mz, it)

    # nearest: per (row, ion) cell, the candidates either side of the ion.
    if row.size == 0:
        return GroupMatches.empty()
    starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (ion[1:] != ion[:-1])])
    ends = np.r_[starts[1:], row.size]
    below = a < q
    # Candidates of a cell are in peak order, so those below the ion come first.
    n_below = np.add.reduceat(below.astype(np.int64), starts)
    has_left = n_below > 0
    has_right = n_below < (ends - starts)
    left = starts + n_below - 1
    right = starts + n_below
    left_c = np.where(has_left, left, starts)
    right_c = np.where(has_right, right, ends - 1)
    q_cell = q[starts]
    d_left = np.abs(q_cell - a[left_c])
    d_right = np.abs(q_cell - a[right_c])
    use_left = has_left & (~has_right | (d_left <= d_right))
    best = np.where(use_left, left_c, right_c)
    within = np.abs(q_cell - a[best]) <= half[starts]
    best = best[within]
    return GroupMatches(
        row=row[best],
        ion=ion[best],
        peak=pk[best],
        n_peaks=np.ones(best.size, np.int64),
        mz=peaks_mz[point[best]],
        intensity=peaks_intensity[point[best]],
    )


def match_group(
    peaks_mz: np.ndarray,
    peaks_intensity: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    ion_mz: np.ndarray,
    rule: MatchRule,
    tolerance: float,
    unit: ToleranceUnit,
    *,
    kernel: Literal["auto", "cells", "peaks"] = "auto",
    crossover: float = 0.25,
    max_elements: int = 1 << 21,
) -> GroupMatches:
    """Match a group with the cheaper kernel: peaks → ions once the ions
    exceed ``crossover`` times a scan's mean peaks, else ions → peaks."""
    if kernel == "auto":
        counts = offsets[rows + 1] - offsets[rows]
        mean_peaks = float(counts.mean()) if counts.size else 0.0
        kernel = "peaks" if ion_mz.size > crossover * max(mean_peaks, 1.0) else "cells"
    if kernel == "peaks":
        return match_peaks(peaks_mz, peaks_intensity, offsets, rows, ion_mz, rule, tolerance, unit)
    if kernel == "cells":
        return match_cells(
            peaks_mz, peaks_intensity, offsets, rows, ion_mz, rule, tolerance, unit,
            max_elements=max_elements,
        )  # fmt: skip
    raise ValueError(f"kernel must be 'auto', 'cells' or 'peaks', got {kernel!r}")


def _concat(parts: list[GroupMatches]) -> GroupMatches:
    if not parts:
        return GroupMatches.empty()
    if len(parts) == 1:
        return parts[0]
    return GroupMatches(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                          ("row", "ion", "peak", "n_peaks", "mz", "intensity")))  # fmt: skip


def _sorted(m: GroupMatches) -> GroupMatches:
    if m.n == 0:
        return m
    by = np.lexsort((m.peak, m.ion, m.row))
    return GroupMatches(m.row[by], m.ion[by], m.peak[by], m.n_peaks[by], m.mz[by], m.intensity[by])
