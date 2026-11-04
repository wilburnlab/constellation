"""Matching one window group against its channels, two ways.

A window group is the scans of a block that share an isolation window (or
every MS1 scan of the block), and so share one vector of channels: the
m/z values the group's queries expect signal at. A cell is one channel in
one scan. Either kernel gives every cell's matches for a rule, row for
row the same:

- :func:`match_by_channel` (channel-driven): each channel is searched
  for in each scan's sorted points with :mod:`core.signal`'s kernels, one
  binary search per cell, whether anything is there or not. It is the
  definition of a match: the other kernel is held to it.
- :func:`match_by_point` (point-driven): the channels are sorted by m/z
  once, and every point is searched for in them, collecting each
  (point, channel) pair within tolerance as a candidate before the rule
  decides. A channel with no point near it costs nothing. This is the
  fragment-index direction (as in MSFragger and Sage), with the index
  built per window group.

Which is cheaper depends on the counts (measured, ``docs/plans/sweep.md``).
Channel-driven work follows the cells; point-driven work follows the
points plus the candidates. On centroids (a peak list: about 750 peaks a
scan, about half a peak within 0.35 Da of a channel) point-driven is as
fast or faster from a quarter of a channel per point up, and 2–11× faster
at proteome DIA scale. On profile points (about 19,500 a scan) a profile
peak puts some ten points within tolerance of each channel: point-driven
collects all ten as candidates and ``nearest`` keeps one, so
channel-driven is 2.4× faster there; ``all`` wants every candidate, and
the two tie.

``match_by_point`` takes every candidate within a slightly widened window
and then applies the exact predicate of the rule it stands in for: for
``nearest``, the choice
:func:`~constellation.core.signal.nearest_within_tolerance` makes between
the points either side of the channel (the lower one on a tie), then its
``|q − a| <= half`` test; for ``all`` and ``sum``, the
``q − half <= a <= q + half`` that :func:`~constellation.core.signal.bounds_within_tolerance`
selects. A point outside the widened window is farther from the channel
than the window, so it can be neither the match nor the reason another
point is not; that is why the two agree exactly.
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

__all__ = ["GroupMatches", "match_by_channel", "match_by_point", "match_group"]

ToleranceUnit = Literal["ppm", "Da"]

#: How much wider than the tolerance the point-driven lookup reaches before
#: the exact test; far above rounding, far below any tolerance.
_WIDEN = 1e-6


@dataclass(frozen=True)
class GroupMatches:
    """Matches of one group, sorted by (row, channel, point)."""

    row: np.ndarray  # position in the group's rows
    channel: np.ndarray  # position in the group's channel vector
    point: np.ndarray  # the point's index within its scan; -1 when aggregated
    n_points: np.ndarray  # points the match stands for
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
    row: np.ndarray, channel: np.ndarray, point: np.ndarray, mz: np.ndarray, intensity: np.ndarray
) -> GroupMatches:
    """One match per cell from its (cell, point) pairs, which arrive sorted:
    summed intensity, intensity-weighted m/z, summed in m/z order."""
    if row.size == 0:
        return GroupMatches.empty()
    starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (channel[1:] != channel[:-1])])
    total = np.add.reduceat(intensity, starts)
    weighted = np.add.reduceat(mz * intensity, starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.where(total > 0, weighted / total, np.add.reduceat(mz, starts) / np.diff(np.r_[starts, row.size]))
    return GroupMatches(
        row=row[starts],
        channel=channel[starts],
        point=np.full(starts.size, -1, np.int64),
        n_points=np.diff(np.r_[starts, row.size]).astype(np.int64),
        mz=centre,
        intensity=total,
    )


def match_by_channel(
    points_mz: np.ndarray,
    points_intensity: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    channel_mz: np.ndarray,
    rule: MatchRule,
    tolerance: float,
    unit: ToleranceUnit,
    *,
    max_elements: int = 1 << 21,
) -> GroupMatches:
    """Channel-driven: every channel searched for in every scan of the
    group, in padded chunks of at most about ``max_elements`` cells and
    points. The scans are copied into a padded matrix first, so even a
    few channels cost a pass over the group's points."""
    if rows.size == 0 or channel_mz.size == 0:
        return GroupMatches.empty()
    counts = offsets[rows + 1] - offsets[rows]
    width = max(int(counts.max()), 1)
    n_channels = channel_mz.size
    chunk = max(1, min(n_channels, max_elements // 2))
    parts: list[GroupMatches] = []
    for i0 in range(0, n_channels, chunk):
        q_np = channel_mz[i0 : i0 + chunk]
        m = q_np.size
        per = max(1, max_elements // (width + m))
        for r0 in range(0, rows.size, per):
            sub = rows[r0 : r0 + per]
            b = sub.size
            axis = np.full((b, width), np.inf)
            inten = np.zeros((b, width))
            flat, row_of, at = _scan_points(offsets, sub)
            axis[row_of, at] = points_mz[flat]
            inten[row_of, at] = points_intensity[flat]
            axis_t = torch.from_numpy(axis)
            q = torch.from_numpy(q_np).unsqueeze(0).expand(b, m)
            if rule.name == "nearest":
                best, within, _ = nearest_within_tolerance(axis_t, q, tolerance, unit)
                bi, mi = np.nonzero(within.numpy())
                pk = best.numpy()[bi, mi]
                parts.append(
                    GroupMatches(
                        row=bi.astype(np.int64) + r0,
                        channel=mi.astype(np.int64) + i0,
                        point=pk.astype(np.int64),
                        n_points=np.ones(bi.size, np.int64),
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
            ch = np.repeat(mi.astype(np.int64), k)
            pk = ragged_arange(lo[bi, mi], k)
            keep = np.isfinite(axis[row, pk])
            row, ch, pk = row[keep], ch[keep], pk[keep]
            mz, it = axis[row, pk], inten[row, pk]
            if rule.name == "all":
                parts.append(GroupMatches(row + r0, ch + i0, pk, np.ones(row.size, np.int64), mz, it))
            else:
                agg = _aggregate(row, ch, pk, mz, it)
                parts.append(
                    GroupMatches(agg.row + r0, agg.channel + i0, agg.point, agg.n_points, agg.mz,
                                 agg.intensity)  # fmt: skip
                )
    return _sorted(_concat(parts))


def _widened(a: np.ndarray, tolerance: float, unit: ToleranceUnit) -> tuple[np.ndarray, np.ndarray]:
    """The channel m/z range that could put each point ``a`` within
    tolerance, a little wider than exact."""
    if unit == "ppm":
        t = tolerance * 1e-6 * (1.0 + _WIDEN) + 1e-12
        return a / (1.0 + t), a / max(1.0 - t, 1e-12)
    if unit == "Da":
        t = tolerance * (1.0 + _WIDEN) + 1e-9
        return a - t, a + t
    raise ValueError(f"unit must be 'ppm' or 'Da', got {unit!r}")


def match_by_point(
    points_mz: np.ndarray,
    points_intensity: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    channel_mz: np.ndarray,
    rule: MatchRule,
    tolerance: float,
    unit: ToleranceUnit,
) -> GroupMatches:
    """Point-driven: every point of the group looked up in the channels
    sorted by m/z, giving (point, channel) candidates, then held to the
    rule's exact test."""
    if rows.size == 0 or channel_mz.size == 0:
        return GroupMatches.empty()
    order = np.argsort(channel_mz, kind="stable")
    sorted_mz = channel_mz[order]
    flat, row_of, at = _scan_points(offsets, rows)
    lo_w, hi_w = _widened(points_mz[flat], tolerance, unit)
    i0 = np.searchsorted(sorted_mz, lo_w, side="left")
    i1 = np.searchsorted(sorted_mz, hi_w, side="right")
    n = i1 - i0
    flat_c = np.repeat(flat, n)
    row = np.repeat(row_of, n)
    pk = np.repeat(at, n)
    ch = order[ragged_arange(i0, n)]
    by = np.lexsort((pk, ch, row))
    flat_c, row, pk, ch = flat_c[by], row[by], pk[by], ch[by]
    q = channel_mz[ch]
    a = points_mz[flat_c]
    half = tolerance_window(torch.from_numpy(q), tolerance, unit).numpy()
    if rule.name in ("all", "sum"):
        keep = (a >= q - half) & (a <= q + half)
        row, ch, pk, flat_c = row[keep], ch[keep], pk[keep], flat_c[keep]
        mz, it = points_mz[flat_c], points_intensity[flat_c]
        if rule.name == "all":
            return GroupMatches(row, ch, pk, np.ones(row.size, np.int64), mz, it)
        return _aggregate(row, ch, pk, mz, it)

    # nearest: per cell, the candidates either side of the channel.
    if row.size == 0:
        return GroupMatches.empty()
    starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (ch[1:] != ch[:-1])])
    ends = np.r_[starts[1:], row.size]
    below = a < q
    # Candidates of a cell are in point order, so those below the channel come first.
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
        channel=ch[best],
        point=pk[best],
        n_points=np.ones(best.size, np.int64),
        mz=points_mz[flat_c[best]],
        intensity=points_intensity[flat_c[best]],
    )


def match_group(
    points_mz: np.ndarray,
    points_intensity: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    channel_mz: np.ndarray,
    rule: MatchRule,
    tolerance: float,
    unit: ToleranceUnit,
    *,
    kernel: Literal["auto", "channel", "point"] = "auto",
    crossover: float = 0.25,
    max_elements: int = 1 << 21,
) -> GroupMatches:
    """Match a group with the cheaper kernel: point-driven once the
    channels exceed ``crossover`` times a scan's mean points, else
    channel-driven.

    Channels per point is the right measure on centroids. For ``nearest``
    on profile points the deciding count is the points within tolerance of
    a channel, which this does not see: profile scans sit below the
    crossover at purified-sample scale, but at proteome scale (over a
    quarter of a channel per point) it would choose point-driven, 3.9×
    slower in a synthetic test. To be settled with the profile rule."""
    if kernel == "auto":
        counts = offsets[rows + 1] - offsets[rows]
        mean_points = float(counts.mean()) if counts.size else 0.0
        kernel = "point" if channel_mz.size > crossover * max(mean_points, 1.0) else "channel"
    if kernel == "point":
        return match_by_point(points_mz, points_intensity, offsets, rows, channel_mz, rule, tolerance, unit)
    if kernel == "channel":
        return match_by_channel(
            points_mz, points_intensity, offsets, rows, channel_mz, rule, tolerance, unit,
            max_elements=max_elements,
        )  # fmt: skip
    raise ValueError(f"kernel must be 'auto', 'channel' or 'point', got {kernel!r}")


def _concat(parts: list[GroupMatches]) -> GroupMatches:
    if not parts:
        return GroupMatches.empty()
    if len(parts) == 1:
        return parts[0]
    return GroupMatches(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                          ("row", "channel", "point", "n_points", "mz", "intensity")))  # fmt: skip


def _sorted(m: GroupMatches) -> GroupMatches:
    if m.n == 0:
        return m
    by = np.lexsort((m.point, m.channel, m.row))
    return GroupMatches(m.row[by], m.channel[by], m.point[by], m.n_points[by], m.mz[by], m.intensity[by])
