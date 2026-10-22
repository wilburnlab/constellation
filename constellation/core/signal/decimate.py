"""Reduce a long 1D trace to what can be drawn at a given width.

A trace drawn as a polyline into ``n`` pixel columns looks the same when
each column keeps only four of its points: the first, the last, the
lowest and the highest (M4; Jugel et al., *M4: A Visualization-Oriented
Time Series Data Aggregation*, PVLDB 7(10), 2014). The first and last
carry the line into and out of the column, and the extremes give its
vertical extent, so nothing visible is lost while the point count drops
to at most ``4 n``.

The function returns *indices* rather than values so a caller keeps every
column of its table (scan numbers, m/z, labels) for the points it keeps:
``table.take(decimate_trace(x, y, n_bins=width).numpy())``.
"""

from __future__ import annotations

import numpy as np
import torch


def _float64(values) -> torch.Tensor:
    """``values`` as a float64 tensor. A read-only array (what Arrow hands
    out) is copied, since torch will not wrap one."""
    if isinstance(values, torch.Tensor):
        return values.to(torch.float64)
    array = np.asarray(values, dtype=np.float64)
    return torch.from_numpy(array if array.flags.writeable else array.copy())


def decimate_trace(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_bins: int,
    x_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Indices of the points of ``(x, y)`` that draw the same line in ``n_bins`` columns.

    ``x`` must be sorted ascending. Both may be tensors or anything numpy
    takes as an array (a writable numpy array is used without a copy). With
    ``x_range``, the columns span that range and the points outside it are
    dropped, except the nearest one on each side, so a line still runs to
    the edges. When the range holds no more than ``4 * n_bins`` points they
    are all kept. Returns sorted int64 indices into ``x``.
    """
    x, y = _float64(x), _float64(y)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D and of the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if x.numel() > 1 and bool((torch.diff(x) < 0).any()):
        raise ValueError("x must be sorted ascending")
    if x.numel() == 0:
        return torch.empty(0, dtype=torch.int64)

    lo, hi = (float(x[0]), float(x[-1])) if x_range is None else map(float, x_range)
    if not hi > lo:
        hi = lo + 1.0
    bounds = torch.tensor([lo, hi], dtype=torch.float64)
    start = int(torch.searchsorted(x, bounds[:1], side="left"))
    stop = int(torch.searchsorted(x, bounds[1:], side="right"))
    edges = torch.tensor([i for i in (start - 1, stop) if 0 <= i < x.numel()], dtype=torch.int64)

    inside = torch.arange(start, stop, dtype=torch.int64)
    if inside.numel() <= 4 * n_bins:
        return torch.unique(torch.cat([inside, edges]))

    bins = ((x[inside] - lo) / (hi - lo) * n_bins).to(torch.int64).clamp(max=n_bins - 1)
    # x is sorted, so each bin is a contiguous run: its first and last points
    # sit at the run boundaries.
    boundary = torch.nonzero(torch.diff(bins)).flatten() + 1
    firsts = torch.cat([boundary.new_zeros(1), boundary])
    lasts = torch.cat([boundary - 1, boundary.new_full((1,), inside.numel() - 1)])
    # Order by y, then (stably) by bin: within each bin the run's ends are
    # its minimum and maximum.
    order = torch.argsort(y[inside], stable=True)
    order = order[torch.argsort(bins[order], stable=True)]
    ordered = bins[order]
    run_start = torch.cat([ordered.new_zeros(1), torch.nonzero(torch.diff(ordered)).flatten() + 1])
    run_end = torch.cat([run_start[1:] - 1, run_start.new_full((1,), order.numel() - 1)])
    keep = torch.cat(
        [inside[firsts], inside[lasts], inside[order[run_start]], inside[order[run_end]], edges]
    )
    return torch.unique(keep)


__all__ = ["decimate_trace"]
