"""Refiners: a peak's apex position finer than its sample, or a fitted shape.

A refiner takes the detection trace and the peaks' apexes and extents
(compacted positions) and gives each apex's position in x, and optionally
fitted shape parameters:

- :class:`NoRefine`: the apex sample's x.
- :class:`Parabolic`: the vertex of the parabola through the apex and its
  two neighbours as points (x, y) (of their logarithms with ``log``, which
  is exact for a Gaussian on any spacing), within half a sample of the
  apex.
- :class:`ShapeFit`: a peak shape fitted to each peak's extent at once
  (``core.signal.peak_fit``), its apex position and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from constellation.core.signal.peaks._types import Trace
from constellation.core.signal.peaks.stages import Stage, register

__all__ = ["NoRefine", "Parabolic", "ShapeFit", "x_at"]


def x_at(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """x at fractional compacted positions, linear between samples; at a
    whole position, that sample's x, whatever its neighbour holds. Until
    2026-11-11 a whole position read its right neighbour through ``0 ·
    x``, so a row's last sample, beside NaN padding, had x NaN."""
    n = x.shape[-1]
    lo = torch.floor(position).clamp(0, max(n - 1, 0)).to(torch.int64)
    hi = (lo + 1).clamp(max=max(n - 1, 0))
    frac = (position - lo.to(position.dtype)).clamp(0, 1)
    xl, xh = torch.gather(x, -1, lo), torch.gather(x, -1, hi)
    return torch.where((hi > lo) & (frac > 0), xl + frac * (xh - xl), xl)


@register
@dataclass(frozen=True)
class NoRefine(Stage):
    """The apex sample's x."""

    kind: ClassVar[str] = "refine"
    name: ClassVar[str] = "none"

    def __call__(self, trace: Trace, apex, left, right, valid):
        return torch.gather(trace.x, -1, apex.clamp(min=0)), None


@register
@dataclass(frozen=True)
class Parabolic(Stage):
    """The vertex of the parabola through the apex and its neighbours, as
    (x, y) points (of log y with ``log``: exact for a Gaussian on any
    spacing), kept within half a sample of the apex (between the midpoints
    to its neighbours); the apex's x where the three are not concave.
    Until 2026-11-11 the parabola assumed equal spacing, so a Gaussian
    centred at 1.3 sampled at x = 0, 1, 4 refined to 0.68."""

    log: bool = False
    kind: ClassVar[str] = "refine"
    name: ClassVar[str] = "parabolic"
    version: ClassVar[int] = 2

    def __call__(self, trace: Trace, apex, left, right, valid):
        y, x = trace.y, trace.x
        n = y.shape[-1]
        p = apex.clamp(min=0)
        before, after = (p - 1).clamp(min=0), (p + 1).clamp(max=n - 1)
        y0, y1, y2 = (torch.gather(y, -1, i) for i in (before, p, after))
        x0, x1, x2 = (torch.gather(x, -1, i) for i in (before, p, after))
        inner = (p > 0) & (p < (trace.count - 1).unsqueeze(-1))
        if self.log:
            positive = (y0 > 0) & (y1 > 0) & (y2 > 0)
            inner &= positive
            y0, y1, y2 = (torch.log(torch.where(positive, v, torch.ones_like(v))) for v in (y0, y1, y2))
        # About the apex, d0 < 0 < d2 and the chords' slopes s0, s2: the
        # curvature is (s0 − s2)/(d0 − d2), concave where s0 > s2, and the
        # vertex is d0/2 − s0·(d0 − d2)/(2(s0 − s2)) from x1.
        d0, d2 = x0 - x1, x2 - x1
        inner &= (d0 < 0) & (d2 > 0)
        one = torch.ones_like(d0)
        s0 = (y0 - y1) / torch.where(inner, d0, -one)
        s2 = (y2 - y1) / torch.where(inner, d2, one)
        concave = inner & (s0 > s2)
        t = 0.5 * d0 - s0 * (d0 - d2) / (2.0 * torch.where(concave, s0 - s2, one))
        vertex = torch.minimum(torch.maximum(x1 + t, 0.5 * (x0 + x1)), 0.5 * (x1 + x2))
        return torch.where(concave, vertex, x1), None


@register
@dataclass(frozen=True)
class ShapeFit(Stage):
    """A peak shape (``form``: a name in ``core.stats.peaks.PEAK_FORMS``)
    fitted to each peak's extent of the trace above its baseline; the apex
    is the fitted shape's, its parameters the peaks' ``components``."""

    form: str = "gaussian"
    max_iter: int = 50
    kind: ClassVar[str] = "refine"
    name: ClassVar[str] = "shape_fit"
    version: ClassVar[int] = 2  # 2026-11-11: scale-free damping and stopping in the fit

    def __call__(self, trace: Trace, apex, left, right, valid):
        from constellation.core.signal.peak_fit import fit_peaks

        fit = fit_peaks(trace.x, trace.y - trace.baseline_or_zero(), trace.valid, left, right, valid,
                        form=self.form, max_iter=self.max_iter)  # fmt: skip
        return fit.apex, fit.params
