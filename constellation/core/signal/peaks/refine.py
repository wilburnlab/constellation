"""Refiners: a peak's apex position finer than its sample, or a fitted shape.

A refiner takes the detection trace and the peaks' apexes and extents
(compacted positions) and gives each apex's position in x, and optionally
fitted shape parameters:

- :class:`NoRefine`: the apex sample's x.
- :class:`Parabolic`: the vertex of the parabola through the apex and its
  two neighbours (of their logarithms with ``log``, which is exact for a
  Gaussian), within half a sample of the apex, mapped to x.
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
    """x at fractional compacted positions, linear between samples."""
    n = x.shape[-1]
    lo = torch.floor(position).clamp(0, max(n - 1, 0)).to(torch.int64)
    hi = (lo + 1).clamp(max=max(n - 1, 0))
    frac = (position - lo.to(position.dtype)).clamp(0, 1)
    xl, xh = torch.gather(x, -1, lo), torch.gather(x, -1, hi)
    return torch.where(hi > lo, xl + frac * (xh - xl), xl)


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
    """The parabola's vertex through the apex and its neighbours."""

    log: bool = False
    kind: ClassVar[str] = "refine"
    name: ClassVar[str] = "parabolic"

    def __call__(self, trace: Trace, apex, left, right, valid):
        y = trace.y
        n = y.shape[-1]
        p = apex.clamp(min=0)
        a = torch.gather(y, -1, (p - 1).clamp(min=0))
        b = torch.gather(y, -1, p)
        c = torch.gather(y, -1, (p + 1).clamp(max=n - 1))
        inner = (p > 0) & (p < (trace.count - 1).unsqueeze(-1))
        if self.log:
            positive = (a > 0) & (b > 0) & (c > 0)
            inner &= positive
            a, b, c = (torch.log(torch.where(positive, v, torch.ones_like(v))) for v in (a, b, c))
        denom = a - 2 * b + c
        shift = torch.where(inner & (denom < 0), 0.5 * (a - c) / torch.where(denom < 0, denom, -torch.ones_like(denom)),
                            torch.zeros_like(b)).clamp(-0.5, 0.5)  # fmt: skip
        return x_at(trace.x, p.to(torch.float64) + shift), None


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

    def __call__(self, trace: Trace, apex, left, right, valid):
        from constellation.core.signal.peak_fit import fit_peaks

        fit = fit_peaks(trace.x, trace.y - trace.baseline_or_zero(), trace.valid, left, right, valid,
                        form=self.form, max_iter=self.max_iter)  # fmt: skip
        return fit.apex, fit.params
