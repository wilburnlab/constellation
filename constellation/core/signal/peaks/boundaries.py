"""Boundary rules: where a peak starts and ends.

A rule takes the boundary trace (the detection trace unless another was
given), the candidates and their apexes on it, and gives each peak's
``left`` / ``right`` (inclusive, compacted positions):

- :class:`DetectorBounds`: the detector's own extent (a run's ends; a
  template's reach); :class:`Valley` where it gave none.
- :class:`Valley`: the prominence's bases, the lowest points between the
  peak and higher ground on each side.
- :class:`Fraction`: where the trace falls to ``fraction`` of the apex's
  height above the baseline (the first sample at or below it).
- :class:`WalkOut`: EncyclopeDIA's ``TransitionRefiner.getIndexRange``:
  out from the apex until the trace has risen for more than ``rise``
  samples after falling below ``half`` of the apex (the boundary is where
  that rise began), or falls below ``stop`` of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from constellation.core.signal._rows import RangeTable
from constellation.core.signal.peaks._types import Candidates, Trace
from constellation.core.signal.peaks.stages import Stage, register

__all__ = ["DetectorBounds", "Valley", "Fraction", "WalkOut"]


def _bases(trace: Trace, apex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from constellation.core.signal.peaks.detect import prominences

    table = RangeTable.build(trace.y, trace.count)
    _, lb, rb = prominences(table, trace.y, apex)
    return lb, rb


@register
@dataclass(frozen=True)
class Valley(Stage):
    """The prominence's bases on the boundary trace."""

    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "valley"

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        return _bases(trace, apex)


@register
@dataclass(frozen=True)
class DetectorBounds(Stage):
    """The detector's own extent, the valleys where it gave none."""

    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "detector"

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        if cand.left is None or cand.right is None:
            return _bases(trace, apex)
        lb, rb = _bases(trace, apex)
        left = torch.where(cand.left >= 0, torch.minimum(cand.left, apex), lb)
        right = torch.where(cand.right >= 0, torch.maximum(cand.right, apex), rb)
        return left, right


@register
@dataclass(frozen=True)
class Fraction(Stage):
    """Where the trace first falls to ``fraction`` of the apex's height above
    the baseline, on each side (the row's ends if it never does)."""

    fraction: float = 0.05
    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "fraction"

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        table = RangeTable.build(trace.y, trace.count)
        p = apex.clamp(min=0)
        base = torch.gather(trace.baseline_or_zero(), -1, p)
        h = torch.gather(trace.y, -1, p)
        level = base + self.fraction * (h - base)
        at_most = torch.nextafter(level, torch.full_like(level, float("inf")))
        left = table.nearest_left(p, at_most, above=False)
        right = table.nearest_right(p, at_most, above=False)
        last = (trace.count - 1).unsqueeze(-1).expand_as(p)
        return torch.where(left >= 0, left, torch.zeros_like(left)), torch.where(right >= 0, right, last)


@register
@dataclass(frozen=True)
class WalkOut(Stage):
    """EncyclopeDIA's walk out from the apex (module docstring); ``rise``
    in samples, ``max_extent`` the most samples walked each way."""

    stop: float = 0.01
    half: float = 0.5
    rise: int = 2
    max_extent: int = 1 << 16
    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "walk_out"

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        y = torch.nan_to_num(trace.y - trace.baseline_or_zero(), nan=0.0)
        p = apex.clamp(min=0)
        h = torch.gather(y, -1, p)
        last = (trace.count - 1).unsqueeze(-1).expand_as(p)
        out = []
        for step in (-1, 1):
            edge = torch.zeros_like(p) if step < 0 else last
            boundary = p.clone()
            valley = p.clone()
            prev = h.clone()
            below_half = torch.zeros_like(p, dtype=torch.bool)
            rising = torch.zeros_like(p)
            active = cand.valid.clone() & (apex >= 0)
            for t in range(1, min(self.max_extent, trace.y.shape[-1]) + 1):
                if not active.any():
                    break
                at = p + step * t
                inside = (at >= 0) & (at <= last)
                ended = active & ~inside
                boundary = torch.where(ended, edge, boundary)
                active &= inside
                v = torch.gather(y, -1, at.clamp(0, y.shape[-1] - 1))
                low = active & (v < self.stop * h)
                boundary = torch.where(low, at, boundary)
                active &= ~low
                below_half |= active & (v < self.half * h)
                up = v > prev
                rising = torch.where(active & up, rising + 1, torch.zeros_like(rising))
                valley = torch.where(active & ~up, at, valley)
                climbed = active & below_half & (rising > self.rise)
                boundary = torch.where(climbed, valley, boundary)
                active &= ~climbed
                boundary = torch.where(active, at, boundary)
                prev = torch.where(active, v, prev)
            out.append(boundary)
        return out[0], out[1]
