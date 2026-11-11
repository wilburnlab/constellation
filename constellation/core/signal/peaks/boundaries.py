"""Boundary rules: where a peak starts and ends.

A rule takes the boundary trace (the detection trace unless another was
given), the candidates and their apexes on it, and gives each peak's
``left`` / ``right`` (inclusive, compacted positions):

- :class:`DetectorBounds`: the detector's own extent (a run's ends; a
  template's reach); :class:`Valley` where it gave none.
- :class:`Valley`: the prominence's bases, the lowest points between the
  peak and higher ground on each side.
- :class:`Fraction`: where the trace above its baseline falls to
  ``fraction`` of the apex's height above it (the first sample at or below
  it).
- :class:`WalkOut`: EncyclopeDIA's ``TransitionRefiner.getIndexRange``:
  out from the apex until the trace has risen for more than ``rise``
  samples after falling below ``half`` of the apex (the boundary is where
  that rise began), or falls below ``stop`` of it.

A boundary trace shares the detection trace's samples and may lack some of
them (its ``valid`` has holes). A sample it lacks is never an endpoint:
a search for a level or a minimum passes it, a walk stops before it, and
a detector's extent is drawn in to the nearest sample it has. Until
2026-11-11 a hole was a NaN the rules read as a value: a walk ended on it,
and a valley's or a fraction's search stopped at it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from constellation.core.signal._rows import RangeTable, valid_neighbours
from constellation.core.signal.peaks._types import Candidates, Trace
from constellation.core.signal.peaks.stages import Stage, register

__all__ = ["DetectorBounds", "Valley", "Fraction", "WalkOut"]


def _bases(trace: Trace, apex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from constellation.core.signal.peaks.detect import prominences

    table = RangeTable.build(trace.y, trace.count, trace.valid)
    _, lb, rb = prominences(table, trace.y, apex)
    return lb, rb


def _ends(trace: Trace, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's first and last valid sample, shaped as ``like [B, M]``."""
    before, after = valid_neighbours(trace.valid)
    first = after[..., :1]
    last = torch.gather(before, -1, (trace.count - 1).clamp(min=0).unsqueeze(-1))
    return first.expand_as(like), last.expand_as(like)


@register
@dataclass(frozen=True)
class Valley(Stage):
    """The prominence's bases on the boundary trace."""

    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "valley"
    version: ClassVar[int] = 2  # 2026-11-11: a boundary trace's holes

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        return _bases(trace, apex)


@register
@dataclass(frozen=True)
class DetectorBounds(Stage):
    """The detector's own extent, drawn in to the boundary trace's valid
    samples; the valleys where it gave none."""

    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "detector"
    version: ClassVar[int] = 2  # 2026-11-11: a boundary trace's holes

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        lb, rb = _bases(trace, apex)
        if cand.left is None or cand.right is None:
            return lb, rb
        before, after = valid_neighbours(trace.valid)
        n = trace.valid.shape[-1]
        inner_left = torch.gather(after, -1, cand.left.clamp(0, n - 1))
        inner_right = torch.gather(before, -1, cand.right.clamp(0, n - 1))
        left = torch.where(cand.left >= 0, torch.minimum(inner_left, apex), lb)
        right = torch.where(cand.right >= 0, torch.maximum(inner_right, apex), rb)
        return left, right


@register
@dataclass(frozen=True)
class Fraction(Stage):
    """Where the trace above its baseline first falls to ``fraction`` of the
    apex's height above it, on each side (the row's first or last valid
    sample if it never does). Compared sample by sample above each one's
    own baseline: until 2026-11-11 the raw trace was compared with a level
    built from the apex's baseline alone, so under a sloping baseline a
    Gaussian's bounds went from [42, 58] to [43, 100]."""

    fraction: float = 0.05
    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "fraction"
    version: ClassVar[int] = 2

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        above = trace.y - trace.baseline_or_zero()
        table = RangeTable.build(above, trace.count, trace.valid)
        p = apex.clamp(min=0)
        level = self.fraction * torch.gather(above, -1, p)
        at_most = torch.nextafter(level, torch.full_like(level, float("inf")))
        left = table.nearest_left(p, at_most, above=False)
        right = table.nearest_right(p, at_most, above=False)
        first, last = _ends(trace, p)
        return torch.where(left >= 0, left, first), torch.where(right >= 0, right, last)


@register
@dataclass(frozen=True)
class WalkOut(Stage):
    """EncyclopeDIA's walk out from the apex (module docstring); ``rise``
    in samples, ``max_extent`` the most samples walked each way. A walk
    stops before a sample the trace lacks, as at the row's end."""

    stop: float = 0.01
    half: float = 0.5
    rise: int = 2
    max_extent: int = 1 << 16
    kind: ClassVar[str] = "boundaries"
    name: ClassVar[str] = "walk_out"
    version: ClassVar[int] = 2  # 2026-11-11: a boundary trace's holes

    def __call__(self, trace: Trace, cand: Candidates, apex: torch.Tensor):
        y = torch.nan_to_num(trace.y - trace.baseline_or_zero(), nan=0.0)
        n = y.shape[-1]
        p = apex.clamp(min=0)
        h = torch.gather(y, -1, p)
        last = (trace.count - 1).unsqueeze(-1).expand_as(p)
        out = []
        for step in (-1, 1):
            # The boundary is where the walk last stood until a rule moves
            # it, so a walk that cannot go on ends there.
            boundary = p.clone()
            valley = p.clone()
            prev = h.clone()
            below_half = torch.zeros_like(p, dtype=torch.bool)
            rising = torch.zeros_like(p)
            active = cand.valid.clone() & (apex >= 0)
            for t in range(1, min(self.max_extent, n) + 1):
                if not active.any():
                    break
                at = p + step * t
                there = torch.gather(trace.valid, -1, at.clamp(0, n - 1))
                active &= (at >= 0) & (at <= last) & there
                v = torch.gather(y, -1, at.clamp(0, n - 1))
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
