"""What every peak is measured by, whichever detector found it.

So that detectors can be compared, each peak gets the same measures (the
:class:`~._types.Peaks` fields): on the detection trace its height,
prominence and bases and its signal-to-noise; on the boundary trace its
half-height width, area, asymmetry and jaggedness. ``jaggedness`` is the
share of the boundary trace's steps inside the extent that reverse the
previous step's direction (TargetedMSQC / MetaClean's measure, without
their flatness allowance): 0 for a clean peak, near 1 for noise.
"""

from __future__ import annotations

import torch

from constellation.core.signal._rows import RangeTable
from constellation.core.signal.peaks._types import Trace
from constellation.core.signal.peaks.refine import x_at

__all__ = ["measure"]


def _cumulative_trapezoid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``C[..., i]``: the trapezoid area of ``y`` over ``x`` from sample 0 to i."""
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * (x[..., 1:] - x[..., :-1])
    seg = torch.nan_to_num(seg, nan=0.0)
    return torch.cat([torch.zeros_like(y[..., :1]), torch.cumsum(seg, dim=-1)], dim=-1)


def measure(det: Trace, bnd: Trace, apex: torch.Tensor, boundary_apex: torch.Tensor, left: torch.Tensor,
            right: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:  # fmt: skip
    """The measures of peaks ``[B, M]`` (compacted positions; module docstring)."""
    from constellation.core.signal.peaks.detect import prominences

    p = apex.clamp(min=0)
    q = boundary_apex.clamp(min=0)
    lo, hi = left.clamp(min=0), right.clamp(min=0)
    det_table = RangeTable.build(det.y, det.count)
    height = torch.gather(det.y, -1, p)
    prominence, left_base, right_base = prominences(det_table, det.y, p)
    snr = (height - torch.gather(det.baseline_or_zero(), -1, p)) / det.noise_at(p)

    # Half height on the boundary trace, above its baseline, inside the extent.
    yb = bnd.y - bnd.baseline_or_zero()
    table = RangeTable.build(yb, bnd.count)
    hb = torch.gather(yb, -1, q)
    half = 0.5 * hb
    at_most = torch.nextafter(half, torch.full_like(half, float("inf")))
    li = table.nearest_left(q, at_most, above=False, limit=lo)
    ri = table.nearest_right(q, at_most, above=False, limit=hi)
    li = torch.where(li >= 0, li, lo)
    ri = torch.where(ri >= 0, ri, hi)
    n = yb.shape[-1]
    yl, yl1 = torch.gather(yb, -1, li), torch.gather(yb, -1, (li + 1).clamp(max=n - 1))
    yr, yr1 = torch.gather(yb, -1, ri), torch.gather(yb, -1, (ri - 1).clamp(min=0))
    left_ip = li.to(torch.float64) + torch.where((yl < half) & (yl1 > yl), (half - yl) / (yl1 - yl), torch.zeros_like(yl))
    right_ip = ri.to(torch.float64) - torch.where((yr < half) & (yr1 > yr), (half - yr) / (yr1 - yr), torch.zeros_like(yr))
    x_apex = torch.gather(bnd.x, -1, q)
    x_left, x_right = x_at(bnd.x, left_ip), x_at(bnd.x, right_ip)
    width = x_right - x_left
    lead, tail = x_apex - x_left, x_right - x_apex
    asymmetry = torch.where(lead > 0, tail / torch.where(lead > 0, lead, torch.ones_like(lead)),
                            torch.full_like(lead, float("nan")))  # fmt: skip

    area_c = _cumulative_trapezoid(bnd.x, torch.where(bnd.valid, yb.clamp(min=0), torch.zeros_like(yb)))
    area = torch.gather(area_c, -1, hi) - torch.gather(area_c, -1, lo)

    step = yb[..., 1:] - yb[..., :-1]
    turn = (step[..., 1:] * step[..., :-1]) < 0  # [B, N-2]: a reversal at sample i+1
    turns = torch.cat([torch.zeros_like(yb[..., :2]), torch.cumsum(turn.to(torch.float64), dim=-1)], dim=-1)
    # Reversals at samples strictly inside the extent: lo+1 .. hi-1.
    inner = (hi - lo - 1).clamp(min=0)
    flips = torch.gather(turns, -1, hi) - torch.gather(turns, -1, (lo + 1).clamp(max=n - 1))
    jaggedness = torch.where(inner > 0, flips / inner.clamp(min=1).to(torch.float64), torch.zeros_like(flips))

    nan = torch.full_like(height, float("nan"))
    keep = valid
    out = {
        "height": torch.where(keep, height, nan),
        "prominence": torch.where(keep, prominence, nan),
        "left_base": torch.where(keep, left_base, -1),
        "right_base": torch.where(keep, right_base, -1),
        "snr": torch.where(keep, snr, nan),
        "width": torch.where(keep, width, nan),
        "area": torch.where(keep, area, nan),
        "asymmetry": torch.where(keep, asymmetry, nan),
        "jaggedness": torch.where(keep, jaggedness, nan),
    }
    return out
