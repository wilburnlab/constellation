"""What every peak is measured by, whichever detector found it.

So that detectors can be compared, each peak gets the same measures (the
:class:`~._types.Peaks` fields): on the detection trace its height,
prominence and bases and its signal-to-noise; on the boundary trace its
half-height width, area, asymmetry and jaggedness. ``jaggedness`` is the
share of the boundary trace's steps inside the extent that reverse the
previous step's direction (TargetedMSQC / MetaClean's measure, without
their flatness allowance): 0 for a clean peak, near 1 for noise.

The boundary trace may lack samples the detection trace has (holes in its
``valid``): its measures pass over them, a half-height crossing
interpolated between the samples on either side, the area joining them and
a step taken between the samples it has. A peak with no boundary apex
(``-1``: no valid boundary sample near its apex) has no boundary measures
(NaN). Until 2026-11-11 a hole was a NaN read as a value, and a boundary
trace with no valid sample still gave every peak a width and an area.
"""

from __future__ import annotations

import torch

from constellation.core.signal._rows import RangeTable, valid_neighbours
from constellation.core.signal.peaks._types import Trace

__all__ = ["measure"]


def _cumulative_trapezoid(x: torch.Tensor, y: torch.Tensor, prior: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``C[..., i]``: the trapezoid area of ``y`` over ``x`` across the valid
    samples up to i, each joined to the valid sample before it (``prior``,
    ``-1`` for none)."""
    has = valid & (prior >= 0)
    at = prior.clamp(min=0)
    seg = 0.5 * (torch.gather(y, -1, at) + y) * (x - torch.gather(x, -1, at))
    return torch.cumsum(torch.where(has, seg, torch.zeros_like(seg)), dim=-1)


def _exclusive_cumsum(t: torch.Tensor) -> torch.Tensor:
    """``E[..., i] = Σ_{j < i} t[..., j]``."""
    return torch.cumsum(t, dim=-1) - t


def measure(det: Trace, bnd: Trace, apex: torch.Tensor, boundary_apex: torch.Tensor, left: torch.Tensor,
            right: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:  # fmt: skip
    """The measures of peaks ``[B, M]`` (compacted positions; module docstring)."""
    from constellation.core.signal.peaks.detect import prominences

    p = apex.clamp(min=0)
    bounded = valid & (boundary_apex >= 0) & (left >= 0) & (right >= 0)
    q = torch.where(bounded, boundary_apex, torch.zeros_like(boundary_apex))
    lo, hi = torch.where(bounded, left, q), torch.where(bounded, right, q)
    det_table = RangeTable.build(det.y, det.count)
    height = torch.gather(det.y, -1, p)
    prominence, left_base, right_base = prominences(det_table, det.y, p)
    snr = (height - torch.gather(det.baseline_or_zero(), -1, p)) / det.noise_at(p)

    # Half height on the boundary trace, above its baseline, inside the extent.
    yb = bnd.y - bnd.baseline_or_zero()
    n = yb.shape[-1]
    before, after = valid_neighbours(bnd.valid)
    # The valid sample strictly before / after each position.
    prior = torch.cat([torch.full_like(before[..., :1], -1), before[..., :-1]], dim=-1)
    following = torch.cat([after[..., 1:], torch.full_like(after[..., :1], n)], dim=-1)
    table = RangeTable.build(yb, bnd.count, bnd.valid)
    hb = torch.gather(yb, -1, q)
    half = 0.5 * hb
    at_most = torch.nextafter(half, torch.full_like(half, float("inf")))
    li = table.nearest_left(q, at_most, above=False, limit=lo)
    ri = table.nearest_right(q, at_most, above=False, limit=hi)
    li = torch.where(li >= 0, li, lo)
    ri = torch.where(ri >= 0, ri, hi)
    # Each crossing lies between the sample found and the next valid one
    # toward the apex.
    li_in = torch.minimum(torch.gather(following, -1, li), q)
    ri_in = torch.maximum(torch.gather(prior, -1, ri), q)
    yl, yl1 = torch.gather(yb, -1, li), torch.gather(yb, -1, li_in)
    yr, yr1 = torch.gather(yb, -1, ri), torch.gather(yb, -1, ri_in)
    fl = torch.where((yl < half) & (yl1 > yl), (half - yl) / torch.where(yl1 > yl, yl1 - yl, torch.ones_like(yl)),
                     torch.zeros_like(yl))  # fmt: skip
    fr = torch.where((yr < half) & (yr1 > yr), (half - yr) / torch.where(yr1 > yr, yr1 - yr, torch.ones_like(yr)),
                     torch.zeros_like(yr))  # fmt: skip
    xl, xl1 = torch.gather(bnd.x, -1, li), torch.gather(bnd.x, -1, li_in)
    xr, xr1 = torch.gather(bnd.x, -1, ri), torch.gather(bnd.x, -1, ri_in)
    x_left = torch.where(fl > 0, xl + fl * (xl1 - xl), xl)
    x_right = torch.where(fr > 0, xr + fr * (xr1 - xr), xr)
    x_apex = torch.gather(bnd.x, -1, q)
    width = x_right - x_left
    lead, tail = x_apex - x_left, x_right - x_apex
    asymmetry = torch.where(lead > 0, tail / torch.where(lead > 0, lead, torch.ones_like(lead)),
                            torch.full_like(lead, float("nan")))  # fmt: skip

    area_c = _cumulative_trapezoid(bnd.x, torch.where(bnd.valid, yb.clamp(min=0), torch.zeros_like(yb)), prior,
                                   bnd.valid)  # fmt: skip
    area = torch.gather(area_c, -1, hi) - torch.gather(area_c, -1, lo)

    # A reversal at a valid sample: the steps from the valid sample before
    # it and to the valid sample after it change sign.
    has_both = bnd.valid & (prior >= 0) & (following < n)
    y_prior = torch.gather(yb, -1, prior.clamp(min=0))
    y_next = torch.gather(yb, -1, following.clamp(max=n - 1))
    turn = has_both & ((yb - y_prior) * (y_next - yb) < 0)
    turns = _exclusive_cumsum(turn.to(torch.float64))
    counted = _exclusive_cumsum(bnd.valid.to(torch.float64))
    # Reversals at the valid samples strictly inside the extent: lo+1 .. hi-1.
    start = torch.minimum(lo + 1, hi)
    flips = torch.gather(turns, -1, hi) - torch.gather(turns, -1, start)
    inner = torch.gather(counted, -1, hi) - torch.gather(counted, -1, start)
    jaggedness = torch.where(inner > 0, flips / inner.clamp(min=1), torch.zeros_like(flips))

    nan = torch.full_like(height, float("nan"))
    out = {
        "height": torch.where(valid, height, nan),
        "prominence": torch.where(valid, prominence, nan),
        "left_base": torch.where(valid, left_base, -1),
        "right_base": torch.where(valid, right_base, -1),
        "snr": torch.where(valid, snr, nan),
        "width": torch.where(bounded, width, nan),
        "area": torch.where(bounded, area, nan),
        "asymmetry": torch.where(bounded, asymmetry, nan),
        "jaggedness": torch.where(bounded, jaggedness, nan),
    }
    return out
