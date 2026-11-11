"""Batched interpolation along an irregular, masked axis.

A trace is sampled at ``x [..., N]`` (ascending among its valid entries;
retention times of a query's scans, say), with ``valid [..., N]`` saying
which samples exist. :func:`bracket` finds, for each new position, the
valid samples on either side and the linear weight between them;
:func:`interp_rows` interpolates values with it. Rows are independent, and
padding may sit anywhere in a row.

Used to put one trace on another's axis (an MS1 trace at MS2 times) and by
the peak layer to map sample indices to x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from constellation.core.signal._rows import compact

__all__ = ["Bracket", "bracket", "interp_rows"]


@dataclass(frozen=True)
class Bracket:
    """For each new position ``[..., M]``: ``left`` the last valid sample at
    or before it and ``right`` the first at or after it (original indices
    into ``N``, ``-1`` for none), ``weight`` the share of ``right`` in a
    linear interpolation (0 where they coincide or one is missing), and
    ``inside`` whether both exist."""

    left: torch.Tensor  # int64
    right: torch.Tensor  # int64
    weight: torch.Tensor  # float64
    inside: torch.Tensor  # bool


def bracket(x: torch.Tensor, x_new: torch.Tensor, *, valid: torch.Tensor | None = None) -> Bracket:
    """The valid samples of ``x`` (``[..., N]``) around each of ``x_new``
    (``[..., M]``, the same leading shape) (:class:`Bracket`). A NaN
    position, or a row with no valid sample, has neither. ``x`` must be
    ascending among each row's valid entries."""
    x = x.to(torch.float64)
    x_new = x_new.to(torch.float64)
    if valid is None:
        valid = torch.isfinite(x)
    valid = valid.bool() & torch.isfinite(x)
    c = compact(valid)
    xc = c.gather(x, fill=float("inf"))
    n = c.count.unsqueeze(-1)
    probe = torch.where(torch.isnan(x_new), torch.zeros_like(x_new), x_new).contiguous()
    first_ge = torch.searchsorted(xc.contiguous(), probe, right=False)
    last_le = torch.searchsorted(xc.contiguous(), probe, right=True) - 1
    known = ~torch.isnan(x_new)
    right_c = torch.where(known & (first_ge < n), first_ge, torch.full_like(first_ge, -1))
    left_c = torch.where(known & (last_le >= 0) & (last_le < n), last_le, torch.full_like(last_le, -1))
    xl = torch.gather(xc, -1, left_c.clamp(min=0))
    xr = torch.gather(xc, -1, right_c.clamp(min=0))
    inside = (left_c >= 0) & (right_c >= 0)
    span = xr - xl
    spread = inside & (span > 0)
    weight = torch.where(spread, (x_new - xl) / torch.where(spread, span, torch.ones_like(span)),
                         torch.zeros_like(x_new))  # fmt: skip
    return Bracket(left=c.position(left_c), right=c.position(right_c), weight=weight, inside=inside)


def interp_rows(
    x: torch.Tensor,
    y: torch.Tensor,
    x_new: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    extrapolate: Literal["nan", "clamp"] = "nan",
    max_gap: float | None = None,
) -> torch.Tensor:
    """``y`` (``[..., N]``, sampled at ``x``) linearly interpolated at
    ``x_new`` (``[..., M]``), row by row. Outside a row's valid samples the
    value is NaN, or with ``extrapolate="clamp"`` the nearest sample's.
    With ``max_gap``, a position between two samples further apart than it
    is NaN. A sample with ``y`` NaN is not a sample."""
    if valid is None:
        valid = torch.ones_like(x, dtype=torch.bool)
    valid = valid.bool() & ~torch.isnan(y)
    b = bracket(x, x_new, valid=valid)
    yv = y.to(torch.float64)
    yl = torch.gather(yv, -1, b.left.clamp(min=0))
    yr = torch.gather(yv, -1, b.right.clamp(min=0))
    out = torch.where(b.inside, yl + b.weight * (yr - yl), torch.full_like(x_new, float("nan"), dtype=torch.float64))
    if max_gap is not None:
        xv = x.to(torch.float64)
        gap = torch.gather(xv, -1, b.right.clamp(min=0)) - torch.gather(xv, -1, b.left.clamp(min=0))
        out = torch.where(b.inside & (gap > max_gap), torch.full_like(out, float("nan")), out)
    if extrapolate == "clamp":
        only_left = (b.left >= 0) & (b.right < 0)
        only_right = (b.right >= 0) & (b.left < 0)
        out = torch.where(only_left, yl, out)
        out = torch.where(only_right, yr, out)
    elif extrapolate != "nan":
        raise ValueError(f"extrapolate must be 'nan' or 'clamp'; got {extrapolate!r}")
    return out.to(y.dtype) if y.is_floating_point() else out
