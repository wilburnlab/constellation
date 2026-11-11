"""A peak shape fitted to every peak of a batch at once.

Each peak's extent ``[left, right]`` of its row is gathered into a window
of its own (``[B·M, L]``, ``L`` the widest extent, padding masked), a form
of ``core.stats.peaks.PEAK_FORMS`` is started from the window's moments
inside its data-derived bounds, and all windows are fitted together by
``core.optim.batched_least_squares``. The fit is on x (time), not sample
index: a chromatogram's scans are not evenly spaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from constellation.core.optim.batched_lsq import batched_least_squares
from constellation.core.stats.peaks import PEAK_FORMS

__all__ = ["PeakFit", "fit_peaks"]


@dataclass(frozen=True)
class PeakFit:
    """``[B, M]`` fits: parameters ``[B, M, P]`` of ``form``, the fitted
    apex position and area, the residual cost, and whether each converged
    (NaN / False for padding)."""

    form: str
    params: torch.Tensor
    apex: torch.Tensor
    area: torch.Tensor
    cost: torch.Tensor
    converged: torch.Tensor
    covariance: torch.Tensor  # [B, M, P, P]


def fit_peaks(
    x: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    peak_valid: torch.Tensor,
    *,
    form: str = "gaussian",
    max_iter: int = 50,
) -> PeakFit:
    """Fit ``form`` to each peak's samples ``[left, right]`` of ``y [B, N]``
    on ``x [B, N]`` (module docstring)."""
    if form not in PEAK_FORMS:
        raise KeyError(f"unknown peak form {form!r}; available: {sorted(PEAK_FORMS)}")
    shape = PEAK_FORMS[form]
    b, n = y.shape
    m = left.shape[-1]
    p = len(shape.parameters)
    nan = torch.full((b, m), float("nan"), dtype=torch.float64)
    if m == 0 or not peak_valid.any():
        return PeakFit(form, torch.full((b, m, p), float("nan"), dtype=torch.float64), nan, nan, nan,
                       torch.zeros((b, m), dtype=torch.bool), torch.full((b, m, p, p), float("nan")))  # fmt: skip
    lo, hi = left.clamp(min=0), right.clamp(min=0)
    span = int((hi - lo).clamp(min=0).max()) + 1
    offsets = torch.arange(span)
    at = (lo.unsqueeze(-1) + offsets).clamp(max=n - 1)  # [B, M, L]
    inside = (lo.unsqueeze(-1) + offsets) <= hi.unsqueeze(-1)
    rows = torch.arange(b).view(b, 1, 1).expand_as(at)
    xw = x.to(torch.float64)[rows, at]
    yw = y.to(torch.float64)[rows, at]
    ok = inside & valid[rows, at] & torch.isfinite(yw) & peak_valid.unsqueeze(-1)
    xw, yw, ok = xw.reshape(b * m, span), yw.reshape(b * m, span), ok.reshape(b * m, span)
    enough = ok.sum(-1) >= p + 1
    xw = torch.where(ok, xw, torch.zeros_like(xw))
    yw = torch.where(ok, yw, torch.zeros_like(yw))
    theta0 = shape.initial(xw, yw, ok)
    lower, upper = shape.bounds(xw, yw, ok)
    theta0 = torch.minimum(torch.maximum(theta0, lower), upper)
    theta0 = torch.nan_to_num(theta0, nan=1.0)

    def residual(theta: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return shape.evaluate(xw[r], theta) - yw[r]

    fit = batched_least_squares(residual, theta0, mask=ok & enough.unsqueeze(-1), lower=lower, upper=upper,
                                max_iter=max_iter)  # fmt: skip
    keep = (peak_valid.reshape(-1) & enough).reshape(b, m)
    params = torch.where(keep.reshape(-1, 1), fit.params, torch.full_like(fit.params, float("nan"))).reshape(b, m, p)
    apex = torch.where(keep, shape.apex(fit.params).reshape(b, m), nan)
    area = torch.where(keep, shape.area(fit.params).reshape(b, m), nan)
    cost = torch.where(keep, fit.cost.reshape(b, m), nan)
    return PeakFit(form=form, params=params, apex=apex, area=area, cost=cost,
                   converged=keep & fit.converged.reshape(b, m),
                   covariance=fit.covariance.reshape(b, m, p, p))  # fmt: skip
