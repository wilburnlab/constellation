"""Several aligned traces made into one, for detection on their joint shape.

A peak that several traces share (the fragments of one precursor eluting
together; the isotopes of one ion; the channels of a diode array) is found
more reliably on one trace made from all of them than on any one of them:
one interfering trace should not make a peak, and one missing should not
remove it. :func:`collapse` takes ``y [..., K, N]`` (K traces of N samples)
and gives ``[..., N]``:

- ``sum`` / ``mean``: the total or average of the valid traces at each
  sample (``weights`` per trace).
- ``median`` / ``quantile``: an order statistic across the traces.
- ``kth_largest``: the k-th largest value at each sample, 0 where fewer than
  k traces are valid: a peak in it needs k traces to rise together, so up to
  k-1 interfering traces cannot make one (EncyclopeDIA's consensus takes
  about the third largest of its normalized fragment traces).

``scale [..., K]`` multiplies each trace first: ``1 / predicted share``
makes every fragment trace an estimate of the precursor's own amount.
"""

from __future__ import annotations

from typing import Literal

import torch

__all__ = ["collapse"]

Method = Literal["sum", "mean", "median", "quantile", "kth_largest"]


def collapse(
    y: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    method: Method = "sum",
    k: int = 3,
    q: float = 0.5,
    scale: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The traces ``y [..., K, N]`` as one ``[..., N]`` (module docstring),
    and how many traces were valid at each sample. A sample no trace is
    valid at is NaN, except under ``kth_largest`` (0 where fewer than ``k``)."""
    ok = torch.isfinite(y) if valid is None else (valid.bool() & torch.isfinite(y))
    v = y.to(torch.float64)
    if scale is not None:
        v = v * scale.to(torch.float64).unsqueeze(-1)
        ok = ok & torch.isfinite(scale).unsqueeze(-1)
    count = ok.sum(dim=-2)
    none = count == 0
    nan = torch.full(count.shape, float("nan"), dtype=torch.float64)
    if method in ("sum", "mean"):
        w = torch.ones_like(v) if weights is None else weights.to(torch.float64).unsqueeze(-1).expand_as(v)
        w = torch.where(ok, w, torch.zeros_like(w))
        total = (torch.where(ok, v, torch.zeros_like(v)) * w).sum(dim=-2)
        if method == "mean":
            denom = w.sum(dim=-2)
            total = torch.where(denom > 0, total / torch.where(denom > 0, denom, torch.ones_like(denom)), nan)
        return torch.where(none, nan, total), count
    if method in ("median", "quantile"):
        masked = torch.where(ok, v, torch.full_like(v, float("nan")))
        out = torch.nanquantile(masked, 0.5 if method == "median" else q, dim=-2)
        return torch.where(none, nan, out), count
    if method == "kth_largest":
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        masked = torch.where(ok, v, torch.full_like(v, float("-inf")))
        kk = min(k, v.shape[-2])
        top = torch.topk(masked, kk, dim=-2).values[..., kk - 1, :]
        enough = count >= k
        return torch.where(enough, top, torch.zeros_like(top)), count
    raise ValueError(f"method must be one of sum, mean, median, quantile, kth_largest; not {method!r}")
