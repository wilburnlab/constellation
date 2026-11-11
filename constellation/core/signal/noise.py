"""Robust noise levels of traces, for signal-to-noise and outlier tests.

Each returns a standard deviation in the trace's units, estimated so that
peaks barely move it: scaled median absolute deviations (``1.4826`` makes a
MAD a Gaussian's sigma). A sparse trace whose MAD is 0 (most samples
equal) gets ``floor``.

- :func:`diff_mad`: from the first differences (``/√2``): blind to a
  slowly varying baseline, so it needs none (Donoho and Johnstone's
  finest-scale estimate in its simplest form).
- :func:`residual_mad`: from the residual to a baseline.
- :func:`rolling_mad`: a local noise level, the residual's MAD over a
  window, for a trace whose noise changes along it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["diff_mad", "residual_mad", "rolling_mad", "MAD_TO_SIGMA"]

#: A MAD of Gaussian samples times this is their standard deviation.
MAD_TO_SIGMA = 1.4826


def _ok(y: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    return torch.isfinite(y) if valid is None else (valid.bool() & torch.isfinite(y))


def _nanmad(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    med = torch.nanmedian(x, dim=dim, keepdim=True).values
    return torch.nanmedian((x - med).abs(), dim=dim).values


def diff_mad(y: torch.Tensor, *, valid: torch.Tensor | None = None, floor: float = 0.0) -> torch.Tensor:
    """``[...]``: each row's sigma from the MAD of its first differences
    between consecutive valid samples, ``/√2``; NaN for a row of under two."""
    from constellation.core.signal._rows import compact

    ok = _ok(y, valid)
    c = compact(ok)
    v = c.gather(y.to(torch.float64), fill=float("nan"))
    d = v[..., 1:] - v[..., :-1]
    sigma = MAD_TO_SIGMA * _nanmad(d) / math.sqrt(2.0) if d.shape[-1] else torch.full(y.shape[:-1], float("nan"))
    return torch.where(torch.isnan(sigma), sigma, sigma.clamp(min=floor))


def residual_mad(
    y: torch.Tensor, baseline: torch.Tensor, *, valid: torch.Tensor | None = None, floor: float = 0.0
) -> torch.Tensor:
    """``[...]``: each row's sigma from the MAD of ``y - baseline`` over its
    valid samples."""
    ok = _ok(y, valid) & torch.isfinite(baseline)
    r = torch.where(ok, (y - baseline).to(torch.float64), torch.full(y.shape, float("nan"), dtype=torch.float64))
    sigma = MAD_TO_SIGMA * _nanmad(r)
    return torch.where(torch.isnan(sigma), sigma, sigma.clamp(min=floor))


def rolling_mad(
    y: torch.Tensor,
    window: int,
    *,
    baseline: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    floor: float = 0.0,
) -> torch.Tensor:
    """``[..., N]``: a sigma at each sample, from the MAD of the residual to
    ``baseline`` (the rolling median when not given) over its window of
    ``window`` (odd) samples."""
    from constellation.core.signal.baseline import rolling_quantile

    ok = _ok(y, valid)
    if baseline is None:
        baseline = rolling_quantile(y, window, 0.5, valid=ok)
    r = torch.where(ok, (y - baseline).to(torch.float64), torch.full(y.shape, float("nan"), dtype=torch.float64))
    shape = tuple(y.shape)
    h = window // 2
    padded = F.pad(r.reshape(-1, shape[-1]), (h, h), value=float("nan"))
    windows = padded.unfold(-1, window, 1)
    sigma = (MAD_TO_SIGMA * _nanmad(windows)).reshape(shape)
    sigma = torch.where(torch.isnan(sigma), sigma, sigma.clamp(min=floor))
    return torch.where(ok, sigma, torch.full_like(sigma, float("nan")))
