"""Batched, masked baselines of traces along their last axis.

A baseline is the level a trace returns to between its peaks: background
in a chromatogram, the drift under an HPLC trace, the roll under an NMR
spectrum. Each function takes ``y [..., N]`` and ``valid [..., N]`` and
returns ``[..., N]`` (NaN where a sample is invalid), by sample index.

- :func:`rolling_quantile`: the q-quantile of the valid samples in a
  window; the moving median at ``q=0.5`` (what EncyclopeDIA subtracts, over
  ``10·w`` scans).
- :func:`opening`: morphological opening, the largest curve under the
  trace that a flat element of the window's width can trace (a min filter
  then a max filter; Perez-Pueyo et al. 2010).
- :func:`snip`: statistics-sensitive non-linear iterative peak clipping
  (Ryan et al. 1988; Morháč 2009), on the log-log-square-root transform by
  default.

Asymmetric least squares (AsLS, arPLS) needs a batched banded solver that
torch does not have; it is not implemented here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["rolling_quantile", "opening", "snip"]


def _ok(y: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    return torch.isfinite(y) if valid is None else (valid.bool() & torch.isfinite(y))


def rolling_quantile(
    y: torch.Tensor,
    window: int,
    q: float = 0.5,
    *,
    valid: torch.Tensor | None = None,
    max_elements: int = 1 << 26,
) -> torch.Tensor:
    """The ``q``-quantile of each sample's window of ``window`` (odd)
    samples, over the valid ones (linear interpolation between order
    statistics, as ``torch.nanquantile``). Rows are taken in chunks so a
    window's unfolded values stay under ``max_elements``."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, not {window}")
    shape = tuple(y.shape)
    ok = _ok(y, valid)
    v = torch.where(ok, y, torch.full_like(y, float("nan"))).to(torch.float64).reshape(-1, shape[-1])
    h = window // 2
    padded = F.pad(v, (h, h), value=float("nan"))
    rows = max(1, max_elements // max(shape[-1] * window, 1))
    out = torch.empty_like(v)
    for a in range(0, v.shape[0], rows):
        windows = padded[a : a + rows].unfold(-1, window, 1)  # [r, N, window]
        out[a : a + rows] = torch.nanquantile(windows, q, dim=-1)
    out = out.reshape(shape)
    return torch.where(ok, out, torch.full_like(out, float("nan"))).to(y.dtype if y.is_floating_point() else torch.float64)


def opening(y: torch.Tensor, window: int, *, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Morphological opening with a flat element of ``window`` (odd)
    samples: each sample's minimum over its window, then the maximum of
    those over its window. Invalid samples are ignored by both."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, not {window}")
    shape = tuple(y.shape)
    ok = _ok(y, valid).reshape(-1, 1, shape[-1])
    v = y.to(torch.float64).reshape(-1, 1, shape[-1])
    h = window // 2
    eroded = -F.max_pool1d(torch.where(ok, -v, torch.full_like(v, float("-inf"))), window, 1, padding=h)
    eroded = torch.where(ok, eroded, torch.full_like(eroded, float("-inf")))
    dilated = F.max_pool1d(eroded, window, 1, padding=h)
    out = torch.where(ok, dilated, torch.full_like(dilated, float("nan"))).reshape(shape)
    return out.to(y.dtype if y.is_floating_point() else torch.float64)


def snip(
    y: torch.Tensor,
    iterations: int,
    *,
    valid: torch.Tensor | None = None,
    transform: bool = True,
    decreasing: bool = True,
) -> torch.Tensor:
    """SNIP baseline: ``iterations`` passes of ``b[i] = min(b[i], (b[i-k] +
    b[i+k]) / 2)``, ``k`` from ``iterations`` down to 1 (``decreasing``,
    Morháč's order) or up. With ``transform`` the clipping runs on
    ``log(log(sqrt(y + 1) + 1) + 1)`` and the result is transformed back,
    which keeps it from cutting into peaks of very different heights. A
    sample whose neighbours at ``±k`` are not both valid keeps its value at
    that pass. ``y`` must be non-negative with ``transform``."""
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, not {iterations}")
    shape = tuple(y.shape)
    ok = _ok(y, valid).reshape(-1, shape[-1])
    b = y.to(torch.float64).reshape(-1, shape[-1])
    if transform:
        b = torch.log(torch.log(torch.sqrt(b.clamp(min=0) + 1) + 1) + 1)
    b = torch.where(ok, b, torch.zeros_like(b))
    n = shape[-1]
    order = range(iterations, 0, -1) if decreasing else range(1, iterations + 1)
    for k in order:
        if k >= n:
            continue
        left = F.pad(b, (k, 0))[:, :n]
        right = F.pad(b, (0, k))[:, k:]
        left_ok = F.pad(ok, (k, 0), value=False)[:, :n]
        right_ok = F.pad(ok, (0, k), value=False)[:, k:]
        both = ok & left_ok & right_ok
        b = torch.where(both, torch.minimum(b, 0.5 * (left + right)), b)
    if transform:
        b = (torch.exp(torch.exp(b) - 1) - 1) ** 2 - 1
    out = torch.where(ok, b, torch.full_like(b, float("nan"))).reshape(shape)
    return out.to(y.dtype if y.is_floating_point() else torch.float64)
