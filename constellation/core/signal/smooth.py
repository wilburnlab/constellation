"""Batched, masked smoothing of traces along their last axis.

Every function takes values ``y [..., N]`` and an optional validity mask
``valid [..., N]`` (default: finite values) and smooths each row on its own,
by sample index. A missing sample is not a zero: each output is a weighted
average (or a local least-squares fit) over the valid samples of its window
alone, so padding and gaps never pull a value down. Outputs are NaN where a
sample is not valid. Windows are in samples; a caller with an irregular
axis converts a width in x to samples first (by the median spacing of the
run, never of a batch, so a row's result does not depend on its neighbours).

``savitzky_golay`` is the local polynomial regression Savitzky and Golay
(1964) tabulated, done exactly on whatever samples a window holds: in the
interior of a fully valid row it is ``scipy.signal.savgol_filter``; at an
edge or a gap the polynomial is fitted to the samples there, where a
tabulated filter would assume samples that are not.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["moving_average", "gaussian", "savitzky_golay", "normalized_convolution"]


def _rows(y: torch.Tensor, valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """``y`` and its weights as ``[B, 1, N]`` float64 (invalid values 0)."""
    shape = tuple(y.shape)
    ok = torch.isfinite(y) if valid is None else (valid.bool() & torch.isfinite(y))
    v = torch.where(ok, y, torch.zeros_like(y)).to(torch.float64).reshape(-1, 1, shape[-1])
    w = ok.to(torch.float64).reshape(-1, 1, shape[-1])
    return v, w, shape


def normalized_convolution(
    y: torch.Tensor, kernel: torch.Tensor, *, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """``Σ k·w·y / Σ k·w`` over each sample's window (``kernel`` odd-length,
    centred): a weighted average that ignores invalid samples. NaN where a
    sample is invalid or its window holds no weight."""
    v, w, shape = _rows(y, valid)
    k = kernel.to(torch.float64).reshape(1, 1, -1)
    h = k.shape[-1] // 2
    num = F.conv1d(v * w, k, padding=h)
    den = F.conv1d(w, k, padding=h)
    out = torch.where((den > 0) & (w > 0), num / torch.where(den > 0, den, torch.ones_like(den)),
                      torch.full_like(num, float("nan")))  # fmt: skip
    return out.reshape(shape).to(y.dtype if y.is_floating_point() else torch.float64)


def moving_average(y: torch.Tensor, window: int, *, valid: torch.Tensor | None = None) -> torch.Tensor:
    """The mean of the valid samples within ``window`` (odd) samples."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, not {window}")
    return normalized_convolution(y, torch.ones(window, dtype=torch.float64), valid=valid)


def gaussian(y: torch.Tensor, sigma: float, *, valid: torch.Tensor | None = None, truncate: float = 4.0) -> torch.Tensor:
    """A Gaussian-weighted average of the valid samples, ``sigma`` in
    samples, the kernel cut at ``truncate`` sigma."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, not {sigma}")
    h = max(int(math.ceil(truncate * sigma)), 1)
    u = torch.arange(-h, h + 1, dtype=torch.float64)
    return normalized_convolution(y, torch.exp(-0.5 * (u / sigma) ** 2), valid=valid)


def savitzky_golay(
    y: torch.Tensor,
    window: int,
    degree: int = 2,
    *,
    derivative: int = 0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """The value (or ``derivative``-th derivative, per sample) at each sample
    of the least-squares polynomial of ``degree`` through the valid samples
    of its ``window`` (odd) samples (module docstring). Where a window holds
    too few samples for the polynomial, the value is their mean (and a
    derivative NaN); NaN where the sample itself is invalid."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, not {window}")
    if not 0 <= degree < window:
        raise ValueError(f"degree must be in [0, window), not {degree}")
    if not 0 <= derivative <= degree:
        raise ValueError(f"derivative must be in [0, degree], not {derivative}")
    v, w, shape = _rows(y, valid)
    h = window // 2
    p = degree + 1
    u = torch.arange(-h, h + 1, dtype=torch.float64)
    # Moments of the weights, Σ w u^m (m < 2p - 1), and of the weighted
    # values, Σ w y u^j (j < p), for every window at once.
    powers_w = torch.stack([u**m for m in range(2 * p - 1)]).unsqueeze(1)
    powers_y = torch.stack([u**j for j in range(p)]).unsqueeze(1)
    moments = F.conv1d(w, powers_w, padding=h)  # [B, 2p-1, N]
    rhs = F.conv1d(v * w, powers_y, padding=h)  # [B, p, N]
    n_rows, _, n = moments.shape
    hankel = torch.stack([moments[:, i : i + p, :] for i in range(p)], dim=1)  # [B, p, p, N]
    a = hankel.permute(0, 3, 1, 2).reshape(-1, p, p)
    b = rhs.permute(0, 2, 1).reshape(-1, p, 1)
    count = moments[:, 0, :].reshape(-1)
    enough = count >= p
    eye = torch.eye(p, dtype=torch.float64)
    solution, info = torch.linalg.solve_ex(torch.where(enough[:, None, None], a, eye.expand_as(a)), b)
    solved = enough & (info == 0)
    coef = solution[:, derivative, 0] * math.factorial(derivative)
    mean = torch.where(count > 0, b[:, 0, 0] / torch.where(count > 0, count, torch.ones_like(count)),
                       torch.full_like(count, float("nan")))  # fmt: skip
    fallback = mean if derivative == 0 else torch.full_like(mean, float("nan"))
    out = torch.where(solved, coef, fallback).reshape(n_rows, n)
    out = torch.where(w[:, 0, :] > 0, out, torch.full_like(out, float("nan")))
    return out.reshape(shape).to(y.dtype if y.is_floating_point() else torch.float64)
