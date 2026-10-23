"""Kernel density estimates on a regular 2-D grid.

A point set is deposited onto a grid (nearest cell, or bilinearly over
the four surrounding cells) and the histogram is convolved with a
kernel stamp by FFT. Because deposition happens before smoothing, the
estimate costs O(n² log n) in the grid size whatever the number of
points, and with nearest-cell deposition it equals stamping a kernel at
every point exactly (the sum of shifted stamps is the histogram
convolved with the stamp).

Two kernels:

  ``gaussian``       anisotropic Gaussian, truncated at ``truncate`` σ.
  ``cosine_approx``  EncyclopeDIA's radial kernel: a cosine lobe of
                     support 2σ, ``A·cos(π r / 4σ)``, evaluated with the
                     jar's fast parabolic cosine. Used by the generic
                     port of its retention-time alignment.

Both stamps are normalized to sum 1, so a grid value is the kernel-
weighted mass of points near that cell.

The grid knows its own coordinates (:class:`GridAxis`), so a density
can be read back at data coordinates (:meth:`GridDensity.sample`,
bilinear or bicubic through ``F.grid_sample``, differentiable in the
query coordinates — the curve refinement in ``core.signal.alignment``
follows its gradient).

Edge handling is the caller's choice: pad the axes beyond the data so
kernels are not truncated, and/or pass ``renormalize=`` a support mask
to divide by the kernel mass that falls inside the support (Jones 1993),
which removes the low bias of a density near the boundary of the region
the data can occupy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

Kernel = Literal["gaussian", "cosine_approx"]
Deposit = Literal["linear", "nearest"]

__all__ = [
    "GridAxis",
    "GridDensity",
    "bin_2d",
    "kernel_stamp",
    "fft_convolve2d",
    "grid_kde_2d",
    "support_mask",
    "weighted_quantile",
    "cosine_approx",
]


@dataclass(frozen=True)
class GridAxis:
    """A regular axis: cell ``i`` sits at ``start + i·step``."""

    start: float
    step: float
    size: int

    @classmethod
    def spanning(cls, lo: float, hi: float, size: int) -> GridAxis:
        """``size`` cells with the first at ``lo`` and the last at ``hi``."""
        if size < 2:
            raise ValueError(f"a grid axis needs at least 2 cells, got {size}")
        if not hi > lo:
            raise ValueError(f"axis span must be positive, got [{lo}, {hi}]")
        return cls(float(lo), (float(hi) - float(lo)) / (size - 1), int(size))

    @property
    def stop(self) -> float:
        return self.start + (self.size - 1) * self.step

    def to_index(self, v: torch.Tensor) -> torch.Tensor:
        """Continuous cell coordinate of each value (0 at ``start``)."""
        return (v - self.start) / self.step

    def to_value(self, i: torch.Tensor) -> torch.Tensor:
        return self.start + i * self.step


@dataclass(frozen=True)
class GridDensity:
    """A density on the grid ``x_axis × y_axis`` (``values[i, j]``)."""

    values: torch.Tensor
    x_axis: GridAxis
    y_axis: GridAxis
    sigma_cells: tuple[float, float]
    kernel: str
    total_weight: float

    def sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
    ) -> torch.Tensor:
        """Density at data coordinates ``(x, y)``; zero outside the grid.

        Differentiable in ``x`` and ``y``. ``bicubic`` is C1 between
        cells, which a gradient-based fit needs.
        """
        x, y = torch.broadcast_tensors(x, y)
        shape = x.shape
        # grid_sample's last coordinate indexes the input's width (dim 1
        # of values, the y axis) and the first its height (the x axis).
        gx = 2.0 * self.y_axis.to_index(y.reshape(-1)) / (self.y_axis.size - 1) - 1.0
        gy = 2.0 * self.x_axis.to_index(x.reshape(-1)) / (self.x_axis.size - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).to(self.values.dtype).view(1, 1, -1, 2)
        out = F.grid_sample(
            self.values[None, None],
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )
        return out.view(shape)


def weighted_quantile(
    values: torch.Tensor,
    q: float | torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile(s) of ``values`` under non-negative ``weights``.

    Sorted values sit at the midpoints of their cumulative weight
    (``p_i = (W_i − w_i/2) / W``) and the quantile is interpolated
    linearly between them, clamped to the extreme values. With equal
    weights this is the Hazen definition (``p_i = (i + ½)/n``). Values
    of zero weight are left out.
    """
    v = values.reshape(-1).to(torch.float64)
    if v.numel() == 0:
        raise ValueError("weighted_quantile of an empty tensor")
    w = (
        torch.ones_like(v)
        if weights is None
        else weights.reshape(-1).to(torch.float64)
    )
    # A value without weight is not an observation; left in, it would
    # still anchor the interpolation.
    positive = w > 0
    v, w = v[positive], w[positive]
    if v.numel() == 0:
        raise ValueError("weights must have a positive sum")
    order = torch.argsort(v)
    v, w = v[order], w[order]
    total = w.sum()
    if not total > 0:
        raise ValueError("weights must have a positive sum")
    p = (torch.cumsum(w, 0) - 0.5 * w) / total
    qt = torch.as_tensor(q, dtype=torch.float64).reshape(-1)
    idx = torch.searchsorted(p, qt).clamp(1, v.numel() - 1) if v.numel() > 1 else None
    if idx is None:
        out = v.expand_as(qt).clone()
    else:
        p0, p1 = p[idx - 1], p[idx]
        v0, v1 = v[idx - 1], v[idx]
        t = ((qt - p0) / (p1 - p0).clamp_min(1e-300)).clamp(0.0, 1.0)
        out = v0 + t * (v1 - v0)
    return out.reshape(torch.as_tensor(q).shape)


def bin_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    x_axis: GridAxis,
    y_axis: GridAxis,
    deposit: Deposit = "linear",
) -> torch.Tensor:
    """Deposit weighted points onto the grid; float64 ``(nx, ny)``.

    ``nearest`` rounds each point to its cell (half up); ``linear``
    ("cloud in cell") splits its weight bilinearly over the four cells
    around it, which keeps the sub-cell position. Weight falling off the
    grid is dropped.
    """
    x = x.reshape(-1).to(torch.float64)
    y = y.reshape(-1).to(torch.float64)
    w = (
        torch.ones_like(x)
        if weights is None
        else weights.reshape(-1).to(torch.float64)
    )
    nx, ny = x_axis.size, y_axis.size
    fi = x_axis.to_index(x)
    fj = y_axis.to_index(y)
    out = torch.zeros(nx * ny, dtype=torch.float64)
    if deposit == "nearest":
        i = torch.floor(fi + 0.5).long()
        j = torch.floor(fj + 0.5).long()
        keep = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        out.index_add_(0, (i * ny + j)[keep], w[keep])
    elif deposit == "linear":
        i0 = torch.floor(fi).long()
        j0 = torch.floor(fj).long()
        tx = fi - i0
        ty = fj - j0
        for di, dj, wt in (
            (0, 0, (1 - tx) * (1 - ty)),
            (1, 0, tx * (1 - ty)),
            (0, 1, (1 - tx) * ty),
            (1, 1, tx * ty),
        ):
            i, j = i0 + di, j0 + dj
            keep = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            out.index_add_(0, (i * ny + j)[keep], (w * wt)[keep])
    else:
        raise ValueError(f"deposit must be 'linear' or 'nearest', got {deposit!r}")
    return out.view(nx, ny)


def cosine_approx(z: float) -> float:
    """EncyclopeDIA's fast cosine: a parabolic sine of ``z + π/2``.

    Kept to the jar's operation order so a stamp built from it carries
    the same values the jar's does (to float rounding of the caller).
    """
    s = z + 1.5707963267948966
    t = 1.2732395447351628 * s + (-0.4052847345693511 * s) * abs(s)
    return 0.225 * (t * abs(t) - t) + t


def kernel_stamp(
    sigma_cells: float | tuple[float, float],
    *,
    kernel: Kernel = "gaussian",
    truncate: float = 4.0,
) -> torch.Tensor:
    """An odd-sized kernel stamp normalized to sum 1 (float64).

    ``sigma_cells`` is the kernel σ in grid cells, one value or one per
    axis. The Gaussian extends to ``truncate`` σ; the cosine kernel's
    support is 2σ (radial, scaled per axis when anisotropic).
    """
    sx, sy = (
        (float(sigma_cells), float(sigma_cells))
        if not isinstance(sigma_cells, tuple)
        else (float(sigma_cells[0]), float(sigma_cells[1]))
    )
    if not (sx > 0 and sy > 0):
        raise ValueError(f"sigma_cells must be positive, got {sigma_cells}")
    if kernel == "gaussian":
        rx, ry = math.ceil(truncate * sx), math.ceil(truncate * sy)
        di = torch.arange(-rx, rx + 1, dtype=torch.float64)[:, None] / sx
        dj = torch.arange(-ry, ry + 1, dtype=torch.float64)[None, :] / sy
        stamp = torch.exp(-0.5 * (di * di + dj * dj))
    elif kernel == "cosine_approx":
        rx, ry = max(1, round(2.0 * sx)), max(1, round(2.0 * sy))
        di = torch.arange(-rx, rx + 1, dtype=torch.float64)[:, None] / sx
        dj = torch.arange(-ry, ry + 1, dtype=torch.float64)[None, :] / sy
        r = torch.sqrt(di * di + dj * dj)  # in units of σ
        s = -r * (math.pi / 4.0) + math.pi / 2.0
        t = (4.0 / math.pi) * s + (-4.0 / math.pi**2) * s * s.abs()
        stamp = torch.where(r <= 2.0, 0.225 * (t * t.abs() - t) + t, 0.0)
        stamp = stamp.clamp_min(0.0)
    else:
        raise ValueError(f"kernel must be 'gaussian' or 'cosine_approx', got {kernel!r}")
    return stamp / stamp.sum()


def fft_convolve2d(grid: torch.Tensor, stamp: torch.Tensor) -> torch.Tensor:
    """Linear (not circular) 2-D convolution, cropped to ``grid``'s shape.

    The stamp must have odd dimensions; it is centred on each cell.
    Mass that would fall outside the grid is lost, as when stamps are
    clipped at the edges.
    """
    if stamp.shape[0] % 2 == 0 or stamp.shape[1] % 2 == 0:
        raise ValueError(f"stamp dimensions must be odd, got {tuple(stamp.shape)}")
    nx, ny = grid.shape
    kx, ky = stamp.shape
    fx, fy = nx + kx - 1, ny + ky - 1
    spec = torch.fft.rfft2(grid.to(torch.float64), s=(fx, fy))
    spec *= torch.fft.rfft2(stamp.to(torch.float64), s=(fx, fy))
    full = torch.fft.irfft2(spec, s=(fx, fy))
    ox, oy = kx // 2, ky // 2
    return full[ox : ox + nx, oy : oy + ny].contiguous()


def support_mask(
    x_axis: GridAxis,
    y_axis: GridAxis,
    inside,
    *,
    subsamples: int = 4,
) -> torch.Tensor:
    """The fraction of each cell inside a region, for ``renormalize=``.

    ``inside(x, y)`` says whether data coordinates lie in the region (the
    box the data can occupy, say). Each cell is sampled at
    ``subsamples²`` points across its width, so a cell the region's edge
    cuts counts in proportion: a cell centred on the edge is half inside,
    which is also the share of its neighbourhood's points that linear
    deposition gives it.
    """
    offsets = (torch.arange(subsamples, dtype=torch.float64) + 0.5) / subsamples - 0.5
    xi = torch.arange(x_axis.size, dtype=torch.float64)[:, None] + offsets[None, :]
    yj = torch.arange(y_axis.size, dtype=torch.float64)[:, None] + offsets[None, :]
    xs = x_axis.to_value(xi)[:, None, :, None]
    ys = y_axis.to_value(yj)[None, :, None, :]
    xs, ys = torch.broadcast_tensors(xs, ys)
    return inside(xs, ys).to(torch.float64).mean(dim=(2, 3))


def grid_kde_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    x_axis: GridAxis,
    y_axis: GridAxis,
    sigma_cells: float | tuple[float, float],
    kernel: Kernel = "gaussian",
    deposit: Deposit = "linear",
    truncate: float = 4.0,
    renormalize: torch.Tensor | None = None,
    renormalize_floor: float = 0.25,
    zero_below: float = 1e-12,
) -> GridDensity:
    """Kernel density of ``(x, y)`` on ``x_axis × y_axis``.

    ``renormalize`` is an optional support mask on the grid (1 where the
    data can lie, 0 elsewhere, fractions on its edge: :func:`support_mask`). The density is divided by the kernel
    mass inside the support at each cell, floored at
    ``renormalize_floor`` (a quarter is the mass a symmetric kernel keeps
    at a corner), which corrects the deficit near the support's edge and
    continues smoothly beyond it. Values below ``zero_below`` × the
    maximum are set to zero, which clears FFT round-off from empty
    regions.
    """
    sx, sy = (
        (float(sigma_cells), float(sigma_cells))
        if not isinstance(sigma_cells, tuple)
        else (float(sigma_cells[0]), float(sigma_cells[1]))
    )
    hist = bin_2d(x, y, weights, x_axis=x_axis, y_axis=y_axis, deposit=deposit)
    stamp = kernel_stamp((sx, sy), kernel=kernel, truncate=truncate)
    values = fft_convolve2d(hist, stamp)
    if renormalize is not None:
        mass = fft_convolve2d(renormalize.to(torch.float64), stamp)
        values = values / mass.clamp_min(renormalize_floor)
    peak = values.max()
    if peak > 0:
        values = torch.where(values < zero_below * peak, 0.0, values)
    else:
        values = values.clamp_min(0.0)
    return GridDensity(
        values=values,
        x_axis=x_axis,
        y_axis=y_axis,
        sigma_cells=(sx, sy),
        kernel=kernel,
        total_weight=float(hist.sum()),
    )
