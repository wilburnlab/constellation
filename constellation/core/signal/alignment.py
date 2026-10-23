"""Monotone alignment of one axis onto another through modal density.

Given paired measurements ``(x, y)`` of the same things on two axes —
a peptide's retention time in a library and in a run — fit a
non-decreasing ``f`` with ``y ≈ f(x)`` that follows where the pairs are
*densest*, not where they average. Wrong pairs (false identifications)
scatter away from the dense band and barely move it, which is the point:
no least-squares step anywhere.

``method="greedy"``
    EncyclopeDIA's algorithm (Searle et al. 2018) in generic form: a
    2-D kernel density on a ``resolution²`` grid (its cosine kernel and
    bandwidth rule), the greedy walk from the densest cell
    (:func:`core.signal.modal_path.greedy_walk`), and the walked cells as
    a piecewise-linear curve. Options add what the jar lacks — padding so
    kernels are not cut at the grid edge, trimmed axis ranges so one
    outlier does not stretch the grid, a robust bandwidth — and default
    to its behaviour.

``method="global"``
    The globally best monotone path. Each axis is standardized (median,
    2.5–97.5 % span), and the plane rotated 45°: ``u = (x̃ + ỹ)/√2`` runs
    along the band, ``v = (ỹ − x̃)/√2`` across it. A path that moves at
    most ``(K−1)/K`` rows of v per column of u is strictly increasing in
    both x and y, so :func:`core.signal.modal_path.monotone_path` over the
    log-density, with a quadratic prior on the slope, finds the best
    strictly monotone curve (the posterior mean, a sub-cell estimate, is
    an option).
    Swapping x and y maps v to −v and the grid onto itself, so the fit
    is the same either way round. A second pass narrows the kernel
    across the band to the spread of the residuals (the jar's single
    bandwidth scales with the RT range, not with the band's width).
    The path, thinned to a knot every half bandwidth, becomes a
    rational-quadratic spline: smooth, strictly increasing, invertible.

``refine="density" | "modal"``
    Gradient refinement of a rational-quadratic spline started from
    either method's curve. The default fit is ``global`` refined by
    ``modal``, the most accurate in the benchmark; ``refine=None`` gives
    the unrefined path. ``density`` maximizes the log-density
    along the curve (sampled bicubically, weighted along u so it is
    symmetric in x and y), annealed over kernels 4, 2 and 1× the
    bandwidth. ``modal`` maximizes an annealed Gaussian kernel of the
    points' distances from the curve (the Welsch loss, i.e. modal
    regression; distances first-order orthogonal in standardized units),
    starting wide and narrowing to the residual spread.

Beyond the calibration range the curve follows an
:class:`core.stats.monotone.Extrapolation` policy (``soft`` by default),
and the residual band is EncyclopeDIA's: the empirical middle
``coverage`` of ``y − f(x)`` (:func:`quantile_band`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping

import torch

from constellation.core.optim import LBFGSOptimizer
from constellation.core.signal.modal_path import greedy_walk, monotone_path
from constellation.core.stats.density import (
    Deposit,
    GridAxis,
    GridDensity,
    grid_kde_2d,
    support_mask,
    weighted_quantile,
)
from constellation.core.stats.monotone import (
    Extrapolation,
    LinearMap,
    MonotoneCurve,
    PiecewiseLinear,
    RationalQuadraticSpline,
    TailedCurve,
    interp,
    steffen_slopes,
)

__all__ = [
    "GreedyOptions",
    "GlobalPathOptions",
    "RefineOptions",
    "MonotoneAlignment",
    "fit_monotone_alignment",
    "quantile_band",
    "theil_sen",
]

_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
#: The narrowest kernel across the band, as a fraction of the bandwidth
#: along it: what noiseless pairs (a residual spread of zero) get.
_MIN_SPREAD = 1e-3
_SQRT2 = math.sqrt(2.0)
_MAD = 1.482602218505602


@dataclass(frozen=True)
class GreedyOptions:
    """Settings for ``method="greedy"``; the defaults are EncyclopeDIA's.

    ``bandwidth="encyclopedia"`` pools the two axes' standard deviations
    in grid cells (Scott's rule for two dimensions) and reads the result
    as a full width at half maximum; ``"robust"`` uses IQR/1.349 instead
    of the standard deviation. The kernel σ is capped at
    ``max_sigma_fraction`` of the grid. ``trim_fraction`` sets each
    axis's range from that quantile instead of the extremes;
    ``pad_sigmas`` extends it by that many kernel σ; ``renormalize``
    corrects the density near the data box's edges. Fits with fewer than
    ``min_points`` pairs use a Theil–Sen line instead.
    """

    resolution: int = 3000
    bandwidth: Literal["encyclopedia", "robust"] = "encyclopedia"
    max_sigma_fraction: float = 1.0 / 40.0
    trim_fraction: float = 0.0
    pad_sigmas: float = 0.0
    renormalize: bool = False
    deposit: Deposit = "nearest"
    tie_rtol: float = 1e-9
    min_points: int = 21


@dataclass(frozen=True)
class GlobalPathOptions:
    """Settings for ``method="global"``.

    The grid has ``cells_per_sigma`` columns per kernel σ along the band
    and ``slope_steps`` (K) rows per column across it, so a path's local
    slope is one of ``2K − 1`` values between ``1/(2K − 1)`` and
    ``2K − 1`` times the ratio of the axes' spans. Each column scores
    ``log(D + density_floor · D̄)`` (``D̄`` the mean of the non-zero
    density; the floor is what a column with no correct pairs scores) and each move
    ``−slope_penalty · t²`` (``t`` the move per column, ``0`` at the span
    ratio), both per kernel σ of path. ``bandwidth_scale`` multiplies
    Scott's rule (the default reads it as a FWHM, as EncyclopeDIA does).
    ``second_pass`` refits with the kernel across the band set to the
    residual spread. ``path`` is the best path (``viterbi``, the default:
    the mode) or the posterior mean of the row in each column (``mean``:
    smoother, but where the band bends away from the prior slope the
    posterior spreads over both and its mean sits between them, as the
    benchmark showed at saturating ends). Points within ``inlier_sigmas`` of the path bound
    the curve's domain and support its tails; knots are placed every
    ``knot_spacing`` kernel σ.
    """

    slope_steps: int = 8
    slope_penalty: float = 1.0
    density_floor: float = 0.05
    cells_per_sigma: int = 4
    bandwidth_scale: float = 1.0 / _FWHM
    second_pass: bool = True
    renormalize: bool = True
    path: Literal["mean", "viterbi"] = "viterbi"
    knot_spacing: float = 0.5
    inlier_sigmas: float = 3.0
    max_columns: int = 2048
    max_rows: int = 8192
    min_points: int = 21


@dataclass(frozen=True)
class RefineOptions:
    """Settings for ``refine=``.

    ``n_bins`` defaults to one bin per two kernel σ of curve (8 to 64).
    ``density_scales`` are the kernel multiples of the density
    objective's annealing stages. The modal objective's kernel starts at
    ``modal_start`` × the residual spread and shrinks by ``modal_ratio``
    per stage until it reaches ``modal_final`` × the spread (at most
    ``modal_max_stages`` stages). ``smoothness`` weighs the roughness of
    the log-slope (neighbouring bins' secants, and each knot's derivative
    against the secants beside it) against the objective, which is a
    mean over points (modal) or over the curve (density); ``samples`` is
    the number of points per axis at which the density objective reads
    the curve.
    """

    n_bins: int | None = None
    density_scales: tuple[float, ...] = (4.0, 2.0, 1.0)
    modal_start: float = 3.0
    modal_ratio: float = 1.5
    modal_final: float = 1.0
    modal_max_stages: int = 8
    smoothness: float = 1e-2
    samples: int = 1024
    max_iter: int = 10


@dataclass(frozen=True)
class MonotoneAlignment:
    """A fitted monotone map, its extrapolation and its residual band."""

    curve: TailedCurve
    method: str
    band: tuple[float, float]
    n_points: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def domain(self) -> tuple[float, float]:
        return self.curve.domain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.curve.forward(x)

    __call__ = forward

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.curve.inverse(y)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.curve.derivative(x)

    def residuals(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``y − f(x)``."""
        return torch.as_tensor(y, dtype=torch.float64) - self.forward(x)

    def within_band(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = self.residuals(x, y)
        return torch.isfinite(r) & (r >= self.band[0]) & (r <= self.band[1])

    def is_extrapolated(self, x: torch.Tensor) -> torch.Tensor:
        return self.curve.is_extrapolated(x)

    def is_clamped(self, x: torch.Tensor) -> torch.Tensor:
        return self.curve.is_clamped(x)


def quantile_band(residuals: torch.Tensor, coverage: float = 0.95) -> tuple[float, float]:
    """The middle ``coverage`` of the finite residuals.

    EncyclopeDIA's rule: with the residuals sorted and ``t = (1 −
    coverage)/2``, the band runs from the value at index
    ``floor((n−1)·t)`` to the value at ``ceil((n−1)·(1−t))``. No finite
    residuals give an unbounded band.
    """
    r = torch.as_tensor(residuals, dtype=torch.float64).reshape(-1)
    r = torch.sort(r[torch.isfinite(r)]).values
    n = r.numel()
    if n == 0:
        return -math.inf, math.inf
    t = (1.0 - coverage) / 2.0
    lo = min(n - 1, max(0, math.floor((n - 1) * t)))
    hi = min(n - 1, max(0, math.ceil((n - 1) * (1.0 - t))))
    return float(r[lo]), float(r[hi])


def theil_sen(
    x: torch.Tensor, y: torch.Tensor, *, max_pairs: int = 1_000_000, seed: int = 0
) -> tuple[float, float]:
    """Median-of-pairwise-slopes line (Theil 1950, Sen 1968).

    The slope is the median over pairs with distinct x; the intercept
    the median of ``y − slope·x``. Robust to about 29 % wrong points and
    no least squares, which is why the small-sample fallback uses it.
    With no spread in x the slope is 1. Above ``max_pairs`` pairs the
    median is taken over that many pairs drawn at random (``seed``), so
    memory stays bounded.
    """
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    n = x.numel()
    if n == 0:
        raise ValueError("theil_sen needs at least one point")
    if not float(x.max() - x.min()) > 0:
        return 1.0, float(torch.median(y - x))
    if n * (n - 1) // 2 <= max_pairs:
        i, j = torch.triu_indices(n, n, 1)
    else:
        g = torch.Generator().manual_seed(seed)
        i = torch.randint(0, n, (max_pairs,), generator=g)
        j = torch.randint(0, n, (max_pairs,), generator=g)
    dx = x[j] - x[i]
    keep = dx != 0
    if bool(keep.any()):
        slope = float(torch.median((y[j] - y[i])[keep] / dx[keep]))
    else:
        slope = 1.0
    return slope, float(torch.median(y - slope * x))


# -- shared helpers -------------------------------------------------------


def _wmad(r: torch.Tensor, w: torch.Tensor) -> float:
    med = weighted_quantile(r, 0.5, w)
    return _MAD * float(weighted_quantile((r - med).abs(), 0.5, w))


def _small_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[MonotoneCurve, dict]:
    x0 = float(x[0])
    if x.numel() == 1 or not float(x.max() - x.min()) > 0:
        # One point, or one x however many points: an offset of slope 1.
        return LinearMap(1.0, float(torch.median(y)) - x0, (x0, x0)), {"fallback": "offset"}
    slope, intercept = theil_sen(x, y)
    if not slope > 0:
        span_x = float(x.max() - x.min())
        span_y = float(y.max() - y.min())
        slope = span_y / span_x if span_x > 0 and span_y > 0 else 1.0
        intercept = float(torch.median(y - slope * x))
    domain = (float(x.min()), float(x.max()))
    return LinearMap(slope, intercept, domain), {"fallback": "theil_sen"}


def _restrict(curve: PiecewiseLinear, lo: float, hi: float) -> PiecewiseLinear:
    """``curve`` cut to ``[lo, hi]`` (within its own domain)."""
    k = curve.knots()
    a, b = max(lo, float(k.x[0])), min(hi, float(k.x[-1]))
    if not b > a:
        return curve
    inside = (k.x > a) & (k.x < b)
    ends = torch.tensor([a, b], dtype=torch.float64)
    x = torch.cat([ends[:1], k.x[inside], ends[1:]])
    y = torch.cat([curve.forward(ends[:1]), k.y[inside], curve.forward(ends[1:])])
    return PiecewiseLinear(x, y)


@dataclass(frozen=True)
class _Frame:
    """Standardization and rotation shared by the global path and refinement."""

    cx: float
    sx: float
    cy: float
    sy: float
    sigma: float  # Scott bandwidth (as σ) in standardized units

    def uv(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xt = (x - self.cx) / self.sx
        yt = (y - self.cy) / self.sy
        return (xt + yt) / _SQRT2, (yt - xt) / _SQRT2

    def xy(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xt = (u - v) / _SQRT2
        yt = (u + v) / _SQRT2
        return self.cx + self.sx * xt, self.cy + self.sy * yt


def _frame(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, scale: float) -> _Frame:
    qx = weighted_quantile(x, torch.tensor([0.025, 0.5, 0.975]), w)
    qy = weighted_quantile(y, torch.tensor([0.025, 0.5, 0.975]), w)
    sx, sy = float(qx[2] - qx[0]), float(qy[2] - qy[0])
    if not (sx > 0 and sy > 0):
        raise ValueError("x and y must each have spread to be aligned")
    xt = (x - qx[1]) / sx
    yt = (y - qy[1]) / sy

    def iqr(v: torch.Tensor) -> float:
        return float(weighted_quantile(v, 0.75, w) - weighted_quantile(v, 0.25, w))

    n_eff = float(w.sum() ** 2 / (w * w).sum())
    spread = (iqr(xt) + iqr(yt)) / (2.0 * 1.349)
    sigma = scale * n_eff ** (-1.0 / 6.0) * max(spread, 1e-6)
    return _Frame(float(qx[1]), sx, float(qy[1]), sy, sigma)


def _rotated_density(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    *,
    sigma_u: float,
    sigma_v: float,
    du: float,
    dv: float,
    box: tuple[float, float, float, float] | None,
    max_rows: int,
) -> GridDensity:
    pad_u, pad_v = 3.0 * sigma_u, 3.0 * sigma_v
    n_u = max(2, math.ceil((float(u.max()) - float(u.min()) + 2 * pad_u) / du) + 1)
    u_axis = GridAxis(float(u.min()) - pad_u, du, n_u)
    half = float(v.abs().max()) + pad_v
    n_half = min(math.ceil(half / dv), max_rows // 2)
    v_axis = GridAxis(-n_half * dv, dv, 2 * n_half + 1)
    mask = None
    if box is not None:
        x_lo, x_hi, y_lo, y_hi = box

        def inside(uu: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
            xt, yt = (uu - vv) / _SQRT2, (uu + vv) / _SQRT2
            return (xt >= x_lo) & (xt <= x_hi) & (yt >= y_lo) & (yt <= y_hi)

        mask = support_mask(u_axis, v_axis, inside, subsamples=2)
    return grid_kde_2d(
        u,
        v,
        w,
        x_axis=u_axis,
        y_axis=v_axis,
        sigma_cells=(sigma_u / du, sigma_v / dv),
        kernel="gaussian",
        deposit="linear",
        renormalize=mask,
    )


def _log_score(density: GridDensity, floor: float) -> tuple[torch.Tensor, float]:
    d = density.values
    nonzero = d[d > 0]
    mean = float(nonzero.mean()) if nonzero.numel() else 1.0
    return torch.log(d + floor * mean) - math.log(mean), mean


# -- greedy -----------------------------------------------------------------


def _fit_greedy(x, y, w, opts: GreedyOptions):
    res = int(opts.resolution)
    if opts.trim_fraction > 0:
        qx = weighted_quantile(x, torch.tensor([opts.trim_fraction, 1 - opts.trim_fraction]), w)
        qy = weighted_quantile(y, torch.tensor([opts.trim_fraction, 1 - opts.trim_fraction]), w)
        x_lo, x_hi, y_lo, y_hi = float(qx[0]), float(qx[1]), float(qy[0]), float(qy[1])
    else:
        x_lo, x_hi, y_lo, y_hi = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    rx, ry = x_hi - x_lo, y_hi - y_lo
    if not (rx > 0 and ry > 0):
        raise ValueError("x and y must each have spread to be aligned")
    if opts.bandwidth == "encyclopedia":
        mx = (w * x).sum() / w.sum()
        my = (w * y).sum() / w.sum()
        scale_x = float(torch.sqrt((w * (x - mx) ** 2).sum() / w.sum()))
        scale_y = float(torch.sqrt((w * (y - my) ** 2).sum() / w.sum()))
    elif opts.bandwidth == "robust":
        scale_x = float(weighted_quantile(x, 0.75, w) - weighted_quantile(x, 0.25, w)) / 1.349
        scale_y = float(weighted_quantile(y, 0.75, w) - weighted_quantile(y, 0.25, w)) / 1.349
    else:
        raise ValueError(f"bandwidth must be 'encyclopedia' or 'robust', got {opts.bandwidth!r}")
    n_eff = float(w.sum() ** 2 / (w * w).sum())
    h_cells = n_eff ** (-1.0 / 6.0) * (scale_x / rx + scale_y / ry) * (res - 1) / 2.0
    sigma_cells = min(res * opts.max_sigma_fraction, h_cells / _FWHM)
    sigma_x = sigma_cells * rx / (res - 1)  # data units
    sigma_y = sigma_cells * ry / (res - 1)
    x_axis = GridAxis.spanning(x_lo - opts.pad_sigmas * sigma_x, x_hi + opts.pad_sigmas * sigma_x, res)
    y_axis = GridAxis.spanning(y_lo - opts.pad_sigmas * sigma_y, y_hi + opts.pad_sigmas * sigma_y, res)
    mask = None
    if opts.renormalize:
        mask = support_mask(
            x_axis,
            y_axis,
            lambda xs, ys: (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi),
            subsamples=2,
        )
    density = grid_kde_2d(
        x,
        y,
        w,
        x_axis=x_axis,
        y_axis=y_axis,
        sigma_cells=(sigma_x / x_axis.step, sigma_y / y_axis.step),
        kernel="cosine_approx",
        deposit=opts.deposit,
        renormalize=mask,
    )
    i, j = greedy_walk(density.values, tie_rtol=opts.tie_rtol)
    curve = PiecewiseLinear(x_axis.to_value(i.to(torch.float64)), y_axis.to_value(j.to(torch.float64)))
    curve = _restrict(curve, float(x.min()), float(x.max()))
    diagnostics = {"sigma_cells": sigma_cells, "resolution": res, "n_knots": int(i.numel())}
    return curve, diagnostics, density


# -- global path --------------------------------------------------------------


def _path(u, v, w, opts: GlobalPathOptions, sigma_u, sigma_v, k_steps, box):
    du = sigma_u / opts.cells_per_sigma
    span_u = float(u.max() - u.min()) + 6.0 * sigma_u
    if span_u / du > opts.max_columns:
        du = span_u / opts.max_columns
    dv = du / k_steps
    density = _rotated_density(
        u, v, w, sigma_u=sigma_u, sigma_v=sigma_v, du=du, dv=dv, box=box, max_rows=opts.max_rows
    )
    score, _ = _log_score(density, opts.density_floor)
    per_column = du / sigma_u
    k = torch.arange(-(k_steps - 1), k_steps, dtype=torch.float64)
    t = k / k_steps
    post = monotone_path(per_column * score, -per_column * opts.slope_penalty * t * t)
    rows = post.mean if opts.path == "mean" else post.viterbi.to(torch.float64)
    cols = torch.arange(density.x_axis.size, dtype=torch.float64)
    return density.x_axis.to_value(cols), density.y_axis.to_value(rows), post, density


def _fit_global(x, y, w, opts: GlobalPathOptions, curve_kind: str):
    frame = _frame(x, y, w, opts.bandwidth_scale)
    u, v = frame.uv(x, y)
    xt, yt = (x - frame.cx) / frame.sx, (y - frame.cy) / frame.sy
    box = (float(xt.min()), float(xt.max()), float(yt.min()), float(yt.max())) if opts.renormalize else None
    sigma = frame.sigma
    k_steps = max(2, int(opts.slope_steps))
    uc, vc, post, density = _path(u, v, w, opts, sigma, sigma, k_steps, box)

    def spread(r, limit):
        near = r.abs() < limit
        return _wmad(r[near], w[near]) if int(near.sum()) >= 3 else limit / 3.0

    r = v - interp(u, uc, vc)
    sigma_v = spread(r, 3.0 * sigma)
    passes = 1
    if opts.second_pass:
        # Narrow the kernel across the band to the residual spread, with
        # rows fine enough to resolve it (a finer v step is a larger K,
        # bounded by the row budget).
        sigma_v = min(max(sigma_v, 1e-6), sigma)
        du = sigma / opts.cells_per_sigma
        half = float(v.abs().max()) + 3.0 * sigma_v
        wanted = math.ceil(4.0 * du / sigma_v)
        affordable = int(opts.max_rows * du / (2.0 * half))
        k2 = max(k_steps, min(wanted, affordable))
        uc, vc, post, density = _path(u, v, w, opts, sigma, sigma_v, k2, box)
        r = v - interp(u, uc, vc)
        sigma_v = spread(r, 3.0 * sigma)
        passes = 2
    # A spread finer than the grid's rows cannot be resolved (noiseless
    # pairs measure zero), and the refinements need it positive.
    sigma_v = max(sigma_v, density.y_axis.step)
    inlier = r.abs() <= opts.inlier_sigmas * sigma_v
    if int(inlier.sum()) < 2:
        inlier = torch.ones_like(inlier)
    x_in = x[inlier]
    lo, hi = float(x_in.min()), float(x_in.max())
    px, py = frame.xy(uc, vc)
    # The path is strictly increasing in both coordinates; cut it to the
    # inlier domain with exact end knots.
    keep = (px > lo) & (px < hi)
    stride = max(1, round(opts.knot_spacing * opts.cells_per_sigma))
    idx = torch.nonzero(keep).reshape(-1)[::stride]
    full = PiecewiseLinear(px, py)
    ends = torch.tensor([lo, hi], dtype=torch.float64)
    kx = torch.cat([ends[:1], px[idx], ends[1:]])
    ky = torch.cat([full.forward(ends[:1]), py[idx], full.forward(ends[1:])])
    order = torch.argsort(kx)
    kx, ky = kx[order], ky[order]
    distinct = torch.cat([torch.tensor([True]), torch.diff(kx) > 0])
    kx, ky = kx[distinct], ky[distinct]
    if curve_kind == "linear":
        curve: MonotoneCurve = PiecewiseLinear(kx, ky)
    else:
        curve = RationalQuadraticSpline.from_knots(kx, ky, steffen_slopes(kx, ky, min_fraction=1e-3))
    diagnostics = {
        "sigma": sigma,
        "sigma_across": sigma_v,
        "passes": passes,
        "grid": tuple(density.values.shape),
        "n_inliers": int(inlier.sum()),
        "log_evidence": post.log_evidence,
        "path_sd": post.sd,
    }
    return curve, diagnostics, frame, sigma_v, inlier


# -- refinement -----------------------------------------------------------------


def _refine(
    curve: MonotoneCurve,
    x, y, w,
    frame: _Frame,
    sigma_across: float,
    kind: str,
    gopts: GlobalPathOptions,
    opts: RefineOptions,
) -> RationalQuadraticSpline:
    lo, hi = curve.domain
    with torch.no_grad():
        y0, y1 = (float(v) for v in curve.forward(torch.tensor([lo, hi], dtype=torch.float64)))
    u_length = ((hi - lo) / frame.sx + (y1 - y0) / frame.sy) / _SQRT2
    n_bins = opts.n_bins or int(min(64, max(8, round(u_length / (2.0 * frame.sigma)))))
    model = RationalQuadraticSpline.from_curve(curve, n_bins)
    if kind == "density":
        _refine_density(model, x, y, w, frame, sigma_across, gopts, opts)
    elif kind == "modal":
        _refine_modal(model, x, y, w, frame, opts)
    else:
        raise ValueError(f"refine must be 'density' or 'modal', got {kind!r}")
    return model


def _smoothness(model: RationalQuadraticSpline) -> torch.Tensor:
    """Roughness of the spline's log-slope.

    A bin's shape is set by its secant and the derivatives at its ends,
    so both are held: neighbouring secants to each other, and each knot
    derivative to the secants on either side of it.
    """
    _, _, w, h, d = model._params()
    log_s = torch.log(h / w)
    log_d = torch.log(d)
    beside = torch.cat([log_s[:1], 0.5 * (log_s[:-1] + log_s[1:]), log_s[-1:]])
    return (torch.diff(log_s) ** 2).sum() + ((log_d - beside) ** 2).sum()


def _lbfgs(model: RationalQuadraticSpline, data: torch.Tensor, loss_fn, max_iter: int) -> None:
    """One annealing stage. The parameters are held in a box around where
    the stage starts (a line search can otherwise overflow the image span),
    and a stage that still ends non-finite is undone."""
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        initial = float(loss_fn(model.forward(data)))
    log_span = model.log_y_span.item()
    span = math.exp(log_span)
    y_lo = model.y_lo.item()
    bounds = {
        "theta_*": (-30.0, 30.0),
        "log_y_span": (log_span - 3.0, log_span + 3.0),
        "y_lo": (y_lo - 10.0 * span, y_lo + 10.0 * span),
    }
    optimizer = LBFGSOptimizer(model, max_iter=25, bounds=bounds)
    # The losses are O(1); a change below 1e-7 of the loss is converged
    # for an RT curve.
    model.fit(data, optimizer=optimizer, loss_fn=loss_fn, max_iter=max_iter, tol=1e-7 * (abs(initial) + 1.0))
    with torch.no_grad():
        final = float(loss_fn(model.forward(data)))
        finite = math.isfinite(final) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
    if not finite or final > initial:
        model.load_state_dict(start)


def _refine_density(model, x, y, w, frame, sigma_across, gopts, opts):
    u, v = frame.uv(x, y)
    xt, yt = (x - frame.cx) / frame.sx, (y - frame.cy) / frame.sy
    box = (float(xt.min()), float(xt.max()), float(yt.min()), float(yt.max())) if gopts.renormalize else None
    lo, hi = model.domain
    for scale in opts.density_scales:
        sigma_u = frame.sigma * scale
        sigma_v = min(max(sigma_across, _MIN_SPREAD * frame.sigma) * scale, sigma_u)
        du = sigma_u / gopts.cells_per_sigma
        # Rows of a quarter kernel, unless that would take more rows than
        # the budget allows to cover the pairs.
        half = float(v.abs().max()) + 3.0 * sigma_v
        dv = max(min(du, sigma_v / 4.0), 2.0 * half / gopts.max_rows)
        density = _rotated_density(
            u, v, w, sigma_u=sigma_u, sigma_v=sigma_v, du=du, dv=dv, box=box, max_rows=gopts.max_rows
        )
        _, mean = _log_score(density, gopts.density_floor)
        with torch.no_grad():
            y_lo, y_hi = model.image
            along_x = torch.linspace(lo, hi, opts.samples + 1, dtype=torch.float64)
            along_y = model.inverse(torch.linspace(y_lo, y_hi, opts.samples + 1, dtype=torch.float64))
            xs = torch.unique(torch.cat([along_x, along_y.clamp(lo, hi)]))
            # The measure each sample stands for is fixed for the stage.
            # Taken from the moving curve, it would pay the curve to grow:
            # a near-vertical end through the band adds length where the
            # density is high.
            measure = torch.diff(frame.uv(xs, model.forward(xs))[0]) / sigma_u

        def loss_fn(pred, density=density, mean=mean, measure=measure):
            us, vs = frame.uv(xs, pred)
            d = density.sample(us, vs, mode="bicubic").clamp_min(0.0)
            s = torch.log(d + gopts.density_floor * mean) - math.log(mean)
            t = torch.diff(vs) / torch.diff(us)
            gain = measure * (0.5 * (s[1:] + s[:-1]) - gopts.slope_penalty * t * t)
            return -gain.sum() / measure.sum() + opts.smoothness * _smoothness(model)

        _lbfgs(model, xs, loss_fn, opts.max_iter)


def _refine_modal(model, x, y, w, frame, opts):
    yt = (y - frame.cy) / frame.sy
    ratio = frame.sx / frame.sy

    def distances() -> torch.Tensor:
        ft = (model.forward(x) - frame.cy) / frame.sy
        slope = model.derivative(x) * ratio
        return (yt - ft) / torch.sqrt(1.0 + slope * slope)

    with torch.no_grad():
        spread = max(_wmad(distances(), w), 1e-4)
    h = opts.modal_start * spread
    for _ in range(opts.modal_max_stages):
        # The orthogonal scaling is fixed for the stage (reweighting, as in
        # IRLS): differentiated, it would pay the curve to turn vertical
        # at its ends, where a vertical stretch is "near" every point.
        with torch.no_grad():
            slope = model.derivative(x) * ratio
            scale = torch.sqrt(1.0 + slope * slope)

        def loss_fn(pred, h=h, scale=scale):
            r = (yt - (pred - frame.cy) / frame.sy) / scale
            kernel = torch.exp(-0.5 * (r / h) ** 2)
            return -(w * kernel).sum() / w.sum() + opts.smoothness * _smoothness(model)

        _lbfgs(model, x, loss_fn, opts.max_iter)
        with torch.no_grad():
            r = distances()
            near = r.abs() < 2.5 * h
            if int(near.sum()) >= 3:
                spread = max(_wmad(r[near], w[near]), 1e-4)
        if h <= opts.modal_final * spread:
            break
        h = max(h / opts.modal_ratio, opts.modal_final * spread, 1e-4)


# -- entry point --------------------------------------------------------------


def fit_monotone_alignment(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    method: Literal["greedy", "global"] = "global",
    curve: Literal["linear", "rational_quadratic"] | None = None,
    refine: Literal["auto", "density", "modal"] | None = "auto",
    extrapolation: Extrapolation | Literal["soft", "linear", "clamp"] = "soft",
    coverage: float = 0.95,
    options: GreedyOptions | GlobalPathOptions | None = None,
    refine_options: RefineOptions | None = None,
) -> MonotoneAlignment:
    """Fit a monotone ``y ≈ f(x)`` through the densest band of the pairs.

    The default is the global path refined by the modal objective, the
    most accurate configuration in the benchmark
    (``docs/plans/rt-alignment.md``). ``refine="auto"`` is each method's
    own refinement: ``modal`` for ``global`` (unless ``curve="linear"``),
    none for ``greedy``; ``refine=None`` turns it off. ``curve`` defaults
    to ``linear`` for an unrefined ``greedy`` fit (the walked cells) and
    otherwise to ``rational_quadratic``; a refinement needs the spline.
    ``weights`` scale each pair's
    contribution to the density (e.g. a confidence). Non-finite pairs are
    dropped. Fewer pairs than the method's ``min_points`` give a
    Theil–Sen line (one pair: an offset of slope 1).
    """
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same length")
    w = (
        torch.ones_like(x)
        if weights is None
        else torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    )
    finite = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0)
    x, y, w = x[finite], y[finite], w[finite]
    if x.numel() == 0:
        raise ValueError("no finite pairs to align")
    policy = Extrapolation.coerce(extrapolation)

    if method == "greedy":
        opts = options if options is not None else GreedyOptions()
        if not isinstance(opts, GreedyOptions):
            raise TypeError("method='greedy' takes GreedyOptions")
    elif method == "global":
        opts = options if options is not None else GlobalPathOptions()
        if not isinstance(opts, GlobalPathOptions):
            raise TypeError("method='global' takes GlobalPathOptions")
    else:
        raise ValueError(f"method must be 'greedy' or 'global', got {method!r}")
    if refine == "auto":
        refine = "modal" if method == "global" and curve in (None, "rational_quadratic") else None
    kind = curve or ("rational_quadratic" if (refine or method == "global") else "linear")
    if refine is not None and kind != "rational_quadratic":
        raise ValueError("refine= needs curve='rational_quadratic'")

    if x.numel() < opts.min_points or float(x.max() - x.min()) == 0.0:
        base, diagnostics = _small_fit(x, y)
        support = torch.ones_like(x, dtype=torch.bool)
        tailed = TailedCurve.build(base, policy, x_support=x[support], w_support=w[support])
        band = quantile_band(y - tailed.forward(x), coverage)
        return MonotoneAlignment(tailed, method, band, int(x.numel()), diagnostics)

    if method == "greedy":
        base, diagnostics, _ = _fit_greedy(x, y, w, opts)
        frame = None
        sigma_across = None
        if kind == "rational_quadratic":
            base = RationalQuadraticSpline.from_curve(base, _bins_for(base, x, y, w))
        support = None
    else:
        base, diagnostics, frame, sigma_across, support = _fit_global(x, y, w, opts, kind)

    if refine is not None:
        gopts = opts if isinstance(opts, GlobalPathOptions) else GlobalPathOptions()
        if frame is None:
            frame = _frame(x, y, w, gopts.bandwidth_scale)
            _, vc = frame.uv(x, y)
            with torch.no_grad():
                _, vf = frame.uv(x, _f64(base.forward(x)))
            r = vc - vf
            near = r.abs() < 3.0 * frame.sigma
            sigma_across = _wmad(r[near], w[near]) if int(near.sum()) >= 3 else frame.sigma
            sigma_across = min(max(sigma_across, _MIN_SPREAD * frame.sigma), frame.sigma)
        base = _refine(base, x, y, w, frame, sigma_across, refine, gopts, refine_options or RefineOptions())
        diagnostics = {**diagnostics, "refine": refine}

    with torch.no_grad():
        residual = y - _f64(base.forward(x))
        band = quantile_band(residual, coverage)
        if support is None:
            support = (residual >= band[0]) & (residual <= band[1])
        tailed = TailedCurve.build(base, policy, x_support=x[support], w_support=w[support])
    return MonotoneAlignment(tailed, method, band, int(x.numel()), diagnostics)


def _f64(v) -> torch.Tensor:
    return torch.as_tensor(v, dtype=torch.float64)


def _bins_for(curve: MonotoneCurve, x, y, w) -> int:
    frame = _frame(x, y, w, GlobalPathOptions().bandwidth_scale)
    lo, hi = curve.domain
    with torch.no_grad():
        y0, y1 = (float(v) for v in curve.forward(torch.tensor([lo, hi], dtype=torch.float64)))
    u_length = ((hi - lo) / frame.sx + (y1 - y0) / frame.sy) / _SQRT2
    return int(min(64, max(8, round(u_length / (2.0 * frame.sigma)))))
