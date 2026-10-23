"""EncyclopeDIA 6.9.23's retention-time fit, replayed bit for bit.

The jar fits each search's library-to-run RT map in
``RetentionTimeFilter.getFilter``: more than 20 points go through
``TwoDimensionalKDE(points, 3000).trace()``, 2 to 20 through a float32
least-squares line, fewer through the identity. This module repeats
that arithmetic, read from the jar's bytecode, so that for the same
points in the same order it returns the same warp, residual and 95 %
flag as the jar's ``rt_fit.txt`` sidecars, to the bit. It is checked on
the lab's four reference sidecars (205,000 rows). It is
``align(method="encyclopedia")``: the reference the other methods are
measured against (``docs/plans/rt-alignment.md``), and what reproduces a
search's own fit.

What the jar does, step by step (float32 throughout unless noted):

1. Each axis spans the float32 minimum to maximum of the points, and a
   value maps to ``round((2999 / range) · (v − start))``, rounded half
   up, clamped to the grid (a zero range maps everything to cell 1500).
2. The kernel σ, in cells, is ``min(75, h / 2.3548)`` with
   ``h = n^(−1/6) · (sd_i + sd_j) / 2`` (double), the standard
   deviations of the cell indices computed with float32 running sums.
3. Every point adds a stamp of ``A·cos(π r / 4σ)`` (``A = π / 8σ``, the
   cosine a parabolic approximation, r the distance in cells, zero
   beyond 2σ) to a float32 grid, in point order, cut at the edges.
4. The walk (:func:`core.signal.modal_path.greedy_walk`) starts at the
   first maximum, walks down and then up, cutting corners where the
   mapped coordinate repeats, and the visited cells map back to times;
   sorted, they are the knots of a piecewise-linear function that clamps
   outside them and locates a query with ``Arrays.binarySearch``.
5. The band is the 2.5th to 97.5th percentile of the float32 residuals,
   by index (``floor`` / ``ceil``).

Times go in as the jar has them, in minutes. Only the arithmetic is
float32; results come back as float64 tensors holding float32 values.
"""

from __future__ import annotations

import math
import struct
from array import array
from dataclasses import dataclass
from typing import Literal

import torch

from constellation.core.signal.modal_path import greedy_walk
from constellation.core.stats.density import cosine_approx, fft_convolve2d
from constellation.core.stats.monotone import Knots

__all__ = [
    "RESOLUTION",
    "MIN_KDE_POINTS",
    "EncyclopediaFit",
    "JarLinearFunction",
    "JarLinearRegression",
    "fit_encyclopedia",
]

#: The grid the search path uses (``TwoDimensionalKDE.DEFAULT_RESOLUTION``).
RESOLUTION = 3000
#: ``getFilter`` uses the KDE above 20 points.
MIN_KDE_POINTS = 21
_BANDWIDTH_TO_STDEV = 2.0 * math.sqrt(2.0 * math.log(2.0))
_NEG_SIXTH = -0.1666666716337204  # (double)(-1f/6f), as the jar writes it


def _f32(v: float) -> float:
    return struct.unpack("f", struct.pack("f", v))[0]


def _round_half_up(v: float) -> int:
    """Java ``Math.round(float)``."""
    return math.floor(v + 0.5)


def _to_f32(values) -> torch.Tensor:
    """``(float) v`` for every value, keeping the shape."""
    return torch.as_tensor(values, dtype=torch.float64).to(torch.float32)


class _Range:
    """``org.searlelab.msrawjava.model.Range`` over float32 bounds."""

    def __init__(self, values: torch.Tensor) -> None:
        self.start = float(values.min())
        self.stop = float(values.max())

    def index(self, v: torch.Tensor, resolution: int) -> torch.Tensor:
        """``linearInterp(v, 0, res − 1)`` then ``Math.round``."""
        top = float(resolution - 1)
        width = _f32(self.stop - self.start)
        if width == 0.0:
            return torch.full(v.shape, _round_half_up(_f32(top / 2.0)), dtype=torch.long)
        scale = torch.tensor(_f32(top / width), dtype=torch.float32)
        val = (scale * (v - torch.tensor(self.start, dtype=torch.float32))).clamp(0.0, top)
        return torch.floor(val.to(torch.float64) + 0.5).long()

    def values(self, resolution: int) -> torch.Tensor:
        """``mapBackToRange(i, 0, res − 1)`` for every cell, as float32."""
        width = _f32(self.stop - self.start)
        if width == 0.0:
            return torch.full((resolution,), self.stop, dtype=torch.float32)
        step = torch.tensor(_f32(width / float(resolution - 1)), dtype=torch.float32)
        cells = torch.arange(resolution, dtype=torch.float32)
        val = step * cells + torch.tensor(self.start, dtype=torch.float32)
        return val.clamp(self.start, self.stop)


def _stdev(values: list[float]) -> float:
    """``General.stdev(float[])``: population SD with float32 running sums."""
    n = len(values)
    if n == 0:
        return 0.0
    acc = array("f", [0.0])
    for v in values:
        acc[0] = acc[0] + v
    mean = _f32(acc[0] / _f32(float(n)))
    acc[0] = 0.0
    for v in values:
        d = _f32(v - mean)
        acc[0] = acc[0] + _f32(d * d)
    return _f32(math.sqrt(_f32(acc[0] / _f32(float(n)))))


def _stamp(sigma: float) -> torch.Tensor:
    """``getStamp(new CosineGaussian(0, σ, 1))``, entries ≤ 0 zeroed."""
    radius = _round_half_up(_f32(2.0 * _f32(sigma)))
    a = math.pi / (8.0 * sigma)
    m = math.pi / (4.0 * sigma)
    lo, hi = 0.0 - 2.0 * sigma, 0.0 + 2.0 * sigma
    width = 2 * radius + 1
    out = [[0.0] * width for _ in range(width)]
    for i in range(width):
        for j in range(width):
            di, dj = abs(i - radius), abs(j - radius)
            r = math.sqrt(float(di * di + dj * dj))
            v = 0.0 if (r < lo or r > hi) else _f32(a * cosine_approx((0.0 - r) * m))
            out[i][j] = v if v > 0.0 else 0.0
    return torch.tensor(out, dtype=torch.float32)


def _sigma(xi: torch.Tensor, yi: torch.Tensor, n: int, resolution: int) -> float:
    sd_x = _stdev([float(v) for v in xi.tolist()])
    sd_y = _stdev([float(v) for v in yi.tolist()])
    h = math.pow(float(n), _NEG_SIXTH) * float(_f32(sd_x + sd_y)) / 2.0
    return min(resolution / 10.0 / 4.0, h / _BANDWIDTH_TO_STDEV)


def _histogram(xi, yi, stamp, resolution, backend):
    radius = stamp.shape[0] // 2
    if backend == "exact":
        grid = torch.zeros(resolution, resolution, dtype=torch.float32)
        for a, b in zip(xi.tolist(), yi.tolist()):
            x0, x1 = max(0, a - radius), min(resolution, a + radius + 1)
            y0, y1 = max(0, b - radius), min(resolution, b + radius + 1)
            grid[x0:x1, y0:y1] += stamp[x0 - a + radius : x1 - a + radius, y0 - b + radius : y1 - b + radius]
        return grid
    if backend == "fft":
        counts = torch.zeros(resolution * resolution, dtype=torch.float64)
        counts.index_add_(0, xi * resolution + yi, torch.ones(xi.numel(), dtype=torch.float64))
        grid = fft_convolve2d(counts.view(resolution, resolution), stamp.to(torch.float64))
        return torch.where(grid < 1e-9 * grid.max(), 0.0, grid)
    raise ValueError(f"backend must be 'exact' or 'fft', got {backend!r}")


def _binary_search(sorted_values: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Java ``Arrays.binarySearch(double[], double)`` for every key.

    Returns the index of a match (the one Java's halving lands on when
    values repeat) or ``-(insertion point) − 1``. ``-0.0`` sorts below
    ``0.0``, as ``Double.doubleToLongBits`` makes it.
    """
    lo = torch.zeros(keys.shape, dtype=torch.long)
    hi = torch.full(keys.shape, sorted_values.numel() - 1, dtype=torch.long)
    found = torch.full(keys.shape, -1, dtype=torch.long)
    key_neg = torch.signbit(keys)
    for _ in range(max(1, sorted_values.numel()).bit_length() + 1):
        active = (lo <= hi) & (found < 0)
        if not bool(active.any()):
            break
        mid = (lo + hi) >> 1
        val = sorted_values[mid.clamp(0, sorted_values.numel() - 1)]
        less = val < keys
        greater = val > keys
        equal = ~less & ~greater
        val_neg = torch.signbit(val)
        same_bits = equal & (val_neg == key_neg)
        go_right = less | (equal & ~same_bits & val_neg & ~key_neg)
        go_left = greater | (equal & ~same_bits & ~val_neg & key_neg)
        found = torch.where(active & same_bits, mid, found)
        lo = torch.where(active & go_right, mid + 1, lo)
        hi = torch.where(active & go_left, mid - 1, hi)
    return torch.where(found >= 0, found, -(lo + 1))


def _linear_interp(x1, x, x2, y1, y2) -> torch.Tensor:
    """``LinearInterpolatedFunction.linearInterp`` (double in, float out)."""
    dx = x2 - x1
    dy = y2 - y1
    flat_x = ((y2 + y1).to(torch.float32) / 2.0).to(torch.float64)
    flat_y = y2.to(torch.float32).to(torch.float64)
    safe = torch.where(dx == 0, 1.0, dx)
    slope = ((y2 - y1) / safe * (x - x1) + y1).to(torch.float32).to(torch.float64)
    return torch.where(dx == 0, flat_x, torch.where(dy == 0, flat_y, slope))


class JarLinearFunction:
    """EncyclopeDIA's ``LinearInterpolatedFunction`` over walk knots.

    Queries are cast to float32 first, located with Java's binary search
    and interpolated in double, then rounded to float32; beyond the knots
    the end values hold. A :class:`core.stats.monotone.MonotoneCurve`.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
        self._y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)

    @staticmethod
    def _evaluate(q, xs, ys) -> torch.Tensor:
        q = _to_f32(q).to(torch.float64)
        k = _binary_search(xs, q)
        k = torch.where(k < 0, -(k + 1), k)
        n = xs.numel()
        lo_val = ys[0].to(torch.float32).to(torch.float64)
        hi_val = ys[-1].to(torch.float32).to(torch.float64)
        kk = k.clamp(1, max(1, n - 1))
        if n >= 2:
            inner = _linear_interp(xs[kk - 1], q, xs[kk], ys[kk - 1], ys[kk])
        else:
            inner = torch.full_like(q, float(lo_val))
        return torch.where(k <= 0, lo_val, torch.where(k >= n, hi_val, inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``getYValue``."""
        return self._evaluate(x, self._x, self._y)

    __call__ = forward

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """``getXValue``."""
        return self._evaluate(y, self._y, self._x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float64)
        if self._x.numel() < 2:
            return torch.zeros_like(x)
        dx = torch.diff(self._x)
        slopes = torch.where(dx > 0, torch.diff(self._y) / torch.where(dx > 0, dx, 1.0), 0.0)
        idx = (torch.searchsorted(self._x, x.contiguous(), right=True) - 1).clamp(0, slopes.numel() - 1)
        inside = (x >= self._x[0]) & (x <= self._x[-1])
        return torch.where(inside, slopes[idx], 0.0)

    @property
    def domain(self) -> tuple[float, float]:
        return float(self._x[0]), float(self._x[-1])

    @property
    def image(self) -> tuple[float, float]:
        return float(self._y[0]), float(self._y[-1])

    def knots(self) -> Knots:
        return Knots(self._x.clone(), self._y.clone(), None, "encyclopedia")


class JarLinearRegression:
    """EncyclopeDIA's ``LinearRegression``: ``x·m + b`` in float32, unclamped."""

    def __init__(self, slope: float, intercept: float, domain: tuple[float, float]) -> None:
        self.slope = slope
        self.intercept = intercept
        self._domain = domain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = _to_f32(x)
        out = q * torch.tensor(self.slope, dtype=torch.float32) + torch.tensor(self.intercept, dtype=torch.float32)
        return out.to(torch.float64)

    __call__ = forward

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        q = _to_f32(y)
        out = (q - torch.tensor(self.intercept, dtype=torch.float32)) / torch.tensor(self.slope, dtype=torch.float32)
        return out.to(torch.float64)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(torch.as_tensor(x, dtype=torch.float64), self.slope)

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    @property
    def image(self) -> tuple[float, float]:
        lo, hi = self.forward(torch.tensor(self._domain, dtype=torch.float64)).tolist()
        return lo, hi

    def knots(self) -> Knots:
        x = torch.tensor(self._domain, dtype=torch.float64)
        return Knots(x, self.forward(x), torch.full_like(x, self.slope), "linear")


def _regression(x: list[float], y: list[float]) -> tuple[float, float]:
    """``LinearRegression.getRegression(float[], float[])``."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, y[0]
    if n == 2:
        m = _f32(_f32(y[1] - y[0]) / _f32(x[1] - x[0]))
        return m, _f32(y[0] - _f32(m * x[0]))
    acc = array("f", [0.0, 0.0, 0.0, 0.0])
    for a, b in zip(x, y):
        acc[0] = acc[0] + a
        acc[1] = acc[1] + b
        acc[2] = acc[2] + _f32(a * b)
        acc[3] = acc[3] + _f32(a * a)
    sx, sy, sxy, sxx = acc
    nf = _f32(float(n))
    m = _f32(_f32(_f32(nf * sxy) - _f32(sx * sy)) / _f32(_f32(nf * sxx) - _f32(sx * sx)))
    b = _f32(_f32(sy - _f32(m * sx)) / nf)
    return m, b


@dataclass(frozen=True)
class EncyclopediaFit:
    """The jar's fit of one point set.

    ``model`` is ``kde``, ``linear`` (2–20 points) or ``identity``.
    ``band`` holds the float32 bounds of the middle 95 % of residuals.
    For the KDE, ``sigma`` and ``radius`` are the kernel σ and stamp
    radius in cells, ``apex`` the walk's start cell.
    """

    curve: JarLinearFunction | JarLinearRegression
    model: Literal["kde", "linear", "identity"]
    band: tuple[float, float]
    resolution: int
    sigma: float | None = None
    radius: int | None = None
    apex: tuple[int, int] | None = None

    def warp(self, x: torch.Tensor) -> torch.Tensor:
        """``getYValue``: where the fit puts each library time."""
        return self.curve.forward(x)

    def delta(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``(float) y − warp(x)`` in float32, as the sidecar's ``delta``."""
        return (_to_f32(y) - self.warp(x).to(torch.float32)).to(torch.float64)

    def within_band(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The sidecar's ``within95p``."""
        d = self.delta(x, y)
        return torch.isfinite(d) & (d >= self.band[0]) & (d <= self.band[1])


def _band(deltas: torch.Tensor, coverage: float) -> tuple[float, float]:
    """``getMiddleDeltas``."""
    d = deltas[torch.isfinite(deltas)]
    if d.numel() == 0:
        return -math.inf, math.inf
    d = torch.sort(d).values
    n = d.numel()
    t = (1.0 - _f32(coverage)) / 2.0
    lo = min(n - 1, max(0, math.floor((n - 1) * t)))
    hi = min(n - 1, max(0, math.ceil((n - 1) * (1.0 - t))))
    return float(d[lo]), float(d[hi])


def fit_encyclopedia(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    resolution: int = RESOLUTION,
    coverage: float = 0.95,
    backend: Literal["exact", "fft"] = "exact",
) -> EncyclopediaFit:
    """Fit ``y ≈ f(x)`` exactly as EncyclopeDIA 6.9.23 does.

    ``x`` and ``y`` are the points in the order the jar would see them
    (for a search, its ``rt_fit.txt`` rows), library and observed time
    in minutes. ``backend="fft"`` builds the grid by FFT instead of
    stamping point by point: faster at high resolution, not bit-exact.
    """
    xs = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    ys = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    if xs.numel() != ys.numel():
        raise ValueError("x and y must have the same length")
    n = xs.numel()
    x32, y32 = xs.to(torch.float32), ys.to(torch.float32)

    if n < MIN_KDE_POINTS:
        if n <= 1:
            # The jar's identity is the regression through (0, 0) and
            # (1, 1), unclamped; its domain is where the point is, so
            # that extrapolation tails do not cut the identity short.
            m, b, model = 1.0, 0.0, "identity"
            dom = (float(x32[0]), float(x32[0])) if n == 1 else (0.0, 0.0)
        else:
            m, b = _regression([float(v) for v in x32.tolist()], [float(v) for v in y32.tolist()])
            model = "linear"
            dom = (float(x32.min()), float(x32.max()))
        curve = JarLinearRegression(m, b, dom)
        deltas = (y32 - curve.forward(x32).to(torch.float32)).to(torch.float64)
        return EncyclopediaFit(curve, model, _band(deltas, coverage), resolution)

    x_range, y_range = _Range(x32), _Range(y32)
    xi = x_range.index(x32, resolution)
    yi = y_range.index(y32, resolution)
    sigma = _sigma(xi, yi, n, resolution)
    stamp = _stamp(sigma)
    grid = _histogram(xi, yi, stamp, resolution, backend)
    x_values = x_range.values(resolution).to(torch.float64)
    y_values = y_range.values(resolution).to(torch.float64)
    tol = {} if backend == "exact" else {"tie_rtol": 1e-6}
    i, j = greedy_walk(grid, x_values=x_values, y_values=y_values, **tol)
    # Knots sort by (x, y) on the mapped values, as Collections.sort does.
    kx, ky = x_values[i], y_values[j]
    order = sorted(range(kx.numel()), key=lambda k: (float(kx[k]), float(ky[k])))
    curve = JarLinearFunction(kx[order], ky[order])
    deltas = (y32 - curve.forward(xs).to(torch.float32)).to(torch.float64)
    apex = (int(torch.argmax(grid.reshape(-1))) // resolution, int(torch.argmax(grid.reshape(-1))) % resolution)
    return EncyclopediaFit(
        curve,
        "kde",
        _band(deltas, coverage),
        resolution,
        sigma=sigma,
        radius=stamp.shape[0] // 2,
        apex=apex,
    )
