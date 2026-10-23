"""Monotone curves: invertible, knot-based maps of one axis onto another.

Calibration curves that must never turn back — retention time onto
retention time, migration time onto fragment size — share one
interface (:class:`MonotoneCurve`): ``forward``, ``inverse``,
``derivative``, the ``domain`` the curve was built on and the ``image``
of that domain, and ``knots()``, one serializable form for every kind.

  ``PiecewiseLinear``          through given knots; non-decreasing, C0.
                               Ties are averaged (equal x → mean y for
                               the forward map, equal y → mean x for the
                               inverse), so a staircase still inverts.
  ``LinearMap``                a line, for fits too small for anything
                               else.
  ``RationalQuadraticSpline``  monotone rational-quadratic spline
                               (Gregory & Delbourgo 1982; the transform
                               of Neural Spline Flows, Durkan et al.
                               2019): strictly increasing, C1, with a
                               closed-form inverse. A ``Parametric``, so
                               it can be fitted; ``from_knots`` builds
                               one through given knots and slopes, which
                               is monotone for any positive slopes (so
                               it replaces a monotone cubic Hermite).

Outside its domain each curve applies its own continuation (clamp for
the piecewise-linear, the boundary slope for the spline). What a caller
wants beyond the calibration range is a separate decision,
:class:`Extrapolation`, applied by wrapping a curve in
:class:`TailedCurve`: hold the end value (``clamp``), continue the end
slope (``linear``), or continue a slope shrunk toward a prior slope for
a bounded margin and hold it beyond (``soft``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

from .parametric import Parametric

__all__ = [
    "Knots",
    "MonotoneCurve",
    "PiecewiseLinear",
    "LinearMap",
    "interp",
    "RationalQuadraticSpline",
    "steffen_slopes",
    "Extrapolation",
    "Tail",
    "TailedCurve",
]


def _f64(v) -> torch.Tensor:
    return torch.as_tensor(v, dtype=torch.float64)


@dataclass(frozen=True)
class Knots:
    """Knot positions, values and (where the curve has them) slopes.

    ``kind`` says how to interpolate between knots: ``linear``,
    ``rational_quadratic`` (needs ``slope``), or a producer-specific
    kind for curves with their own evaluation rules.
    """

    x: torch.Tensor
    y: torch.Tensor
    slope: torch.Tensor | None
    kind: str


@runtime_checkable
class MonotoneCurve(Protocol):
    """A non-decreasing map with an inverse."""

    @property
    def domain(self) -> tuple[float, float]: ...

    @property
    def image(self) -> tuple[float, float]: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def inverse(self, y: torch.Tensor) -> torch.Tensor: ...

    def derivative(self, x: torch.Tensor) -> torch.Tensor: ...

    def knots(self) -> Knots: ...


def interp(q: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear interpolation on strictly increasing ``xp``, clamped outside.

    The torch counterpart of ``numpy.interp``; ``fp`` need not be
    monotone.
    """
    q = _f64(q)
    if xp.numel() == 1:
        return fp[0].expand_as(q).clone()
    idx = torch.searchsorted(xp, q.contiguous(), right=True).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = ((q - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return f0 + t * (f1 - f0)


def _collapse(keys: torch.Tensor, vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge runs of equal ``keys`` (already sorted) into their mean ``vals``."""
    uniq, inverse, counts = torch.unique_consecutive(
        keys, return_inverse=True, return_counts=True
    )
    sums = torch.zeros_like(uniq).index_add_(0, inverse, vals)
    return uniq, sums / counts


class PiecewiseLinear:
    """Linear interpolation through non-decreasing knots.

    Knots are sorted by x. Runs of equal x collapse to their mean y and,
    for the inverse, runs of equal y to their mean x, so vertical and
    horizontal stretches (a walk on a grid makes both) neither break the
    forward map nor the inverse. Clamps outside its domain.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = _f64(x).reshape(-1)
        y = _f64(y).reshape(-1)
        if x.numel() == 0 or x.numel() != y.numel():
            raise ValueError("PiecewiseLinear needs equal-length, non-empty x and y")
        order = torch.argsort(x, stable=True)
        x, y = x[order], y[order]
        self._x, self._y = _collapse(x, y)
        if bool((torch.diff(self._y) < 0).any()):
            raise ValueError("PiecewiseLinear knots must be non-decreasing in y")
        self._yi, self._xi = _collapse(self._y, self._x)

    @property
    def domain(self) -> tuple[float, float]:
        return float(self._x[0]), float(self._x[-1])

    @property
    def image(self) -> tuple[float, float]:
        return float(self._y[0]), float(self._y[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return interp(x, self._x, self._y)

    __call__ = forward

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return interp(y, self._yi, self._xi)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        x = _f64(x)
        if self._x.numel() == 1:
            return torch.zeros_like(x)
        slopes = torch.diff(self._y) / torch.diff(self._x)
        idx = torch.searchsorted(self._x, x.contiguous(), right=True) - 1
        inside = (x >= self._x[0]) & (x <= self._x[-1])
        d = slopes[idx.clamp(0, slopes.numel() - 1)]
        return torch.where(inside, d, 0.0)

    def knots(self) -> Knots:
        return Knots(self._x.clone(), self._y.clone(), None, "linear")


class LinearMap:
    """``y = slope·x + intercept`` (``slope > 0``), with a nominal domain."""

    def __init__(self, slope: float, intercept: float, domain: tuple[float, float]) -> None:
        if not slope > 0:
            raise ValueError(f"LinearMap needs a positive slope, got {slope}")
        self.slope = float(slope)
        self.intercept = float(intercept)
        self._domain = (float(domain[0]), float(domain[1]))

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    @property
    def image(self) -> tuple[float, float]:
        lo, hi = self._domain
        return self.slope * lo + self.intercept, self.slope * hi + self.intercept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slope * _f64(x) + self.intercept

    __call__ = forward

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return (_f64(y) - self.intercept) / self.slope

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(_f64(x), self.slope)

    def knots(self) -> Knots:
        x = _f64(self._domain)
        return Knots(x, self.forward(x), torch.full_like(x, self.slope), "linear")


def steffen_slopes(
    x: torch.Tensor, y: torch.Tensor, *, min_fraction: float = 0.0
) -> torch.Tensor:
    """Knot slopes that keep a Hermite-type interpolant monotone.

    Steffen (1990): at an interior knot, the parabola through it and its
    neighbours gives a slope, limited to at most twice either adjacent
    secant and set to zero at a local extremum; the end slopes use the
    one-sided parabola under the same limit. For non-decreasing data
    every slope is ≥ 0; ``min_fraction`` floors them at that fraction of
    the overall secant (a rational-quadratic spline needs them
    positive).
    """
    x = _f64(x).reshape(-1)
    y = _f64(y).reshape(-1)
    if x.numel() < 2:
        raise ValueError("steffen_slopes needs at least two knots")
    h = torch.diff(x)
    s = torch.diff(y) / h
    if x.numel() == 2:
        d = s.repeat(2)
    else:
        p = (s[:-1] * h[1:] + s[1:] * h[:-1]) / (h[:-1] + h[1:])
        inner = (torch.sign(s[:-1]) + torch.sign(s[1:])) * torch.minimum(
            torch.minimum(s[:-1].abs(), s[1:].abs()), 0.5 * p.abs()
        )

        def end(s0, s1, h0, h1):
            p0 = s0 * (1 + h0 / (h0 + h1)) - s1 * h0 / (h0 + h1)
            if float(p0 * s0) <= 0:
                return torch.zeros_like(s0)
            if float(p0.abs()) > 2 * float(s0.abs()):
                return 2 * s0
            return p0

        d0 = end(s[0], s[1], h[0], h[1])
        dn = end(s[-1], s[-2], h[-1], h[-2])
        d = torch.cat([d0.reshape(1), inner, dn.reshape(1)])
    if min_fraction > 0:
        overall = (y[-1] - y[0]) / (x[-1] - x[0])
        d = d.clamp_min(min_fraction * float(overall))
    return d


def _inv_softplus(v: torch.Tensor) -> torch.Tensor:
    v = v.clamp_min(1e-12)
    return v + torch.log(-torch.expm1(-v))


class RationalQuadraticSpline(Parametric):
    """Monotone rational-quadratic spline on ``[x_lo, x_hi]``.

    ``n_bins`` bins whose widths and heights are softmax fractions of
    the domain and the image (each at least ``min_bin_fraction``), and
    ``n_bins + 1`` knot derivatives that are softplus multiples of the
    mean slope (each at least ``min_slope_fraction`` of it). In bin k,
    with ``ξ`` the position within the bin and ``s`` its secant::

        y = y_k + h_k (s ξ² + δ_k ξ(1−ξ)) / (s + (δ_{k+1} + δ_k − 2s) ξ(1−ξ))

    which is strictly increasing for positive ``δ``, C1 at the knots,
    and inverts in closed form (a quadratic in ξ). Beyond the domain it
    continues with the boundary derivative, so it is defined and
    differentiable everywhere; :class:`TailedCurve` sets a different
    policy. The domain is fixed; the image's offset and span are fitted.
    """

    def __init__(
        self,
        x_lo: float,
        x_hi: float,
        n_bins: int = 16,
        *,
        y_lo: float = 0.0,
        y_hi: float = 1.0,
        min_bin_fraction: float = 1e-3,
        min_slope_fraction: float = 1e-3,
    ) -> None:
        super().__init__()
        if not x_hi > x_lo:
            raise ValueError(f"need x_hi > x_lo, got [{x_lo}, {x_hi}]")
        if not y_hi > y_lo:
            raise ValueError(f"need y_hi > y_lo, got [{y_lo}, {y_hi}]")
        k = int(n_bins)
        if k < 1:
            raise ValueError(f"n_bins must be ≥ 1, got {n_bins}")
        self.n_bins = k
        self.min_width = min(float(min_bin_fraction), 0.5 / k)
        self.min_height = min(float(min_bin_fraction), 0.5 / k)
        self.min_slope = float(min_slope_fraction)
        self.register_buffer("x_lo", _f64(x_lo))
        self.register_buffer("x_hi", _f64(x_hi))
        self.theta_w = nn.Parameter(torch.zeros(k, dtype=torch.float64))
        self.theta_h = nn.Parameter(torch.zeros(k, dtype=torch.float64))
        unit = _inv_softplus(_f64(1.0 - self.min_slope))
        self.theta_d = nn.Parameter(unit.repeat(k + 1))
        self.y_lo = nn.Parameter(_f64(y_lo))
        self.log_y_span = nn.Parameter(_f64(math.log(y_hi - y_lo)))

    # -- construction ---------------------------------------------------

    @classmethod
    def from_knots(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        slopes: torch.Tensor,
        **kwargs,
    ) -> RationalQuadraticSpline:
        """The spline through ``(x, y)`` with derivative ``slopes`` there.

        Knots must be strictly increasing in x and non-decreasing in y;
        bins narrower or flatter than the minimum fractions, and slopes
        below the minimum, are raised to them.
        """
        x = _f64(x).reshape(-1)
        y = _f64(y).reshape(-1)
        slopes = _f64(slopes).reshape(-1)
        if x.numel() < 2 or not (x.numel() == y.numel() == slopes.numel()):
            raise ValueError("from_knots needs ≥ 2 knots with matching y and slopes")
        if bool((torch.diff(x) <= 0).any()):
            raise ValueError("from_knots needs strictly increasing x")
        k = x.numel() - 1
        model = cls(float(x[0]), float(x[-1]), k, y_lo=float(y[0]), y_hi=float(y[-1]), **kwargs)
        width, span = float(x[-1] - x[0]), float(y[-1] - y[0])

        def logits(frac: torch.Tensor, floor: float) -> torch.Tensor:
            free = ((frac - floor) / (1.0 - k * floor)).clamp_min(1e-12)
            return torch.log(free / free.sum())

        with torch.no_grad():
            model.theta_w.copy_(logits(torch.diff(x) / width, model.min_width))
            model.theta_h.copy_(logits(torch.diff(y) / span, model.min_height))
            mean_slope = span / width
            model.theta_d.copy_(_inv_softplus(slopes / mean_slope - model.min_slope))
        return model

    @classmethod
    def from_curve(
        cls, curve: MonotoneCurve, n_bins: int, *, n_dense: int = 4096, **kwargs
    ) -> RationalQuadraticSpline:
        """A spline with ``n_bins`` bins approximating ``curve`` on its domain.

        Bin edges fall at equal steps of ``(x − x_lo)/W + (y − y_lo)/H``
        along the curve (W, H the domain and image spans), so steep and
        shallow stretches get bins in proportion to how far the curve
        travels; knot slopes are Steffen's through the edges.
        """
        lo, hi = curve.domain
        xs = torch.linspace(lo, hi, n_dense, dtype=torch.float64)
        with torch.no_grad():
            ys = _f64(curve.forward(xs))
        width, span = hi - lo, float(ys[-1] - ys[0])
        if not (width > 0 and span > 0):
            raise ValueError("from_curve needs a curve with a non-degenerate domain and image")
        travelled = (xs - lo) / width + (ys - ys[0]) / span
        edges = interp(torch.linspace(0.0, 2.0, n_bins + 1, dtype=torch.float64), travelled, xs)
        edges[0], edges[-1] = lo, hi
        edges = torch.unique_consecutive(edges)
        with torch.no_grad():
            ye = _f64(curve.forward(edges))
        ye = torch.cummax(ye, 0).values
        slopes = steffen_slopes(edges, ye, min_fraction=1e-3)
        return cls.from_knots(edges, ye, slopes, **kwargs)

    # -- evaluation -----------------------------------------------------

    def _params(self):
        k = self.n_bins
        width = self.x_hi - self.x_lo
        span = torch.exp(self.log_y_span)
        w = width * (self.min_width + (1.0 - k * self.min_width) * torch.softmax(self.theta_w, -1))
        h = span * (self.min_height + (1.0 - k * self.min_height) * torch.softmax(self.theta_h, -1))
        xk = torch.cat([self.x_lo.reshape(1), self.x_lo + torch.cumsum(w, -1)[:-1], self.x_hi.reshape(1)])
        yk = torch.cat([self.y_lo.reshape(1), self.y_lo + torch.cumsum(h, -1)[:-1], (self.y_lo + span).reshape(1)])
        d = (span / width) * (self.min_slope + F.softplus(self.theta_d))
        return xk, yk, w, h, d

    def _bin(self, edges: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(edges[1:-1].contiguous(), v.contiguous(), right=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _f64(x)
        xk, yk, w, h, d = self._params()
        idx = self._bin(xk, x)
        x0, y0, wk, hk = xk[idx], yk[idx], w[idx], h[idx]
        d0, d1 = d[idx], d[idx + 1]
        s = hk / wk
        xi = ((x - x0) / wk).clamp(0.0, 1.0)
        mix = xi * (1.0 - xi)
        inside = y0 + hk * (s * xi * xi + d0 * mix) / (s + (d1 + d0 - 2.0 * s) * mix)
        below = yk[0] + d[0] * (x - xk[0])
        above = yk[-1] + d[-1] * (x - xk[-1])
        return torch.where(x < xk[0], below, torch.where(x > xk[-1], above, inside))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        x = _f64(x)
        xk, yk, w, h, d = self._params()
        idx = self._bin(xk, x)
        wk, hk = w[idx], h[idx]
        d0, d1 = d[idx], d[idx + 1]
        s = hk / wk
        xi = ((x - xk[idx]) / wk).clamp(0.0, 1.0)
        mix = xi * (1.0 - xi)
        beta = s + (d1 + d0 - 2.0 * s) * mix
        inside = s * s * (d1 * xi * xi + 2.0 * s * mix + d0 * (1.0 - xi) ** 2) / (beta * beta)
        return torch.where(x < xk[0], d[0], torch.where(x > xk[-1], d[-1], inside))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = _f64(y)
        xk, yk, w, h, d = self._params()
        idx = self._bin(yk, y)
        x0, y0, wk, hk = xk[idx], yk[idx], w[idx], h[idx]
        d0, d1 = d[idx], d[idx + 1]
        s = hk / wk
        dy = torch.minimum((y - y0).clamp_min(0.0), hk)
        c2 = d1 + d0 - 2.0 * s
        a = hk * (s - d0) + dy * c2
        b = hk * d0 - dy * c2
        c = -s * dy
        disc = (b * b - 4.0 * a * c).clamp_min(0.0)
        xi = 2.0 * c / (-b - torch.sqrt(disc))
        inside = x0 + xi * wk
        below = xk[0] + (y - yk[0]) / d[0]
        above = xk[-1] + (y - yk[-1]) / d[-1]
        return torch.where(y < yk[0], below, torch.where(y > yk[-1], above, inside))

    # -- description ----------------------------------------------------

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.x_lo), float(self.x_hi)

    @property
    def image(self) -> tuple[float, float]:
        lo = float(self.y_lo)
        return lo, lo + float(torch.exp(self.log_y_span))

    def knots(self) -> Knots:
        with torch.no_grad():
            xk, yk, _, _, d = self._params()
        return Knots(xk.clone(), yk.clone(), d.clone(), "rational_quadratic")

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            xk, yk, _, _, d = self._params()
        return {"x_knots": xk.clone(), "y_knots": yk.clone(), "slopes": d.clone()}


# -- extrapolation ------------------------------------------------------


@dataclass(frozen=True)
class Extrapolation:
    """What a curve does beyond its domain.

    ``clamp``   hold the end value.
    ``linear``  continue the end slope without bound.
    ``soft``    continue a slope shrunk toward the prior slope, for at
                most ``margin`` of the domain (or of the image, whichever
                is reached first), then hold. The end slope is the secant
                over the last ``window`` of the domain (default a tenth of
                it); the shrinkage weight toward it is ``n / (n + shrinkage)``
                with ``n`` the support weight inside that window, so a
                well-populated end keeps its own slope and a sparse one
                falls back to the prior.
    """

    mode: Literal["soft", "linear", "clamp"] = "soft"
    margin: float = 0.25
    shrinkage: float = 20.0
    window: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("soft", "linear", "clamp"):
            raise ValueError(f"extrapolation mode must be soft, linear or clamp; got {self.mode!r}")

    @classmethod
    def coerce(cls, value: Extrapolation | str) -> Extrapolation:
        return value if isinstance(value, Extrapolation) else cls(mode=value)


@dataclass(frozen=True)
class Tail:
    """One end of a :class:`TailedCurve`.

    From ``(x, y)`` the tail runs at ``slope`` for ``reach`` along x
    (``inf`` for no bound) and holds beyond. ``end_slope`` and
    ``support`` record what the slope was derived from.
    """

    x: float
    y: float
    slope: float
    reach: float
    end_slope: float
    support: float


class TailedCurve:
    """A monotone curve with an extrapolation policy beyond its domain."""

    def __init__(self, curve: MonotoneCurve, lower: Tail, upper: Tail, policy: Extrapolation) -> None:
        self.curve = curve
        self.lower = lower
        self.upper = upper
        self.policy = policy

    @classmethod
    def build(
        cls,
        curve: MonotoneCurve,
        policy: Extrapolation | str = "soft",
        *,
        prior_slope: float | None = None,
        x_support: torch.Tensor | None = None,
        w_support: torch.Tensor | None = None,
    ) -> TailedCurve:
        """Wrap ``curve`` with ``policy``.

        ``prior_slope`` defaults to the curve's secant over its domain.
        ``x_support`` / ``w_support`` are the points (and weights) the
        curve was fitted to, counted near each end for the soft
        shrinkage; without them the end slope is used unshrunk. A curve
        whose domain has no width gets unbounded tails, since a margin
        relative to the domain would be zero.
        """
        with torch.no_grad():
            return cls._build(curve, Extrapolation.coerce(policy), prior_slope, x_support, w_support)

    @classmethod
    def _build(cls, curve, policy, prior_slope, x_support, w_support) -> TailedCurve:
        lo, hi = curve.domain
        y_lo = float(curve.forward(_f64([lo]))[0])
        y_hi = float(curve.forward(_f64([hi]))[0])
        width, span = hi - lo, y_hi - y_lo
        if prior_slope is None:
            prior_slope = span / width if width > 0 and span > 0 else 1.0
        prior = float(prior_slope)
        if policy.mode == "clamp":
            return cls(
                curve,
                Tail(lo, y_lo, 0.0, math.inf, 0.0, 0.0),
                Tail(hi, y_hi, 0.0, math.inf, 0.0, 0.0),
                policy,
            )
        window = policy.window if policy.window is not None else 0.1 * width
        window = min(window, width)
        if window > 0:
            ends = curve.forward(_f64([lo + window, hi - window]))
            s_lo = (float(ends[0]) - y_lo) / window
            s_hi = (y_hi - float(ends[1])) / window
        else:
            s_lo = s_hi = prior
        s_lo = min(max(s_lo, prior / 50.0), prior * 50.0)
        s_hi = min(max(s_hi, prior / 50.0), prior * 50.0)

        def support(a: float, b: float) -> float:
            if x_support is None:
                return math.inf
            xs = _f64(x_support).reshape(-1)
            ws = torch.ones_like(xs) if w_support is None else _f64(w_support).reshape(-1)
            return float(ws[(xs >= a) & (xs <= b)].sum())

        n_lo = support(lo, lo + window)
        n_hi = support(hi - window, hi)

        def tail(x0: float, y0: float, s_end: float, n: float) -> Tail:
            if policy.mode == "linear":
                return Tail(x0, y0, s_end, math.inf, s_end, n)
            weight = 1.0 if math.isinf(n) else n / (n + policy.shrinkage)
            slope = math.exp(weight * math.log(s_end) + (1.0 - weight) * math.log(prior))
            if width > 0:
                reach = policy.margin * width
                if span > 0:
                    reach = min(reach, policy.margin * span / slope)
            else:
                reach = math.inf
            return Tail(x0, y0, slope, reach, s_end, n)

        return cls(curve, tail(lo, y_lo, s_lo, n_lo), tail(hi, y_hi, s_hi, n_hi), policy)

    @property
    def domain(self) -> tuple[float, float]:
        return self.curve.domain

    @property
    def image(self) -> tuple[float, float]:
        return self.lower.y, self.upper.y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _f64(x)
        lo, hi = self.lower.x, self.upper.x
        inner = _f64(self.curve.forward(x.clamp(lo, hi)))
        below = self.lower.y - self.lower.slope * (lo - x).clamp(0.0, self.lower.reach)
        above = self.upper.y + self.upper.slope * (x - hi).clamp(0.0, self.upper.reach)
        return torch.where(x < lo, below, torch.where(x > hi, above, inner))

    __call__ = forward

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = _f64(y)
        lo, hi = self.lower.x, self.upper.x
        inner = _f64(self.curve.inverse(y.clamp(self.lower.y, self.upper.y)))

        def run(tail: Tail, dy: torch.Tensor) -> torch.Tensor:
            if tail.slope <= 0:
                return torch.zeros_like(dy)
            return (dy / tail.slope).clamp(0.0, tail.reach)

        below = lo - run(self.lower, self.lower.y - y)
        above = hi + run(self.upper, y - self.upper.y)
        return torch.where(y < self.lower.y, below, torch.where(y > self.upper.y, above, inner))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        x = _f64(x)
        lo, hi = self.lower.x, self.upper.x
        inner = _f64(self.curve.derivative(x.clamp(lo, hi)))
        below = torch.where(lo - x <= self.lower.reach, self.lower.slope, 0.0)
        above = torch.where(x - hi <= self.upper.reach, self.upper.slope, 0.0)
        return torch.where(x < lo, below, torch.where(x > hi, above, inner))

    def is_extrapolated(self, x: torch.Tensor) -> torch.Tensor:
        """Beyond the domain the curve was built on."""
        x = _f64(x)
        return (x < self.lower.x) | (x > self.upper.x)

    def is_clamped(self, x: torch.Tensor) -> torch.Tensor:
        """Beyond the domain and past where the tail holds constant."""
        x = _f64(x)
        low = (x < self.lower.x) & ((self.lower.x - x > self.lower.reach) | (self.lower.slope == 0))
        high = (x > self.upper.x) & ((x - self.upper.x > self.upper.reach) | (self.upper.slope == 0))
        return low | high

    def knots(self) -> Knots:
        return self.curve.knots()
