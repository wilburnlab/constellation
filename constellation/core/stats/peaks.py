"""Observable peak shapes — `PeakShape` ABC subclasses.

Peak shapes parameterize the time-dependent count `N(t)` underlying a
chromatographic / electrophoretic peak. Distinct from probability
densities (`distributions.py`): peaks model observed intensity, not
samples drawn from a normalized density. They share the `Parametric`
umbrella but split below it: peaks have `.integrate()` / `.bounds()`
and may not have a clean `.log_prob()` / `.cdf()` (HyperEMG ships
without one; Warped / Spline deferred).

v1 ships:
    GaussianPeak    3 params: log_N_max, t_apex, log_sigma
    EMGPeak         4 params: log_N_max, t_apex, log_sigma, log_tau
    HyperEMGPeak    6 params: log_N_total, mu, log_sigma, log_tau_r,
                    log_tau_l, logit_eta — weighted left+right EMG
                    mixture, parameterized by total area; fit via
                    `core.optim.DifferentialEvolution` (LBFGS won't).

Deferred (need DE optimizer in `core.optim` — LBFGS won't fit them):
    WarpedEMGPeak   piecewise cubic Hermite refinement of an EMG fit
    SplinePeak      B-spline knots, fit by reference-shape minimization

Numerical conventions:
    - All ops in float64 by default — peak fitting is precision-sensitive.
    - Positive parameters live in log-space; locations stay in real space.
    - The EMG closed form involves `exp(a)·erfc(b)` where both args may
      be large; we use `erfcx(b) = exp(b²)·erfc(b)` so the product
      becomes `exp(a-b²)·erfcx(b)` and stays in range. The algebraic
      identity `a - b² = -(t-μ)²/(2σ²)` is what makes this work.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .parametric import PeakShape

__all__ = [
    "GaussianPeak",
    "EMGPeak",
    "HyperEMGPeak",
    "emg_pdf",
    "emg_log_pdf",
    "emg_left_pdf",
    "emg_left_log_pdf",
    "emg_cdf",
    "emg_left_cdf",
    "gaussian_peak",
    "emg_peak",
    "PeakForm",
    "PEAK_FORMS",
]

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# ──────────────────────────────────────────────────────────────────────
# Numerical helpers
# ──────────────────────────────────────────────────────────────────────


def _erfcx(x: torch.Tensor) -> torch.Tensor:
    """Scaled complementary error function: `erfcx(x) = exp(x²)·erfc(x)`.

    Stable for all `x`. Direct computation where `x < 25`; for `x ≥ 25`
    a three-term asymptotic series (Abramowitz & Stegun 7.1.23). Torch
    does not ship `erfcx` natively as of 2.11.

    Uses `torch.where` rather than boolean indexing so the function is
    `torch.func.vmap`-compatible (DE's population path vmaps the model;
    boolean masking with dynamic shape is rejected by vmap). The
    `clamp(max=25)` keeps `exp(x²)` from overflowing on the unused
    branch — `torch.where` evaluates both branches before selecting.
    """
    x_clamped = x.clamp(max=25.0)
    direct = torch.exp(x_clamped**2) * torch.erfc(x)
    # asymptotic branch: divisions need a non-zero denominator on the
    # unused (small-x) side, otherwise inf/nan can leak through autograd.
    x_safe = torch.where(x > 1e-10, x, torch.full_like(x, 1.0))
    inv_x2 = 1.0 / (x_safe**2)
    asymp = (1.0 / (x_safe * math.sqrt(math.pi))) * (
        1.0 - 0.5 * inv_x2 + 0.75 * inv_x2**2
    )
    return torch.where(x < 25.0, direct, asymp)


def _log_erfcx(x: torch.Tensor) -> torch.Tensor:
    """Log of the scaled complementary error function, `log(erfcx(x))`, stable for
    ALL `x` — including large negative `x`, where `erfcx` itself overflows.

    For `x ≥ 0` `erfcx(x)` is bounded, so `log(_erfcx(x))` is taken directly. For
    `x < 0`, `erfcx(x) = exp(x²)·erfc(x)` overflows to `inf`, but
    `log(erfcx(x)) = x² + log(erfc(x))` with `erfc(x) ∈ (1, 2]` is finite. Using
    this branch (rather than `log` of an overflowed `_erfcx`) is what lets the EMG
    tail avoid the `0·inf` trap — an underflowed Gaussian times an overflowed
    `erfcx` — so the densities stay finite into the far tails (e.g. a HyperEMG with
    `σ ≪ τ` evaluated over its own `bounds()`). `torch.where` with substituted-safe
    inputs keeps both branches NaN-free under autograd / vmap (the `_erfcx`
    convention).
    """
    x_pos = torch.where(x >= 0.0, x, torch.ones_like(x))
    x_neg = torch.where(x < 0.0, x, -torch.ones_like(x))
    log_pos = torch.log(_erfcx(x_pos).clamp(min=1e-300))
    log_neg = x_neg * x_neg + torch.log(torch.erfc(x_neg))
    return torch.where(x >= 0.0, log_pos, log_neg)


def _emg_mode_offset(sigma: torch.Tensor, tau: torch.Tensor, *, iterations: int = 40) -> torch.Tensor:
    """Where the right-tailed EMG peaks, in units of σ from the Gaussian
    centre: ``(t_mode − μ) / σ``.

    With ``z = (t − μ)/σ``, ``r = σ/τ`` and ``w = (r − z)/√2`` the log
    density is ``−z²/2 + log erfcx(w)`` up to a constant, and setting its
    derivative to zero reduces to ``erfcx(w*) = √(2/π)/r``. ``erfcx`` is
    strictly decreasing, so the root is unique; the mode is
    ``z* = r − √2·w*``. It goes to ``1/r`` (μ + τ) as τ → 0 and to 0 as
    τ → ∞.

    Newton's method on ``g(w) = log erfcx(w) − log c`` (convex and
    decreasing, so once left of the root it closes in from the left)
    runs a fixed number of steps on detached inputs; one more step on the
    live inputs gives the implicit-function derivative, in either mode. Stays finite for
    any ``w`` through :func:`_log_erfcx` and is ``vmap``-safe.
    """
    r = sigma / tau
    log_c = 0.5 * math.log(2.0 / math.pi) - torch.log(r)

    def step(w: torch.Tensor, log_c: torch.Tensor) -> torch.Tensor:
        log_e = _log_erfcx(w)
        # d/dw log erfcx(w) = 2w − 2/(√π·erfcx(w)), with 1/erfcx from its log.
        slope = 2.0 * w - (2.0 / math.sqrt(math.pi)) * torch.exp(-log_e)
        return w - (log_e - log_c) / slope

    # The iterations run on detached inputs, so no derivative of either
    # mode (backward, or forward through ``torch.func.jvp``) is carried
    # through them; the last step, on the live ``log_c``, carries the
    # implicit-function derivative.
    log_c_fixed = log_c.detach()
    c = torch.exp(log_c_fixed)
    # Large w: erfcx(w) ≈ 1/(w√π); very negative w: erfcx(w) ≈ 2·exp(w²).
    w = torch.where(
        c < 1.0,
        1.0 / (c.clamp(min=1e-300) * math.sqrt(math.pi)),
        -torch.sqrt(torch.log((c / 2.0).clamp(min=1.0))),
    )
    for _ in range(iterations):
        w = step(w, log_c_fixed)
    w = step(w.detach(), log_c)
    return r - _SQRT2 * w


def emg_pdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Exponentially-modified Gaussian density. Free function — used
    by `core.signal` parametric peak fits and by `EMGPeak`.

    `f(t) = (1/2τ)·exp(-(t-μ)²/(2σ²))·erfcx(σ/(τ√2) - z/√2)`
    where `z = (t-μ)/σ`. Evaluated in log-space (`_log_erfcx`) so the far tail
    stays finite instead of forming `0·inf` (underflowed Gaussian × overflowed
    `erfcx`).
    """
    return torch.exp(emg_log_pdf(t, mu, sigma, tau))


def emg_log_pdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Log of `emg_pdf`, numerically stable for fitting under NLL — and into the
    tails: `_log_erfcx` keeps `-z²/2 + log erfcx(b)` finite where `erfcx`
    overflows."""
    z = (t - mu) / sigma
    b = sigma / (tau * _SQRT2) - z / _SQRT2
    return -math.log(2.0) - torch.log(tau) - 0.5 * z * z + _log_erfcx(b)


def emg_left_pdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Left-tailing (fronting) EMG density — the mirror of `emg_pdf`, with the
    exponential tail extending to the *left* of the Gaussian.

    `f(t) = (1/2τ)·exp(-z²/2)·erfcx(σ/(τ√2) + z/√2)`, `z = (t-μ)/σ`. The only
    change from `emg_pdf` is the `+ z/√2` (vs `- z/√2`), which reflects the
    tail about μ; like `emg_pdf` it integrates to 1. Evaluated in log-space so a
    wide left tail (`σ ≪ τ`) stays finite far out.
    """
    return torch.exp(emg_left_log_pdf(t, mu, sigma, tau))


def emg_left_log_pdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Log of `emg_left_pdf`, numerically stable for fitting under NLL — and into
    the tails (see `_log_erfcx`)."""
    z = (t - mu) / sigma
    b = sigma / (tau * _SQRT2) + z / _SQRT2
    return -math.log(2.0) - torch.log(tau) - 0.5 * z * z + _log_erfcx(b)


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF Φ(z), from ``erfc`` so the lower tail keeps its
    relative precision."""
    return 0.5 * torch.erfc(-z / _SQRT2)


def emg_cdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """CDF of the right-tailed EMG density :func:`emg_pdf`, exactly:
    ``F(t) = Φ(z) − τ·f(t)`` with ``z = (t − μ)/σ``. The density is taken
    from its stable log form, so the CDF is finite into both tails, and it
    broadcasts and is ``vmap``-safe like the densities. ``t = ±inf`` gives
    0 and 1."""
    t = torch.as_tensor(t, dtype=torch.as_tensor(mu).dtype)
    finite = torch.isfinite(t)
    t_safe = torch.where(finite, t, torch.zeros_like(t))
    z = (t_safe - mu) / sigma
    value = _normal_cdf(z) - tau * emg_pdf(t_safe, mu, sigma, tau)
    value = value.clamp(min=0.0, max=1.0)
    limit = torch.where(t > 0, torch.ones_like(value), torch.zeros_like(value))
    return torch.where(finite, value, limit)


def emg_left_cdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """CDF of the left-tailed (fronting) EMG density :func:`emg_left_pdf`:
    the mirror of :func:`emg_cdf`, ``F(t) = Φ(z) + τ·f_L(t)``."""
    t = torch.as_tensor(t, dtype=torch.as_tensor(mu).dtype)
    finite = torch.isfinite(t)
    t_safe = torch.where(finite, t, torch.zeros_like(t))
    z = (t_safe - mu) / sigma
    value = _normal_cdf(z) + tau * emg_left_pdf(t_safe, mu, sigma, tau)
    value = value.clamp(min=0.0, max=1.0)
    limit = torch.where(t > 0, torch.ones_like(value), torch.zeros_like(value))
    return torch.where(finite, value, limit)


def _limits(
    t_lo: float | torch.Tensor | None, t_hi: float | torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integration limits as float64 tensors, a missing one as ``∓inf``."""
    lo = torch.tensor(-math.inf, dtype=torch.float64) if t_lo is None else torch.as_tensor(t_lo, dtype=torch.float64)
    hi = torch.tensor(math.inf, dtype=torch.float64) if t_hi is None else torch.as_tensor(t_hi, dtype=torch.float64)
    return lo, hi


def gaussian_peak(t: torch.Tensor, height: torch.Tensor, center: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """A Gaussian peak of ``height`` at ``center``: ``height·exp(−½((t − center)/σ)²)``.
    The one formula ``GaussianPeak`` and the batched ``gaussian`` form use."""
    z = (t - center) / sigma
    return height * torch.exp(-0.5 * z * z)


def emg_peak(t: torch.Tensor, height: torch.Tensor, apex: torch.Tensor, sigma: torch.Tensor,
             tau: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """A right-tailed EMG peak whose maximum is ``height`` at ``apex`` (the
    exact mode, :func:`_emg_mode_offset`), its ratio to the apex taken in
    log space so the far tail stays finite. The one formula ``EMGPeak`` and
    the batched ``emg`` form use."""
    mu = apex - sigma * _emg_mode_offset(sigma, tau)
    return height * torch.exp(emg_log_pdf(t, mu, sigma, tau) - emg_log_pdf(apex, mu, sigma, tau))


def _moments(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor):
    """Per row: the positive part's centroid and spread, the height and the
    smallest sample spacing (for starting values and bounds)."""
    w = torch.where(valid, y.clamp(min=0), torch.zeros_like(y))
    total = w.sum(-1).clamp(min=1e-300)
    xs = torch.where(valid, x, torch.zeros_like(x))
    center = (w * xs).sum(-1) / total
    spread = torch.sqrt(((w * (xs - center.unsqueeze(-1)) ** 2).sum(-1) / total).clamp(min=0))
    height = torch.where(valid, y, torch.full_like(y, float("-inf"))).amax(-1).clamp(min=0)
    big = torch.full_like(x, float("inf"))
    gaps = torch.where(valid[..., 1:] & valid[..., :-1], (x[..., 1:] - x[..., :-1]).abs(), big[..., 1:])
    step = gaps.amin(-1) if gaps.shape[-1] else torch.ones_like(center)
    step = torch.where(torch.isfinite(step) & (step > 0), step, torch.ones_like(step))
    lo = torch.where(valid, x, big).amin(-1)
    hi = torch.where(valid, x, -big).amax(-1)
    return center, spread.clamp(min=step), height, step, lo, hi


@dataclass(frozen=True)
class PeakForm:
    """A peak shape as functions of batched parameters ``theta [B, P]``, for
    fitting many peaks at once (``core.signal.peak_fit``): its value at
    ``x [B, N]``, its whole area, its apex position, starting values and
    box bounds from the data of each row. Named in :data:`PEAK_FORMS`."""

    name: str
    version: int
    parameters: tuple[str, ...]
    evaluate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    area: Callable[[torch.Tensor], torch.Tensor]
    apex: Callable[[torch.Tensor], torch.Tensor]
    initial: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    bounds: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _gauss_init(x, y, valid):
    center, spread, height, _, _, _ = _moments(x, y, valid)
    return torch.stack([height, center, spread], -1)


def _gauss_bounds(x, y, valid):
    _, _, height, step, lo, hi = _moments(x, y, valid)
    lower = torch.stack([torch.zeros_like(lo), lo, step / 4], -1)
    upper = torch.stack([10 * height.clamp(min=1e-300), hi, (hi - lo).clamp(min=step)], -1)
    return lower, upper


def _emg_init(x, y, valid):
    center, spread, height, _, _, _ = _moments(x, y, valid)
    apex = torch.gather(x, -1, torch.where(valid, y, torch.full_like(y, float("-inf"))).argmax(-1, keepdim=True))
    return torch.stack([height, apex.squeeze(-1), spread / 1.5, spread / 2], -1)


def _emg_bounds(x, y, valid):
    _, _, height, step, lo, hi = _moments(x, y, valid)
    span = (hi - lo).clamp(min=step)
    lower = torch.stack([torch.zeros_like(lo), lo, step / 4, step / 20], -1)
    upper = torch.stack([10 * height.clamp(min=1e-300), hi, span, 2 * span], -1)
    return lower, upper


def _emg_area(theta):
    h, apex, s, tau = theta.unbind(-1)
    mu = apex - s * _emg_mode_offset(s, tau)
    return h * torch.exp(-emg_log_pdf(apex, mu, s, tau))


#: The batched peak forms, by name.
PEAK_FORMS: dict[str, PeakForm] = {
    "gaussian": PeakForm(
        name="gaussian", version=1, parameters=("height", "center", "sigma"),
        evaluate=lambda x, th: gaussian_peak(x, th[..., 0:1], th[..., 1:2], th[..., 2:3]),
        area=lambda th: th[..., 0] * th[..., 2] * _SQRT2PI,
        apex=lambda th: th[..., 1],
        initial=_gauss_init, bounds=_gauss_bounds,
    ),
    "emg": PeakForm(
        name="emg", version=1, parameters=("height", "apex", "sigma", "tau"),
        evaluate=lambda x, th: emg_peak(x, th[..., 0:1], th[..., 1:2], th[..., 2:3], th[..., 3:4]),
        area=_emg_area,
        apex=lambda th: th[..., 1],
        initial=_emg_init, bounds=_emg_bounds,
    ),
}  # fmt: skip


# ──────────────────────────────────────────────────────────────────────
# Peak shapes
# ──────────────────────────────────────────────────────────────────────


class GaussianPeak(PeakShape):
    """Gaussian peak: `N(t) = N_max · exp(-½((t-t_apex)/σ)²)`.

    Params: `log_N_max`, `t_apex`, `log_sigma`. Three parameters; the
    simplest shape, useful as a baseline or fallback when the trailing
    edge is too short to constrain `τ`.
    """

    def __init__(
        self, N_max: float = 1.0, t_apex: float = 0.0, sigma: float = 1.0
    ) -> None:
        super().__init__()
        self.log_N_max = nn.Parameter(
            torch.tensor(math.log(N_max), dtype=torch.float64)
        )
        self.t_apex = nn.Parameter(torch.tensor(t_apex, dtype=torch.float64))
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma), dtype=torch.float64)
        )

    @property
    def N_max(self) -> torch.Tensor:
        return self.log_N_max.exp()

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return gaussian_peak(t, self.N_max, self.t_apex, self.sigma)

    def integrate(
        self,
        t_lo: float | torch.Tensor | None = None,
        t_hi: float | torch.Tensor | None = None,
        n_points: int = 500,
    ) -> torch.Tensor:
        # Closed form when no limits are given: ∫ N_max·exp(-½z²) dt = N_max·σ·√(2π)
        if t_lo is None and t_hi is None:
            return self.N_max * self.sigma * _SQRT2PI
        # Bounded integral via the Gaussian CDF; a missing limit is ∓inf.
        lo, hi = _limits(t_lo, t_hi)
        sigma = self.sigma
        cdf_hi = _normal_cdf((hi - self.t_apex) / sigma)
        cdf_lo = _normal_cdf((lo - self.t_apex) / sigma)
        return self.N_max * sigma * _SQRT2PI * (cdf_hi - cdf_lo)

    def bounds(self, n_sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma
        return (self.t_apex - n_sigma * sigma, self.t_apex + n_sigma * sigma)

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "N_max": self.N_max.detach(),
            "t_apex": self.t_apex.detach(),
            "sigma": self.sigma.detach(),
        }


class EMGPeak(PeakShape):
    """Exponentially-modified Gaussian — Gaussian convolved with a
    right-exponential decay. Captures the standard chromatographic
    tailing pattern.

    Params: `log_N_max`, `t_apex`, `log_sigma`, `log_tau`.

    `t_apex` is the *observed peak maximum* (where `N` reaches `N_max`),
    not the underlying Gaussian centroid `μ`. We parameterize on the
    observable apex and solve back to `μ` internally (the exact mode,
    :func:`_emg_mode_offset`) so fits stay well-conditioned.
    """

    def __init__(
        self,
        N_max: float = 1.0,
        t_apex: float = 0.0,
        sigma: float = 1.0,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.log_N_max = nn.Parameter(
            torch.tensor(math.log(N_max), dtype=torch.float64)
        )
        self.t_apex = nn.Parameter(torch.tensor(t_apex, dtype=torch.float64))
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma), dtype=torch.float64)
        )
        self.log_tau = nn.Parameter(
            torch.tensor(math.log(tau), dtype=torch.float64)
        )

    @property
    def N_max(self) -> torch.Tensor:
        return self.log_N_max.exp()

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def mu(self) -> torch.Tensor:
        """Underlying Gaussian centroid: ``t_apex`` less the exact distance
        from centre to mode (:func:`_emg_mode_offset`). Until 2026-11-07
        this was ``t_apex − σ²/τ``, which put the curve's maximum away
        from ``t_apex`` (by several σ when τ ≪ σ)."""
        return self.t_apex - self.sigma * _emg_mode_offset(self.sigma, self.tau)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # N_max at t_apex: the density over its value there, in log space
        # (emg_peak; until 2026-11-11 a ratio of densities clamped at 1e-30).
        return emg_peak(t, self.N_max, self.t_apex, self.sigma, self.tau)

    def integrate(
        self,
        t_lo: float | torch.Tensor | None = None,
        t_hi: float | torch.Tensor | None = None,
        n_points: int = 500,
    ) -> torch.Tensor:
        """Area under the peak over ``[t_lo, t_hi]``, exactly: the EMG
        density integrates to 1, so the whole area is ``N_max / f(t_apex)``
        and a bounded one that times the CDF difference (:func:`emg_cdf`).
        A missing limit is ``∓inf``. Differentiable in the parameters and
        the limits; ``n_points`` is unused. Until 2026-11-11 this was a
        500-point trapezoid over ``bounds()`` by default, which cut the
        tail off (−1.7 % of the area at τ = 5σ, −3.7 % at 20σ) and, through
        ``.item()``, stopped gradients through the limits."""
        sigma, tau, mu = self.sigma, self.tau, self.mu
        area = self.N_max / emg_pdf(self.t_apex, mu, sigma, tau).clamp(min=1e-300)
        if t_lo is None and t_hi is None:
            return area
        lo, hi = _limits(t_lo, t_hi)
        return area * (emg_cdf(hi, mu, sigma, tau) - emg_cdf(lo, mu, sigma, tau))

    def bounds(self, n_sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma
        tau = self.tau
        # Asymmetric: leading edge ~ Gaussian; trailing edge stretches
        # by ~3τ before decaying to noise.
        return (
            self.t_apex - n_sigma * sigma,
            self.t_apex + n_sigma * sigma + 3.0 * tau,
        )

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "N_max": self.N_max.detach(),
            "t_apex": self.t_apex.detach(),
            "sigma": self.sigma.detach(),
            "tau": self.tau.detach(),
            "mu": self.mu.detach(),
        }


class HyperEMGPeak(PeakShape):
    """Hyper-EMG(1,1): a Gaussian convolved with independent left and right
    exponential tails.

    `N(t) = N_total · [η·emg_pdf(t; μ,σ,τ_r) + (1−η)·emg_left_pdf(t; μ,σ,τ_l)]`

    Both mixture components are proper PDFs (integrate to 1), so
    `∫ N(t) dt = N_total` for *any* shape — the integrated ion count is a direct
    parameter (contrast `EMGPeak`, which parameterizes the apex height `N_max`).
    With `η → 1` and a small `τ_l` this reduces to a right-tailing EMG. Its CDF
    is the η mixture of :func:`emg_cdf` and :func:`emg_left_cdf`; the η mixture
    is multimodal, so fit with
    `core.optim.DifferentialEvolution` (+ optional LBFGS polish), not LBFGS
    alone.

    Params: `log_N_total`, `mu` (the Gaussian center, ≈ apex for a symmetric
    peak), `log_sigma`, `log_tau_r`, `log_tau_l`, `logit_eta`
    (η = sigmoid(logit_eta) ∈ (0, 1) is the right-tail mixture weight).
    """

    def __init__(
        self,
        N_total: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        tau_r: float = 1.0,
        tau_l: float = 1.0,
        eta: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0.0 < eta < 1.0:
            raise ValueError(f"eta must be in (0, 1); got {eta}")
        self.log_N_total = nn.Parameter(
            torch.tensor(math.log(N_total), dtype=torch.float64)
        )
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float64))
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma), dtype=torch.float64)
        )
        self.log_tau_r = nn.Parameter(
            torch.tensor(math.log(tau_r), dtype=torch.float64)
        )
        self.log_tau_l = nn.Parameter(
            torch.tensor(math.log(tau_l), dtype=torch.float64)
        )
        # logit parameterization: eta = sigmoid(logit_eta).
        self.logit_eta = nn.Parameter(
            torch.tensor(math.log(eta / (1.0 - eta)), dtype=torch.float64)
        )

    @property
    def N_total(self) -> torch.Tensor:
        return self.log_N_total.exp()

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    @property
    def tau_r(self) -> torch.Tensor:
        return self.log_tau_r.exp()

    @property
    def tau_l(self) -> torch.Tensor:
        return self.log_tau_l.exp()

    @property
    def eta(self) -> torch.Tensor:
        """Right-tail mixture weight η ∈ (0, 1)."""
        return torch.sigmoid(self.logit_eta)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # vmap-safe: only broadcasting ops + the `torch.where`-based `_erfcx`,
        # so DE's population (vmap'd) closure can evaluate it.
        eta = self.eta
        right = emg_pdf(t, self.mu, self.sigma, self.tau_r)
        left = emg_left_pdf(t, self.mu, self.sigma, self.tau_l)
        return self.N_total * (eta * right + (1.0 - eta) * left)

    def integrate(
        self,
        t_lo: float | torch.Tensor | None = None,
        t_hi: float | torch.Tensor | None = None,
        n_points: int = 500,
    ) -> torch.Tensor:
        """Area over ``[t_lo, t_hi]``, exactly: the mixture components are
        normalized densities, so the whole area is ``N_total`` and a
        bounded one is ``N_total`` times the mixture's CDF difference
        (:func:`emg_cdf`, :func:`emg_left_cdf`). A missing limit is
        ``∓inf``; ``n_points`` is unused. Until 2026-11-11 a bounded area
        was a trapezoid, with a missing limit taken from ``bounds()``."""
        if t_lo is None and t_hi is None:
            return self.N_total
        lo, hi = _limits(t_lo, t_hi)
        eta = self.eta

        def cdf(t: torch.Tensor) -> torch.Tensor:
            return eta * emg_cdf(t, self.mu, self.sigma, self.tau_r) + (1.0 - eta) * emg_left_cdf(
                t, self.mu, self.sigma, self.tau_l
            )

        return self.N_total * (cdf(hi) - cdf(lo))

    def bounds(self, n_sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma
        # Asymmetric on BOTH sides: a fronting tail (τ_l) to the left and a
        # tailing tail (τ_r) to the right.
        return (
            self.mu - n_sigma * sigma - 3.0 * self.tau_l,
            self.mu + n_sigma * sigma + 3.0 * self.tau_r,
        )

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "N_total": self.N_total.detach(),
            "mu": self.mu.detach(),
            "sigma": self.sigma.detach(),
            "tau_r": self.tau_r.detach(),
            "tau_l": self.tau_l.detach(),
            "eta": self.eta.detach(),
        }

    @classmethod
    def from_emg(cls, emg: "EMGPeak") -> "HyperEMGPeak":
        """Warm-start a HyperEMG from a fitted `EMGPeak`: mostly right-tailing
        (η≈0.95), `τ_l = τ_r`, total area `N_total = N_max / emg_pdf(t_apex)`."""
        with torch.no_grad():
            mu = emg.mu
            pdf_apex = emg_pdf(emg.t_apex, mu, emg.sigma, emg.tau).clamp(min=1e-30)
            n_total = (emg.N_max / pdf_apex).clamp(min=1e-30)
        model = cls(eta=0.95)
        with torch.no_grad():
            model.log_N_total.copy_(torch.log(n_total))
            model.mu.copy_(mu)
            model.log_sigma.copy_(emg.log_sigma.detach())
            model.log_tau_r.copy_(emg.log_tau.detach())
            model.log_tau_l.copy_(emg.log_tau.detach())
        return model


# Reserved for the next peak-numerics session (need DE — now shipped):
#   - WarpedEMGPeak    piecewise cubic Hermite refinement of an EMG fit
#   - SplinePeak       B-spline knots, fit by reference-shape minimization
