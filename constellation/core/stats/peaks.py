"""Observable peak shapes — `PeakShape` ABC subclasses.

Peak shapes parameterize the time-dependent count `N(t)` underlying a
chromatographic / electrophoretic peak. Distinct from probability
densities (`distributions.py`): peaks model observed intensity, not
samples drawn from a normalized density. They share the `Parametric`
umbrella but split below it: peaks have `.integrate()` / `.bounds()`
and may not have a clean `.log_prob()` / `.cdf()` (HyperEMG, Warped,
Spline — deferred to a later session).

v1 ships:
    GaussianPeak    3 params: log_N_max, t_apex, log_sigma
    EMGPeak         4 params: log_N_max, t_apex, log_sigma, log_tau

Deferred (need DE optimizer in `core.optim` — LBFGS won't fit them):
    HyperEMGPeak    weighted left+right EMG mixture; no closed-form CDF
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

import torch
from torch import nn

from .parametric import PeakShape

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# ──────────────────────────────────────────────────────────────────────
# Numerical helpers
# ──────────────────────────────────────────────────────────────────────


def _erfcx(x: torch.Tensor) -> torch.Tensor:
    """Scaled complementary error function: `erfcx(x) = exp(x²)·erfc(x)`.

    Stable for all `x`. Direct computation where `x < 25`; for `x ≥ 25`
    a three-term asymptotic series (Abramowitz & Stegun 7.1.23). Torch
    does not ship `erfcx` natively as of 2.5.x.
    """
    safe = x < 25.0
    result = torch.empty_like(x)

    x_safe = x[safe]
    result[safe] = torch.exp(x_safe**2) * torch.erfc(x_safe)

    x_big = x[~safe]
    inv_x2 = 1.0 / (x_big**2)
    result[~safe] = (1.0 / (x_big * math.sqrt(math.pi))) * (
        1.0 - 0.5 * inv_x2 + 0.75 * inv_x2**2
    )
    return result


def emg_pdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Exponentially-modified Gaussian density. Free function — used
    by `core.signal` parametric peak fits and by `EMGPeak`.

    `f(t) = (1/2τ)·exp(-(t-μ)²/(2σ²))·erfcx(σ/(τ√2) - z/√2)`
    where `z = (t-μ)/σ`.
    """
    z = (t - mu) / sigma
    b = sigma / (tau * _SQRT2) - z / _SQRT2
    gauss = torch.exp(-0.5 * z * z)
    return (1.0 / (2.0 * tau)) * gauss * _erfcx(b)


def emg_log_pdf(
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Log of `emg_pdf`, numerically stable for fitting under NLL."""
    z = (t - mu) / sigma
    b = sigma / (tau * _SQRT2) - z / _SQRT2
    log_gauss = -0.5 * z * z
    log_erfcx = torch.log(_erfcx(b).clamp(min=1e-30))
    return -math.log(2.0) - torch.log(tau) + log_gauss + log_erfcx


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
        z = (t - self.t_apex) / self.sigma
        return self.N_max * torch.exp(-0.5 * z * z)

    def integrate(
        self,
        t_lo: float | torch.Tensor | None = None,
        t_hi: float | torch.Tensor | None = None,
        n_points: int = 500,
    ) -> torch.Tensor:
        # Closed form when no limits are given: ∫ N_max·exp(-½z²) dt = N_max·σ·√(2π)
        if t_lo is None and t_hi is None:
            return self.N_max * self.sigma * _SQRT2PI
        # Bounded integral via the Gaussian CDF.
        lo, hi = self._resolve_limits(t_lo, t_hi)
        sigma = self.sigma
        cdf_hi = 0.5 * (1.0 + torch.erf((hi - self.t_apex) / (sigma * _SQRT2)))
        cdf_lo = 0.5 * (1.0 + torch.erf((lo - self.t_apex) / (sigma * _SQRT2)))
        return self.N_max * sigma * _SQRT2PI * (cdf_hi - cdf_lo)

    def bounds(self, n_sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma
        return (self.t_apex - n_sigma * sigma, self.t_apex + n_sigma * sigma)

    def _resolve_limits(
        self,
        t_lo: float | torch.Tensor | None,
        t_hi: float | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b_lo, b_hi = self.bounds()
        lo = b_lo if t_lo is None else torch.as_tensor(t_lo, dtype=torch.float64)
        hi = b_hi if t_hi is None else torch.as_tensor(t_hi, dtype=torch.float64)
        return lo, hi

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
    not the underlying Gaussian centroid `μ` — they differ by `σ²/τ`
    in the high-`σ/τ` regime. We parameterize on the observable apex
    and solve back to `μ` internally so fits stay well-conditioned.
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
        """Underlying Gaussian centroid (first-order mode correction:
        `μ = t_apex - σ²/τ`)."""
        return self.t_apex - self.sigma**2 / self.tau

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma
        tau = self.tau
        mu = self.mu
        # Normalize so that the peak max equals N_max. The unnormalized
        # `emg_pdf` integrates to 1; we scale by the value at t_apex.
        pdf_t = emg_pdf(t, mu, sigma, tau)
        pdf_apex = emg_pdf(self.t_apex, mu, sigma, tau)
        return self.N_max * pdf_t / pdf_apex.clamp(min=1e-30)

    def integrate(
        self,
        t_lo: float | torch.Tensor | None = None,
        t_hi: float | torch.Tensor | None = None,
        n_points: int = 500,
    ) -> torch.Tensor:
        # Trapezoidal integration over the resolved limits — the closed
        # form for the bounded integral pulls in the EMG CDF, which is
        # itself an erfcx call; the trapezoidal route is just as accurate
        # at n_points=500 and avoids re-deriving the CDF.
        lo, hi = self._resolve_limits(t_lo, t_hi)
        grid = torch.linspace(
            lo.item() if torch.is_tensor(lo) else lo,
            hi.item() if torch.is_tensor(hi) else hi,
            n_points,
            dtype=torch.float64,
        )
        values = self.forward(grid)
        return torch.trapezoid(values, grid)

    def bounds(self, n_sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma
        tau = self.tau
        # Asymmetric: leading edge ~ Gaussian; trailing edge stretches
        # by ~3τ before decaying to noise.
        return (
            self.t_apex - n_sigma * sigma,
            self.t_apex + n_sigma * sigma + 3.0 * tau,
        )

    def _resolve_limits(
        self,
        t_lo: float | torch.Tensor | None,
        t_hi: float | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b_lo, b_hi = self.bounds()
        lo = b_lo if t_lo is None else torch.as_tensor(t_lo, dtype=torch.float64)
        hi = b_hi if t_hi is None else torch.as_tensor(t_hi, dtype=torch.float64)
        return lo, hi

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "N_max": self.N_max.detach(),
            "t_apex": self.t_apex.detach(),
            "sigma": self.sigma.detach(),
            "tau": self.tau.detach(),
            "mu": self.mu.detach(),
        }


# Reserved for future sessions (need core.optim.DifferentialEvolution to
# fit reliably; LBFGS gets stuck in the local minima these create):
#   - HyperEMGPeak     weighted left+right EMG mixture
#   - WarpedEMGPeak    piecewise cubic Hermite refinement
#   - SplinePeak       B-spline knots
