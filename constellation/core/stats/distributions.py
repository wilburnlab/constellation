"""Probability distributions — `Distribution` ABC subclasses.

Continuous: NormalDistribution, StudentT, GeneralizedNormal, Beta,
Gamma, LogNormal.
Discrete: Poisson, Multinomial.
Vector / simplex: Dirichlet.

All carry `.log_prob(x)` and `.cdf(x)` (where closed-form). Multinomial
and Dirichlet `.cdf` raise — no closed form for K ≥ 3. Parameters are
stored in unconstrained reals (log-space for positives, raw for
locations) so they fit cleanly under any gradient-based optimizer.

Numerical conventions:
    - Default float64. Construct on CPU; users `.to(device)` if needed.
    - log_prob is the *log probability density / mass*, not the
      negative log-likelihood — `Parametric.fit` negates and sums.
    - `from_logits=True` (where applicable) accepts unconstrained
      input parameterizations directly.

CDF gap. torch lacks `betainc` as of 2.11, so `StudentT.cdf` and
`Beta.cdf` raise `NotImplementedError` in v1. A scipy bridge would
work for the forward pass but break autograd, and a custom
autograd Function felt heavy for the actual v1 use cases — fitting
always goes through `log_prob` (pure torch), and CDFs aren't on the
`Parametric.fit` path. Lift the gap when torch ships `betainc` or
when a concrete use case shows up (gradient-aware quantile loss,
Cartographer port phase, etc.).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .parametric import Distribution

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────
# Continuous — location/scale
# ──────────────────────────────────────────────────────────────────────


class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution. Params: `mu`, `log_sigma`."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float64))
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma), dtype=torch.float64)
        )

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma
        z = (x - self.mu) / sigma
        return -0.5 * (z * z + _LOG_2PI) - self.log_sigma

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf((x - self.mu) / (self.sigma * _SQRT_2)))

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"mu": self.mu.detach(), "sigma": self.sigma.detach()}


class StudentT(Distribution):
    """Student-t distribution. Params: `mu`, `log_sigma`, `log_nu`.

    Used for MS m/z error models — the heavy tails accommodate
    occasional large miscalibrations without distorting the bulk fit.
    `log_nu` is clamped at `log(0.5)` from below to avoid the lgamma
    pole at ν=0.
    """

    _LOG_NU_MIN: float = math.log(0.5)

    def __init__(
        self, mu: float = 0.0, sigma: float = 1.0, nu: float = 4.0
    ) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float64))
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma), dtype=torch.float64)
        )
        self.log_nu = nn.Parameter(torch.tensor(math.log(nu), dtype=torch.float64))

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    @property
    def nu(self) -> torch.Tensor:
        return self.log_nu.clamp(min=self._LOG_NU_MIN).exp()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        nu = self.nu
        sigma = self.sigma
        half_nu = 0.5 * nu
        half_nup1 = 0.5 * (nu + 1.0)
        z = (x - self.mu) / sigma
        return (
            torch.lgamma(half_nup1)
            - torch.lgamma(half_nu)
            - 0.5 * torch.log(nu * math.pi)
            - torch.log(sigma)
            - half_nup1 * torch.log1p(z * z / nu)
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        # Pinned: torch lacks betainc as of 2.11; a scipy bridge would
        # break autograd, and a custom autograd Function felt heavy for
        # the v1 use cases (fitting goes through log_prob). See module
        # docstring "CDF gap" — lift when torch ships betainc or when a
        # concrete gradient-aware-CDF use case appears.
        raise NotImplementedError(
            "StudentT.cdf is not implemented in v1 (torch lacks betainc). "
            "Use scipy.stats.t.cdf at the boundary if you need numerical "
            "values; fitting goes through log_prob (pure torch)."
        )

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mu": self.mu.detach(),
            "sigma": self.sigma.detach(),
            "nu": self.nu.detach(),
        }


class GeneralizedNormal(Distribution):
    """Generalized Normal (exponential power) distribution.

    `f(x) = β / (2α Γ(1/β)) · exp(-(|x-μ|/α)^β)`. Reduces to Normal
    at β=2 (with `α = σ·√2`) and Laplace at β=1. Params: `mu`,
    `log_alpha`, `log_beta`. `log_beta` is clamped to `[log(0.1), log(20)]`
    to keep the Γ(1/β) call stable.
    """

    _LOG_BETA_MIN: float = math.log(0.1)
    _LOG_BETA_MAX: float = math.log(20.0)

    def __init__(
        self, mu: float = 0.0, alpha: float = 1.0, beta: float = 2.0
    ) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float64))
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha), dtype=torch.float64)
        )
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(beta), dtype=torch.float64)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.clamp(
            min=self._LOG_BETA_MIN, max=self._LOG_BETA_MAX
        ).exp()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        beta = self.beta
        z = (x - self.mu).abs() / alpha
        return (
            torch.log(beta)
            - math.log(2.0)
            - torch.log(alpha)
            - torch.lgamma(1.0 / beta)
            - z**beta
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        beta = self.beta
        z = (x - self.mu) / alpha
        # Symmetric CDF via regularized lower incomplete gamma:
        #   F(x) = 0.5 + sign(z) · 0.5 · γ(1/β, |z|^β) / Γ(1/β)
        gammainc = torch.special.gammainc(1.0 / beta, z.abs() ** beta)
        return 0.5 + 0.5 * torch.sign(z) * gammainc

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mu": self.mu.detach(),
            "alpha": self.alpha.detach(),
            "beta": self.beta.detach(),
        }


# ──────────────────────────────────────────────────────────────────────
# Continuous — positive support
# ──────────────────────────────────────────────────────────────────────


class Beta(Distribution):
    """Beta distribution on `(0, 1)`. Params: `log_alpha`, `log_beta`."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha), dtype=torch.float64)
        )
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(beta), dtype=torch.float64)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        beta = self.beta
        x_safe = x.clamp(min=_EPS, max=1.0 - _EPS)
        return (
            (alpha - 1.0) * torch.log(x_safe)
            + (beta - 1.0) * torch.log1p(-x_safe)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
            + torch.lgamma(alpha + beta)
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        # Pinned: see module docstring "CDF gap". Same situation as
        # StudentT.cdf — torch lacks betainc as of 2.11.
        raise NotImplementedError(
            "Beta.cdf is not implemented in v1 (torch lacks betainc). "
            "Use scipy.stats.beta.cdf at the boundary if you need "
            "numerical values; fitting goes through log_prob (pure torch)."
        )

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"alpha": self.alpha.detach(), "beta": self.beta.detach()}


class Gamma(Distribution):
    """Gamma distribution on `(0, ∞)`, rate parameterization. Params:
    `log_alpha` (shape), `log_beta` (rate). Mean = α/β."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha), dtype=torch.float64)
        )
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(beta), dtype=torch.float64)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        beta = self.beta
        x_safe = x.clamp(min=_EPS)
        return (
            alpha * torch.log(beta)
            - torch.lgamma(alpha)
            + (alpha - 1.0) * torch.log(x_safe)
            - beta * x_safe
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = x.clamp(min=0.0)
        return torch.special.gammainc(self.alpha, self.beta * x_safe)

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"alpha": self.alpha.detach(), "beta": self.beta.detach()}


class LogNormal(Distribution):
    """LogNormal distribution: `log(x) ~ Normal(mu, sigma)`. Params:
    `mu`, `log_sigma`. Defined on `(0, ∞)`."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float64))
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma), dtype=torch.float64)
        )

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = x.clamp(min=_EPS)
        log_x = torch.log(x_safe)
        sigma = self.sigma
        z = (log_x - self.mu) / sigma
        return -0.5 * (z * z + _LOG_2PI) - self.log_sigma - log_x

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = x.clamp(min=_EPS)
        log_x = torch.log(x_safe)
        return 0.5 * (1.0 + torch.erf((log_x - self.mu) / (self.sigma * _SQRT_2)))

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"mu": self.mu.detach(), "sigma": self.sigma.detach()}


# ──────────────────────────────────────────────────────────────────────
# Discrete
# ──────────────────────────────────────────────────────────────────────


class Poisson(Distribution):
    """Poisson distribution. Params: `log_rate`. PMF on non-negative
    integers; `log_prob` accepts integer or float-valued tensors."""

    def __init__(self, rate: float = 1.0) -> None:
        super().__init__()
        self.log_rate = nn.Parameter(
            torch.tensor(math.log(rate), dtype=torch.float64)
        )

    @property
    def rate(self) -> torch.Tensor:
        return self.log_rate.exp()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        rate = self.rate
        x_f = x.to(torch.float64)
        return -rate + x_f * self.log_rate - torch.lgamma(x_f + 1.0)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        # F(k; λ) = Q(⌊k⌋ + 1, λ)  where Q is the regularized upper
        # incomplete gamma. torch.special.gammaincc is Q.
        k = torch.floor(x.to(torch.float64))
        return torch.special.gammaincc(k + 1.0, self.rate)

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"rate": self.rate.detach()}


class Multinomial(Distribution):
    """Multinomial distribution. Param: `log_alpha` shape `(K,)`.

    `log_alpha` is treated as logits (`from_logits=True` is the
    default since the parameter name says "log") — softmaxed
    internally. `log_prob(x)` accepts integer counts of shape
    `(..., K)`; `n` is inferred from `x.sum(-1)`. `cdf` raises
    (no closed form for K≥3).
    """

    def __init__(self, logits: torch.Tensor | None = None, K: int = 2) -> None:
        super().__init__()
        if logits is None:
            init = torch.zeros(K, dtype=torch.float64)
        else:
            init = torch.as_tensor(logits, dtype=torch.float64)
            if init.ndim != 1:
                raise ValueError(f"logits must be 1-D; got shape {tuple(init.shape)}")
        self.log_alpha = nn.Parameter(init)

    @property
    def K(self) -> int:
        return int(self.log_alpha.shape[0])

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.log_alpha, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.to(torch.float64)
        n = x_f.sum(dim=-1)
        log_p = torch.log_softmax(self.log_alpha, dim=-1)
        log_factorial_n = torch.lgamma(n + 1.0)
        log_factorial_x = torch.lgamma(x_f + 1.0).sum(dim=-1)
        return log_factorial_n - log_factorial_x + (x_f * log_p).sum(dim=-1)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Multinomial CDF has no closed form for K>=3; for K=2 use the "
            "binomial CDF directly."
        )

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"probs": self.probs.detach()}


# ──────────────────────────────────────────────────────────────────────
# Vector / simplex
# ──────────────────────────────────────────────────────────────────────


class Dirichlet(Distribution):
    """Dirichlet distribution on the K-simplex. Param: `log_alpha`
    shape `(K,)`. `log_prob(x)` accepts simplex inputs (or pre-softmax
    logits via `from_logits=True`). `cdf` raises (no closed form)."""

    def __init__(
        self, alpha: torch.Tensor | None = None, K: int = 3
    ) -> None:
        super().__init__()
        if alpha is None:
            init = torch.zeros(K, dtype=torch.float64)  # alpha=1 (uniform)
        else:
            a = torch.as_tensor(alpha, dtype=torch.float64)
            if a.ndim != 1:
                raise ValueError(f"alpha must be 1-D; got shape {tuple(a.shape)}")
            init = torch.log(a.clamp(min=_EPS))
        self.log_alpha = nn.Parameter(init)

    @property
    def K(self) -> int:
        return int(self.log_alpha.shape[0])

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def log_prob(
        self, x: torch.Tensor, *, from_logits: bool = False
    ) -> torch.Tensor:
        if from_logits:
            log_x = torch.log_softmax(x, dim=-1)
        else:
            log_x = x.clamp(min=_EPS).log()
        alpha = self.alpha
        log_norm = torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum()
        return log_norm + ((alpha - 1.0) * log_x).sum(dim=-1)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Dirichlet CDF has no closed form for K>=3; use Monte Carlo "
            "integration for empirical estimates."
        )

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"alpha": self.alpha.detach()}
