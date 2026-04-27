"""Parametric form ABCs — `Parametric` / `Distribution` / `PeakShape`.

`Parametric` is the umbrella ABC: an `nn.Module` carrying a `.fit()` driver
that runs an external optimizer over the model's parameters. Two
specializations sit underneath:

    Distribution     adds .log_prob(x), .cdf(x); default loss = NLL
    PeakShape        adds .integrate(), .bounds(); default loss = MSE
                     log_prob/cdf optional (raise NotImplementedError)

Optimizer policy: this module does not ship any optimizer implementation.
`Parametric.fit(data, *, optimizer, ...)` accepts any object exposing
``step(closure) -> Tensor`` (duck-typed; type hint is `object` in v1).
The formal `Optimizer` Protocol lives in `core.optim` once that ships;
the type hint upgrades then without code changes here.

Convention (documented, not enforced): subclasses parameterize positive
quantities in log-space (`log_sigma`, `log_tau`, `log_N_max`) and bounded
quantities in logit-space (`logit_eta`, mixture weights). Default dtype
is float64 — peak-shape and density numerics are precision-sensitive.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────
# FitResult
# ──────────────────────────────────────────────────────────────────────


@dataclass
class FitResult:
    """Outcome of a `Parametric.fit()` call."""

    loss_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    final_loss: float = math.nan


# ──────────────────────────────────────────────────────────────────────
# Parametric ABC
# ──────────────────────────────────────────────────────────────────────


class Parametric(nn.Module, abc.ABC):
    """Base class for fittable parametric forms.

    Subclasses must implement `forward(x)` and `parameters_dict()`.
    `_default_loss(pred, target)` is overridden by `Distribution`
    (NLL) and `PeakShape` (MSE); leaf subclasses (calibration models)
    inherit MSE from this base.

    Parameters live as `nn.Parameter` attributes; subclasses register
    them in `__init__`. Default tensor dtype is float64.
    """

    def __init__(self) -> None:
        super().__init__()

    # -- abstract surface --------------------------------------------------

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the parametric form at `x`."""

    @abc.abstractmethod
    def parameters_dict(self) -> dict[str, torch.Tensor]:
        """Detached named-parameter snapshot. Used for logging /
        round-trip checks; subclasses decode log-space / logit-space
        parameterizations into natural units here."""

    # -- default loss (overridable) ---------------------------------------

    def _default_loss(
        self, pred: torch.Tensor, target: torch.Tensor | None
    ) -> torch.Tensor:
        if target is None:
            raise ValueError(
                "default loss is MSE and requires `target`; "
                "pass `loss_fn=` for unsupervised fits"
            )
        return ((pred - target) ** 2).mean()

    # -- fit driver -------------------------------------------------------

    def fit(
        self,
        data: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        optimizer: object,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        max_iter: int = 200,
        tol: float = 1e-8,
        callback: Callable[[int, float], None] | None = None,
    ) -> FitResult:
        """Fit parameters to `data` (and optional `target`) via an external optimizer.

        `optimizer` must expose ``step(closure) -> Tensor``. The closure
        returns a scalar loss and is responsible for `zero_grad()` +
        `backward()`. Convergence: ``|loss_{i} - loss_{i-1}| < tol``.
        NaN loss aborts (`converged=False`).
        """
        result = FitResult()
        prev_loss = math.inf

        for i in range(max_iter):
            def closure():
                # `optimizer` is responsible for ordering; we just compute
                # the loss + gradients on each call. Some optimizers (LBFGS)
                # call this multiple times per outer step.
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                pred = self.forward(data)
                if loss_fn is not None:
                    loss = loss_fn(pred, target) if target is not None else loss_fn(pred)
                elif isinstance(self, Distribution):
                    # NLL on the data itself; `pred` ignored.
                    loss = -self.log_prob(data).sum()
                else:
                    loss = self._default_loss(pred, target)
                loss.backward()
                return loss

            loss_t = optimizer.step(closure)
            if torch.is_tensor(loss_t):
                loss_val = float(loss_t.detach())
            else:
                loss_val = float(loss_t)

            if not math.isfinite(loss_val):
                result.n_iter = i + 1
                result.final_loss = loss_val
                result.converged = False
                if callback is not None:
                    callback(i, loss_val)
                return result

            result.loss_history.append(loss_val)
            if callback is not None:
                callback(i, loss_val)

            if abs(prev_loss - loss_val) < tol:
                result.n_iter = i + 1
                result.final_loss = loss_val
                result.converged = True
                return result
            prev_loss = loss_val

        result.n_iter = max_iter
        result.final_loss = prev_loss
        result.converged = False
        return result


# ──────────────────────────────────────────────────────────────────────
# Distribution ABC
# ──────────────────────────────────────────────────────────────────────


class Distribution(Parametric, abc.ABC):
    """Probability distribution: `.log_prob(x)` and `.cdf(x)` required.

    Default loss is negative log-likelihood on `data` (target is
    ignored — the data IS the target for density fitting).
    """

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability density (or mass, for discrete) at `x`."""

    @abc.abstractmethod
    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution function at `x`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward = log_prob. Override only if a model needs a
        different evaluation (e.g. mixtures returning component-wise)."""
        return self.log_prob(x)


# ──────────────────────────────────────────────────────────────────────
# PeakShape ABC
# ──────────────────────────────────────────────────────────────────────


class PeakShape(Parametric, abc.ABC):
    """Observable peak shape: `.integrate()` and `.bounds()` required.

    `log_prob` and `cdf` raise `NotImplementedError` by default; peak
    shapes that ALSO carry a clean density (Gaussian, EMG) may
    override. Default loss is MSE — peaks fit observed counts, not
    samples from a density.
    """

    @abc.abstractmethod
    def integrate(
        self,
        t_lo: float | torch.Tensor | None = None,
        t_hi: float | torch.Tensor | None = None,
        n_points: int = 500,
    ) -> torch.Tensor:
        """Integrated area under the peak over [t_lo, t_hi].
        Defaults to the peak's `.bounds()` when limits are None."""

    @abc.abstractmethod
    def bounds(self, n_sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
        """`(t_lo, t_hi)` covering the peak's significant support."""

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} is a PeakShape, not a Distribution. "
            "Use a Distribution subclass (e.g. NormalDistribution) for "
            "density evaluation."
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} is a PeakShape, not a Distribution. "
            "Use `.integrate(t_lo, t_hi)` for area under the peak."
        )
