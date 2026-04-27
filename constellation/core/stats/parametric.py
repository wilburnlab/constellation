"""Parametric form ABCs — `Parametric` / `Distribution` / `PeakShape`.

`Parametric` is the umbrella ABC: an `nn.Module` carrying a `.fit()` driver
that runs an external optimizer over the model's parameters. Two
specializations sit underneath:

    Distribution     adds .log_prob(x), .cdf(x); default loss = NLL
    PeakShape        adds .integrate(), .bounds(); default loss = MSE
                     log_prob/cdf optional (raise NotImplementedError)

`Parametric.fit(data, *, optimizer, ...)` dispatches on the optimizer
shape: an `Optimizer` (gradient-based; LBFGS, Adam, ...) gets a scalar
closure that returns un-backed loss; a `PopulationOptimizer` (DE) gets
a population closure that calls a vmap'd `model_view` and returns a
`(pop_size,)` loss vector. Both Protocols live in `core.optim`.

Vmap implication: every subclass's `forward` (and `log_prob` / `cdf`
for `Distribution`) must be `torch.func.vmap`-compatible — basic
broadcasting torch ops are fine; boolean indexing with dynamic shape
breaks vmap and must be rewritten with `torch.where`.

Convention (documented, not enforced): subclasses parameterize positive
quantities in log-space (`log_sigma`, `log_tau`, `log_N_max`) and bounded
quantities in logit-space (`logit_eta`, mixture weights). Default dtype
is float64 — peak-shape and density numerics are precision-sensitive.
DE bounds operate on the registered name (e.g. `log_sigma`); use
`core.optim.bounds_in_natural_units` to convert from natural units.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from constellation.core.optim import Optimizer, PopulationOptimizer


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
        optimizer: Optimizer | PopulationOptimizer,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        max_iter: int = 200,
        tol: float = 1e-8,
        polish_on_converge: bool = False,
        callback: Callable[[int, float], None] | None = None,
    ) -> FitResult:
        """Fit parameters to `data` (and optional `target`) via an external optimizer.

        Dispatches on optimizer type. For `Optimizer` (gradient-based)
        the closure returns a scalar un-backed loss; the optimizer
        manages `zero_grad` / `backward` internally. For
        `PopulationOptimizer` (DE) the closure receives a vmap'd
        `model_view` and returns a `(pop_size,)` loss vector.

        Outer convergence: ``|loss_{i} - loss_{i-1}| < tol`` for the
        scalar path; ``optimizer.is_converged`` short-circuits the
        population path. NaN loss aborts (`converged=False`).

        `polish_on_converge=True` triggers `optimizer.polish(scalar_closure)`
        after the loop terminates if the optimizer exposes the method
        (e.g. DifferentialEvolution).
        """
        if isinstance(optimizer, PopulationOptimizer):
            return self._fit_population(
                data, target, optimizer, loss_fn, max_iter, tol,
                polish_on_converge, callback,
            )
        return self._fit_scalar(
            data, target, optimizer, loss_fn, max_iter, tol,
            polish_on_converge, callback,
        )

    def _build_scalar_closure(
        self,
        data: torch.Tensor,
        target: torch.Tensor | None,
        loss_fn: Callable[..., torch.Tensor] | None,
    ) -> Callable[[], torch.Tensor]:
        """Returns a closure() -> loss tensor. The closure does NOT call
        backward — that responsibility belongs to the optimizer."""

        def closure() -> torch.Tensor:
            pred = self.forward(data)
            if loss_fn is not None:
                return loss_fn(pred, target) if target is not None else loss_fn(pred)
            if isinstance(self, Distribution):
                return -self.log_prob(data).sum()
            return self._default_loss(pred, target)

        return closure

    def _fit_scalar(
        self,
        data: torch.Tensor,
        target: torch.Tensor | None,
        optimizer: Optimizer,
        loss_fn: Callable[..., torch.Tensor] | None,
        max_iter: int,
        tol: float,
        polish_on_converge: bool,
        callback: Callable[[int, float], None] | None,
    ) -> FitResult:
        result = FitResult()
        prev_loss = math.inf
        closure = self._build_scalar_closure(data, target, loss_fn)

        for i in range(max_iter):
            loss_t = optimizer.step(closure)
            loss_val = (
                float(loss_t.detach()) if torch.is_tensor(loss_t) else float(loss_t)
            )

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
                if polish_on_converge and hasattr(optimizer, "polish"):
                    optimizer.polish(closure)
                return result
            prev_loss = loss_val

        result.n_iter = max_iter
        result.final_loss = prev_loss
        result.converged = False
        return result

    def _fit_population(
        self,
        data: torch.Tensor,
        target: torch.Tensor | None,
        optimizer: PopulationOptimizer,
        loss_fn: Callable[..., torch.Tensor] | None,
        max_iter: int,
        tol: float,
        polish_on_converge: bool,
        callback: Callable[[int, float], None] | None,
    ) -> FitResult:
        """Population-fit driver. Builds a vectorized closure that calls
        the vmap'd `model_view` and reduces loss per individual."""
        result = FitResult()
        is_distribution = isinstance(self, Distribution)

        def pop_closure(model_view) -> torch.Tensor:
            if loss_fn is not None:
                pred = model_view(data)   # (pop, *forward_shape)
                if target is not None:
                    return loss_fn(pred, target)
                return loss_fn(pred)
            if is_distribution:
                # Distribution.forward defaults to log_prob (see Distribution
                # ABC); the vmap'd model_view substitutes batched params and
                # broadcasts log_prob over the leading population dim.
                lp_per_individual = model_view(data)   # (pop, *data_shape)
                if lp_per_individual.dim() == 1:
                    return -lp_per_individual
                reduce_dims = tuple(range(1, lp_per_individual.dim()))
                return -lp_per_individual.sum(dim=reduce_dims)
            # PeakShape / calibration: MSE per individual against target
            pred = model_view(data)   # (pop, *data_shape)
            if target is None:
                raise ValueError(
                    "default loss is MSE and requires `target`; "
                    "pass `loss_fn=` for unsupervised population fits"
                )
            diff = pred - target
            reduce_dims = tuple(range(1, diff.dim()))
            return (diff * diff).mean(dim=reduce_dims)

        for i in range(max_iter):
            best_t = optimizer.step(pop_closure)
            loss_val = (
                float(best_t.detach()) if torch.is_tensor(best_t) else float(best_t)
            )
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

            if optimizer.is_converged:
                result.n_iter = i + 1
                result.final_loss = loss_val
                result.converged = True
                if polish_on_converge and hasattr(optimizer, "polish"):
                    scalar_closure = self._build_scalar_closure(
                        data, target, loss_fn
                    )
                    optimizer.polish(scalar_closure)
                    # Re-evaluate after polish
                    with torch.no_grad():
                        polished = scalar_closure()
                    polished_val = float(polished.detach())
                    result.loss_history.append(polished_val)
                    result.final_loss = polished_val
                return result

        result.n_iter = max_iter
        result.final_loss = (
            result.loss_history[-1] if result.loss_history else math.inf
        )
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
