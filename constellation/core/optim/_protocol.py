"""Optimizer Protocols — `Optimizer` and `PopulationOptimizer`.

Two structurally-typed protocols cover the optimizer space `Parametric.fit()`
needs to dispatch over:

    Optimizer            scalar closure -> Tensor; gradient-based
                         (LBFGSOptimizer, torch.optim.LBFGS, Adam, ...)

    PopulationOptimizer  population closure -> (pop_size,) Tensor;
                         gradient-free (DifferentialEvolution; future
                         CMA-ES / PSO would also satisfy this)

Both are `@runtime_checkable` so `Parametric.fit()` can dispatch via
`isinstance(opt, PopulationOptimizer)`. Closures own the loss
computation; the optimizer owns gradient bookkeeping (zero_grad,
backward) and population state. This is the inversion of the v0 duck-
typed contract where the closure called `loss.backward()` itself —
moving that responsibility into the optimizer is what lets DE share
the same Protocol family without paying for unused autograd.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import torch

ScalarClosure = Callable[[], torch.Tensor]
"""closure() -> 0-d loss tensor; the optimizer is responsible for grad mgmt."""

ModelView = Callable[..., torch.Tensor]
"""A model-shaped callable that broadcasts forward over a leading population
dim. Built by DifferentialEvolution from the model + a batched-params dict
via `torch.func.vmap(torch.func.functional_call(...))`."""

PopulationClosure = Callable[[ModelView], torch.Tensor]
"""closure(model_view) -> (pop_size,) loss tensor (one fitness per individual).
Runs under torch.no_grad() — DE never reads gradients."""


@runtime_checkable
class Optimizer(Protocol):
    """Torch-optimizer-shaped contract. One step per call."""

    def step(self, closure: ScalarClosure) -> torch.Tensor: ...


@runtime_checkable
class PopulationOptimizer(Protocol):
    """Population-based optimizer. Each `step()` advances one generation
    and returns the current best fitness as a 0-d tensor. `is_converged`
    flips True when the optimizer's internal stopping criteria fire;
    `Parametric.fit()` short-circuits its outer loop on that signal."""

    def step(self, closure: PopulationClosure) -> torch.Tensor: ...

    @property
    def is_converged(self) -> bool: ...
