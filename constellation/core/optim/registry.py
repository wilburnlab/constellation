"""Optimizer factory registry — name → callable returning an `Optimizer`.

Per Principle 5, DE and the torch.optim families share one registry so
`Parametric.fit(optimizer=OPTIMIZER_REGISTRY["de"](model, **kwargs))`
works without import-time discrimination. The registry is small in v1
(LBFGS + DE) and grows as more torch.optim wrappers earn their keep —
add Adam/SGD when a real `core.nn` training loop demands them, not
preemptively.

Factories accept `(model, **kwargs)` and return an instance satisfying
either the `Optimizer` or `PopulationOptimizer` Protocol. Callers
discriminate via `isinstance` if they care (e.g. `Parametric.fit`).
"""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from ._de import DifferentialEvolution
from ._lbfgs import LBFGSOptimizer

OptimizerFactory = Callable[..., Any]

OPTIMIZER_REGISTRY: dict[str, OptimizerFactory] = {
    "lbfgs": LBFGSOptimizer,
    "de": DifferentialEvolution,
}


def make_optimizer(name: str, model: nn.Module, **kwargs: Any) -> Any:
    """Construct an optimizer by name. Raises `KeyError` with the
    available names if `name` is not registered."""
    try:
        factory = OPTIMIZER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown optimizer {name!r}; available: {sorted(OPTIMIZER_REGISTRY)}"
        ) from None
    return factory(model, **kwargs)
