"""Optimizers — gradient-based and gradient-free under one Protocol family.

`core.optim` ships:
    Optimizer            scalar closure -> Tensor (Protocol)
    PopulationOptimizer  population closure -> (pop_size,) loss (Protocol)
    LBFGSOptimizer       torch.optim.LBFGS wrapper with optional bounds clamping
    DifferentialEvolution generic over nn.Module; vectorized via vmap +
                         functional_call; 6 mutation strategies; Sobol/LHS/
                         uniform init; binomial/exponential crossover; F/CR
                         dithering; clip/reflect bounds handling; explicit
                         terminal LBFGS polish reusing LBFGSOptimizer.
    OPTIMIZER_REGISTRY    name → factory for both above (Principle 5)
    bounds_in_natural_units helper for log/logit-parameterized models

DAG: `core.optim` is leaf-tier — depends only on `torch`. `core.stats`
imports the Protocols here; `Parametric.fit()` dispatches on optimizer
shape via `isinstance(opt, PopulationOptimizer)`.
"""

from ._bounds import bounds_in_natural_units
from ._de import DifferentialEvolution
from ._lbfgs import LBFGSOptimizer
from ._protocol import (
    ModelView,
    Optimizer,
    PopulationClosure,
    PopulationOptimizer,
    ScalarClosure,
)
from ._strategies import Crossover, MutationStrategy
from .registry import OPTIMIZER_REGISTRY, make_optimizer

InitMethod = str  # public alias matching the runtime kwargs

__all__ = [
    "Optimizer",
    "PopulationOptimizer",
    "ScalarClosure",
    "PopulationClosure",
    "ModelView",
    "LBFGSOptimizer",
    "DifferentialEvolution",
    "OPTIMIZER_REGISTRY",
    "make_optimizer",
    "bounds_in_natural_units",
    "MutationStrategy",
    "Crossover",
    "InitMethod",
]
