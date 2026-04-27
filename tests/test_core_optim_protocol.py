"""Protocol-conformance tests for `core.optim`."""

from __future__ import annotations

import torch
from torch import nn

from constellation.core.optim import (
    DifferentialEvolution,
    LBFGSOptimizer,
    Optimizer,
    PopulationOptimizer,
)


class _Quad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = nn.Parameter(torch.zeros(2, dtype=torch.float64) + 1.0)

    def forward(self) -> torch.Tensor:
        return (self.x ** 2).sum()


def test_lbfgs_optimizer_satisfies_optimizer_protocol() -> None:
    opt = LBFGSOptimizer(_Quad())
    assert isinstance(opt, Optimizer)


def test_lbfgs_optimizer_does_not_satisfy_population_optimizer() -> None:
    opt = LBFGSOptimizer(_Quad())
    assert not isinstance(opt, PopulationOptimizer)


def test_de_satisfies_both_protocols() -> None:
    # DE has step(closure) and is_converged — satisfies both
    # Optimizer (structurally — step accepts a callable) and PopulationOptimizer.
    opt = DifferentialEvolution(_Quad(), bounds={"x": (-5.0, 5.0)}, seed=0)
    assert isinstance(opt, PopulationOptimizer)
    assert isinstance(opt, Optimizer)


def test_torch_lbfgs_satisfies_optimizer_protocol() -> None:
    """torch.optim.LBFGS already returns Tensor from .step(closure); the
    Protocol is structurally satisfied without an adapter."""
    p = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    raw = torch.optim.LBFGS([p])
    assert isinstance(raw, Optimizer)


def test_minimal_fake_optimizer_satisfies_protocol() -> None:
    class Fake:
        def step(self, closure):
            return torch.tensor(0.0)

    assert isinstance(Fake(), Optimizer)
    # No is_converged → not a PopulationOptimizer.
    assert not isinstance(Fake(), PopulationOptimizer)
