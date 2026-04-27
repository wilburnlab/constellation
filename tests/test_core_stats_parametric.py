"""Tests for core.stats.parametric — ABCs, FitResult, fit driver."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from constellation.core.stats import (
    Distribution,
    FitResult,
    NormalDistribution,
    Parametric,
    PeakShape,
)

from conftest import LBFGSAdapter


# ──────────────────────────────────────────────────────────────────────
# ABC instantiation
# ──────────────────────────────────────────────────────────────────────


def test_parametric_abc_not_instantiable():
    with pytest.raises(TypeError):
        Parametric()


def test_distribution_abc_not_instantiable():
    with pytest.raises(TypeError):
        Distribution()


def test_peakshape_abc_not_instantiable():
    with pytest.raises(TypeError):
        PeakShape()


# ──────────────────────────────────────────────────────────────────────
# FitResult dataclass
# ──────────────────────────────────────────────────────────────────────


def test_fit_result_defaults():
    r = FitResult()
    assert r.loss_history == []
    assert r.n_iter == 0
    assert r.converged is False
    assert math.isnan(r.final_loss)


# ──────────────────────────────────────────────────────────────────────
# fit() — affine recovery smoke test
# ──────────────────────────────────────────────────────────────────────


class _Affine(Parametric):
    """y = a·x + b — smallest non-trivial Parametric for fit-tests."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, x):
        return self.a * x + self.b

    def parameters_dict(self):
        return {"a": self.a.detach(), "b": self.b.detach()}


def test_fit_recovers_affine():
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
    target = 2.0 * x + 3.0  # exact target — no noise

    model = _Affine()
    opt = LBFGSAdapter(model, max_iter=50)
    result = model.fit(x, target, optimizer=opt, max_iter=50, tol=1e-12)

    params = model.parameters_dict()
    assert math.isclose(params["a"].item(), 2.0, abs_tol=1e-4)
    assert math.isclose(params["b"].item(), 3.0, abs_tol=1e-4)
    assert result.final_loss < 1e-8


def test_fit_optimizer_required():
    """`optimizer` is keyword-only and required — no default in v1."""
    model = _Affine()
    x = torch.tensor([1.0, 2.0])
    with pytest.raises(TypeError):
        model.fit(x, x)  # missing optimizer


def test_fit_records_loss_history():
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    target = 2.0 * x + 3.0
    model = _Affine()
    opt = LBFGSAdapter(model, max_iter=5)
    result = model.fit(x, target, optimizer=opt, max_iter=10, tol=1e-12)
    assert len(result.loss_history) == result.n_iter
    # Loss should never increase between recorded steps for LBFGS on convex problem
    for prev, curr in zip(result.loss_history, result.loss_history[1:]):
        assert curr <= prev + 1e-9


# ──────────────────────────────────────────────────────────────────────
# Distribution.fit — NLL on data
# ──────────────────────────────────────────────────────────────────────


def test_distribution_fit_recovers_normal():
    torch.manual_seed(42)
    samples = torch.randn(2000, dtype=torch.float64) * 2.5 + 1.5
    model = NormalDistribution(mu=0.0, sigma=1.0)
    opt = LBFGSAdapter(model, max_iter=30)
    model.fit(samples, optimizer=opt, max_iter=50)
    params = model.parameters_dict()
    assert math.isclose(params["mu"].item(), 1.5, abs_tol=0.1)
    assert math.isclose(params["sigma"].item(), 2.5, rel_tol=0.1)


# ──────────────────────────────────────────────────────────────────────
# PeakShape ABC default raises for log_prob/cdf
# ──────────────────────────────────────────────────────────────────────


def test_peakshape_log_prob_raises_by_default():
    from constellation.core.stats import GaussianPeak

    p = GaussianPeak()
    # GaussianPeak inherits the PeakShape default — does not override
    # log_prob/cdf even though it could.
    with pytest.raises(NotImplementedError):
        p.log_prob(torch.tensor(0.0))
    with pytest.raises(NotImplementedError):
        p.cdf(torch.tensor(0.0))
