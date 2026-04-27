"""Tests for `core.optim.LBFGSOptimizer`."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from constellation.core.optim import LBFGSOptimizer


class _Quad(nn.Module):
    """f(x) = (x - target)^2 — minimum at target, possibly outside bounds."""

    def __init__(self, init: float = 0.0, target: float = 0.0) -> None:
        super().__init__()
        self.x = nn.Parameter(torch.tensor(init, dtype=torch.float64))
        self.register_buffer(
            "target", torch.tensor(target, dtype=torch.float64)
        )

    def forward(self) -> torch.Tensor:
        return (self.x - self.target) ** 2


# ──────────────────────────────────────────────────────────────────────
# Closure-contract inversion
# ──────────────────────────────────────────────────────────────────────


def test_closure_does_not_call_backward() -> None:
    """The new contract: closure returns un-backed loss; optimizer
    handles backward. Verify by writing a closure that explicitly does
    NOT call backward and confirming convergence."""
    m = _Quad(init=2.5, target=0.0)
    opt = LBFGSOptimizer(m, max_iter=20)

    def closure() -> torch.Tensor:
        return m()  # NO backward call

    for _ in range(10):
        loss = opt.step(closure)
        if float(loss.detach()) < 1e-12:
            break
    assert float(m.x.detach()) == pytest.approx(0.0, abs=1e-6)


def test_step_returns_tensor() -> None:
    m = _Quad(init=1.0)
    opt = LBFGSOptimizer(m)

    def closure() -> torch.Tensor:
        return m()

    out = opt.step(closure)
    assert torch.is_tensor(out)
    assert out.dim() == 0


# ──────────────────────────────────────────────────────────────────────
# Bounds clamping
# ──────────────────────────────────────────────────────────────────────


def test_bounds_clamping_keeps_params_in_box() -> None:
    """Unconstrained LBFGS would converge to x=10, but bounds restrict
    to [-1, 1]; clamping must hold the result at the boundary."""
    m = _Quad(init=0.0, target=10.0)  # unconstrained optimum at x=10
    opt = LBFGSOptimizer(m, bounds={"x": (-1.0, 1.0)}, max_iter=50)

    def closure() -> torch.Tensor:
        return m()

    for _ in range(20):
        opt.step(closure)
    assert -1.0 <= float(m.x.detach()) <= 1.0
    # Should be pulled to the upper bound (closest feasible point to target=10).
    assert float(m.x.detach()) == pytest.approx(1.0, abs=1e-6)


def test_bounds_require_module_not_iterable() -> None:
    p = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="bounds"):
        LBFGSOptimizer([p], bounds={"x": (-1.0, 1.0)})


def test_no_bounds_works_with_iterable_params() -> None:
    p = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    opt = LBFGSOptimizer([p])

    def closure() -> torch.Tensor:
        return p ** 2

    for _ in range(10):
        opt.step(closure)
    assert float(p.detach()) == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────
# Polish-bug regression (cartographer 0cee299)
# ──────────────────────────────────────────────────────────────────────


def test_polish_bounds_regression_unconstrained_opt_outside_box() -> None:
    """Cartographer 0cee299 fix: before bounds-clamping was folded into
    the optimizer, an LBFGS polish on the global-best individual could
    escape DE bounds (e.g. tau_l → 190s creating flat baselines). With
    the clamp inside .step(), even an aggressive line search must
    respect bounds."""
    m = _Quad(init=-0.5, target=100.0)  # would slingshot far out of box
    opt = LBFGSOptimizer(
        m, bounds={"x": (-1.0, 1.0)}, lr=1.0, max_iter=100
    )

    def closure() -> torch.Tensor:
        return m()

    for _ in range(5):
        opt.step(closure)
    assert -1.0 <= float(m.x.detach()) <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Glob-pattern bounds
# ──────────────────────────────────────────────────────────────────────


class _LogQuad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        # Drives log_a and log_b way down (huge negative); c to 100.
        return (
            (self.log_a + 50.0) ** 2
            + (self.log_b + 50.0) ** 2
            + (self.c - 100.0) ** 2
        )


def test_glob_pattern_bounds() -> None:
    m = _LogQuad()
    opt = LBFGSOptimizer(
        m, bounds={"log_*": (-3.0, 3.0), "c": (-1.0, 1.0)}, max_iter=50
    )

    def closure() -> torch.Tensor:
        return m()

    for _ in range(10):
        opt.step(closure)
    assert -3.0 <= float(m.log_a.detach()) <= 3.0
    assert -3.0 <= float(m.log_b.detach()) <= 3.0
    assert -1.0 <= float(m.c.detach()) <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Recovers an Affine through Parametric.fit
# ──────────────────────────────────────────────────────────────────────


def test_recovers_affine_through_parametric_fit() -> None:
    """Same recovery test as `test_fit_recovers_affine` but exercising
    LBFGSOptimizer through the public Parametric.fit driver — proves
    the closure-contract refactor doesn't break end-to-end fitting."""
    from constellation.core.stats import Parametric

    class Affine(Parametric):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
            self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.a * x + self.b

        def parameters_dict(self) -> dict[str, torch.Tensor]:
            return {"a": self.a.detach(), "b": self.b.detach()}

    torch.manual_seed(7)
    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    y = 3.0 * x - 2.0 + 0.01 * torch.randn(50, dtype=torch.float64)

    m = Affine()
    opt = LBFGSOptimizer(m, max_iter=30)
    res = m.fit(x, y, optimizer=opt, max_iter=20, tol=1e-12)

    assert res.converged
    assert float(m.a.detach()) == pytest.approx(3.0, abs=0.05)
    assert float(m.b.detach()) == pytest.approx(-2.0, abs=0.05)
    assert math.isfinite(res.final_loss)
