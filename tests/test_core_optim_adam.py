"""Tests for `core.optim.AdamOptimizer`."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from constellation.core.optim import (
    AdamOptimizer,
    Optimizer,
    PopulationOptimizer,
    OPTIMIZER_REGISTRY,
    make_optimizer,
)


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
    """The closure returns un-backed loss; the optimizer handles backward.
    Verify by writing a closure that explicitly does NOT call backward and
    confirming convergence."""
    m = _Quad(init=2.5, target=0.0)
    opt = AdamOptimizer(m, lr=0.2)

    def closure() -> torch.Tensor:
        return m()  # NO backward call

    for _ in range(2000):
        loss = opt.step(closure)
        if float(loss.detach()) < 1e-12:
            break
    assert float(m.x.detach()) == pytest.approx(0.0, abs=1e-4)


def test_step_returns_scalar_tensor() -> None:
    m = _Quad(init=1.0)
    opt = AdamOptimizer(m)

    def closure() -> torch.Tensor:
        return m()

    out = opt.step(closure)
    assert torch.is_tensor(out)
    assert out.dim() == 0


def test_step_decreases_loss() -> None:
    m = _Quad(init=3.0, target=0.0)
    opt = AdamOptimizer(m, lr=0.1)

    def closure() -> torch.Tensor:
        return m()

    first = float(opt.step(closure).detach())
    for _ in range(50):
        last = float(opt.step(closure).detach())
    assert last < first


# ──────────────────────────────────────────────────────────────────────
# Bounds clamping (shared path with LBFGSOptimizer)
# ──────────────────────────────────────────────────────────────────────


def test_bounds_clamping_keeps_params_in_box() -> None:
    """Unconstrained Adam would march to x=10, but bounds restrict to
    [-1, 1]; clamping after each step must hold the result at the
    boundary."""
    m = _Quad(init=0.0, target=10.0)  # unconstrained optimum at x=10
    opt = AdamOptimizer(m, bounds={"x": (-1.0, 1.0)}, lr=0.2)

    def closure() -> torch.Tensor:
        return m()

    for _ in range(200):
        opt.step(closure)
    assert -1.0 <= float(m.x.detach()) <= 1.0
    # Pulled to the upper bound (closest feasible point to target=10).
    assert float(m.x.detach()) == pytest.approx(1.0, abs=1e-6)


def test_bounds_require_module_not_iterable() -> None:
    p = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="bounds"):
        AdamOptimizer([p], bounds={"x": (-1.0, 1.0)})


def test_no_bounds_works_with_iterable_params() -> None:
    p = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    opt = AdamOptimizer([p], lr=0.2)

    def closure() -> torch.Tensor:
        return p ** 2

    for _ in range(2000):
        opt.step(closure)
    assert float(p.detach()) == pytest.approx(0.0, abs=1e-4)


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
    opt = AdamOptimizer(
        m, bounds={"log_*": (-3.0, 3.0), "c": (-1.0, 1.0)}, lr=0.5
    )

    def closure() -> torch.Tensor:
        return m()

    for _ in range(300):
        opt.step(closure)
    assert -3.0 <= float(m.log_a.detach()) <= 3.0
    assert -3.0 <= float(m.log_b.detach()) <= 3.0
    assert -1.0 <= float(m.c.detach()) <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Registry + Protocol membership
# ──────────────────────────────────────────────────────────────────────


def test_registered_as_adam() -> None:
    assert OPTIMIZER_REGISTRY["adam"] is AdamOptimizer
    m = _Quad()
    opt = make_optimizer("adam", m, lr=0.1)
    assert isinstance(opt, AdamOptimizer)


def test_satisfies_optimizer_protocol_not_population() -> None:
    opt = AdamOptimizer(_Quad())
    assert isinstance(opt, Optimizer)
    assert not isinstance(opt, PopulationOptimizer)


# ──────────────────────────────────────────────────────────────────────
# Non-finite-gradient guard (robustness for noisy MC objectives, e.g. ELBO)
# ──────────────────────────────────────────────────────────────────────


def test_skips_nonfinite_gradient_then_resumes() -> None:
    """A NaN gradient (e.g. an extreme Monte-Carlo ELBO draw) must NOT move the
    parameters — stepping would poison both the params and Adam's moments."""
    p = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    opt = AdamOptimizer([p], lr=0.1)
    bad = [True]

    def closure() -> torch.Tensor:
        return p * float("nan") if bad[0] else p**2

    before = float(p.detach())
    opt.step(closure)  # NaN gradient → skipped
    assert float(p.detach()) == before
    bad[0] = False
    opt.step(closure)  # finite gradient → moves
    assert float(p.detach()) != before
    assert math.isfinite(float(p.detach()))


# ──────────────────────────────────────────────────────────────────────
# Recovers an Affine through Parametric.fit
# ──────────────────────────────────────────────────────────────────────


def test_recovers_affine_through_parametric_fit() -> None:
    """Exercise AdamOptimizer through the public Parametric.fit driver —
    proves the scalar-closure contract drives end-to-end fitting (the same
    path the ELBO objective will use)."""
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
    opt = AdamOptimizer(m, lr=0.1)
    res = m.fit(x, y, optimizer=opt, max_iter=2000, tol=1e-14)

    assert float(m.a.detach()) == pytest.approx(3.0, abs=0.05)
    assert float(m.b.detach()) == pytest.approx(-2.0, abs=0.05)
    assert math.isfinite(res.final_loss)
