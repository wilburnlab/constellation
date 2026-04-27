"""Tests for `core.optim.DifferentialEvolution`."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from constellation.core.optim import DifferentialEvolution
from constellation.core.optim._strategies import (
    CROSSOVER_SCHEMES,
    MUTATION_STRATEGIES,
    _distinct_indices,
)


# ──────────────────────────────────────────────────────────────────────
# Standard test functions (sanity)
# ──────────────────────────────────────────────────────────────────────


class _Sphere(nn.Module):
    """f(x) = Σ xᵢ² — minimum 0 at x=0."""

    def __init__(self, dim: int = 5) -> None:
        super().__init__()
        self.x = nn.Parameter(
            torch.full((dim,), 3.0, dtype=torch.float64)
        )

    def forward(self) -> torch.Tensor:
        return (self.x ** 2).sum()


class _Rosenbrock(nn.Module):
    """Sum_i 100·(x_{i+1} - xᵢ²)² + (1 - xᵢ)² — minimum 0 at x=(1,...,1)."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.x = nn.Parameter(torch.zeros(dim, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        x = self.x
        return (
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
        ).sum()


class _Rastrigin(nn.Module):
    """f(x) = 10·n + Σ (xᵢ² - 10·cos(2π·xᵢ)) — minimum 0 at x=0; multimodal."""

    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.x = nn.Parameter(
            torch.full((dim,), 2.5, dtype=torch.float64)
        )

    def forward(self) -> torch.Tensor:
        x = self.x
        return 10.0 * x.numel() + (x ** 2 - 10.0 * torch.cos(2.0 * math.pi * x)).sum()


class _Ackley(nn.Module):
    """Ackley function — minimum 0 at x=0; many local minima."""

    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.x = nn.Parameter(torch.full((dim,), 1.5, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        x = self.x
        n = x.numel()
        a, b, c = 20.0, 0.2, 2.0 * math.pi
        s1 = (x ** 2).sum() / n
        s2 = torch.cos(c * x).sum() / n
        return -a * torch.exp(-b * torch.sqrt(s1)) - torch.exp(s2) + a + math.e


def _drive_to_convergence(de: DifferentialEvolution, model: nn.Module, max_iter: int = 500):
    def pop_closure(model_view):
        return model_view()

    i = 0
    while not de.is_converged and i < max_iter:
        de.step(pop_closure)
        i += 1
    return i


@pytest.mark.parametrize(
    "fn_cls,dim,bounds_lo,bounds_hi,target,strategy",
    [
        (_Sphere, 5, -5.0, 5.0, 1e-4, "best1"),
        (_Rosenbrock, 4, -2.0, 2.0, 1.0, "best1"),
        # Rastrigin is highly multimodal — needs an exploratory strategy
        # and a larger population. The random strategies escape the
        # tight local basins (Δ=1.0 between local minima) more reliably.
        (_Rastrigin, 3, -5.12, 5.12, 2.0, "rand1"),
        (_Ackley, 3, -32.0, 32.0, 1e-2, "best1"),
    ],
)
def test_de_converges_on_standard_functions(
    fn_cls, dim, bounds_lo, bounds_hi, target, strategy
):
    torch.manual_seed(0)
    m = fn_cls(dim) if dim != fn_cls().x.numel() else fn_cls()
    de = DifferentialEvolution(
        m,
        bounds={"x": (bounds_lo, bounds_hi)},
        pop_size=60,
        seed=0,
        strategy=strategy,
        max_evals=60000,
        patience=120,
    )
    _drive_to_convergence(de, m, max_iter=1500)
    assert de.best_fitness <= target, f"{fn_cls.__name__}: best={de.best_fitness:.3e}"


# ──────────────────────────────────────────────────────────────────────
# Bounds enforcement
# ──────────────────────────────────────────────────────────────────────


def test_population_never_escapes_bounds():
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-2.0, 2.0)}, pop_size=20, seed=0
    )
    def pop_closure(model_view):
        return model_view()
    for _ in range(50):
        de.step(pop_closure)
        # Inspect the population directly
        pop = de._population
        assert (pop >= -2.0 - 1e-12).all()
        assert (pop <= 2.0 + 1e-12).all()


def test_reflect_bounds_handling_works():
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-1.0, 1.0)}, pop_size=20, seed=0,
        bounds_handling="reflect",
    )
    def pop_closure(model_view):
        return model_view()
    for _ in range(50):
        de.step(pop_closure)
        pop = de._population
        assert (pop >= -1.0 - 1e-12).all()
        assert (pop <= 1.0 + 1e-12).all()


# ──────────────────────────────────────────────────────────────────────
# Sobol warm-start
# ──────────────────────────────────────────────────────────────────────


def test_sobol_warm_start_places_current_params_at_slot_0():
    m = _Sphere(3)
    # Move x off the default; warm-start must bring it back to slot 0.
    with torch.no_grad():
        m.x.copy_(torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64))
    de = DifferentialEvolution(
        m, bounds={"x": (-3.0, 3.0)}, pop_size=10, seed=0,
        warm_start=True,
    )
    expected = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64)
    assert torch.allclose(de._population[0], expected)


# ──────────────────────────────────────────────────────────────────────
# Vectorized population eval shape
# ──────────────────────────────────────────────────────────────────────


def test_model_view_returns_population_leading_dim():
    """Verify the vmap'd model_view broadcasts forward over a leading
    population dim of the right size."""
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-2.0, 2.0)}, pop_size=12, seed=0
    )

    captured = {}

    def pop_closure(model_view):
        out = model_view()
        captured["shape"] = out.shape
        return out

    de.step(pop_closure)
    assert captured["shape"] == (12,)


def test_per_individual_loss_matches_scalar_reference():
    """Compare vectorized eval against per-individual scalar eval to
    ensure vmap doesn't mix individuals."""
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-3.0, 3.0)}, pop_size=8, seed=0
    )
    pop = de._population.clone()

    # Vectorized
    def pop_closure(model_view):
        return model_view()
    vec_losses = de._evaluate(pop, pop_closure)

    # Scalar reference
    from torch.func import functional_call
    scalar = []
    for i in range(pop.shape[0]):
        params = {"x": pop[i]}
        scalar.append(float(functional_call(m, params).item()))
    scalar_t = torch.tensor(scalar, dtype=torch.float64)

    assert torch.allclose(vec_losses, scalar_t, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Polish
# ──────────────────────────────────────────────────────────────────────


def test_polish_lowers_fitness_and_respects_bounds():
    """DE alone might land near but not at the optimum; polish must
    refine it without leaving the box."""
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-2.0, 2.0)}, pop_size=20, seed=0,
        patience=10,  # stop early so polish has work to do
    )
    def pop_closure(model_view):
        return model_view()
    for _ in range(15):
        de.step(pop_closure)

    pre = de.best_fitness
    def scalar_closure():
        return m()
    de.polish(scalar_closure)
    assert de.best_fitness <= pre + 1e-12
    # Bounds still respected
    assert (-2.0 <= m.x.detach()).all() and (m.x.detach() <= 2.0).all()


def test_polish_does_not_escape_tight_bounds():
    """Regression for the cartographer 0cee299 fix: polish on a peak
    whose unconstrained optimum lies outside bounds must NOT escape."""
    class Quad(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        def forward(self):
            return (self.x - 100.0) ** 2  # unconstrained min at x=100

    m = Quad()
    de = DifferentialEvolution(
        m, bounds={"x": (-1.0, 1.0)}, pop_size=10, seed=0,
        patience=5,
    )
    def pop_closure(model_view):
        return model_view()
    for _ in range(10):
        de.step(pop_closure)
    def scalar_closure():
        return m()
    de.polish(scalar_closure)
    assert -1.0 - 1e-12 <= float(m.x.detach()) <= 1.0 + 1e-12


# ──────────────────────────────────────────────────────────────────────
# Mutation strategies — table test on Sphere
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", sorted(MUTATION_STRATEGIES.keys()))
def test_each_mutation_strategy_makes_progress_on_sphere(strategy):
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-5.0, 5.0)}, pop_size=30, seed=0,
        strategy=strategy, patience=80, max_evals=20000,
    )
    initial = float((m.x ** 2).sum().item())
    _drive_to_convergence(de, m, max_iter=500)
    assert de.best_fitness < initial * 0.05, (
        f"strategy={strategy}: initial={initial:.2e}, best={de.best_fitness:.2e}"
    )


@pytest.mark.parametrize("crossover", sorted(CROSSOVER_SCHEMES.keys()))
def test_each_crossover_scheme_converges(crossover):
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-5.0, 5.0)}, pop_size=30, seed=0,
        crossover=crossover, patience=80, max_evals=20000,
    )
    _drive_to_convergence(de, m, max_iter=500)
    assert de.best_fitness < 1e-3, (
        f"crossover={crossover}: best={de.best_fitness:.2e}"
    )


def test_distinct_indices_invariant():
    gen = torch.Generator()
    gen.manual_seed(0)
    pop_size, k = 8, 5
    idx = _distinct_indices(pop_size, k, gen)
    for i in range(pop_size):
        row = idx[i].tolist()
        assert len(set(row)) == k, f"row {i} has duplicates: {row}"
        assert i not in row, f"row {i} contains self-index: {row}"


# ──────────────────────────────────────────────────────────────────────
# F/CR dithering
# ──────────────────────────────────────────────────────────────────────


def test_F_CR_dither_tuple_inputs():
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-3.0, 3.0)}, pop_size=20, seed=0,
        F=(0.5, 1.0), CR=(0.5, 1.0), patience=50, max_evals=15000,
    )
    _drive_to_convergence(de, m, max_iter=300)
    assert de.best_fitness < 1e-2


# ──────────────────────────────────────────────────────────────────────
# Convergence flags
# ──────────────────────────────────────────────────────────────────────


def test_target_fitness_triggers_convergence():
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-5.0, 5.0)}, pop_size=20, seed=0,
        target_fitness=1.0,
    )
    def pop_closure(model_view):
        return model_view()
    for _ in range(200):
        de.step(pop_closure)
        if de.is_converged:
            break
    assert de.is_converged
    assert de.best_fitness <= 1.0


def test_max_evals_triggers_convergence():
    m = _Sphere(3)
    de = DifferentialEvolution(
        m, bounds={"x": (-5.0, 5.0)}, pop_size=20, seed=0,
        max_evals=100,  # ≤ ceil(100/20) = 5 generations of trial evals + 1 init eval
    )
    def pop_closure(model_view):
        return model_view()
    for _ in range(100):
        de.step(pop_closure)
        if de.is_converged:
            break
    assert de.is_converged


# ──────────────────────────────────────────────────────────────────────
# Vmap-compatibility smoke test for shipped Parametric subclasses
# ──────────────────────────────────────────────────────────────────────


def _make_subclasses():
    """Instantiate each shipped Parametric subclass with default args."""
    from constellation.core.stats import (
        Beta,
        Dirichlet,
        EMGPeak,
        Gamma,
        GaussianPeak,
        GeneralizedNormal,
        Hill,
        LogLinear,
        LogNormal,
        Multinomial,
        NormalDistribution,
        Poisson,
        Sigmoidal,
        StudentT,
    )
    return [
        ("NormalDistribution", NormalDistribution(), torch.linspace(-2, 2, 7, dtype=torch.float64), "log_prob"),
        ("StudentT", StudentT(), torch.linspace(-2, 2, 7, dtype=torch.float64), "log_prob"),
        ("GeneralizedNormal", GeneralizedNormal(), torch.linspace(-2, 2, 7, dtype=torch.float64), "log_prob"),
        ("Beta", Beta(2.0, 5.0), torch.linspace(0.1, 0.9, 5, dtype=torch.float64), "log_prob"),
        ("Gamma", Gamma(2.0, 1.0), torch.linspace(0.1, 5.0, 5, dtype=torch.float64), "log_prob"),
        ("LogNormal", LogNormal(), torch.linspace(0.1, 5.0, 5, dtype=torch.float64), "log_prob"),
        ("Poisson", Poisson(3.0), torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0], dtype=torch.float64), "log_prob"),
        ("GaussianPeak", GaussianPeak(), torch.linspace(-3, 3, 9, dtype=torch.float64), "forward"),
        ("EMGPeak", EMGPeak(), torch.linspace(-3, 5, 9, dtype=torch.float64), "forward"),
        ("Sigmoidal", Sigmoidal(), torch.linspace(-3, 3, 9, dtype=torch.float64), "forward"),
        ("Hill", Hill(), torch.linspace(0.1, 5.0, 9, dtype=torch.float64), "forward"),
        ("LogLinear", LogLinear(), torch.linspace(0.1, 5.0, 9, dtype=torch.float64), "forward"),
        # Multinomial / Dirichlet take vector data; skipped from the
        # smoke test until DE-friendly count-data fits land. They're
        # not blocked from DE — the user just needs a custom loss_fn
        # with the right reduction over vector dims.
    ]


@pytest.mark.parametrize(
    "name,model,data,method",
    _make_subclasses(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_subclass_is_vmap_compatible(name, model, data, method):
    """Each shipped Parametric subclass must support batched parameter
    substitution — one call with population dim 4 should return a
    (4, *data_shape) tensor."""
    from torch.func import functional_call, vmap

    pop_size = 4
    batched_params = {}
    for pname, p in model.named_parameters():
        # Replicate scalar params with leading pop dim; for vector params,
        # add small jitter so each individual differs.
        if p.dim() == 0:
            batched_params[pname] = (
                p.detach().expand(pop_size).clone()
                + 0.01 * torch.linspace(-1, 1, pop_size, dtype=torch.float64)
            )
        else:
            batched_params[pname] = (
                p.detach().unsqueeze(0).expand(pop_size, *p.shape).clone()
            )

    # Distribution.forward defaults to log_prob, so functional_call(model, ...)
    # exercises the right code path for both branches in `method`.
    del method  # informational only — both branches go through forward

    def fmodel(params):
        return functional_call(model, params, data)

    out = vmap(fmodel, in_dims=(0,))(batched_params)
    assert out.shape[0] == pop_size, f"{name}: leading dim {out.shape[0]} ≠ {pop_size}"


# ──────────────────────────────────────────────────────────────────────
# End-to-end through Parametric.fit()
# ──────────────────────────────────────────────────────────────────────


def test_de_fits_gaussian_peak_through_parametric_fit():
    """DE drives a noisy peak fit through the standard Parametric.fit
    interface — proves it's a drop-in replacement for LBFGS on the
    GaussianPeak workload."""
    from constellation.core.stats import GaussianPeak

    torch.manual_seed(7)
    t = torch.linspace(0, 30, 200, dtype=torch.float64)
    true_peak = GaussianPeak(N_max=10.0, t_apex=15.0, sigma=2.0)
    y = true_peak.forward(t).detach() + 0.05 * torch.randn_like(t)

    fit_peak = GaussianPeak(N_max=1.0, t_apex=12.0, sigma=4.0)
    de = DifferentialEvolution(
        fit_peak,
        bounds={"log_N_max": (-2, 5), "t_apex": (0, 30), "log_sigma": (-3, 3)},
        pop_size=30,
        seed=11,
        patience=40,
        max_evals=20000,
    )
    res = fit_peak.fit(t, y, optimizer=de, max_iter=500)
    pd = fit_peak.parameters_dict()
    assert float(pd["t_apex"]) == pytest.approx(15.0, abs=0.3)
    assert float(pd["sigma"]) == pytest.approx(2.0, rel=0.2)
    assert float(pd["N_max"]) == pytest.approx(10.0, rel=0.2)
    assert math.isfinite(res.final_loss)


def test_de_short_circuits_parametric_fit_on_convergence():
    from constellation.core.stats import GaussianPeak

    torch.manual_seed(1)
    t = torch.linspace(0, 10, 50, dtype=torch.float64)
    y = GaussianPeak(N_max=5.0, t_apex=5.0, sigma=1.0).forward(t).detach()

    fit_peak = GaussianPeak(N_max=1.0, t_apex=4.0, sigma=2.0)
    de = DifferentialEvolution(
        fit_peak,
        bounds={"log_N_max": (-2, 5), "t_apex": (0, 10), "log_sigma": (-3, 3)},
        pop_size=20,
        seed=0,
        patience=15,
    )
    res = fit_peak.fit(t, y, optimizer=de, max_iter=500)
    assert res.converged or res.n_iter < 500


# ──────────────────────────────────────────────────────────────────────
# Construction validation
# ──────────────────────────────────────────────────────────────────────


def test_unknown_strategy_raises():
    m = _Sphere(2)
    with pytest.raises(ValueError, match="strategy"):
        DifferentialEvolution(m, strategy="not_a_strategy", seed=0)


def test_unknown_crossover_raises():
    m = _Sphere(2)
    with pytest.raises(ValueError, match="crossover"):
        DifferentialEvolution(m, crossover="not_a_crossover", seed=0)


def test_pop_size_too_small_raises():
    m = _Sphere(2)
    with pytest.raises(ValueError, match="pop_size"):
        DifferentialEvolution(m, pop_size=2, seed=0)


def test_default_pop_size_scales_with_n_params():
    m = _Sphere(5)  # 5 params → 15*5=75
    de = DifferentialEvolution(m, seed=0)
    assert de._pop_size == 75
