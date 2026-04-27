"""`DifferentialEvolution` — population-based gradient-free optimizer.

Generic over `nn.Module`. One `step()` advances one generation; the
caller drives the outer loop (typically `Parametric.fit()` watching
`is_converged`). Vectorized population evaluation via
`torch.func.vmap(torch.func.functional_call(...))` — the model's
`forward` is invoked once per generation, with all `pop_size`
candidate parameter sets stacked as a leading batch dim.

Vectorization implication: every `Parametric` subclass's `forward`
(and `log_prob` / `cdf` for `Distribution`) must be `vmap`-compatible.
In practice this means using `torch.where` instead of boolean
indexing for branch logic; basic broadcasting torch ops Just Work.

Mutation strategies, crossover schemes, init methods, and bounds
helpers all live in sibling modules so the DE class stays focused on
the evolutionary loop itself.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.func import functional_call, vmap

from ._bounds import apply_bounds, resolve_bounds
from ._init import initialize_population
from ._layout import (
    build_layout,
    flatten_params,
    params_dict_from_flat,
    params_dict_from_population,
    unflatten_into_model,
)
from ._lbfgs import LBFGSOptimizer
from ._protocol import PopulationClosure, ScalarClosure
from ._strategies import (
    CROSSOVER_SCHEMES,
    MUTATION_STRATEGIES,
    Crossover,
    MutationStrategy,
)


class DifferentialEvolution:
    """Differential evolution over an `nn.Module`'s parameters.

    Parameters
    ----------
    model:
        The module being optimized. The model's parameters are read at
        construction (for layout + warm-start) and written back to
        match the global-best individual after each `step()` and again
        after `polish()`.
    bounds:
        `dict[name, (lo, hi)]` of per-parameter box bounds. Applied to
        the parameter as registered (e.g. `log_sigma`); use
        `core.optim.bounds_in_natural_units` to convert from natural
        units. Unspecified parameters get ±1e10 (effectively unbounded
        but bracketable for init).
    pop_size:
        Population size. Defaults to `max(15 * n_params, 20)`.
    F, CR:
        Mutation factor and crossover rate. Scalar OR `(lo, hi)` tuple
        for per-generation dithering.
    strategy:
        One of "rand1", "rand2", "best1", "best2", "current_to_best1",
        "current_to_rand1".
    crossover:
        "binomial" (default) or "exponential".
    init:
        "sobol" (default; warm-started), "lhs", or "uniform".
    bounds_handling:
        "clip" (default) or "reflect" applied to trial vectors.
    warm_start:
        If True (default), population[0] is the model's current
        parameters clamped into bounds.
    max_evals:
        Hard budget on closure evaluations (one closure call per
        generation evaluates `pop_size` individuals). Optional.
    tol:
        Convergence trips when `fitness.std() < tol`.
    patience:
        Convergence trips after this many generations without an
        improvement to the global best.
    target_fitness:
        Convergence trips if best fitness ≤ this value.
    seed:
        For reproducibility — seeds an internal `torch.Generator`.
    dtype, device:
        Population storage dtype/device. Cartographer locked these to
        float64/CPU; same default here. GPU population eval works for
        small models but the speed-up is marginal vs. CPU on the
        peak-fitting workloads this was built for.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        bounds: dict[str, tuple[float, float]] | None = None,
        pop_size: int | None = None,
        F: float | tuple[float, float] = 0.7,
        CR: float | tuple[float, float] = 0.9,
        strategy: MutationStrategy = "best1",
        crossover: Crossover = "binomial",
        init: Literal["sobol", "lhs", "uniform"] = "sobol",
        bounds_handling: Literal["clip", "reflect"] = "clip",
        warm_start: bool = True,
        max_evals: int | None = None,
        tol: float = 1e-8,
        patience: int = 30,
        target_fitness: float | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        if strategy not in MUTATION_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; want one of "
                f"{sorted(MUTATION_STRATEGIES)}"
            )
        if crossover not in CROSSOVER_SCHEMES:
            raise ValueError(
                f"unknown crossover {crossover!r}; want one of "
                f"{sorted(CROSSOVER_SCHEMES)}"
            )
        self._model = model
        self._bounds_dict = bounds
        self._strategy = strategy
        self._crossover = crossover
        self._bounds_handling = bounds_handling
        self._F = F
        self._CR = CR
        self._tol = float(tol)
        self._patience = int(patience)
        self._target_fitness = (
            float(target_fitness) if target_fitness is not None else None
        )
        self._max_evals = max_evals
        self._dtype = dtype
        self._device = torch.device(device)

        self._layout = build_layout(model)
        self._lower, self._upper = resolve_bounds(
            self._layout, bounds, dtype=dtype, device=self._device
        )

        if pop_size is None:
            pop_size = max(15 * self._layout.n_flat, 20)
        elif pop_size < 5:
            raise ValueError(f"pop_size must be ≥ 5; got {pop_size}")
        self._pop_size = pop_size

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

        current_flat = flatten_params(model, self._layout).to(dtype).to(self._device)
        self._population = initialize_population(
            init,
            self._pop_size,
            current_flat,
            self._lower,
            self._upper,
            seed=seed,
            dtype=dtype,
            warm_start=warm_start,
        )
        self._fitness = torch.full(
            (self._pop_size,), float("inf"), dtype=dtype, device=self._device
        )
        self._best_idx = 0
        self._best_fitness = float("inf")
        self._gens_without_improvement = 0
        self._n_generations = 0
        self._n_evals = 0
        self._converged = False
        self._first_step = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_converged(self) -> bool:
        return self._converged

    @property
    def best_fitness(self) -> float:
        return self._best_fitness

    @property
    def n_generations(self) -> int:
        return self._n_generations

    @property
    def n_evals(self) -> int:
        return self._n_evals

    def step(self, closure: PopulationClosure) -> torch.Tensor:
        """Advance one generation. Returns current best fitness as 0-d tensor.

        On the FIRST call, the initial population is evaluated to seed
        `self._fitness`; mutation/crossover then runs as usual. This
        keeps the Protocol contract clean — every `.step()` makes
        forward progress.
        """
        if self._first_step:
            self._fitness = self._evaluate(self._population, closure)
            self._best_idx = int(torch.argmin(self._fitness).item())
            self._best_fitness = float(self._fitness[self._best_idx].item())
            self._first_step = False

        F = self._sample_param(self._F)
        CR = self._sample_param(self._CR)

        mutants = MUTATION_STRATEGIES[self._strategy](
            self._population, self._best_idx, F, self._generator
        )
        trials = CROSSOVER_SCHEMES[self._crossover](
            self._population, mutants, CR, self._generator
        )
        trials = apply_bounds(
            trials, self._lower, self._upper, mode=self._bounds_handling
        )

        trial_fitness = self._evaluate(trials, closure)

        # Greedy selection: trial wins iff it strictly improves on the parent.
        better = trial_fitness < self._fitness
        self._population = torch.where(
            better.unsqueeze(1), trials, self._population
        )
        self._fitness = torch.where(better, trial_fitness, self._fitness)

        new_best_idx = int(torch.argmin(self._fitness).item())
        new_best_fitness = float(self._fitness[new_best_idx].item())
        if new_best_fitness < self._best_fitness:
            self._best_idx = new_best_idx
            self._best_fitness = new_best_fitness
            self._gens_without_improvement = 0
        else:
            self._gens_without_improvement += 1

        self._n_generations += 1

        # Push global-best back into the model so callers can inspect.
        unflatten_into_model(
            self._model, self._population[self._best_idx], self._layout
        )

        self._update_convergence()
        return torch.as_tensor(self._best_fitness, dtype=self._dtype)

    def polish(
        self, scalar_closure: ScalarClosure, **lbfgs_kwargs
    ) -> torch.Tensor:
        """Terminal LBFGS refinement of the global-best individual.

        `scalar_closure` is a gradient-aware closure (DE's population
        closure runs under `no_grad`; LBFGS needs autograd active).
        Bounds are passed through to `LBFGSOptimizer` so the same
        clamping path applies — no separate polish-bounds bug surface.
        """
        # Ensure the model holds the current best (defensive — step()
        # already writes it, but a polish-without-step would not).
        unflatten_into_model(
            self._model, self._population[self._best_idx], self._layout
        )
        polisher = LBFGSOptimizer(
            self._model, bounds=self._bounds_dict, **lbfgs_kwargs
        )
        polisher.step(scalar_closure)
        # `torch.optim.LBFGS.step` returns the loss from the first closure
        # call (before line search), so re-evaluate after the step to get
        # the actual polished fitness.
        with torch.no_grad():
            polished_loss_t = scalar_closure()
        polished_loss = float(polished_loss_t.detach().item())
        # Capture the polished position back into the population so a
        # subsequent .step() benefits from the refinement.
        polished_flat = flatten_params(self._model, self._layout).to(self._dtype)
        self._population[self._best_idx] = polished_flat
        self._fitness[self._best_idx] = polished_loss
        if polished_loss < self._best_fitness:
            self._best_fitness = polished_loss
            self._best_idx = self._best_idx
        return torch.as_tensor(polished_loss, dtype=self._dtype)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(
        self, population: torch.Tensor, closure: PopulationClosure
    ) -> torch.Tensor:
        """Vectorized loss evaluation over a `(pop_size, n_flat)` population.

        Builds a model_view callable that vmap-substitutes the batched
        params, hands it to the user's closure, and returns the
        `(pop_size,)` loss vector. NaN/Inf losses are coerced to +inf
        so DE never selects a numerical blowup as "best".
        """
        batched_params = params_dict_from_population(population, self._layout)
        model = self._model

        # vmap over the population dim of `params` only; data args flow in
        # via Python closure capture so vmap doesn't need to know how many
        # there are. This also lets a model with no positional args
        # (e.g. test functions like Sphere) be evaluated cleanly.
        def model_view(*args, **kwargs):
            def fmodel(params):
                return functional_call(model, params, args, kwargs)

            return vmap(fmodel, in_dims=(0,))(batched_params)

        with torch.no_grad():
            losses = closure(model_view)
        if not torch.is_tensor(losses):
            raise TypeError(
                f"population closure must return a Tensor; got {type(losses).__name__}"
            )
        if losses.shape != (population.shape[0],):
            raise ValueError(
                f"population closure must return shape ({population.shape[0]},); "
                f"got {tuple(losses.shape)}"
            )
        losses = losses.to(self._dtype).to(self._device)
        # Coerce non-finite to +inf — out-of-domain trial vectors must
        # never out-rank a finite parent in selection.
        losses = torch.where(
            torch.isfinite(losses), losses, torch.full_like(losses, float("inf"))
        )
        self._n_evals += int(population.shape[0])
        return losses

    def _sample_param(
        self, value: float | tuple[float, float]
    ) -> float:
        """Resolve scalar-or-dither hyperparameter for this generation."""
        if isinstance(value, tuple):
            lo, hi = value
            u = torch.rand((), generator=self._generator).item()
            return float(lo) + u * (float(hi) - float(lo))
        return float(value)

    def _update_convergence(self) -> None:
        if self._fitness.std().item() < self._tol:
            self._converged = True
            return
        if self._gens_without_improvement >= self._patience:
            self._converged = True
            return
        if (
            self._target_fitness is not None
            and self._best_fitness <= self._target_fitness
        ):
            self._converged = True
            return
        if self._max_evals is not None and self._n_evals >= self._max_evals:
            self._converged = True
            return

    # ------------------------------------------------------------------
    # Inspection helpers (not part of either Protocol; user-facing convenience)
    # ------------------------------------------------------------------

    def best_params_dict(self) -> dict[str, torch.Tensor]:
        """Named-tensor dict for the current global best — useful for
        logging without mutating the model."""
        return params_dict_from_flat(self._population[self._best_idx], self._layout)
