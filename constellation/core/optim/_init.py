"""Population initialization schemes for DifferentialEvolution.

Three methods:
    sobol           low-discrepancy quasi-random; default. Slot 0 is the
                    warm-start (current model parameters); the rest fill
                    a Sobol sequence over the bounds.
    lhs             Latin hypercube — stratified random sample, one per
                    bin per dimension.
    uniform         plain torch.rand uniform over the bounds.

Unbounded dimensions (where the resolved bound is ±_DEFAULT_BOUND) get a
local box of `±_INIT_RANGE` centered on the warm-start value rather than
sampling from the absurd ±1e10 range. This carries forward the
cartographer 0cee299 convention; without it, Sobol's first few samples
land 10⁹ units from the warm-start point.
"""

from __future__ import annotations

import torch

from ._bounds import _DEFAULT_BOUND

_INIT_RANGE: float = 3.0


def _resolve_init_box(
    current_flat: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For unbounded dims, narrow the init range to ±_INIT_RANGE around
    the warm-start value. Returns (init_lower, init_upper) the actual
    sampling box; the *enforcement* box (lower, upper) stays unchanged."""
    unbounded = (lower < -_DEFAULT_BOUND + 1) | (upper > _DEFAULT_BOUND - 1)
    init_lo = lower.clone()
    init_hi = upper.clone()
    init_lo[unbounded] = current_flat[unbounded] - _INIT_RANGE
    init_hi[unbounded] = current_flat[unbounded] + _INIT_RANGE
    return init_lo, init_hi


def sobol_population(
    n_pop: int,
    current_flat: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    warm_start: bool = True,
) -> torch.Tensor:
    """Sobol quasi-random initialization. Slot 0 holds the warm-start
    (clamped into bounds) when `warm_start=True`."""
    n_flat = current_flat.numel()
    init_lo, init_hi = _resolve_init_box(current_flat, lower, upper)

    engine = torch.quasirandom.SobolEngine(
        dimension=n_flat, scramble=True, seed=seed
    )
    raw = engine.draw(n_pop, dtype=dtype)
    population = init_lo.unsqueeze(0) + raw * (init_hi - init_lo).unsqueeze(0)
    if warm_start:
        population[0] = current_flat.clamp(lower, upper)
    return population.clamp(lower, upper)


def lhs_population(
    n_pop: int,
    current_flat: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    warm_start: bool = True,
) -> torch.Tensor:
    """Latin hypercube — one sample per stratified bin per dimension.
    Each dimension is split into `n_pop` equal bins; one uniform draw
    per bin; then independently permute each dimension."""
    n_flat = current_flat.numel()
    init_lo, init_hi = _resolve_init_box(current_flat, lower, upper)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # Stratified samples in [0, 1): one per bin, jittered uniformly.
    bin_edges = torch.arange(n_pop, dtype=dtype) / n_pop
    jitter = torch.rand(n_pop, n_flat, generator=gen, dtype=dtype) / n_pop
    samples = bin_edges.unsqueeze(1) + jitter   # (n_pop, n_flat)

    # Permute each dimension independently.
    for j in range(n_flat):
        perm = torch.randperm(n_pop, generator=gen)
        samples[:, j] = samples[perm, j]

    population = init_lo.unsqueeze(0) + samples * (init_hi - init_lo).unsqueeze(0)
    if warm_start:
        population[0] = current_flat.clamp(lower, upper)
    return population.clamp(lower, upper)


def uniform_population(
    n_pop: int,
    current_flat: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    warm_start: bool = True,
) -> torch.Tensor:
    """Plain uniform sampling over the init box."""
    n_flat = current_flat.numel()
    init_lo, init_hi = _resolve_init_box(current_flat, lower, upper)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    raw = torch.rand(n_pop, n_flat, generator=gen, dtype=dtype)
    population = init_lo.unsqueeze(0) + raw * (init_hi - init_lo).unsqueeze(0)
    if warm_start:
        population[0] = current_flat.clamp(lower, upper)
    return population.clamp(lower, upper)


_INITIALIZERS = {
    "sobol": sobol_population,
    "lhs": lhs_population,
    "uniform": uniform_population,
}


def initialize_population(
    method: str,
    n_pop: int,
    current_flat: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    warm_start: bool = True,
) -> torch.Tensor:
    """Dispatch by name."""
    try:
        fn = _INITIALIZERS[method]
    except KeyError:
        raise ValueError(
            f"unknown init method {method!r}; want one of "
            f"{sorted(_INITIALIZERS)}"
        ) from None
    return fn(
        n_pop,
        current_flat,
        lower,
        upper,
        seed=seed,
        dtype=dtype,
        warm_start=warm_start,
    )
