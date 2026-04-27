"""DE mutation and crossover schemes — registry-driven, vectorized.

Each mutation strategy is a function
    `(population, best_idx, F, generator) -> mutants`
returning a `(pop_size, n_flat)` tensor.

Distinctness invariant: the per-individual auxiliary indices `r1, r2, ...`
are sampled WITHOUT replacement and ALL distinct from `i` (the target
index). Cartographer's relaxed-distinctness implementation
(`torch.randint` independently per slot) is intentionally not carried
forward — it produces redundant mutations on small populations.

Crossover: binomial (default) and exponential. Both guarantee at least
one trial coordinate comes from the mutant (`j_rand`).
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

MutationFn = Callable[[torch.Tensor, int, float, torch.Generator], torch.Tensor]
CrossoverFn = Callable[
    [torch.Tensor, torch.Tensor, float, torch.Generator], torch.Tensor
]


# ──────────────────────────────────────────────────────────────────────
# Distinct-index sampling
# ──────────────────────────────────────────────────────────────────────


def _distinct_indices(
    pop_size: int, k: int, generator: torch.Generator
) -> torch.Tensor:
    """For each `i` in [0, pop_size), sample `k` indices from
    [0, pop_size) \\ {i} without replacement. Returns shape `(pop_size, k)`.

    Implementation: per-row Fisher-Yates over the candidate set
    `[0, pop_size) \\ {i}`. Vectorized via a sort of random keys —
    fast for the populations we run (≤200) and correct for any size.
    """
    if pop_size <= k:
        raise ValueError(
            f"need pop_size > k+1 for distinct sampling; got pop_size={pop_size}, k={k}"
        )
    # (pop_size, pop_size) random keys; mask the diagonal so each row's own
    # index sorts to the end. Take the first k of the sorted argsort.
    keys = torch.rand(pop_size, pop_size, generator=generator)
    diag_idx = torch.arange(pop_size)
    keys[diag_idx, diag_idx] = float("inf")
    return keys.argsort(dim=1)[:, :k]


# ──────────────────────────────────────────────────────────────────────
# Mutation strategies
# ──────────────────────────────────────────────────────────────────────


def _mutate_rand1(
    population: torch.Tensor, best_idx: int, F: float, generator: torch.Generator
) -> torch.Tensor:
    """v_i = x_r1 + F * (x_r2 - x_r3)."""
    pop_size = population.shape[0]
    idx = _distinct_indices(pop_size, 3, generator)
    r1, r2, r3 = idx[:, 0], idx[:, 1], idx[:, 2]
    return population[r1] + F * (population[r2] - population[r3])


def _mutate_rand2(
    population: torch.Tensor, best_idx: int, F: float, generator: torch.Generator
) -> torch.Tensor:
    """v_i = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)."""
    pop_size = population.shape[0]
    idx = _distinct_indices(pop_size, 5, generator)
    r1, r2, r3, r4, r5 = (idx[:, k] for k in range(5))
    return (
        population[r1]
        + F * (population[r2] - population[r3])
        + F * (population[r4] - population[r5])
    )


def _mutate_best1(
    population: torch.Tensor, best_idx: int, F: float, generator: torch.Generator
) -> torch.Tensor:
    """v_i = x_best + F * (x_r1 - x_r2)."""
    pop_size = population.shape[0]
    idx = _distinct_indices(pop_size, 2, generator)
    r1, r2 = idx[:, 0], idx[:, 1]
    return population[best_idx].unsqueeze(0) + F * (population[r1] - population[r2])


def _mutate_best2(
    population: torch.Tensor, best_idx: int, F: float, generator: torch.Generator
) -> torch.Tensor:
    """v_i = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)."""
    pop_size = population.shape[0]
    idx = _distinct_indices(pop_size, 4, generator)
    r1, r2, r3, r4 = (idx[:, k] for k in range(4))
    return (
        population[best_idx].unsqueeze(0)
        + F * (population[r1] - population[r2])
        + F * (population[r3] - population[r4])
    )


def _mutate_current_to_best1(
    population: torch.Tensor, best_idx: int, F: float, generator: torch.Generator
) -> torch.Tensor:
    """v_i = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)."""
    pop_size = population.shape[0]
    idx = _distinct_indices(pop_size, 2, generator)
    r1, r2 = idx[:, 0], idx[:, 1]
    best = population[best_idx].unsqueeze(0)
    return population + F * (best - population) + F * (population[r1] - population[r2])


def _mutate_current_to_rand1(
    population: torch.Tensor, best_idx: int, F: float, generator: torch.Generator
) -> torch.Tensor:
    """v_i = x_i + F * (x_r1 - x_i) + F * (x_r2 - x_r3). Stochastic
    counterpart of current-to-best/1 — the attractor is a random
    population member, not the current best."""
    pop_size = population.shape[0]
    idx = _distinct_indices(pop_size, 3, generator)
    r1, r2, r3 = idx[:, 0], idx[:, 1], idx[:, 2]
    return (
        population
        + F * (population[r1] - population)
        + F * (population[r2] - population[r3])
    )


MUTATION_STRATEGIES: dict[str, MutationFn] = {
    "rand1": _mutate_rand1,
    "rand2": _mutate_rand2,
    "best1": _mutate_best1,
    "best2": _mutate_best2,
    "current_to_best1": _mutate_current_to_best1,
    "current_to_rand1": _mutate_current_to_rand1,
}

MutationStrategy = Literal[
    "rand1", "rand2", "best1", "best2", "current_to_best1", "current_to_rand1"
]


# ──────────────────────────────────────────────────────────────────────
# Crossover
# ──────────────────────────────────────────────────────────────────────


def _crossover_binomial(
    population: torch.Tensor,
    mutants: torch.Tensor,
    CR: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Standard binomial crossover. Each coordinate inherits from the
    mutant with probability `CR`, from the parent otherwise. Per-row
    `j_rand` guarantees at least one mutant coordinate so trial ≠ parent."""
    pop_size, n_flat = population.shape
    mask = torch.rand(pop_size, n_flat, generator=generator) < CR
    j_rand = torch.randint(0, n_flat, (pop_size,), generator=generator)
    mask[torch.arange(pop_size), j_rand] = True
    return torch.where(mask, mutants, population)


def _crossover_exponential(
    population: torch.Tensor,
    mutants: torch.Tensor,
    CR: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Exponential crossover. Starting at a random position `j_start`,
    successive coordinates inherit from the mutant until the streak
    breaks (each successive position passes the inheritance test with
    probability `CR`). Always copies at least the starting position so
    trial ≠ parent.

    Generates a (pop_size, n_flat) "active" mask. Vectorized: use the
    cumulative AND of per-position Bernoulli draws starting from
    `j_start`, rotated per-row.
    """
    pop_size, n_flat = population.shape
    j_start = torch.randint(0, n_flat, (pop_size,), generator=generator)

    # Per-position Bernoulli(CR), but the first slot in the rotated
    # ordering is forced True (the j_start coord always inherits).
    bern = torch.rand(pop_size, n_flat, generator=generator) < CR
    bern[:, 0] = True
    streak = bern.cumprod(dim=1).bool()   # streak breaks at the first False

    # Rotate so that streak position 0 lands at column j_start.
    cols = torch.arange(n_flat).unsqueeze(0).expand(pop_size, -1)
    src = (cols - j_start.unsqueeze(1)) % n_flat
    mask = streak.gather(1, src)
    return torch.where(mask, mutants, population)


CROSSOVER_SCHEMES: dict[str, CrossoverFn] = {
    "binomial": _crossover_binomial,
    "exponential": _crossover_exponential,
}

Crossover = Literal["binomial", "exponential"]
