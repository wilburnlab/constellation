"""Monotone paths through a density grid: EncyclopeDIA's greedy walk and the DP path."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from constellation.core.signal.modal_path import greedy_walk, monotone_path


def _cells(i: torch.Tensor, j: torch.Tensor) -> list[tuple[int, int]]:
    return list(zip(i.tolist(), j.tolist()))


def test_a_turn_from_x_to_y_cuts_the_corner() -> None:
    grid = torch.zeros(4, 4)
    grid[1, 0] = 5  # from (0,0): x is densest
    grid[1, 1] = 2  # from (1,0): y is densest
    grid[2, 1] = 1
    grid[2, 2] = 9  # from (1,1): diagonal
    grid[3, 2] = grid[2, 3] = 4  # from (2,2): the sides tie, so diagonal
    i, j = greedy_walk(grid, start=(0, 0))
    # (0,0)'s second copy and the (1,0) corner are cut: the L is a diagonal.
    assert _cells(i, j) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_a_straight_run_keeps_its_points() -> None:
    grid = torch.zeros(5, 5)
    grid[0, 1] = grid[0, 2] = grid[0, 3] = 1.0  # up the first column
    grid[1, 4] = 1.0
    i, j = greedy_walk(grid, start=(0, 0))
    # The first y-step cuts the start's second copy; the run up the column
    # keeps every point.
    assert _cells(i, j) == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4)]


def test_empty_density_is_crossed_diagonally() -> None:
    i, j = greedy_walk(torch.zeros(6, 4), start=(2, 1))
    assert _cells(i, j) == [(1, 0), (2, 1), (2, 1), (3, 2), (4, 3)]


def test_the_walk_starts_at_the_first_maximum() -> None:
    grid = torch.zeros(5, 5)
    grid[2, 3] = grid[3, 1] = 7.0
    i, j = greedy_walk(grid)
    assert (2, 3) in _cells(i, j) and (3, 1) not in _cells(i, j)
    assert _cells(*greedy_walk(torch.zeros(3, 3)))[0] == (0, 0)


def test_a_tie_tolerance_treats_near_equal_neighbours_as_equal() -> None:
    grid = torch.zeros(3, 3, dtype=torch.float64)
    grid[1, 0], grid[0, 1], grid[1, 1] = 1.0, 1.0 + 1e-12, 0.5
    exact = _cells(*greedy_walk(grid, start=(0, 0)))
    tolerant = _cells(*greedy_walk(grid, start=(0, 0), tie_rtol=1e-9))
    assert exact == [(0, 0), (1, 1), (2, 2)]  # y, then x: the corner is cut
    assert tolerant == [(0, 0), (0, 0), (1, 1), (2, 2)]  # a tie: diagonal


def test_corner_cutting_can_compare_mapped_coordinates() -> None:
    grid = torch.zeros(4, 4)
    grid[1, 0] = grid[2, 0] = grid[3, 0] = 5.0  # along x
    by_index = _cells(*greedy_walk(grid, start=(0, 0)))
    # Rows that map to one value look like a column, so each step cuts.
    by_value = _cells(*greedy_walk(grid, start=(0, 0), x_values=torch.zeros(4)))
    assert by_index == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert by_value == [(0, 0), (3, 0)]


def _brute_force(score: torch.Tensor, step: torch.Tensor):
    n_cols, n_rows = score.shape
    p = step.numel() // 2
    best, best_path, paths, weights = -math.inf, None, [], []
    for path in itertools.product(range(n_rows), repeat=n_cols):
        moves = [b - a for a, b in zip(path, path[1:])]
        if any(abs(k) > p for k in moves):
            continue
        total = sum(float(score[c, r]) for c, r in enumerate(path)) + sum(float(step[k + p]) for k in moves)
        paths.append(path)
        weights.append(total)
        if total > best:
            best, best_path = total, path
    w = torch.tensor(weights, dtype=torch.float64)
    prob = torch.softmax(w, 0)
    rows = torch.tensor(paths, dtype=torch.float64)
    mean = (prob[:, None] * rows).sum(0)
    sd = torch.sqrt((prob[:, None] * (rows - mean) ** 2).sum(0))
    return best_path, mean, sd, float(torch.logsumexp(w, 0))


def test_the_dp_path_is_the_brute_force_optimum_and_posterior() -> None:
    torch.manual_seed(6)
    score = torch.randn(5, 6, dtype=torch.float64)
    step = torch.tensor([-1.5, -0.3, 0.0, -0.4, -2.0], dtype=torch.float64)
    post = monotone_path(score, step)
    best, mean, sd, log_z = _brute_force(score, step)
    assert tuple(post.viterbi.tolist()) == best
    assert torch.allclose(post.mean, mean, atol=1e-10)
    assert torch.allclose(post.sd, sd, atol=1e-10)
    assert post.log_evidence == pytest.approx(log_z)


def test_forbidden_moves_are_never_taken() -> None:
    torch.manual_seed(7)
    score = torch.randn(30, 40, dtype=torch.float64)
    step = torch.tensor([-math.inf, 0.0, 0.0, 0.0, -math.inf], dtype=torch.float64)
    post = monotone_path(score, step, posterior=False)
    assert int(torch.diff(post.viterbi).abs().max()) <= 1


def test_mirroring_the_rows_mirrors_the_path() -> None:
    torch.manual_seed(8)
    score = torch.randn(12, 21, dtype=torch.float64)
    step = -0.5 * torch.arange(-3, 4, dtype=torch.float64) ** 2
    a = monotone_path(score, step)
    b = monotone_path(score.flip(1), step)
    assert torch.equal(a.viterbi, 20 - b.viterbi)
    assert torch.allclose(a.mean, 20 - b.mean, atol=1e-9)


def test_a_ridge_is_followed() -> None:
    rows = torch.arange(50, dtype=torch.float64)
    centre = 10 + 0.6 * torch.arange(40, dtype=torch.float64)
    score = -0.5 * ((rows[None, :] - centre[:, None]) / 1.5) ** 2
    step = -0.1 * torch.arange(-2, 3, dtype=torch.float64) ** 2
    post = monotone_path(score, step)
    assert float((post.mean - centre).abs().max()) < 0.5
