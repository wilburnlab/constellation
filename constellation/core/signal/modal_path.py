"""Monotone paths through the mode of a gridded density.

Two ways to trace where a 2-D point cloud is densest while only ever
moving up and to the right — the shape of a calibration between two
axes that measure the same thing (retention times of one peptide in two
runs, say):

``greedy_walk``
    EncyclopeDIA's walk (Searle et al. 2018). From the densest cell,
    step toward whichever of the three forward neighbours (x, diagonal,
    y) is densest — diagonally if the diagonal is the densest or the two
    side neighbours tie, which carries the walk diagonally through empty
    stretches — until an edge; then the same backward. At each turn from
    one axis to the other the corner point is dropped, so an L becomes a
    diagonal. Fast and local: it cannot recover from a wrong turn, and it
    stops at the first edge it reaches.

``monotone_path``
    The globally best path, by dynamic programming. A path takes one row
    per column and moves by a bounded number of rows between columns;
    its score is the sum of per-cell scores plus a log-prior on each
    move. Viterbi gives the best path; forward–backward gives, for each
    column, the posterior mean and SD of the row, a sub-cell estimate
    whose spread widens where the density is flat. The caller decides
    what columns and rows are (``core.signal.alignment`` uses a frame
    rotated 45° so that a bounded move is a bounded slope).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["greedy_walk", "monotone_path", "PathPosterior"]


def greedy_walk(
    grid: torch.Tensor,
    *,
    start: tuple[int, int] | None = None,
    tie_rtol: float = 0.0,
    tie_atol: float = 0.0,
    corner_cut: bool = True,
    x_values: torch.Tensor | None = None,
    y_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy monotone walk through ``grid`` from its maximum.

    Returns the visited cells' ``(i, j)`` indices sorted by ``(i, j)``.
    The start cell appears twice (once per direction), as in
    EncyclopeDIA. ``start`` defaults to the first maximum in row-major
    order (``(0, 0)`` for an all-zero grid).

    Two values are equal for the walk's rules when they differ by at
    most ``tie_atol + tie_rtol · max(|a|, |b|)``; exact equality (the
    default) is what a grid accumulated in a fixed order needs, a
    tolerance what an FFT-convolved grid needs. Corner cutting compares
    the cells' coordinates; pass ``x_values`` / ``y_values`` (one value
    per row / column) to compare mapped coordinates instead, as the jar
    does.
    """
    if grid.dim() != 2:
        raise ValueError(f"grid must be 2-D, got shape {tuple(grid.shape)}")
    nx, ny = grid.shape
    if start is None:
        flat = int(torch.argmax(grid.reshape(-1)))
        if not float(grid.reshape(-1)[flat]) > 0.0:
            flat = 0
        start = divmod(flat, ny)
    xv = x_values.tolist() if x_values is not None else list(range(nx))
    yv = y_values.tolist() if y_values is not None else list(range(ny))
    exact = tie_rtol == 0.0 and tie_atol == 0.0
    rows: dict[int, list[float]] = {}

    def row(i: int) -> list[float]:
        r = rows.get(i)
        if r is None:
            r = rows[i] = grid[i].tolist()
        return r

    def same(a: float, b: float) -> bool:
        if exact:
            return a == b
        return abs(a - b) <= tie_atol + tie_rtol * max(abs(a), abs(b))

    points: list[tuple[int, int]] = []

    def walk(i: int, j: int, step: int) -> None:
        while True:
            points.append((i, j))
            if step > 0 and (i >= nx - 1 or j >= ny - 1):
                return
            if step < 0 and (i <= 0 or j <= 0):
                return
            a = row(i + step)[j]
            d = row(i + step)[j + step]
            b = row(i)[j + step]
            m = max(a, b, d)
            if same(d, m) or same(a, b):
                i += step
                j += step
            elif same(a, m):
                if corner_cut and len(points) >= 2 and xv[points[-2][0]] == xv[i]:
                    points.pop()
                i += step
            else:
                if corner_cut and len(points) >= 2 and yv[points[-2][1]] == yv[j]:
                    points.pop()
                j += step

    walk(start[0], start[1], -1)
    walk(start[0], start[1], +1)
    points.sort()
    ij = torch.tensor(points, dtype=torch.long)
    return ij[:, 0], ij[:, 1]


@dataclass(frozen=True)
class PathPosterior:
    """Result of :func:`monotone_path`, one entry per column.

    ``viterbi`` is the best path's row (int64); ``mean`` / ``sd`` the
    posterior mean and SD of the row (float64, fractional);
    ``log_evidence`` the log of the summed score of all paths.
    """

    viterbi: torch.Tensor
    mean: torch.Tensor
    sd: torch.Tensor
    log_evidence: float


def monotone_path(
    score: torch.Tensor,
    step_logp: torch.Tensor,
    *,
    posterior: bool = True,
) -> PathPosterior:
    """Best path through ``score`` (``(n_columns, n_rows)`` log-potentials).

    ``step_logp`` has ``2P + 1`` entries, the log-prior of moving by
    ``k = −P … P`` rows from one column to the next (``-inf`` forbids a
    move). The path may start and end on any row. Cost is
    ``O(n_columns · n_rows · (2P + 1))``; memory one float64 grid for the
    forward pass.
    """
    if score.dim() != 2:
        raise ValueError(f"score must be 2-D, got shape {tuple(score.shape)}")
    if step_logp.dim() != 1 or step_logp.numel() % 2 != 1:
        raise ValueError("step_logp must be 1-D with an odd number of entries")
    score = score.to(torch.float64)
    step = step_logp.to(torch.float64)
    n_cols, n_rows = score.shape
    p = step.numel() // 2
    # In a window over the previous column, entry j is row v + j − P, so
    # the move into row v is k = P − j: the prior is step reversed.
    into = step.flip(0)

    def windows(v: torch.Tensor) -> torch.Tensor:
        return F.pad(v, (p, p), value=float("-inf")).unfold(0, 2 * p + 1, 1)

    back = torch.empty(n_cols, n_rows, dtype=torch.int16)
    best = score[0].clone()
    for c in range(1, n_cols):
        val, arg = (windows(best) + into).max(-1)
        back[c] = (p - arg).to(torch.int16)
        best = score[c] + val
    path = torch.empty(n_cols, dtype=torch.long)
    path[-1] = int(torch.argmax(best))
    for c in range(n_cols - 1, 0, -1):
        path[c - 1] = path[c] - int(back[c, path[c]])

    if not posterior:
        f = path.to(torch.float64)
        return PathPosterior(path, f, torch.zeros_like(f), float("nan"))

    alpha = torch.empty(n_cols, n_rows, dtype=torch.float64)
    alpha[0] = score[0]
    for c in range(1, n_cols):
        alpha[c] = score[c] + torch.logsumexp(windows(alpha[c - 1]) + into, -1)
    log_z = float(torch.logsumexp(alpha[-1], 0))
    rows = torch.arange(n_rows, dtype=torch.float64)
    mean = torch.empty(n_cols, dtype=torch.float64)
    sd = torch.empty(n_cols, dtype=torch.float64)
    beta = torch.zeros(n_rows, dtype=torch.float64)
    for c in range(n_cols - 1, -1, -1):
        g = torch.softmax(alpha[c] + beta, 0)
        m = (g * rows).sum()
        mean[c] = m
        sd[c] = torch.sqrt((g * (rows - m) ** 2).sum().clamp_min(0.0))
        if c > 0:
            # Out of row v' the move is k = j − P to row v' + k.
            beta = torch.logsumexp(windows(score[c] + beta) + step, -1)
    return PathPosterior(path, mean, sd, log_z)
