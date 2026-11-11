"""Many small, independent least-squares problems at once.

A batch of fits (one peak shape per candidate peak, one curve per row)
is B problems of P parameters each. Fitting them one at a time pays the
optimizer's overhead B times; stacking them into one parameter vector for
a general optimizer couples their convergence. :func:`batched_least_squares`
runs Levenberg–Marquardt on all rows together, each with its own damping,
step acceptance and convergence, as batched ``[B, P, P]`` solves: on 512
Gaussian peaks a 40-line prototype of it took 35 ms where one L-BFGS over
the stacked problems took 216 ms and one fit at a time 6.2 s, all equally
accurate (review of 2026-11-10, §6).

It is not an :class:`~constellation.core.optim.Optimizer`: that protocol
steps one problem's ``nn.Module`` under a scalar closure, and a batch of
independent residual vectors is a different contract (rows converge and
leave at their own pace). So it is not in ``OPTIMIZER_REGISTRY`` either: a
departure from the one-registry principle (CLAUDE.md, principle 6), made
because forcing the contract would cost the batching it exists for.

The residual function takes the parameters of some rows and which rows
they are, ``residual(params [b, P], rows [b]) -> [b, N]``, so that
converged rows are left out of later iterations. Rows must be independent:
row i's residual depends on row i's parameters only. The Jacobian is
forward-mode (``torch.func.jvp``, one pass per parameter) unless given.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["LeastSquaresResult", "batched_least_squares"]

Residual = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

#: ``LeastSquaresResult.status`` values.
RUNNING, F_TOL, X_TOL, G_TOL, MAX_ITER = 0, 1, 2, 3, -1


@dataclass(frozen=True)
class LeastSquaresResult:
    """Per row: the parameters, ``cost = ½ Σ r²``, whether it converged and
    by which test (``status``: 1 cost change, 2 step, 3 gradient, -1 out of
    iterations), its iterations, and the covariance of the parameters,
    ``(JᵀJ)⁻¹ · 2·cost / (n − p)`` over its free parameters (NaN where
    singular or without degrees of freedom)."""

    params: torch.Tensor  # [B, P]
    cost: torch.Tensor  # [B]
    converged: torch.Tensor  # [B] bool
    status: torch.Tensor  # [B] int
    n_iter: torch.Tensor  # [B] int
    covariance: torch.Tensor  # [B, P, P]


def _jacobian(residual: Residual, params: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(r [b, N], J [b, N, P])`` by forward mode: one ``jvp`` per
    parameter, the P of them vectorized (``vmap``) into one call."""
    from torch.func import jvp, vmap

    p = params.shape[-1]
    tangents = torch.eye(p, dtype=params.dtype).unsqueeze(1).expand(p, *params.shape)

    def one(t):
        return jvp(lambda q: residual(q, rows), (params,), (t,))

    r, cols = vmap(one, out_dims=(None, 0))(tangents) if p else (residual(params, rows), None)
    return r, cols.permute(1, 2, 0)


def batched_least_squares(
    residual: Residual,
    params0: torch.Tensor,
    *,
    jacobian: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    mask: torch.Tensor | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    free: torch.Tensor | None = None,
    max_iter: int = 100,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-12,
    damping: float = 1e-3,
) -> LeastSquaresResult:
    """Fit every row of ``params0 [B, P]`` (module docstring).

    ``mask [B, N]`` says which residuals count; ``lower`` / ``upper``
    (``[B, P]`` or ``[P]``) bound the parameters, a step projected onto the
    box; ``free [B, P]`` marks the parameters a row fits (the others stay
    as given: a padded row of fewer components). ``jacobian(params, rows)``
    may give ``(r, J)`` analytically. ``damping`` is the first λ relative to
    the largest diagonal of ``JᵀJ`` (Nielsen's update after)."""
    params = params0.detach().to(torch.float64).clone()
    b, p = params.shape
    lo = None if lower is None else torch.broadcast_to(lower.to(torch.float64), params.shape)
    hi = None if upper is None else torch.broadcast_to(upper.to(torch.float64), params.shape)
    fr = torch.ones_like(params, dtype=torch.bool) if free is None else free.bool().expand_as(params)

    def clamp(q, rows):
        if lo is not None:
            q = torch.maximum(q, lo[rows])
        if hi is not None:
            q = torch.minimum(q, hi[rows])
        return q

    def evaluate(q, rows, with_jacobian):
        if with_jacobian:
            r, j = (jacobian or (lambda a, c: _jacobian(residual, a, c)))(q, rows)
        else:
            r, j = residual(q, rows), None
        if mask is not None:
            m = mask[rows]
            r = torch.where(m, r, torch.zeros_like(r))
            if j is not None:
                j = torch.where(m.unsqueeze(-1), j, torch.zeros_like(j))
        r = torch.nan_to_num(r.to(torch.float64), nan=0.0)
        if j is not None:
            j = torch.nan_to_num(j.to(torch.float64), nan=0.0) * fr[rows].unsqueeze(-2)
        return r, j

    params = clamp(params, torch.arange(b))
    all_rows = torch.arange(b)
    r, j = evaluate(params, all_rows, True)
    cost = 0.5 * (r * r).sum(-1)
    jtj = j.transpose(-1, -2) @ j
    lam = damping * torch.diagonal(jtj, dim1=-2, dim2=-1).amax(-1).clamp(min=1e-12)
    nu = torch.full((b,), 2.0, dtype=torch.float64)
    status = torch.zeros(b, dtype=torch.int64)
    n_iter = torch.zeros(b, dtype=torch.int64)
    grad = (j.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
    status = torch.where(grad.abs().amax(-1) <= gtol, torch.full_like(status, G_TOL), status)
    eye = torch.eye(p, dtype=torch.float64)
    jac_cache = j
    r_cache = r
    for _ in range(max_iter):
        active = torch.nonzero(status == RUNNING).squeeze(-1)
        if active.numel() == 0:
            break
        q = params[active]
        jj = jac_cache[active]
        rr = r_cache[active]
        a = jj.transpose(-1, -2) @ jj
        g = (jj.transpose(-1, -2) @ rr.unsqueeze(-1)).squeeze(-1)
        diag = torch.diagonal(a, dim1=-2, dim2=-1).clamp(min=1e-12)
        lhs = a + lam[active].unsqueeze(-1).unsqueeze(-1) * torch.diag_embed(diag)
        # A fixed parameter's row and column are zero (its Jacobian column
        # is): a unit diagonal there makes its step 0.
        lhs = lhs + torch.diag_embed((~fr[active]).to(torch.float64))
        chol, info = torch.linalg.cholesky_ex(lhs)
        step = torch.cholesky_solve(-g.unsqueeze(-1), chol).squeeze(-1)
        bad = info != 0
        if bad.any():
            step[bad] = torch.linalg.lstsq(lhs[bad], -g[bad].unsqueeze(-1)).solution.squeeze(-1)
        step = torch.where(fr[active], step, torch.zeros_like(step))
        trial = clamp(q + step, active)
        taken = trial - q
        r_new, _ = evaluate(trial, active, False)
        cost_new = 0.5 * (r_new * r_new).sum(-1)
        predicted = -(taken * g).sum(-1) - 0.5 * (taken.unsqueeze(-2) @ a @ taken.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        actual = cost[active] - cost_new
        rho = actual / predicted.clamp(min=1e-300)
        better = (actual > 0) & torch.isfinite(cost_new)
        n_iter[active] += 1
        # Accepted rows move and get a fresh Jacobian; refused ones raise λ.
        acc = active[better]
        if acc.numel():
            params[acc] = trial[better]
            r2, j2 = evaluate(params[acc], acc, True)
            r_cache[acc] = r2
            jac_cache[acc] = j2
            old = cost[acc]
            cost[acc] = cost_new[better]
            shrink = torch.clamp(1.0 - (2.0 * rho[better] - 1.0) ** 3, min=1.0 / 3.0)
            lam[acc] = lam[acc] * shrink
            nu[acc] = 2.0
            grad_acc = (j2.transpose(-1, -2) @ r2.unsqueeze(-1)).squeeze(-1)
            small_f = (old - cost[acc]).abs() <= ftol * old.clamp(min=1e-300)
            small_x = taken[better].norm(dim=-1) <= xtol * (params[acc].norm(dim=-1) + xtol)
            small_g = grad_acc.abs().amax(-1) <= gtol
            st = torch.zeros(acc.numel(), dtype=torch.int64)
            st = torch.where(small_g, torch.full_like(st, G_TOL), st)
            st = torch.where(small_x, torch.full_like(st, X_TOL), st)
            st = torch.where(small_f, torch.full_like(st, F_TOL), st)
            status[acc] = st
        rej = active[~better]
        if rej.numel():
            lam[rej] = lam[rej] * nu[rej]
            nu[rej] = nu[rej] * 2.0
            # A step this small that still fails: converged where it stands.
            tiny = taken[~better].norm(dim=-1) <= xtol * (params[rej].norm(dim=-1) + xtol)
            status[rej] = torch.where(tiny, torch.full_like(status[rej], X_TOL), status[rej])
    status = torch.where(status == RUNNING, torch.full_like(status, MAX_ITER), status)
    # Covariance at the solution, over the free parameters.
    j = jac_cache
    a = j.transpose(-1, -2) @ j
    a = a + eye * (~fr).to(torch.float64).unsqueeze(-1)
    n_obs = (torch.ones_like(r_cache) if mask is None else mask.to(torch.float64)).sum(-1)
    dof = n_obs - fr.sum(-1)
    inv, info = torch.linalg.inv_ex(a)
    scale = torch.where(dof > 0, 2.0 * cost / dof.clamp(min=1), torch.full_like(cost, float("nan")))
    cov = inv * scale.unsqueeze(-1).unsqueeze(-1)
    cov = torch.where((info == 0).unsqueeze(-1).unsqueeze(-1), cov, torch.full_like(cov, float("nan")))
    cov = torch.where(fr.unsqueeze(-1) & fr.unsqueeze(-2), cov, torch.zeros_like(cov))
    return LeastSquaresResult(params=params, cost=cost, converged=status > 0, status=status, n_iter=n_iter,
                              covariance=cov)  # fmt: skip
