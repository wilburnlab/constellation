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

Everything a row decides is free of the units its residuals and
parameters are in, so a peak fitted in counts and the same peak in
millions of counts take the same path. The damping is Marquardt's,
``JᵀJ + μ·diag(JᵀJ)`` with μ dimensionless (Nielsen's update); a step has
converged when every parameter's change is within ``xtol`` of that
parameter, the cost when its decrease is within ``ftol`` of it, and the
gradient when the largest cosine between the residuals and a column of the
Jacobian is within ``gtol`` (MINPACK's test). Until 2026-11-11 λ started
at ``damping · max diag(JᵀJ)`` and then multiplied ``diag(JᵀJ)``, so the
scale entered twice, and the step test was on the whole parameter vector,
which the amplitude dominates: an exact EMG fitted at height 1e6 stopped,
"converged", at σ 2.44 and τ 1.83 instead of 2.1 and 3.

A residual or Jacobian that is not finite where it counts is never taken
for zero: at the start the row fails (status −2); a trial step that makes
one is refused. Until 2026-11-11 both were replaced by zeros, so fitting
``log(p)`` from p = −1 was a perfect fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["LeastSquaresResult", "batched_least_squares"]

Residual = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

#: ``LeastSquaresResult.status`` values.
RUNNING, F_TOL, X_TOL, G_TOL, MAX_ITER, NOT_FINITE = 0, 1, 2, 3, -1, -2


@dataclass(frozen=True)
class LeastSquaresResult:
    """Per row: the parameters, ``cost = ½ Σ r²``, whether it converged and
    by which test (``status``: 1 cost change, 2 step, 3 gradient, -1 out of
    iterations, -2 a residual or Jacobian not finite at the start, the
    parameters then as started and the cost and covariance NaN), its
    iterations, and the covariance of the parameters,
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


def _gradient_cosine(j: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``[b]``: the largest ``|J_iᵀ r| / (‖J_i‖ ‖r‖)`` over the columns,
    MINPACK's scale-free gradient test (0 for an exact fit, and a column
    the residuals do not depend on is left out)."""
    g = (j.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
    col = torch.linalg.vector_norm(j, dim=-2)
    rn = torch.linalg.vector_norm(r, dim=-1).unsqueeze(-1)
    scale = col * rn
    cos = torch.where(scale > 0, g.abs() / torch.where(scale > 0, scale, torch.ones_like(scale)),
                      torch.zeros_like(g))  # fmt: skip
    return cos.amax(-1) if cos.shape[-1] else torch.zeros(r.shape[0], dtype=r.dtype)


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
    may give ``(r, J)`` analytically. ``damping`` is the first μ of
    ``JᵀJ + μ·diag(JᵀJ)``, dimensionless (Nielsen's update after). A row
    stops when every free parameter's step is within ``xtol·(|p| + xtol)``,
    when the cost falls by no more than ``ftol`` of itself, or when the
    gradient's cosine with the residuals is within ``gtol``."""
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
        """``(r, J, finite)``: the residuals that count (the others 0), the
        Jacobian over them and the free parameters, and whether all of
        those are finite. A row that is not has its r and J zeroed, for
        the arithmetic only: the caller refuses it."""
        if with_jacobian:
            r, j = (jacobian or (lambda a, c: _jacobian(residual, a, c)))(q, rows)
        else:
            r, j = residual(q, rows), None
        r = r.to(torch.float64)
        counts = torch.ones_like(r, dtype=torch.bool) if mask is None else mask[rows]
        r = torch.where(counts, r, torch.zeros_like(r))
        finite = torch.isfinite(r).all(-1)
        if j is not None:
            j = j.to(torch.float64)
            j = torch.where(counts.unsqueeze(-1) & fr[rows].unsqueeze(-2), j, torch.zeros_like(j))
            finite &= torch.isfinite(j).flatten(-2).all(-1)
            j = torch.where(finite.view(-1, 1, 1), j, torch.zeros_like(j))
        r = torch.where(finite.unsqueeze(-1), r, torch.zeros_like(r))
        return r, j, finite

    def small_step(taken, q):
        """Every parameter's step within ``xtol`` of it (fixed ones take none)."""
        return (taken.abs() <= xtol * (q.abs() + xtol)).all(-1)

    params = clamp(params, torch.arange(b))
    r, j, finite = evaluate(params, torch.arange(b), True)
    cost = torch.where(finite, 0.5 * (r * r).sum(-1), torch.full((b,), float("nan"), dtype=torch.float64))
    mu = torch.full((b,), float(damping), dtype=torch.float64)
    nu = torch.full((b,), 2.0, dtype=torch.float64)
    status = torch.where(finite, torch.full((b,), RUNNING), torch.full((b,), NOT_FINITE))
    n_iter = torch.zeros(b, dtype=torch.int64)
    status = torch.where(finite & (_gradient_cosine(j, r) <= gtol), torch.full_like(status, G_TOL), status)
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
        # Marquardt's scaling; a parameter the residuals do not depend on
        # (its column zero, as a fixed one's is) gets a unit diagonal, so
        # its step is 0.
        diag = torch.diagonal(a, dim1=-2, dim2=-1)
        diag = torch.where(diag > 0, diag, torch.ones_like(diag))
        lhs = a + mu[active].view(-1, 1, 1) * torch.diag_embed(diag)
        lhs = lhs + torch.diag_embed((~fr[active]).to(torch.float64))
        chol, info = torch.linalg.cholesky_ex(lhs)
        step = torch.cholesky_solve(-g.unsqueeze(-1), chol).squeeze(-1)
        bad = info != 0
        if bad.any():
            step[bad] = torch.linalg.lstsq(lhs[bad], -g[bad].unsqueeze(-1)).solution.squeeze(-1)
        # A step that overflowed (μ beyond float range after many refusals)
        # is no step: the row then stops where it stands.
        step = torch.where(fr[active] & torch.isfinite(step), step, torch.zeros_like(step))
        trial = clamp(q + step, active)
        taken = trial - q
        r_new, _, ok_new = evaluate(trial, active, False)
        cost_new = torch.where(ok_new, 0.5 * (r_new * r_new).sum(-1), torch.full_like(cost[active], float("inf")))
        predicted = -(taken * g).sum(-1) - 0.5 * (taken.unsqueeze(-2) @ a @ taken.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        actual = cost[active] - cost_new
        rho = actual / predicted.clamp(min=1e-300)
        better = (actual > 0) & ok_new
        n_iter[active] += 1
        # A better trial moves its row if its Jacobian there is finite too;
        # the others raise μ.
        moved = torch.zeros_like(better)
        cand = torch.nonzero(better).squeeze(-1)
        if cand.numel():
            rows = active[cand]
            r2, j2, ok2 = evaluate(trial[cand], rows, True)
            cand, rows, r2, j2 = cand[ok2], rows[ok2], r2[ok2], j2[ok2]
            moved[cand] = True
        if moved.any():
            params[rows] = trial[cand]
            r_cache[rows] = r2
            jac_cache[rows] = j2
            old = cost[rows]
            cost[rows] = cost_new[cand]
            shrink = torch.clamp(1.0 - (2.0 * rho[cand] - 1.0) ** 3, min=1.0 / 3.0)
            mu[rows] = mu[rows] * shrink
            nu[rows] = 2.0
            small_f = (old - cost[rows]) <= ftol * old.clamp(min=1e-300)
            small_x = small_step(taken[cand], params[rows])
            small_g = _gradient_cosine(j2, r2) <= gtol
            st = torch.zeros(rows.numel(), dtype=torch.int64)
            st = torch.where(small_g, torch.full_like(st, G_TOL), st)
            st = torch.where(small_x, torch.full_like(st, X_TOL), st)
            st = torch.where(small_f, torch.full_like(st, F_TOL), st)
            status[rows] = st
        refused = torch.nonzero(~moved).squeeze(-1)
        if refused.numel():
            rej = active[refused]
            mu[rej] = mu[rej] * nu[rej]
            nu[rej] = nu[rej] * 2.0
            # A step this small that still fails: converged where it stands.
            tiny = small_step(taken[refused], params[rej])
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
    cov = torch.where((status == NOT_FINITE).view(-1, 1, 1), torch.full_like(cov, float("nan")), cov)
    return LeastSquaresResult(params=params, cost=cost, converged=status > 0, status=status, n_iter=n_iter,
                              covariance=cov)  # fmt: skip
