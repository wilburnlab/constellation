"""Uncertainty quantification — credible intervals, Laplace covariance,
profile likelihood.

Three rungs of a ladder, cheapest last to richest first:

  * `credible_interval(samples, level)` — empirical quantiles of posterior
    samples. The primary path: draw from a fitted `VariationalGuide`,
    decode to natural units, take quantiles. Needs no parametric-form CDF,
    which is exactly how Counter sidesteps the pinned `StudentT.cdf` gap.
  * `laplace_cov(model, data)` — Gaussian (Laplace) approximation: the
    inverse Hessian of the negative log-joint at the current (MAP) point.
    A cheap covariance when full VI is overkill.
  * `profile_interval(model, data, param)` — profile-likelihood interval:
    sweep one parameter, re-optimize the rest, find where the deviance
    crosses `χ²₁(level)`. The classic MLE/MAP-compatible interval.

All operate in whatever space the model stores its parameters; decode to
natural units at the call site.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.func import functional_call

from constellation.core.optim import (
    LBFGSOptimizer,
    build_layout,
    flatten_params,
    params_dict_from_flat,
    unflatten_into_model,
)

from .variational import PriorFn, _resolve_active_indices

__all__ = ["credible_interval", "laplace_cov", "profile_interval"]


def credible_interval(
    samples: torch.Tensor, level: float = 0.95, *, dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-tailed empirical credible interval of `samples` along `dim`.

    Returns `(lo, hi)` at the `(1±level)/2` quantiles. `samples` is any
    posterior-draw tensor — e.g. `exp(guide.named_samples(model, n)["log_N_total"])`
    for the credible interval on `N_total` in natural units.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    alpha = (1.0 - level) / 2.0
    q = torch.tensor([alpha, 1.0 - alpha], dtype=samples.dtype, device=samples.device)
    lo, hi = torch.quantile(samples, q, dim=dim)
    return lo, hi


def laplace_cov(
    model: nn.Module,
    data: torch.Tensor,
    *,
    subset: Sequence[str] | None = None,
    log_prior: PriorFn | None = None,
) -> torch.Tensor:
    """Laplace covariance over the `subset` parameters (default: all) — the
    inverse Hessian of the negative log-joint at the model's current params.

    The model must be at (or near) its MAP/MLE for this to be meaningful.
    Returns a `(k, k)` covariance in the model's stored parameterization,
    ordered by the flat layout; marginal stds are `cov.diagonal().sqrt()`.
    `model.forward` must return per-datum log-density (the `Distribution`
    contract).
    """
    layout = build_layout(model)
    active = _resolve_active_indices(layout, subset)
    base = flatten_params(model, layout)
    theta0 = base[active].clone()

    def neg_log_joint(active_vec: torch.Tensor) -> torch.Tensor:
        full = base.index_copy(0, active, active_vec)
        params = params_dict_from_flat(full, layout)
        lp = functional_call(model, params, (data,)).sum()
        if log_prior is not None:
            lp = lp + log_prior(params)
        return -lp

    hess = torch.autograd.functional.hessian(neg_log_joint, theta0)
    hess = hess.reshape(active.numel(), active.numel())
    return torch.linalg.inv(hess)


def _chi2_1_half(level: float) -> float:
    """`0.5·χ²₁⁻¹(level)` — the profile-likelihood deviance threshold. Uses
    `χ²₁⁻¹(p) = (Φ⁻¹((1+p)/2))²` (torch `ndtri`), so no scipy dependency."""
    p = torch.tensor((1.0 + level) / 2.0, dtype=torch.float64)
    z = float(torch.special.ndtri(p))
    return 0.5 * z * z


def _cross(
    g0: float, f0: float, g1: float, f1: float, thresh: float
) -> float:
    """Linear-interpolated crossing of `thresh` between `(g0,f0)` and `(g1,f1)`."""
    if f1 == f0:
        return g1
    return g0 + (thresh - f0) * (g1 - g0) / (f1 - f0)


def profile_interval(
    model: nn.Module,
    data: torch.Tensor,
    param_name: str,
    *,
    level: float = 0.95,
    n_grid: int = 41,
    radius: float | None = None,
    log_prior: PriorFn | None = None,
    refit_steps: int = 30,
) -> tuple[float, float]:
    """Profile-likelihood interval for a scalar parameter `param_name`.

    Sweeps `param_name` over a grid centred on its current (MAP) value,
    re-optimizing every *other* parameter at each grid point (LBFGS), and
    returns where the deviance `2·(nll(θ) − nll_min)` crosses `χ²₁(level)`.
    The model is restored to its incoming parameter values on return.

    `radius` defaults to `5·σ` from a one-parameter `laplace_cov`. If the
    deviance does not cross the threshold within the grid on a side, that
    bound is reported at the grid edge (interval is open there).
    """
    params = dict(model.named_parameters())
    if param_name not in params:
        raise KeyError(f"{param_name!r} not in model; have {list(params)}")
    target = params[param_name]
    if target.numel() != 1:
        raise ValueError(
            f"profile_interval supports scalar parameters; {param_name!r} "
            f"has {target.numel()} elements"
        )
    layout = build_layout(model)
    base = flatten_params(model, layout)  # MAP snapshot to restore
    others = [p for n, p in model.named_parameters() if n != param_name]

    def nll() -> torch.Tensor:
        lp = model(data).sum()
        if log_prior is not None:
            lp = lp + log_prior(dict(model.named_parameters()))
        return -lp

    with torch.no_grad():
        nll_min = float(nll().detach())
    v0 = float(target.detach())

    if radius is None:
        cov = laplace_cov(model, data, subset=[param_name], log_prior=log_prior)
        std = float(cov.reshape(()).clamp(min=1e-24).sqrt())
        radius = 5.0 * std

    grid = torch.linspace(v0 - radius, v0 + radius, n_grid, dtype=target.dtype)
    profile: list[float] = []
    target.requires_grad_(False)
    try:
        for v in grid:
            with torch.no_grad():
                unflatten_into_model(model, base, layout)  # reset to MAP
                target.data.fill_(float(v))
            if others:
                opt = LBFGSOptimizer(others, max_iter=refit_steps)
                prev = math.inf
                for _ in range(40):
                    cur = float(opt.step(nll).detach())
                    if abs(prev - cur) < 1e-10:
                        break
                    prev = cur
            with torch.no_grad():
                profile.append(float(nll().detach()))
    finally:
        target.requires_grad_(True)
        with torch.no_grad():
            unflatten_into_model(model, base, layout)  # restore MAP

    g = grid.to(torch.float64).tolist()
    thresh = nll_min + _chi2_1_half(level)
    imin = min(range(len(profile)), key=lambda i: profile[i])

    lo = g[0]
    for i in range(imin, 0, -1):
        if profile[i - 1] >= thresh > profile[i]:
            lo = _cross(g[i - 1], profile[i - 1], g[i], profile[i], thresh)
            break
    hi = g[-1]
    for i in range(imin, len(profile) - 1):
        if profile[i] < thresh <= profile[i + 1]:
            hi = _cross(g[i], profile[i], g[i + 1], profile[i + 1], thresh)
            break
    return lo, hi
