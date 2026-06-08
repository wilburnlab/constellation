"""Variational inference — reparameterized Gaussian guides + Monte-Carlo ELBO.

Pure-PyTorch VI that slots into the existing `Parametric` / `core.optim`
machinery rather than a separate probabilistic-programming stack:

  * A `VariationalGuide` snapshots a target model's **flat parameter
    layout** (`core.optim.ParamLayout`) over a chosen *subset* of params
    (the ones we want a posterior over). The complement is held fixed at
    the model's current (point / MAP) values during VI.
  * `elbo(model, guide, data)` draws reparameterized samples, writes each
    into a functional copy of the model via `torch.func.functional_call`
    + `vmap` — **the identical mechanism `DifferentialEvolution` uses** —
    so the model's existing vmap-safe `log_prob` evaluates a whole batch
    of samples at once, and gradients flow back to the guide.
  * `fit_vb` runs the `-ELBO` loss through any `core.optim.Optimizer`;
    `AdamOptimizer` is the natural choice (its moment estimates absorb the
    Monte-Carlo gradient noise that breaks LBFGS's line search). DE can
    warm-start the model first; the guide's `loc` then inits at that point.

The guide operates in the model's **stored** parameterization (log / logit
space — `core.stats` convention), so a Gaussian `q` is well matched and
positive/bounded quantities stay valid after the model decodes them; the
caller decodes guide samples to natural units for credible intervals (see
`core.stats.intervals`).

Default-loss note: `elbo` returns the ELBO (to maximize); `fit_vb` negates
it for the minimizing `Optimizer.step` contract — closures stay grad-free,
the optimizer owns `backward`.
"""

from __future__ import annotations

import abc
import fnmatch
import math
from typing import Callable, Sequence

import torch
from torch import nn
from torch.func import functional_call, vmap

from constellation.core.optim import (
    ParamLayout,
    build_layout,
    flatten_params,
    params_dict_from_population,
)

from .parametric import FitResult

__all__ = [
    "VariationalGuide",
    "MeanFieldGuide",
    "LowRankGuide",
    "elbo",
    "fit_vb",
]

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_SCALE_MIN = math.log(1e-6)  # floor on log-scale to prevent posterior collapse

# A prior callable maps the batched (vmap-shaped) param dict to a (n_samples,)
# log-prior. None means a flat (improper) prior — ELBO reduces to E_q[loglik]+H.
PriorFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


def _resolve_active_indices(
    layout: ParamLayout, subset: Sequence[str] | None
) -> torch.Tensor:
    """Flat indices of the parameters a guide is variational over.

    `subset` entries match parameter names exactly or by fnmatch glob
    (e.g. ``"log_*"``, ``"peak.log_sigma"``); ``None`` selects every
    parameter. Order follows the flat layout, not the subset list.
    """
    if subset is None:
        return torch.arange(layout.n_flat, dtype=torch.long)
    selected: list[int] = []
    matched: set[str] = set()
    for name, shape, off in zip(layout.names, layout.shapes, layout.offsets):
        if any(fnmatch.fnmatchcase(name, pat) for pat in subset):
            n = max(1, int(torch.zeros(shape).numel()))
            selected.extend(range(off, off + n))
            matched.add(name)
    unmatched = [
        p
        for p in subset
        if not any(fnmatch.fnmatchcase(name, p) for name in layout.names)
    ]
    if unmatched:
        raise KeyError(
            f"subset pattern(s) {unmatched} matched no parameter; "
            f"available: {list(layout.names)}"
        )
    if not selected:
        raise ValueError("subset selected zero parameters")
    return torch.tensor(sorted(selected), dtype=torch.long)


class VariationalGuide(nn.Module, abc.ABC):
    """Reparameterized posterior over a subset of a model's flat parameters.

    Subclasses implement `rsample` (reparameterized draws, `(n, k)`) and
    `entropy` (closed-form `H[q]`, scalar). `k` is the number of active
    flat parameter entries; the inactive complement is read *live* from the
    model at ELBO time (so it tracks any prior MAP fit).
    """

    def __init__(self, model: nn.Module, *, subset: Sequence[str] | None = None):
        super().__init__()
        self._layout = build_layout(model)
        active = _resolve_active_indices(self._layout, subset)
        self.register_buffer("active_idx", active)
        self._k = int(active.numel())
        dtype = next(model.parameters()).dtype
        self.register_buffer(
            "_init_loc", flatten_params(model, self._layout)[active].to(dtype)
        )

    @property
    def layout(self) -> ParamLayout:
        return self._layout

    @property
    def k(self) -> int:
        """Number of active (variational) flat parameter entries."""
        return self._k

    @abc.abstractmethod
    def rsample(self, n: int) -> torch.Tensor:
        """`(n, k)` reparameterized draws (differentiable w.r.t. guide params)."""

    @abc.abstractmethod
    def entropy(self) -> torch.Tensor:
        """Closed-form differential entropy `H[q]` (0-d tensor)."""

    @property
    @abc.abstractmethod
    def loc(self) -> torch.Tensor:
        """Posterior mean over the active params, `(k,)` — the point estimate."""

    def sample(self, n: int) -> torch.Tensor:
        """No-grad draws, `(n, k)` — for credible intervals."""
        with torch.no_grad():
            return self.rsample(n)

    def full_draws(
        self, model: nn.Module, n: int, *, requires_grad: bool
    ) -> torch.Tensor:
        """`(n, n_flat)` parameter samples: active columns drawn from `q`,
        inactive columns held at the model's current values (read live)."""
        draws = self.rsample(n) if requires_grad else self.sample(n)
        base = flatten_params(model, self._layout)  # (n_flat,), detached
        full = base.unsqueeze(0).repeat(n, 1)
        return full.index_copy(1, self.active_idx, draws)

    def named_samples(self, model: nn.Module, n: int) -> dict[str, torch.Tensor]:
        """No-grad draws mapped back to `{name: (n, *shape)}` in the model's
        stored (log/logit) parameterization. Active params vary across the
        leading dim; inactive params are constant. Decode to natural units
        (e.g. `exp` for `log_*`) before computing credible intervals."""
        full = self.full_draws(model, n, requires_grad=False)
        return params_dict_from_population(full, self._layout)

    @torch.no_grad()
    def to_model(self, model: nn.Module) -> None:
        """Write the posterior mean (`loc`) into the model's active params."""
        flat = flatten_params(model, self._layout)
        flat = flat.index_copy(0, self.active_idx, self.loc.detach())
        params = dict(model.named_parameters())
        for name, shape, off in zip(
            self._layout.names, self._layout.shapes, self._layout.offsets
        ):
            n = max(1, int(torch.zeros(shape).numel()))
            params[name].data.copy_(flat[off : off + n].reshape(shape))


class MeanFieldGuide(VariationalGuide):
    """Diagonal-Gaussian guide `q(θ) = N(loc, diag(exp(2·log_scale)))`.

    Reparameterized sampling `loc + exp(log_scale)·ε`; closed-form entropy.
    `log_scale` is floored at `log(1e-6)` so the posterior cannot collapse
    to a point (mirrors the `StudentT` `log_nu` clamp).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        subset: Sequence[str] | None = None,
        init_scale: float = 0.1,
    ):
        super().__init__(model, subset=subset)
        self.loc_param = nn.Parameter(self._init_loc.clone())
        self.log_scale = nn.Parameter(
            torch.full_like(self._init_loc, math.log(init_scale))
        )

    @property
    def loc(self) -> torch.Tensor:
        return self.loc_param

    @property
    def scale(self) -> torch.Tensor:
        """Posterior marginal std per active param, `(k,)` (floored)."""
        return self._scale()

    def _scale(self) -> torch.Tensor:
        return self.log_scale.clamp(min=_LOG_SCALE_MIN).exp()

    def rsample(self, n: int) -> torch.Tensor:
        eps = torch.randn(
            (n, self._k), dtype=self.loc_param.dtype, device=self.loc_param.device
        )
        return self.loc_param.unsqueeze(0) + self._scale().unsqueeze(0) * eps

    def entropy(self) -> torch.Tensor:
        # H = 0.5·k·(1 + log 2π) + Σ log σ_i
        log_scale = self.log_scale.clamp(min=_LOG_SCALE_MIN)
        return 0.5 * self._k * (1.0 + _LOG_2PI) + log_scale.sum()


class LowRankGuide(VariationalGuide):
    """Low-rank-plus-diagonal Gaussian `q(θ) = N(loc, diag(d²) + W Wᵀ)`.

    Captures parameter correlations the mean-field guide cannot (e.g.
    `N_total ↔ peak width`). `rank` is the number of factor columns in `W`.
    Entropy uses the matrix-determinant lemma so it stays `O(k·r² + r³)`.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        subset: Sequence[str] | None = None,
        rank: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__(model, subset=subset)
        if rank < 1:
            raise ValueError(f"rank must be ≥ 1; got {rank}")
        self._rank = int(rank)
        self.loc_param = nn.Parameter(self._init_loc.clone())
        self.log_diag = nn.Parameter(
            torch.full_like(self._init_loc, math.log(init_scale))
        )
        # Small random factor so the initial covariance is ~diagonal.
        self.factor = nn.Parameter(
            0.01 * torch.randn((self._k, self._rank), dtype=self._init_loc.dtype)
        )

    @property
    def loc(self) -> torch.Tensor:
        return self.loc_param

    def _diag(self) -> torch.Tensor:
        return self.log_diag.clamp(min=_LOG_SCALE_MIN).exp()

    def rsample(self, n: int) -> torch.Tensor:
        d = self._diag()
        eps_d = torch.randn(
            (n, self._k), dtype=self.loc_param.dtype, device=self.loc_param.device
        )
        eps_r = torch.randn(
            (n, self._rank), dtype=self.loc_param.dtype, device=self.loc_param.device
        )
        return (
            self.loc_param.unsqueeze(0)
            + d.unsqueeze(0) * eps_d
            + eps_r @ self.factor.transpose(0, 1)
        )

    def entropy(self) -> torch.Tensor:
        # logdet(diag(d²) + W Wᵀ) = Σ log d² + logdet(I_r + Wᵀ diag(d⁻²) W)
        d = self._diag()
        w = self.factor
        wt_dinv2_w = (w / (d * d).unsqueeze(1)).transpose(0, 1) @ w  # (r, r)
        m = torch.eye(self._rank, dtype=w.dtype, device=w.device) + wt_dinv2_w
        logdet = 2.0 * self.log_diag.clamp(min=_LOG_SCALE_MIN).sum() + torch.logdet(m)
        return 0.5 * self._k * (1.0 + _LOG_2PI) + 0.5 * logdet


def elbo(
    model: nn.Module,
    guide: VariationalGuide,
    data: torch.Tensor,
    *,
    log_prior: PriorFn | None = None,
    n_samples: int = 8,
) -> torch.Tensor:
    """Monte-Carlo ELBO `E_q[log p(data|θ) + log p(θ)] + H[q]` (a 0-d tensor
    to MAXIMIZE).

    `model.forward` must return per-datum log-density (the `Distribution`
    contract — `forward` defaults to `log_prob`). Reparameterized samples
    are substituted via `functional_call` + `vmap`, so `model.log_prob` runs
    once over the whole `(n_samples, …)` batch and gradients flow to the
    guide. `log_prior(batched_params) -> (n_samples,)` adds a prior in the
    model's stored parameterization; `None` is a flat prior (variational
    MLE — the entropy term still regularizes the scale toward the
    likelihood curvature).
    """
    full = guide.full_draws(model, n_samples, requires_grad=True)
    batched = params_dict_from_population(full, guide.layout)

    def model_view(*args, **kwargs):
        def fmodel(params):
            return functional_call(model, params, args, kwargs)

        return vmap(fmodel, in_dims=(0,))(batched)

    log_density = model_view(data)  # (n_samples, *data_shape)
    if log_density.dim() > 1:
        log_density = log_density.flatten(1).sum(dim=1)
    elif log_density.dim() == 0:
        log_density = log_density.reshape(1)
    total = log_density  # (n_samples,)
    if log_prior is not None:
        total = total + log_prior(batched)
    return total.mean() + guide.entropy()


def fit_vb(
    model: nn.Module,
    guide: VariationalGuide,
    data: torch.Tensor,
    *,
    optimizer,
    log_prior: PriorFn | None = None,
    n_samples: int = 8,
    max_iter: int = 300,
    tol: float = 1e-6,
    callback: Callable[[int, float], None] | None = None,
) -> FitResult:
    """Optimize a guide's parameters to maximize the ELBO via a gradient
    `Optimizer` (use `AdamOptimizer(guide)`). Returns a `FitResult` whose
    `loss_history` is the `-ELBO` trajectory. Outer convergence: `|Δloss| <
    tol`. NaN loss aborts (`converged=False`)."""
    result = FitResult()
    prev_loss = math.inf

    def closure() -> torch.Tensor:
        return -elbo(
            model, guide, data, log_prior=log_prior, n_samples=n_samples
        )

    for i in range(max_iter):
        loss_t = optimizer.step(closure)
        loss_val = float(loss_t.detach()) if torch.is_tensor(loss_t) else float(loss_t)
        if not math.isfinite(loss_val):
            result.n_iter = i + 1
            result.final_loss = loss_val
            result.converged = False
            if callback is not None:
                callback(i, loss_val)
            return result
        result.loss_history.append(loss_val)
        if callback is not None:
            callback(i, loss_val)
        if abs(prev_loss - loss_val) < tol:
            result.n_iter = i + 1
            result.final_loss = loss_val
            result.converged = True
            return result
        prev_loss = loss_val

    result.n_iter = max_iter
    result.final_loss = result.loss_history[-1] if result.loss_history else math.inf
    result.converged = False
    return result
