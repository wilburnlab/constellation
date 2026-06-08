"""`AdamOptimizer` — production wrapper around `torch.optim.Adam`.

The same two responsibilities `LBFGSOptimizer` takes on, so Adam slots
into the `Optimizer` Protocol family identically:

1. **Closure inversion.** The user's scalar closure returns un-backed
   loss; `AdamOptimizer.step` calls `zero_grad` + `backward` itself, so
   closures stay grad-free and shared with the LBFGS / DE paths.

2. **Bounds-clamping after step.** When `bounds=` is supplied, parameters
   are clamped back into the box after every `step()` — the same layout +
   `resolve_bounds` path `LBFGSOptimizer` uses, so a single source of
   truth for box constraints across all gradient optimizers.

Unlike LBFGS (a quasi-Newton optimizer that runs an internal line search
of `max_iter` evaluations per `.step()`), Adam's `.step()` is a *single*
first-order update; the iteration count is owned by the outer
`Parametric.fit` loop. Adam is the optimizer of choice for the Monte-Carlo
ELBO objective (`core.stats.variational`): its noisy stochastic gradients
break LBFGS's line search, whereas Adam's running moment estimates handle
them gracefully.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ._bounds import resolve_bounds
from ._layout import build_layout, flatten_params, unflatten_into_model
from ._protocol import ScalarClosure


class AdamOptimizer:
    """Adam optimizer wrapping `torch.optim.Adam` with optional box-bounds
    clamping after each step.

    Parameters
    ----------
    params:
        Either an `nn.Module` (preferred — needed for bounds clamping) or
        an iterable of `nn.Parameter`. Bound enforcement requires the
        module form so the optimizer can name parameters.
    lr, betas, eps, weight_decay, amsgrad:
        Forwarded to `torch.optim.Adam`. The `lr=1e-2` default is a touch
        higher than torch's `1e-3` — the log/logit-space parameterizations
        in `core.stats` (and the ELBO objective Adam exists for) tolerate
        and benefit from the larger step.
    bounds:
        Optional `dict[name, (lo, hi)]`. Resolution is exact match, then
        fnmatch glob, then unbounded (±1e10). Bounds operate on the
        parameter as registered (e.g. `log_sigma`); use
        `core.optim.bounds_in_natural_units` to convert from natural units.
    """

    def __init__(
        self,
        params: nn.Module | Iterable[nn.Parameter],
        *,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if isinstance(params, nn.Module):
            self._model: nn.Module | None = params
            param_iter = list(params.parameters())
        else:
            self._model = None
            param_iter = list(params)
            if bounds is not None:
                raise ValueError(
                    "bounds= requires `params` to be an nn.Module so parameter "
                    "names are available; got an iterable of Parameters"
                )
        self._inner = torch.optim.Adam(
            param_iter,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self._bounds_dict = bounds
        if bounds is not None and self._model is not None:
            self._layout = build_layout(self._model)
            dtype = next(self._model.parameters()).dtype
            self._lower, self._upper = resolve_bounds(
                self._layout, bounds, dtype=dtype
            )
        else:
            self._layout = None
            self._lower = None
            self._upper = None

    def step(self, closure: ScalarClosure) -> torch.Tensor:
        """Run one Adam update. The user's `closure` returns un-backed loss;
        we zero grads, evaluate, backward, and step internally, then clamp
        into the bounds box if `bounds=` was supplied. Returns the loss at
        the pre-step parameters (a 0-d tensor)."""
        self._inner.zero_grad(set_to_none=False)
        loss = closure()
        loss.backward()
        self._inner.step()
        if self._lower is not None:
            self._clamp()
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss)
        return loss

    def _clamp(self) -> None:
        assert self._model is not None and self._layout is not None
        flat = flatten_params(self._model, self._layout)
        clamped = flat.clamp(self._lower, self._upper)
        if not torch.equal(flat, clamped):
            unflatten_into_model(self._model, clamped, self._layout)
