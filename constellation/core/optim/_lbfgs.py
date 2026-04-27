"""`LBFGSOptimizer` — production wrapper around `torch.optim.LBFGS`.

Two responsibilities the test-shim version (formerly in
`tests/conftest.py`) didn't take on:

1. **Closure inversion.** The user's scalar closure returns un-backed
   loss; `LBFGSOptimizer` builds a wrapped closure that calls
   `zero_grad` + `backward` itself. This unifies LBFGS with DE
   (gradient-free) under the same `Optimizer` Protocol — DE doesn't
   waste compute on a `loss.backward()` it never reads.

2. **Bounds-clamping after step.** When `bounds=` is supplied,
   parameters are clamped back into the box after every `step()`.
   This is the carry-forward of the cartographer 0cee299 polish bug
   fix: an unconstrained LBFGS line search can escape DE bounds
   (e.g. `tau_l → 190s` creating flat baselines). Folding the clamp
   into the optimizer means DE's `polish()` reuses this exact path
   instead of duplicating the logic.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ._bounds import resolve_bounds
from ._layout import build_layout, flatten_params, unflatten_into_model
from ._protocol import ScalarClosure


class LBFGSOptimizer:
    """L-BFGS optimizer wrapping `torch.optim.LBFGS` with optional
    box-bounds clamping after each step.

    Parameters
    ----------
    params:
        Either an `nn.Module` (preferred — needed for bounds clamping)
        or an iterable of `nn.Parameter`. Bound enforcement requires
        the module form so the optimizer can name parameters.
    lr, max_iter, tolerance_grad, tolerance_change, history_size,
    line_search_fn:
        Forwarded to `torch.optim.LBFGS`. `line_search_fn="strong_wolfe"`
        is the default (vs. torch's None) — it's far more robust on the
        peak-fitting workloads this optimizer was designed for.
    bounds:
        Optional `dict[name, (lo, hi)]`. Resolution is exact match,
        then fnmatch glob, then unbounded (±1e10). Bounds operate on
        the parameter as registered (e.g. `log_sigma`).
    """

    def __init__(
        self,
        params: nn.Module | Iterable[nn.Parameter],
        *,
        lr: float = 1.0,
        max_iter: int = 20,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = "strong_wolfe",
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
        self._inner = torch.optim.LBFGS(
            param_iter,
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
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
        """Run one LBFGS step. The user's `closure` returns un-backed
        loss; we wrap it to manage gradients internally."""

        def wrapped() -> torch.Tensor:
            self._inner.zero_grad(set_to_none=False)
            loss = closure()
            loss.backward()
            return loss

        loss = self._inner.step(wrapped)
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
