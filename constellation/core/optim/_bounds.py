"""Per-parameter bounds resolution and natural-units helpers.

Bounds always operate on the parameter as registered on the `nn.Module`
(e.g. `log_sigma`, not `sigma`). The `bounds_in_natural_units` helper
converts a natural-units dict into a registered-name dict for users
who'd rather think in `(sigma_min, sigma_max)` than `(log(sigma_min),
log(sigma_max))` — but the conversion is opt-in and explicit; no name-
prefix detection happens automatically.

`_DEFAULT_BOUND` (±1e10) is the sentinel for "unbounded" — large enough
to never bind in practice but finite so DE init logic can still bracket
unbounded dimensions around the warm-start point.
"""

from __future__ import annotations

import fnmatch
import math
from typing import Iterable

import torch
from torch import nn

from ._layout import ParamLayout

_DEFAULT_BOUND: float = 1e10


def resolve_bounds(
    layout: ParamLayout,
    spec: dict[str, tuple[float, float]] | None,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flat (lower, upper) tensors aligned with `layout`.

    Resolution order per parameter name: exact match → first-matching
    fnmatch glob → ±_DEFAULT_BOUND. Bounds in `spec` may use shell-glob
    patterns (`"log_*": (-5, 5)`).
    """
    lower = torch.full((layout.n_flat,), -_DEFAULT_BOUND, dtype=dtype, device=device)
    upper = torch.full((layout.n_flat,), _DEFAULT_BOUND, dtype=dtype, device=device)
    if spec is None:
        return lower, upper

    exact = {k: v for k, v in spec.items() if not _is_glob(k)}
    globs = [(k, v) for k, v in spec.items() if _is_glob(k)]

    for name, shape, off in zip(layout.names, layout.shapes, layout.offsets):
        n = int(torch.tensor(shape).prod().item()) if len(shape) > 0 else 1
        bound = exact.get(name)
        if bound is None:
            for pattern, b in globs:
                if fnmatch.fnmatchcase(name, pattern):
                    bound = b
                    break
        if bound is None:
            continue
        lo, hi = bound
        if not (lo < hi):
            raise ValueError(
                f"bounds for {name!r} must satisfy lo < hi; got ({lo}, {hi})"
            )
        lower[off : off + n] = float(lo)
        upper[off : off + n] = float(hi)
    return lower, upper


def _is_glob(s: str) -> bool:
    return any(c in s for c in "*?[")


def bounds_in_natural_units(
    model: nn.Module,
    natural_bounds: dict[str, tuple[float, float]],
    *,
    log_params: Iterable[str] = (),
    logit_params: Iterable[str] = (),
) -> dict[str, tuple[float, float]]:
    """Translate natural-units bounds into the model's registered-parameter
    names. Caller declares which natural names map to log- or logit-space
    parameters; no name-prefix magic.

    Example:
        bounds_in_natural_units(
            peak,
            {"sigma": (1e-3, 1e3), "tau": (1e-2, 1e2), "t_apex": (0.0, 60.0)},
            log_params=["sigma", "tau"],
        )
        # → {"log_sigma": (log(1e-3), log(1e3)),
        #    "log_tau":   (log(1e-2), log(1e2)),
        #    "t_apex":    (0.0, 60.0)}

    Validates that the converted names exist on `model`. Unknown
    natural names raise; this is intentional — silent omissions are
    the failure mode the explicit API exists to prevent.
    """
    log_set = set(log_params)
    logit_set = set(logit_params)
    registered = set(dict(model.named_parameters()).keys())
    out: dict[str, tuple[float, float]] = {}
    for name, (lo, hi) in natural_bounds.items():
        if not (lo < hi):
            raise ValueError(
                f"bounds for {name!r} must satisfy lo < hi; got ({lo}, {hi})"
            )
        if name in log_set:
            target = f"log_{name}"
            if lo <= 0:
                raise ValueError(
                    f"log-space conversion requires lo > 0; got {name}={lo}"
                )
            out[target] = (math.log(float(lo)), math.log(float(hi)))
        elif name in logit_set:
            target = f"logit_{name}"
            if not (0 < lo < 1 and 0 < hi < 1):
                raise ValueError(
                    f"logit-space conversion requires 0 < lo < hi < 1; "
                    f"got {name}=({lo}, {hi})"
                )
            out[target] = (
                math.log(float(lo) / (1.0 - float(lo))),
                math.log(float(hi) / (1.0 - float(hi))),
            )
        else:
            target = name
            out[target] = (float(lo), float(hi))
        if target not in registered:
            raise KeyError(
                f"parameter {target!r} not registered on {type(model).__name__}; "
                f"available: {sorted(registered)}"
            )
    return out


def apply_bounds(
    population: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    mode: str = "clip",
) -> torch.Tensor:
    """Project a population back into bounds. `mode="clip"` clamps;
    `mode="reflect"` mirrors out-of-bounds positions back into [lo, hi].
    Reflection handles arbitrarily-large overshoots via fold-twice."""
    if mode == "clip":
        return population.clamp(lower, upper)
    if mode == "reflect":
        span = (upper - lower).clamp(min=1e-30)
        # Fold into [0, 2*span], then reflect into [0, span], then translate.
        shifted = (population - lower) % (2.0 * span)
        reflected = torch.where(shifted > span, 2.0 * span - shifted, shifted)
        return lower + reflected
    raise ValueError(f"unknown bounds mode {mode!r}; want 'clip' or 'reflect'")
