"""Parameter layout helpers — flatten / unflatten `nn.Module` parameters
between the natural (named, multi-tensor) view and the population view
((pop_size, n_flat) tensor).

DE keeps its population as a single 2-D tensor for vectorized mutation
and crossover; converting back to the per-name dict that
`torch.func.functional_call` consumes happens once per generation in
`_params_dict_from_population`. LBFGSOptimizer reuses the same layout
+ flatten/unflatten for bounds-clamping after a step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParamLayout:
    """Flat-vector layout for `nn.Module` parameters.

    `names`, `shapes`, `offsets` are aligned: parameter `names[i]`
    occupies `flat[offsets[i] : offsets[i] + numel(shapes[i])]` and
    reshapes to `shapes[i]`. `n_flat` is the total parameter count.
    """

    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    offsets: tuple[int, ...]
    n_flat: int


def build_layout(model: nn.Module) -> ParamLayout:
    """Snapshot `model.named_parameters()` into a frozen layout."""
    names: list[str] = []
    shapes: list[torch.Size] = []
    offsets: list[int] = []
    offset = 0
    for name, p in model.named_parameters():
        names.append(name)
        shapes.append(p.shape)
        offsets.append(offset)
        offset += p.numel()
    return ParamLayout(
        names=tuple(names),
        shapes=tuple(shapes),
        offsets=tuple(offsets),
        n_flat=offset,
    )


def flatten_params(model: nn.Module, layout: ParamLayout) -> torch.Tensor:
    """Concatenate the model's current parameters into a 1-D tensor
    in `layout` order. Detached, on the model's parameter dtype/device."""
    pieces: list[torch.Tensor] = []
    params = dict(model.named_parameters())
    for name in layout.names:
        pieces.append(params[name].detach().reshape(-1))
    return torch.cat(pieces)


def unflatten_into_model(
    model: nn.Module, flat: torch.Tensor, layout: ParamLayout
) -> None:
    """Write `flat` back into the model's parameters in-place. Uses
    `param.data.copy_(...)` so the assignment doesn't touch the autograd
    graph — appropriate for DE/polish post-step writes."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, shape, off in zip(layout.names, layout.shapes, layout.offsets):
            n = int(torch.tensor(shape).prod().item()) if len(shape) > 0 else 1
            params[name].data.copy_(flat[off : off + n].reshape(shape))


def params_dict_from_population(
    population: torch.Tensor, layout: ParamLayout
) -> dict[str, torch.Tensor]:
    """Slice a `(pop_size, n_flat)` population into the named-tensor
    dict that `torch.func.functional_call` accepts. Each entry has a
    leading `pop_size` dim followed by the parameter's natural shape:
    `{"log_sigma": (pop_size,), "weights": (pop_size, K), ...}`."""
    pop_size = population.shape[0]
    out: dict[str, torch.Tensor] = {}
    for name, shape, off in zip(layout.names, layout.shapes, layout.offsets):
        n = int(torch.tensor(shape).prod().item()) if len(shape) > 0 else 1
        out[name] = population[:, off : off + n].reshape((pop_size, *shape))
    return out


def params_dict_from_flat(
    flat: torch.Tensor, layout: ParamLayout
) -> dict[str, torch.Tensor]:
    """Single-individual variant — slices a `(n_flat,)` vector into the
    named-tensor dict. Used by polish closures that operate on the
    global-best individual without the population dim."""
    out: dict[str, torch.Tensor] = {}
    for name, shape, off in zip(layout.names, layout.shapes, layout.offsets):
        n = int(torch.tensor(shape).prod().item()) if len(shape) > 0 else 1
        out[name] = flat[off : off + n].reshape(shape)
    return out
