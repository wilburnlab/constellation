"""Row helpers shared by the batched trace operations.

Every trace operation in ``core.signal`` takes values ``[..., N]`` with a
validity mask of the same shape, and treats each row on its own. Kernels
that slide along a row (convolutions, searches) are simplest when a row's
valid entries are contiguous and its padding is at the end, so they first
:func:`compact` it: valid entries move to the front in their order, and the
permutation maps results back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Compacted", "compact", "expand"]


@dataclass(frozen=True)
class Compacted:
    """A row-wise permutation putting each row's valid entries first, in
    their order: ``order[..., j]`` is the original position of compacted
    entry ``j``, ``count[...]`` how many entries are valid."""

    order: torch.Tensor  # int64 [..., N]
    count: torch.Tensor  # int64 [...]

    @property
    def valid(self) -> torch.Tensor:
        """``[..., N]``: whether compacted entry ``j`` is a valid one."""
        n = self.order.shape[-1]
        return torch.arange(n, device=self.order.device) < self.count.unsqueeze(-1)

    def gather(self, values: torch.Tensor, fill: float | int | bool | None = None) -> torch.Tensor:
        """``values`` (``[..., N]``) in compacted order; with ``fill``, the
        padding entries set to it."""
        out = torch.gather(values, -1, self.order)
        if fill is None:
            return out
        return torch.where(self.valid, out, torch.full_like(out, fill))

    def position(self, index: torch.Tensor) -> torch.Tensor:
        """Original positions of compacted indices ``index`` (``[..., M]``,
        ``-1`` for none), ``-1`` kept."""
        safe = index.clamp(min=0)
        out = torch.gather(self.order, -1, safe)
        return torch.where(index >= 0, out, torch.full_like(out, -1))


def compact(valid: torch.Tensor) -> Compacted:
    """The permutation moving each row's valid entries to its front, stably
    (:class:`Compacted`)."""
    valid = valid.bool()
    # A stable sort on "invalid" keeps the valid entries in their order.
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    return Compacted(order=order, count=valid.sum(dim=-1))


def expand(compacted: Compacted, values: torch.Tensor, fill: float | int | bool) -> torch.Tensor:
    """Compacted ``values`` (``[..., N]``) back in original order, with
    ``fill`` where the original entry was not valid."""
    out = torch.full_like(values, fill)
    kept = torch.where(compacted.valid, values, torch.full_like(values, fill))
    return out.scatter(-1, compacted.order, kept)
