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

__all__ = ["Compacted", "compact", "expand", "valid_neighbours"]


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


def valid_neighbours(valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(at_or_before, at_or_after)``, each ``[..., N]``: the nearest valid
    position at or before each position (``-1`` for none) and at or after
    it (``N`` for none). For rows with holes in their validity."""
    n = valid.shape[-1]
    idx = torch.arange(n, device=valid.device).expand_as(valid)
    before = torch.cummax(torch.where(valid, idx, torch.full_like(idx, -1)), dim=-1).values
    after = torch.where(valid, idx, torch.full_like(idx, n)).flip(-1).cummin(dim=-1).values.flip(-1)
    return before, after


# ── range queries ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RangeTable:
    """Sparse tables of a row's range minima and maxima, for searches that
    every peak in a batch asks at once in ``O(log N)`` vectorized steps:
    the nearest sample left or right of a position past a level (where a
    peak's prominence search stops, where a width crossing is), and the
    minimum over a range with the position it is at (a prominence base).

    Built from compacted values ``y [..., N]`` and each row's count of valid
    samples; padding never answers a search, nor does a sample ``valid``
    (if given) leaves out, a hole: a search passes it, as a minimum or a
    maximum ignores it. ``levels[l][..., i]`` holds the max (min) of
    ``y[i : i + 2**l]``. Memory ``O(B N log N)``.
    """

    lo: tuple[torch.Tensor, ...]  # range minima per level
    hi: tuple[torch.Tensor, ...]  # range maxima per level
    lo_tie_right: tuple[torch.Tensor, ...]  # where the minimum is, the larger index on a tie
    lo_tie_left: tuple[torch.Tensor, ...]  # where the minimum is, the smaller index on a tie
    count: torch.Tensor  # [...]

    @property
    def n(self) -> int:
        return int(self.lo[0].shape[-1])

    @classmethod
    def build(cls, y: torch.Tensor, count: torch.Tensor, valid: torch.Tensor | None = None) -> RangeTable:
        n = y.shape[-1]
        idx = torch.arange(n, device=y.device).expand_as(y)
        inside = idx < count.unsqueeze(-1)
        if valid is not None:
            inside = inside & valid.bool()
        lo0 = torch.where(inside, y, torch.full_like(y, float("inf")))
        hi0 = torch.where(inside, y, torch.full_like(y, float("-inf")))
        lo, hi, at_r, at_l = [lo0], [hi0], [idx], [idx]
        step = 1
        while 2 * step <= max(n, 1):
            a_lo, b_lo = lo[-1][..., : n - step], lo[-1][..., step:]
            a_hi, b_hi = hi[-1][..., : n - step], hi[-1][..., step:]
            pad_lo = lo[-1][..., n - step:]
            pad_hi = hi[-1][..., n - step:]
            lo.append(torch.cat([torch.minimum(a_lo, b_lo), pad_lo], dim=-1))
            hi.append(torch.cat([torch.maximum(a_hi, b_hi), pad_hi], dim=-1))
            ra, rb = at_r[-1][..., : n - step], at_r[-1][..., step:]
            la, lb = at_l[-1][..., : n - step], at_l[-1][..., step:]
            # Minimum's position: the right one on a tie (a left base is the
            # minimum nearest its peak) or the left one (a right base).
            at_r.append(torch.cat([torch.where(b_lo <= a_lo, rb, ra), at_r[-1][..., n - step:]], dim=-1))
            at_l.append(torch.cat([torch.where(a_lo <= b_lo, la, lb), at_l[-1][..., n - step:]], dim=-1))
            step *= 2
        return cls(tuple(lo), tuple(hi), tuple(at_r), tuple(at_l), count)

    def _get(self, tables: tuple[torch.Tensor, ...], level: int, at: torch.Tensor) -> torch.Tensor:
        return torch.gather(tables[level], -1, at.clamp(0, max(self.n - 1, 0)))

    def nearest_left(self, at: torch.Tensor, level: torch.Tensor, *, above: bool = True,
                     limit: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
        """For positions ``at [..., M]``, the nearest index ``j < at`` with
        ``y[j] > level`` (``above``) or ``y[j] < level`` (not ``above``),
        not before ``limit`` (default 0); ``-1`` for none."""
        tables = self.hi if above else self.lo
        j = at.clone()
        floor = torch.zeros_like(at) if limit is None else limit.clamp(min=0)
        for lv in range(len(tables) - 1, -1, -1):
            size = 1 << lv
            start = j - size
            ok = start >= floor
            block = self._get(tables, lv, start)
            skip = ok & ((block <= level) if above else (block >= level))
            j = torch.where(skip, start, j)
        found = j - 1
        hit = found >= floor
        value = torch.gather(tables[0], -1, found.clamp(0, max(self.n - 1, 0)))
        hit &= (value > level) if above else (value < level)
        return torch.where(hit, found, torch.full_like(found, -1))

    def nearest_right(self, at: torch.Tensor, level: torch.Tensor, *, above: bool = True,
                      limit: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
        """For positions ``at``, the nearest index ``j > at`` with
        ``y[j] > level`` (``above``) or ``< level``, not past ``limit``
        (default the row's last valid sample); ``-1`` for none."""
        tables = self.hi if above else self.lo
        last = (self.count - 1).unsqueeze(-1).expand_as(at) if limit is None else limit
        last = torch.minimum(last, (self.count - 1).unsqueeze(-1).expand_as(at))
        j = at + 1
        for lv in range(len(tables) - 1, -1, -1):
            size = 1 << lv
            ok = j + size - 1 <= last
            block = self._get(tables, lv, j)
            skip = ok & ((block <= level) if above else (block >= level))
            j = torch.where(skip, j + size, j)
        hit = j <= last
        value = torch.gather(tables[0], -1, j.clamp(0, max(self.n - 1, 0)))
        hit &= (value > level) if above else (value < level)
        return torch.where(hit, j, torch.full_like(j, -1))

    def argmin(self, a: torch.Tensor, b: torch.Tensor, *, prefer: str = "right") -> tuple[torch.Tensor, torch.Tensor]:
        """The minimum over ``[a, b]`` (inclusive; ``a <= b``) and where it
        is: on a tie, the position nearest ``b`` (``prefer="right"``) or
        nearest ``a``."""
        length = (b - a + 1).clamp(min=1)
        lv = torch.floor(torch.log2(length.to(torch.float64))).to(torch.int64)
        out_v = torch.full(a.shape, float("inf"), dtype=self.lo[0].dtype, device=a.device)
        out_i = a.clone()
        for level in range(len(self.lo)):
            sel = lv == level
            if not sel.any():
                continue
            size = 1 << level
            first, second = a, b - size + 1
            v1, v2 = self._get(self.lo, level, first), self._get(self.lo, level, second)
            at = self.lo_tie_left if prefer == "left" else self.lo_tie_right
            i1, i2 = self._get(at, level, first), self._get(at, level, second)
            if prefer == "right":
                take2 = v2 <= v1
            else:
                take2 = v2 < v1
            v = torch.where(take2, v2, v1)
            i = torch.where(take2, i2, i1)
            out_v = torch.where(sel, v, out_v)
            out_i = torch.where(sel, i, out_i)
        return out_v, out_i
