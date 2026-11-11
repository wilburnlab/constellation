"""What the stages of a peak picker hand one another.

A :class:`Trace` is a batch of traces after preprocessing, *compacted*:
each row's valid samples first, in order, padding at the end, so every
kernel sees padding only past a row's ``count``. :class:`Candidates` are a
detector's apexes in that compacted space. :class:`Peaks` is what a
``PeakPicker`` returns, its indices mapped back to the positions the
caller gave.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["Trace", "Candidates", "Peaks"]


@dataclass(frozen=True)
class Trace:
    """Rows of a batch, compacted (module docstring): values ``y`` (after
    smoothing), their axis ``x`` (float64), validity, each row's count of
    samples, and what preprocessing measured: a ``baseline`` per sample and
    a ``noise`` sigma per row or per sample (None where no stage gave one).
    Padding values are NaN. A detection trace's valid samples are the first
    ``count`` of its row; a boundary trace keeps the detection trace's
    samples and count, and its ``valid`` can have holes among them."""

    y: torch.Tensor  # [B, N] float64
    x: torch.Tensor  # [B, N] float64
    valid: torch.Tensor  # [B, N] bool, a prefix of each row (holes in a boundary trace)
    count: torch.Tensor  # [B] int64
    baseline: torch.Tensor | None = None  # [B, N]
    noise: torch.Tensor | None = None  # [B] or [B, N]
    raw: torch.Tensor | None = None  # [B, N], before smoothing

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.y.shape)  # type: ignore[return-value]

    def baseline_or_zero(self) -> torch.Tensor:
        return torch.zeros_like(self.y) if self.baseline is None else torch.nan_to_num(self.baseline, nan=0.0)

    def noise_at(self, index: torch.Tensor) -> torch.Tensor:
        """The noise sigma at compacted positions ``index [B, M]`` (NaN where none)."""
        if self.noise is None:
            return torch.full(index.shape, float("nan"), dtype=torch.float64)
        if self.noise.dim() == 1:
            return self.noise.unsqueeze(-1).expand(index.shape).to(torch.float64)
        return torch.gather(self.noise.to(torch.float64), -1, index.clamp(min=0))


@dataclass(frozen=True)
class Candidates:
    """A detector's peaks, ``[B, M]`` in compacted positions: the apex, a
    score (higher is a better peak; what it measures is the detector's), and
    the detector's own boundaries where it has them (``-1`` none). ``extras``
    holds anything else a detector measures, ``[B, M]`` each. ``baseline``
    (``[B, N]``) and ``noise`` (``[B]`` or ``[B, N]``) are what a detector
    estimated for itself where the trace gave it none, for the picker to
    measure the peaks with."""

    apex: torch.Tensor  # int64, -1 padding
    score: torch.Tensor  # float64
    valid: torch.Tensor  # bool
    left: torch.Tensor | None = None
    right: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    baseline: torch.Tensor | None = None
    noise: torch.Tensor | None = None

    @property
    def m(self) -> int:
        return int(self.apex.shape[-1])

    def take(self, order: torch.Tensor) -> Candidates:
        """The candidates in ``order`` (``[B, M']`` positions into M)."""

        def g(t):
            return None if t is None else torch.gather(t, -1, order)

        return Candidates(
            apex=g(self.apex), score=g(self.score), valid=g(self.valid), left=g(self.left), right=g(self.right),
            extras={k: g(v) for k, v in self.extras.items()}, baseline=self.baseline, noise=self.noise,
        )  # fmt: skip


@dataclass(frozen=True)
class Peaks:
    """Peaks of a batch, padded to ``[..., M]`` with ``valid``, best
    ``score`` first. Indices are positions along the caller's last axis
    (-1 for padding); ``*_x`` are on the caller's axis ``x``.

    - ``apex``: where the detector put the peak, on the detection trace;
      ``apex_x``: its refined position. ``boundary_apex``: the apex on the
      boundary trace (the same unless a separate boundary trace was given
      and the apex re-centred on it).
    - ``left`` / ``right``: the peak's extent (inclusive), and their x.
    - ``left_base`` / ``right_base``: the prominence's bases (where the
      lowest points between this peak and higher ground are), kept apart
      from the extent.
    - ``height``: the detection trace at the apex; ``prominence``; ``snr``:
      height above the baseline over the noise; ``score``: the detector's.
    - ``width``: full width at half the boundary trace's height above its
      baseline, in x; ``area``: the boundary trace above its baseline over
      the extent (trapezoid in x).
    - ``asymmetry``: the right half-width over the left, at half height;
      ``jaggedness``: the share of direction changes of the boundary trace
      inside the extent.
    - ``extras``: detector-specific measures; ``components``: fitted shape
      parameters ``[..., M, P]`` of ``form`` (a refiner's), else None.
    """

    valid: torch.Tensor
    apex: torch.Tensor
    apex_x: torch.Tensor
    boundary_apex: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    left_x: torch.Tensor
    right_x: torch.Tensor
    left_base: torch.Tensor
    right_base: torch.Tensor
    height: torch.Tensor
    prominence: torch.Tensor
    snr: torch.Tensor
    score: torch.Tensor
    width: torch.Tensor
    area: torch.Tensor
    asymmetry: torch.Tensor
    jaggedness: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    components: torch.Tensor | None = None
    form: str | None = None

    @property
    def n(self) -> torch.Tensor:
        """``[...]``: how many peaks each row has."""
        return self.valid.sum(dim=-1)
