"""The phase of one periodic component of a signal on a continuous axis.

A signal folded onto a period ``P`` is a sum of arrows on a circle, one per
point, each at the angle of ``x`` modulo ``P`` and as long as its weight. The
direction of their sum is that component's phase: it says where the signal
clusters within the period. It is one Fourier coefficient, at a frequency
chosen freely, evaluated directly rather than by binning the axis and taking
a whole FFT.

A symmetric peak shape only scales that coefficient (convolution multiplies
it by the shape's transform, which for a symmetric shape is real), so the
phase of points smeared by such a shape is the phase of their centres.

:class:`PhaseFold` accumulates the coefficient in fine bins of the axis and
in blocks of the data, in one pass over data of any size. Any window of the
axis is then a sum of bins, and leaving out one block at a time gives a
jackknife standard error of its phase, which is what says whether a window
holds enough signal to believe. Generic: nothing here knows about masses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["PhaseFold", "WindowPhase", "wrapped"]


def wrapped(angle: torch.Tensor | float) -> torch.Tensor:
    """An angle in radians, in ``(-pi, pi]``."""
    value = torch.as_tensor(angle, dtype=torch.float64)
    return torch.atan2(torch.sin(value), torch.cos(value))


@dataclass(frozen=True)
class WindowPhase:
    """One window of the axis: the phase of what it holds, in radians."""

    #: Against a reference, the phase difference; otherwise the phase.
    phase: float
    #: Jackknife standard error of ``phase`` over the blocks; infinite with
    #: fewer than two blocks holding anything.
    se: float
    #: Length of the sum over the sum of lengths: 1 when every arrow points
    #: the same way, near 0 when they cancel.
    coherence: float
    weight: float
    blocks: int


class PhaseFold:
    """The coefficient at one period, by bin of the axis and block of the data."""

    def __init__(
        self,
        period: float,
        lo: float,
        hi: float,
        *,
        bin_width: float = 1.0,
        n_blocks: int = 32,
    ) -> None:
        if period <= 0 or bin_width <= 0 or hi <= lo or n_blocks < 1:
            raise ValueError("a fold needs a positive period and bin width, and hi above lo")
        self.period = float(period)
        self.lo = float(lo)
        self.bin_width = float(bin_width)
        self.n_bins = int(math.ceil((hi - lo) / bin_width))
        self.n_blocks = int(n_blocks)
        shape = (self.n_blocks, self.n_bins)
        self._re = torch.zeros(shape, dtype=torch.float64)
        self._im = torch.zeros(shape, dtype=torch.float64)
        self._weight = torch.zeros(shape, dtype=torch.float64)

    def add(self, x: torch.Tensor, weights: torch.Tensor, block: int = 0) -> None:
        """Add points at ``x`` with ``weights`` to ``block``."""
        bins = torch.floor((x - self.lo) / self.bin_width).long()
        keep = (bins >= 0) & (bins < self.n_bins)
        bins, x, weights = bins[keep], x[keep], weights[keep]
        angle = 2.0 * math.pi * x / self.period
        block %= self.n_blocks
        self._re[block].index_add_(0, bins, weights * torch.cos(angle))
        self._im[block].index_add_(0, bins, weights * torch.sin(angle))
        self._weight[block].index_add_(0, bins, weights)

    def _span(self, lo: float, hi: float) -> tuple[int, int]:
        a = max(int(math.floor((lo - self.lo) / self.bin_width)), 0)
        b = min(int(math.ceil((hi - self.lo) / self.bin_width)), self.n_bins)
        return a, max(a, b)

    def occupied(self) -> tuple[float, float] | None:
        """The extent of the axis holding anything; None when nothing does."""
        filled = torch.nonzero(self._weight.sum(0) > 0)
        if filled.numel() == 0:
            return None
        first, last = int(filled[0]), int(filled[-1])
        return self.lo + first * self.bin_width, self.lo + (last + 1) * self.bin_width

    def coefficient(self, lo: float, hi: float) -> complex:
        """The summed coefficient of ``[lo, hi)``."""
        a, b = self._span(lo, hi)
        return complex(self._re[:, a:b].sum().item(), self._im[:, a:b].sum().item())

    def window(
        self,
        lo: float,
        hi: float,
        *,
        reference: complex | None = None,
        blocks: torch.Tensor | None = None,
    ) -> WindowPhase:
        """The phase of ``[lo, hi)``, against ``reference`` when one is given.

        ``blocks`` (a boolean mask) restricts the window to some of the data.
        """
        a, b = self._span(lo, hi)
        mask = torch.ones(self.n_blocks, dtype=torch.bool) if blocks is None else blocks
        re = self._re[:, a:b].sum(1) * mask
        im = self._im[:, a:b].sum(1) * mask
        weight = self._weight[:, a:b].sum(1) * mask
        total = torch.complex(re.sum(), im.sum())
        turn = torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
        if reference is not None and reference != 0:
            turn = torch.conj(torch.tensor(reference, dtype=torch.complex128))
            turn = turn / torch.abs(turn)
        phase = torch.angle(total * turn)
        filled = weight > 0
        n = int(filled.sum())
        if n >= 2:
            # Each block left out, measured against the whole.
            rest = total - torch.complex(re, im)
            leave_one_out = torch.angle(rest[filled] * torch.conj(total))
            se = math.sqrt((n - 1) / n * float((leave_one_out**2).sum()))
        else:
            se = math.inf
        weight_total = float(weight.sum())
        coherence = float(torch.abs(total)) / weight_total if weight_total > 0 else 0.0
        return WindowPhase(float(phase), se, coherence, weight_total, n)
