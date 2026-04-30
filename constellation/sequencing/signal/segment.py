"""Squiggle segmentation — translocation events, dwell times, k-mer windows.

Segmentation identifies the points in a squiggle where the
translocating strand transitions from one k-mer to the next. The
dwell time at each k-mer is the key observable that physical-model
basecallers consume; the lab's planned synthetic-library work targets
this directly via in-house oligonucleotide standards.

Status: STUB. Pending Phase 13.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_PHASE = "Phase 13 (signal/segment)"


@dataclass(frozen=True)
class SegmentationResult:
    """Output of squiggle segmentation.

    boundaries: (n_segments + 1,) int64 sample-index breakpoints
    means:      (n_segments,) float32 mean current per segment
    dwells:     (n_segments,) float32 segment lengths in samples
    """

    boundaries: torch.Tensor
    means: torch.Tensor
    dwells: torch.Tensor


def segment_squiggle(
    signal: torch.Tensor,
    *,
    method: str = "tcombo",  # 't-test combo' — placeholder
    min_dwell: int = 4,
) -> SegmentationResult:
    """Segment a normalized squiggle into per-k-mer event regions.

    ``method`` selects the segmentation algorithm. Phase 13 ships a
    couple of options (t-test sliding windows, change-point detection,
    HMM); ``"tcombo"`` is the default placeholder.
    """
    raise NotImplementedError(f"segment_squiggle pending {_PHASE}")


__all__ = [
    "SegmentationResult",
    "segment_squiggle",
]
