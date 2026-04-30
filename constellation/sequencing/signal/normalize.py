"""Squiggle normalization — int16 ADC → picoamperes → mad-normalized.

Two-stage normalization the lab's downstream signal models will need:

    raw_to_picoamperes   undo the int16 quantization with the per-read
                         ``scale`` / ``offset`` from RAW_SIGNAL_TABLE
                         (current_pA = scale * raw + offset)

    mad_normalize        standardize against the per-read median + MAD
                         (median absolute deviation) — robust to outlier
                         spikes that mean/std normalization would let
                         dominate

Both operate on torch tensors (the format custom basecallers will
consume) and accept either single-read 1D tensors or padded batch
tensors with an explicit ``length`` per row.

Status: STUB. Pending Phase 13.
"""

from __future__ import annotations

import torch


_PHASE = "Phase 13 (signal/* + modifications/*)"


def raw_to_picoamperes(
    raw: torch.Tensor,
    *,
    scale: float | torch.Tensor,
    offset: float | torch.Tensor,
) -> torch.Tensor:
    """Convert int16 ADC counts to picoamperes via per-read scale + offset.

    ``raw`` may be 1D (single read) or 2D ``(batch, length)`` — scale
    and offset broadcast accordingly. Output is float32.
    """
    raise NotImplementedError(f"raw_to_picoamperes pending {_PHASE}")


def mad_normalize(
    signal_pa: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Median + MAD-scaled normalization, robust to outlier spikes.

    For padded batches, supply per-row ``lengths`` so statistics
    aren't biased by padding.
    """
    raise NotImplementedError(f"mad_normalize pending {_PHASE}")


__all__ = [
    "raw_to_picoamperes",
    "mad_normalize",
]
