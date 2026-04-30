"""Phred quality codec — offset-33 ASCII <-> Q-score tensors.

FASTQ / BAM encode per-base quality as ASCII characters offset by 33
(``'!'`` = Q0, ``'I'`` = Q40). Q-scores in turn relate to error
probability as ``Q = -10 * log10(p_error)``. ``READ_TABLE.quality``
stores the raw ASCII string for round-trip fidelity; this module
exposes the decode helpers downstream code uses when it needs the
numeric form.

Status: STUB. Function bodies pending Phase 2 (Reader/Writer
Protocols + readers/{fastx, sam_bam, pod5}). See plan file
in-our-development-of-fuzzy-quilt.md.
"""

from __future__ import annotations

import torch


_PHASE = "Phase 2 (Reader/Writer Protocols + readers)"

PHRED_OFFSET: int = 33
"""Offset between ASCII codepoint and Q-score (FASTQ / BAM convention)."""


def decode_phred(quality: str) -> torch.Tensor:
    """Decode an offset-33 ASCII Phred string to a 1D int8 Q-score tensor."""
    raise NotImplementedError(f"decode_phred pending {_PHASE}")


def encode_phred(q_scores: torch.Tensor) -> str:
    """Encode a 1D int8 Q-score tensor as offset-33 ASCII."""
    raise NotImplementedError(f"encode_phred pending {_PHASE}")


def q_to_p_error(q_scores: torch.Tensor) -> torch.Tensor:
    """Convert Q-scores to per-base error probabilities ``10**(-Q/10)``.

    Returns float32 to match downstream-tensor convention; callers that
    want float64 should ``.to(torch.float64)`` at the call site.
    """
    raise NotImplementedError(f"q_to_p_error pending {_PHASE}")


def mean_q(quality: str) -> float:
    """Per-read aggregate Q-score using the *probability* mean
    (not arithmetic mean of Q values, which is misleading because Q is
    a log scale).

    ``Q_mean = -10 * log10(mean(p_error))``
    """
    raise NotImplementedError(f"mean_q pending {_PHASE}")


__all__ = [
    "PHRED_OFFSET",
    "decode_phred",
    "encode_phred",
    "q_to_p_error",
    "mean_q",
]
