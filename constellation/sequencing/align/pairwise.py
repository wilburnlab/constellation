"""Pairwise sequence alignment — edlib / parasail backends.

Single ``pairwise_align(query, ref, *, backend, mode, ...)`` function
hides the backend choice. ``edlib`` is the default for edit-distance-
shaped problems (adapter / barcode location, demux scoring); parasail
is faster for SIMD-accelerated Smith-Waterman with arbitrary scoring
matrices.

Both deps live in the ``[sequencing]`` extra; imports are deferred to
inside the function body so the module imports without them installed.

Status: STUB. Pending Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_PHASE = "Phase 4 (align/pairwise + edlib/parasail)"


@dataclass(frozen=True)
class PairwiseResult:
    """Result of a pairwise alignment.

    score:        edit distance (edlib) or alignment score (parasail)
    query_start:  0-based half-open positions on the query
    query_end:    ditto
    ref_start:    0-based half-open positions on the reference
    ref_end:      ditto
    cigar:        CIGAR string ('50M5I20M') if requested; else None
    """

    score: int
    query_start: int
    query_end: int
    ref_start: int
    ref_end: int
    cigar: str | None = None


def pairwise_align(
    query: str,
    ref: str,
    *,
    backend: Literal["edlib", "parasail"] = "edlib",
    mode: Literal["global", "local", "infix", "prefix"] = "infix",
    max_distance: int | None = None,
    return_cigar: bool = False,
) -> PairwiseResult:
    """Align ``query`` against ``ref`` with the chosen backend.

    Modes:
        global    end-to-end alignment of both sequences
        local     best local match (parasail only — Smith-Waterman)
        infix     query embedded in ref (edlib's HW mode); the demux
                  workhorse for adapter/barcode location
        prefix    query at the ref start (edlib's SHW mode)

    edlib supports infix / prefix / global; parasail supports local /
    global. Backend choice should follow the alignment shape.

    ``max_distance`` (edlib only) bails early when edit distance
    exceeds a threshold — useful for fast adapter rejection.
    """
    raise NotImplementedError(f"pairwise_align pending {_PHASE}")


__all__ = [
    "PairwiseResult",
    "pairwise_align",
]
