"""Pairwise sequence alignment — edlib backend (parasail deferred).

Single ``pairwise_align(query, ref, *, backend, mode, ...)`` function
hides the backend choice. ``edlib`` is the default for edit-distance-
shaped problems (adapter / barcode location, demux scoring). Parasail
support (SIMD Smith-Waterman with arbitrary scoring matrices) is
declared in the API but not yet implemented — it lands when a
non-edit-distance use case (assembly polishing, BLAST-style alignment)
needs it.

Edlib lives in the ``[sequencing]`` extra; import is deferred to
inside the function body so this module imports even if edlib isn't
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


# edlib mode names — the function exposes the conceptual mode names
# above; this maps them to edlib's three-letter codes internally.
_EDLIB_MODE = {
    "global": "NW",
    "infix": "HW",
    "prefix": "SHW",
}


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
        global    end-to-end alignment of both sequences (edlib NW)
        local     best local match (parasail only — Smith-Waterman)
        infix     query embedded in ref (edlib's HW mode); the demux
                  workhorse for adapter/barcode location
        prefix    query at the ref start (edlib's SHW mode)

    edlib supports infix / prefix / global; parasail supports local /
    global. Backend choice should follow the alignment shape.

    ``max_distance`` (edlib only) bails early when edit distance
    exceeds a threshold — useful for fast adapter rejection. Use ``-1``
    or ``None`` to leave it unbounded (edlib's default).

    Raises ``ValueError`` for invalid (backend, mode) pairs and
    ``RuntimeError`` if no alignment is found within the
    ``max_distance`` budget.
    """
    if backend == "parasail":
        raise NotImplementedError(
            "parasail backend is not yet implemented; pending an explicit "
            "non-edit-distance use case (S1 demux uses edlib)."
        )
    if backend != "edlib":
        raise ValueError(f"unknown backend {backend!r}")

    if mode == "local":
        raise ValueError(
            "mode='local' is parasail-only; use 'infix' for edlib's HW mode "
            "(query embedded in ref) instead"
        )
    if mode not in _EDLIB_MODE:
        raise ValueError(f"unknown mode {mode!r} for edlib backend")

    import edlib  # type: ignore[import-untyped]

    edlib_mode = _EDLIB_MODE[mode]
    task = "path" if return_cigar else "locations"
    k = -1 if max_distance is None else max_distance
    a = edlib.align(query, ref, mode=edlib_mode, task=task, k=k)

    ed = a.get("editDistance", -1)
    if ed < 0:
        raise RuntimeError(
            f"no alignment within max_distance={max_distance!r} (mode={mode!r})"
        )

    locations = a.get("locations") or []
    if not locations:
        # Shouldn't happen with editDistance >= 0, but guard explicitly.
        raise RuntimeError("edlib returned editDistance >= 0 with no locations")

    # Pick the first (start, end) location — edlib reports inclusive
    # endpoints; convert to half-open.
    start, end_inclusive = locations[0]

    cigar: str | None = None
    if return_cigar:
        cigar_raw = a.get("cigar")
        cigar = cigar_raw if isinstance(cigar_raw, str) else None

    return PairwiseResult(
        score=int(ed),
        query_start=0,
        query_end=len(query),
        ref_start=int(start),
        ref_end=int(end_inclusive) + 1,
        cigar=cigar,
    )


__all__ = [
    "PairwiseResult",
    "pairwise_align",
]
