"""Error-tolerant substring location — anchors barcode demux.

The two-step demux algorithm needs:

    locate_substring(query, ref, max_distance) -> first match
        Find the first edit-distance-≤k occurrence of `query` in
        `ref`. The lab's polyA anchor uses this to find the polyA
        tail (`'AAAA'` with edge_distance=1).

    locate_kmer_runs(seq, kmer, min_count) -> contiguous runs
        Find runs of consecutive ``kmer`` occurrences exceeding
        ``min_count`` (telomeric repeat detection, polyA tail length
        estimation).

Built on edlib's HW (infix) mode at low edit distance — fast even on
nanopore reads. Status: STUB. Pending Phase 4 / 5.
"""

from __future__ import annotations

from dataclasses import dataclass


_PHASE = "Phase 4-5 (align/locate; demux uses this)"


@dataclass(frozen=True)
class LocateMatch:
    """One occurrence of an error-tolerant substring search."""

    start: int                  # 0-based half-open
    end: int
    edit_distance: int


def locate_substring(
    query: str,
    ref: str,
    *,
    max_distance: int = 0,
    return_all: bool = False,
) -> LocateMatch | list[LocateMatch] | None:
    """Find ``query`` in ``ref`` with up to ``max_distance`` edits.

    Returns the first match (``return_all=False``, default) or all
    matches (``return_all=True``). ``None`` if no match fits within
    the distance budget.
    """
    raise NotImplementedError(f"locate_substring pending {_PHASE}")


def locate_kmer_runs(
    seq: str,
    kmer: str,
    *,
    min_count: int = 4,
    max_gap: int = 0,
) -> list[tuple[int, int, int]]:
    """Find contiguous runs of ``kmer`` in ``seq``.

    Returns ``(start, end, count)`` tuples — start/end are 0-based
    half-open offsets bracketing the run; count is the number of
    consecutive ``kmer`` occurrences. ``max_gap`` allows occasional
    single-base deviations between consecutive copies (homopolymer
    error tolerance).
    """
    raise NotImplementedError(f"locate_kmer_runs pending {_PHASE}")


__all__ = [
    "LocateMatch",
    "locate_substring",
    "locate_kmer_runs",
]
