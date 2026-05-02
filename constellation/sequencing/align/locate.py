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
nanopore reads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocateMatch:
    """One occurrence of an error-tolerant substring search.

    ``start`` / ``end`` are 0-based half-open offsets into the reference
    string. ``edit_distance`` is the edit distance of this occurrence
    against the query (0 = exact match).
    """

    start: int
    end: int
    edit_distance: int


def _edlib_align(query: str, ref: str, *, k: int):
    """Internal wrapper around ``edlib.align`` that defers the import.

    edlib lives in the ``[sequencing]`` extra; this lets the module
    import even when edlib isn't installed (the call site fails clearly
    when actually used).
    """
    import edlib  # type: ignore[import-untyped]

    return edlib.align(query, ref, mode="HW", task="locations", k=k)


def locate_substring(
    query: str,
    ref: str,
    *,
    max_distance: int = 0,
    return_all: bool = False,
) -> LocateMatch | list[LocateMatch] | None:
    """Find ``query`` in ``ref`` with up to ``max_distance`` edits.

    Returns the first match (``return_all=False``, default) or all
    matches (``return_all=True``). ``None`` (or ``[]``) if no match
    fits within the edit-distance budget.

    Edlib's HW mode reports each match's ``(start, end)`` as the
    inclusive edit-distance-minimizing endpoints in ``ref``; we convert
    to half-open by adding 1 to ``end``. ``edit_distance`` is the
    overall match's distance; edlib does not break it down per-location
    but the value applies to all returned locations equally (they all
    minimize the same alignment cost).
    """
    if not query:
        raise ValueError("query is empty")
    if max_distance < 0:
        raise ValueError(f"max_distance must be ≥ 0, got {max_distance}")
    if not ref:
        return [] if return_all else None

    a = _edlib_align(query, ref, k=max_distance)
    locations = a.get("locations") or []
    if not locations or a.get("editDistance", -1) < 0:
        return [] if return_all else None

    ed = int(a["editDistance"])
    matches = [
        LocateMatch(start=int(s), end=int(e) + 1, edit_distance=ed)
        for s, e in locations
    ]
    if return_all:
        return matches
    # Sort by start to make "first match" deterministic across edlib
    # versions; edlib sometimes returns locations in non-sorted order.
    matches.sort(key=lambda m: (m.start, m.end))
    return matches[0]


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

    Pending — not on the S1 transcriptomics-port path. The polyA
    anchor uses ``locate_substring`` instead, with edlib's edit-
    distance tolerance subsuming the ``max_gap`` use case.
    """
    raise NotImplementedError(
        "locate_kmer_runs is not yet implemented; the S1 polyA anchor uses "
        "locate_substring with max_distance >= 1 instead."
    )


__all__ = [
    "LocateMatch",
    "locate_substring",
    "locate_kmer_runs",
]
