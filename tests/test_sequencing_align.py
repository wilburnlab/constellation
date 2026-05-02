"""Tests for ``constellation.sequencing.align.{locate,pairwise}``.

Synthetic-string fixtures with hand-computed expected values; ground-
truth checks against direct edlib calls so any future drift in our
wrapper vs the underlying library is caught.
"""

from __future__ import annotations

import pytest

from constellation.sequencing.align.locate import LocateMatch, locate_substring
from constellation.sequencing.align.pairwise import PairwiseResult, pairwise_align


# ──────────────────────────────────────────────────────────────────────
# locate_substring
# ──────────────────────────────────────────────────────────────────────


def test_locate_exact_match() -> None:
    """'AAAA' in 'GTAAAAGT' starts at index 2 (half-open end 6)."""
    m = locate_substring("AAAA", "GTAAAAGT", max_distance=0)
    assert isinstance(m, LocateMatch)
    assert m.start == 2
    assert m.end == 6
    assert m.edit_distance == 0


def test_locate_with_one_mismatch() -> None:
    """'AAAA' should locate in 'GTAATAGT' at ed=1 with max_distance=1."""
    m = locate_substring("AAAA", "GTAATAGT", max_distance=1)
    assert isinstance(m, LocateMatch)
    assert m.edit_distance == 1


def test_locate_no_match_within_budget() -> None:
    """No 'AAAA'-like in 'GGGGGG' at any reasonable budget."""
    m = locate_substring("AAAA", "GGGGGG", max_distance=1)
    assert m is None


def test_locate_empty_ref() -> None:
    assert locate_substring("AAAA", "", max_distance=2) is None


def test_locate_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="query is empty"):
        locate_substring("", "ACGT", max_distance=0)


def test_locate_negative_distance_raises() -> None:
    with pytest.raises(ValueError, match="max_distance must be"):
        locate_substring("AAAA", "ACGT", max_distance=-1)


def test_locate_return_all_multiple_matches() -> None:
    """'AAA' occurs 3 times in 'AAAGAAATAAAG' — return_all should
    enumerate them. (edlib HW reports all minimal-edit-distance
    locations; with max_distance=0 those are exactly the exact-match
    starts.)"""
    matches = locate_substring("AAA", "AAAGAAATAAAG", max_distance=0, return_all=True)
    assert isinstance(matches, list)
    assert len(matches) == 3
    starts = sorted(m.start for m in matches)
    assert starts == [0, 4, 8]


def test_locate_return_all_empty_when_no_match() -> None:
    matches = locate_substring("AAAA", "GGGGGG", max_distance=1, return_all=True)
    assert matches == []


def test_locate_first_match_is_leftmost() -> None:
    """When multiple matches exist, the default (return_all=False) one
    is the leftmost — required for parity with NanoporeAnalysis's
    ``np.min([loc[0] for loc in primer_locations])`` 3'-primer find."""
    m = locate_substring("AAA", "GGAAACCAAAGG", max_distance=0)
    assert isinstance(m, LocateMatch)
    assert m.start == 2  # first occurrence wins


# ──────────────────────────────────────────────────────────────────────
# pairwise_align
# ──────────────────────────────────────────────────────────────────────


def test_pairwise_infix_exact() -> None:
    r = pairwise_align("CAT", "GGCATTT")
    assert isinstance(r, PairwiseResult)
    assert r.score == 0
    assert r.ref_start == 2
    assert r.ref_end == 5  # half-open
    assert r.query_start == 0
    assert r.query_end == 3


def test_pairwise_infix_with_mismatch() -> None:
    r = pairwise_align("CAT", "GGCGTTT", max_distance=1)
    assert r.score == 1


def test_pairwise_global_mode() -> None:
    """Global (NW) mode forces end-to-end alignment."""
    r = pairwise_align("CAT", "CAT", mode="global")
    assert r.score == 0
    r2 = pairwise_align("CAT", "CGT", mode="global")
    assert r2.score == 1


def test_pairwise_max_distance_overshoot_raises() -> None:
    with pytest.raises(RuntimeError, match="no alignment within"):
        pairwise_align("CAT", "GGGGGG", max_distance=1)


def test_pairwise_local_rejected_for_edlib() -> None:
    with pytest.raises(ValueError, match="parasail-only"):
        pairwise_align("CAT", "CGT", mode="local")


def test_pairwise_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        pairwise_align("CAT", "CAT", mode="bogus")  # type: ignore[arg-type]


def test_pairwise_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        pairwise_align("CAT", "CAT", backend="bogus")  # type: ignore[arg-type]


def test_pairwise_parasail_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="parasail backend"):
        pairwise_align("CAT", "CAT", backend="parasail")


def test_pairwise_with_cigar() -> None:
    """``return_cigar=True`` should populate the cigar field. Edlib
    uses extended CIGAR (``=`` / ``X`` for match / mismatch, ``I``/``D``
    for indels)."""
    r = pairwise_align("CAT", "GGCATTT", return_cigar=True)
    assert r.cigar is not None
    # 3 matched chars in a row → '3='
    assert "=" in r.cigar or "M" in r.cigar


# ──────────────────────────────────────────────────────────────────────
# Cross-check against direct edlib (catches wrapper drift)
# ──────────────────────────────────────────────────────────────────────


def test_locate_matches_direct_edlib_call() -> None:
    """Our locate_substring should agree with a direct edlib.align call
    on the (start, end_inclusive) → half-open conversion."""
    import edlib

    query = "AAGCAGTGGTATCAACGCAGAGTAC"  # 25-nt 5' SSP prefix
    ref = (
        "TTTGCGAAGCAGTGGTATCAACGCAGAGTACCACATATCAGAGTGCGT"
    )  # SSP variant + barcode + ...
    a = edlib.align(query, ref, mode="HW", task="locations", k=2)
    assert a["editDistance"] >= 0  # edlib found at least one match

    m = locate_substring(query, ref, max_distance=2)
    assert isinstance(m, LocateMatch)
    assert m.edit_distance == a["editDistance"]
    # Half-open conversion: edlib reports inclusive end; our wrapper
    # adds 1.
    expected_first_loc = sorted(a["locations"], key=lambda x: (x[0], x[1]))[0]
    assert m.start == expected_first_loc[0]
    assert m.end == expected_first_loc[1] + 1
