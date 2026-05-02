"""Tests for ``constellation.sequencing.transcriptome.classify``.

Table-driven enumeration of every (5p, 3p, BC) presence combination ×
fragment / non-fragment, asserting the resulting status string matches
NanoporeAnalysis's ``cDNA status`` column exactly.
"""

from __future__ import annotations

import pytest

from constellation.sequencing.transcriptome.classify import (
    Classification,
    ReadStatus,
    classify_read,
    status_string,
)
from constellation.sequencing.transcriptome.scoring import (
    AdapterVerdict,
    BarcodeVerdict,
)


def _ssp(found: bool) -> AdapterVerdict:
    return AdapterVerdict(found=found, edit_distance=0 if found else None,
                          boundary=10 if found else None)


def _primer3(found: bool) -> AdapterVerdict:
    return AdapterVerdict(found=found, edit_distance=0 if found else None,
                          boundary=5 if found else None)


def _barcode(found: bool) -> BarcodeVerdict:
    if found:
        return BarcodeVerdict(found=True, index=0, name="BC01", edit_distance=0)
    return BarcodeVerdict(found=False)


@pytest.mark.parametrize(
    "ssp_found, p3_found, bc_found, expected",
    [
        (True, True, True, ReadStatus.COMPLETE),
        (False, True, True, ReadStatus.THREE_PRIME_ONLY),
        (True, True, False, ReadStatus.MISSING_BARCODE),
        (True, False, False, ReadStatus.FIVE_PRIME_ONLY),
        (False, False, False, ReadStatus.UNKNOWN),
        # Degenerate combinations — fall through to UNKNOWN (mirrors
        # NanoporeAnalysis's else-branch behavior)
        (False, True, False, ReadStatus.UNKNOWN),
        (False, False, True, ReadStatus.UNKNOWN),
        (True, False, True, ReadStatus.UNKNOWN),
    ],
)
def test_status_truth_table(
    ssp_found: bool, p3_found: bool, bc_found: bool, expected: ReadStatus
) -> None:
    c = classify_read(
        ssp=_ssp(ssp_found),
        primer3=_primer3(p3_found),
        barcode=_barcode(bc_found),
        transcript_length=400,
        min_transcript_length=200,
    )
    assert c.status == expected
    assert c.is_fragment is False  # 400 >= 200


def test_fragment_flag_set_when_transcript_short() -> None:
    c = classify_read(
        ssp=_ssp(True), primer3=_primer3(True), barcode=_barcode(True),
        transcript_length=150, min_transcript_length=200,
    )
    assert c.status == ReadStatus.COMPLETE
    assert c.is_fragment is True


def test_fragment_flag_unset_when_transcript_unknown() -> None:
    """transcript_length=None (no SSP/no transcript window) should NOT
    set the fragment flag."""
    c = classify_read(
        ssp=_ssp(False), primer3=_primer3(False), barcode=_barcode(False),
        transcript_length=None, min_transcript_length=200,
    )
    assert c.status == ReadStatus.UNKNOWN
    assert c.is_fragment is False


def test_fragment_flag_set_for_unknown_with_short_seq() -> None:
    """Unknown reads can still be Fragment-flagged if their (pseudo-)
    transcript is shorter than the threshold."""
    c = classify_read(
        ssp=_ssp(False), primer3=_primer3(False), barcode=_barcode(False),
        transcript_length=100, min_transcript_length=200,
    )
    assert c.status == ReadStatus.UNKNOWN
    assert c.is_fragment is True


def test_status_string_concatenates_fragment_suffix() -> None:
    c = Classification(status=ReadStatus.COMPLETE, is_fragment=True)
    assert status_string(c) == "Complete Fragment"
    c2 = Classification(status=ReadStatus.COMPLETE, is_fragment=False)
    assert status_string(c2) == "Complete"


def test_status_string_for_three_prime_only() -> None:
    """The string must include the apostrophe-space format
    NanoporeAnalysis stores."""
    c = Classification(status=ReadStatus.THREE_PRIME_ONLY, is_fragment=False)
    assert status_string(c) == "3' Only"
    c2 = Classification(status=ReadStatus.THREE_PRIME_ONLY, is_fragment=True)
    assert status_string(c2) == "3' Only Fragment"


def test_status_enum_string_values() -> None:
    """ReadStatus values must match NanoporeAnalysis's column strings
    byte-for-byte (downstream parity test compares strings directly)."""
    assert ReadStatus.COMPLETE.value == "Complete"
    assert ReadStatus.THREE_PRIME_ONLY.value == "3' Only"
    assert ReadStatus.FIVE_PRIME_ONLY.value == "5' Only"
    assert ReadStatus.MISSING_BARCODE.value == "Missing Barcode"
    assert ReadStatus.UNKNOWN.value == "Unknown"
    assert ReadStatus.COMPLEX.value == "Complex"
