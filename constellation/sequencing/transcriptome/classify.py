"""Per-read classification — ``ReadStatus`` enum + ``classify_read``.

NanoporeAnalysis's ``annotate_read`` produces six base statuses
(``Complete``, ``3' Only``, ``5' Only``, ``Missing Barcode``,
``Unknown``, ``Complex``) with an optional ``" Fragment"`` suffix when
the resolved transcript is shorter than ``min_transcript_len``. The
Constellation port keeps the categorical statuses byte-for-byte for
parity but lifts the fragment suffix into a separate boolean column —
schema-cleaner, and the parity test maps `cDNA status` strings onto
``(status, is_fragment)`` tuples explicitly.

S2 will populate ``Truncated`` (transcript ends without 3' adapter
despite polyA), and S3 will populate ``Palindromic`` and
``TerminalDropout`` as artifact detection lands. Those values are
declared on the enum today but are never produced by the S1 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from constellation.sequencing.transcriptome.scoring import (
    AdapterVerdict,
    BarcodeVerdict,
)


class ReadStatus(StrEnum):
    """Classification of a per-read demux outcome.

    The string values match NanoporeAnalysis's ``cDNA status`` column
    verbatim (including the leading apostrophe + space convention) so
    parity tests can compare strings directly.
    """

    # Base classifications (S1)
    COMPLETE = "Complete"
    THREE_PRIME_ONLY = "3' Only"
    FIVE_PRIME_ONLY = "5' Only"
    MISSING_BARCODE = "Missing Barcode"
    UNKNOWN = "Unknown"
    COMPLEX = "Complex"

    # Extended classifications (populated by S2 / S3, declared on enum
    # for stable schema)
    TRUNCATED = "Truncated"
    PALINDROMIC = "Palindromic"
    TERMINAL_DROPOUT = "TerminalDropout"


@dataclass(frozen=True)
class Classification:
    """Result of classifying a read against the construct's expected
    structural slots.

    ``status`` is the categorical label (one of ``ReadStatus`` values);
    ``is_fragment`` is True when the resolved transcript window is
    shorter than ``min_transcript_len`` (NanoporeAnalysis's
    ``" Fragment"`` suffix lifted to a boolean).
    """

    status: ReadStatus
    is_fragment: bool


def classify_read(
    *,
    ssp: AdapterVerdict,
    primer3: AdapterVerdict,
    barcode: BarcodeVerdict,
    transcript_length: int | None,
    min_transcript_length: int,
) -> Classification:
    """Compute ``(status, is_fragment)`` for a single read's verdicts.

    Mirrors NanoporeAnalysis's ``annotate_read`` decision tree:

        all three found     →  Complete
        primer3 + barcode   →  3' Only        (ssp absent)
        ssp + primer3       →  Missing Barcode (barcode absent)
        ssp only            →  5' Only         (primer3 + barcode absent)
        all three absent    →  Unknown
        anything else       →  Unknown        (degenerate; rare)

    ``Complex`` is a strand-resolution outcome, not a per-orientation
    classification; the orchestrator (``demux._annotate_strand``)
    upgrades the status from this function's output to ``Complex``
    when forward and reverse orientations both produce non-Unknown
    classifications.

    The fragment flag is derived from ``transcript_length`` — set to
    True iff a transcript_length is known (not None) and falls below
    ``min_transcript_length``. NanoporeAnalysis appends " Fragment" to
    the status string in this case; we keep them as separate fields.
    """
    s_found = ssp.found
    p_found = primer3.found
    b_found = barcode.found

    if s_found and p_found and b_found:
        status = ReadStatus.COMPLETE
    elif p_found and b_found and not s_found:
        status = ReadStatus.THREE_PRIME_ONLY
    elif s_found and p_found and not b_found:
        status = ReadStatus.MISSING_BARCODE
    elif s_found and not p_found and not b_found:
        status = ReadStatus.FIVE_PRIME_ONLY
    elif not s_found and not p_found and not b_found:
        status = ReadStatus.UNKNOWN
    else:
        # Other combinations (e.g. barcode without primer3) shouldn't
        # be reachable in practice — barcode requires primer3 location
        # to define putative_umi — but fall through to Unknown to mirror
        # NanoporeAnalysis's else-branch behavior.
        status = ReadStatus.UNKNOWN

    is_fragment = (
        transcript_length is not None and transcript_length < min_transcript_length
    )
    return Classification(status=status, is_fragment=is_fragment)


def status_string(classification: Classification) -> str:
    """Render a ``Classification`` into NanoporeAnalysis's
    ``cDNA status`` column format ('Complete Fragment', "3' Only", etc.)
    — useful for parity testing and human-readable output."""
    s = classification.status.value
    if classification.is_fragment:
        return f"{s} Fragment"
    return s


__all__ = [
    "Classification",
    "ReadStatus",
    "classify_read",
    "status_string",
]
