"""Tests for ``constellation.sequencing.align.consensus``.

Validate the per-position weighted-PWM consensus kernel:

    1. Identical members reproduce the reference window exactly.
    2. Substitution wins under sufficient weight; loses under
       insufficient weight.
    3. Deletion-dominant positions drop out of the emitted consensus
       (the output gets shorter).
    4. Splice operators advance ref_pos without contributing votes.
    5. Insertions are skipped (no projection onto the genome frame).
    6. Members aligning outside the requested window are clipped.
    7. Missing alignment_cs raises a clear error.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.align.consensus import build_consensus
from constellation.sequencing.schemas.alignment import ALIGNMENT_CS_TABLE


def _cs(rows: list[tuple[int, str]]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"alignment_id": int(a), "cs_string": str(s)} for a, s in rows],
        schema=ALIGNMENT_CS_TABLE,
    )


def test_identical_members_reproduce_reference() -> None:
    ref = "AAAACCCCGGGG"
    cs_table = _cs([(1, ":12"), (2, ":12"), (3, ":12")])
    out = build_consensus(
        member_alignment_ids=[1, 2, 3],
        member_weights=[1.0, 1.0, 1.0],
        member_ref_starts=[100, 100, 100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_weighted_substitution_loses_under_low_weight() -> None:
    ref = "AAAACCCCGGGG"
    # Member 2 substitutes C→T at ref position 4 (offset 4 in the
    # window). Weight 1 vs 3 ref-matchers: ref wins.
    cs_table = _cs([(1, ":12"), (2, ":4*ct:7")])
    out = build_consensus(
        member_alignment_ids=[1, 2],
        member_weights=[3.0, 1.0],
        member_ref_starts=[100, 100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_weighted_substitution_wins_under_high_weight() -> None:
    ref = "AAAACCCCGGGG"
    cs_table = _cs([(1, ":12"), (2, ":4*ct:7")])
    out = build_consensus(
        member_alignment_ids=[1, 2],
        member_weights=[1.0, 5.0],
        member_ref_starts=[100, 100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    # Position 4 flips C → T; rest match reference.
    assert out == "AAAATCCCGGGG"


def test_deletion_drops_positions_from_consensus() -> None:
    ref = "AAAACCCCGGGG"
    # Single-member deletion of CCCC at positions 4-7 → consensus
    # length shrinks by 4.
    cs_table = _cs([(1, ":4-cccc:4")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == "AAAAGGGG"


def test_splice_advances_ref_pos_without_voting() -> None:
    """A `~ag10gt` operator skips 10 ref positions; positions 4-13 of a
    24-bp window get no votes from this member, so they fall back to
    the reference base when no other member covers them."""
    ref = "AAAA" + "TTTTTTTTTT" + "GGGGGGGGGG"  # 24 bp
    # Member spans 0..4 + splice 10 + 4..14 (last 10 of ref)
    # cs:long: ":4~ag10gt:10"
    cs_table = _cs([(1, ":4~ag10gt:10")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    # Spliced region falls back to ref because nobody voted there
    # (min_depth=1.0 default; no votes → fallback to ref base).
    assert out == ref


def test_insertion_is_skipped_no_projection() -> None:
    """`+xxx` insertion in query is not projected onto the genome
    frame — consensus length stays equal to the window length."""
    ref = "AAAACCCC"
    cs_table = _cs([(1, ":4+gg:4")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_member_outside_window_is_clipped() -> None:
    """Member alignment starting before the window has its early votes
    clipped; only the in-window positions accumulate."""
    ref = "CCCCCCCC"  # window starts at 100
    # Member starts at 96, all match — its votes for positions 96-99
    # are clipped, votes for 100-103 land at window offsets 0-3.
    cs_table = _cs([(1, ":12")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[96],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_empty_alignment_cs_raises() -> None:
    cs_table = ALIGNMENT_CS_TABLE.empty_table()
    with pytest.raises(ValueError, match="alignment_cs"):
        build_consensus(
            member_alignment_ids=[1],
            member_weights=[1.0],
            member_ref_starts=[0],
            alignment_cs=cs_table,
            reference_sequence="AAAA",
            reference_start=0,
        )


def test_member_array_length_mismatch_raises() -> None:
    cs_table = _cs([(1, ":4")])
    with pytest.raises(ValueError, match="length mismatch"):
        build_consensus(
            member_alignment_ids=[1, 2],
            member_weights=[1.0],
            member_ref_starts=[0, 0],
            alignment_cs=cs_table,
            reference_sequence="AAAA",
            reference_start=0,
        )


def test_no_members_returns_reference_unchanged() -> None:
    cs_table = _cs([(1, ":4")])
    out = build_consensus(
        member_alignment_ids=[],
        member_weights=[],
        member_ref_starts=[],
        alignment_cs=cs_table,
        reference_sequence="ACGT",
        reference_start=0,
    )
    assert out == "ACGT"


def test_long_match_form_equivalent_to_short() -> None:
    """`=AAAA` and `:4` should produce the same consensus contribution."""
    ref = "AAAACCCC"
    cs_short = _cs([(1, ":8")])
    cs_long = _cs([(2, "=AAAACCCC")])
    out_short = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[0],
        alignment_cs=cs_short,
        reference_sequence=ref,
        reference_start=0,
    )
    out_long = build_consensus(
        member_alignment_ids=[2],
        member_weights=[1.0],
        member_ref_starts=[0],
        alignment_cs=cs_long,
        reference_sequence=ref,
        reference_start=0,
    )
    assert out_short == out_long == ref
