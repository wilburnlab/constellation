"""Tests for ``constellation.sequencing.align.cigar``.

CIGAR + cs:long parsers — the algorithmically delicate piece of
Phase 1's alignment-block extraction. Hand-crafted fixtures with
hand-computed expected values; the goal is to nail down the
block-break rule (N + long D, but never long I), the soft/hard-clip
exclusion, and the cs:long match/mismatch counting.
"""

from __future__ import annotations

import pytest

from constellation.sequencing.align.cigar import (
    Block,
    Junction,
    blocks_to_junctions,
    parse_cigar_blocks,
    parse_cs_long_blocks,
    query_start_from_cigar,
)


# ──────────────────────────────────────────────────────────────────────
# parse_cigar_blocks
# ──────────────────────────────────────────────────────────────────────


def test_cigar_simple_match() -> None:
    """A single 100M block at ref=1000 has no internal indels."""
    blocks = parse_cigar_blocks("100M", ref_start=1000)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.block_index == 0
    assert b.ref_start == 1000
    assert b.ref_end == 1100
    assert b.query_start == 0
    assert b.query_end == 100
    assert b.n_match is None and b.n_mismatch is None
    assert b.n_insert == 0 and b.n_delete == 0


def test_cigar_unmapped_returns_empty() -> None:
    """``"*"`` and empty CIGAR both yield no blocks."""
    assert parse_cigar_blocks("*", ref_start=0) == []
    assert parse_cigar_blocks("", ref_start=0) == []


def test_cigar_intron_breaks_block() -> None:
    """``100M500N200M`` → 2 blocks separated by a 500-bp intron."""
    blocks = parse_cigar_blocks("100M500N200M", ref_start=1000)
    assert len(blocks) == 2
    assert blocks[0].ref_start == 1000 and blocks[0].ref_end == 1100
    assert blocks[1].ref_start == 1600 and blocks[1].ref_end == 1800
    # Query coords don't advance through N
    assert blocks[0].query_end == 100
    assert blocks[1].query_start == 100
    assert blocks[1].query_end == 300


def test_cigar_long_deletion_breaks_block() -> None:
    """A D ≥ intron_min_bp ends the current block (assembly use-case)."""
    blocks = parse_cigar_blocks("100M50D200M", ref_start=0, intron_min_bp=25)
    assert len(blocks) == 2
    assert blocks[0].ref_end == 100
    assert blocks[1].ref_start == 150  # ref advances through the D


def test_cigar_short_deletion_stays_within_block() -> None:
    """A small D contributes to ``n_delete``, doesn't break the block."""
    blocks = parse_cigar_blocks("100M3D200M", ref_start=0, intron_min_bp=25)
    assert len(blocks) == 1
    assert blocks[0].ref_start == 0
    assert blocks[0].ref_end == 303
    assert blocks[0].n_delete == 3


def test_cigar_long_insertion_does_not_break_block() -> None:
    """Long I shifts query but never breaks a block — insertions add
    bases to the query within the same aligned region.
    """
    blocks = parse_cigar_blocks("100M500I200M", ref_start=0, intron_min_bp=25)
    assert len(blocks) == 1
    assert blocks[0].ref_start == 0 and blocks[0].ref_end == 300
    assert blocks[0].query_end == 800  # 100 + 500 + 200
    assert blocks[0].n_insert == 500


def test_cigar_soft_clip_at_start_advances_query() -> None:
    """Leading S advances ``query_start``; the block starts past it."""
    blocks = parse_cigar_blocks("50S100M30S", ref_start=1000)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.ref_start == 1000 and b.ref_end == 1100
    assert b.query_start == 50
    assert b.query_end == 150


def test_cigar_hard_clip_does_not_advance_query() -> None:
    """H bases aren't in SEQ — query_start stays at 0."""
    blocks = parse_cigar_blocks("50H100M", ref_start=0)
    assert len(blocks) == 1
    assert blocks[0].query_start == 0
    assert blocks[0].query_end == 100


def test_cigar_three_block_intron_pattern() -> None:
    """Realistic minimap2 splice pattern: 50S 100M 500N 200M2I50M 50N 100M 30S."""
    blocks = parse_cigar_blocks(
        "50S100M500N200M2I50M50N100M30S", ref_start=1000
    )
    assert len(blocks) == 3
    # Block 0: 100M
    assert blocks[0].ref_start == 1000 and blocks[0].ref_end == 1100
    assert blocks[0].query_start == 50 and blocks[0].query_end == 150
    # Block 1: 200M2I50M (across the I, ref advances 250, query 252)
    assert blocks[1].ref_start == 1600 and blocks[1].ref_end == 1850
    assert blocks[1].query_start == 150 and blocks[1].query_end == 402
    assert blocks[1].n_insert == 2
    # Block 2: 100M
    assert blocks[2].ref_start == 1900 and blocks[2].ref_end == 2000
    assert blocks[2].query_start == 402 and blocks[2].query_end == 502


def test_cigar_eq_and_x_count_as_match_blocks() -> None:
    """``=`` and ``X`` are first-class block-builders (consume both axes)."""
    blocks = parse_cigar_blocks("50=2X48=", ref_start=0)
    assert len(blocks) == 1
    assert blocks[0].ref_end == 100


def test_cigar_padding_is_skipped() -> None:
    """``P`` (padding) consumes neither axis."""
    blocks = parse_cigar_blocks("100M5P200M", ref_start=0)
    assert len(blocks) == 1
    assert blocks[0].ref_end == 300


def test_cigar_malformed_raises() -> None:
    with pytest.raises(ValueError):
        parse_cigar_blocks("100Z", ref_start=0)
    with pytest.raises(ValueError):
        parse_cigar_blocks("M100", ref_start=0)


# ──────────────────────────────────────────────────────────────────────
# parse_cs_long_blocks
# ──────────────────────────────────────────────────────────────────────


def test_cs_long_simple_match() -> None:
    """``=AAAA`` is a 4-bp match block."""
    blocks = parse_cs_long_blocks("=AAAA", ref_start=10)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.ref_start == 10 and b.ref_end == 14
    assert b.n_match == 4
    assert b.n_mismatch == 0
    assert b.n_insert == 0 and b.n_delete == 0


def test_cs_short_form_match() -> None:
    """``:N`` short-form match contributes to ``n_match``."""
    blocks = parse_cs_long_blocks(":100", ref_start=0)
    assert len(blocks) == 1
    assert blocks[0].n_match == 100
    assert blocks[0].ref_end == 100


def test_cs_substitution_counts() -> None:
    """``*at`` is one mismatched base."""
    blocks = parse_cs_long_blocks("=ACGT*ag=ACGT", ref_start=0)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.n_match == 8
    assert b.n_mismatch == 1
    assert b.ref_end == 9
    assert b.query_end == 9


def test_cs_insertion_shifts_query_only() -> None:
    """``+ggg`` advances query but not ref."""
    blocks = parse_cs_long_blocks("=AAAA+ggg=TTTT", ref_start=0)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.ref_end == 8       # 4 + 4
    assert b.query_end == 11    # 4 + 3 + 4
    assert b.n_insert == 3
    assert b.n_match == 8


def test_cs_deletion_shifts_ref_only() -> None:
    """``-aaa`` advances ref but not query."""
    blocks = parse_cs_long_blocks("=AAAA-ggg=TTTT", ref_start=0)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.ref_end == 11      # 4 + 3 + 4
    assert b.query_end == 8     # 4 + 4
    assert b.n_delete == 3
    assert b.n_match == 8


def test_cs_splice_breaks_block() -> None:
    """``~aa<N>bb`` ends the current block."""
    blocks = parse_cs_long_blocks("=AAAA~gt500ag=TTTT", ref_start=1000)
    assert len(blocks) == 2
    assert blocks[0].ref_start == 1000 and blocks[0].ref_end == 1004
    assert blocks[1].ref_start == 1504 and blocks[1].ref_end == 1508
    assert blocks[0].query_end == 4
    assert blocks[1].query_start == 4
    assert blocks[0].n_match == 4
    assert blocks[1].n_match == 4


def test_cs_realistic_three_block() -> None:
    """A spliced read with two introns, mismatches, and an indel."""
    cs = "=AAAACC*ag=TT~gt100ag=GGGG-aa=CCCC~gt50ag=TTTT"
    blocks = parse_cs_long_blocks(cs, ref_start=2000)
    assert len(blocks) == 3
    # Block 0: =AAAACC *ag =TT  → 8 matches via the lengths in =/= sections
    # plus 1 mismatch; 9 ref advanced, 9 query advanced.
    b0 = blocks[0]
    assert b0.ref_start == 2000 and b0.ref_end == 2009
    assert b0.n_match == 8
    assert b0.n_mismatch == 1
    # Block 1: =GGGG -aa =CCCC → 8 matches, 2 ref-only deletes
    b1 = blocks[1]
    assert b1.ref_start == 2109 and b1.ref_end == 2119
    assert b1.n_match == 8
    assert b1.n_delete == 2
    # Block 2: =TTTT
    b2 = blocks[2]
    assert b2.ref_start == 2169 and b2.ref_end == 2173
    assert b2.n_match == 4


def test_cs_empty_string_returns_empty() -> None:
    assert parse_cs_long_blocks("", ref_start=0) == []


def test_cs_malformed_raises() -> None:
    with pytest.raises(ValueError):
        parse_cs_long_blocks("@AAAA", ref_start=0)


# ──────────────────────────────────────────────────────────────────────
# blocks_to_junctions
# ──────────────────────────────────────────────────────────────────────


def test_junctions_from_two_blocks() -> None:
    blocks = [
        Block(0, 1000, 1100, 0, 100, None, None, 0, 0),
        Block(1, 1600, 1800, 100, 300, None, None, 0, 0),
    ]
    junctions = blocks_to_junctions(blocks)
    assert junctions == [Junction(donor_pos=1100, acceptor_pos=1600)]


def test_junctions_empty_for_single_block() -> None:
    blocks = [Block(0, 0, 100, 0, 100, None, None, 0, 0)]
    assert blocks_to_junctions(blocks) == []


def test_junctions_three_blocks_yields_two() -> None:
    blocks = [
        Block(0, 0, 100, 0, 100, None, None, 0, 0),
        Block(1, 600, 800, 100, 300, None, None, 0, 0),
        Block(2, 1300, 1400, 300, 400, None, None, 0, 0),
    ]
    junctions = blocks_to_junctions(blocks)
    assert junctions == [
        Junction(donor_pos=100, acceptor_pos=600),
        Junction(donor_pos=800, acceptor_pos=1300),
    ]


# ──────────────────────────────────────────────────────────────────────
# query_start_from_cigar
# ──────────────────────────────────────────────────────────────────────


def test_query_start_from_leading_soft_clip() -> None:
    assert query_start_from_cigar("50S100M") == 50


def test_query_start_with_leading_hard_then_soft() -> None:
    """H is invisible to SEQ, so it doesn't count, but a following S does."""
    assert query_start_from_cigar("10H50S100M") == 50


def test_query_start_with_only_hard_clip() -> None:
    """H alone shouldn't advance query_start."""
    assert query_start_from_cigar("10H100M") == 0


def test_query_start_no_clip() -> None:
    assert query_start_from_cigar("100M") == 0


def test_query_start_empty() -> None:
    assert query_start_from_cigar("") == 0
    assert query_start_from_cigar("*") == 0


# ──────────────────────────────────────────────────────────────────────
# Cross-parser equivalence
# ──────────────────────────────────────────────────────────────────────


def test_cigar_and_cs_agree_on_block_boundaries() -> None:
    """For a spliced read, CIGAR and cs:long must produce identical
    (ref_start, ref_end) per block — block boundaries are the
    invariant; only match/mismatch attribution differs.
    """
    cigar_blocks = parse_cigar_blocks("100M500N200M", ref_start=1000)
    cs_blocks = parse_cs_long_blocks(":100~gt500ag:200", ref_start=1000)
    assert len(cigar_blocks) == len(cs_blocks)
    for a, b in zip(cigar_blocks, cs_blocks):
        assert (a.ref_start, a.ref_end) == (b.ref_start, b.ref_end)
        assert (a.query_start, a.query_end) == (b.query_start, b.query_end)
