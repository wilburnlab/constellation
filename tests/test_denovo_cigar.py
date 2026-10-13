"""CIGAR helper tests for the de novo consensus kernel.

``parse_cigar`` is strict on purpose: the permissive version silently
returned ``[(90, "M")]`` for ``"10S90M"``, which would shift every projected
member by the clip length with no error anywhere.
"""

from __future__ import annotations

import pytest

from constellation.sequencing.transcriptome.cluster.denovo._cigar import (
    base_codes,
    cigar_stats,
    indel_runs,
    parse_cigar,
    parse_cigar_clipped,
)


# ── parse_cigar strictness ────────────────────────────────────────────


def test_parse_cigar_extended_alphabet():
    assert parse_cigar("9=1I3=") == [(9, "="), (1, "I"), (3, "=")]
    assert parse_cigar("5M2X1D") == [(5, "M"), (2, "X"), (1, "D")]
    assert parse_cigar("") == []


def test_parse_cigar_rejects_soft_clips_instead_of_dropping_them():
    # The whole point: the permissive regex returned [(90, "M")] here, so a
    # member projected 10 bases off its true start with nothing to notice.
    with pytest.raises(ValueError, match="10S90M"):
        parse_cigar("10S90M")
    with pytest.raises(ValueError, match="parse_cigar_clipped"):
        parse_cigar("10S90M5S")


def test_parse_cigar_rejects_intron_and_pad_ops():
    with pytest.raises(ValueError):
        parse_cigar("50M100N50M")
    with pytest.raises(ValueError):
        parse_cigar("10M5P10M")


def test_parse_cigar_rejects_trailing_garbage():
    with pytest.raises(ValueError):
        parse_cigar("10=junk")


# ── parse_cigar_clipped ───────────────────────────────────────────────


def test_parse_cigar_clipped_splits_terminal_clips():
    assert parse_cigar_clipped("10S90M") == (10, [(90, "M")], 0)
    assert parse_cigar_clipped("10S90M5S") == (10, [(90, "M")], 5)
    assert parse_cigar_clipped("90M") == (0, [(90, "M")], 0)
    # Hard + soft clips at the same end sum.
    assert parse_cigar_clipped("5H10S90M") == (15, [(90, "M")], 0)


def test_parse_cigar_clipped_rejects_interior_clip():
    with pytest.raises(ValueError, match="interior"):
        parse_cigar_clipped("10M5S10M")


# ── indel_runs ────────────────────────────────────────────────────────


def test_indel_runs_counts_events_not_bases():
    # One 2-base deletion is one event, not two.
    assert indel_runs("5=1I3=2D4=") == (1, 1)
    assert indel_runs("100=") == (0, 0)
    assert indel_runs("5=1D3=1D4=") == (0, 2)
    assert indel_runs("5=3I5=") == (1, 0)


def test_indel_runs_merges_adjacent_same_op():
    # A concatenated / transposed CIGAR can carry split runs; they are one
    # event, which is what the frameshift-pair rule's "exactly one indel run"
    # has to see.
    assert indel_runs("5=1I1I3=") == (1, 0)
    assert indel_runs("5=1D1D1D3=") == (0, 1)


# ── unchanged helpers still behave ────────────────────────────────────


def test_cigar_stats_folds_M_into_matches():
    assert cigar_stats("9=1X2I3D") == (9, 1, 2, 3)
    assert cigar_stats("10M") == (10, 0, 0, 0)


def test_base_codes_maps_ambiguous_to_four():
    assert list(base_codes("ACGT")) == [0, 1, 2, 3]
    assert list(base_codes("acgt")) == [0, 1, 2, 3]
    assert list(base_codes("ANRT")) == [0, 4, 4, 3]
