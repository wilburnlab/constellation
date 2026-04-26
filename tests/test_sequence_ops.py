"""Tests for constellation.core.sequence.ops."""

from __future__ import annotations

import pytest

from constellation.core.sequence.alphabets import (
    AA,
    AA_IUPAC,
    DNA,
    DNA_IUPAC,
    RNA,
    RNA_IUPAC,
)
from constellation.core.sequence.ops import (
    format_modified_sequence,
    hamming_distance,
    identify_alphabet,
    kmerize,
    normalize,
    parse_modified_sequence,
    sliding_window,
    validate,
)


# ──────────────────────────────────────────────────────────────────────
# identify_alphabet
# ──────────────────────────────────────────────────────────────────────


def test_identify_dna():
    assert identify_alphabet("ACGTACGT") is DNA


def test_identify_rna():
    """U presence picks RNA over DNA."""
    assert identify_alphabet("ACGUACGU") is RNA


def test_identify_protein():
    assert identify_alphabet("PEPTIDEMASK") is AA


def test_identify_iupac_dna():
    """Adding N triggers DNA_IUPAC."""
    assert identify_alphabet("ACGTN") is DNA_IUPAC


def test_identify_iupac_rna():
    assert identify_alphabet("ACGUN") is RNA_IUPAC


def test_identify_iupac_aa():
    """X is an IUPAC AA token only."""
    assert identify_alphabet("PEPTIDEX") is AA_IUPAC


def test_identify_canonical_preferred_over_iupac():
    """A pure-canonical sequence shouldn't escalate to IUPAC."""
    seq = "ACGT"  # both DNA and DNA_IUPAC fit; canonical wins
    assert identify_alphabet(seq) is DNA


def test_identify_dna_preferred_over_aa_for_dna_letters():
    """GATTACA fits both DNA and AA — disambiguate to DNA by default
    (iteration order in `_DEFAULT_CANDIDATES`)."""
    assert identify_alphabet("GATTACA") is DNA


def test_identify_unknown_raises():
    with pytest.raises(ValueError):
        identify_alphabet("ZZZZ@@@")


def test_identify_empty_raises():
    with pytest.raises(ValueError):
        identify_alphabet("")


# ──────────────────────────────────────────────────────────────────────
# validate
# ──────────────────────────────────────────────────────────────────────


def test_validate_passes():
    validate("ACGT", DNA)
    validate("PEPTIDE", AA)


def test_validate_lists_offending_positions():
    with pytest.raises(ValueError, match=r"@\d") as exc:
        validate("PEPTIDEZ", AA)
    assert "Z" in str(exc.value)


def test_validate_empty_no_op():
    validate("", DNA)  # no exception


# ──────────────────────────────────────────────────────────────────────
# normalize
# ──────────────────────────────────────────────────────────────────────


def test_normalize_strips_whitespace_and_uppercases():
    assert normalize("  acgt acgt  ", DNA) == "ACGTACGT"


def test_normalize_dna_substitutes_u_to_t():
    """Pasting an RNA sequence with a DNA alphabet should fix U→T."""
    assert normalize("ACGU", DNA) == "ACGT"


def test_normalize_rna_substitutes_t_to_u():
    assert normalize("ACGT", RNA) == "ACGU"


def test_normalize_aa_leaves_alone():
    assert normalize("PEPTIDE  ", AA) == "PEPTIDE"


# ──────────────────────────────────────────────────────────────────────
# kmerize / sliding_window
# ──────────────────────────────────────────────────────────────────────


def test_kmerize_basic():
    assert kmerize("ABCDEF", 3) == ["ABC", "BCD", "CDE", "DEF"]


def test_kmerize_step():
    assert kmerize("ABCDEF", 3, step=2) == ["ABC", "CDE"]


def test_kmerize_too_short_returns_empty():
    assert kmerize("AB", 3) == []


def test_kmerize_k_equals_length():
    assert kmerize("ABC", 3) == ["ABC"]


def test_kmerize_k_zero_raises():
    with pytest.raises(ValueError):
        kmerize("ABC", 0)


def test_kmerize_negative_step_raises():
    with pytest.raises(ValueError):
        kmerize("ABCDEF", 3, step=0)


def test_sliding_window_is_iterator():
    it = sliding_window("ABCDEF", 3)
    assert next(it) == "ABC"
    assert next(it) == "BCD"


def test_sliding_window_empty_when_too_short():
    assert list(sliding_window("AB", 3)) == []


# ──────────────────────────────────────────────────────────────────────
# hamming_distance
# ──────────────────────────────────────────────────────────────────────


def test_hamming_zero_for_identical():
    assert hamming_distance("ACGT", "ACGT") == 0


def test_hamming_full_for_disjoint():
    assert hamming_distance("AAAA", "CCCC") == 4


def test_hamming_partial():
    assert hamming_distance("ACGT", "AAGT") == 1


def test_hamming_length_mismatch_raises():
    with pytest.raises(ValueError):
        hamming_distance("ACG", "ACGT")


def test_hamming_degenerate_counts_as_mismatch():
    """Without IUPAC-aware matching, N vs A is a mismatch."""
    assert hamming_distance("AANT", "AAAT") == 1


# ──────────────────────────────────────────────────────────────────────
# parse_modified_sequence / format_modified_sequence
# ──────────────────────────────────────────────────────────────────────


def test_parse_no_mods():
    seq, mods = parse_modified_sequence("PEPTIDE")
    assert seq == "PEPTIDE"
    assert mods == {}


def test_parse_unimod_mod():
    seq, mods = parse_modified_sequence("PEPC[UNIMOD:4]TIDE")
    assert seq == "PEPCTIDE"
    assert mods == {3: "UNIMOD:4"}


def test_parse_mass_notation():
    seq, mods = parse_modified_sequence("PEPC[+57.021]TIDE")
    assert seq == "PEPCTIDE"
    assert mods == {3: 57.021}


def test_parse_n_terminal_mod():
    seq, mods = parse_modified_sequence("[+42.011]MASTERPROTEIN")
    assert seq == "MASTERPROTEIN"
    assert mods == {0: 42.011}


def test_parse_multiple_mods():
    seq, mods = parse_modified_sequence("M[UNIMOD:35]AC[UNIMOD:4]E")
    assert seq == "MACE"
    assert mods == {0: "UNIMOD:35", 2: "UNIMOD:4"}


def test_parse_unterminated_bracket_raises():
    with pytest.raises(ValueError):
        parse_modified_sequence("PEP[UNIMOD:4")


def test_parse_format_round_trip_unimod():
    original = "PEPC[UNIMOD:4]TIDE"
    seq, mods = parse_modified_sequence(original)
    assert format_modified_sequence(seq, mods) == original


def test_format_no_mods_passthrough():
    assert format_modified_sequence("PEPTIDE", {}) == "PEPTIDE"


def test_format_mass_notation_signed():
    """Float values format with explicit sign for parser-disambiguation."""
    out = format_modified_sequence("PEPTIDE", {3: 15.99})
    assert out == "PEPT[+15.990000]IDE"


def test_parse_mass_notation_round_trip():
    seq, mods = parse_modified_sequence("PEPT[+15.990000]IDE")
    assert seq == "PEPTIDE"
    assert isinstance(mods[3], float)
    assert abs(mods[3] - 15.99) < 1e-9
