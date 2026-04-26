"""Tests for constellation.core.sequence.nucleic."""

from __future__ import annotations

import pytest
import torch

from constellation.core.sequence.alphabets import DNA, DNA_IUPAC, RNA, RNA_IUPAC
from constellation.core.sequence.nucleic import (
    CODON_TABLES,
    STANDARD,
    CodonTable,
    Orf,
    best_orf,
    complement,
    find_orfs,
    gc_content,
    reverse_complement,
    translate,
)


# ──────────────────────────────────────────────────────────────────────
# Reverse-complement / complement
# ──────────────────────────────────────────────────────────────────────


def test_revcomp_basic():
    assert reverse_complement("ATCG") == "CGAT"


def test_revcomp_palindrome():
    assert reverse_complement("AATT") == "AATT"


def test_revcomp_double_is_identity():
    seq = "ACGTNRYSWKMBDHV"
    assert reverse_complement(reverse_complement(seq)) == seq


def test_revcomp_iupac_pairs():
    """N→N (palindromic), R→Y, K→M, B→V, D→H."""
    assert reverse_complement("NRYSKMBVDH") == reverse_complement("NRYSKMBVDH")
    assert reverse_complement("N") == "N"
    assert reverse_complement("R") == "Y"
    assert reverse_complement("Y") == "R"


def test_revcomp_rejects_unknown():
    with pytest.raises(KeyError):
        reverse_complement("ACGTZ")


def test_revcomp_rna():
    assert reverse_complement("ACGU", alphabet=RNA_IUPAC) == "ACGU"
    assert reverse_complement("AAUU", alphabet=RNA_IUPAC) == "AAUU"  # palindrome


def test_complement_no_reversal():
    assert complement("ACGT") == "TGCA"


def test_complement_rejects_aa():
    """AA alphabet has no complement table."""
    from constellation.core.sequence.alphabets import AA

    with pytest.raises(ValueError):
        complement("PEPTIDE", alphabet=AA)


# ──────────────────────────────────────────────────────────────────────
# Codon tables
# ──────────────────────────────────────────────────────────────────────


def test_codon_tables_loaded():
    assert sorted(CODON_TABLES.keys()) == [1, 2, 3, 4, 5, 6, 11]


def test_standard_is_table_1():
    assert STANDARD is CODON_TABLES[1]
    assert STANDARD.transl_table == 1


def test_standard_codon_count():
    """Every codon (4³ = 64) covered."""
    assert len(STANDARD.forward) == 64


def test_standard_anchor_translations():
    assert STANDARD.translate_codon("ATG") == "M"
    assert STANDARD.translate_codon("TGG") == "W"
    assert STANDARD.translate_codon("TAA") == "*"
    assert STANDARD.translate_codon("TAG") == "*"
    assert STANDARD.translate_codon("TGA") == "*"


def test_translate_codon_handles_rna_input():
    """U→T normalization happens inside translate_codon."""
    assert STANDARD.translate_codon("AUG") == "M"


def test_vertebrate_mito_recodes():
    """Table 2: AGA/AGG → stop; TGA → W; ATA → M."""
    t2 = CODON_TABLES[2]
    assert t2.translate_codon("AGA") == "*"
    assert t2.translate_codon("AGG") == "*"
    assert t2.translate_codon("TGA") == "W"
    assert t2.translate_codon("ATA") == "M"


def test_ciliate_table_recodes():
    """Table 6: TAA/TAG → Q (stop reassignment)."""
    t6 = CODON_TABLES[6]
    assert t6.translate_codon("TAA") == "Q"
    assert t6.translate_codon("TAG") == "Q"
    assert t6.translate_codon("TGA") == "*"


# ──────────────────────────────────────────────────────────────────────
# translate
# ──────────────────────────────────────────────────────────────────────


def test_translate_simple():
    assert translate("ATGGCATCAGCATCAGCAGCATAA") == "MASASAA*"


def test_translate_to_stop():
    assert translate("ATGGCATCAGCATCAGCAGCATAA", to_stop=True) == "MASASAA"


def test_translate_frame():
    assert translate("AATGGCATCA", frame=1) == "MAS"
    assert translate("GAATGGCATC", frame=2) == "MA"


def test_translate_rejects_bad_frame():
    with pytest.raises(ValueError):
        translate("ACGT", frame=3)


# ── Degenerate-codon collapse ─────────────────────────────────────────


def test_translate_wobble_synonymy_ctn_to_l():
    """CTN spans CTT,CTC,CTA,CTG, all of which encode L."""
    assert translate("CTN") == "L"


def test_translate_wobble_aay_to_n():
    """AAY = AAC|AAT, both N."""
    assert translate("AAY") == "N"


def test_translate_aar_to_k():
    """AAR = AAA|AAG, both K — no ambiguity."""
    assert translate("AAR") == "K"


def test_translate_genuinely_ambiguous_to_x():
    """AAB (= AA[CGT]) spans K and N → no defined ambiguity code."""
    out = translate("AAB")
    assert out == "X"


def test_translate_b_for_n_or_d():
    """RAY = R(A|G) + A + Y(C|T) = AAC|AAT|GAC|GAT
    = N|N|D|D → {N,D} → B."""
    assert translate("RAY") == "B"


def test_translate_z_for_q_or_e():
    """SAR = S(C|G) + A + R(A|G) = CAA|CAG|GAA|GAG = Q|Q|E|E."""
    assert translate("SAR") == "Z"


def test_translate_mixed_stop_and_coding_to_x():
    """TAR = TAA|TAG: both stops → *. Already exercised by tests using TAA."""
    assert translate("TAR") == "*"


def test_translate_completely_unknown_codon_to_x():
    """Z is not a base — codon containing Z resolves to X."""
    assert translate("AZZ") == "X"


# ── Trailing partial codon ────────────────────────────────────────────


def test_translate_partial_discard_default():
    """10 nt = 3 full codons + 1 leftover; leftover dropped."""
    assert translate("ATGGCATCAA") == "MAS"


def test_translate_partial_pad():
    """`pad` extends with N(s) and translates to a 4-codon protein."""
    out = translate("ATGGCATCAA", partial="pad")
    assert len(out) == 4


def test_translate_partial_raise():
    with pytest.raises(ValueError, match="not divisible by 3"):
        translate("ATGGCATCAA", partial="raise")


def test_translate_alt_start_with_flag():
    """Standard table has TTG as a start; with the flag enabled it
    becomes M instead of L."""
    plain = translate("TTGAAATAA")
    starty = translate("TTGAAATAA", treat_first_codon_as_start=True)
    assert plain == "LK*"
    assert starty == "MK*"


def test_translate_alt_start_only_at_first_codon():
    """Internal start-codon hits should NOT become M."""
    out = translate("ATGCTGAAATAA", treat_first_codon_as_start=True)
    # ATG=M, CTG=L (internal CTG should not become M), AAA=K, TAA=*
    assert out == "MLK*"


# ── Pluggable codon table ─────────────────────────────────────────────


def test_translate_with_mito_table_recodes_aga():
    """Vert mito table reassigns AGA → stop."""
    out = translate("ATGAGAATG", codon_table=CODON_TABLES[2])
    assert out == "M*M"


def test_translate_with_ciliate_table_recodes_taa():
    out = translate("ATGTAATGA", codon_table=CODON_TABLES[6])
    assert out == "MQ*"


def test_codon_table_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        STANDARD.transl_table = 99  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# find_orfs / best_orf
# ──────────────────────────────────────────────────────────────────────


def test_find_orf_basic():
    """Single forward ORF."""
    orfs = find_orfs("AAAATGGCATCAGCATCAGCAGCATAATTT", min_aa_length=2)
    pluses = [o for o in orfs if o.strand == "+"]
    assert pluses
    assert pluses[0].protein == "MASASAA"
    assert pluses[0].length == 7


def test_find_orf_both_strands():
    """ATG... TAA on the forward strand, plus a complementary ORF on -."""
    seq = "ATGGCATCAGCATCAGCAGCATAA"
    orfs = find_orfs(seq, min_aa_length=2, both_strands=True)
    strands = {o.strand for o in orfs}
    assert "+" in strands


def test_find_orf_min_length_filter():
    """A 2-residue ORF is rejected at min_aa_length=10."""
    orfs = find_orfs("ATGGCATAA", min_aa_length=10)
    assert orfs == []


def test_find_orf_longest_per_stop():
    """Two alternative starts feeding the same stop should collapse to
    the longest ORF when longest_per_stop=True (default)."""
    # ATG ... ATG ... TAA: same stop but two starts
    seq = "ATGAAAATGAAATAA"  # outer ORF: MKMK*; inner: MK*
    orfs = find_orfs(seq, min_aa_length=2, both_strands=False)
    pluses = [o for o in orfs if o.strand == "+"]
    # Longest-per-stop: keep only the outer ORF
    longest = max(pluses, key=lambda o: o.length)
    assert longest.protein == "MKMK"


def test_find_orf_orf_dataclass_fields():
    orfs = find_orfs("ATGGCATAA", min_aa_length=1)
    assert orfs
    o = orfs[0]
    assert isinstance(o, Orf)
    assert o.start == 0
    assert o.end == 9
    assert o.length == 2
    assert o.protein == "MA"
    assert o.transl_table == 1


def test_best_orf_picks_longest():
    seq = "ATGAAATAA" + "CCC" + "ATGAAAGGGAAAGGGTAA"  # short then long
    b = best_orf(seq, min_aa_length=1, both_strands=False)
    assert b is not None
    assert b.length >= 5  # the longer ORF


def test_best_orf_returns_none_when_no_orf():
    assert best_orf("CCCCCC", min_aa_length=10) is None


# ──────────────────────────────────────────────────────────────────────
# gc_content
# ──────────────────────────────────────────────────────────────────────


def test_gc_content_canonical():
    assert gc_content("ACGT") == 0.5
    assert gc_content("GCGC") == 1.0
    assert gc_content("ATAT") == 0.0


def test_gc_content_iupac_fractional():
    """N → 0.5; S → 1.0; B → 2/3; D → 1/3."""
    assert gc_content("N") == pytest.approx(0.5)
    assert gc_content("S") == pytest.approx(1.0)
    assert gc_content("B") == pytest.approx(2 / 3)
    assert gc_content("D") == pytest.approx(1 / 3)


def test_gc_content_empty():
    assert gc_content("") == 0.0


def test_gc_content_window_returns_tensor():
    out = gc_content("ATCGATCGATCG", window=4, step=2)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    # 12 bases, window=4, step=2 → starts at 0,2,4,6,8 → 5 windows
    assert out.numel() == 5


def test_gc_content_window_too_large_returns_empty_tensor():
    out = gc_content("ACG", window=10)
    assert isinstance(out, torch.Tensor)
    assert out.numel() == 0


def test_gc_content_rejects_unknown():
    with pytest.raises(KeyError):
        gc_content("ACGTZ")


# Silence unused-import warnings — DNA / RNA referenced in module-level
# comments but not directly in tests.
_ = (DNA, DNA_IUPAC, RNA, CodonTable)
