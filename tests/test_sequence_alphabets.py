"""Tests for constellation.core.sequence.alphabets."""

from __future__ import annotations

import pytest

from constellation.core.chem.composition import Composition
from constellation.core.sequence.alphabets import (
    AA,
    AA_IUPAC,
    ALPHABETS,
    COMPLEMENT_DNA,
    COMPLEMENT_DNA_IUPAC,
    COMPLEMENT_RNA,
    COMPLEMENT_RNA_IUPAC,
    DEGENERATE_AA,
    DEGENERATE_DNA,
    DEGENERATE_RNA,
    DNA,
    DNA_IUPAC,
    RNA,
    RNA_IUPAC,
    Alphabet,
    canonical_for,
    degenerate_ok,
    expand_token,
    expansion_table,
    requires_canonical,
)


# ──────────────────────────────────────────────────────────────────────
# Construction & invariants
# ──────────────────────────────────────────────────────────────────────


def test_canonical_dna_tokens():
    assert DNA.tokens == ("A", "C", "G", "T")
    assert DNA.degenerate is False
    assert DNA.compositions is None  # nucleic alphabets don't carry compositions


def test_canonical_rna_tokens():
    assert RNA.tokens == ("A", "C", "G", "U")
    assert RNA.degenerate is False


def test_canonical_aa_tokens_and_compositions():
    assert AA.tokens == tuple("ACDEFGHIKLMNPQRSTVWY")
    assert AA.degenerate is False
    assert AA.compositions is not None
    assert set(AA.compositions.keys()) == set(AA.tokens)
    for r, comp in AA.compositions.items():
        assert isinstance(comp, Composition)


def test_iupac_alphabets_are_degenerate():
    assert DNA_IUPAC.degenerate is True
    assert RNA_IUPAC.degenerate is True
    assert AA_IUPAC.degenerate is True


def test_iupac_alphabets_carry_no_compositions():
    assert DNA_IUPAC.compositions is None
    assert RNA_IUPAC.compositions is None
    assert AA_IUPAC.compositions is None


def test_iupac_alphabets_extend_canonical():
    for canonical, iupac in ((DNA, DNA_IUPAC), (RNA, RNA_IUPAC), (AA, AA_IUPAC)):
        assert set(canonical.tokens).issubset(iupac.tokens)


def test_degenerate_alphabet_with_compositions_raises():
    """Cannot construct a degenerate alphabet with compositions attached."""
    with pytest.raises(ValueError):
        Alphabet(
            name="bad",
            tokens=("A",),
            kind="dna",
            degenerate=True,
            compositions={"A": Composition.zeros()},
        )


def test_alphabets_registry_complete():
    assert set(ALPHABETS.keys()) == {"DNA", "RNA", "AA", "DNA_IUPAC", "RNA_IUPAC", "AA_IUPAC"}


# ──────────────────────────────────────────────────────────────────────
# Residue compositions — cross-check vs known masses
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "residue,expected_residue_mass",
    [
        ("A", 71.03711),  # Alanine residue (C3H5NO)
        ("G", 57.02146),  # Glycine
        ("L", 113.08406),
        ("M", 131.04049),  # Methionine (with sulfur)
        ("R", 156.10111),
        ("W", 186.07931),
    ],
)
def test_aa_residue_masses_match_published(residue: str, expected_residue_mass: float):
    """Residue (in-chain) masses should match standard published values
    to ~5 decimal places."""
    comp = AA.compositions[residue]
    assert abs(comp.mass - expected_residue_mass) < 1e-4


# ──────────────────────────────────────────────────────────────────────
# Containment & validation
# ──────────────────────────────────────────────────────────────────────


def test_dna_contains():
    assert DNA.contains("A")
    assert DNA.contains("-")  # gap by default
    assert not DNA.contains("U")
    assert not DNA.contains("N")  # IUPAC-only


def test_aa_contains_stop():
    assert AA.contains("*")
    assert not DNA.contains("*")  # nucleic alphabets have no stop


def test_iupac_contains_extra():
    assert DNA_IUPAC.contains("N")
    assert DNA_IUPAC.contains("R")
    assert AA_IUPAC.contains("X")
    assert AA_IUPAC.contains("U")  # selenocysteine


def test_validate_passes_clean_seq():
    assert DNA.validate("ACGT")
    assert AA.validate("PEPTIDE")


def test_validate_rejects_alien_char():
    assert not DNA.validate("ACGTZ")
    assert not AA.validate("PEPTIDEZ")


# ──────────────────────────────────────────────────────────────────────
# Degeneracy expansion
# ──────────────────────────────────────────────────────────────────────


def test_degenerate_dna_table_is_complete():
    """Each IUPAC code should have an expansion."""
    assert set(DEGENERATE_DNA.keys()) == set("RYSWKMBDHVN")


def test_degenerate_rna_substitutes_u_for_t():
    """RNA expansion table mirrors DNA but with U in place of T."""
    for code, dna_bases in DEGENERATE_DNA.items():
        rna_bases = DEGENERATE_RNA[code]
        assert rna_bases == tuple("U" if b == "T" else b for b in dna_bases)


def test_degenerate_n_expands_to_all_four():
    assert set(DEGENERATE_DNA["N"]) == {"A", "C", "G", "T"}


def test_degenerate_b_excludes_a():
    """B = not-A → C|G|T."""
    assert set(DEGENERATE_DNA["B"]) == {"C", "G", "T"}


def test_aa_ambiguity_codes():
    assert set(DEGENERATE_AA["B"]) == {"N", "D"}
    assert set(DEGENERATE_AA["Z"]) == {"Q", "E"}
    assert set(DEGENERATE_AA["J"]) == {"I", "L"}
    assert set(DEGENERATE_AA["X"]) == set(AA.tokens)


def test_expand_canonical_token_returns_self():
    assert expand_token("A", DNA_IUPAC) == ("A",)


def test_expand_degenerate_token():
    assert set(expand_token("N", DNA_IUPAC)) == {"A", "C", "G", "T"}
    assert set(expand_token("R", RNA_IUPAC)) == {"A", "G"}


def test_expand_unknown_raises():
    with pytest.raises(KeyError):
        expand_token("Q", DNA_IUPAC)


def test_expansion_table_by_kind():
    assert expansion_table(DNA_IUPAC) is DEGENERATE_DNA
    assert expansion_table(RNA_IUPAC) is DEGENERATE_RNA
    assert expansion_table(AA_IUPAC) is DEGENERATE_AA


def test_canonical_for_returns_self_when_canonical():
    assert canonical_for(DNA) is DNA
    assert canonical_for(AA) is AA


def test_canonical_for_strips_degeneracy():
    assert canonical_for(DNA_IUPAC) is DNA
    assert canonical_for(RNA_IUPAC) is RNA
    assert canonical_for(AA_IUPAC) is AA


# ──────────────────────────────────────────────────────────────────────
# Complement maps
# ──────────────────────────────────────────────────────────────────────


def test_complement_dna_canonical():
    assert COMPLEMENT_DNA == {"A": "T", "T": "A", "C": "G", "G": "C"}


def test_complement_rna_canonical():
    assert COMPLEMENT_RNA == {"A": "U", "U": "A", "C": "G", "G": "C"}


def test_complement_iupac_self_pairs():
    """S=C|G and W=A|T are self-complementary; N also pairs with itself."""
    for c in ("S", "W", "N"):
        assert COMPLEMENT_DNA_IUPAC[c] == c
        assert COMPLEMENT_RNA_IUPAC[c] == c


def test_complement_iupac_swaps():
    """R↔Y (purines ↔ pyrimidines); K↔M; B↔V; D↔H."""
    assert COMPLEMENT_DNA_IUPAC["R"] == "Y"
    assert COMPLEMENT_DNA_IUPAC["Y"] == "R"
    assert COMPLEMENT_DNA_IUPAC["K"] == "M"
    assert COMPLEMENT_DNA_IUPAC["B"] == "V"
    assert COMPLEMENT_DNA_IUPAC["D"] == "H"


# ──────────────────────────────────────────────────────────────────────
# Decorators
# ──────────────────────────────────────────────────────────────────────


def test_requires_canonical_marks_function():
    @requires_canonical
    def f(x: str) -> str:
        return x.upper()

    assert getattr(f, "_requires_canonical", False) is True


def test_requires_canonical_with_alphabet_kwarg_rejects_degenerate():
    @requires_canonical
    def needs_canonical(seq: str, *, alphabet: Alphabet = AA) -> str:
        return seq

    # Canonical alphabet → passes through
    assert needs_canonical("PEPTIDE", alphabet=AA) == "PEPTIDE"
    # Degenerate alphabet → raises
    with pytest.raises(ValueError, match="canonical"):
        needs_canonical("PEPTIDE", alphabet=AA_IUPAC)


def test_degenerate_ok_marks_function():
    @degenerate_ok
    def f(x: str) -> str:
        return x

    assert getattr(f, "_degenerate_ok", False) is True


# ──────────────────────────────────────────────────────────────────────
# Dataclass behavior
# ──────────────────────────────────────────────────────────────────────


def test_alphabet_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        DNA.name = "foo"  # type: ignore[misc]


def test_alphabet_repr_is_short():
    """Tokens can be long; repr should not dump them."""
    r = repr(AA)
    assert "AA" in r
    # Don't bake in too tight a contract — just sanity-check brevity.
    assert len(r) < 100
