"""Tests for normalize_encyclopedia_modseq — EncyclopeDIA → ProForma 2.0.

The motivating case: EncyclopeDIA collapses ``[Mod]-X`` and ``X[Mod]``
into one ``X[Mod]`` form, but they are chemically distinct molecules.
This module's job is to disambiguate at the reader boundary by looking
up the mass delta in UNIMOD and applying the modification's specificity
constraints.
"""

from __future__ import annotations

import pytest

from constellation.thirdparty.encyclopedia import normalize_encyclopedia_modseq


# ──────────────────────────────────────────────────────────────────────
# Pass-through cases (no mods)
# ──────────────────────────────────────────────────────────────────────


def test_unmodified_passes_through():
    assert normalize_encyclopedia_modseq("PEPTIDE") == "PEPTIDE"


def test_empty_input_returns_empty():
    assert normalize_encyclopedia_modseq("") == ""


# ──────────────────────────────────────────────────────────────────────
# Side-chain modifications (residue position, not N-terminus)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input,expected",
    [
        # Carbamidomethyl on C — universal Cam
        ("PEPC[+57.02146]TIDE", "PEPC[UNIMOD:4]TIDE"),
        # Phospho on S
        ("PEPS[+79.96633]TIDE", "PEPS[UNIMOD:21]TIDE"),
        # Phospho on T
        ("PEPT[+79.96633]IDE", "PEPT[UNIMOD:21]IDE"),
        # Phospho on Y
        ("PEPY[+79.96633]IDE", "PEPY[UNIMOD:21]IDE"),
        # Oxidation on M
        ("PEPM[+15.99492]TIDE", "PEPM[UNIMOD:35]TIDE"),
        # K side-chain Acetyl (mid-sequence)
        ("PEPK[+42.01057]TIDE", "PEPK[UNIMOD:1]TIDE"),
    ],
)
def test_side_chain_modifications(input: str, expected: str):
    assert normalize_encyclopedia_modseq(input) == expected


# ──────────────────────────────────────────────────────────────────────
# N-terminal disambiguation — the K[Acetyl]VERPD vs [Acetyl]-KVERPD case
# ──────────────────────────────────────────────────────────────────────


def test_acetyl_on_k_at_residue_zero_defaults_to_side_chain():
    """When the residue at position 0 supports both N-terminal and
    side-chain placement (e.g. K + Acetyl), default to the side-chain
    form. EncyclopeDIA's literal rendering is X[Mod], and we have no
    localization evidence to override that without explicit localization
    metadata. The downstream tooling can rewrite to the terminal form
    when localization scoring resolves it."""
    got = normalize_encyclopedia_modseq("K[+42.01057]VERPD")
    assert got == "K[UNIMOD:1]VERPD"


def test_acetyl_on_a_at_residue_zero_promotes_to_n_terminus():
    """A residue's side chain cannot host Acetyl (Ac is K-side or N-term
    only). So at position 0, the only chemically valid placement is
    N-terminal — promote it."""
    got = normalize_encyclopedia_modseq("A[+42.01057]ARK")
    assert got == "[UNIMOD:1]-AARK"


def test_methionine_oxidation_at_residue_zero_stays_side_chain():
    """M can host Oxidation on its side chain regardless of position."""
    got = normalize_encyclopedia_modseq("M[+15.99492]PEPTIDE")
    assert got == "M[UNIMOD:35]PEPTIDE"


@pytest.mark.parametrize(
    "n_term_residue",
    ["A", "G", "L", "F", "S"],  # residues that don't host Acetyl on side chain
)
def test_acetyl_at_residue_zero_for_non_k_residues_promotes_to_terminal(
    n_term_residue: str,
):
    inp = f"{n_term_residue}[+42.01057]PEP"
    expected = f"[UNIMOD:1]-{n_term_residue}PEP"
    assert normalize_encyclopedia_modseq(inp) == expected


# ──────────────────────────────────────────────────────────────────────
# Unknown / ambiguous mass deltas — fallthrough to mass-delta notation
# ──────────────────────────────────────────────────────────────────────


def test_unknown_mass_delta_passes_through():
    """Mass not in UNIMOD → ProForma mass-delta form preserved."""
    got = normalize_encyclopedia_modseq("PEPK[+999.999]TIDE")
    assert got == "PEPK[+999.999]TIDE"


def test_proline_oxidation_not_in_unimod_specificity_passes_through():
    """Oxidation (UNIMOD:35) does not declare a P specificity. The
    chemistry layer can't validate this assignment, so leave as a
    mass-delta annotation rather than guess."""
    got = normalize_encyclopedia_modseq("PEPP[+15.99492]TIDE")
    assert got == "PEPP[+15.99492]TIDE"


# ──────────────────────────────────────────────────────────────────────
# Multiple mods in one peptide
# ──────────────────────────────────────────────────────────────────────


def test_multiple_modifications_round_trip():
    inp = "C[+57.02146]EPM[+15.99492]TIS[+79.96633]K"
    got = normalize_encyclopedia_modseq(inp)
    assert got == "C[UNIMOD:4]EPM[UNIMOD:35]TIS[UNIMOD:21]K"


def test_n_term_acetyl_with_internal_mods():
    """N-term Ac on A + internal Cam on C — both should resolve correctly."""
    inp = "A[+42.01057]EPC[+57.02146]TIDE"
    got = normalize_encyclopedia_modseq(inp)
    assert got == "[UNIMOD:1]-AEPC[UNIMOD:4]TIDE"


# ──────────────────────────────────────────────────────────────────────
# Output is parseable as ProForma 2.0
# ──────────────────────────────────────────────────────────────────────


def test_output_is_valid_proforma():
    """The whole point of the normalizer is to emit ProForma 2.0 strings
    that the proforma parser accepts losslessly."""
    from constellation.core.sequence.proforma import parse_proforma

    cases = [
        "K[+42.01057]VERPD",
        "PEPK[+42.01057]TIDE",
        "A[+42.01057]EPC[+57.02146]TIDE",
        "PEPP[+15.99492]TIDE",  # mass-delta passthrough
        "PEPK[+999.999]TIDE",  # unknown mass passthrough
    ]
    for inp in cases:
        out = normalize_encyclopedia_modseq(inp)
        parsed = parse_proforma(out)  # must not raise
        assert parsed is not None


# ──────────────────────────────────────────────────────────────────────
# Malformed input
# ──────────────────────────────────────────────────────────────────────


def test_lowercase_residue_rejected():
    with pytest.raises(ValueError, match="unexpected character"):
        normalize_encyclopedia_modseq("pepK[+42]TIDE")


def test_unmatched_bracket_rejected():
    with pytest.raises(ValueError):
        normalize_encyclopedia_modseq("K[+42")
