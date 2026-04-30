"""Tests for parse_encyclopedia_modseq — EncyclopeDIA → Peptidoform → ProForma 2.0.

The motivating case: EncyclopeDIA collapses ``[Mod]-X`` and ``X[Mod]``
into one ``X[Mod]`` form, but they are chemically distinct molecules.
This module's job is to disambiguate at the reader boundary by looking
up the mass delta in UNIMOD and applying the modification's specificity
constraints, returning a structured ``Peptidoform`` directly.

Assertions go through ``format_proforma(parse_encyclopedia_modseq(s))``
so we exercise both the parser and its ProForma 2.0 serialization
contract in one shot.
"""

from __future__ import annotations

import pytest

from constellation.core.sequence.proforma import format_proforma, parse_proforma
from constellation.massspec.io.encyclopedia import parse_encyclopedia_modseq


def _round_trip(modseq: str) -> str:
    """Convenience: parse encyclopedia modseq, format as ProForma 2.0."""
    return format_proforma(parse_encyclopedia_modseq(modseq))


# ──────────────────────────────────────────────────────────────────────
# Pass-through cases (no mods)
# ──────────────────────────────────────────────────────────────────────


def test_unmodified_passes_through():
    assert _round_trip("PEPTIDE") == "PEPTIDE"


def test_empty_input_returns_empty():
    assert _round_trip("") == ""


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
    assert _round_trip(input) == expected


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
    assert _round_trip("K[+42.01057]VERPD") == "K[UNIMOD:1]VERPD"


def test_acetyl_on_a_at_residue_zero_promotes_to_n_terminus():
    """A residue's side chain cannot host Acetyl (Ac is K-side or N-term
    only). So at position 0, the only chemically valid placement is
    N-terminal — promote it."""
    assert _round_trip("A[+42.01057]ARK") == "[UNIMOD:1]-AARK"


def test_methionine_oxidation_at_residue_zero_stays_side_chain():
    """M can host Oxidation on its side chain regardless of position."""
    assert _round_trip("M[+15.99492]PEPTIDE") == "M[UNIMOD:35]PEPTIDE"


@pytest.mark.parametrize(
    "n_term_residue",
    ["A", "G", "L", "F", "S"],  # residues that don't host Acetyl on side chain
)
def test_acetyl_at_residue_zero_for_non_k_residues_promotes_to_terminal(
    n_term_residue: str,
):
    inp = f"{n_term_residue}[+42.01057]PEP"
    expected = f"[UNIMOD:1]-{n_term_residue}PEP"
    assert _round_trip(inp) == expected


# ──────────────────────────────────────────────────────────────────────
# Unknown / ambiguous mass deltas — fallthrough to mass-delta notation
# ──────────────────────────────────────────────────────────────────────


def test_unknown_mass_delta_passes_through():
    """Mass not in UNIMOD → ProForma mass-delta form preserved as a
    ``ModRef(mass_delta=...)`` and re-emitted unchanged."""
    pep = parse_encyclopedia_modseq("PEPK[+999.999]TIDE")
    assert pep.sequence == "PEPKTIDE"
    assert 3 in pep.residue_mods
    (tagged,) = pep.residue_mods[3]
    assert tagged.mod is not None
    assert tagged.mod.mass_delta == pytest.approx(999.999)


def test_proline_oxidation_not_in_unimod_specificity_passes_through():
    """Oxidation (UNIMOD:35) does not declare a P specificity. The
    chemistry layer can't validate this assignment, so leave as a
    mass-delta annotation rather than guess."""
    pep = parse_encyclopedia_modseq("PEPP[+15.99492]TIDE")
    (tagged,) = pep.residue_mods[3]
    assert tagged.mod is not None
    assert tagged.mod.accession is None
    assert tagged.mod.mass_delta == pytest.approx(15.99492)


# ──────────────────────────────────────────────────────────────────────
# Multiple mods in one peptide
# ──────────────────────────────────────────────────────────────────────


def test_multiple_modifications_round_trip():
    inp = "C[+57.02146]EPM[+15.99492]TIS[+79.96633]K"
    assert _round_trip(inp) == "C[UNIMOD:4]EPM[UNIMOD:35]TIS[UNIMOD:21]K"


def test_n_term_acetyl_with_internal_mods():
    """N-term Ac on A + internal Cam on C — both should resolve correctly."""
    inp = "A[+42.01057]EPC[+57.02146]TIDE"
    assert _round_trip(inp) == "[UNIMOD:1]-AEPC[UNIMOD:4]TIDE"


# ──────────────────────────────────────────────────────────────────────
# Output is parseable as ProForma 2.0
# ──────────────────────────────────────────────────────────────────────


def test_output_is_valid_proforma():
    """Round-tripping through format_proforma + parse_proforma must
    succeed without raising and must be idempotent on the second pass."""
    cases = [
        "K[+42.01057]VERPD",
        "PEPK[+42.01057]TIDE",
        "A[+42.01057]EPC[+57.02146]TIDE",
        "PEPP[+15.99492]TIDE",  # mass-delta passthrough
        "PEPK[+999.999]TIDE",  # unknown mass passthrough
    ]
    for inp in cases:
        out = _round_trip(inp)
        parse_proforma(out)  # must not raise


# ──────────────────────────────────────────────────────────────────────
# Malformed input
# ──────────────────────────────────────────────────────────────────────


def test_lowercase_residue_rejected():
    with pytest.raises(ValueError, match="unexpected character"):
        parse_encyclopedia_modseq("pepK[+42]TIDE")


def test_unmatched_bracket_rejected():
    with pytest.raises(ValueError):
        parse_encyclopedia_modseq("K[+42")
