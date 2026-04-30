"""Tests for format_encyclopedia_modseq — Peptidoform → EncyclopeDIA modseq.

The writer-side inverse of parse_encyclopedia_modseq. Lossy on the
chemical-fidelity axis: terminal mods collapse onto residue 0
(EncyclopeDIA's grammar can't distinguish [Mod]-X from X[Mod]).
"""

from __future__ import annotations

import pytest

from constellation.core.sequence.proforma import (
    ModRef,
    Peptidoform,
    TaggedMod,
    parse_proforma,
)
from constellation.massspec.io.encyclopedia import (
    format_encyclopedia_modseq,
    parse_encyclopedia_modseq,
)


# ──────────────────────────────────────────────────────────────────────
# Pass-through cases
# ──────────────────────────────────────────────────────────────────────


def test_unmodified_passes_through():
    assert format_encyclopedia_modseq(parse_proforma("PEPTIDE")) == "PEPTIDE"


def test_empty_input_returns_empty():
    assert format_encyclopedia_modseq(Peptidoform(sequence="")) == ""


# ──────────────────────────────────────────────────────────────────────
# Canonical UNIMOD lookups → mass-delta notation
# ──────────────────────────────────────────────────────────────────────


def test_carbamidomethyl_round_trip():
    out = format_encyclopedia_modseq(parse_proforma("PEPC[UNIMOD:4]TIDE"))
    # Carbamidomethyl: 57.0214637
    assert out.startswith("PEPC[+57.0214")
    assert out.endswith("]TIDE")


def test_phospho_round_trip():
    out = format_encyclopedia_modseq(parse_proforma("PEPS[UNIMOD:21]TIDE"))
    assert out.startswith("PEPS[+79.9663")
    assert out.endswith("]TIDE")


def test_oxidation_round_trip():
    out = format_encyclopedia_modseq(parse_proforma("PEPM[UNIMOD:35]TIDE"))
    assert out.startswith("PEPM[+15.9949")
    assert out.endswith("]TIDE")


# ──────────────────────────────────────────────────────────────────────
# Terminal collapse — the lossy axis
# ──────────────────────────────────────────────────────────────────────


def test_n_terminal_acetyl_collapses_to_residue_zero():
    """[UNIMOD:1]-AARK loses the 'this is on the N-terminus' fact and
    becomes A[+42.011]ARK in EncyclopeDIA's grammar. The downstream
    parse_encyclopedia_modseq will re-promote to N-terminus when it
    sees A[+42.011] (because A's side chain can't host Acetyl)."""
    pep = parse_proforma("[UNIMOD:1]-AARK")
    out = format_encyclopedia_modseq(pep)
    assert out.startswith("A[+42.010")
    assert out.endswith("]ARK")


def test_n_term_collapse_round_trips_through_parser():
    """Round-trip [UNIMOD:1]-AARK → format → parse → format gives the
    same EncyclopeDIA string (the parse re-promotes to N-term, then
    format collapses again)."""
    once = format_encyclopedia_modseq(parse_proforma("[UNIMOD:1]-AARK"))
    twice = format_encyclopedia_modseq(parse_encyclopedia_modseq(once))
    assert once == twice


# ──────────────────────────────────────────────────────────────────────
# Mass-delta passthrough
# ──────────────────────────────────────────────────────────────────────


def test_unknown_mass_delta_passes_through():
    """A ModRef carrying only mass_delta (no UNIMOD accession) renders
    its delta directly without a UNIMOD lookup."""
    pep = Peptidoform(
        sequence="PEPKTIDE",
        residue_mods={3: (TaggedMod(mod=ModRef(mass_delta=999.999)),)},
    )
    out = format_encyclopedia_modseq(pep)
    assert out.startswith("PEPK[+999.999")
    assert out.endswith("]TIDE")


def test_negative_mass_delta_renders_with_minus_sign():
    pep = Peptidoform(
        sequence="PEPK",
        residue_mods={3: (TaggedMod(mod=ModRef(mass_delta=-17.02655)),)},
    )
    out = format_encyclopedia_modseq(pep)
    assert out.startswith("PEPK[-17.026")


# ──────────────────────────────────────────────────────────────────────
# Round-trip with parse_encyclopedia_modseq
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "encyclopedia_input",
    [
        "PEPTIDE",
        "PEPC[+57.02146]TIDE",
        "PEPS[+79.96633]TIDE",
        "PEPM[+15.99492]TIDE",
        "PEPK[+42.01057]TIDE",  # K side-chain Ac
        "C[+57.02146]EPM[+15.99492]TIS[+79.96633]K",  # multi-mod
    ],
)
def test_parse_format_idempotent_for_unimod_inputs(encyclopedia_input: str):
    """parse → format → parse → format converges (idempotent on the
    second pass for canonical UNIMOD lookups)."""
    pep1 = parse_encyclopedia_modseq(encyclopedia_input)
    out1 = format_encyclopedia_modseq(pep1)
    pep2 = parse_encyclopedia_modseq(out1)
    out2 = format_encyclopedia_modseq(pep2)
    assert out1 == out2


# ──────────────────────────────────────────────────────────────────────
# Unsupported features raise NotImplementedError
# ──────────────────────────────────────────────────────────────────────


def test_c_terminal_mod_rejected():
    pep = parse_proforma("PEPTIDE-[UNIMOD:2]")
    with pytest.raises(NotImplementedError, match="C-terminal"):
        format_encyclopedia_modseq(pep)


def test_global_isotope_rejected():
    pep = parse_proforma("<13C>PEPTIDE")
    with pytest.raises(NotImplementedError, match="isotope"):
        format_encyclopedia_modseq(pep)


def test_formula_only_mod_rejected():
    pep = Peptidoform(
        sequence="PEPK",
        residue_mods={3: (TaggedMod(mod=ModRef(formula="H2O")),)},
    )
    with pytest.raises(NotImplementedError, match="[Ff]ormula"):
        format_encyclopedia_modseq(pep)
