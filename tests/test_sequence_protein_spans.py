"""Digestion with span kinds, context-aware modification placement and
``permute_peptidoform`` (2026-11-05). The default paths are held to the
frozen pre-change copies in ``tests/_cleave_oracle.py``."""

from __future__ import annotations

import random

import pytest

import _cleave_oracle as oracle
from constellation.core.sequence.proforma import format_proforma, parse_proforma, permute_peptidoform
from constellation.core.sequence.protein import (
    PROTEASES,
    SiteContext,
    cleave,
    enumerate_modforms,
    peptide_mass,
)

AAS = "ACDEFGHIKLMNPQRSTVWY"


def test_specific_digestion_is_unchanged() -> None:
    rng = random.Random(11)
    for _ in range(150):
        seq = ("M" if rng.random() < 0.5 else "") + "".join(rng.choice(AAS) for _ in range(rng.randint(1, 160)))
        protease = rng.choice(list(PROTEASES.ids()))
        met = rng.random() < 0.5
        kw = dict(
            missed_cleavages=rng.randint(0, 3),
            min_length=rng.randint(1, 8),
            # with Met excision the old code clipped only parents within max_length
            max_length=None if met else rng.choice([None, 20, 40]),
            excise_initiator_met=met,
        )
        assert cleave(seq, protease, **kw) == oracle.cleave(seq, protease, **kw)
        new = [(p.start, p.end) for p in cleave(seq, protease, return_spans=True, **kw)]
        old = sorted({(p.start, p.end) for p in oracle.cleave(seq, protease, return_spans=True, **kw)})
        assert new == old


def test_met_clipped_form_no_longer_needs_its_parent_within_max_length() -> None:
    seq = "MAAAAAK" + "GEPTIDEK"  # the N-terminal span is 7 long, clipped 6
    old = oracle.cleave(seq, "Trypsin", min_length=1, max_length=6, excise_initiator_met=True)
    new = cleave(seq, "Trypsin", min_length=1, max_length=6, excise_initiator_met=True)
    assert "AAAAAK" not in old and "AAAAAK" in new


def test_span_kinds_and_flanks() -> None:
    seq = "MKPEPTIDERAAAK"
    spans = {
        (p.start, p.end): p
        for p in cleave(seq, "Trypsin", missed_cleavages=1, min_length=1, excise_initiator_met=True, return_spans=True)
    }
    first = spans[(0, 10)]  # MKPEPTIDER: K before P is no cut
    assert (first.n_term, first.c_term, first.preceding, first.following) == ("protein", "enzyme", "-", "A")
    clipped = spans[(1, 10)]
    assert clipped.is_protein_n_terminal and clipped.preceding == "M"
    last = spans[(10, 14)]
    assert (last.n_term, last.c_term, last.following, last.n_missed) == ("enzyme", "protein", "-", 0)
    assert spans[(0, 14)].n_missed == 1


def test_semi_specific_floats_either_end() -> None:
    seq = "AAAAKCCCCK"
    spans = {(p.start, p.end): p for p in cleave(seq, "Trypsin", specificity="semi", min_length=3, max_length=None, return_spans=True)}
    assert spans[(1, 5)].n_term == "nonspecific" and spans[(1, 5)].c_term == "enzyme"  # N floats
    assert spans[(0, 3)].n_term == "protein" and spans[(0, 3)].c_term == "nonspecific"  # C floats
    assert (0, 5) in spans and (5, 10) in spans  # the specific spans are semi too
    assert all(p.n_term != "nonspecific" or p.c_term != "nonspecific" for p in spans.values())
    assert all(p.n_missed == 0 for p in spans.values())
    assert cleave(seq, "Trypsin", semi_specific=True, min_length=3, max_length=None) == cleave(
        seq, "Trypsin", specificity="semi", min_length=3, max_length=None
    )


def test_unspecific_is_every_span_in_bounds() -> None:
    seq = "".join(random.Random(5).choice(AAS) for _ in range(120))
    spans = cleave(seq, "Trypsin", specificity="unspecific", min_length=7, max_length=30, return_spans=True)
    assert len(spans) == sum(len(seq) - n + 1 for n in range(7, 31))
    assert {(p.start, p.end) for p in spans} == {(s, s + n) for n in range(7, 31) for s in range(len(seq) - n + 1)}
    with pytest.raises(ValueError):
        cleave(seq, specificity="unspecific", max_length=None)


def test_declared_mature_terminus() -> None:
    # A signal peptide ending at 5: the mature chain starts there.
    seq = "MAAAGDTHKSEIAK"
    spans = {(p.start, p.end): p for p in cleave(seq, "Trypsin", min_length=1, extra_n_termini=[5], return_spans=True)}
    assert spans[(5, 9)].n_term == "protein" and spans[(5, 9)].sequence == "DTHK"
    assert (0, 9) in spans  # the precursor's own span is kept
    with pytest.raises(ValueError):
        cleave(seq, extra_n_termini=[99])


# ── modification placement ───────────────────────────────────────────


def test_context_free_enumeration_is_unchanged() -> None:
    rng = random.Random(13)
    for _ in range(150):
        seq = "".join(rng.choice(AAS) for _ in range(rng.randint(1, 22)))
        kw = dict(
            fixed={"C": "UNIMOD:4"},
            variable={"M": "UNIMOD:35", "S": "UNIMOD:21", "T": "UNIMOD:21", "N-term": "UNIMOD:1"},
            max_variable=rng.randint(0, 3),
            validate_specificity=False,
        )
        assert [format_proforma(f) for f in enumerate_modforms(seq, **kw)] == [
            format_proforma(f) for f in oracle.enumerate_modforms(seq, **kw)
        ]


def test_protein_n_term_needs_the_protein_terminus() -> None:
    var = {"Protein N-term": "UNIMOD:1", "M": "UNIMOD:35"}
    at_terminus = enumerate_modforms("MKPEPTIDE", variable=var, context=SiteContext(protein_n_term=True))
    inside = enumerate_modforms("MKPEPTIDE", variable=var, context=SiteContext())
    assert "[UNIMOD:1]-MKPEPTIDE" in {format_proforma(f) for f in at_terminus}
    assert not any(f.n_term_mods for f in inside)
    with pytest.raises(ValueError, match="needs a context"):
        enumerate_modforms("MKPEPTIDE", variable=var)


def test_skip_unlicensed_applies_a_rule_only_where_unimod_places_it() -> None:
    var = {"N-term": "UNIMOD:28"}  # Gln->pyro-Glu: N-terminal Q only
    with pytest.raises(ValueError):
        enumerate_modforms("PEPTIDE", variable=var)
    assert [format_proforma(f) for f in enumerate_modforms("PEPTIDE", variable=var, skip_unlicensed=True)] == ["PEPTIDE"]
    assert len(enumerate_modforms("QPEPTIDE", variable=var, skip_unlicensed=True)) == 2


def test_hidden_placements_validate() -> None:
    # SILAC K (UNIMOD:259) has no shown placement; it used to be refused.
    with pytest.raises(ValueError):
        oracle.enumerate_modforms("PEPTIDEK", fixed={"K": "UNIMOD:259"})
    forms = enumerate_modforms("PEPTIDEK", fixed={"K": "UNIMOD:259"})
    assert [format_proforma(f) for f in forms] == ["PEPTIDEK[UNIMOD:259]"]


def test_max_per_mod_and_its_count() -> None:
    kw = dict(variable={"S": "UNIMOD:21", "T": "UNIMOD:21", "M": "UNIMOD:35"}, max_variable=3)
    forms = enumerate_modforms("SSTTMM", max_per_mod={"UNIMOD:21": 1}, **kw)
    assert all(format_proforma(f).count("UNIMOD:21") <= 1 for f in forms)
    every = enumerate_modforms("SSTTMM", **kw)
    assert len(forms) < len(every)
    assert len(enumerate_modforms("SSTTMM", max_per_mod={"UNIMOD:21": 1}, max_forms=len(forms), **kw)) == len(forms)
    with pytest.raises(ValueError, match="exceeding max_forms"):
        enumerate_modforms("SSTTMM", max_per_mod={"UNIMOD:21": 1}, max_forms=len(forms) - 1, **kw)


def test_return_origin() -> None:
    out = enumerate_modforms(
        "MCK",
        fixed={"C": "UNIMOD:4"},
        variable={"M": "UNIMOD:35", "Protein N-term": "UNIMOD:1"},
        context=SiteContext(protein_n_term=True),
        return_origin=True,
    )
    by_text = {format_proforma(f): placed for f, placed in out}
    placed = by_text["[UNIMOD:1]-M[UNIMOD:35]C[UNIMOD:4]K"]
    assert [(p.slot, p.key, p.origin, p.site) for p in placed] == [
        ("N-term", "UNIMOD:1", "variable", "Protein N-term"),
        (0, "UNIMOD:35", "variable", "M"),
        (1, "UNIMOD:4", "fixed", "C"),
    ]


# ── permutation ───────────────────────────────────────────────────────


def test_permute_keeps_composition_and_moves_mods_with_residues() -> None:
    pf = parse_proforma("<13C>[UNIMOD:1]-PEM[UNIMOD:35]TS[UNIMOD:21]IDEK/2")
    n = len(pf.sequence)
    order = [0, *range(n - 2, 0, -1), n - 1]  # EncyclopeDIA's reversal
    rev = permute_peptidoform(pf, order)
    assert format_proforma(rev) == "<13C>[UNIMOD:1]-PEDIS[UNIMOD:21]TM[UNIMOD:35]EK/2"
    assert peptide_mass(rev) == peptide_mass(pf)
    with pytest.raises(ValueError):
        permute_peptidoform(pf, [0, 0, 1])
    with pytest.raises(ValueError):
        permute_peptidoform(parse_proforma("PE(PT)[Phospho]IDE"), list(range(7)))
