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
    count_modforms,
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


def test_a_placement_names_every_site_that_reaches_it_whatever_their_order() -> None:
    """Until 2026-11-11 a placement two sites reached (N-term and Protein
    N-term at a protein's N-terminus) named only the first in the spec's
    order, so the provenance built on it followed the rules' order (review
    of 2026-11-10, B11). The forms do not change."""
    out = {}
    for fixed, variable in (
        ({"C": "UNIMOD:4", "H": "UNIMOD:4"}, {"N-term": "UNIMOD:1", "Protein N-term": "UNIMOD:1", "M": "UNIMOD:35"}),
        ({"H": "UNIMOD:4", "C": "UNIMOD:4"}, {"M": "UNIMOD:35", "Protein N-term": "UNIMOD:1", "N-term": "UNIMOD:1"}),
    ):
        forms = enumerate_modforms("MCHK", fixed=fixed, variable=variable, context=SiteContext(protein_n_term=True),
                                   return_origin=True, validate_specificity=False)  # fmt: skip
        out[tuple(variable)] = {format_proforma(f): {(p.slot, p.key, p.origin, p.sites) for p in placed} for f, placed in forms}
    first, second = out.values()
    assert first == second
    assert first["[UNIMOD:1]-MC[UNIMOD:4]H[UNIMOD:4]K"] == {
        ("N-term", "UNIMOD:1", "variable", ("N-term", "Protein N-term")),
        (1, "UNIMOD:4", "fixed", ("C",)),
        (2, "UNIMOD:4", "fixed", ("H",)),
    }
    # A fixed modification two sites reach: both named.
    ((_form, placed),) = enumerate_modforms("MCK", fixed={"N-term": "UNIMOD:1", "Protein N-term": "UNIMOD:1"},
                                            context=SiteContext(protein_n_term=True), return_origin=True)  # fmt: skip
    assert [p.sites for p in placed] == [("N-term", "Protein N-term")]


def _texts(forms) -> list[str]:
    return [format_proforma(f) for f in forms]


def test_an_isotopic_substitution_stacks_with_a_chemical_modification() -> None:
    # heavy lysine (UNIMOD:259) leaves the amine free for an acetyl
    fixed_label = enumerate_modforms("PEPKK", fixed={"K": "UNIMOD:259"}, variable={"K": "UNIMOD:1"}, max_variable=2)
    assert "PEPK[UNIMOD:1][UNIMOD:259]K[UNIMOD:1][UNIMOD:259]" in {canonical(f) for f in fixed_label}
    assert len(fixed_label) == 4
    both_variable = enumerate_modforms("PEPK", variable={"K": ["UNIMOD:259", "UNIMOD:1"]}, max_variable=2)
    assert {canonical(f) for f in both_variable} == {
        "PEPK", "PEPK[UNIMOD:259]", "PEPK[UNIMOD:1]", "PEPK[UNIMOD:1][UNIMOD:259]"
    }
    # a fixed label and a fixed chemical modification share the residue
    assert _texts(enumerate_modforms("PEPK", fixed={"K": ["UNIMOD:259", "UNIMOD:36"]})) == [
        "PEPK[UNIMOD:36][UNIMOD:259]"
    ]
    with pytest.raises(ValueError, match="fixed site"):
        enumerate_modforms("PEPK", fixed={"K": ["UNIMOD:259", "UNIMOD:188"]})


def canonical(form) -> str:
    from constellation.massspec.library.ids import canonical_peptide_key

    return canonical_peptide_key(form)


def test_a_side_chain_cyclization_replaces_the_residue_modification() -> None:
    # fixed CAM: the pyro-carbamidomethyl form carries no CAM on its C
    forms = _texts(enumerate_modforms("CPEK", fixed={"C": "UNIMOD:4"}, variable={"N-term": "UNIMOD:26"}))
    assert forms == ["C[UNIMOD:4]PEK", "[UNIMOD:26]-CPEK"]
    # a deamidated Q is not also Gln->pyro-Glu
    kw = dict(variable={"Q": "UNIMOD:7", "N-term": "UNIMOD:28"}, max_variable=3, skip_unlicensed=True)
    forms = _texts(enumerate_modforms("QPQK", **kw))
    assert "[UNIMOD:28]-QPQ[UNIMOD:7]K" in forms
    assert not any(f.startswith("[UNIMOD:28]-Q[UNIMOD:7]") for f in forms)
    assert count_modforms("QPQK", **kw) == len(forms) == 6  # 2**3, less the 2 with both on Q1


def test_two_fixed_modifications_on_one_slot_raise() -> None:
    ctx = SiteContext(protein_n_term=True)
    with pytest.raises(ValueError, match="two fixed modifications"):
        enumerate_modforms("APEK", fixed={"Protein N-term": "UNIMOD:1", "N-term": "UNIMOD:36"}, context=ctx,
                           validate_specificity=False)
    # inside the protein only one of them applies
    assert _texts(enumerate_modforms(
        "APEK", fixed={"Protein N-term": "UNIMOD:1", "N-term": "UNIMOD:36"}, context=SiteContext(),
        validate_specificity=False,
    )) == ["[UNIMOD:36]-APEK"]


def test_counts_are_what_enumeration_returns() -> None:
    """The DP count and the pruned walk agree on random specs mixing
    limits, labels, cyclizations and fixed slots; the order is by number
    of variable modifications, no form twice."""
    rng = random.Random(29)
    pool = {
        "M": ["UNIMOD:35"], "S": ["UNIMOD:21"], "T": ["UNIMOD:21"], "Q": ["UNIMOD:7"], "N": ["UNIMOD:7"],
        "K": ["UNIMOD:1", "UNIMOD:259"], "C": ["UNIMOD:4"], "N-term": ["UNIMOD:1", "UNIMOD:26", "UNIMOD:28"],
    }
    for _ in range(300):
        seq = rng.choice("CQEMSK") + "".join(rng.choice("ACDEKMNQST") for _ in range(rng.randint(1, 10)))
        variable = {site: keys for site, keys in pool.items() if rng.random() < 0.6}
        fixed = {"C": "UNIMOD:4"} if rng.random() < 0.5 else {}
        variable.pop("C", None) if fixed else None
        limits = {"UNIMOD:21": rng.randint(0, 2)} if rng.random() < 0.5 else None
        kw = dict(fixed=fixed, variable=variable, max_variable=rng.randint(0, 4), max_per_mod=limits,
                  skip_unlicensed=True)
        forms = enumerate_modforms(seq, **kw)
        assert len(forms) == count_modforms(seq, **kw)
        texts = [canonical(f) for f in forms]
        assert len(set(texts)) == len(texts)


def test_a_placement_counts_toward_every_rule_that_reaches_it() -> None:
    """N-term and Protein N-term are one slot at a protein's N-terminus; the
    limit was taken from the first rule that reached it, so the forms
    depended on the rules' order, and one order broke the second rule's
    limit (Codex, 2026-11-10: four forms of PEPTIDEK, three reversed)."""
    context = SiteContext(protein_n_term=True)
    limit = [("UNIMOD:1", ["Protein N-term", "K"], 1)]
    orders = (
        {"N-term": ["UNIMOD:1"], "Protein N-term": ["UNIMOD:1"], "K": ["UNIMOD:1"]},
        {"Protein N-term": ["UNIMOD:1"], "K": ["UNIMOD:1"], "N-term": ["UNIMOD:1"]},
    )
    for variable in orders:
        kw = dict(variable=variable, context=context, limit_groups=limit)
        forms = sorted(canonical(f) for f in enumerate_modforms("PEPTIDEK", **kw))
        assert forms == ["PEPTIDEK", "PEPTIDEK[UNIMOD:1]", "[UNIMOD:1]-PEPTIDEK"]
        assert count_modforms("PEPTIDEK", **kw) == 3


def test_a_variable_modification_replacing_a_fixed_cyclization_frees_residue_0() -> None:
    """A fixed cyclization took residue 0's fixed modification away in every
    form, also where a variable N-terminal one replaced the cyclization:
    [UNIMOD:1]-CPEPTIDEK without its carbamidomethyl (Codex, 2026-11-10).
    Residue 0 is the cyclization's only in the forms that keep it."""
    kw = dict(fixed={"N-term": "UNIMOD:26", "C": "UNIMOD:4"}, variable={"N-term": ["UNIMOD:1"]})
    forms = enumerate_modforms("CPEPTIDEK", **kw)
    assert [canonical(f) for f in forms] == ["[UNIMOD:26]-CPEPTIDEK", "[UNIMOD:1]-C[UNIMOD:4]PEPTIDEK"]
    assert peptide_mass(forms[1]) == pytest.approx(peptide_mass(parse_proforma("[Acetyl]-C[Carbamidomethyl]PEPTIDEK")))
    # Residue 0's variable modifications go with the forms that free it.
    kw = dict(fixed={"N-term": "UNIMOD:28"}, variable={"N-term": ["UNIMOD:1"], "Q": ["UNIMOD:7"]})
    forms = sorted(canonical(f) for f in enumerate_modforms("QPEPTIDEKQ", **kw))
    assert forms == sorted([
        "[UNIMOD:28]-QPEPTIDEKQ", "[UNIMOD:28]-QPEPTIDEKQ[UNIMOD:7]", "[UNIMOD:1]-QPEPTIDEKQ",
        "[UNIMOD:1]-Q[UNIMOD:7]PEPTIDEKQ", "[UNIMOD:1]-QPEPTIDEKQ[UNIMOD:7]", "[UNIMOD:1]-Q[UNIMOD:7]PEPTIDEKQ[UNIMOD:7]",
    ])  # fmt: skip
    assert count_modforms("QPEPTIDEKQ", **kw) == 6


def test_counts_and_forms_with_fixed_cyclizations_terminal_contexts_and_rules_in_any_order() -> None:
    """Random specs with a fixed N-terminal cyclization, a protein terminus
    and limits over sites two rules reach: the count is the enumeration,
    the forms do not depend on the order of the rules, and residue 0
    carries a chemical modification only where no cyclization holds the
    N-terminus."""
    from constellation.core.sequence.protein import SIDE_CHAIN_CYCLIZATIONS

    rng = random.Random(31)
    pool = {"N-term": ["UNIMOD:1", "UNIMOD:28"], "Protein N-term": ["UNIMOD:1"], "K": ["UNIMOD:1"],
            "Q": ["UNIMOD:7"], "M": ["UNIMOD:35"], "C": ["UNIMOD:4"]}  # fmt: skip
    for _ in range(300):
        seq = rng.choice("CQK") + "".join(rng.choice("ACKMQ") for _ in range(rng.randint(1, 8)))
        variable = {site: keys for site, keys in pool.items() if rng.random() < 0.6}
        fixed = {}
        if rng.random() < 0.5:
            fixed["N-term"] = rng.choice(["UNIMOD:26", "UNIMOD:28"])
        if rng.random() < 0.5:
            fixed["C"] = "UNIMOD:4"
            variable.pop("C", None)
        limits = [("UNIMOD:1", ["Protein N-term", "K"], rng.randint(0, 2))] if rng.random() < 0.6 else None
        context = SiteContext(protein_n_term=rng.random() < 0.7)
        k = rng.randint(0, 4)
        seen = []
        for spec in (variable, dict(reversed(list(variable.items())))):
            kw = dict(fixed=fixed, variable=spec, max_variable=k, context=context, limit_groups=limits,
                      skip_unlicensed=True)  # fmt: skip
            forms = enumerate_modforms(seq, **kw)
            assert len(forms) == count_modforms(seq, **kw)
            texts = [canonical(f) for f in forms]
            assert len(set(texts)) == len(texts)
            for f in forms:
                n_term = {t.mod.accession for t in f.n_term_mods}
                if n_term & {k.split(":")[1] for k in SIDE_CHAIN_CYCLIZATIONS}:
                    assert 0 not in f.residue_mods, texts
            seen.append(sorted(texts))
        assert seen[0] == seen[1], (seq, fixed, variable, limits)


def test_a_limit_prunes_the_walk() -> None:
    import time

    start = time.perf_counter()
    forms = enumerate_modforms("M" * 40, variable={"M": "UNIMOD:35"}, max_variable=40, max_per_mod={"UNIMOD:35": 2})
    assert len(forms) == 1 + 40 + 40 * 39 // 2
    assert count_modforms("M" * 60, variable={"M": "UNIMOD:35"}, max_variable=60) == 2**60
    assert time.perf_counter() - start < 2.0


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
