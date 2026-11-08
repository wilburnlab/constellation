"""``massspec.library.search_space``: FASTA → a provenance-keeping search
space, its decoys, budgets and rules, and ``digest.precursors_from_fasta``
as a projection of it (held to the frozen old one)."""

from __future__ import annotations

import random
from pathlib import Path

import _digest_oracle as old_digest
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from constellation.core.io.fasta import write_fasta
from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.library.decoys import DecoySet
from constellation.massspec.library.digest import precursors_from_fasta
from constellation.massspec.library.io import load_library, save_library
from constellation.massspec.library.search_space import (
    SEARCH_SPACE_METADATA_KEY,
    Digestion,
    ModRule,
    SearchSpaceBudgetError,
    SearchSpaceSettings,
    build_search_space,
    count_search_space,
    parse_label,
    parse_protein_terminus,
)
from constellation.massspec.peptide.mz import precursor_mz

AAS = "ACDEFGHIKLMNPQRSTVWY"
# Protein 1 starts MAVNEMTEFAK...; protein 2 carries AVNEMTEFAK inside, so
# the same sequence is protein-N-terminal (after Met) in one and internal in
# the other.
P1 = "MAVNEMTEFAKSSRQWVLLGCK"
P2 = "GSSRAVNEMTEFAKLLTEMQRYCAK"


@pytest.fixture
def fasta(tmp_path: Path) -> Path:
    p = tmp_path / "two.fasta"
    write_fasta(p, [P1, P2], ["sp|P1|ONE_TEST", "sp|P2|TWO_TEST"], descriptions=["One OS=Test GN=ONE", "Two"])
    return p


def _settings(**kw) -> SearchSpaceSettings:
    base = dict(
        digestion=Digestion(missed_cleavages=1, min_length=4, max_length=30),
        rules=(ModRule.parse("Carbamidomethyl@C", "fixed"), ModRule.parse("Oxidation@M", "variable")),
        charges=(2, 3),
        min_mz=0.0,
        max_mz=1e9,
        decoys=(),
    )
    base.update(kw)
    return SearchSpaceSettings(**base)


def _peptide(lib, modseq: str) -> int:
    t = lib.peptides.filter(pc.equal(lib.peptides.column("modified_sequence"), modseq))
    assert t.num_rows == 1, modseq
    return t.column("peptide_id")[0].as_py()


def _spans_of(lib, peptide_id: int) -> list[tuple[str, int, int]]:
    acc = dict(zip(lib.proteins.column("protein_id").to_pylist(), lib.proteins.column("accession").to_pylist(), strict=True))
    e = lib.span_peptide.filter(pc.equal(lib.span_peptide.column("peptide_id"), peptide_id))
    return sorted((acc[r["protein_id"]], r["start"], r["end"]) for r in e.to_pylist())


# ── the graph ──────────────────────────────────────────────────────────


def test_shared_peptide_is_one_peptidoform_with_two_spans(fasta: Path) -> None:
    lib = build_search_space([fasta], _settings())
    pid = _peptide(lib, "AVNEMTEFAK")
    assert _spans_of(lib, pid) == [("sp|P1|ONE_TEST", 1, 11), ("sp|P2|TWO_TEST", 4, 14)]
    pairs = set(zip(lib.protein_peptide.column("protein_id").to_pylist(), lib.protein_peptide.column("peptide_id").to_pylist(), strict=True))
    assert len(pairs) == lib.protein_peptide.num_rows  # the projection, no duplicates
    span = lib.peptide_spans.filter(pc.and_(pc.equal(lib.peptide_spans.column("start"), 1), pc.equal(lib.peptide_spans.column("end"), 11)))
    assert span.to_pylist()[0] | {"protein_id": 0} == {
        "protein_id": 0, "start": 1, "end": 11, "sequence": "AVNEMTEFAK", "missed_cleavages": 0,
        "n_term": "protein", "c_term": "enzyme", "preceding": "M", "following": "S",
    }


def test_protein_n_term_form_only_from_the_protein_n_terminal_span(fasta: Path) -> None:
    rules = (ModRule.parse("Acetyl@Protein N-term", "variable"),)
    lib = build_search_space([fasta], _settings(rules=rules))
    acetyl = _peptide(lib, "[UNIMOD:1]-AVNEMTEFAK")
    assert _spans_of(lib, acetyl) == [("sp|P1|ONE_TEST", 1, 11)]  # not P2's internal copy
    assert len(_spans_of(lib, _peptide(lib, "AVNEMTEFAK"))) == 2
    mods = lib.peptide_modifications.filter(pc.equal(lib.peptide_modifications.column("peptide_id"), acetyl)).to_pylist()
    assert mods == [{"peptide_id": acetyl, "position": None, "terminus": "N", "modification": "UNIMOD:1",
                     "origin": "variable", "rule": "Acetyl@Protein N-term"}]


def test_modification_rows_record_fixed_and_variable(fasta: Path) -> None:
    lib = build_search_space([fasta], _settings())
    pid = _peptide(lib, "LLTEM[UNIMOD:35]QR")
    rows = lib.peptide_modifications.filter(pc.equal(lib.peptide_modifications.column("peptide_id"), pid)).to_pylist()
    assert [(r["position"], r["modification"], r["origin"], r["rule"]) for r in rows] == [
        (4, "UNIMOD:35", "variable", "Oxidation@M")
    ]
    cys = _peptide(lib, "YC[UNIMOD:4]AK")
    rows = lib.peptide_modifications.filter(pc.equal(lib.peptide_modifications.column("peptide_id"), cys)).to_pylist()
    assert [(r["position"], r["origin"], r["rule"]) for r in rows] == [(1, "fixed", "Carbamidomethyl@C")]


def test_labels_make_heavy_forms_on_the_same_spans(fasta: Path) -> None:
    labels = (parse_label("sp|P2|TWO_TEST=13C,15N"), parse_label("P2=light"))  # by id and by accession
    lib = build_search_space([fasta], _settings(labels=labels))
    heavy = _peptide(lib, "<13C><15N>LLTEMQR")
    light = _peptide(lib, "LLTEMQR")
    assert _spans_of(lib, heavy) == _spans_of(lib, light)
    rows = lib.peptide_modifications.filter(pc.equal(lib.peptide_modifications.column("peptide_id"), heavy)).to_pylist()
    assert sorted(r["modification"] for r in rows if r["origin"] == "label") == ["13C", "15N"]
    # only P2's peptides are labelled
    assert not any("<13C>" in m and "QWVLLGC" in m for m in lib.peptides.column("modified_sequence").to_pylist())
    with pytest.raises(ValueError, match="name no FASTA"):
        build_search_space([fasta], _settings(labels=(parse_label("nope.fasta=13C"),)))


def test_declared_mature_terminus(fasta: Path) -> None:
    termini = (parse_protein_terminus("P2=5"),)  # mature P2 starts at its 5th residue
    rules = (ModRule.parse("Acetyl@Protein N-term", "variable"),)
    lib = build_search_space([fasta], _settings(rules=rules, protein_termini=termini))
    spans = _spans_of(lib, _peptide(lib, "[UNIMOD:1]-AVNEMTEFAK"))
    assert ("sp|P2|TWO_TEST", 4, 14) in spans
    assert parse_protein_terminus("P02769=25-607") == ("P02769", 25, 607)
    assert parse_protein_terminus("X=-40") == ("X", None, 40)


def test_noncanonical_residues_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "sec.fasta"
    write_fasta(p, ["AAUAAKGEPTIDEK*"], ["SEL"])
    lib = build_search_space([p], _settings(digestion=Digestion(missed_cleavages=0, min_length=4)))
    assert lib.peptides.column("sequence").to_pylist() == ["GEPTIDEK"]
    assert lib.metadata[SEARCH_SPACE_METADATA_KEY]["counts"]["spans_noncanonical_skipped"] == 1


def test_precursors_and_mz_window(fasta: Path) -> None:
    lib = build_search_space([fasta], _settings(min_mz=400.0, max_mz=700.0, charges=(1, 2, 3)))
    mz = lib.precursors.column("precursor_mz").to_numpy()
    assert ((mz >= 400) & (mz <= 700)).all()
    rows = lib.precursors.filter(pc.equal(lib.precursors.column("peptide_id"), _peptide(lib, "AVNEMTEFAK"))).to_pylist()
    assert rows
    for r in rows:  # the bits precursor_mz gives
        assert r["precursor_mz"] == precursor_mz(parse_proforma("AVNEMTEFAK"), charge=r["charge"])
    # every peptide has a precursor in range; spans only where a peptide survived
    assert set(lib.precursors.column("peptide_id").to_pylist()) == set(lib.peptides.column("peptide_id").to_pylist())


# ── decoys ─────────────────────────────────────────────────────────────


def test_peptide_decoys_keep_composition_termini_and_mods(fasta: Path) -> None:
    lib = build_search_space([fasta], _settings(decoys=(DecoySet(),)))
    peps = {r["peptide_id"]: r for r in lib.peptides.to_pylist()}
    decoys = [r for r in peps.values() if r["is_decoy"]]
    assert decoys and all(r["decoy_set"] == 0 and r["decoy_of"] in peps for r in decoys)
    mz_of = {}
    for r in lib.precursors.to_pylist():
        mz_of[(r["peptide_id"], r["charge"])] = r["precursor_mz"]
    for d in decoys:
        t = peps[d["decoy_of"]]
        assert d["sequence"][0] == t["sequence"][0] and d["sequence"][-1] == t["sequence"][-1]
        assert sorted(d["sequence"]) == sorted(t["sequence"])
        for z in (2, 3):
            assert mz_of.get((d["peptide_id"], z)) == mz_of.get((t["peptide_id"], z))
    ox = next(d for d in decoys if peps[d["decoy_of"]]["modified_sequence"] == "LLTEM[UNIMOD:35]QR")
    assert ox["modified_sequence"] == "LQM[UNIMOD:35]ETLR"
    rows = lib.peptide_modifications.filter(pc.equal(lib.peptide_modifications.column("peptide_id"), ox["peptide_id"])).to_pylist()
    assert [(r["position"], r["modification"]) for r in rows] == [(2, "UNIMOD:35")]
    # no decoy is a target sequence (I = L)
    targets = {r["sequence"].replace("I", "L") for r in peps.values() if not r["is_decoy"]}
    assert not any(d["sequence"].replace("I", "L") in targets for d in decoys)


def test_entrapment_sets_are_independent_and_deterministic(fasta: Path) -> None:
    sets = (DecoySet(), DecoySet("shuffle-interior", index=1, seed=3), DecoySet("shuffle-interior", index=2, seed=3))
    a = build_search_space([fasta], _settings(decoys=sets))
    b = build_search_space([fasta], _settings(decoys=sets))
    assert a.peptides.equals(b.peptides) and a.precursors.equals(b.precursors)
    by_set: dict[int, set[str]] = {}
    for r in a.peptides.to_pylist():
        if r["is_decoy"]:
            by_set.setdefault(r["decoy_set"], set()).add(r["sequence"])
    assert set(by_set) == {0, 1, 2}
    assert not (by_set[0] & by_set[1]) and not (by_set[1] & by_set[2])


def test_shuffles_do_not_depend_on_input_order(tmp_path: Path) -> None:
    f1, f2 = tmp_path / "a.fasta", tmp_path / "b.fasta"
    write_fasta(f1, [P1, P2], ["A", "B"])
    write_fasta(f2, [P2, P1], ["B", "A"])
    s = _settings(decoys=(DecoySet("shuffle-interior"),))
    def decoys(path):
        lib = build_search_space([path], s)
        return {r["modified_sequence"] for r in lib.peptides.to_pylist() if r["is_decoy"]}
    assert decoys(f1) == decoys(f2)


def test_protein_reversal(fasta: Path) -> None:
    lib = build_search_space([fasta], _settings(decoys=(DecoySet("reverse-protein"),)))
    acc = lib.proteins.column("accession").to_pylist()
    assert acc == ["sp|P1|ONE_TEST", "sp|P2|TWO_TEST", "DECOY_sp|P1|ONE_TEST", "DECOY_sp|P2|TWO_TEST"]
    assert lib.proteins.column("is_decoy").to_pylist() == [False, False, True, True]
    decoys = lib.peptides.filter(pc.fill_null(lib.peptides.column("is_decoy"), False))
    assert decoys.num_rows and decoys.column("decoy_of").null_count == decoys.num_rows
    assert set(decoys.column("decoy_set").to_pylist()) == {0}


# ── budgets, counts, determinism, rules ────────────────────────────────


def test_budgets_refuse_before_building(fasta: Path) -> None:
    wide = _settings(rules=(ModRule.parse("Phospho@S,T,Y", "variable"), ModRule.parse("Oxidation@M", "variable")),
                     max_variable=3, digestion=Digestion(specificity="unspecific", min_length=4, max_length=20))
    counts = count_search_space([fasta], replace_budget(wide, None, None))
    with pytest.raises(SearchSpaceBudgetError, match="peptidoforms"):
        build_search_space([fasta], replace_budget(wide, counts["peptidoforms_planned"] - 1, None))
    with pytest.raises(SearchSpaceBudgetError, match="precursors"):
        build_search_space([fasta], replace_budget(wide, None, counts["precursors"] - 1))
    lib = build_search_space([fasta], replace_budget(wide, None, None))
    assert lib.n_precursors == counts["precursors"]


def replace_budget(s: SearchSpaceSettings, peptidoforms, precursors) -> SearchSpaceSettings:
    from dataclasses import replace

    return replace(s, max_peptidoforms=peptidoforms, max_precursors=precursors)


def test_writes_the_same_bytes_twice(fasta: Path, tmp_path: Path) -> None:
    s = _settings(decoys=(DecoySet(),))
    for name in ("one", "two"):
        save_library(build_search_space([fasta], s), tmp_path / name, format="parquet_dir")
    for f in sorted((tmp_path / "one").glob("*.parquet")):
        assert f.read_bytes() == (tmp_path / "two" / f.name).read_bytes(), f.name
    back = load_library(tmp_path / "one", format="parquet_dir")
    assert back.span_peptide.num_rows > 0 and back.metadata[SEARCH_SPACE_METADATA_KEY]["inputs"][0]["sha256"]


def test_rule_parsing() -> None:
    assert set(ModRule.parse("Phospho", "variable").sites) == {"S", "T", "Y"}  # UNIMOD's shown sites
    assert ModRule.parse("UNIMOD:259", "fixed").sites == ("K",)  # shown nowhere: its hidden ones
    assert "H" in ModRule.parse("Phospho@*", "variable").sites
    r = ModRule.parse("UNIMOD:21@S,T:max=2", "variable")
    assert (r.modification, r.sites, r.max_per_peptide) == ("UNIMOD:21", ("S", "T"), 2)
    assert ModRule.parse("Acetyl@protein-n-term", "variable").sites == ("Protein N-term",)
    for bad, kind in [("Oxidation@A", "variable"), ("Nope@M", "variable"), ("Oxidation@Z", "variable"),
                      ("Carbamidomethyl@C:max=1", "fixed")]:
        with pytest.raises(ValueError):
            ModRule.parse(bad, kind)
    with pytest.raises(ValueError, match="two fixed"):
        SearchSpaceSettings(rules=(ModRule.parse("Carbamidomethyl@C", "fixed"), ModRule.parse("Propionamide@C", "fixed")))


def test_fixed_rules_conflict_only_where_both_apply() -> None:
    # Gln->pyro-Glu and Glu->pyro-Glu are both N-terminal, on different residues
    SearchSpaceSettings(rules=(ModRule.parse("UNIMOD:28", "fixed"), ModRule.parse("UNIMOD:27", "fixed")))
    # SILAC lysine and a chemical modification of lysine share the residue
    SearchSpaceSettings(rules=(ModRule.parse("UNIMOD:259@K", "fixed"), ModRule.parse("Dimethyl@K", "fixed")))
    with pytest.raises(ValueError, match="two fixed"):
        SearchSpaceSettings(rules=(ModRule.parse("UNIMOD:259@K", "fixed"), ModRule.parse("UNIMOD:188@K", "fixed")))
    with pytest.raises(ValueError, match="two fixed"):
        SearchSpaceSettings(rules=(ModRule.parse("Amidated@Protein C-term", "fixed"), ModRule.parse("Methyl@C-term", "fixed")))


def test_links_are_refused_and_monolinks_are_not() -> None:
    for linked in ("UNIMOD:374@C", "UNIMOD:1898@K", "UNIMOD:1896@K"):
        with pytest.raises(ValueError, match="links two residues"):
            ModRule.parse(linked, "variable")
    assert ModRule.parse("UNIMOD:1020@K", "variable").sites == ("K",)  # water-quenched DSS monolink
    assert ModRule.parse("UNIMOD:824@C", "variable").sites == ("C",)  # BMOE: intact or monolink


def test_label_spellings_are_one_state() -> None:
    assert parse_label("P=D,2H,15N") == ("P", ("15N", "2H"))
    for bad in ("P=33S,34S", "P=99C", "P=13C,12C", "P=Cx"):
        with pytest.raises(ValueError):
            parse_label(bad)
    s = SearchSpaceSettings(labels=(("P", ("D",)), ("P", ("2H",))))
    assert s.labels == (("P", ("2H",)),)


def test_labels_and_isotopic_substitutions_do_not_mix(fasta: Path) -> None:
    with pytest.raises(ValueError, match="label one residue's atoms twice"):
        build_search_space(
            [fasta], _settings(labels=(("P1", ("13C",)),), rules=(ModRule.parse("UNIMOD:259@K", "fixed"),))
        )


def test_unmatched_termini_are_refused(fasta: Path) -> None:
    with pytest.raises(ValueError, match="protein termini name no protein"):
        build_search_space([fasta], _settings(protein_termini=(("P9", 3, None),)))


def test_heavy_lysine_keeps_its_acetylated_forms(fasta: Path) -> None:
    lib = build_search_space(
        [fasta], _settings(rules=(ModRule.parse("UNIMOD:259@K", "fixed"), ModRule.parse("Acetyl@K", "variable")))
    )
    texts = set(lib.peptides.column("modified_sequence").to_pylist())
    assert "AVNEMTEFAK[UNIMOD:259]" in texts and "AVNEMTEFAK[UNIMOD:1][UNIMOD:259]" in texts
    rows = lib.peptide_modifications.to_pylist()
    assert {(r["modification"], r["origin"]) for r in rows} >= {("UNIMOD:259", "fixed"), ("UNIMOD:1", "variable")}


def test_subsetting_precursors_keeps_a_decoy_without_its_target(fasta: Path) -> None:
    """A decoy stands on its own: dropping its target's precursors keeps it
    (until 2026-11-07 it went with its target), its decoy_of a record."""
    lib = build_search_space([fasta], _settings(decoys=(DecoySet(),)))
    peptides = lib.peptides.to_pylist()
    target = next(r for r in peptides if not r["is_decoy"] and any(d["decoy_of"] == r["peptide_id"] for d in peptides))
    drop = pc.equal(lib.precursors.column("peptide_id"), target["peptide_id"])
    sub = lib.subset_precursors(lib.precursors.filter(pc.invert(drop)).column("precursor_id"))
    left = set(sub.peptides.column("peptide_id").to_pylist())
    assert target["peptide_id"] not in left
    assert any(r["decoy_of"] == target["peptide_id"] for r in sub.peptides.to_pylist())
    assert sub.peptide_spans.num_rows <= lib.peptide_spans.num_rows
    assert set(sub.span_peptide.column("peptide_id").to_pylist()) <= left


# ── digest.precursors_from_fasta, as a projection ──────────────────────


def test_precursors_from_fasta_matches_the_old_enumerator(tmp_path: Path) -> None:
    rng = random.Random(21)
    seqs = [("M" if rng.random() < 0.6 else "") + "".join(rng.choice(AAS) for _ in range(rng.randint(30, 300))) for _ in range(40)]
    p = tmp_path / "rand.fasta"
    write_fasta(p, seqs, [f"P{i}" for i in range(len(seqs))])
    kw = dict(missed_cleavages=2, fixed_mods={"C": "UNIMOD:4"}, variable_mods={"M": "UNIMOD:35"}, max_variable_mods=2, charges=(2, 3, 4))
    old = old_digest.precursors_from_fasta(p, **kw)
    new = precursors_from_fasta(p, **kw)
    old_set = set(old)
    extra = [x for x in new if x not in old_set]
    # the one difference: a Met-clipped peptide of exactly max_length whose
    # unclipped parent was one too long (the old code never clipped it)
    assert all(len(x.sequence) == 30 and any(s.startswith("M" + x.sequence) for s in seqs) for x in extra)
    assert [x for x in new if x in old_set] == old
    assert pa.array([x.precursor_mz for x in new]).null_count == 0
