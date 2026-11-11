"""Defects of the search space found by the review of 2026-11-09
(``docs/reviews/pre-scoring-2026-11-09.md``, B1-B7), each held by a test
that fails on the code before: a resume keyed on the code that made it,
decoys on decoy proteins, a variable modification replacing a fixed one,
reverse-c-fixed and N-terminal rings, a limit per rule, decoy sets and
their counts, and one key per molecule. And those of the second review
(``docs/reviews/pre-scoring-2026-11-10.md``, B8-B11): the layout in the
metadata, the packaged tables in the resume key, reverse-c-fixed under any
residue-specific N-terminal modification, and provenance whatever the
rules' order."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pyarrow.compute as pc
import pytest

from constellation.core.sequence.proforma import parse_proforma
from constellation.core.sequence.protein import SIDE_CHAIN_CYCLIZATIONS, count_modforms, enumerate_modforms
from constellation.massspec.library.decoys import DecoySet
from constellation.massspec.library.search_space import (
    Digestion,
    ModRule,
    SearchSpaceSettings,
    build_search_space,
)

P1 = "MAVNEMTEFAKSSRQWVLLGCKAVNEMTEFAKLLTEMQRYCAK"
P2 = "GSSRAVNEMTEFAKLLTEMQRYCAKPEPTIDERLVNELTEFAK"  # shares AVNEMTEFAK and LLTEMQR with P1


def _fasta(tmp_path: Path, records: dict[str, str], name: str = "p.fa") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text("".join(f">{acc}\n{seq}\n" for acc, seq in records.items()))
    return path


def _settings(**kw) -> SearchSpaceSettings:
    base = SearchSpaceSettings(digestion=Digestion(missed_cleavages=1, excise_initiator_met=False), charges=(2,),
                               min_mz=0, max_mz=1e6, decoys=(DecoySet(),))  # fmt: skip
    return replace(base, **kw)


# ── B1: the resume key ──────────────────────────────────────────────


def _cli(*argv: str) -> int:
    from constellation.cli.__main__ import _build_parser

    args = _build_parser().parse_args(["massspec", "search-space", *argv])
    return args.func(args)


def test_resume_is_keyed_by_what_the_code_writes_not_its_install(tmp_path, monkeypatch, capsys):
    import constellation
    from constellation.massspec.library import search_space as module

    fasta = _fasta(tmp_path, {"P1": P1})
    out = tmp_path / "space"
    argv = ("--fasta", str(fasta), "--output-dir", str(out), "--min-mz", "0")
    assert _cli(*argv) == 0
    capsys.readouterr()
    assert _cli(*argv, "--resume") == 0
    assert "already built" in capsys.readouterr().out
    # The install's version is not in the key: an editable install never
    # rewrites it, so it let older code's output pass as current.
    monkeypatch.setattr(constellation, "__version__", "9.9.9")
    assert _cli(*argv, "--resume") == 0
    assert "already built" in capsys.readouterr().out
    # What the code writes, and the vocabulary it read, are.
    monkeypatch.setattr(module, "SEARCH_SPACE_LAYOUT", module.SEARCH_SPACE_LAYOUT + 1)
    assert _cli(*argv, "--resume") == 0
    assert "already built" not in capsys.readouterr().out
    monkeypatch.setattr(module, "_vocabulary_digest", lambda: "another vocabulary")
    assert _cli(*argv, "--resume") == 0
    assert "already built" not in capsys.readouterr().out


# ── B2: decoys on decoy proteins ────────────────────────────────────


def test_peptide_level_decoys_have_decoy_proteins(tmp_path):
    lib = build_search_space([_fasta(tmp_path, {"P1": P1, "P2": P2})], _settings())
    proteins = {r["protein_id"]: r for r in lib.proteins.to_pylist()}
    assert sorted(r["accession"] for r in proteins.values() if r["is_decoy"]) == ["DECOY_P1", "DECOY_P2"]
    decoys = lib.peptides.filter(pc.equal(lib.peptides.column("is_decoy"), True))
    edges: dict[int, set[str]] = {}
    for e in lib.protein_peptide.to_pylist():
        edges.setdefault(e["peptide_id"], set()).add(proteins[e["protein_id"]]["accession"])
    assert decoys.num_rows > 0
    for d in decoys.to_pylist():
        # Every decoy is on the decoys of its target's proteins, and only there.
        target = {p for p in edges[d["decoy_of"]]}
        assert edges[d["peptide_id"]] == {f"DECOY_{p}" for p in target}
    # A peptide shared by P1 and P2 gives a decoy shared by their decoys.
    assert any(len(v) == 2 and all(a.startswith("DECOY_") for a in v) for v in edges.values())
    # The decoy spans are the targets' places, with the decoys' sequences.
    assert lib.peptide_spans.num_rows == 2 * lib.peptide_spans.filter(
        pc.is_in(lib.peptide_spans.column("protein_id"),
                 value_set=lib.proteins.filter(pc.invert(lib.proteins.column("is_decoy"))).column("protein_id"))
    ).num_rows  # fmt: skip


def test_a_subset_keeps_a_decoy_whose_target_was_filtered_away(tmp_path):
    lib = build_search_space([_fasta(tmp_path, {"P1": P1, "P2": P2})], _settings())
    target = lib.peptides.filter(pc.invert(lib.peptides.column("is_decoy"))).column("peptide_id")[0].as_py()
    decoy = lib.peptides.filter(pc.equal(lib.peptides.column("decoy_of"), target)).column("peptide_id")[0].as_py()
    keep = lib.precursors.filter(pc.not_equal(lib.precursors.column("peptide_id"), target)).column("precursor_id")
    filtered = lib.subset_precursors(keep)
    p1 = filtered.proteins.filter(pc.equal(filtered.proteins.column("accession"), "P1")).column("protein_id")[0].as_py()
    sub = filtered.subset_proteins([p1, *filtered.proteins.filter(
        pc.equal(filtered.proteins.column("accession"), "P2")).column("protein_id").to_pylist()])  # fmt: skip
    kept = set(sub.peptides.column("peptide_id").to_pylist())
    assert target not in kept and decoy in kept
    # A named target brings its decoy protein.
    one = lib.subset_proteins([p1])
    assert sorted(one.proteins.column("accession").to_pylist()) == ["DECOY_P1", "P1"]
    assert pc.sum(one.peptides.column("is_decoy")).as_py() > 0


def test_fasta_entries_named_as_decoys(tmp_path):
    with pytest.raises(ValueError, match="accessions of the decoys made"):
        build_search_space([_fasta(tmp_path, {"P1": P1, "DECOY_P1": P2})], _settings())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        lib = build_search_space([_fasta(tmp_path, {"P1": P1, "rev_P2": P2}, "q.fa")], _settings())
    assert any("named as decoys" in str(w.message) for w in seen)
    assert lib.metadata["x.constellation.search_space"]["counts"]["decoy_named_targets"] == 1


# ── B3: a variable modification replaces a fixed one ────────────────


def _forms(tmp_path, rules, sequence):
    lib = build_search_space([_fasta(tmp_path, {"P1": sequence})],
                             _settings(rules=rules, decoys=(), digestion=Digestion(protease="No_enzyme", min_length=len(sequence),
                                                                                   max_length=len(sequence), excise_initiator_met=False)))  # fmt: skip
    return sorted(lib.peptides.column("modified_sequence").to_pylist()), lib


def test_a_variable_modification_replaces_a_fixed_one_on_its_site(tmp_path):
    tmt, acetyl = "UNIMOD:737", "UNIMOD:1"
    forms, lib = _forms(tmp_path, (ModRule.parse("TMT6plex@K", "fixed"), ModRule.parse("Acetyl@K", "variable")),
                        "PEPKTIDEK")  # fmt: skip
    assert f"PEPK[{acetyl}]TIDEK[{tmt}]" in forms and f"PEPK[{tmt}]TIDEK[{acetyl}]" in forms
    assert f"PEPK[{tmt}]TIDEK[{tmt}]" in forms
    assert not any(f"[{tmt}][{acetyl}]" in f or f"[{acetyl}][{tmt}]" in f for f in forms)  # never stacked
    # Its modification rows hold the acetyl, not the TMT it replaced there.
    by_id = {r["peptide_id"]: r["modified_sequence"] for r in lib.peptides.to_pylist()}
    for row in lib.peptide_modifications.to_pylist():
        form = parse_proforma(by_id[row["peptide_id"]])
        if row["position"] is not None:
            on = [t.mod.accession for t in form.residue_mods.get(row["position"], ())]
            assert row["modification"].split(":")[1] in on
    # The protein N-terminus: acetylated, its amine takes no TMT.
    forms, _ = _forms(tmp_path / "n", (ModRule.parse("TMT6plex@N-term", "fixed"),
                                       ModRule.parse("Acetyl@Protein N-term", "variable")), "PEPTIDEK")  # fmt: skip
    assert f"[{acetyl}]-PEPTIDEK" in forms and f"[{tmt}]-PEPTIDEK" in forms


def test_counts_with_replacements_are_what_enumeration_returns():
    fixed = {"K": "UNIMOD:737", "C": "UNIMOD:4", "N-term": "UNIMOD:737"}
    for variable in ({"K": "UNIMOD:1"}, {"K": ["UNIMOD:1", "UNIMOD:34"], "M": "UNIMOD:35"},
                     {"N-term": "UNIMOD:1", "C": "UNIMOD:39"}):  # fmt: skip
        for peptide in ("PEPKTIDEKMK", "CKMCKK", "MKCK"):
            for k in (0, 1, 2, 3):
                kw = dict(fixed=fixed, variable=variable, max_variable=k, validate_specificity=False)
                assert count_modforms(peptide, **kw) == len(enumerate_modforms(peptide, **kw))


# ── B4: reverse-c-fixed keeps an N-terminal ring's residue ──────────


def test_reverse_c_fixed_keeps_a_ringed_first_residue_in_place(tmp_path):
    from constellation.massspec.library.decoys import _ring_key

    fasta = _fasta(tmp_path, {"P1": "QAVNEMTEFAKQLLTEMQRYCAKEPEPTIDERLVNELTEFAK", "P2": P2})
    rules = (ModRule.parse("Gln->pyro-Glu@N-term", "variable"), ModRule.parse("Glu->pyro-Glu@N-term", "variable"))
    lib = build_search_space([fasta], _settings(rules=rules, decoys=(DecoySet("reverse-c-fixed"),)))
    decoys = lib.peptides.filter(pc.equal(lib.peptides.column("is_decoy"), True)).column("modified_sequence")
    ringed = [parse_proforma(s) for s in decoys.to_pylist()
              if any(_ring_key(t) in SIDE_CHAIN_CYCLIZATIONS for t in parse_proforma(s).n_term_mods)]  # fmt: skip
    assert ringed
    assert all(p.sequence[0] in "QE" for p in ringed)
    # A sequence with no ring is reversed as before: all but the C-terminal residue.
    plain = {p.sequence for p in map(parse_proforma, decoys.to_pylist())}
    targets = {r["sequence"] for r in lib.peptides.filter(pc.invert(lib.peptides.column("is_decoy"))).to_pylist()}
    assert "AVNEMTEFAK" in targets and "AFETMENVAK" in plain


# ── B5: a limit per rule ────────────────────────────────────────────


def test_a_limit_is_on_its_own_rules_placements(tmp_path):
    rules = (ModRule.parse("Oxidation@M", "variable"), ModRule.parse("Oxidation@W:max=1", "variable"))
    forms, _ = _forms(tmp_path, rules, "MWMWK")
    ox = "[UNIMOD:35]"
    assert f"M{ox}WM{ox}WK" in forms  # two methionines: the W limit does not cap them
    assert not any(f.count(ox) == 2 and f"W{ox}" in f and f.count(f"W{ox}") == 2 for f in forms)
    assert sum(f.count(f"W{ox}") >= 1 for f in forms) > 0
    assert max(f.count(f"W{ox}") for f in forms) == 1
    with pytest.raises(ValueError, match="different limits"):
        _settings(rules=(ModRule.parse("Oxidation@M", "variable"), ModRule.parse("Oxidation@M:max=1", "variable")))


def test_limits_per_group_count_as_they_enumerate():
    groups = [("UNIMOD:35", ["W"], 1), ("UNIMOD:35", ["M", "W"], 2)]
    for peptide in ("MWMWK", "MMMWWW", "WWK"):
        for k in (1, 2, 3, 4):
            kw = dict(variable={"M": "UNIMOD:35", "W": "UNIMOD:35"}, max_variable=k, limit_groups=groups,
                      validate_specificity=False)  # fmt: skip
            forms = enumerate_modforms(peptide, **kw)
            assert count_modforms(peptide, **kw) == len(forms)
            for f in forms:
                w = sum(1 for i, mods in f.residue_mods.items() if peptide[i] == "W" and mods)
                assert w <= 1 and len(f.residue_mods) <= 2


# ── B6: decoy sets and their counts ─────────────────────────────────


def test_a_second_set_of_an_unseeded_method_is_refused():
    with pytest.raises(ValueError, match="copy the first"):
        _settings(decoys=(DecoySet("reverse-protein", 0), DecoySet("reverse-protein", 1)))
    with pytest.raises(ValueError, match="copy the first"):
        _settings(decoys=(DecoySet("reverse-interior", 0), DecoySet("reverse-interior", 1, seed=3)))
    _settings(decoys=(DecoySet("reverse-interior", 0), DecoySet("shuffle-interior", 1), DecoySet("shuffle-interior", 2)))


def test_two_methods_under_one_set_index_are_refused_and_apart_are_built(tmp_path):
    """Two methods at index 0 were one set of twice the decoys, and since
    peptide-level decoys have proteins, two proteins under one accession
    (Codex, 2026-11-10). Each set is one method's; numbered apart, the two
    build, every protein id once."""
    with pytest.raises(ValueError, match="decoy set 0 given twice"):
        _settings(decoys=(DecoySet("reverse-protein"), DecoySet("reverse-interior")))
    fasta = _fasta(tmp_path, {"P1": P1, "P2": P2})
    lib = build_search_space([fasta], _settings(decoys=(DecoySet("reverse-protein", 0), DecoySet("reverse-interior", 1))))
    ids = lib.proteins.column("protein_id").to_pylist()
    assert len(ids) == len(set(ids))
    accessions = lib.proteins.column("accession").to_pylist()
    assert {"DECOY_P1", "DECOY_P2", "DECOY1_P1", "DECOY1_P2"} <= set(accessions)


def test_the_cli_refuses_reverse_protein_sets_of_two(tmp_path, capsys):
    fasta = _fasta(tmp_path, {"P1": P1})
    assert _cli("--fasta", str(fasta), "--output-dir", str(tmp_path / "o"), "--decoys", "reverse-protein:sets=2") == 2
    assert "copy the first" in capsys.readouterr().err


def test_protein_level_decoys_are_counted_as_decoys(tmp_path):
    fasta = _fasta(tmp_path, {"P1": P1, "P2": P2})
    lib = build_search_space([fasta], _settings(decoys=(DecoySet("reverse-protein"),)))
    counts = lib.metadata["x.constellation.search_space"]["counts"]
    decoy = pc.sum(lib.peptides.column("is_decoy")).as_py()
    assert counts["decoy_peptidoforms"] == decoy > 0
    assert counts["peptidoforms"] == lib.n_peptides - decoy
    decoy_ids = lib.peptides.filter(lib.peptides.column("is_decoy")).column("peptide_id")
    n_decoy_precursors = lib.precursors.filter(pc.is_in(lib.precursors.column("peptide_id"), value_set=decoy_ids)).num_rows
    assert counts["decoy_precursors"] == n_decoy_precursors
    assert counts["precursors"] == lib.n_precursors - n_decoy_precursors


# ── the second review (2026-11-10), B8-B11 ──────────────────────────


def _without_layout(lib, layout=None):
    """``lib`` as older code wrote it: its search-space record with no
    layout, or with ``layout``."""
    record = {k: v for k, v in lib.metadata["x.constellation.search_space"].items() if k != "layout"}
    if layout is not None:
        record["layout"] = layout
    return lib.with_metadata({"x.constellation.search_space": record})


def test_a_search_space_records_its_layout_and_one_older_code_made_is_warned_about(tmp_path):
    """B8: the layout was only in the resume key, so a search space built
    before PR #110 (no decoy proteins) was predicted and subset silently."""
    from constellation.massspec.library import load_library, save_library
    from constellation.massspec.library.search_space import SEARCH_SPACE_LAYOUT, check_search_space_layout

    lib = build_search_space([_fasta(tmp_path, {"P1": P1, "P2": P2})], _settings())
    assert lib.metadata["x.constellation.search_space"]["layout"] == SEARCH_SPACE_LAYOUT == 2
    p1 = lib.proteins.filter(pc.equal(lib.proteins.column("accession"), "P1")).column("protein_id").to_pylist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a current one passes without a word
        check_search_space_layout(lib)
        lib.subset_proteins(p1)
        # and so does a library that is no search space (a jar's, a .dlib)
        check_search_space_layout(replace(lib, metadata_extras={}))
    with pytest.warns(UserWarning, match=r"records no layout, and the current layout is 2.*massspec search-space"):
        _without_layout(lib).subset_proteins(p1)
    with pytest.warns(UserWarning, match=r"has layout 1, and the current layout is 2"):
        check_search_space_layout(_without_layout(lib, 1))
    # Written and read back, the record survives.
    save_library(lib, tmp_path / "space", format="parquet_dir")
    assert load_library(tmp_path / "space").metadata["x.constellation.search_space"]["layout"] == 2


def test_the_resume_key_covers_the_protease_and_element_tables(tmp_path, monkeypatch):
    """B9: proteases.json and elements.json change what a search space
    holds, and were not in its key."""
    import importlib.resources as resources

    from constellation.massspec.library.search_space import search_space_key

    fasta, settings = _fasta(tmp_path, {"P1": P1}), _settings()
    key = search_space_key([fasta], settings)
    packaged, real_files = resources.files("constellation.data"), resources.files

    def data_with(edited: str | None) -> Path:
        copy = tmp_path / f"data-{edited}"
        copy.mkdir()
        for name in ("unimod.json", "proteases.json", "elements.json"):
            (copy / name).write_bytes(packaged.joinpath(name).read_bytes() + (b"\n" if name == edited else b""))
        return copy

    for edited, same in ((None, True), ("proteases.json", False), ("elements.json", False)):
        copy = data_with(edited)
        monkeypatch.setattr(
            resources, "files", lambda package, copy=copy: copy if package == "constellation.data" else real_files(package)
        )
        assert (search_space_key([fasta], settings) == key) is same, edited


def test_reverse_c_fixed_keeps_the_first_residue_under_any_residue_specific_n_terminal_modification(tmp_path):
    """B10: only the side-chain cyclizations (UNIMOD:26/27/28) kept residue
    0 in place; UNIMOD:385 (ammonia loss, at an N-terminus only on C, S or
    T) moved with the rest, leaving the loss on a residue that cannot make
    it."""
    from constellation.core.sequence.proforma import ModRef, Peptidoform, TaggedMod
    from constellation.massspec.library.decoys import peptide_decoys

    fasta = _fasta(tmp_path, {"P1": "CAVNEMTEFAKCLLTEMQRYCAKEPEPTIDERLVNELTEFAK"})
    lib = build_search_space([fasta], _settings(rules=(ModRule.parse("Ammonia-loss@N-term", "variable"),),
                                                decoys=(DecoySet("reverse-c-fixed"),)))  # fmt: skip
    decoys = [parse_proforma(s) for s in lib.peptides.filter(lib.peptides.column("is_decoy")).column("modified_sequence").to_pylist()]
    lost = [p for p in decoys if any(t.mod.accession == "385" for t in p.n_term_mods)]
    assert lost and all(p.sequence[0] == "C" for p in lost)
    # Every form of the sequence shares its decoy's order, the bare one too.
    assert "CAFETMENVAK" in {p.sequence for p in decoys}
    # A sequence that cannot take one (no C, S or T first) is reversed as before.
    assert "AFETLENVLK" in {p.sequence for p in decoys}

    def first_residue(n_term: ModRef) -> str:
        target = Peptidoform("CAVNEMTEFAK", n_term_mods=(TaggedMod(mod=n_term),))
        out = peptide_decoys([target], DecoySet("reverse-c-fixed"), taken=set(), key=repr)
        return out.decoy[0].sequence[0]

    assert first_residue(ModRef(cv="UNIMOD", accession="385")) == "C"
    assert first_residue(ModRef(cv="UNIMOD", accession="28")) == "C"  # a ring, kept as before
    # On the N-term site itself (acetyl), with no accession, or not UNIMOD's: moved, as before.
    assert first_residue(ModRef(cv="UNIMOD", accession="1")) == "A"
    assert first_residue(ModRef(name="Ammonia-loss")) == "A"
    assert first_residue(ModRef(cv="MOD", accession="00040")) == "A"


def test_the_provenance_of_a_placement_does_not_depend_on_the_rules_order(tmp_path):
    """B11: a placement two rules reached was recorded under whichever rule
    came first; now it has a row for each, so permuting the rules gives the
    same forms and the same modification table."""
    import itertools

    rules = (ModRule.parse("Acetyl@N-term:max=1", "variable"), ModRule.parse("Acetyl@Protein N-term:max=1", "variable"),
             ModRule.parse("Oxidation@M", "variable"), ModRule.parse("Oxidation@M,W", "variable"),
             ModRule.parse("Carbamidomethyl@C", "fixed"), ModRule.parse("Carbamidomethyl@C,H", "fixed"))  # fmt: skip
    fasta = _fasta(tmp_path, {"P1": P1, "P2": "MWHEMTEFAKCLLTEMQRYCAK"})
    seen = set()
    for order in itertools.islice(itertools.permutations(range(len(rules))), 0, None, 37):
        lib = build_search_space([fasta], _settings(rules=tuple(rules[i] for i in order)))
        forms = tuple(sorted(lib.peptides.column("modified_sequence").to_pylist()))
        rows = tuple(sorted((tuple(r.values()) for r in lib.peptide_modifications.to_pylist()), key=repr))
        seen.add((forms, rows))
    assert len(seen) == 1
    ((_forms, rows),) = seen
    by_placement: dict[tuple, set[str]] = {}
    for pid, position, terminus, mod, origin, rule in rows:
        by_placement.setdefault((pid, position, terminus, mod, origin), set()).add(rule)
    # The protein's N-terminal acetyl is both N-term rules'; a methionine's
    # oxidation both oxidation rules'; a cysteine's carbamidomethyl both fixed rules'.
    assert {"Acetyl@N-term:max=1", "Acetyl@Protein N-term:max=1"} in by_placement.values()
    assert {"Oxidation@M", "Oxidation@M,W"} in by_placement.values()
    assert {"Carbamidomethyl@C", "Carbamidomethyl@C,H"} in by_placement.values()
    assert {"Oxidation@M,W"} in by_placement.values()  # a tryptophan's: one rule's
