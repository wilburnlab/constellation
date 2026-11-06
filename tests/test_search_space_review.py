"""Independent chemistry and enumeration checks for the search-space review.

The defects the review found (R1-R12, G1; docs/reviews/search-space-2026-11-05.md)
were fixed on the branch it reviewed; each test named for one now holds
the fix (the strict-xfail markers it was written with are gone).
"""

from __future__ import annotations

import itertools
import math
import signal
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pyteomics import mass as reference_mass

from constellation.core.chem.composition import Composition
from constellation.core.chem.elements import PROTON_MASS
from constellation.core.chem.isotopes import isotope_envelope
from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.proforma import Peptidoform, format_proforma, parse_proforma
from constellation.core.sequence.protein import (
    SiteContext,
    cleave,
    count_modforms,
    enumerate_modforms,
    peptide_composition,
    peptide_mass,
    peptide_masses,
)
from constellation.massspec.library.decoys import DecoySet, il_key
from constellation.massspec.library.digest import precursors_from_library
from constellation.massspec.library.io import load_library, save_library
from constellation.massspec.library.search_space import (
    Digestion,
    ModRule,
    SearchSpaceBudgetError,
    SearchSpaceSettings,
    build_search_space,
    parse_label,
)
from constellation.massspec.peptide.envelope import peptide_envelope


def _settings(**kwargs):
    base = SearchSpaceSettings(
        digestion=Digestion(
            protease="No_enzyme", min_length=1, max_length=50,
            excise_initiator_met=False,
        ),
        charges=(2,), min_mz=0, max_mz=1e6, decoys=(),
    )
    return replace(base, **kwargs)


def _build(tmp_path, sequence="PEPTIDEK", **kwargs):
    fasta = tmp_path / "protein.fa"
    fasta.write_text(f">P\n{sequence}\n")
    return build_search_space([fasta], _settings(**kwargs))


def _texts(lib):
    return set(lib.peptides["modified_sequence"].to_pylist())


def _neutral_mass(lib):
    assert lib.n_precursors == 1
    row = lib.precursors.to_pylist()[0]
    return (row["precursor_mz"] - PROTON_MASS) * row["charge"]


# Each expectation enumerates all substrings and counts internal cuts directly;
# it does not call cleave_sites, a regex, or the frozen implementation oracle.
@pytest.mark.parametrize("enzyme", ["Trypsin", "AspN", "No_enzyme"])
@pytest.mark.parametrize("specificity", ["specific", "semi", "unspecific"])
@pytest.mark.parametrize("missed", [0, 1, 3])
@pytest.mark.parametrize("met", [False, True])
def test_digestion_matches_independent_substring_oracle(enzyme, specificity, missed, met):
    sequences = ["M" + "".join(middle) + "DK" for middle in itertools.product("AKRP", repeat=3)]
    for seq in sequences:
        n = len(seq)
        cuts = {
            i for i in range(1, n)
            if (enzyme == "Trypsin" and seq[i - 1] in "KR" and seq[i] != "P")
            or (enzyme == "AspN" and seq[i] == "D")
        }
        nterm = {0, 3} | ({1} if met else set())
        cterm = {n, n - 1}
        expected = {}
        for start in range(n):
            for end in range(start + 2, min(n, start + 5) + 1):
                mc = sum(start < cut < end for cut in cuts)
                ns, cs = start in nterm | cuts, end in cterm | cuts
                keep = specificity == "unspecific" or (
                    mc <= missed and (ns and cs if specificity == "specific" else ns or cs)
                )
                if keep:
                    expected[start, end] = (
                        seq[start:end], mc,
                        "protein" if start in nterm else "enzyme" if start in cuts else "nonspecific",
                        "protein" if end in cterm else "enzyme" if end in cuts else "nonspecific",
                        seq[start - 1] if start else "-", seq[end] if end < n else "-",
                    )
        spans = cleave(
            seq, enzyme, specificity=specificity, missed_cleavages=missed,
            min_length=2, max_length=5, excise_initiator_met=met,
            extra_n_termini=[3], extra_c_termini=[n - 1], return_spans=True,
        )
        actual = {
            (p.start, p.end): (p.sequence, p.n_missed, p.n_term, p.c_term, p.preceding, p.following)
            for p in spans
        }
        assert len(actual) == len(spans)
        assert actual == expected, (seq, enzyme, specificity, missed, met)


@pytest.mark.parametrize("max_variable", [0, 1, 2, 4])
@pytest.mark.parametrize("oxidations", [0, 1, 2])
@pytest.mark.parametrize("phosphates", [0, 1, 2])
def test_modification_counts_and_localizations_match_cartesian_oracle(max_variable, oxidations, phosphates):
    # CAM on C, alternative oxidation/dioxidation on each M, phospho on S/T.
    sequence = "CMMSST"
    choices = [(), ("UNIMOD:35",), ("UNIMOD:425",)]
    expected = set()
    for states in itertools.product(choices, choices, [(), ("UNIMOD:21",)], [(), ("UNIMOD:21",)], [(), ("UNIMOD:21",)]):
        selected = [key for state in states for key in state]
        if len(selected) > max_variable or selected.count("UNIMOD:35") > oxidations or selected.count("UNIMOD:21") > phosphates:
            continue
        text = "C[UNIMOD:4]"
        text += "".join(aa + "".join(f"[{key}]" for key in state) for aa, state in zip(sequence[1:], states, strict=True))
        expected.add(text)
    kwargs = dict(
        fixed={"C": "UNIMOD:4"},
        variable={"M": ["UNIMOD:35", "UNIMOD:425"], "S": "UNIMOD:21", "T": "UNIMOD:21"},
        max_variable=max_variable,
        max_per_mod={"UNIMOD:35": oxidations, "UNIMOD:21": phosphates},
    )
    forms = enumerate_modforms(sequence, **kwargs)
    assert {format_proforma(f) for f in forms} == expected
    assert len(forms) == count_modforms(sequence, **kwargs) == len(expected)


@pytest.mark.parametrize("sequence,modseq,delta", [
    ("ACDMK", "AC[UNIMOD:4]DM[UNIMOD:35]K", 57.021463735 + 15.994914620),
    ("NQSTK", "N[UNIMOD:7]QS[UNIMOD:21]TK", 0.984015583 + 79.966330890),
    ("QPEPTIDE", "[UNIMOD:28]-QPEPTIDE", -17.026549101),
    ("EPEPTIDE", "[UNIMOD:27]-EPEPTIDE", -18.010564684),
    ("CPEPTIDE", "[UNIMOD:26]-CPEPTIDE", 39.994914620),
    ("PEPTIDEK", "PEPTIDEK[UNIMOD:188]", 6.020129),
    ("PEPTIDEK", "PEPTIDEK[UNIMOD:259]", 8.014199),
    ("PEPTIDEK", "PEPTIDEK[UNIMOD:737]", 229.162932),
])
def test_masses_match_independent_residue_masses_and_tabulated_deltas(sequence, modseq, delta):
    expected = reference_mass.fast_mass(sequence) + delta
    form = parse_proforma(modseq)
    assert peptide_mass(form) == pytest.approx(expected, abs=2e-6)
    assert peptide_masses([form]).item() == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize("labels", [("13C",), ("15N",), ("13C", "15N")])
def test_global_labels_on_unmodified_peptides_use_exact_atom_counts(tmp_path, labels):
    seq = "ACDEK"
    # ACDEK = C21 H36 N6 O10 S; counts independently summed from residues + water.
    shift = (21 * 1.00335483507 if "13C" in labels else 0) + (6 * 0.99703489445 if "15N" in labels else 0)
    lib = _build(tmp_path, seq, labels=(("P", labels),))
    assert _neutral_mass(lib) == pytest.approx(reference_mass.fast_mass(seq) + shift, abs=2e-6)


def test_c_terminal_context_and_mature_c_terminus(tmp_path):
    lib = _build(
        tmp_path, "PEPTIDEK", protein_termini=(("P", None, 5),),
        rules=(ModRule.parse("Amidated@Protein C-term", "variable"),),
    )
    assert {"PEPTI", "PEPTI-[UNIMOD:2]", "PEPTIDEK", "PEPTIDEK-[UNIMOD:2]"} == _texts(lib)
    internal = enumerate_modforms(
        "PEPTI", context=SiteContext(), variable={"Protein C-term": "UNIMOD:2"},
    )
    assert [format_proforma(f) for f in internal] == ["PEPTI"]


def test_light_and_heavy_forms_are_filtered_by_their_own_masses(tmp_path):
    light = reference_mass.fast_mass("ACDEK") / 2 + PROTON_MASS
    lib = _build(
        tmp_path, "ACDEK", labels=(("P", ()), ("P", ("13C", "15N"))),
        min_mz=light + 1, max_mz=light + 20,
    )
    assert _texts(lib) == {"<13C><15N>ACDEK"}


def test_default_decoys_preserve_complex_terminal_labelled_forms_and_roundtrip(tmp_path):
    lib = _build(
        tmp_path, "QCMSTPEK", labels=(("P", ("13C", "15N")),),
        rules=tuple(ModRule.parse(rule, kind) for rule, kind in [
            ("Carbamidomethyl@C", "fixed"), ("UNIMOD:28", "variable"),
            ("Oxidation@M", "variable"), ("Phospho@S,T", "variable"),
        ]), max_variable=3, decoys=(DecoySet(),),
    )
    peptides = {r["peptide_id"]: r for r in lib.peptides.to_pylist()}
    decoys = [r for r in peptides.values() if r["is_decoy"]]
    assert decoys
    for decoy in decoys:
        source = peptides[decoy["decoy_of"]]
        a, b = parse_proforma(source["modified_sequence"]), parse_proforma(decoy["modified_sequence"])
        assert a.sequence[0] == b.sequence[0] and a.sequence[-1] == b.sequence[-1]
        assert peptide_composition(a) == peptide_composition(b)
        assert peptide_mass(a) == pytest.approx(peptide_mass(b), abs=1e-10)
    save_library(lib, tmp_path / "library", format="parquet_dir")
    back = load_library(tmp_path / "library")
    for name in ("proteins", "peptides", "precursors", "peptide_spans", "span_peptide", "peptide_modifications"):
        assert getattr(lib, name).equals(getattr(back, name))


@pytest.mark.parametrize("labels", [("13C",), ("15N",), ("13C", "15N")])
def test_metabolic_label_does_not_label_added_alkylation_reagent(tmp_path, labels):
    lib = _build(
        tmp_path, "ACDEK", labels=(("P", labels),),
        rules=(ModRule.parse("Carbamidomethyl@C", "fixed"),),
    )
    shift = (21 * 1.00335483507 if "13C" in labels else 0) + (6 * 0.99703489445 if "15N" in labels else 0)
    expected = reference_mass.fast_mass("ACDEK") + shift + 57.021463735
    assert _neutral_mass(lib) == pytest.approx(expected, abs=2e-6)


def test_fixed_silac_keeps_compatible_acetylated_lysine_forms(tmp_path):
    lib = _build(tmp_path, rules=(ModRule.parse("UNIMOD:259@K", "fixed"), ModRule.parse("Acetyl@K", "variable")))
    masses = sorted((r["precursor_mz"] - PROTON_MASS) * r["charge"] for r in lib.precursors.to_pylist())
    expected = reference_mass.fast_mass("PEPTIDEK") + 8.014199
    assert masses == pytest.approx([expected, expected + 42.010564684], abs=2e-6)


def test_pyro_cysteine_combination_is_resolved_or_rejected(tmp_path):
    try:
        lib = _build(tmp_path, "CPEPTIDEK", rules=(ModRule.parse("Carbamidomethyl@C", "fixed"), ModRule.parse("UNIMOD:26", "variable")))
    except ValueError:
        return  # Explicit incompatibility is acceptable; silently wrong chemistry is not.
    masses = sorted((r["precursor_mz"] - PROTON_MASS) * r["charge"] for r in lib.precursors.to_pylist())
    base = reference_mass.fast_mass("CPEPTIDEK")
    assert masses == pytest.approx([base + 39.994914620, base + 57.021463735], abs=2e-6)


@pytest.mark.parametrize("sequence,rule", [("ACDEK", "UNIMOD:374@C"), ("GKKG", "UNIMOD:1898@K")])
def test_unsupported_bridge_chemistry_is_rejected(tmp_path, sequence, rule):
    with pytest.raises((ValueError, NotImplementedError)):
        _build(tmp_path, sequence, rules=(ModRule.parse(rule, "variable"),))


class _DeadlineExceeded(AssertionError):
    pass


@contextmanager
def _deadline(seconds=2):
    if not hasattr(signal, "setitimer"):
        pytest.skip("bounded combinatorial regression uses POSIX setitimer")
    def expired(*_):
        raise _DeadlineExceeded("tiny count/result exceeded the combinatorial work deadline")
    prior = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


def test_large_count_is_computed_without_enumerating_assignments():
    with _deadline():
        assert count_modforms("M" * 30, variable={"M": "UNIMOD:35"}, max_variable=30) == 2**30


def test_per_modification_cap_prunes_enumeration():
    with _deadline():
        forms = enumerate_modforms("M" * 30, variable={"M": "UNIMOD:35"}, max_variable=30, max_per_mod={"UNIMOD:35": 1})
    assert len(forms) == 31


def test_tiny_budget_rejects_wide_modification_search_promptly(tmp_path):
    with _deadline(), pytest.raises(SearchSpaceBudgetError):
        _build(tmp_path, "M" * 30, rules=(ModRule.parse("Oxidation@M", "variable"),), max_variable=30, max_peptidoforms=1)


class _SyntheticClient:
    """A Koina stand-in answering every precursor with one y1+ fragment."""

    model_inputs = {"peptide_sequences": None, "precursor_charges": None, "collision_energies": None}
    model_outputs = {"intensities": None, "mz": None, "annotation": None}

    def predict(self, arrays, *, min_intensity=1e-4, mode="semi_async"):
        n = len(np.asarray(arrays["peptide_sequences"]).ravel())
        return {"intensities": np.ones((n, 1)), "mz": np.zeros((n, 1)), "annotation": np.full((n, 1), "y1+1")}


def test_prediction_roundtrip_preserves_decoy_flags_and_spans(tmp_path):
    from constellation.massspec.library.koina.api import predict_library

    lib = _build(tmp_path, decoys=(DecoySet(),))
    path = tmp_path / "library"
    save_library(lib, path, format="parquet_dir")
    predicted, _ = predict_library(
        library=load_library(path), ms2_model_name="Prosit_2020_intensity_HCD", rt_model_name=None,
        collision_energy=30.0, ms2_client=_SyntheticClient(),
    )
    before = {r["modified_sequence"]: r["is_decoy"] for r in lib.peptides.to_pylist()}
    after = {r["modified_sequence"]: r["is_decoy"] for r in predicted.peptides.to_pylist()}
    assert after == before
    for name in ("peptides", "precursors", "peptide_spans", "span_peptide", "peptide_modifications", "protein_peptide"):
        assert getattr(predicted, name).select(getattr(lib, name).column_names).equals(getattr(lib, name)), name
    assert predicted.fragments.num_rows == lib.n_precursors
    assert set(predicted.fragments.column("precursor_id").to_pylist()) == set(lib.precursors.column("precursor_id").to_pylist())
    assert predicted.metadata_extras["x.koina.precursors_not_predicted"] == 0
    # the hand-off precursors_from_library makes is still the bare grid
    assert len(precursors_from_library(path)) == lib.n_precursors


def test_attach_predictions_carries_ccs_and_keeps_unset_values_from_the_source(tmp_path):
    """attach_predictions copied RT and dropped CCS (fixed 2026-11-05)."""
    from constellation.massspec.library.library import assign_ids, attach_predictions

    lib = _build(tmp_path, "PEPTIDEKMAGICSEQR", decoys=(DecoySet(),))
    rows = lib.precursors.join(lib.peptides.select(["peptide_id", "modified_sequence"]), keys="peptide_id")
    rows = sorted(rows.to_pylist(), key=lambda r: (r["modified_sequence"], r["charge"]))
    predicted_rows = [
        {"modified_sequence": r["modified_sequence"], "charge": r["charge"], "precursor_mz": r["precursor_mz"],
         "rt_predicted": 100.0 + i, "ccs_predicted": -1.0 if i % 2 else 400.0 + i}
        for i, r in enumerate(rows)
    ]
    predicted = assign_ids(
        proteins=[], peptides=[{"modified_sequence": m} for m in {r["modified_sequence"] for r in rows}],
        precursors=predicted_rows,
    )
    merged, counts = attach_predictions(lib, predicted)
    assert counts["precursors_not_predicted"] == 0
    got = merged.precursors.join(merged.peptides.select(["peptide_id", "modified_sequence"]), keys="peptide_id")
    got = {(r["modified_sequence"], r["charge"]): r for r in got.to_pylist()}
    for want in predicted_rows:
        row = got[(want["modified_sequence"], want["charge"])]
        assert row["rt_predicted"] == want["rt_predicted"]
        # an unset predicted CCS leaves the source's (also unset here)
        assert row["ccs_predicted"] == (want["ccs_predicted"] if want["ccs_predicted"] >= 0 else -1.0)


def test_a_dlib_holds_the_targets_and_leaves_decoys_out(tmp_path):
    """The writer wrote decoys into entries with isDecoy false, so the jar
    searched them as targets and made decoys of them (fixed 2026-11-05)."""
    import sqlite3

    lib = _build(tmp_path, "PEPTIDEKMAGICSEQR", decoys=(DecoySet(),))
    peptides = lib.peptides.to_pylist()
    targets = {p["sequence"] for p in peptides if not p["is_decoy"]}
    decoys = {p["sequence"] for p in peptides if p["is_decoy"]}
    target_ids = {p["peptide_id"] for p in peptides if not p["is_decoy"]}
    n_target_precursors = sum(1 for r in lib.precursors.to_pylist() if r["peptide_id"] in target_ids)
    assert decoys and not decoys & targets

    with pytest.warns(UserWarning, match=rf"{len(decoys)} decoy peptide\(s\)"):
        save_library(lib, tmp_path / "space.dlib", format="encyclopedia.dlib")
    con = sqlite3.connect(f"file:{tmp_path / 'space.dlib'}?mode=ro", uri=True)
    entries = {r[0] for r in con.execute("select PeptideSeq from entries")}
    mapped = {r[0] for r in con.execute("select PeptideSeq from peptidetoprotein")}
    assert con.execute("select count(*) from entries").fetchone()[0] == n_target_precursors
    assert entries == targets and mapped == targets
    assert con.execute("select count(*) from peptidetoprotein where isDecoy").fetchone()[0] == 0


def test_a_dlib_of_a_library_without_decoys_warns_of_nothing(tmp_path):
    import warnings

    lib = _build(tmp_path, "PEPTIDEKMAGICSEQR")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_library(lib, tmp_path / "space.dlib", format="encyclopedia.dlib")


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_fixed_terminal_aliases_are_rejected(tmp_path, reverse):
    rules = (ModRule.parse("Acetyl@Protein N-term", "fixed"), ModRule.parse("Dimethyl@N-term", "fixed"))
    with pytest.raises(ValueError):
        _build(tmp_path, rules=rules[::-1] if reverse else rules)


def test_reverse_protein_decoys_cannot_be_spectrally_identical_to_targets(tmp_path):
    lib = _build(tmp_path, "AILA", decoys=(DecoySet("reverse-protein"),))
    rows = lib.peptides.to_pylist()
    target = {il_key(r["sequence"]) for r in rows if not r["is_decoy"]}
    decoy = {il_key(r["sequence"]) for r in rows if r["is_decoy"]}
    assert not target & decoy


def test_shuffle_collision_resolution_is_input_order_independent(tmp_path):
    def pairs(sequences):
        path = tmp_path / "order.fa"
        path.write_text("".join(f">{seq}\n{seq}\n" for seq in sequences))
        lib = build_search_space([path], _settings(decoys=(DecoySet("shuffle-interior"),)))
        rows = lib.peptides.to_pylist()
        by_id = {r["peptide_id"]: r["modified_sequence"] for r in rows}
        return {by_id[r["decoy_of"]]: r["modified_sequence"] for r in rows if r["is_decoy"]}
    assert pairs(["PADCEK", "PAECDK"]) == pairs(["PAECDK", "PADCEK"])


def test_unmatched_protein_terminus_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _build(tmp_path, protein_termini=(("TYPO", 3, None),))


def test_deuterium_aliases_do_not_double_the_label_shift(tmp_path):
    try:
        duplicate = _build(tmp_path, labels=(parse_label("P=D,2H"),))
    except ValueError:
        return
    single = _build(tmp_path, labels=(parse_label("P=2H"),))
    assert _neutral_mass(duplicate) == pytest.approx(_neutral_mass(single), abs=1e-9)


def test_contradictory_global_isotopes_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        _build(tmp_path, "MPEPTIDEK", labels=(parse_label("P=33S,34S"),))


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_peptidoform_keeps_fixed_and_variable_origins(tmp_path, reverse):
    records = [("A", "GPEPTIDEK"), ("B", "AAAKGPEPTIDEK")]
    path = tmp_path / "contexts.fa"
    path.write_text("".join(f">{acc}\n{seq}\n" for acc, seq in (records[::-1] if reverse else records)))
    lib = build_search_space([path], _settings(
        digestion=Digestion(min_length=3, missed_cleavages=0, excise_initiator_met=False),
        rules=(ModRule.parse("Acetyl@Protein N-term", "fixed"), ModRule.parse("Acetyl@N-term", "variable")),
    ))
    peptide = next(r for r in lib.peptides.to_pylist() if r["modified_sequence"] == "[UNIMOD:1]-GPEPTIDEK")
    edges = [r for r in lib.span_peptide.to_pylist() if r["peptide_id"] == peptide["peptide_id"]]
    assert len(edges) == 2
    origins = {r["origin"] for r in lib.peptide_modifications.to_pylist() if r["peptide_id"] == peptide["peptide_id"]}
    assert origins == {"fixed", "variable"}


def test_fully_labelled_envelope_only_varies_unlabelled_elements():
    _, actual = peptide_envelope(parse_proforma("<13C><15N>ACDEK"), charge=2, n_peaks=5)
    # Fully specified C/N isotopes have no natural-abundance variance.
    _, expected = isotope_envelope(Composition.from_dict({"H": 36, "O": 10, "S": 1}), n_peaks=5)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-7, rtol=1e-6)
