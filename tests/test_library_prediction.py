"""Predictions put onto a library: the contract, requests, orchestration,
assembly and JChronologer's decoders, without the jar.

JChronologer's predictors run here on a fake bundle
(``_jchronologer_fixtures.FakeBundle``: the real preprocessing files with
stand-ins for the TorchScript models), so their inputs and decoding are
tested in CI. ``tests/test_jchronologer_golden.py`` runs the real models
against the jar's own predictions where EncyclopeDIA is installed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.library.decoys import DecoySet
from constellation.massspec.library.jchronologer import models as jm
from constellation.massspec.library.library import Library, attach_predictions
from constellation.massspec.library.prediction import registry
from constellation.massspec.library.prediction.base import (
    CHARGE_OUT_OF_RANGE,
    Capabilities,
    FragmentArrays,
    Refusals,
    Requests,
    ValueArray,
    merge_refusals,
    sites_key,
)
from constellation.massspec.library.prediction.collision_energy import CollisionEnergy
from constellation.massspec.library.prediction.requests import build_requests, unlabelled_sites
from constellation.massspec.library.prediction.run import predict_onto
from constellation.massspec.library.prediction.torch_runner import run_batched
from constellation.massspec.library.search_space import (
    Digestion,
    ModRule,
    SearchSpaceSettings,
    build_search_space,
)
from constellation.massspec.peptide.ions import IonType, fragment_ladder_arrays_batch
from _jchronologer_fixtures import FakeBundle


# ── stand-in models ─────────────────────────────────────────────────


class Recorder:
    """A model that remembers its inputs and returns ``fn(*inputs)``."""

    def __init__(self, fn) -> None:
        self.fn = fn
        self.inputs: list[tuple[torch.Tensor, ...]] = []

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        self.inputs.append(tuple(x.clone() for x in inputs))
        return self.fn(*inputs)


def _requests(peptides: list[str], charges: list[int], energy: float = 30.0) -> Requests:
    sites = tuple(unlabelled_sites(p)[0] for p in peptides)
    return Requests(
        sites, np.arange(len(peptides)), np.array(charges, np.int32), np.full(len(peptides), energy, np.float32)
    )


# ── JChronologer's inputs and decoding ───────────────────────────────


def test_cartographer_decoding_follows_the_jars_layout():
    out = np.zeros((2, 174), np.float32)
    out[0, 0] = 0.5  # y1 1+
    out[0, 4] = 0.3  # b1 2+
    out[0, 9] = 0.005  # b2 1+, under the 0.01 floor
    out[0, 17] = 0.4  # b3 3+, above the precursor's charge 2
    out[0, 6 * 7 + 1] = 0.7  # y8 2+, ion number 8 = length: impossible
    out[0, 6 * 2 + 1] = np.nan  # y3 2+
    out[0, 6 * 3 + 2] = -0.2  # y4 3+
    out[1, 6 * 28 + 2] = 0.9  # y29 3+ on a 31-residue peptide at 4+
    out[1, 6 * 28 + 5] = np.float32(0.01)  # b29 3+, exactly the floor: kept
    got = jm.decode_cartographer(out, np.array([8, 31]), np.array([2, 4]))
    ions = sorted(
        zip(got.request.tolist(), [IonType(t).name for t in got.ion_type], got.ion_number.tolist(),
            got.fragment_charge.tolist(), np.float32(got.intensity).tolist())
    )
    assert ions == [
        (0, "B", 1, 2, np.float32(0.3)),
        (0, "Y", 1, 1, np.float32(0.5)),
        (1, "B", 29, 3, np.float32(0.01)),
        (1, "Y", 29, 3, np.float32(0.9)),
    ]


def test_electrician_normalisation_is_the_jars():
    out = np.array([[-1.0, 2.0, 6.0, np.nan, 0.0, 2.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]], np.float32)
    got = jm.normalize_charge_distribution(out)
    assert got.dtype == np.float32
    assert got[0].tolist() == pytest.approx([0.0, 0.2, 0.6, 0.0, 0.0, 0.2])
    assert got[0, 1] == np.float32(np.float64(np.float32(2.0)) / 10.0)
    assert not got[1].any()


def test_the_four_predictors_give_the_jars_inputs_and_outputs():
    chron = Recorder(lambda t: torch.full((t.shape[0], 1), 12.5, dtype=torch.float32))
    cart = Recorder(lambda t, z, e: torch.full((t.shape[0], 174), 0.25, dtype=torch.float32))
    elec = Recorder(lambda t: torch.tensor([[1.0, 3.0, 0.0, 0.0, 0.0, 0.0]] * t.shape[0]))
    sculpt = Recorder(lambda t, z: torch.ones((t.shape[0], 1)))
    bundle = FakeBundle({"chronologer": chron, "cartographer": cart, "electrician": elec, "sculptor": sculpt})
    requests = _requests(["[UNIMOD:28]-QPEPTIDEK", "LVNELTEFAK"], [2, 3], energy=27.3)

    rt = jm.ChronologerRetentionTime(bundle).predict(requests)
    assert rt.values.tolist() == [750.0, 750.0] and rt.extra["scale"] == jm.RT_SCALE
    assert chron.inputs[0][0].shape == (2, 52)
    assert bundle.tokenizers["chronologer"].vocabulary.decode(chron.inputs[0][0].numpy())[0] == "(ePEPTIDEK_"

    frags = jm.CartographerFragments(bundle).predict(requests)
    tokens, charge, energy = cart.inputs[0]
    assert tokens.shape == (2, 33) and charge.tolist()[0] == [0, 1, 0, 0, 0, 0] and charge.tolist()[1][2] == 1
    assert energy.dtype == torch.float32 and energy[0, 0] == np.float32(np.float64(np.float32(27.3)) / 100.0)
    # 9 residues at 2+: ion numbers 1..8, fragment charges 1..2, b and y.
    assert int((frags.request == 0).sum()) == 8 * 2 * 2
    assert int((frags.request == 1).sum()) == 9 * 3 * 2

    probability = jm.ElectricianChargeState(bundle).predict(requests)
    assert probability.values.tolist() == pytest.approx([0.75, 0.0])

    ccs = jm.SculptorCcs(bundle).predict(requests)
    pre = bundle.preprocessing("sculptor")
    want = np.float32(1.0) * np.float32(pre["train_ccs_std"]) + np.float32(pre["train_ccs_mean"])
    assert ccs.values.tolist() == [float(want)] * 2
    assert sculpt.inputs[0][1].tolist()[1][2] == 1


def test_checks_refuse_what_the_jar_refuses():
    bundle = FakeBundle({})
    requests = Requests(
        tuple(unlabelled_sites(p)[0] for p in ("PEPTIDEK", "PEPTW[UNIMOD:35]IDEK")),
        np.array([0, 0, 0, 1]),
        np.array([2, 7, 2, 2], np.int32),
        np.array([30.0, 30.0, 61.0, 30.0], np.float32),
    )
    cart = jm.CartographerFragments(bundle).check(requests)
    assert cart.reasons == (None, CHARGE_OUT_OF_RANGE, "collision_energy_out_of_range", None)
    chron = jm.ChronologerRetentionTime(bundle).check(requests)
    assert chron.reasons == (None, None, None, "unsupported_modification")
    no_energy = replace(requests, collision_energy=np.full(4, np.nan, np.float32))
    assert "no collision energy" in jm.CartographerFragments(bundle).check(no_energy).details[0]


# ── the runner ───────────────────────────────────────────────────────


def test_the_runner_does_not_depend_on_the_thread_count():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8)).eval()
    x = np.random.default_rng(0).standard_normal((1000, 16)).astype(np.float32)
    one = run_batched(model, [x], batch_size=64, threads=1)
    four = run_batched(model, [x], batch_size=64, threads=4)
    assert one.shape == (1000, 8) and np.array_equal(one, four)
    with pytest.raises(ValueError, match="no inputs"):
        run_batched(model, [x[:0]])


# ── requests ─────────────────────────────────────────────────────────


def _search_space(tmp_path: Path, proteins: dict[str, str], **kwargs) -> Library:
    fasta = tmp_path / "proteins.fa"
    fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in proteins.items()))
    settings = SearchSpaceSettings(
        digestion=Digestion(protease="Trypsin", missed_cleavages=0, min_length=6, max_length=30, excise_initiator_met=False),
        charges=(2, 3), min_mz=0, max_mz=1e6, decoys=(DecoySet(),),
    )
    return build_search_space([fasta], replace(settings, **kwargs))


PROTEINS = {"P1": "MPEPTIDEKLVNELTEFAKSAMPLERHLVDEPQNLIK", "P2": "AEFVEVTKYLYEIARQTALVELLK"}


def test_labels_come_off_and_heavy_and_light_share_requests(tmp_path):
    lib = _search_space(tmp_path, PROTEINS, labels=(("P1", ()), ("P1", ("13C", "15N"))))
    plan = build_requests(lib, CollisionEnergy.scaled_from_reference(30.0))
    texts = lib.peptides.column("modified_sequence").to_pylist()
    assert any(t.startswith("<13C><15N>") for t in texts)
    assert plan.labelled_peptides == sum(t.startswith("<") for t in texts)
    # A heavy and a light precursor of one peptide ask the same request.
    by_text = dict(zip(lib.peptides.column("peptide_id").to_pylist(), texts))
    asked: dict[tuple[str, int], set[int]] = {}
    for pid, z, r in zip(plan.peptide_id.tolist(), lib.precursors.column("charge").to_pylist(), plan.request.tolist()):
        asked.setdefault((by_text[pid].replace("<13C><15N>", ""), z), set()).add(r)
    assert all(len(v) == 1 for v in asked.values())
    keys = [(sites_key(plan.requests.peptides[p]), int(z)) for p, z in zip(plan.requests.peptide, plan.requests.charge)]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert plan.requests.collision_energy.tolist() == [
        float(CollisionEnergy.scaled_from_reference(30.0).resolve(np.array([z]))[0]) for z in plan.requests.charge
    ]


def test_silac_labels_come_off_and_chemical_modifications_stay():
    sites, labelled = unlabelled_sites("PEPTK[UNIMOD:259][UNIMOD:1]IDEK[UNIMOD:259]")
    assert labelled and sites.residue_mods[4] == ("UNIMOD:1",) and sites.residue_mods[-1] == ()
    sites, labelled = unlabelled_sites("[UNIMOD:737]-PEPTK[UNIMOD:737]IDEK")
    assert not labelled and sites.n_term == ("UNIMOD:737",)


def test_a_target_and_its_decoys_at_one_charge_are_one_group(tmp_path):
    lib = _search_space(tmp_path, PROTEINS)
    plan = build_requests(lib)
    peptides = {r["peptide_id"]: r for r in lib.peptides.to_pylist()}
    charges = lib.precursors.column("charge").to_pylist()
    for row, pid in enumerate(plan.peptide_id.tolist()):
        target = peptides[pid]["decoy_of"] if peptides[pid]["is_decoy"] else pid
        same = [
            r for r, p in enumerate(plan.peptide_id.tolist())
            if (peptides[p]["decoy_of"] if peptides[p]["is_decoy"] else p) == target and charges[r] == charges[row]
        ]
        assert set(np.flatnonzero(plan.group == plan.group[row]).tolist()) == set(same)
    assert plan.is_decoy.any() and (~plan.is_decoy).any()


# ── the orchestration ────────────────────────────────────────────────


class FakePredictor:
    """Predicts a property from a peptide's residues; refuses peptides whose
    sites key is in ``refuse``."""

    def __init__(self, prop: str, refuse: set[str] = frozenset(), nan: bool = False) -> None:
        self.name = f"fake:{prop}"
        self.capabilities = Capabilities(
            property=prop,
            needs_charge=prop in ("fragments", "ccs"),
            ion_types=(IonType.B, IonType.Y) if prop == "fragments" else (),
            max_fragment_charge=2 if prop == "fragments" else 0,
            rt_scale="fake_seconds" if prop == "retention_time" else None,
        )
        self.refuse = set(refuse)
        self.nan = nan
        self.asked: list[Requests] = []

    def identity(self) -> dict:
        return {"backend": "fake", "model": self.capabilities.property}

    def check(self, requests: Requests) -> Refusals:
        keys = [sites_key(requests.peptides[p]) for p in requests.peptide.tolist()]
        return merge_refusals(len(requests), {i: ("unsupported_modification", k) for i, k in enumerate(keys) if k in self.refuse})

    def predict(self, requests: Requests):
        self.asked.append(requests)
        length = np.array([len(requests.peptides[p].residues) for p in requests.peptide])
        if self.capabilities.property == "fragments":
            n = len(requests)
            return FragmentArrays(
                request=np.repeat(np.arange(n), 2),
                ion_type=np.tile(np.array([IonType.B, IonType.Y], np.int8), n),
                ion_number=np.tile(np.array([2, 1], np.int32), n),
                fragment_charge=np.ones(2 * n, np.int32),
                intensity=np.tile([0.5, 1.0], n) * (requests.charge.repeat(2) / 10),
            )
        values = np.full(len(requests), np.nan) if self.nan else length * 10.0 + requests.charge
        return ValueArray(values)


def _predictors(**kw) -> dict:
    return {p: FakePredictor(p, **kw) for p in ("fragments", "retention_time", "ccs", "charge_state")}


def _decoy_and_target_of(lib: Library) -> tuple[str, str]:
    peptides = lib.peptides.to_pylist()
    decoy = next(p for p in peptides if p["is_decoy"])
    target = next(p for p in peptides if p["peptide_id"] == decoy["decoy_of"])
    return target["modified_sequence"], decoy["modified_sequence"]


@pytest.mark.parametrize("which", ["target", "decoy"])
def test_a_refused_precursor_takes_its_partner_with_it(tmp_path, which):
    lib = _search_space(tmp_path, PROTEINS)
    target, decoy = _decoy_and_target_of(lib)
    refused = target if which == "target" else decoy
    predictors = _predictors()
    predictors["ccs"] = FakePredictor("ccs", refuse={refused})
    out, report = predict_onto(lib, predictors)
    left = set(out.peptides.column("modified_sequence").to_pylist())
    assert target not in left and decoy not in left
    assert report.refused == {"ccs": {"unsupported_modification": 2}}  # the peptide at 2+ and 3+
    assert report.dropped_with_partner == 2 and report.dropped == 4
    assert report.precursors_kept == lib.precursors.num_rows - 4
    assert out.precursors.num_rows == report.precursors_kept


def test_predictions_land_on_the_precursors_that_asked(tmp_path):
    lib = _search_space(tmp_path, PROTEINS)
    out, report = predict_onto(lib, _predictors(), collision_energy=CollisionEnergy.uniform(30.0))
    rows = out.precursors.join(out.peptides.select(["peptide_id", "sequence"]), keys="peptide_id").to_pylist()
    for row in rows:
        want = len(row["sequence"]) * 10.0 + row["charge"]
        assert row["rt_predicted"] == want and row["ccs_predicted"] == want and row["charge_probability"] == want
    assert report.fragments == out.fragments.num_rows == 2 * out.precursors.num_rows
    assert report.requests_predicted == report.requests and report.dropped == 0
    meta = out.metadata_extras
    assert meta["x.massspec.prediction.predictors"]["fragments"]["backend"] == "fake"
    assert meta["x.massspec.prediction.collision_energy"]["origin"] == "uniform"
    assert meta["x.massspec.prediction.rt_scale"] == "fake_seconds"
    # Decoy pairing and provenance survive.
    assert out.peptides.column("decoy_of").equals(lib.peptides.column("decoy_of"))
    assert out.span_peptide.num_rows == lib.span_peptide.num_rows


def test_an_unpredicted_value_leaves_the_sources(tmp_path):
    lib = _search_space(tmp_path, PROTEINS)
    out, _ = predict_onto(lib, {"ccs": FakePredictor("ccs", nan=True), "charge_state": FakePredictor("charge_state", nan=True)})
    assert set(out.precursors.column("ccs_predicted").to_pylist()) == {-1.0}
    assert out.precursors.column("charge_probability").null_count == out.precursors.num_rows
    assert out.fragments.num_rows == lib.fragments.num_rows


def test_a_labelled_precursor_gets_its_own_mz_and_its_light_forms_intensities(tmp_path):
    lib = _search_space(tmp_path, PROTEINS, labels=(("P1", ()), ("P1", ("13C", "15N"))), decoys=())
    predictors = {"fragments": FakePredictor("fragments")}
    out, report = predict_onto(lib, predictors)
    assert len(predictors["fragments"].asked[0]) < out.precursors.num_rows  # heavy and light share requests
    precursors = out.precursors.select(["precursor_id", "peptide_id", "charge"]).rename_columns(
        ["precursor_id", "peptide_id", "precursor_charge"]
    )
    frags = out.fragments.join(precursors, keys="precursor_id")
    frags = frags.join(out.peptides.select(["peptide_id", "modified_sequence"]), keys="peptide_id").to_pylist()
    for f in frags:
        ladder = fragment_ladder_arrays_batch(
            [parse_proforma(f["modified_sequence"])], ion_types=(IonType.B, IonType.Y), max_fragment_charge=2
        )
        rows = np.flatnonzero(
            (ladder.ion_type == f["ion_type"]) & (ladder.position == f["position"]) & (ladder.charge == f["charge"])
        )
        assert f["mz_theoretical"] == ladder.mz[rows[0]]

    def by_light_form(heavy: bool) -> dict:
        return {
            (f["modified_sequence"].removeprefix("<13C><15N>"), f["precursor_charge"], f["ion_type"], f["position"]):
                f["intensity_predicted"]
            for f in frags
            if f["modified_sequence"].startswith("<13C><15N>") == heavy
        }

    heavy, light = by_light_form(True), by_light_form(False)
    assert heavy and all(light[k] == v for k, v in heavy.items())
    assert report.caveats and "deuterium" in report.caveats[0]


def test_the_result_does_not_depend_on_the_librarys_row_order(tmp_path):
    lib = _search_space(tmp_path, PROTEINS)
    order = np.random.default_rng(1).permutation(lib.precursors.num_rows)
    shuffled = replace(lib, precursors=lib.precursors.take(pa.array(order)))
    a, _ = predict_onto(lib, _predictors(), collision_energy=CollisionEnergy.uniform(30.0))
    b, _ = predict_onto(shuffled, _predictors(), collision_energy=CollisionEnergy.uniform(30.0))

    def canonical(t: pa.Table, keys: list[str]) -> pa.Table:
        return t.take(pc.sort_indices(t, sort_keys=[(k, "ascending") for k in keys]))

    assert canonical(a.precursors, ["precursor_id"]).equals(canonical(b.precursors, ["precursor_id"]))
    keys = ["precursor_id", "ion_type", "position", "charge"]
    assert canonical(a.fragments, keys).equals(canonical(b.fragments, keys))


def test_a_peptide_with_no_sites_reading_is_refused_as_input(tmp_path):
    lib = _search_space(tmp_path, {"P": "PEPTIDEKLVNELTEFAK"}, decoys=())
    peptides = lib.peptides.set_column(
        lib.peptides.schema.get_field_index("modified_sequence"),
        "modified_sequence",
        pa.array([s.replace("PEPTIDEK", "PEPTIDE[+15.995]K") for s in lib.peptides.column("modified_sequence").to_pylist()]),
    )
    out, report = predict_onto(replace(lib, peptides=peptides), {"retention_time": FakePredictor("retention_time")})
    assert report.refused == {"input": {"unsupported_modification": 2}}
    assert out.precursors.num_rows == lib.precursors.num_rows - 2


def test_predictors_must_match_their_property_and_have_an_energy():
    with pytest.raises(ValueError, match="predicts ccs, not fragments"):
        predict_onto(None, {"fragments": FakePredictor("ccs")})
    bundle = FakeBundle({})
    with pytest.raises(ValueError, match="needs a collision energy"):
        predict_onto(None, {"fragments": jm.CartographerFragments(bundle)})


def test_attach_predictions_carries_charge_probability(tmp_path):
    lib = _search_space(tmp_path, PROTEINS, decoys=())
    predicted, _ = predict_onto(lib, {"charge_state": FakePredictor("charge_state")})
    merged, counts = attach_predictions(lib, predicted)
    assert merged.precursors.column("charge_probability").equals(predicted.precursors.column("charge_probability"))
    assert counts["precursors_not_predicted"] == 0


def test_the_registry_names_and_refuses():
    assert registry.property_of("jchronologer:sculptor") == "ccs"
    with pytest.raises(ValueError, match="unknown predictor"):
        registry.predictor("jchronologer:scout")
    with pytest.raises(ValueError, match="unknown backend"):
        registry.default_predictors("nope")
    bundle = FakeBundle({})
    got = registry.default_predictors("jchronologer", ("fragments", "ccs"), bundle=bundle, threads=2)
    assert set(got) == {"fragments", "ccs"} and got["fragments"].threads == 2
    assert got["ccs"].name == "jchronologer:sculptor"
