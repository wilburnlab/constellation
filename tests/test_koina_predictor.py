"""Koina's models behind the prediction contract, and models mixed per property.

Offline: the fragment model replays captured responses
(``test_koina_offline.FakeKoinaClient``) and the retention-time model is a
small recording stand-in, so nothing here reaches the network.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from _jchronologer_fixtures import FakeBundle
from constellation.massspec.library import load_library, save_library
from constellation.massspec.library.decoys import DecoySet
from constellation.massspec.library.koina import KoinaInputError
from constellation.massspec.library.koina import client as koina_client
from constellation.massspec.library.koina.api import predict_library
from constellation.massspec.library.koina.predictor import KoinaFragments, KoinaRetentionTime, koina_sequence
from constellation.massspec.library.prediction import registry
from constellation.massspec.library.prediction.base import Requests
from constellation.massspec.library.prediction.collision_energy import CollisionEnergy
from constellation.massspec.library.prediction.requests import unlabelled_sites
from constellation.massspec.library.prediction.run import predict_onto
from constellation.massspec.library.search_space import Digestion, SearchSpaceSettings, build_search_space
from constellation.massspec.peptide.ions import IonType
from test_koina_offline import HCD, FakeKoinaClient, fixture_specs


class RtClient:
    """Chronologer_RT stand-in: rt (minutes) = residues / 10; remembers what was sent."""

    model_inputs = {"peptide_sequences": None}
    model_outputs = {"rt": None}

    def __init__(self, *_: object, **__: object) -> None:
        self.sent: list[str] = []

    def predict(self, arrays, **_):
        sequences = [str(s) for s in np.asarray(arrays["peptide_sequences"]).ravel().tolist()]
        self.sent += sequences
        return {"rt": np.array([[len(unlabelled_sites(s)[0].residues) / 10] for s in sequences])}


class OneIonClient:
    """A fragment model answering every request with y1+ and b2+ (and their m/z unknown)."""

    model_inputs = {"peptide_sequences": None, "precursor_charges": None, "collision_energies": None}
    model_outputs = {"intensities": None, "mz": None, "annotation": None}

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, arrays, *, min_intensity=1e-4, **_):
        self.calls += 1
        n = len(np.asarray(arrays["peptide_sequences"]).ravel())
        return {
            "intensities": np.tile([1.0, 0.5, -1.0], (n, 1)),
            "mz": np.zeros((n, 3)),
            "annotation": np.tile(np.array(["y1+1", "b2+1", "y3+1"], dtype=object), (n, 1)),
        }


class LossClient(OneIonClient):
    """A fragment model whose answers include neutral losses."""

    def predict(self, arrays, *, min_intensity=1e-4, **_):
        self.calls += 1
        n = len(np.asarray(arrays["peptide_sequences"]).ravel())
        return {
            "intensities": np.tile([1.0, 0.5, 0.25], (n, 1)),
            "mz": np.zeros((n, 3)),
            "annotation": np.tile(np.array(["y1+1", "y3-H2O+1", "b2-NH3+1"], dtype=object), (n, 1)),
        }


def _requests(specs) -> Requests:
    peptides = sorted({s.modified_sequence for s in specs})
    index = {p: i for i, p in enumerate(peptides)}
    return Requests(
        tuple(unlabelled_sites(p)[0] for p in peptides),
        np.array([index[s.modified_sequence] for s in specs]),
        np.array([s.charge for s in specs], np.int32),
        CollisionEnergy.uniform(30.0).resolve(np.array([s.charge for s in specs])),
    )


def _search_space(tmp_path: Path, **kwargs):
    fasta = tmp_path / "proteins.fa"
    fasta.write_text(">P1\nPEPTIDEKLVNELTEFAKSAMPLERCMEPEPTIDER\n>P2\nAEFVEVTKYLYEIARQTALVELLK\n")
    settings = SearchSpaceSettings(
        digestion=Digestion(missed_cleavages=0, excise_initiator_met=False),
        charges=(2, 3), min_mz=0, max_mz=1e6, decoys=(DecoySet(),),
    )
    return build_search_space([fasta], replace(settings, **kwargs))


# ── the predictors ───────────────────────────────────────────────────


def test_fragments_decode_as_the_old_assembly_did():
    specs = fixture_specs(HCD)
    requests = _requests(specs)
    got = KoinaFragments(HCD, client=FakeKoinaClient(HCD)).predict(requests)
    old, _ = predict_library(specs=specs, ms2_client=FakeKoinaClient(HCD), rt_model_name=None, collision_energy=30.0)
    typed = old.fragments.filter(pc.is_valid(old.fragments.column("ion_type")))
    assert len(got) == typed.num_rows > 0
    # Same intensities, ion by ion (ion number here, bond position there).
    assert sorted(np.round(got.intensity, 9).tolist()) == sorted(np.round(typed.column("intensity_predicted").to_numpy(), 9).tolist())
    assert got.mz_reported is not None and np.all(got.mz_reported > 0)


def test_checks_are_local_and_name_the_reason():
    client = FakeKoinaClient(HCD)
    model = KoinaFragments(HCD, client=client)
    peptides = ["[UNIMOD:1]-PEPTIDEK", "PEPS[UNIMOD:21]IDEK", "A" * 31, "PEPTIDEK", "PEPTIDEK", "PEPTUDEK"]
    requests = Requests(
        tuple(unlabelled_sites(p)[0] for p in peptides),
        np.arange(6), np.array([2, 2, 2, 7, 2, 2], np.int32),
        np.array([30, 30, 30, 30, np.nan, 30], np.float32),
    )
    refusals = model.check(requests)
    assert refusals.reasons == (
        "unsupported_modification", "unsupported_modification", "length_out_of_range",
        "charge_out_of_range", "collision_energy_out_of_range", "unknown_residue",
    )
    assert "UNIMOD:21" in refusals.details[1]
    assert not hasattr(client, "last_request")  # no request was sent


def test_retention_time_is_sent_in_koinas_dialect():
    client = RtClient()
    model = KoinaRetentionTime("Chronologer_RT", client=client)
    requests = Requests(
        tuple(unlabelled_sites(p)[0] for p in ("[UNIMOD:28]-QPEPC[UNIMOD:4]K", "<13C>LVNELTEFAK")),
        np.array([0, 0, 1]), np.array([2, 3, 2], np.int32), np.full(3, np.nan, np.float32),
    )
    values = model.predict(requests).values
    # Pyro-Glu on its residue, as Koina's Chronologer reads it.
    assert client.sent == ["Q[UNIMOD:28]PEPC[UNIMOD:4]K", "LVNELTEFAK"]
    assert values.tolist() == pytest.approx([6 / 10 * 60, 6 / 10 * 60, 60.0])
    assert model.capabilities.rt_scale == "koina:Chronologer_RT:minutes"
    assert koina_sequence(unlabelled_sites("[UNIMOD:1]-M[UNIMOD:35]K")[0]) == "[UNIMOD:1]-M[UNIMOD:35]K"


# ── a library through Koina ──────────────────────────────────────────


def test_a_labelled_search_space_goes_through_koina(tmp_path):
    lib = _search_space(tmp_path, labels=(("P1", ()), ("P1", ("13C", "15N"))))
    client = OneIonClient()
    out, stats = predict_library(library=lib, ms2_client=client, rt_model_name=None, collision_energy=30.0)
    # Labelled precursors are predicted (as their light form) rather than refused.
    assert out.precursors.num_rows == lib.precursors.num_rows
    assert out.metadata_extras["x.koina.precursors_not_predicted"] == 0
    peptides = dict(zip(out.peptides.column("peptide_id").to_pylist(), out.peptides.column("modified_sequence").to_pylist()))
    frags = out.fragments.join(out.precursors.select(["precursor_id", "peptide_id"]), keys="precursor_id").to_pylist()
    heavy = [f for f in frags if peptides[f["peptide_id"]].startswith("<13C><15N>")]
    light = {(peptides[f["peptide_id"]], f["ion_type"]): f["mz_theoretical"] for f in frags}
    assert heavy and all(f["mz_theoretical"] > light[(peptides[f["peptide_id"]][10:], f["ion_type"])] for f in heavy
                         if (peptides[f["peptide_id"]][10:], f["ion_type"]) in light)
    assert stats.n_fragments == out.fragments.num_rows == 2 * out.precursors.num_rows


def test_fragments_with_neutral_losses_are_assembled(tmp_path):
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.peptide.ions import fragment_ladder_arrays_batch
    from constellation.massspec.peptide.neutral_losses import LOSS_REGISTRY

    lib = _search_space(tmp_path, labels=(("P1", ()), ("P1", ("13C", "15N"))))
    out, stats = predict_library(library=lib, ms2_client=LossClient(), rt_model_name=None, collision_energy=30.0)
    assert out.precursors.num_rows == lib.precursors.num_rows
    frags = out.fragments.join(out.precursors.select(["precursor_id", "peptide_id"]), keys="precursor_id")
    frags = frags.join(out.peptides.select(["peptide_id", "modified_sequence"]), keys="peptide_id").to_pylist()
    assert {f["loss_id"] for f in frags} == {None, "H2O", "NH3"}
    shift = {"H2O": 0.0, "NH3": 0.997034886}  # what a 13C/15N chain adds to the loss
    for f in frags:
        if f["loss_id"] is None:
            continue
        ladder = fragment_ladder_arrays_batch([parse_proforma(f["modified_sequence"])], max_fragment_charge=1)
        intact = ladder.mz[(ladder.ion_type == f["ion_type"]) & (ladder.position == f["position"])][0]
        heavy = f["modified_sequence"].startswith("<13C><15N>")
        lost = LOSS_REGISTRY.get(f["loss_id"]).delta_mass + (shift[f["loss_id"]] if heavy else 0.0)
        assert f["mz_theoretical"] == pytest.approx(intact - lost, abs=1e-6)
    assert stats.n_fragments == out.fragments.num_rows == 3 * out.precursors.num_rows


def test_error_mode_refuses_before_any_request_and_skip_drops_pairs(tmp_path):
    lib = _search_space(tmp_path, rules=())
    sequences = lib.peptides.column("modified_sequence").to_pylist()
    phospho = [s.replace("S", "S[UNIMOD:21]", 1) if "S" in s else s for s in sequences]
    column = lib.peptides.schema.get_field_index("modified_sequence")
    lib = replace(lib, peptides=lib.peptides.set_column(column, "modified_sequence", pa.array(phospho)))
    client = OneIonClient()
    with pytest.raises(KoinaInputError, match="UNIMOD:21"):
        predict_library(library=lib, ms2_client=client, rt_model_name=None, collision_energy=30.0)
    assert client.calls == 0
    out, _ = predict_library(library=lib, ms2_client=client, rt_model_name=None, collision_energy=30.0, on_unsupported="skip")
    left = out.peptides.column("modified_sequence").to_pylist()
    assert left and not any("UNIMOD:21" in s for s in left)
    # Each phosphopeptide's partner went with it.
    # Each phosphopeptide's precursors (2+ and 3+) went, and its partner's with them.
    assert out.metadata_extras["x.koina.precursors_not_predicted"] == 2 * 2 * sum(
        "UNIMOD:21" in s for s, d in zip(phospho, lib.peptides.column("is_decoy").to_pylist()) if not d
    )


def test_models_mix_per_property(tmp_path):
    lib = _search_space(tmp_path)
    bundle = FakeBundle()
    predictors = {
        "fragments": registry.predictor("jchronologer:cartographer", bundle=bundle),
        "retention_time": registry.predictor("koina:Chronologer_RT", client=RtClient()),
    }
    out, report = predict_onto(lib, predictors, collision_energy=CollisionEnergy.scaled_from_reference(30.0))
    assert report.predictors["fragments"]["backend"] == "jchronologer"
    assert report.predictors["retention_time"] == {"backend": "koina", "model": "Chronologer_RT", "server": koina_client.resolve_server(), "scale": "minutes"}
    rows = out.precursors.join(out.peptides.select(["peptide_id", "sequence"]), keys="peptide_id").to_pylist()
    assert all(r["rt_predicted"] == pytest.approx(len(r["sequence"]) / 10 * 60) for r in rows)
    assert out.fragments.num_rows > 0 and set(out.fragments.column("ion_type").to_pylist()) <= {int(IonType.B), int(IonType.Y)}


def test_the_registry_builds_koina_models_by_name():
    assert isinstance(registry.predictor("koina:Prosit_2020_intensity_HCD", client=FakeKoinaClient(HCD)), KoinaFragments)
    assert registry.property_of("koina:Chronologer_RT") == "retention_time"
    with pytest.raises(ValueError, match="predicts retention_time, not fragments"):
        registry.predictor("koina:Chronologer_RT", property="fragments", client=RtClient())
    with pytest.raises(ValueError, match="say which property"):
        registry.predictor("koina:SomeModelAddedLater", client=RtClient())
    with pytest.warns(UserWarning, match="no local overlay"):
        model = registry.predictor("koina:SomeModelAddedLater", property="retention_time", client=RtClient())
    assert isinstance(model, KoinaRetentionTime)


# ── the CLI ──────────────────────────────────────────────────────────


@pytest.fixture
def offline(monkeypatch):
    import constellation.massspec.library.jchronologer as package

    monkeypatch.setattr(package, "locate_bundle", lambda jar=None: FakeBundle())
    monkeypatch.setattr(koina_client, "_CLIENT_FACTORY", lambda model, **_: RtClient() if "RT" in model else FakeKoinaClient(model))


def _run(*argv: str) -> int:
    from constellation.cli.__main__ import _build_parser

    args = _build_parser().parse_args(["massspec", "predict-library", *argv])
    return args.func(args)


def test_the_cli_mixes_models(offline, tmp_path):
    import json

    path = tmp_path / "ss"
    save_library(_search_space(tmp_path), path, format="parquet_dir")
    out = tmp_path / "out"
    assert _run("--backend", "jchronologer", "--from-library", str(path), "--output-dir", str(out), "--no-progress",
                "--rt-model", "koina:Chronologer_RT", "--ccs-model", "none") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    predictors = manifest["prediction"]["predictors"]
    assert predictors["retention_time"]["backend"] == "koina" and predictors["fragments"]["backend"] == "jchronologer"
    assert "ccs" not in predictors
    library = load_library(out / "library_pqdir")
    assert set(library.precursors.column("ccs_predicted").to_pylist()) == {-1.0}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--ms2-model", "Prosit_2020_intensity_HCD"], "backend:model"),
        (["--koina-url", "localhost:1"], "no koina:<model> is chosen"),
        (["--no-ccs", "--ccs-model", "jchronologer:sculptor"], "contradict"),
        (["--on-unsupported-mod", "skip"], "--on-unsupported-mod"),
    ],
)
def test_the_cli_refuses_unclear_model_choices(offline, tmp_path, capsys, argv, message):
    assert _run("--backend", "jchronologer", "--from-library", str(tmp_path), "--output-dir", str(tmp_path / "o"), *argv) == 1
    assert message in capsys.readouterr().err


def test_the_koina_backend_replaces_only_its_own_export(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(koina_client, "_CLIENT_FACTORY", lambda model, **_: OneIonClient())
    path = tmp_path / "ss"
    save_library(_search_space(tmp_path), path, format="parquet_dir")
    out, dlib = tmp_path / "out", tmp_path / "lib.dlib"
    base = ("--backend", "koina", "--from-library", str(path), "--output-dir", str(out), "--no-progress",
            "--rt-model", "none")
    assert _run(*base, "--output-dlib", str(dlib)) == 0
    # Other settings, resumed: its own export is replaced (it used to stop at the end on FileExistsError).
    assert _run(*base, "--output-dlib", str(dlib), "--collision-energy", "25", "--resume") == 0
    assert load_library(dlib).precursors.num_rows > 0
    # A file it did not write is refused before anything is predicted, and left alone.
    foreign = tmp_path / "foreign.dlib"
    foreign.write_bytes(b"not ours")
    key = (out / "_SUCCESS").read_text()
    assert _run(*base, "--output-dlib", str(foreign), "--collision-energy", "27", "--resume") == 1
    assert "was not exported from" in capsys.readouterr().err
    assert foreign.read_bytes() == b"not ours" and (out / "_SUCCESS").read_text() == key
