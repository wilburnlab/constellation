"""Tier A — the whole Koina stack, offline and without koinapy.

Every test here replays the captured fixtures under ``tests/data/koina/``
through a fake client implementing the same ``PredictClient`` Protocol as
``KoinaClient``, so the assembly, translation and validation logic is
covered with no network and no ``[ms]`` extra. The live regression checks
live in ``test_koina_live.py`` behind an env gate.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.library.digest import PrecursorSpec
from constellation.massspec.library.koina import (
    KoinaInputError,
    KoinaModSeqError,
    adjust_nce,
    assemble_library,
    format_koina_modseq,
    ms2_model,
    parse_koina_annotation,
    parse_koina_modseq,
    rt_model,
)
from constellation.massspec.library.koina.client import validate_columns

FIXTURES = Path(__file__).parent / "data" / "koina"

HCD = "Prosit_2020_intensity_HCD"
CID = "Prosit_2020_intensity_CID"


# ── fixture replay ────────────────────────────────────────────────────


def load_fixture(model: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / f"{model}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


class FakeKoinaClient:
    """Replays a captured response; satisfies the PredictClient Protocol."""

    def __init__(self, model: str, **_: object) -> None:
        self.model = model
        self._data = load_fixture(model)
        self._inputs = {
            k[len("in_") :]: ((-1, 1), "BYTES")
            for k in self._data
            if k.startswith("in_")
        }
        self._outputs = {
            k[len("out_") :]: "FP32" for k in self._data if k.startswith("out_")
        }

    @property
    def model_inputs(self):
        return self._inputs

    @property
    def model_outputs(self):
        return self._outputs

    def predict(self, arrays, *, min_intensity=1e-4, mode="semi_async"):
        validate_columns(arrays.keys(), self._inputs, model=self.model)
        return {k[len("out_") :]: v for k, v in self._data.items() if k.startswith("out_")}


def fixture_specs(model: str) -> list[PrecursorSpec]:
    """The precursor grid the fixture was captured from, in order."""
    data = load_fixture(model)
    seqs = [
        s.decode() if isinstance(s, bytes) else str(s)
        for s in data["in_peptide_sequences"].ravel().tolist()
    ]
    charges = data["in_precursor_charges"].ravel().tolist()
    return [
        PrecursorSpec(
            modified_sequence=s,
            sequence=parse_proforma(s).sequence,
            charge=int(z),
            precursor_mz=500.0,
            proteins=("P00000",),
        )
        for s, z in zip(seqs, charges, strict=True)
    ]


# ── annotation grammar ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("token", "seq_len", "expected"),
    [
        (b"y1+1", 8, (4, 6, 1, None)),
        (b"b1+1", 8, (1, 0, 1, None)),
        (b"b12+2", 20, (1, 11, 2, None)),
        (b"y3-H2O+1", 8, (4, 4, 1, "H2O")),
        ("y1+1", 8, (4, 6, 1, None)),  # str accepted as well as bytes
        (b"y8+1", 8, None),  # off the end of an 8-mer's 7 bonds
        (b"y2-XX+1", 8, None),  # unregistered loss
        (b"garbage", 8, None),
        (b"", 8, None),
    ],
)
def test_parse_koina_annotation(token, seq_len, expected):
    assert parse_koina_annotation(token, seq_len) == expected


def test_position_convention_is_bond_indexed():
    """Prosit emits 1-indexed Roepstorff; we store 0-indexed bond positions.

    For PEPTIDEK (L=8, 7 bonds) the two ion series must run in opposite
    directions — this is the check that catches an inverted conversion.
    """
    b_positions = [parse_koina_annotation(f"b{i}+1", 8)[1] for i in range(1, 8)]
    y_positions = [parse_koina_annotation(f"y{i}+1", 8)[1] for i in range(1, 8)]
    assert b_positions == [0, 1, 2, 3, 4, 5, 6]
    assert y_positions == [6, 5, 4, 3, 2, 1, 0]


def test_every_fixture_annotation_parses():
    """All 174 grid tokens are real b/y ions — no silent partial-ID rows."""
    data = load_fixture(HCD)
    tokens = {t for row in data["out_annotation"] for t in row}
    assert len(tokens) == 174
    for token in tokens:
        assert parse_koina_annotation(token, 30) is not None


# ── modseq translation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "modseq",
    ["PEPTIDEK", "YIC[UNIMOD:4]DNQDTISSK", "PEPTC[UNIMOD:4]IDEM[UNIMOD:35]K"],
)
def test_modseq_roundtrip(modseq):
    assert format_koina_modseq(parse_koina_modseq(modseq)) == modseq


def test_terminal_mods_rejected():
    """Measured live: Koina's Prosit models reject every N-term form."""
    with pytest.raises(KoinaModSeqError, match="terminal modification"):
        format_koina_modseq(parse_proforma("[UNIMOD:1]-PEPTIDEK"))


def test_unsupported_mod_named_in_error():
    with pytest.raises(KoinaModSeqError, match="UNIMOD:21"):
        format_koina_modseq(
            parse_proforma("PEPTY[UNIMOD:21]DEK"),
            supported={"UNIMOD:4", "UNIMOD:35"},
        )


def test_mass_delta_mod_rejected():
    with pytest.raises(KoinaModSeqError, match="no CV accession"):
        format_koina_modseq(parse_proforma("PEPT[+79.9663]IDEK"))


# ── input validation ──────────────────────────────────────────────────


def test_missing_declared_input_errors():
    with pytest.raises(KoinaInputError, match="precursor_charges"):
        validate_columns(
            ["peptide_sequences"],
            {"peptide_sequences": None, "precursor_charges": None},
        )


def test_undeclared_extra_input_warns_not_silently_dropped():
    """The CID-at-six-energies trap: koinapy would drop this silently."""
    with pytest.warns(UserWarning, match="collision_energies"):
        validate_columns(
            ["peptide_sequences", "precursor_charges", "collision_energies"],
            {"peptide_sequences": None, "precursor_charges": None},
        )


def test_cid_fixture_declares_no_collision_energy():
    assert "collision_energies" not in FakeKoinaClient(CID).model_inputs
    assert "collision_energies" in FakeKoinaClient(HCD).model_inputs


# ── NCE ───────────────────────────────────────────────────────────────


def test_adjust_nce_matches_searle_table():
    charges = np.array([[1], [2], [3], [4], [5]])
    got = adjust_nce(30.0, charges).ravel()
    assert got == pytest.approx([30.0, 27.0, 25.5, 24.0, 22.5])


def test_adjust_nce_clamps_above_table():
    assert adjust_nce(30.0, np.array([[9]])).ravel()[0] == pytest.approx(22.5)


def test_adjust_nce_disabled_is_passthrough():
    got = adjust_nce(30.0, np.array([[2], [3]]), enabled=False).ravel()
    assert got == pytest.approx([30.0, 30.0])


# ── assembly ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", [HCD, CID])
def test_assemble_library_from_fixture(model):
    specs = fixture_specs(model)
    response = {
        k[len("out_") :]: v for k, v in load_fixture(model).items() if k.startswith("out_")
    }
    lib, stats = assemble_library(specs, response, ms2_model=ms2_model(model))

    # Library's __post_init__ enforces PK uniqueness + FK closure, so
    # construction succeeding is itself most of the assertion.
    assert lib.precursors.num_rows == len(specs)
    assert lib.peptides.num_rows == len({s.modified_sequence for s in specs})
    assert stats.n_fragments > 0
    assert lib.fragments.num_rows == stats.n_fragments


def test_mz_crosscheck_agrees_with_local_ladder():
    """Our ladder vs Koina's returned m/z — the translation-integrity check.

    A disagreement means we described a different molecule to the server
    than the one we think we did.
    """
    specs = fixture_specs(HCD)
    response = {
        k[len("out_") :]: v for k, v in load_fixture(HCD).items() if k.startswith("out_")
    }
    _, stats = assemble_library(specs, response, ms2_model=ms2_model(HCD))
    assert stats.max_abs_ppm_deviation < 5.0, stats.worst_ppm_annotation
    assert stats.n_unparseable_annotations == 0


def test_padding_sentinel_is_not_emitted_as_a_peak():
    """Prosit pads its fixed 174-wide grid with -1; those are absent ions."""
    specs = fixture_specs(HCD)
    response = {
        k[len("out_") :]: v for k, v in load_fixture(HCD).items() if k.startswith("out_")
    }
    lib, _ = assemble_library(specs, response, ms2_model=ms2_model(HCD))
    intensities = lib.fragments.column("intensity_predicted").to_pylist()
    assert intensities and min(intensities) > 0.0


def test_response_row_mismatch_is_an_error():
    response = {
        k[len("out_") :]: v for k, v in load_fixture(HCD).items() if k.startswith("out_")
    }
    with pytest.raises(ValueError, match="not aligned"):
        assemble_library(fixture_specs(HCD)[:3], response, ms2_model=ms2_model(HCD))


def test_partial_id_rows_survive_assembly():
    """An unparseable annotation is kept with NULLs + the raw string."""
    specs = fixture_specs(HCD)[:1]
    response = {
        "intensities": np.array([[0.5, 0.25]]),
        "mz": np.array([[147.1128, 300.0]]),
        "annotation": np.array([[b"y1+1", b"Int/PEPT"]], dtype=object),
    }
    lib, stats = assemble_library(specs, response, ms2_model=ms2_model(HCD))
    assert stats.n_unparseable_annotations == 1
    rows = lib.fragments.to_pylist()
    partial = [r for r in rows if r["ion_type"] is None]
    assert len(partial) == 1
    assert partial[0]["annotation"] == "Int/PEPT"


# ── RT ────────────────────────────────────────────────────────────────


def test_rt_unit_conversion_is_per_model():
    assert rt_model("Chronologer_RT").to_seconds == 60.0
    assert rt_model("AlphaPept_rt_generic").to_seconds == 1.0


def test_unregistered_model_warns_but_works():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overlay = ms2_model("Some_Future_Model_2031")
    assert not overlay.registered
    assert any("no local overlay" in str(w.message) for w in caught)


# ── end-to-end through the layered API ────────────────────────────────


def _library_from_fixture(model=HCD):
    from constellation.massspec.library.koina.api import predict_library

    return predict_library(
        specs=fixture_specs(model),
        ms2_model_name=model,
        rt_model_name=None,
        collision_energy=30.0 if model == HCD else None,
        ms2_client=FakeKoinaClient(model),
    )


def test_predict_library_end_to_end_offline():
    lib, stats = _library_from_fixture()
    assert lib.precursors.num_rows == 8
    assert stats.n_fragments == lib.fragments.num_rows
    assert lib.metadata_extras["x.koina.ms2_model"] == HCD


def test_library_roundtrips_through_parquet_dir(tmp_path):
    """The format score_fragments.py already reads."""
    from constellation.massspec.library import load_library, save_library

    lib, _ = _library_from_fixture()
    save_library(lib, tmp_path / "lib", format="parquet_dir")
    back = load_library(tmp_path / "lib")
    assert back.fragments.num_rows == lib.fragments.num_rows
    assert back.precursors.num_rows == lib.precursors.num_rows


def test_library_exports_as_dlib(tmp_path):
    """Koina predictions can be handed to EncyclopeDIA as a .dlib."""
    from constellation.massspec.library import save_library

    lib, _ = _library_from_fixture()
    save_library(lib, tmp_path / "pred.dlib", format="encyclopedia.dlib")
    assert (tmp_path / "pred.dlib").stat().st_size > 0


def test_collision_energy_rejected_for_cid():
    """Six energies against CID would yield six identical libraries."""
    from constellation.massspec.library.koina.api import predict_fragments

    with pytest.raises(KoinaInputError, match="does not accept a collision energy"):
        predict_fragments(
            fixture_specs(CID),
            model=CID,
            collision_energy=30.0,
            client=FakeKoinaClient(CID),
        )


def test_collision_energy_required_for_hcd():
    from constellation.massspec.library.koina.api import predict_fragments

    with pytest.raises(KoinaInputError, match="requires a collision energy"):
        predict_fragments(
            fixture_specs(HCD),
            model=HCD,
            collision_energy=None,
            client=FakeKoinaClient(HCD),
        )


def test_preflight_rejects_overlong_peptide():
    from constellation.massspec.library.koina.api import predict_fragments

    long_spec = PrecursorSpec(
        modified_sequence="A" * 31 + "K",
        sequence="A" * 31 + "K",
        charge=2,
        precursor_mz=500.0,
    )
    with pytest.raises(KoinaInputError, match="exceeds"):
        predict_fragments(
            [long_spec], model=HCD, collision_energy=30.0, client=FakeKoinaClient(HCD)
        )


def test_preflight_rejects_out_of_range_charge():
    from constellation.massspec.library.koina.api import predict_fragments

    spec = PrecursorSpec(
        modified_sequence="PEPTIDEK",
        sequence="PEPTIDEK",
        charge=9,
        precursor_mz=500.0,
    )
    with pytest.raises(KoinaInputError, match="charge outside"):
        predict_fragments(
            [spec], model=HCD, collision_energy=30.0, client=FakeKoinaClient(HCD)
        )


@pytest.mark.parametrize(
    "shape",
    [
        {"peptide_sequences": ["LVNELTEFAK"], "precursor_charges": [2]},
        [{"peptide_sequences": "LVNELTEFAK", "precursor_charges": 2}],
    ],
)
def test_predict_accepts_multiple_input_shapes(shape):
    """dict-of-lists and sequence-of-row-dicts reach the same request."""
    import pyarrow as pa

    from constellation.massspec.library.koina.api import predict

    out = predict(CID, shape, client=FakeKoinaClient(CID))
    assert isinstance(out, pa.Table)
    assert "intensities" in out.column_names


def test_predict_drops_padding_rows():
    """Prosit's -1 padding must not reach the caller as if it were data."""
    from constellation.massspec.library.koina.api import predict

    out = predict(
        CID,
        {"peptide_sequences": ["LVNELTEFAK"], "precursor_charges": [2]},
        client=FakeKoinaClient(CID),
    )
    intensities = out.column("intensities").to_pylist()
    assert intensities, "everything was filtered"
    assert min(intensities) > 0.0
    # The fixture grid is 8 precursors x 174 slots; real ions are far fewer.
    assert out.num_rows < 8 * 174


def test_predict_accepts_arrow_table():
    import pyarrow as pa

    from constellation.massspec.library.koina.api import predict

    table = pa.table(
        {"peptide_sequences": ["LVNELTEFAK"], "precursor_charges": pa.array([2], pa.int32())}
    )
    out = predict(CID, table, client=FakeKoinaClient(CID))
    assert out.num_rows > 0


# ── the no-pandas boundary ────────────────────────────────────────────


def test_library_package_does_not_import_pandas():
    """Constellation bans pandas internally; keep the boundary from eroding."""
    root = Path(__file__).parent.parent / "constellation" / "massspec" / "library"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "import pandas" in path.read_text()
    ]
    assert offenders == []
