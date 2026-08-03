"""Tier B — live regression checks against the public Koina server.

Gated on ``CONSTELLATION_KOINA_TESTS=1`` so the default suite stays
offline. These exist to catch the server changing out from under the
assumptions baked into ``models.py`` and ``_annotation.py`` — the
captured fixtures can't notice that, because they're frozen.

    CONSTELLATION_KOINA_TESTS=1 pytest tests/test_koina_live.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CONSTELLATION_KOINA_TESTS") != "1",
        reason="set CONSTELLATION_KOINA_TESTS=1 to run live Koina tests",
    ),
]

pytest.importorskip("koinapy")

HCD = "Prosit_2020_intensity_HCD"
CID = "Prosit_2020_intensity_CID"

PEPTIDES = ["LVNELTEFAK", "YIC[UNIMOD:4]DNQDTISSK", "HLVDEPQNLIK", "LGEYGFQNALIVR"]


@pytest.fixture(scope="module")
def hcd_client():
    from constellation.massspec.library.koina.client import KoinaClient

    return KoinaClient(HCD)


@pytest.fixture(scope="module")
def cid_client():
    from constellation.massspec.library.koina.client import KoinaClient

    return KoinaClient(CID)


def test_declared_inputs_still_differ_between_hcd_and_cid(hcd_client, cid_client):
    """The asymmetry the multi-energy guard depends on.

    If CID ever gains a collision-energy input, the CLI's sweep rejection
    becomes wrong and should be relaxed.
    """
    assert "collision_energies" in hcd_client.model_inputs
    assert "collision_energies" not in cid_client.model_inputs


def test_rt_output_column_names_are_stable():
    """models.py's unit conversion is keyed to which head a model exposes."""
    from constellation.massspec.library.koina.client import KoinaClient

    assert "rt" in KoinaClient("Chronologer_RT").model_outputs
    assert "irt" in KoinaClient("AlphaPept_rt_generic").model_outputs


def test_hcd_responds_to_collision_energy(hcd_client):
    from constellation.massspec.library.koina.api import predict

    def at(energy: float) -> np.ndarray:
        table = predict(
            HCD,
            {
                "peptide_sequences": ["LVNELTEFAK"],
                "precursor_charges": [2],
                "collision_energies": [energy],
            },
            client=hcd_client,
        )
        return np.asarray(table.column("intensities").to_pylist())

    low, high = at(15.0), at(45.0)
    assert not np.allclose(low, high), "HCD ignored collision energy"


def test_cid_is_energy_independent(cid_client):
    """Documented behaviour, and the reason a CID sweep is refused."""
    from constellation.massspec.library.koina.api import predict

    out = predict(
        CID,
        {"peptide_sequences": ["LVNELTEFAK"], "precursor_charges": [2]},
        client=cid_client,
    )
    assert out.num_rows > 0


def test_annotation_grammar_unchanged(hcd_client):
    """The regression canary for ``_annotation.py``'s regex."""
    from constellation.massspec.library.koina._annotation import parse_koina_annotation

    response = hcd_client.predict(
        {
            "peptide_sequences": np.array(["LVNELTEFAK"], dtype=object).reshape(-1, 1),
            "precursor_charges": np.array([[2]], dtype=np.int32),
            "collision_energies": np.array([[30.0]], dtype=np.float32),
        }
    )
    tokens = {t for row in response["annotation"] for t in row}
    assert tokens, "no annotations returned"
    for token in tokens:
        assert parse_koina_annotation(token, 30) is not None, token


def test_mz_crosscheck_within_tolerance(hcd_client):
    """Our ladder vs the server's m/z — catches modseq-translation drift."""
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.library.digest import PrecursorSpec
    from constellation.massspec.library.koina.api import predict_fragments
    from constellation.massspec.library.koina.assemble import assemble_library
    from constellation.massspec.library.koina.models import ms2_model

    specs = [
        PrecursorSpec(
            modified_sequence=modseq,
            sequence=parse_proforma(modseq).sequence,
            charge=z,
            precursor_mz=500.0,
            proteins=("P02769",),
        )
        for modseq in PEPTIDES
        for z in (2, 3)
    ]
    sent, response = predict_fragments(
        specs, model=HCD, collision_energy=30.0, client=hcd_client
    )
    _, stats = assemble_library(sent, response, ms2_model=ms2_model(HCD))

    assert stats.max_abs_ppm_deviation < 5.0, stats.worst_ppm_annotation
    assert stats.n_unparseable_annotations == 0
    assert stats.n_ladder_misses == 0


def test_model_limits_still_hold(hcd_client):
    """``models.py`` claims length <= 30 and charge <= 6 for Prosit 2020."""
    from constellation.massspec.library.koina.client import KoinaError

    def try_predict(peptide: str, charge: int) -> bool:
        try:
            hcd_client.predict(
                {
                    "peptide_sequences": np.array([peptide], dtype=object).reshape(-1, 1),
                    "precursor_charges": np.array([[charge]], dtype=np.int32),
                    "collision_energies": np.array([[30.0]], dtype=np.float32),
                }
            )
            return True
        except KoinaError:
            return False

    assert try_predict("A" * 29 + "K", 2), "30-mer should be accepted"
    assert not try_predict("A" * 30 + "K", 2), "31-mer should be rejected"
    assert try_predict("LVNELTEFAK", 6), "charge 6 should be accepted"
    assert not try_predict("LVNELTEFAK", 7), "charge 7 should be rejected"
