"""JChronologer's models in PyTorch against the jar's own predictions.

``tests/data/jchronologer/1.0.10/predictions.tsv`` holds what
``ChronologerLibraryPredictor`` itself returned for 80 requests
(``scripts/survey-jchronologer-predictions.py``). These tests run
Constellation's predictors on the same requests with the models read from
the installed jar, so they are skipped where EncyclopeDIA is not
installed. Tolerances are float32 noise: the jar batches one request at a
time, these tests all at once.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.library.prediction.assemble import Ladders
from constellation.massspec.library.prediction.base import Requests
from constellation.massspec.library.prediction.collision_energy import CollisionEnergy
from constellation.massspec.library.prediction.requests import unlabelled_sites
from constellation.massspec.library.prediction.run import predict_onto
from constellation.massspec.peptide.ions import IonType, bond_position

DATA = Path(__file__).parent / "data" / "jchronologer" / "1.0.10" / "predictions.tsv"


@pytest.fixture(scope="module")
def predictors():
    from constellation.massspec.library.prediction.registry import default_predictors
    from constellation.massspec.tokenize.jchronologer import UnknownModelError
    from constellation.thirdparty.registry import ToolNotFoundError

    try:
        return default_predictors("jchronologer", threads=4)
    except (ToolNotFoundError, FileNotFoundError, UnknownModelError) as e:
        pytest.skip(f"JChronologer is not installed here: {e}")


def _rows():
    return [
        line.split("\t")
        for line in DATA.read_text().splitlines()
        if line and not line.startswith(("#", "proforma\t"))
    ]


@pytest.fixture(scope="module")
def golden():
    entries = [r for r in _rows() if r[3] != "ERROR"]
    peptides = sorted({r[0] for r in entries})
    index = {p: i for i, p in enumerate(peptides)}
    requests = Requests(
        tuple(unlabelled_sites(p)[0] for p in peptides),
        np.array([index[r[0]] for r in entries]),
        np.array([int(r[4]) for r in entries], np.int32),
        np.array([float(r[2]) for r in entries], np.float32),
    )
    return entries, peptides, requests


def test_retention_times(predictors, golden):
    entries, _, requests = golden
    ours = predictors["retention_time"].predict(requests).values
    jar = np.array([float(r[6]) for r in entries])
    assert np.max(np.abs(ours - jar)) < 1e-3


def test_ccs_and_where_there_is_none(predictors, golden):
    entries, _, requests = golden
    sculptor = predictors["ccs"]
    refused = sculptor.check(requests).refused
    jar = np.array([np.nan if r[7] == "NA" else float(r[7]) for r in entries])
    # The jar gives no CCS exactly where Sculptor has no tokens (TMT).
    assert np.array_equal(np.isnan(jar), refused) and refused.any()
    taken = np.flatnonzero(~refused)
    ours = sculptor.predict(requests.take(taken)).values
    assert np.max(np.abs(ours - jar[taken])) < 1e-3


def test_charge_probabilities(predictors, golden):
    entries, _, requests = golden
    ours = predictors["charge_state"].predict(requests).values
    jar = np.array([np.nan if r[8] == "NA" else float(r[8]) for r in entries])
    given = ~np.isnan(jar)
    assert given.sum() > 50
    assert np.max(np.abs(ours[given] - jar[given])) < 1e-6


def test_fragments(predictors, golden):
    entries, peptides, requests = golden
    got = predictors["fragments"].predict(requests)
    ladders = Ladders([parse_proforma(p) for p in peptides], (IonType.B, IonType.Y), 3)
    index = {p: i for i, p in enumerate(peptides)}
    n_ions = 0
    for e, row in enumerate(entries):
        jar: dict[str, tuple[float, float]] = {}
        for ion in row[9].split(";") if row[9] else []:
            label, mz, intensity = ion.split("=")
            jar[label] = (float(mz), float(np.float32(intensity)))
        sel = got.request == e
        ours = {
            f"{c}+{'y' if t == IonType.Y else 'b'}{n}": v
            for t, n, c, v in zip(got.ion_type[sel], got.ion_number[sel], got.fragment_charge[sel], got.intensity[sel])
        }
        # Only ions within float noise of the 0.01 floor may be on one side alone.
        for label in set(jar) ^ set(ours):
            value = jar[label][1] if label in jar else ours[label]
            assert abs(value - 0.01) < 1e-4, (row[0], row[4], label)
        for label in set(jar) & set(ours):
            assert abs(jar[label][1] - ours[label]) < 5e-5, (row[0], label)
            charge, kind, number = int(label[0]), label[2], int(label[3:])
            t = IonType.Y if kind == "y" else IonType.B
            length = len(requests.peptides[index[row[0]]].residues)
            mz = ladders.mz(
                np.array([index[row[0]]]), np.array([int(t)], np.int8),
                np.array([bond_position(int(t), number, length)]), np.array([charge]),
            )[0]
            assert abs(mz - jar[label][0]) < 1e-5, (row[0], label)
        n_ions += len(jar)
    assert n_ions > 4000


def test_what_the_jar_refuses_is_refused(predictors):
    refused = [r[0] for r in _rows() if r[3] == "ERROR"]
    assert len(refused) == 3
    requests = Requests(
        tuple(unlabelled_sites(p)[0] for p in refused),
        np.arange(len(refused)), np.full(len(refused), 2, np.int32), np.full(len(refused), 27.5, np.float32),
    )
    assert predictors["retention_time"].check(requests).refused.all()


def test_a_search_space_predicted_end_to_end(predictors, tmp_path):
    from constellation.massspec.library.decoys import DecoySet
    from constellation.massspec.library.search_space import Digestion, SearchSpaceSettings, build_search_space

    fasta = tmp_path / "bsa.fa"
    fasta.write_text(">BSA_fragment\nDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEK\n")
    settings = SearchSpaceSettings(
        digestion=Digestion(missed_cleavages=1, excise_initiator_met=False),
        charges=(2, 3), min_mz=300, max_mz=1800, decoys=(DecoySet(),),
    )
    library = build_search_space([fasta], settings)
    out, report = predict_onto(library, predictors, collision_energy=CollisionEnergy.scaled_from_reference(30.0))
    assert report.precursors_kept == out.precursors.num_rows > 0
    precursors = out.precursors.to_pylist()
    assert all(p["rt_predicted"] > 0 and p["ccs_predicted"] > 0 and 0 <= p["charge_probability"] <= 1 for p in precursors)
    with_fragments = set(out.fragments.column("precursor_id").to_pylist())
    assert with_fragments == {p["precursor_id"] for p in precursors}
    decoys = out.peptides.filter(out.peptides.column("is_decoy")).column("decoy_of").to_pylist()
    assert decoys and set(decoys) <= set(out.peptides.column("peptide_id").to_pylist())
    # A different collision energy for one charge changes the fragments.
    other, _ = predict_onto(
        library, predictors, collision_energy=CollisionEnergy.explicit({2: 25.0, 3: 30.0})
    )
    assert not other.fragments.column("intensity_predicted").equals(out.fragments.column("intensity_predicted"))
    assert replace(other, fragments=out.fragments).precursors.equals(out.precursors)
