# ruff: noqa: F811  (the stand-in jar's fixtures are imported, then named as arguments)
"""Reading what an experiment's searches found (massspec.results).

Experiments are made by the real search workflow against the stand-in
jar of ``test_massspec_search_workflow``; the searched library is a real
EncyclopeDIA library file, since the lookup reads it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from constellation.massspec.io.encyclopedia import import_dia, write_dia
from constellation.massspec.library.ids import precursor_id
from constellation.massspec.manifest import table_partition_dir
from constellation.massspec.quant.targets import targets_from_experiment
from constellation.massspec.results import (
    acquisition_bundles,
    identifications,
    library_spectrum,
    precursor_detail,
    rt_mapping,
    search_units,
    searched_library_spectrum,
    trace_targets,
)
from constellation.massspec.search.encyclopedia.params import SearchParameters, Tolerance
from constellation.massspec.spectra.annotate import ion_ladder
from constellation.massspec.workflows.encyclopedia import report_searches, search_acquisition

from _encyclopedia_fixtures import build_result_library
from _massspec_fixtures import dda_scans
from test_massspec_search_workflow import _PEPTIDES, _bundle, _entry, inputs, jar  # noqa: F401

PARAMS = SearchParameters(precursor_tolerance=Tolerance(0.35, "Da"), threads=1, extra={"-fixed": ""})


@pytest.fixture
def searched(tmp_path: Path, jar, inputs):
    """An experiment holding one peptide-centric search of one bundle."""
    library = tmp_path / "searched.dlib"
    build_result_library(library, [_entry(m, z, t, "lib") for m, z, t in _PEPTIDES])
    bundle = _bundle(tmp_path, "run_01")
    experiment = tmp_path / "experiment"
    result = search_acquisition(bundle, experiment_dir=experiment, parameters=PARAMS,
                                fasta=inputs["fasta"], library=library)  # fmt: skip
    return experiment, result.unit, bundle, library


@pytest.fixture
def scribed(tmp_path: Path, jar, inputs):
    """An experiment holding one spectrum-centric (ScribeTwo) search of a
    DDA run, given 0.35 Da tolerances that the stand-in, like ScribeTwo,
    adjusts (to 0.42 Da) and records."""
    params = SearchParameters(
        precursor_tolerance=Tolerance(0.35, "Da"), fragment_tolerance=Tolerance(0.35, "Da"),
        library_fragment_tolerance=Tolerance(0.35, "Da"),
    )  # fmt: skip
    experiment = tmp_path / "experiment"
    result = search_acquisition(_bundle(tmp_path, "dda_run", dda_scans()), experiment_dir=experiment,
                                parameters=params, fasta=inputs["fasta"])  # fmt: skip
    return experiment, result.unit


def _pid(modseq_proforma: str, charge: int) -> int:
    return precursor_id(modseq_proforma, charge)


def test_the_finished_searches_are_listed(searched) -> None:
    experiment, unit, _, library = searched
    (found,) = search_units(experiment)
    assert found["unit"] == unit
    assert found["search_mode"] == "peptide-centric"
    assert found["acquisition_ids"] == [0]
    assert found["library"] == str(library)
    assert found["precursor_tolerance"] == {"value": 0.35, "unit": "Da"}
    assert found["merged_from"] is None


def test_identifications_are_the_quantified_precursors_with_their_peptides(searched) -> None:
    experiment, unit, _, _ = searched
    ids = identifications(experiment, max_qvalue=None)
    targets = targets_from_experiment(experiment)
    assert sorted(ids.column("precursor_id").to_pylist()) == sorted(targets.column("precursor_id").to_pylist())
    assert set(ids.column("unit").to_pylist()) == {unit}
    assert set(ids.column("acquisition_id").to_pylist()) == {0}
    row = ids.filter(pc.equal(ids.column("modified_sequence"), "PEPTIDEK")).to_pylist()[0]
    assert row["sequence"] == "PEPTIDEK" and row["charge"] == 2
    assert row["precursor_id"] == _pid("PEPTIDEK", 2)
    assert row["proteins"] == "P1"
    assert row["scan"] is None and row["is_decoy"] is False
    assert row["rt"] == pytest.approx(600.0)

    one = identifications(experiment, precursor_id=row["precursor_id"], max_qvalue=None)
    assert one.num_rows == 1
    assert identifications(experiment, unit="no-such-search").num_rows == 0
    assert identifications(experiment, acquisition_id=7).num_rows == 0


def test_q_values_and_decoys_filter_rows(searched) -> None:
    experiment, *_ = searched
    every = identifications(experiment, max_qvalue=None)
    strict = identifications(experiment, max_qvalue=-1.0)
    q = every.column("qvalue")
    assert strict.num_rows == q.null_count  # only rows without a q-value pass a negative cut
    assert identifications(experiment).num_rows <= every.num_rows


def test_a_spectrum_centric_match_is_accepted_with_its_precursor(scribed) -> None:
    # ScribeTwo scores the best match of each precursor; its other matches
    # carry no q-value and stand or fall with it.
    experiment, _ = scribed
    every = identifications(experiment, max_qvalue=None)
    best = {}
    for r in every.to_pylist():
        if r["qvalue"] is not None:
            best[r["precursor_id"]] = min(r["qvalue"], best.get(r["precursor_id"], 1.0))
    assert sorted(best.values())[:2] == pytest.approx([0.001, 0.002])
    (accepted,) = [pid for pid, q in best.items() if q < 0.0015]
    strict = identifications(experiment, max_qvalue=0.0015)
    assert set(strict.column("precursor_id").to_pylist()) == {accepted}
    assert sorted(strict.column("qvalue").to_pylist(), key=lambda q: q is None) == [pytest.approx(0.001), None]
    assert len(set(strict.column("scan").to_pylist())) == 2


def test_tolerances_are_the_ones_the_search_ran_with(scribed) -> None:
    experiment, unit = scribed
    (found,) = search_units(experiment)
    assert found["fragment_tolerance"] == {"value": 0.42, "unit": "Da"}
    assert found["precursor_tolerance"] == {"value": 0.42, "unit": "Da"}
    pid = identifications(experiment).column("precursor_id")[0].as_py()
    detail = precursor_detail(experiment, pid, unit=unit, acquisition_id=0)
    assert detail["fragment_tolerance"] == {"value": 0.42, "unit": "Da"}


def test_a_spectrum_centric_precursor_is_detailed_at_the_match_asked_for(scribed) -> None:
    experiment, unit = scribed
    every = identifications(experiment, max_qvalue=None)
    pid = every.filter(pc.equal(every.column("qvalue"), 0.001)).column("precursor_id")[0].as_py()
    rows = every.filter(pc.equal(every.column("precursor_id"), pid)).to_pylist()
    (best,) = [r for r in rows if r["qvalue"] is not None]
    (other,) = [r for r in rows if r["qvalue"] is None]
    assert precursor_detail(experiment, pid, unit=unit, acquisition_id=0)["identification"]["scan"] == best["scan"]
    found = precursor_detail(experiment, pid, unit=unit, acquisition_id=0, scan=other["scan"])["identification"]
    assert (found["scan"], found["rt"], found["qvalue"]) == (other["scan"], other["rt"], None)
    assert precursor_detail(experiment, pid, unit=unit, acquisition_id=0, scan=10**6)["identification"] is None


def test_a_report_is_not_a_search_of_the_run(searched, inputs) -> None:
    # A quantitative report re-quantifies the runs under a key of its own;
    # what the run's searches identified does not include it.
    experiment, unit, _, _ = searched
    report = report_searches(experiment, fasta=inputs["fasta"])
    reported = pq.ParquetDataset(experiment / "tables" / "precursor_quant").read().column("unit").to_pylist()
    assert report.key in set(map(str, reported))
    assert set(identifications(experiment, max_qvalue=None).column("unit").to_pylist()) == {unit}


def test_the_searched_bundle_is_found_from_the_search(searched) -> None:
    experiment, unit, bundle, _ = searched
    (entry,) = acquisition_bundles(experiment)
    assert entry["acquisition_id"] == 0
    assert entry["bundle"] == bundle and entry["via"] == "search input"
    assert entry["units"] == [unit]


def test_a_copied_experiment_finds_its_own_bundle(tmp_path: Path, jar, inputs) -> None:
    import shutil

    original = tmp_path / "experiment"
    bundle = _bundle(original / "bundles", "run_01")
    library = tmp_path / "searched.dlib"
    build_result_library(library, [_entry(m, z, t, "lib") for m, z, t in _PEPTIDES])
    search_acquisition(bundle, experiment_dir=original, parameters=PARAMS, fasta=inputs["fasta"], library=library)
    copy = tmp_path / "elsewhere" / "experiment"
    shutil.copytree(original, copy, symlinks=True)
    # Found in the copy itself, also while the original is still there...
    (found,) = acquisition_bundles(copy)
    assert found["bundle"] == copy / "bundles" / "run_01" and found["via"] == "experiment bundles"
    # ...and once it is gone.
    shutil.rmtree(original)
    (found,) = acquisition_bundles(copy)
    assert found["bundle"] == copy / "bundles" / "run_01"


def test_a_run_searched_from_a_spectra_file_needs_its_bundle_attached(tmp_path: Path, jar, inputs) -> None:
    source = _bundle(tmp_path / "src", "run_02")
    spectra = write_dia([source], tmp_path / "given" / "run_02.dia").path
    experiment = tmp_path / "experiment"
    search_acquisition(spectra, experiment_dir=experiment, parameters=PARAMS, **inputs)
    (alone,) = acquisition_bundles(experiment)
    assert alone["bundle"] is None and alone["via"] is None

    (imported,) = import_dia(spectra, tmp_path / "imported")
    unrelated = _bundle(tmp_path / "other", "run_99")
    (linked,) = acquisition_bundles(experiment, candidates=[unrelated, imported])
    assert linked["bundle"] == imported and linked["via"] == "attached"


def test_the_result_spectrum_is_labelled_by_ion_number(searched) -> None:
    experiment, unit, _, _ = searched
    spectrum = library_spectrum(experiment, _pid("PEPTIDEK", 2), unit=unit, acquisition_id=0)
    assert spectrum.num_rows > 0
    assert spectrum.column("mz").to_pylist() == sorted(spectrum.column("mz").to_pylist())
    ladder = {r["annotation"]: r["mz"] for r in ion_ladder("PEPTIDEK").to_pylist()}
    for row in spectrum.to_pylist():
        if row["ion_type"] in ("b", "y"):
            # The label names the ion whose mass is at that m/z.
            assert ladder[row["annotation"]] == pytest.approx(row["mz"], abs=0.02)
    assert library_spectrum(experiment, 12345, unit=unit, acquisition_id=0).num_rows == 0


def test_the_searched_librarys_spectrum_is_found_by_content(searched) -> None:
    experiment, unit, _, library = searched
    found = searched_library_spectrum(experiment, _pid("PEPTIDEK", 2), unit=unit)
    assert found is not None
    expected = _entry("PEPTIDEK", 2, 0.0, "lib")
    assert found.column("mz").to_pylist() == pytest.approx(sorted(expected.mz))
    assert float(found.schema.metadata[b"x.massspec.rt_seconds"]) == pytest.approx(600.0)
    # The N-terminally acetylated peptide is found through the library's spelling.
    acetyl = searched_library_spectrum(experiment, _pid("[Acetyl]-MDFQHRPGGK", 3), unit=unit)
    assert acetyl is not None and acetyl.num_rows > 0
    assert searched_library_spectrum(experiment, 12345, unit=unit) is None

    # A library replaced since the search is not read.
    library.write_bytes(library.read_bytes() + b"\0")
    assert searched_library_spectrum(experiment, _pid("PEPTIDEK", 2), unit=unit) is None


def test_precursor_detail(searched) -> None:
    experiment, unit, _, library = searched
    detail = precursor_detail(experiment, _pid("ELVISLIVESK", 2), unit=unit, acquisition_id=0)
    assert detail["identification"]["modified_sequence"] == "ELVISLIVESK"
    assert detail["library"] == str(library)
    assert detail["precursor_tolerance"] == {"value": 0.35, "unit": "Da"}
    assert isinstance(detail["quantified_ions"], list) and isinstance(detail["chromatogram"], list)
    missing = precursor_detail(experiment, 12345, unit=unit, acquisition_id=0)
    assert missing["identification"] is None and missing["quantified_ions"] == []


def test_rt_mapping_takes_the_final_pass_without_decoys_in_order(searched) -> None:
    experiment, unit, _, _ = searched
    table = pa.table(
        {
            "engine": ["encyclopedia"] * 6,
            "pass": ["first", "final", "final", "final", "final", "final"],
            "modified_sequence": ["A"] * 6,
            "peptide_id": pa.array([1] * 6, pa.int64()),
            "charge": pa.array([2] * 6, pa.int32()),
            "library_rt": [100.0, 300.0, 100.0, 200.0, 250.0, 400.0],
            "observed_rt": [0.0] * 6,
            "fitted_rt": [999.0, 330.0, 110.0, 205.0, 200.0, 999.0],
            "residual": [0.0] * 6,
            "within_95": [True] * 6,
            "is_decoy": [False, False, False, False, False, True],
        }
    )
    # Replace what the stand-in jar's calibration sidecars gave.
    directory = table_partition_dir(experiment, "rt_alignment", unit, 0)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.parquet"):
        old.unlink()
    pq.write_table(table, directory / "part-0.parquet")
    mapping = rt_mapping(experiment, unit=unit, acquisition_id=0)
    observed = mapping.column("observed_rt").to_numpy()
    library_rt = mapping.column("library_rt").to_numpy()
    assert np.all(np.diff(observed) > 0) and np.all(np.diff(library_rt) > 0)
    assert library_rt.tolist() == [100.0, 200.0, 300.0]  # 250 s folds into 200 s's fit; the decoy is out
    assert np.interp(205.0, observed, library_rt) == pytest.approx(200.0)


def test_trace_targets_are_the_most_intense_stored_ions(searched) -> None:
    experiment, unit, _, _ = searched
    targets = trace_targets(experiment, _pid("ELVISLIVESK", 2), unit=unit, acquisition_id=0, top_n=3)
    assert targets.num_rows == 3
    intensities = targets.column("intensity").to_pylist()
    assert intensities == sorted(intensities, reverse=True)
    assert set(targets.column("ion_type").to_pylist()) <= {"b", "y"}
    ladder = {r["annotation"]: r["mz"] for r in ion_ladder("ELVISLIVESK").to_pylist()}
    for row in targets.to_pylist():
        assert ladder[row["label"]] == pytest.approx(row["mz"], abs=0.02)
    assert trace_targets(experiment, 12345, unit=unit, acquisition_id=0).num_rows == 0
