# ruff: noqa: F811  (the stand-in jar's fixtures are imported, then named as arguments)
"""Phase 6: the spectrum-centric search (ScribeTwo).

The stand-in jar of ``test_massspec_search_workflow`` behaves as
ScribeTwo does where the workflow can tell: it writes ``<input>.dlib``
beside its input and its reports in ``<input>.scribe2/`` (the captured
inventory), stores whole observed spectra, records itself as a
``ScribeTwo...`` search, and records its own adjusted tolerances instead
of the ones it was given. What it was measured to do is in
``docs/plans/encyclopedia-6.9.23-survey.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from constellation.core.manifests import read_manifest
from constellation.massspec.io.encyclopedia import read_encyclopedia
from constellation.massspec.manifest import SearchManifest
from constellation.massspec.search.encyclopedia.params import SearchParameters, Tolerance
from constellation.massspec.workflows import encyclopedia as workflow
from constellation.massspec.workflows.encyclopedia import (
    WorkflowError,
    quantify,
    read_experiment_table,
    report_searches,
    search_acquisition,
    search_acquisitions,
)

from _massspec_fixtures import dda_scans as _dda_scans
from test_massspec_search_workflow import _NOISE_PEAKS, _PEPTIDES, _bundle, inputs, jar  # noqa: F401


def _dda(root: Path, name: str = "dda_run") -> Path:
    return _bundle(root, name, _dda_scans())


def _library_free(inputs: dict) -> dict:
    return {"fasta": inputs["fasta"]}


def test_a_dda_run_is_searched_spectrum_centric_without_a_library(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    bundle = _dda(tmp_path)
    result = search_acquisition(bundle, experiment_dir=experiment, **_library_free(inputs))

    (call,) = jar.calls
    assert call.program == "scribetwo" and call.flags["-searchType"] == "auto"
    assert "-l" not in call.flags
    manifest = result.manifest
    assert manifest.search_mode == "spectrum-centric" and manifest.program == "scribetwo"
    assert manifest.status == "finished" and "library" not in manifest.inputs
    # The jar's own names: the result beside the input, the reports in a
    # directory named for it.
    assert manifest.outputs["result_library"] == "jar/dda_run.dia.dlib"
    assert manifest.outputs["<input>.scribe2/<input>.scribe2.features.txt"] == (
        "jar/dda_run.dia.scribe2/dda_run.dia.scribe2.features.txt"
    )

    # Scores and identifications, under the engine that made them.
    scores = read_experiment_table(experiment, "precursor_score", unit=result.unit)
    assert set(scores.column("engine").to_pylist()) == {"scribetwo"}
    precursors = read_experiment_table(experiment, "precursor", unit=result.unit)
    assert precursors.num_rows == len(_PEPTIDES)

    # Every scored match, with its scan; the best of each precursor carries
    # the scores and joins the precursor table by its content id.
    psms = read_experiment_table(experiment, "psm", unit=result.unit)
    assert psms.num_rows == 2 * (len(_PEPTIDES) + 1)  # two per precursor, and a decoy
    assert set(psms.column("engine").to_pylist()) == {"scribetwo"}
    best = psms.filter(pc.equal(psms.column("psm_type"), "best"))
    targets = best.filter(pc.invert(best.column("is_decoy")))
    assert targets.num_rows == len(_PEPTIDES)
    assert set(targets.column("precursor_id").to_pylist()) == set(
        precursors.column("precursor_id").to_pylist()
    )
    assert all(q is not None for q in targets.column("qvalue").to_pylist())
    other = psms.filter(pc.equal(psms.column("psm_type"), "other"))
    assert other.column("qvalue").null_count == other.num_rows
    decoys = psms.filter(psms.column("is_decoy"))
    assert decoys.column("precursor_id").null_count == decoys.num_rows
    # The scans are the acquisition's: they join its scan metadata.
    scans = pq.read_table(bundle / "scan_metadata.parquet", columns=["scan", "rt"]).to_pylist()
    rt_of = {r["scan"]: r["rt"] for r in scans}
    for row in psms.to_pylist():
        assert row["acquisition_id"] == 0 and row["raw_file"] == "dda_run.raw"
        assert row["retention_time_s"] == pytest.approx(rt_of[row["scan"]], abs=1e-3)

    # The stored spectra are observed spectra: only the peptide's ions are
    # fragment rows, and the rest is counted.
    fragments = read_experiment_table(experiment, "fragment", unit=result.unit)
    assert fragments.num_rows and not set(_NOISE_PEAKS) & set(fragments.column("mz_theoretical").to_pylist())
    assert any("peaks this ScribeTwo result stores are not ions" in c for c in manifest.caveats)


def test_scribetwo_reads_a_copy_of_the_fasta_kept_in_the_experiment(tmp_path, jar, inputs) -> None:
    """ScribeTwo writes a 400 MB index beside the FASTA it is given and
    follows links to do so (measured), so it gets a copy of its own."""
    experiment = tmp_path / "experiment"
    first = search_acquisition(_dda(tmp_path, "dda_a"), experiment_dir=experiment, **_library_free(inputs))
    (copy,) = jar.fasta_seen
    assert copy.parent.parent == experiment / "fasta" and copy.name == inputs["fasta"].name
    assert copy.read_bytes() == inputs["fasta"].read_bytes() and not copy.is_symlink()
    assert first.manifest.inputs["fasta"]["path"] == str(inputs["fasta"].resolve())
    assert first.manifest.inputs["fasta"]["copy"] == str(copy.relative_to(experiment))
    stamp = copy.stat().st_mtime_ns
    search_acquisition(_dda(tmp_path, "dda_b"), experiment_dir=experiment, **_library_free(inputs))
    assert jar.fasta_seen[1] == copy and copy.stat().st_mtime_ns == stamp  # made once
    assert sorted(p.name for p in (experiment / "fasta").iterdir() if p.is_dir()) == [copy.parent.name]
    # A peptide-centric search reads the FASTA where it is.
    search_acquisition(_bundle(tmp_path, "dia_run"), experiment_dir=experiment, **inputs)
    assert jar.calls[-1].flags["-f"] == inputs["fasta"].resolve()


def test_the_tolerances_scribetwo_adjusts_are_said_so_not_warned_about(tmp_path, jar, inputs) -> None:
    params = SearchParameters(
        precursor_tolerance=Tolerance(0.35, "Da"), fragment_tolerance=Tolerance(0.35, "Da"),
        library_fragment_tolerance=Tolerance(0.35, "Da"),
    )  # fmt: skip
    result = search_acquisition(
        _dda(tmp_path), experiment_dir=tmp_path / "e", parameters=params, **_library_free(inputs)
    )
    assert result.disagreements == ()
    notes = [c for c in result.manifest.caveats if c.startswith("ScribeTwo adjusted")]
    assert notes == [
        "ScribeTwo adjusted the precursor tolerance from 0.35 to 0.42 Da (its first pass on a "
        "sample of spectra); the result records the adjusted value",
        "ScribeTwo adjusted the fragment tolerance from 0.35 to 0.42 Da (its first pass on a "
        "sample of spectra); the result records the adjusted value",
    ]
    # Fragment rows are annotated at the adjusted tolerance the result records.
    assert result.manifest.engine_settings["-ftol"] == "0.42"


def test_named_modifications_and_the_prediction_cache(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    bundle = _dda(tmp_path)
    cache_a, cache_b = tmp_path / "cache_a", tmp_path / "cache_b"
    first = search_acquisition(
        bundle, experiment_dir=experiment, **_library_free(inputs),
        parameters=SearchParameters(ptms={"Oxidation": "var", "Carbamidomethyl": "fix"},
                                    prediction_cache=cache_a),
    )  # fmt: skip
    flags = jar.calls[0].flags
    assert flags["-ptmOxidation"] == "var" and flags["-ptmCarbamidomethyl"] == "fix"
    assert flags["-predictionCache"] == cache_a
    assert first.manifest.parameters["ptms"] == {"Oxidation": "var", "Carbamidomethyl": "fix"}
    # The cache only saves time: a search in another cache is the same search.
    again = search_acquisition(
        bundle, experiment_dir=experiment, **_library_free(inputs), resume=True,
        parameters=SearchParameters(ptms={"Oxidation": "var", "Carbamidomethyl": "fix"},
                                    prediction_cache=cache_b),
    )  # fmt: skip
    assert again.reused and again.unit == first.unit and len(jar.calls) == 1
    # Other modifications are another search.
    other = search_acquisition(
        bundle, experiment_dir=experiment, **_library_free(inputs),
        parameters=SearchParameters(ptms={"Oxidation": "off", "Carbamidomethyl": "fix"}),
    )  # fmt: skip
    assert other.unit != first.unit


def test_what_the_spectrum_centric_search_refuses(tmp_path, jar, inputs, monkeypatch) -> None:
    with pytest.raises(ValueError, match="no modification 'Oxidized'"):
        SearchParameters(ptms={"Oxidized": "var"})
    with pytest.raises(ValueError, match="must be one of off, var, fix"):
        SearchParameters(ptms={"Oxidation": "variable"})
    with pytest.raises(ValueError, match="do not mix"):
        SearchParameters(ptms={"Oxidation": "var"}, extra={"-fixed": ""})
    experiment = tmp_path / "experiment"
    a, b = _dda(tmp_path, "dda_a"), _dda(tmp_path, "dda_b")
    with pytest.raises(WorkflowError, match="a merged search is peptide-centric"):
        search_acquisitions([a, b], experiment_dir=experiment, merge=True, **_library_free(inputs))
    with pytest.raises(ValueError, match="--search-type dda was given for a DIA acquisition"):
        search_acquisition(_bundle(tmp_path, "dia_run"), experiment_dir=experiment,
                           search_mode="spectrum-centric", search_type="dda", **inputs)  # fmt: skip
    assert jar.calls == []

    # ScribeTwo scores with its own JavaPot: a Percolator this machine
    # cannot run does not stop it.
    monkeypatch.setattr(workflow, "percolator_blocker", lambda *a, **k: "Percolator cannot run here")
    assert search_acquisition(a, experiment_dir=experiment, **_library_free(inputs)).manifest.status == "finished"

    # Reports and the chromatogram-library chain are peptide-centric.
    unit = search_acquisition(b, experiment_dir=experiment, **_library_free(inputs)).unit
    with pytest.raises(WorkflowError, match="is spectrum-centric"):
        report_searches(experiment, fasta=inputs["fasta"], searches=[unit])
    with pytest.raises(WorkflowError, match="has no finished peptide-centric search"):
        report_searches(experiment, fasta=inputs["fasta"])
    monkeypatch.setattr(workflow, "percolator_blocker", lambda *a, **k: None)
    with pytest.raises(ValueError, match="no peptide-centric search of DDA"):
        quantify(experiment, runs=[a], fasta=inputs["fasta"], library=inputs["library"])


def test_a_dia_run_searched_spectrum_centric_with_a_library(tmp_path, jar, inputs) -> None:
    result = search_acquisition(
        _bundle(tmp_path, "dia_run"), experiment_dir=tmp_path / "e", search_mode="spectrum-centric",
        search_type="dia", **inputs,
    )  # fmt: skip
    flags = jar.calls[0].flags
    assert jar.calls[0].program == "scribetwo" and flags["-searchType"] == "dia"
    assert flags["-l"] == inputs["library"].resolve()
    assert result.manifest.inputs["library"]["path"] == str(inputs["library"].resolve())
    psms = read_experiment_table(tmp_path / "e", "psm", unit=result.unit)
    assert psms.num_rows and set(psms.column("engine").to_pylist()) == {"scribetwo"}


def test_scribetwo_against_an_elib_writes_an_elib(tmp_path, jar, inputs) -> None:
    """Measured: ScribeTwo names its result after the library it searched,
    so a chromatogram library (.elib) gives <input>.elib."""
    chromatogram = tmp_path / "gpf.elib"
    chromatogram.write_bytes(b"a chromatogram library")
    result = search_acquisition(
        _bundle(tmp_path, "dia_run"), experiment_dir=tmp_path / "e", search_mode="spectrum-centric",
        search_type="dia", fasta=inputs["fasta"], library=chromatogram,
    )  # fmt: skip
    assert result.manifest.outputs["result_library"] == "jar/dia_run.dia.elib"
    assert read_experiment_table(tmp_path / "e", "psm", unit=result.unit).num_rows


def test_read_encyclopedia_finds_the_reports_beside_a_scribetwo_result(tmp_path, jar, inputs) -> None:
    result = search_acquisition(_dda(tmp_path), experiment_dir=tmp_path / "e", **_library_free(inputs))
    library = result.unit_dir / result.manifest.outputs["result_library"]
    read = read_encyclopedia(library, annotate_fragments=False)
    assert read.search.psms.num_rows == 2 * (len(_PEPTIDES) + 1)
    assert read_encyclopedia(library, annotate_fragments=False, psms=False).search.psms.num_rows == 0
    # Without its reports a ScribeTwo result is still read, with no matches.
    lone = tmp_path / "lone.dia.dlib"
    lone.write_bytes(library.read_bytes())
    assert read_encyclopedia(lone, annotate_fragments=False).search.psms.num_rows == 0


def test_the_command(tmp_path, jar, inputs, capsys) -> None:
    from constellation.cli.__main__ import main

    bundle = _dda(tmp_path)
    argv = ["massspec", "search", "--input", str(bundle), "--fasta", str(inputs["fasta"]),
            "--output-dir", str(tmp_path / "e"), "--ptm", "Oxidation=var",
            "--prediction-cache", str(tmp_path / "cache"), "--no-progress"]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "spectrum-centric search of dda_run (acquisition 0)" in out
    flags = jar.calls[0].flags
    assert flags["-ptmOxidation"] == "var" and flags["-predictionCache"] == tmp_path / "cache"
    assert main([*argv[:-5], "--ptm", "Oxidation=sometimes", "--no-progress"]) == 1
    assert "must be one of off, var, fix" in capsys.readouterr().err
    assert os.path.isdir(tmp_path / "e" / "fasta")
    record = read_manifest(next((tmp_path / "e" / "units").glob("dda_run-*")), expect=SearchManifest)
    assert record.parameters["prediction_cache"] == str(tmp_path / "cache")


def test_the_summary_says_what_a_spectrum_centric_search_is(tmp_path, jar, inputs) -> None:
    from constellation.massspec.experiment import format_summary, summarize

    params = SearchParameters(precursor_tolerance=Tolerance(0.35, "Da"), ptms={"Oxidation": "var"})
    result = search_acquisition(
        _dda(tmp_path), experiment_dir=tmp_path / "e", parameters=params, **_library_free(inputs)
    )
    (s,) = summarize(tmp_path / "e")["searches"]
    assert s["search_mode"] == "spectrum-centric" and s["search_type"] == "auto"
    assert ["search type", "auto"] in s["settings"]
    assert ["precursor tolerance", "0.42 Da (adjusted by ScribeTwo; given 0.35)"] in s["settings"]
    assert ["scoring", "JavaPot (ScribeTwo's own)"] in s["settings"]
    assert not [p for p in s["settings"] if p[0] == "Percolator"]
    text = " ".join(format_summary(summarize(tmp_path / "e")).split())
    assert "library none: predicted from the FASTA" in text
    detail = " ".join(format_summary(summarize(result.unit_dir)).split())
    assert "ScribeTwo read the experiment's copy, fasta/" in detail
    assert "-ptmOxidation var" in detail


def test_scribetwo_spells_no_modifications_with_a_dash_and_its_fatal_errors_are_shown(tmp_path, jar, inputs) -> None:
    """Measured on 6.9.23: ``-fixed ''`` (the peptide-centric search's
    'none') stops ScribeTwo with an error, after which it exits 0."""
    from constellation.massspec.search.encyclopedia.outputs import MissingOutputError

    bundle = _dda(tmp_path)
    with pytest.raises(WorkflowError, match="ScribeTwo spells 'none' for -fixed as '-'"):
        search_acquisition(bundle, experiment_dir=tmp_path / "e", **_library_free(inputs),
                           parameters=SearchParameters(extra={"-fixed": ""}))  # fmt: skip
    assert jar.calls == []
    jar.fatal = "edu.washington.EncyclopediaException: Error parsing fixed modifications from []"
    with pytest.raises(MissingOutputError, match="scribetwo stopped on an error and exited 0: edu.washington"):
        search_acquisition(bundle, experiment_dir=tmp_path / "e", **_library_free(inputs),
                           parameters=SearchParameters(extra={"-fixed": "-"}))  # fmt: skip
    (record,) = [read_manifest(d, expect=SearchManifest) for d in (tmp_path / "e" / "units").iterdir() if d.is_dir()]
    assert record.status == "failed" and "Error parsing fixed modifications" in record.error


def test_fatal_error_reads_the_jars_report(tmp_path) -> None:
    from constellation.massspec.search.encyclopedia.outputs import fatal_error

    assert fatal_error(None) is None and fatal_error(tmp_path / "absent.log") is None
    log = tmp_path / "stdout.log"
    log.write_text(  # as 6.9.23's ScribeTwo wrote it, 2026-10-17
        "[00:37:51] Thermo server: waiting for gRPC port 36705 to accept connections...\n"
        "[00:37:51] Encountered Fatal Error!\n"
        "[00:37:51] edu.washington.gs.maccoss.encyclopedia.utils.EncyclopediaException: Error "
        "parsing fixed modifications from [], expected comma-delimited site=mass rules or '-'\n"
        "\tat edu.washington.gs.maccoss.encyclopedia.filereaders.FixedModificationParser.error("
        "FixedModificationParser.java:78)\n"
    )
    assert fatal_error(log) == (
        "edu.washington.gs.maccoss.encyclopedia.utils.EncyclopediaException: Error parsing fixed "
        "modifications from [], expected comma-delimited site=mass rules or '-'"
    )
    log.write_text("[00:00:01] Finished analysis!\n")
    assert fatal_error(log) is None
