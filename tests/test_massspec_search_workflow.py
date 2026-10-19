"""The single-acquisition search workflow and the experiment it writes.

The jar is stood in for by a function that behaves as the jar does in
the two ways that matter here: it writes its outputs beside its input,
named for it (the captured file inventory, with a result library built
from the captured table definitions), and it cannot be made to search
where a result already is. Everything around it is real: the
acquisition registry, the spectra file Constellation writes, the link
into the search's directory, the flag validation, the tables, the
manifests, resume. A real search of real data is the opt-in
``test_encyclopedia_search_reference.py``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
import time
import types
from pathlib import Path

import pyarrow.compute as pc
import pytest

from constellation.core.manifests import manifest_kind, read_manifest
from constellation.massspec.acquisitions import load_acquisitions
from constellation.massspec.io.encyclopedia import DiaFile, write_dia
from constellation.massspec.io.encyclopedia._modseq import parse_encyclopedia_modseq
from constellation.massspec.library.ids import ID_SCHEME, content_id
from constellation.massspec.manifest import ExperimentManifest, SearchManifest
from constellation.massspec.peptide.ions import fragment_ladder
from constellation.massspec.peptide.mz import precursor_mz
from constellation.massspec.quant.targets import load_targets, targets_from_experiment
from constellation.massspec.search.encyclopedia.outputs import expected_names, output_inventory
from constellation.massspec.search.encyclopedia.params import SearchParameters, Tolerance
from constellation.massspec.search.encyclopedia.runner import UnknownFlagError
from constellation.massspec.workflows import encyclopedia as workflow
from constellation.massspec.experiment import summarize
from constellation.massspec.workflows.encyclopedia import (
    WorkflowError,
    read_experiment_table,
    search_acquisition,
)

from _encyclopedia_fixtures import Entry, build_result_library
from test_encyclopedia_dia_codec import _dda_scans, _dia_scans, _write_bundle

_PEPTIDES = [("PEPTIDEK", 2, 5.0e5), ("ELVISLIVESK", 2, 2.0e6), ("[+42.010565]MDFQHRPGGK", 3, 0.0)]


def _bundle(root: Path, name: str, scans=None) -> Path:
    """A converted-bundle directory, manifest included."""
    bundle = _write_bundle(root, name, scans if scans is not None else _dia_scans())
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "thermo_acquisition",
                "source_file": f"/data/{name}.raw",
                "source_sha256": f"sha-of-{name}",
                "run_metadata": {
                    "instrument_model": "Stellar",
                    "instrument_serial": "QLT-S-10199",
                    "creation_date": "10/07/2026 12:15:40",
                },
            }
        )
    )
    return bundle


def _entry(modseq: str, charge: int, total: float, source: str) -> Entry:
    form = parse_encyclopedia_modseq(modseq)
    table, _ = fragment_ladder(form, max_fragment_charge=1, return_tensor=False)
    mz = sorted(table.column("mz_theoretical").to_pylist())
    return Entry(
        modseq=modseq,
        sequence=form.sequence,
        charge=charge,
        precursor_mz=float(precursor_mz(form, charge=charge)),
        mz=mz,
        intensity=[100.0 + i for i in range(len(mz))],
        total_intensity=total,
        source=source,
    )


def _fake_libexport(flags: dict, state) -> types.SimpleNamespace:
    """What the real -libexport does, as far as the workflow can tell: it
    reads each run's spectra file and result from its input directory,
    writes Percolator files into that directory, and writes a report in
    which every peptide is an entry under one run, quantified in every
    run, with the protein scores under the run ``global``."""
    inputs, output = Path(str(flags["-i"])), Path(str(flags["-o"]))
    assert "-f" in flags, "the real jar refuses a report without -f"
    runs = sorted(p.stem for p in inputs.glob("*.dia"))
    assert runs, "no spectra file in the report's input directory"
    for run in runs:
        assert (inputs / f"{run}.dia").is_symlink(), "the spectra file is linked, not copied"
        assert (inputs / f"{run}.dia.elib").is_file(), f"{run}'s result is not beside its spectra"
    stem = output.name.removesuffix(".elib")
    for name in ("features", "results", "decoy"):
        (inputs / f"{stem}_concatenated_{name}.txt").write_text("")
    sources = [f"{run}.raw" for run in runs]
    entries = [
        dataclasses.replace(
            _entry(m, z, total, sources[0]),
            other_runs=[(other, total * (k + 2)) for k, other in enumerate(sources[1:])],
        )
        for m, z, total in state.peptides
    ]
    build_result_library(
        output,
        entries,
        decoy_peptides=[("KEDITPEP", "DECOY_1")],
        source=sources[0],
        protein_source="global",
        metadata={"-ftol": str(flags.get("-ftol", "10")), "-a": str(flags.get("-a"))},
    )
    for suffix in (".peptides.txt", ".proteins.txt", ".diffacto.csv", ".diffacto.samples.lst"):
        output.with_name(output.name + suffix).write_text("")
    return types.SimpleNamespace(
        argv=["java", "-jar", "encyclopedia.jar", "-libexport"],
        elapsed_seconds=0.5,
        returncode=0,
        java_version="openjdk 21",
        jar_sha256="cafe",
    )


#: Peaks a ScribeTwo result stores that are no ion of its peptide: it
#: keeps the whole observed spectrum.
_NOISE_PEAKS = (151.37, 333.91, 777.77)


def _as_scribetwo(entries, settings, flags, source):
    """What a ScribeTwo result looks like where it differs from a
    peptide-centric one: whole observed spectra (here: the ions plus
    noise), ``<source> search type`` ``ScribeTwo...``, and the precursor
    and fragment tolerances it was given replaced by its own estimates."""
    observed = []
    for e in entries:
        mz = list(e.mz) + list(_NOISE_PEAKS)
        intensity = list(e.intensity) + [5.0] * len(_NOISE_PEAKS)
        order = sorted(range(len(mz)), key=mz.__getitem__)
        observed.append(dataclasses.replace(e, mz=[mz[i] for i in order],
                                            intensity=[intensity[i] for i in order]))  # fmt: skip
    kind = {"dia": "DIA"}.get(str(flags.get("-searchType")), "DDA")
    settings = {**settings, f"{source} search type": f"ScribeTwo{kind}", "PercolatorVersion": "JavaPot"}
    for flag in ("-ptol", "-ftol"):
        if flag in flags:
            settings[flag] = str(float(flags[flag]) * 1.2)
    return observed, settings


def _scribe2_reports(given: Path, entries, source: str) -> None:
    """ScribeTwo's reports beside its result, in the captured layout:
    every match in the features file (two per precursor, on real scans of
    the spectra file), the best of each in the results, one decoy."""
    import sqlite3

    con = sqlite3.connect(f"file:{given}?mode=ro", uri=True)
    try:
        scans = con.execute(
            "SELECT SpectrumIndex, ScanStartTime FROM spectra ORDER BY SpectrumIndex"
        ).fetchall()
    finally:
        con.close()
    reports = given.parent / f"{given.name}.scribe2"
    reports.mkdir(exist_ok=True)
    stem = f"{given.name}.scribe2"
    features = ["id\tLabel\tScanNr\tscribeScore\tExpMass\tprecursorMz\tprecursorMass\tsequence\tProteins"]
    results = ["PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tproteinIds"]
    decoys = list(results)
    matches = [(e, False) for e in entries] + [(entries[0], True)]
    for k, (e, decoy) in enumerate(matches):
        spelled = ("decoy" if decoy else "") + e.modseq
        for j in (0, 1):  # the best match, then the same precursor in a later scan
            scan, rt = scans[(2 * k + j) % len(scans)]
            psm = f"{source}:{rt}:{spelled}+{e.charge}"
            label = -1 if decoy else 1
            mass = (e.precursor_mz - 1.007276466812) * e.charge
            features.append(f"{psm}\t{label}\t{scan}\t0.5\t{e.precursor_mz + 0.01}\t{e.precursor_mz}"
                            f"\t{mass}\t-.{e.sequence}.-\t{';'.join(e.proteins)}")  # fmt: skip
            if j == 0:
                row = f"{psm}\t{2.0 - k * 0.1}\t{0.001 * (k + 1)}\t{0.002 * (k + 1)}\t-.{e.sequence}.-\t{';'.join(e.proteins)}"
                (decoys if decoy else results).append(row)
    (reports / f"{stem}.features.txt").write_text("\n".join(features) + "\n")
    (reports / f"{stem}.txt").write_text("\n".join(results) + "\npi_0=null\n")
    (reports / f"{stem}.decoy.txt").write_text("\n".join(decoys) + "\n")


def _percolator_sidecars(given: Path, entries, source: str, columns: dict) -> None:
    """A peptide-centric search's text files, as the jar lays them out:
    Percolator's output with every entry within the threshold, one more
    target beyond it and a decoy; its features; the final retention-time
    fit. (The other files keep the bare header the stand-in writes.)"""
    stem = f"{given.parent / given.name}"
    targets = [(e.modseq, e.charge, 0.001 * (k + 1)) for k, e in enumerate(entries)] + [("ELVISK", 2, 0.4)]
    lines = ["\t".join(columns["<input>.encyclopedia.txt"])]
    lines += [f"{source}:{600 + k}.0:{m}+{z}\t{1.5 - k}\t{q}\t{q / 2}\t-.{m}.-\tP1" for k, (m, z, q) in enumerate(targets)]
    Path(f"{stem}.encyclopedia.txt").write_text("\n".join(lines) + "\npi_0=0.5\n")
    Path(f"{stem}.encyclopedia.decoy.txt").write_text(
        "\t".join(columns["<input>.encyclopedia.decoy.txt"])
        + f"\n{source}:650.0:decoyKEDITPEP+2\t-1.0\t0.5\t0.5\t-.KEDITPEP.-\tDECOY_1\n"
    )
    header = columns["<input>.features.txt"]
    rows = []
    for k, (m, z, _q) in enumerate(targets):
        values = {"id": f"{source}:{600 + k}.0:{m}+{z}", "Label": "1", "ScanNr": str(k),
                  "sequence": f"-.{m}.-", "Proteins": "P1"}  # fmt: skip
        rows.append("\t".join(values.get(c, "0.5") for c in header))
    Path(f"{stem}.features.txt").write_text("\n".join(["\t".join(header), *rows]) + "\n")
    fit = ["\t".join(columns["<input>.encyclopedia.txt.final.rt_fit.txt"])]
    fit += [f"10.0\t{10.0 + k}\t{10.0 + k}\t0.0\ttrue\tfalse\t{e.modseq}" for k, e in enumerate(entries)]
    Path(f"{stem}.encyclopedia.txt.final.rt_fit.txt").write_text("\n".join(fit) + "\n")


def _recorded_settings(program: str, flags: dict) -> dict[str, str]:
    """What the real jar writes into a result's ``metadata``: every
    setting it records, at its documented default unless passed."""
    from constellation.massspec.search.encyclopedia import cli_spec
    from constellation.massspec.search.encyclopedia.outputs import output_inventory
    from constellation.massspec.search.encyclopedia.runner import format_value

    keys = [k for k in output_inventory(program)["result_metadata_keys"] if k.startswith("-")]
    recorded = {k: cli_spec.flag_default(program, k) for k in keys}
    recorded = {k: v for k, v in recorded.items() if v is not None}
    recorded.update({k: format_value(v) for k, v in flags.items() if k in keys})
    return recorded


@pytest.fixture
def jar(monkeypatch):
    """Stand-ins for the jar. ``jar.calls`` lists the runs it was asked for;
    ``jar.source`` overrides the run name it writes into its result,
    ``jar.metadata`` what it records about its settings, ``jar.fail`` an
    exception it raises instead of finishing, and ``jar.while_running`` a
    function called with the search's directory before it writes anything."""
    state = types.SimpleNamespace(
        calls=[], source=None, peptides=_PEPTIDES, metadata=None, fail=None, while_running=None,
        fail_for={}, delay={}, wait_for={}, fasta_seen=[], fatal=None,
    )
    lock = threading.Lock()

    def run_program(program, flags, *, log_dir, cwd, **kwargs):
        from constellation.massspec.search.encyclopedia.runner import compose_argv

        argv = compose_argv(program, flags, allow_undocumented=kwargs.get("allow_undocumented", ()))
        with lock:
            state.calls.append(types.SimpleNamespace(program=program, flags=dict(flags), argv=argv))
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "stdout.log").write_text("EncyclopeDIA version 6.9.23\n")
        if program == "libexport":
            return _fake_libexport(flags, state)
        given = Path(str(flags["-i"]))
        run_name = given.stem
        time.sleep(state.delay.get(run_name, 0.0))
        if run_name in state.wait_for:
            assert state.wait_for[run_name].wait(timeout=30), f"{run_name} waited in vain"
        if state.while_running is not None:
            state.while_running(Path(cwd).parent)
        if state.fail is not None:
            raise state.fail
        if state.fatal is not None:
            # What 6.9.23's ScribeTwo does on a bad setting: says so, exits 0.
            log = Path(log_dir) / "stdout.log"
            log.write_text(f"[00:00:01] Encountered Fatal Error!\n[00:00:01] {state.fatal}\n\tat x.y(Z.java:1)\n")
            return types.SimpleNamespace(argv=["java"], elapsed_seconds=0.1, returncode=0,
                                         java_version="openjdk 21", jar_sha256="cafe", stdout_log=log)  # fmt: skip
        if run_name in state.fail_for:
            raise state.fail_for[run_name]
        assert given.is_file() and given.parent == Path(cwd), "the input is in the search's directory"
        scribetwo = program == "scribetwo"
        result_name = "<input>.dlib" if scribetwo else "<input>.elib"
        names = expected_names(program, given.name)
        result = given.parent / names[result_name]
        if scribetwo and str(flags.get("-l", "")).endswith(".elib"):
            result = result.with_suffix(".elib")  # ScribeTwo's result follows its library
        # Beside the input, named for it. A result already there would be
        # taken for a finished search by the real jar; here the stand-in
        # refuses it, so a rerun over old outputs fails loudly.
        assert not result.exists(), "a result is already there"
        # The real jar copies the spectra file's own record of its source:
        # the .raw for a single run, the merged file itself for a merge.
        with DiaFile(given) as dia:
            recorded = str(dia.metadata.get("sourcename") or f"{given.stem}.raw")
        source = state.source or recorded
        columns = {f["name"]: f.get("columns") for f in output_inventory(program)["files"]}
        for name, file_name in names.items():
            if name != result_name:
                # As the jar writes them: a table's header even with no rows.
                path = given.parent / file_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\t".join(columns[name]) + "\n" if columns.get(name) else "")
        entries = [_entry(m, z, total, source) for m, z, total in state.peptides]
        settings = {**_recorded_settings(program, flags), **(state.metadata or {})}
        if scribetwo:
            entries, settings = _as_scribetwo(entries, settings, flags, source)
            state.fasta_seen.append(Path(str(flags["-f"])))
        build_result_library(
            result,
            entries,
            decoy_peptides=[("KEDITPEP", "DECOY_1")],
            source=source,
            metadata=settings,
        )
        if scribetwo:
            _scribe2_reports(given, entries, source)
        else:
            _percolator_sidecars(given, entries, source, columns)
        return types.SimpleNamespace(
            argv=["java", "-jar", "encyclopedia.jar", *argv],
            elapsed_seconds=1.5,
            returncode=0,
            java_version="openjdk 21",
            jar_sha256="cafe",
        )

    monkeypatch.setattr(workflow, "run_program", run_program)
    monkeypatch.setattr(workflow, "resolve_jar", lambda: types.SimpleNamespace(version="6.9.23"))
    monkeypatch.setattr(workflow, "percolator_blocker", lambda *a, **k: None)
    return state


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    (tmp_path / "db.fasta").write_text(">P1\nPEPTIDEK\n")
    (tmp_path / "lib.dlib").write_bytes(b"library")
    return {"fasta": tmp_path / "db.fasta", "library": tmp_path / "lib.dlib"}


def test_a_bundle_is_searched_into_an_experiment(tmp_path: Path, jar, inputs) -> None:
    bundle = _bundle(tmp_path, "run_01")
    experiment = tmp_path / "experiment"
    params = SearchParameters(
        precursor_tolerance=Tolerance(0.35, "Da"), threads=4, extra={"-fixed": ""}
    )
    result = search_acquisition(bundle, experiment_dir=experiment, parameters=params, **inputs)

    # ── what was run ────────────────────────────────────────────────
    (call,) = jar.calls
    # The search's own directory: named for what was searched, with a
    # digest of how.
    unit = result.unit_dir
    assert re.fullmatch(r"run_01-[0-9a-f]{8}", result.unit)
    assert unit == experiment / "units" / result.unit
    assert call.program == "encyclopedia"
    # The spectra file is written once, outside any search's directory,
    # and the jar is given a link to it in a directory of its own. No -o:
    # the jar writes beside its input, under its own names.
    spectra = experiment / "spectra" / "run_01" / "run_01.dia"
    link = unit / "jar" / "run_01.dia"
    assert spectra.is_file() and not spectra.is_symlink()
    assert (spectra.parent / "_SUCCESS").is_file()
    assert call.flags["-i"] == link and "-o" not in call.flags
    assert link.is_symlink() and link.resolve() == spectra
    assert os.readlink(link) == "../../../spectra/run_01/run_01.dia"  # the experiment can be moved
    assert (unit / "jar" / "run_01.dia.elib").is_file()
    assert sorted(p.name for p in spectra.parent.iterdir()) == [
        "_SUCCESS", "run_01.dia", "run_01.dia.scans.parquet",
    ]  # fmt: skip
    assert call.argv[call.argv.index("-ptolunits") + 1] == "AMU"

    # ── what the directory says it is ───────────────────────────────
    assert manifest_kind(experiment) == "massspec_experiment"
    root = read_manifest(experiment, expect=ExperimentManifest)
    assert root.units == {result.unit: f"units/{result.unit}"} and root.id_scheme == ID_SCHEME
    assert set(root.tables) == {
        "protein", "peptide", "precursor", "fragment", "protein_peptide",
        "precursor_quant", "peptide_quant", "precursor_score", "peptide_score", "protein_score",
        "rt_alignment", "percolator_feature",  # from the sidecars (phase 7)
    }  # fmt: skip
    manifest = read_manifest(unit, expect=SearchManifest)
    assert manifest == result.manifest
    assert (manifest.engine, manifest.engine_version, manifest.program, manifest.search_mode) == (
        "encyclopedia", "6.9.23", "encyclopedia", "peptide-centric",
    )  # fmt: skip
    assert manifest.acquisition_ids == [0]
    assert manifest.search_key.startswith(result.unit.rsplit("-", 1)[1])
    # The binding of the result's rows to the acquisition, recorded.
    assert manifest.sources == {"run_01.raw": 0}
    # Settings as given (Da), next to the command line as run (AMU).
    assert manifest.parameters == {
        "precursor_tolerance": {"value": 0.35, "unit": "Da"},
        "threads": 4,
        "extra": {"-fixed": ""},
    }
    assert "AMU" in manifest.argv and manifest.inputs["fasta"]["sha256"]
    assert manifest.inputs["source"]["acquisition_type"] == "DIA"
    assert manifest.inputs["spectra_file"] == {
        "path": "spectra/run_01/run_01.dia",
        "written_by": "constellation",
        "link": "jar/run_01.dia",
        "linked_as": "symlink",
    }
    assert manifest.outputs["result_library"] == "jar/run_01.dia.elib"
    assert manifest.outputs["<input>.features.txt"] == "jar/run_01.dia.features.txt"
    assert (unit / "_SUCCESS").read_text() == manifest.stage_key != ""

    # ── the acquisition ─────────────────────────────────────────────
    acquisitions = load_acquisitions(experiment / "acquisitions")
    (row,) = acquisitions.table.to_pylist()
    assert (row["acquisition_id"], row["source_file"], row["source_sha256"]) == (
        0, "/data/run_01.raw", "sha-of-run_01",
    )  # fmt: skip
    assert acquisitions.id_for_source("run_01.dia") == 0  # the file handed to the jar

    # ── the tables ──────────────────────────────────────────────────
    # Every precursor scored: the three identified, a decoy, and one beyond
    # the FDR threshold that only Percolator's own output has.
    assert result.table_rows["precursor"] == 3 and result.table_rows["precursor_score"] == 5
    precursors = read_experiment_table(experiment, "precursor")
    peptides = read_experiment_table(experiment, "peptide")
    key = dict(
        zip(
            peptides.column("peptide_id").to_pylist(),
            peptides.column("modified_sequence").to_pylist(),
            strict=True,
        )
    )
    assert "[UNIMOD:1]-MDFQHRPGGK" in key.values()
    assert all(i == content_id(m) for i, m in key.items())
    assert all(
        r["precursor_id"] == content_id(f"{key[r['peptide_id']]}/{r['charge']}")
        for r in precursors.to_pylist()
    )
    quant = read_experiment_table(experiment, "precursor_quant").to_pylist()
    assert sorted(q["intensity"] for q in quant) == [0.0, 5.0e5, 2.0e6]  # the jar's quantities
    assert {(q["unit"], q["acquisition_id"]) for q in quant} == {(result.unit, 0)}
    assert (
        experiment / "tables" / "precursor_quant" / f"unit={result.unit}" / "acquisition_id=0"
    ).is_dir()
    ids = set(precursors.column("precursor_id").to_pylist())
    for table in ("fragment", "precursor_quant"):
        assert set(read_experiment_table(experiment, table).column("precursor_id").to_pylist()) <= ids
    scores = read_experiment_table(experiment, "precursor_score")
    assert pc.sum(scores.column("is_decoy")).as_py() == 1


def test_resume_finds_the_very_same_search(tmp_path: Path, jar, inputs) -> None:
    bundle = _bundle(tmp_path, "run_01")
    experiment = tmp_path / "experiment"
    first = search_acquisition(bundle, experiment_dir=experiment, **inputs)
    again = search_acquisition(bundle, experiment_dir=experiment, resume=True, **inputs)
    assert again.reused and len(jar.calls) == 1
    assert again.manifest == first.manifest and again.unit == first.unit

    # Without resume the same search is run again, in the same directory.
    rerun = search_acquisition(bundle, experiment_dir=experiment, **inputs)
    assert not rerun.reused and len(jar.calls) == 2 and rerun.unit == first.unit
    # How the tables are written is not part of what the search is.
    plain = search_acquisition(bundle, experiment_dir=experiment, annotate_fragments=False, **inputs)
    assert plain.unit == first.unit and plain.manifest.search_key == first.manifest.search_key
    assert plain.manifest.stage_key != first.manifest.stage_key
    # Nor is the number of threads it ran on.
    threaded = search_acquisition(
        bundle, experiment_dir=experiment, resume=True, annotate_fragments=False,
        parameters=SearchParameters(threads=3), **inputs,
    )  # fmt: skip
    assert threaded.reused and threaded.unit == first.unit
    assert sorted(read_manifest(experiment, expect=ExperimentManifest).units) == [first.unit]


def test_another_search_of_an_acquisition_does_not_replace_the_first(tmp_path, jar, inputs) -> None:
    """Every search has its own directory. The same acquisition with
    other settings, or another library, goes beside what is there."""
    bundle = _bundle(tmp_path, "run_01")
    experiment = tmp_path / "experiment"
    first = search_acquisition(bundle, experiment_dir=experiment, **inputs)
    spectra = experiment / "spectra" / "run_01" / "run_01.dia"
    written = spectra.stat()

    def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
        return {
            str(p.relative_to(directory)): (p.lstat().st_ino, p.lstat().st_mtime_ns)
            for p in sorted(directory.rglob("*"))
        }

    before = snapshot(first.unit_dir)
    first_tables = snapshot(experiment / "tables" / "precursor_quant" / f"unit={first.unit}")

    jar.peptides = _PEPTIDES[:2]  # what a narrower search might find
    narrow = search_acquisition(
        bundle,
        experiment_dir=experiment,
        resume=True,  # makes no difference: it is not the same search
        parameters=SearchParameters(precursor_tolerance=Tolerance(10.0)),
        **inputs,
    )
    (inputs["library"].parent / "other.dlib").write_bytes(b"another library")
    other = search_acquisition(
        bundle, experiment_dir=experiment, fasta=inputs["fasta"], library=inputs["library"].parent / "other.dlib"
    )
    assert len({first.unit, narrow.unit, other.unit}) == 3 and len(jar.calls) == 3
    assert (first.acquisition_id, narrow.acquisition_id, other.acquisition_id) == (0, 0, 0)

    # The first search is exactly as it was: every file, and its rows.
    assert snapshot(first.unit_dir) == before
    assert snapshot(experiment / "tables" / "precursor_quant" / f"unit={first.unit}") == first_tables
    assert read_manifest(first.unit_dir, expect=SearchManifest) == first.manifest
    assert set(read_manifest(experiment, expect=ExperimentManifest).units) == {
        first.unit, narrow.unit, other.unit,
    }  # fmt: skip

    # Each has its own rows, read by naming the search.
    quant = read_experiment_table(experiment, "precursor_quant")
    assert quant.num_rows == 3 + 2 + 2
    assert read_experiment_table(experiment, "precursor_quant", unit=first.unit).num_rows == 3
    assert read_experiment_table(experiment, "precursor_quant", unit=narrow.unit).num_rows == 2
    assert read_experiment_table(experiment, "precursor_quant", unit=narrow.unit, acquisition_id=0).num_rows == 2
    assert read_experiment_table(experiment, "precursor_quant", unit=narrow.unit, acquisition_id=5).num_rows == 0

    # It cost searches, not conversions: one spectra file, written once,
    # linked into all three directories.
    assert (spectra.stat().st_ino, spectra.stat().st_mtime_ns) == (written.st_ino, written.st_mtime_ns)
    assert first.manifest.runtime["spectra_file_seconds"] > 0.0
    assert narrow.manifest.runtime["spectra_file_seconds"] == 0.0
    for one in (first, narrow, other):
        assert (one.unit_dir / "jar" / "run_01.dia").resolve() == spectra

    # "The result for acquisition 0" now needs the search named.
    with pytest.raises(ValueError, match="holds 3 results that fit"):
        load_targets(experiment)
    assert load_targets(narrow.unit_dir).num_rows == 2
    assert targets_from_experiment(experiment, unit=first.unit).num_rows == 3

    summary = summarize(experiment)
    assert [s["unit"] for s in summary["searches"]] == sorted([first.unit, narrow.unit, other.unit])
    assert {s["unit"]: s["found"]["precursors"] for s in summary["searches"]}[narrow.unit] == 2


def test_a_run_name_with_awkward_characters_keeps_its_rows(tmp_path: Path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    result = search_acquisition(_bundle(tmp_path, "run 01=a%"), experiment_dir=experiment, **inputs)
    assert result.unit.startswith("run 01=a%-")
    quant = read_experiment_table(experiment, "precursor_quant", unit=result.unit)
    assert quant.num_rows == 3 and set(quant.column("unit").to_pylist()) == {result.unit}
    assert load_targets(result.unit_dir).num_rows == 3


def test_a_unit_is_emptied_before_it_is_searched_again(tmp_path: Path, jar, inputs) -> None:
    """A result beside its input is the jar's sign that the search is
    finished, so a search that is meant to run must find none there. (The
    stand-in refuses to run where a result already is.)"""
    bundle = _bundle(tmp_path, "run_01")
    experiment = tmp_path / "experiment"
    first = search_acquisition(bundle, experiment_dir=experiment, **inputs)
    unit = first.unit_dir
    (unit / "jar" / "left_by_the_first_search.txt").write_text("")
    (unit / "logs" / "old.log").write_text("")
    stale = experiment / "tables" / "retired_table" / f"unit={first.unit}" / "acquisition_id=0"
    stale.mkdir(parents=True)
    (stale / "part-0.parquet").write_bytes(b"")

    # A search that fails after the unit was emptied leaves no marker and
    # no tables: nothing that reads as the old result. Its manifest says
    # what it was and why it stopped.
    jar.source = "stranger.raw"
    with pytest.raises(KeyError, match="stranger.raw"):
        search_acquisition(bundle, experiment_dir=experiment, **inputs)
    assert not (unit / "_SUCCESS").exists()
    failed = read_manifest(unit, expect=SearchManifest)
    assert failed.status == "failed" and "stranger.raw" in failed.error
    assert failed.engine_settings == {} and failed.search_key == first.manifest.search_key
    assert not (unit / "jar" / "left_by_the_first_search.txt").exists()
    assert not (unit / "logs" / "old.log").exists()
    assert not list((experiment / "tables").glob("*/unit=*"))

    jar.source = None
    again = search_acquisition(bundle, experiment_dir=experiment, **inputs)
    assert again.table_rows == first.table_rows
    assert not stale.exists()
    assert read_experiment_table(experiment, "precursor").num_rows == 3


def test_the_experiments_own_spectra_file_can_be_given_as_the_input(tmp_path, jar, inputs) -> None:
    """By its link in a search's directory or by itself: it is the same
    acquisition and, with the same settings, the same search."""
    experiment = tmp_path / "experiment"
    first = search_acquisition(_bundle(tmp_path, "run_01"), experiment_dir=experiment, **inputs)
    link = first.unit_dir / "jar" / "run_01.dia"
    for given in (link, experiment / "spectra" / "run_01" / "run_01.dia"):
        again = search_acquisition(given, experiment_dir=experiment, resume=True, **inputs)
        assert again.reused and again.unit == first.unit and again.acquisition_id == 0
    assert len(jar.calls) == 1
    rerun = search_acquisition(link, experiment_dir=experiment, **inputs)
    assert rerun.unit == first.unit and link.is_symlink() and link.is_file()
    assert rerun.manifest.inputs["spectra_file"] == first.manifest.inputs["spectra_file"]


def test_without_symlinks_the_spectra_file_is_hard_linked_or_copied(tmp_path, jar, inputs, monkeypatch) -> None:
    def no_symlinks(self, target, target_is_directory=False):
        raise OSError("symbolic links are not supported here")

    monkeypatch.setattr(Path, "symlink_to", no_symlinks)
    experiment = tmp_path / "experiment"
    result = search_acquisition(_bundle(tmp_path, "run_01"), experiment_dir=experiment, **inputs)
    spectra = experiment / "spectra" / "run_01" / "run_01.dia"
    link = result.unit_dir / "jar" / "run_01.dia"
    assert result.manifest.inputs["spectra_file"]["linked_as"] == "hardlink"
    assert not link.is_symlink() and link.stat().st_ino == spectra.stat().st_ino

    def no_hard_links(src, dst, **_kwargs):
        raise OSError("hard links are not supported here")

    monkeypatch.setattr(workflow.os, "link", no_hard_links)
    result = search_acquisition(_bundle(tmp_path, "run_02"), experiment_dir=experiment, **inputs)
    link = result.unit_dir / "jar" / "run_02.dia"
    assert result.manifest.inputs["spectra_file"]["linked_as"] == "copy"
    assert link.is_file() and link.stat().st_ino != (experiment / "spectra" / "run_02" / "run_02.dia").stat().st_ino


def test_settings_the_result_records_differently_are_reported(tmp_path: Path, jar, inputs) -> None:
    from constellation.massspec.workflows.encyclopedia import parameter_disagreements

    # What 6.9.23 wrote for a search asked for at 0.35 Da on 12 threads.
    recorded = {"-ptol": "0.35", "-ptolunits": "AMU", "-numberOfThreadsUsed": "12", "-enzyme": "Trypsin", "-fixed": ""}  # fmt: skip
    asked = {"-ptol": 0.35, "-ptolunits": "AMU", "-numberOfThreadsUsed": 12, "-enzyme": "trypsin", "-fixed": "",
             "-i": Path("/x/run.dia"), "-percolatorProteinThreshold": 0.01}  # fmt: skip
    assert parameter_disagreements(recorded, asked) == []
    assert parameter_disagreements({"-ptol": "10"}, {"-ptol": 10.0}) == []
    assert parameter_disagreements(recorded, {**asked, "-ptol": 0.2}) == [
        "the result records -ptol '0.35' but the search asked for '0.2'"
    ]

    jar.metadata = {"-ptol": "0.35", "-ptolunits": "AMU"}
    result = search_acquisition(
        _bundle(tmp_path, "run_01"),
        experiment_dir=tmp_path / "experiment",
        parameters=SearchParameters(precursor_tolerance=Tolerance(10.0)),
        **inputs,
    )
    assert result.manifest.caveats == [
        "the result records -ptol '0.35' but the search asked for '10.0'",
        "the result records -ptolunits 'AMU' but the search asked for 'PPM'",
    ]
    assert list(result.disagreements) == result.manifest.caveats


def test_a_second_acquisition_joins_the_experiment(tmp_path: Path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    one = search_acquisition(_bundle(tmp_path, "run_01"), experiment_dir=experiment, **inputs)
    two = search_acquisition(_bundle(tmp_path, "run_02"), experiment_dir=experiment, **inputs)
    assert (one.acquisition_id, two.acquisition_id) == (0, 1)
    assert set(read_manifest(experiment, expect=ExperimentManifest).units) == {one.unit, two.unit}
    assert one.unit.startswith("run_01-") and two.unit.startswith("run_02-")

    quant = read_experiment_table(experiment, "precursor_quant")
    assert sorted(pc.unique(quant.column("acquisition_id")).to_pylist()) == [0, 1]
    assert read_experiment_table(experiment, "precursor_quant", acquisition_id=1).num_rows == 3
    # The same precursor has the same id under both acquisitions.
    by_run = [
        set(read_experiment_table(experiment, "precursor", acquisition_id=i).column("precursor_id").to_pylist())
        for i in (0, 1)
    ]
    assert by_run[0] == by_run[1] and len(by_run[0]) == 3

    # A target list is for one run: the experiment alone is ambiguous now,
    # a search's directory is not, and neither is naming the acquisition.
    with pytest.raises(ValueError, match="holds 2 results that fit"):
        load_targets(experiment)
    assert targets_from_experiment(experiment, acquisition_id=1).equals(load_targets(two.unit_dir))
    targets = load_targets(two.unit_dir)
    assert targets.num_rows == 3
    row = targets.filter(pc.equal(targets.column("modified_sequence"), "PEPTIDEK")).to_pylist()[0]
    assert (row["rt_center"], row["rt_start"], row["rt_end"]) == (600.0, 590.0, 610.0)
    assert row["target_id"] == row["precursor_id"] == content_id("PEPTIDEK/2")


def test_a_searched_experiment_is_a_target_list(tmp_path: Path, jar, inputs) -> None:
    """What `chromatogram extract --targets` takes: the experiment while
    it holds one acquisition, or a unit directory at any time."""
    from constellation.massspec.quant.schemas import XIC_TARGET_TABLE

    experiment = tmp_path / "experiment"
    result = search_acquisition(_bundle(tmp_path, "run_01"), experiment_dir=experiment, **inputs)
    targets = load_targets(experiment)
    assert targets.schema.equals(XIC_TARGET_TABLE, check_metadata=False)
    assert targets.equals(load_targets(result.unit_dir))
    assert targets.equals(targets_from_experiment(experiment, acquisition_id=0))
    # Identified but not quantified is still somewhere to look.
    assert sorted(targets.column("modified_sequence").to_pylist()) == [
        "ELVISLIVESK", "PEPTIDEK", "[UNIMOD:1]-MDFQHRPGGK",
    ]  # fmt: skip
    assert targets.equals(targets_from_experiment(experiment, unit=result.unit))
    with pytest.raises(ValueError, match="no results for acquisition 7"):
        targets_from_experiment(experiment, acquisition_id=7)
    with pytest.raises(ValueError, match="no results for search nonesuch"):
        targets_from_experiment(experiment, unit="nonesuch")


def test_a_raw_file_is_converted_into_the_experiment_once(tmp_path, jar, inputs, monkeypatch) -> None:
    """The Thermo reader is stood in for; what is checked is where the
    bundle goes, that a finished one is not converted again, and that an
    unfinished one (no manifest) is."""
    import shutil

    from constellation.massspec.readers import thermo

    template = _bundle(tmp_path / "template", "run_01")
    converted = []

    def convert(source, output_dir, *, force=False, **_kwargs):
        converted.append((Path(source).name, force))
        shutil.copytree(template, output_dir, dirs_exist_ok=True)

    monkeypatch.setattr(thermo, "convert", convert)
    raw = tmp_path / "run_01.raw"
    raw.write_bytes(b"")
    experiment = tmp_path / "experiment"

    first = search_acquisition(raw, experiment_dir=experiment, **inputs)
    bundle = experiment / "bundles" / "run_01"
    assert (bundle / "manifest.json").is_file() and converted == [("run_01.raw", True)]
    assert first.manifest.inputs["source"]["kind"] == "bundle"
    assert first.manifest.sources == {"run_01.raw": 0}

    search_acquisition(raw, experiment_dir=experiment, **inputs)
    assert len(converted) == 1  # the bundle is there already

    (bundle / "manifest.json").unlink()  # as an interrupted conversion leaves it
    search_acquisition(raw, experiment_dir=experiment, **inputs)
    assert len(converted) == 2

    def no_dlls(*_args, **_kwargs):
        raise ImportError("the Thermo DLLs are not installed; run scripts/install-thermo-dlls.sh")

    monkeypatch.setattr(thermo, "convert", no_dlls)
    other = tmp_path / "run_09.raw"
    other.write_bytes(b"")
    with pytest.raises(WorkflowError, match="cannot read run_09.raw: the Thermo DLLs are not installed"):
        search_acquisition(other, experiment_dir=experiment, **inputs)


def test_a_given_spectra_file_is_linked_in_and_left_alone(tmp_path: Path, jar, inputs) -> None:
    """A spectra file from elsewhere is not copied and nothing is written
    beside it; a finished search sitting next to it (as in any directory
    EncyclopeDIA has already been run in) is not taken for this one."""
    bundle = _bundle(tmp_path, "run_01")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    spectra = write_dia([bundle], elsewhere / "run_01.dia").path
    (elsewhere / "run_01.dia.elib").write_bytes(b"someone else's finished search")
    before = sorted(p.name for p in elsewhere.iterdir())

    experiment = tmp_path / "experiment"
    result = search_acquisition(spectra, experiment_dir=experiment, **inputs)
    link = result.unit_dir / "jar" / "run_01.dia"
    assert jar.calls[0].flags["-i"] == link
    assert link.is_symlink() and os.readlink(link) == str(spectra)  # absolute: it is outside
    assert sorted(p.name for p in elsewhere.iterdir()) == before
    assert not (experiment / "spectra").exists()  # nothing was written for it
    assert result.manifest.inputs["spectra_file"] == {
        "path": str(spectra),
        "written_by": "given",
        "link": "jar/run_01.dia",
        "linked_as": "symlink",
    }
    assert result.table_rows["precursor"] == 3
    (row,) = load_acquisitions(experiment / "acquisitions").table.to_pylist()
    assert row["source_file"] == "run_01.raw" and row["source_aliases"] == ["run_01.dia"]
    assert result.manifest.sources == {"run_01.raw": 0}


def test_what_cannot_be_searched_is_refused_before_anything_runs(tmp_path, jar, inputs, monkeypatch) -> None:
    experiment = tmp_path / "experiment"
    (tmp_path / "run.mzML").write_text("")
    with pytest.raises(WorkflowError, match="mzML input is not supported yet"):
        search_acquisition(tmp_path / "run.mzML", experiment_dir=experiment, **inputs)
    with pytest.raises(WorkflowError, match="no such input"):
        search_acquisition(tmp_path / "absent.raw", experiment_dir=experiment, **inputs)

    dia = _bundle(tmp_path, "dia_run")
    with pytest.raises(WorkflowError, match="needs --library"):
        search_acquisition(dia, experiment_dir=experiment, fasta=inputs["fasta"])
    with pytest.raises(WorkflowError, match="--library not found"):
        search_acquisition(dia, experiment_dir=experiment, fasta=inputs["fasta"], library=tmp_path / "no.dlib")
    with pytest.raises(UnknownFlagError, match="-ptoll"):
        search_acquisition(
            dia, experiment_dir=experiment, parameters=SearchParameters(extra={"-ptoll": "1"}), **inputs
        )

    # A DDA run defaults to the spectrum-centric engine; asking for the
    # peptide-centric one on it is simply wrong, as are the spectrum-centric
    # engine's own settings on a peptide-centric search.
    dda = _bundle(tmp_path, "dda_run", _dda_scans())
    with pytest.raises(ValueError, match="no peptide-centric search of DDA"):
        search_acquisition(dda, experiment_dir=experiment, search_mode="peptide-centric", **inputs)
    with pytest.raises(WorkflowError, match="--ptm belong to the spectrum-centric search"):
        search_acquisition(
            dia, experiment_dir=experiment, parameters=SearchParameters(ptms={"Oxidation": "var"}),
            **inputs,
        )  # fmt: skip

    monkeypatch.setattr(workflow, "percolator_blocker", lambda *a, **k: "Percolator cannot run here")
    with pytest.raises(WorkflowError, match="Percolator cannot run here"):
        search_acquisition(dia, experiment_dir=experiment, **inputs)
    assert jar.calls == []

    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "notes.txt").write_text("mine")
    monkeypatch.setattr(workflow, "percolator_blocker", lambda *a, **k: None)
    with pytest.raises(WorkflowError, match="is not an experiment directory"):
        search_acquisition(dia, experiment_dir=tmp_path / "busy", **inputs)


def test_a_result_that_names_another_run_is_refused(tmp_path: Path, jar, inputs) -> None:
    """The unit searched run_02; a result whose rows say run_01 must not
    be filed under either acquisition."""
    experiment = tmp_path / "experiment"
    search_acquisition(_bundle(tmp_path, "run_01"), experiment_dir=experiment, **inputs)
    jar.source = "run_01.raw"
    with pytest.raises(WorkflowError, match="but this unit searched acquisition 1"):
        search_acquisition(_bundle(tmp_path, "run_02"), experiment_dir=experiment, **inputs)
    assert not list((experiment / "units").glob("run_02-*/_SUCCESS"))
    # And a run the experiment has never heard of cannot be bound at all.
    jar.source = "stranger.raw"
    with pytest.raises(KeyError, match="no acquisition is named 'stranger.raw'"):
        search_acquisition(_bundle(tmp_path, "run_03"), experiment_dir=experiment, **inputs)


def test_the_search_command_drives_the_workflow(tmp_path: Path, jar, inputs, capsys) -> None:
    from constellation.cli.__main__ import main

    bundle = _bundle(tmp_path, "run_01")
    argv = [
        "massspec", "search",
        "--input", str(bundle),
        "--library", str(inputs["library"]),
        "--fasta", str(inputs["fasta"]),
        "--output-dir", str(tmp_path / "experiment"),
        "--precursor-tolerance", "0.35", "--precursor-tolerance-unit", "Da",
        "--encyclopedia-arg=-fixed=",
        "--no-progress",
    ]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "peptide-centric search of run_01 (acquisition 0): 3 precursors" in out
    assert jar.calls[0].flags["-fixed"] == "" and jar.calls[0].flags["-ptolunits"] == "AMU"

    assert main([*argv, "--resume"]) == 0
    assert "already searched with these inputs and settings" in capsys.readouterr().out
    assert len(jar.calls) == 1

    # Refusals are exit status 1 with the reason, and nothing runs.
    (tmp_path / "run.mzML").write_text("")
    bad = [a if a != str(bundle) else str(tmp_path / "run.mzML") for a in argv]
    assert main(bad) == 1
    assert "mzML input is not supported yet" in capsys.readouterr().err
    assert main([*argv, "--encyclopedia-arg=-bogus=1"]) == 1
    assert "does not document these flags" in capsys.readouterr().err
    assert main([*argv, "--encyclopedia-arg=-o=/elsewhere/report.txt"]) == 1
    assert "-o is not passed through" in capsys.readouterr().err
    assert main([*argv, "--ptm", "Oxidation=var"]) == 1
    assert "(--ptm) and -fixed / -variable do not mix" in capsys.readouterr().err
    no_fixed = [a for a in argv if a != "--encyclopedia-arg=-fixed="]
    assert main([*no_fixed, "--ptm", "Oxidation=var"]) == 1
    assert "--ptm belong to the spectrum-centric search" in capsys.readouterr().err
    assert main([*argv, "--search-mode", "spectrum-centric", "--ptm", "Oxidation"]) == 1
    assert "write it as NAME=STATE" in capsys.readouterr().err
    assert len(jar.calls) == 1

    # A result that records other settings than the ones given is said so.
    jar.metadata = {"-ptol": "10", "-ptolunits": "PPM"}
    assert main(argv) == 0
    err = capsys.readouterr().err
    assert "warning: the result records -ptol '10' but the search asked for '0.35'" in err
    assert "check the jar's log before using it" in err
