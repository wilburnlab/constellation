# ruff: noqa: F811  (the stand-in jar's fixtures are imported, then named as arguments)
"""Phase 5: many acquisitions, merged fractions, the report, the chain.

The jar is the stand-in of ``test_massspec_search_workflow`` (search and
``-libexport``); everything around it is real. The real-data checks of
the same paths are recorded in docs/plans/encyclopedia-integration.md.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pyarrow.compute as pc
import pytest

from constellation.core.manifests import read_manifest
from constellation.massspec.acquisitions import MERGED_KIND, load_acquisitions
from constellation.massspec.experiment import format_summary, summarize
from constellation.massspec.io.encyclopedia import DiaFile
from constellation.massspec.manifest import (
    ALL_ACQUISITIONS,
    ExperimentManifest,
    QuantReportManifest,
    SearchManifest,
)
from constellation.massspec.search.encyclopedia.params import SearchParameters, Tolerance
from constellation.massspec.workflows import encyclopedia as workflow
from constellation.massspec.workflows.encyclopedia import (
    CollisionSettings,
    WorkflowError,
    quantify,
    read_experiment_table,
    report_searches,
    search_acquisitions,
)
from constellation.thirdparty.jvm import JvmRunError

from test_massspec_search_workflow import _PEPTIDES, _bundle, inputs, jar  # noqa: F401


def _bundles(root: Path, *names: str) -> list[Path]:
    return [_bundle(root, name) for name in names]


# ── several acquisitions ────────────────────────────────────────────────


def test_acquisitions_are_numbered_in_the_order_given_however_they_finish(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    bundles = _bundles(tmp_path, "run_a", "run_b", "run_c")
    done = {name: threading.Event() for name in ("run_a", "run_b", "run_c")}
    # run_c finishes first and run_a last: each waits for the one after it.
    jar.wait_for = {"run_a": done["run_b"], "run_b": done["run_c"]}
    order: list[str] = []

    def progress(event, plan, outcome):
        order.append(f"{event}:{plan.run}")
        if event == "done":
            done[plan.run].set()

    fan = search_acquisitions(bundles, experiment_dir=experiment, jobs=3, progress=progress, **inputs)
    assert fan.ok and len(fan.results) == 3
    finished = [e.split(":")[1] for e in order if e.startswith("done")]
    assert finished == ["run_c", "run_b", "run_a"]
    # Results come back in the order given, under ids in the order given.
    assert [(r.run, r.acquisition_id) for r in fan.results] == [("run_a", 0), ("run_b", 1), ("run_c", 2)]
    # Each search registered itself in the experiment without losing the
    # others' entries, though they finished at once.
    root = read_manifest(experiment, expect=ExperimentManifest)
    assert set(root.units) == {r.unit for r in fan.results}
    quant = read_experiment_table(experiment, "precursor_quant")
    assert sorted(pc.unique(quant.column("acquisition_id")).to_pylist()) == [0, 1, 2]


def test_a_failed_search_does_not_stop_the_others_and_resume_reruns_only_it(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    bundles = _bundles(tmp_path, "run_a", "run_b", "run_c")
    jar.fail_for = {"run_b": JvmRunError(1, ["java"], "boom", tmp_path / "e.log")}
    fan = search_acquisitions(bundles, experiment_dir=experiment, jobs=2, **inputs)
    assert not fan.ok
    assert [r.run for r in fan.results] == ["run_a", "run_c"]
    ((plan, exc),) = fan.failures
    assert plan.run == "run_b" and isinstance(exc, JvmRunError)
    failed = read_manifest(experiment / "units" / plan.unit, expect=SearchManifest)
    assert failed.status == "failed" and "the jar exited 1" in failed.error

    jar.fail_for = {}
    calls = len(jar.calls)
    again = search_acquisitions(bundles, experiment_dir=experiment, jobs=2, resume=True, **inputs)
    assert again.ok and len(jar.calls) == calls + 1  # only run_b ran
    assert [r.reused for r in again.results] == [True, False, True]
    assert [r.unit for r in again.results] == [r.unit for r in fan.results[:1]] + [plan.unit] + [fan.results[1].unit]


def test_the_same_input_twice_is_refused(tmp_path, jar, inputs) -> None:
    (bundle,) = _bundles(tmp_path, "run_a")
    with pytest.raises(WorkflowError, match="run_a is given twice"):
        search_acquisitions([bundle, bundle], experiment_dir=tmp_path / "e", **inputs)


# ── merged fractions ────────────────────────────────────────────────────


def test_fractions_are_merged_into_one_acquisition_that_names_them(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    fractions = _bundles(tmp_path, "gpf-400-500_x", "gpf-500-600_x", "gpf-600-700_x")
    fan = search_acquisitions(fractions, experiment_dir=experiment, merge=True, **inputs)
    (result,) = fan.results
    assert result.run == "gpf-merged_x"
    acquisitions = load_acquisitions(experiment / "acquisitions").table.to_pylist()
    assert [(a["acquisition_id"], a["source_kind"], a["merged_from"]) for a in acquisitions] == [
        (0, "thermo_acquisition", None),
        (1, "thermo_acquisition", None),
        (2, "thermo_acquisition", None),
        (3, MERGED_KIND, [0, 1, 2]),
    ]
    # The result names the merged file, and is filed under the merged acquisition.
    assert result.acquisition_id == 3 and result.manifest.sources == {"gpf-merged_x.dia": 3}
    assert [s["acquisition_id"] for s in result.manifest.inputs["sources"]] == [0, 1, 2]
    spectra = experiment / "spectra" / "gpf-merged_x" / "gpf-merged_x.dia"
    with DiaFile(spectra) as dia:
        assert len(dia.fraction_ids()) == 3
    assert jar.calls[0].flags["-i"] == result.unit_dir / "jar" / "gpf-merged_x.dia"
    quant = read_experiment_table(experiment, "precursor_quant", unit=result.unit)
    assert set(quant.column("acquisition_id").to_pylist()) == {3}

    # The same merge again is the same search; another order is another merge.
    again = search_acquisitions(fractions, experiment_dir=experiment, merge=True, resume=True, **inputs)
    assert again.results[0].reused
    with pytest.raises(ValueError, match="cannot also be a merge of"):
        search_acquisitions(list(reversed(fractions)), experiment_dir=experiment, merge=True, **inputs)
    named = search_acquisitions(
        list(reversed(fractions)), experiment_dir=experiment, merge=True, merged_name="pool", **inputs
    )
    assert named.results[0].run == "pool"


def test_what_cannot_be_merged_is_refused(tmp_path, jar, inputs) -> None:
    from constellation.massspec.io.encyclopedia import write_dia

    (only,) = _bundles(tmp_path, "run_a")
    with pytest.raises(WorkflowError, match="at least two"):
        search_acquisitions([only], experiment_dir=tmp_path / "e", merge=True, **inputs)
    spectra = write_dia([only], tmp_path / "given.dia").path
    (other,) = _bundles(tmp_path, "run_b")
    with pytest.raises(WorkflowError, match="these are spectra files: given.dia"):
        search_acquisitions([other, spectra], experiment_dir=tmp_path / "e", merge=True, **inputs)
    with pytest.raises(WorkflowError, match="only applies to a merged search"):
        search_acquisitions([other], experiment_dir=tmp_path / "e", merged_name="x", **inputs)


def test_merged_names_are_cut_at_separators() -> None:
    from constellation.massspec.workflows.encyclopedia import _merged_name

    assert _merged_name(["261006_u2os_dia_gpf-400-500_100min-IOa_HCD26",
                         "261006_u2os_dia_gpf-500-600_100min-IOa_HCD26",
                         "261006_u2os_dia_gpf-900-1000_100min-IOa_HCD26"]) == (
        "261006_u2os_dia_gpf-merged_100min-IOa_HCD26"
    )  # fmt: skip
    assert _merged_name(["a", "b"]) == "merged"


# ── the report ──────────────────────────────────────────────────────────


def test_a_report_quantifies_every_run_and_keeps_run_agnostic_scores_apart(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    fan = search_acquisitions(_bundles(tmp_path, "run_a", "run_b"), experiment_dir=experiment, **inputs)
    searches_before = {
        r.unit: sorted((p.name, p.stat().st_mtime_ns) for p in (r.unit_dir / "jar").iterdir())
        for r in fan.results
    }
    report = report_searches(experiment, fasta=inputs["fasta"])

    # What the jar was given.
    call = jar.calls[-1]
    assert call.program == "libexport" and call.flags["-f"] == inputs["fasta"].resolve()
    assert call.flags["-l"] == inputs["library"].resolve() and call.flags["-a"] is True
    assert call.flags["-o"] == report.report_dir / "report.elib"
    # The report's own directory: links in, its files beside the report.
    assert report.key.startswith("report-") and report.report_dir == experiment / "reports" / report.key
    assert (report.report_dir / "inputs" / "report_concatenated_features.txt").is_file()
    for r in fan.results:
        assert sorted((p.name, p.stat().st_mtime_ns) for p in (r.unit_dir / "jar").iterdir()) == searches_before[r.unit]

    manifest = report.manifest
    assert isinstance(manifest, QuantReportManifest) and manifest.status == "finished"
    assert manifest.units == sorted(r.unit for r in fan.results)
    assert manifest.acquisition_ids == [0, 1]
    assert manifest.sources == {"run_a.raw": 0, "run_b.raw": 1}
    assert {"report", "report.elib.peptides.txt", "report.elib.proteins.txt"} <= set(manifest.outputs)
    root = read_manifest(experiment, expect=ExperimentManifest)
    assert root.reports == {report.key: f"reports/{report.key}"}

    # Every precursor quantified in every run: the entries name run_a, the
    # quantities both.
    quant = read_experiment_table(experiment, "precursor_quant", unit=report.key)
    by_run = quant.group_by("acquisition_id").aggregate([("intensity", "sum")]).sort_by("acquisition_id")
    total = sum(t for _, _, t in _PEPTIDES)
    assert by_run.to_pylist() == [
        {"acquisition_id": 0, "intensity_sum": total},
        {"acquisition_id": 1, "intensity_sum": 2 * total},
    ]
    # Protein scores span every run, so do the library tables.
    proteins = read_experiment_table(experiment, "protein_score", unit=report.key)
    assert set(proteins.column("acquisition_id").to_pylist()) == {ALL_ACQUISITIONS}
    precursors = read_experiment_table(experiment, "precursor", unit=report.key)
    assert set(precursors.column("acquisition_id").to_pylist()) == {ALL_ACQUISITIONS}

    # Resume finds it; without, it is made again in its own directory.
    again = report_searches(experiment, fasta=inputs["fasta"], resume=True)
    assert again.reused and again.key == report.key and len(jar.calls) == 3
    remade = report_searches(experiment, fasta=inputs["fasta"])
    assert remade.key == report.key and len(jar.calls) == 4
    assert read_experiment_table(experiment, "precursor_quant", unit=report.key).num_rows == quant.num_rows

    # A report's precursors are shared by its runs; targets for one run find them.
    from constellation.massspec.quant.targets import targets_from_experiment

    targets = targets_from_experiment(experiment, unit=report.key, acquisition_id=1)
    assert targets.num_rows == len(_PEPTIDES)

    summary = summarize(experiment)
    (listed,) = summary["reports"]
    assert listed["key"] == report.key and listed["status"] == "finished"
    assert listed["found"] == {"precursors": 3, "proteins": 1}
    full = format_summary(summarize(report.report_dir))
    assert f"report {report.key}" in full and "acquisition 1 (run_b.raw)" in full


def test_what_a_report_cannot_be_made_from_is_refused(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    a, b = _bundles(tmp_path, "run_a", "run_b")
    one = search_acquisitions([a], experiment_dir=experiment, **inputs).results[0]
    (inputs["library"].parent / "other.dlib").write_bytes(b"another library")
    search_acquisitions([b], experiment_dir=experiment, fasta=inputs["fasta"],
                        library=inputs["library"].parent / "other.dlib")  # fmt: skip
    with pytest.raises(WorkflowError, match="used 2"):
        report_searches(experiment, fasta=inputs["fasta"])
    # Narrowed to one library, it can be made.
    assert report_searches(experiment, fasta=inputs["fasta"], library=inputs["library"]).manifest.units == [one.unit]
    with pytest.raises(WorkflowError, match="not finished searches of experiment: nonesuch"):
        report_searches(experiment, fasta=inputs["fasta"], searches=["nonesuch"])
    # The same run searched twice cannot be in one report.
    twice = search_acquisitions([a], experiment_dir=experiment,
                                parameters=SearchParameters(precursor_tolerance=Tolerance(5.0)), **inputs)  # fmt: skip
    with pytest.raises(WorkflowError, match="are both searches of acquisition 0"):
        report_searches(experiment, fasta=inputs["fasta"], searches=[one.unit, twice.results[0].unit])


def test_a_report_is_made_from_the_library_contents_its_searches_used(tmp_path, jar, inputs) -> None:
    """One library is one file's contents: searches made before and after
    the file at the library's path changed are two libraries, and a
    report is refused once the file is no longer the one searched."""
    experiment = tmp_path / "experiment"
    a, b = _bundles(tmp_path, "run_a", "run_b")
    library = inputs["library"]
    first = search_acquisitions([a], experiment_dir=experiment, **inputs).results[0]
    library.write_bytes(library.read_bytes() + b", changed")
    second = search_acquisitions([b], experiment_dir=experiment, **inputs).results[0]
    with pytest.raises(WorkflowError, match="used 2"):
        report_searches(experiment, fasta=inputs["fasta"])
    with pytest.raises(WorkflowError, match="has changed since the searches used it"):
        report_searches(experiment, fasta=inputs["fasta"], searches=[first.unit])
    report = report_searches(experiment, fasta=inputs["fasta"], searches=[second.unit])
    assert report.manifest.inputs["library"]["sha256"] == second.manifest.inputs["library"]["sha256"]


def test_a_report_does_not_take_its_files_as_pass_through_flags(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    search_acquisitions(_bundles(tmp_path, "run_a"), experiment_dir=experiment, **inputs)
    for flag in ("-l", "-f", "-i", "-o", "-a"):
        with pytest.raises(WorkflowError, match=f"{flag} is not passed through"):
            report_searches(experiment, fasta=inputs["fasta"], extra={flag: "x"})


def test_a_report_refuses_searches_that_disagree_on_a_setting_it_takes(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    a, b = _bundles(tmp_path, "run_a", "run_b")
    search_acquisitions([a], experiment_dir=experiment,
                        parameters=SearchParameters(fragment_tolerance=Tolerance(0.35, "Da")), **inputs)  # fmt: skip
    search_acquisitions([b], experiment_dir=experiment,
                        parameters=SearchParameters(fragment_tolerance=Tolerance(0.5, "Da")), **inputs)  # fmt: skip
    with pytest.raises(WorkflowError, match="the searches disagree on -ftol"):
        report_searches(experiment, fasta=inputs["fasta"])


# ── the chain ───────────────────────────────────────────────────────────


def test_quantify_is_the_steps_run_in_order(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    fractions = _bundles(tmp_path, "gpf-400-500_x", "gpf-500-600_x")
    runs = _bundles(tmp_path, "run_a", "run_b")
    result = quantify(experiment, runs=runs, fractions=fractions, jobs=2, **inputs)

    programs = [c.program for c in jar.calls]
    assert programs == ["encyclopedia"] * 3 + ["libexport"]
    library_search = result.library_search
    chromatogram_library = library_search.unit_dir / "jar" / "gpf-merged_x.dia.elib"
    assert result.library == chromatogram_library and result.collision_filtered is None
    assert jar.calls[0].flags["-l"] == inputs["library"].resolve()  # fractions: the given library
    assert {c.flags["-l"] for c in jar.calls[1:3]} == {chromatogram_library}  # runs: the chromatogram library
    assert result.runs.ok and [r.run for r in result.runs.results] == ["run_a", "run_b"]
    assert result.report.manifest.units == sorted(r.unit for r in result.runs.results)
    assert jar.calls[3].flags["-l"] == chromatogram_library

    # Chained or by hand, the steps are the same searches: everything is found.
    calls = len(jar.calls)
    search_acquisitions(fractions, experiment_dir=experiment, merge=True, resume=True, **inputs)
    search_acquisitions(runs, experiment_dir=experiment, resume=True, fasta=inputs["fasta"],
                        library=chromatogram_library)  # fmt: skip
    again = quantify(experiment, runs=runs, fractions=fractions, resume=True, **inputs)
    assert len(jar.calls) == calls and again.report.reused

    summary = summarize(experiment)
    assert [a["merged_from"] for a in summary["acquisitions"]][2] == [0, 1]
    assert {s["status"] for s in summary["searches"]} == {"finished"} and len(summary["reports"]) == 1
    assert "merged from 0, 1" in format_summary(summary)


def test_quantify_waits_with_the_report_until_every_run_is_searched(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    runs = _bundles(tmp_path, "run_a", "run_b")
    jar.fail_for = {"run_b": RuntimeError("disk full")}
    result = quantify(experiment, runs=runs, **inputs)
    assert result.library_search is None and result.library == inputs["library"].resolve()
    assert result.report is None and [p.run for p, _ in result.runs.failures] == ["run_b"]
    assert "libexport" not in [c.program for c in jar.calls]
    jar.fail_for = {}
    resumed = quantify(experiment, runs=runs, resume=True, **inputs)
    assert resumed.report is not None and [r.reused for r in resumed.runs.results] == [True, False]


def test_the_collision_filter_runs_only_when_asked(tmp_path, jar, inputs, monkeypatch) -> None:
    from constellation.massspec.search import collision

    seen = {}

    def apply(elib, dia, **kwargs):
        seen.update(elib=Path(elib), dia=Path(dia), **kwargs)
        return {"PEPTIDEK"}, {"n_windows": 1, "n_entries": 3, "clusters": [{"winner": "X", "losers": ["PEPTIDEK"]}]}

    def filter_by(elib, losers, out):
        Path(out).write_bytes(Path(elib).read_bytes())
        return {"entries": len(losers)}

    monkeypatch.setattr(collision, "apply_collision_filter", apply)
    monkeypatch.setattr(collision, "filter_elib_by_losers", filter_by)
    experiment = tmp_path / "experiment"
    fractions = _bundles(tmp_path, "gpf-1_x", "gpf-2_x")
    runs = _bundles(tmp_path, "run_a")
    settings = CollisionSettings(fragment_ppm=500.0)
    result = quantify(experiment, runs=runs, fractions=fractions, collision=settings, **inputs)
    filtered = result.collision_filtered
    assert filtered == result.library_search.unit_dir / "collision" / "gpf-merged_x.dia.collision-filtered.elib"
    assert result.library == filtered and jar.calls[1].flags["-l"] == filtered
    assert seen["frag_ppm_tol"] == 500.0 and seen["dia"].name == "gpf-merged_x.dia"
    record = json.loads((filtered.parent / "collision.json").read_text())
    assert record["n_losers"] == 1 and record["losers"] == ["PEPTIDEK"]
    with pytest.raises(WorkflowError, match="give fractions"):
        quantify(experiment, runs=runs, collision=settings, **inputs)


# ── concurrency, detaching, cancellation ────────────────────────────────


def test_the_file_lock_keeps_threads_apart(tmp_path) -> None:
    from constellation.core.locks import file_lock

    inside: list[int] = []
    overlaps = []

    def hold(i: int) -> None:
        with file_lock(tmp_path / ".lock"):
            inside.append(i)
            if len(inside) > 1:
                overlaps.append(list(inside))
            time.sleep(0.05)
            inside.remove(i)

    threads = [threading.Thread(target=hold, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []


def test_a_detached_run_is_followed_to_its_exit_status(tmp_path) -> None:
    import io

    from constellation.core.detach import follow, latest, launch

    script = "import sys, time; print('one', flush=True); time.sleep(0.4); print('two'); sys.exit(3)"
    run = launch([sys.executable, "-c", script], run_dir=tmp_path / "runs", label="probe")
    assert run.state() in ("running", "failed")
    out = io.StringIO()
    assert follow(run, out=out, poll_s=0.05) == 3
    assert out.getvalue() == "one\ntwo\n"
    assert latest(tmp_path / "runs").state() == "failed"


def test_a_detached_run_stopped_by_its_group_still_records_its_status(tmp_path) -> None:
    """kill -TERM -<pid> reaches the wrapping shell too; it must outlive
    the command to write the status, and the command must still get
    the signal (a trap, not an ignore, so it is not inherited)."""
    import signal

    from constellation.core.detach import follow, launch

    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    run = launch([sys.executable, "-c", script], run_dir=tmp_path / "runs", label="probe")
    for _ in range(200):
        if run.log.exists() and "ready" in run.log.read_text():
            break
        time.sleep(0.05)
    os.killpg(run.pid, signal.SIGTERM)
    assert follow(run, out=open(os.devnull, "w"), poll_s=0.05) == 143
    assert run.state() == "failed"


def test_detach_starts_the_same_command_again_without_it(tmp_path, monkeypatch, capsys) -> None:
    from constellation.cli.__main__ import main
    from constellation.core import detach

    launched = {}

    def fake_launch(argv, *, run_dir, label, env=None):
        launched.update(argv=argv, run_dir=run_dir, label=label, env=env)
        return detach.DetachedRun(
            record=run_dir / "x.json", pid=4242, argv=tuple(argv), log=run_dir / "x.log",
            exit_file=run_dir / "x.exit", started_at="now", label=label,
        )  # fmt: skip

    monkeypatch.setattr(detach, "launch", fake_launch)
    experiment = tmp_path / "experiment"
    argv = ["massspec", "report", "--output-dir", str(experiment), "--fasta", "f.fasta", "--detach"]
    assert main(argv) == 0
    assert launched["argv"][1:3] == ["-m", "constellation.cli"]
    assert launched["argv"][3:] == argv[:-1]  # --detach removed
    assert launched["run_dir"] == experiment.resolve() / "runs" and launched["label"] == "report"
    assert launched["env"]["CONSTELLATION_DETACHED"] == "1"
    assert "kill -TERM -4242" in capsys.readouterr().out
    # Inside the detached run, --detach is ignored (it runs in the foreground).
    monkeypatch.setenv("CONSTELLATION_DETACHED", "1")
    launched.clear()
    assert main(argv) == 1  # the report itself runs, and refuses: no experiment
    assert launched == {}


def test_summarize_follows_the_latest_detached_run(tmp_path, jar, inputs, capsys) -> None:
    from constellation.cli.__main__ import main
    from constellation.core.detach import launch

    experiment = tmp_path / "experiment"
    search_acquisitions(_bundles(tmp_path, "run_a"), experiment_dir=experiment, **inputs)
    launch([sys.executable, "-c", "print('working'); raise SystemExit(0)"],
           run_dir=experiment / "runs", label="quantify")  # fmt: skip
    assert main(["massspec", "summarize", str(experiment), "--follow"]) == 0
    captured = capsys.readouterr()
    assert "working" in captured.err and "ended with exit status 0" in captured.err
    assert "detached runs" in captured.out and "quantify  finished" in captured.out
    assert main(["massspec", "summarize", str(tmp_path / "nowhere"), "--follow"]) == 2


def test_a_search_stopped_by_a_signal_in_a_worker_thread_is_recorded_as_cancelled(tmp_path, jar, inputs, monkeypatch) -> None:
    """In a worker thread the JVM's exit looks like any failure; the
    runner's cancellation flag is what tells them apart."""
    # Raised once the jar is running (earlier, the spectra file's writer
    # would stop first: the test below).
    monkeypatch.setattr(workflow, "cancellation_requested", lambda: 15 if jar.calls else None)
    jar.fail = JvmRunError(143, ["java"], "", tmp_path / "e.log")
    fan = search_acquisitions(_bundles(tmp_path, "run_a"), experiment_dir=tmp_path / "e", **inputs)
    ((plan, _),) = fan.failures
    record = read_manifest(tmp_path / "e" / "units" / plan.unit, expect=SearchManifest)
    assert record.status == "cancelled"
    # Why it ended is the signal, not the (here empty) end of the jar's stderr.
    assert record.error == "stopped by SIGTERM (the jar exited 143)"


def test_a_report_after_its_searches_were_run_again_is_made_again(tmp_path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    bundles = _bundles(tmp_path, "run_a", "run_b")
    search_acquisitions(bundles, experiment_dir=experiment, **inputs)
    report = report_searches(experiment, fasta=inputs["fasta"])
    time.sleep(0.01)
    search_acquisitions(bundles[:1], experiment_dir=experiment, **inputs)  # the same search, run again
    again = report_searches(experiment, fasta=inputs["fasta"], resume=True)
    assert not again.reused and again.key != report.key


def test_a_report_takes_each_acquisition_once(tmp_path, jar, inputs) -> None:
    """The same acquisition searched as a bundle and as a spectra file of
    another name (the P4 comparison) is two searches of one run."""
    import shutil

    experiment = tmp_path / "experiment"
    (bundle,) = _bundles(tmp_path, "run_a")
    search_acquisitions([bundle], experiment_dir=experiment, **inputs)
    given = tmp_path / "run_a_ref.dia"
    shutil.copy(experiment / "spectra" / "run_a" / "run_a.dia", given)
    (other,) = search_acquisitions([given], experiment_dir=experiment, **inputs).results
    assert other.acquisition_id == 0
    calls = len(jar.calls)
    with pytest.raises(WorkflowError, match="are both searches of acquisition 0"):
        report_searches(experiment, fasta=inputs["fasta"])
    assert len(jar.calls) == calls  # refused before the jar


def test_a_search_running_in_another_command_is_left_alone(tmp_path, jar, inputs) -> None:
    """A detached run still going, and the same command again with
    --resume: the second must not clear the first's directory."""
    from constellation.core.locks import file_lock

    experiment = tmp_path / "e"
    (bundle,) = _bundles(tmp_path, "run_a")
    (first,) = search_acquisitions([bundle], experiment_dir=experiment, **inputs).results
    (first.unit_dir / "_SUCCESS").unlink()  # as if it were still running
    calls = len(jar.calls)
    with file_lock(experiment / "units" / f".{first.unit}.lock"):  # held by the other command
        fan = search_acquisitions([bundle], experiment_dir=experiment, resume=True, **inputs)
    ((_, exc),) = fan.failures
    assert isinstance(exc, WorkflowError) and "is running in another command now" in str(exc)
    assert len(jar.calls) == calls
    assert (first.unit_dir / "jar" / "run_a.dia.elib").is_file()
    assert read_manifest(first.unit_dir, expect=SearchManifest).status == "finished"
    # Once it is free, resume runs it.
    assert search_acquisitions([bundle], experiment_dir=experiment, resume=True, **inputs).ok

    (first.unit_dir / "_SUCCESS").touch()
    report = report_searches(experiment, fasta=inputs["fasta"])
    (report.report_dir / "_SUCCESS").unlink()
    with file_lock(report.report_dir.parent / f".{report.key}.lock"):
        with pytest.raises(WorkflowError, match="is running in another command now"):
            report_searches(experiment, fasta=inputs["fasta"], resume=True)
    assert (report.report_dir / "report.elib").is_file()


def test_ctrl_c_stops_searches_running_in_worker_threads(tmp_path, jar, inputs, monkeypatch) -> None:
    """Ctrl-C reaches the main thread only; JVMs started by worker
    threads (each in a session of its own) must be told to stop."""
    import signal

    from constellation.thirdparty import jvm

    monkeypatch.setattr(jvm, "_cancelled", None)  # and put back afterwards
    both = threading.Barrier(2)

    def running(unit_dir: Path) -> None:
        if both.wait(timeout=10) == 0:
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        deadline = time.monotonic() + 10
        while jvm.cancellation_requested() is None:  # a JVM, until it is told
            assert time.monotonic() < deadline, "the stop never reached the worker"
            time.sleep(0.01)
        raise JvmRunError(143, ["java"], "", tmp_path / "e.log")

    jar.while_running = running
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        search_acquisitions(
            _bundles(tmp_path, "run_a", "run_b"), experiment_dir=tmp_path / "e", jobs=2, **inputs
        )
    assert time.monotonic() - started < 10
    units = [d for d in (tmp_path / "e" / "units").iterdir() if d.is_dir()]
    assert len(units) == 2
    for unit_dir in units:
        record = read_manifest(unit_dir, expect=SearchManifest)
        assert record.status == "cancelled" and record.error == "stopped by SIGINT (the jar exited 143)"


def test_a_spectra_file_being_written_when_the_run_is_stopped(tmp_path, jar, inputs, monkeypatch) -> None:
    """A worker thread writing a spectra file asks whether to stop, so a
    stop does not wait for the whole file; nothing ran, so nothing is
    recorded, and the next write clears what a killed one left."""
    experiment = tmp_path / "e"
    (bundle,) = _bundles(tmp_path, "run_a")
    monkeypatch.setattr(workflow, "cancellation_requested", lambda: 15)
    fan = search_acquisitions([bundle], experiment_dir=experiment, **inputs)
    ((plan, exc),) = fan.failures
    assert type(exc).__name__ == "SpectraWriteStopped"
    spectra = experiment / "spectra" / "run_a"
    assert not (spectra / "_SUCCESS").exists() and not (experiment / "units" / plan.unit).exists()
    assert jar.calls == []

    monkeypatch.setattr(workflow, "cancellation_requested", lambda: None)
    stale = spectra / ".run_a.dia.tmp-999999"  # what a SIGKILLed writer leaves
    stale.write_bytes(b"half a file")
    assert search_acquisitions([bundle], experiment_dir=experiment, **inputs).ok
    assert not stale.exists() and (spectra / "run_a.dia").is_file()


def test_a_jar_failure_with_nothing_on_stderr_says_so(tmp_path, jar, inputs) -> None:
    jar.fail = JvmRunError(1, ["java"], "  \n", tmp_path / "e.log")
    fan = search_acquisitions(_bundles(tmp_path, "run_a"), experiment_dir=tmp_path / "e", **inputs)
    ((plan, _),) = fan.failures
    record = read_manifest(tmp_path / "e" / "units" / plan.unit, expect=SearchManifest)
    assert record.status == "failed"
    assert record.error == "the jar exited 1 and printed nothing to stderr"


def test_a_stopped_command_says_how_to_finish(tmp_path, jar, inputs, capsys) -> None:
    from constellation.cli.__main__ import main

    jar.fail = SystemExit(143)  # what the JVM runner's cancellation scope raises on SIGTERM
    (bundle,) = _bundles(tmp_path, "run_a")
    with pytest.raises(SystemExit):
        main(["massspec", "search", "--input", str(bundle), "--output-dir", str(tmp_path / "e"),
              "--library", str(inputs["library"]), "--fasta", str(inputs["fasta"])])  # fmt: skip
    err = capsys.readouterr().err
    assert "stopped: every search that had not finished is recorded as cancelled" in err
    assert "--resume runs only what did not finish" in err


def test_the_commands(tmp_path, jar, inputs, capsys) -> None:
    from constellation.cli.__main__ import main

    experiment = tmp_path / "experiment"
    fractions = [str(p) for p in _bundles(tmp_path, "gpf-1_x", "gpf-2_x")]
    runs = [str(p) for p in _bundles(tmp_path, "run_a", "run_b")]
    common = ["--fasta", str(inputs["fasta"]), "--output-dir", str(experiment), "--no-progress"]
    assert main(["massspec", "quantify", "--runs", *runs, "--fractions", *fractions,
                 "--library", str(inputs["library"]), "--jobs", "2", *common]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "chromatogram library:" in out and "report across 2 searches (2 acquisitions)" in out
    # The steps on their own find what the chain made.
    assert main(["massspec", "search", "--input", *fractions, "--merge", "--library",
                 str(inputs["library"]), "--resume", *common]) == 0  # fmt: skip
    assert "already searched" in capsys.readouterr().out
    assert main(["massspec", "report", "--resume", "--fasta", str(inputs["fasta"]),
                 "--output-dir", str(experiment)]) == 0  # fmt: skip
    assert "already made" in capsys.readouterr().out
    # Several inputs, one failing: exit 1, the others done.
    jar.fail_for = {"run_c": RuntimeError("no")}
    more = [str(p) for p in _bundles(tmp_path, "run_c", "run_d")]
    assert main(["massspec", "search", "--input", *more, "--library", str(inputs["library"]), *common]) == 1
    err = capsys.readouterr().err
    assert "1 of 2 searches failed (1 done)" in err and "--resume runs only what did not finish" in err
    assert os.path.isdir(experiment / "reports")
    # The chain's merged search failing: the jar's own status, and which step.
    jar.fail_for = {}
    jar.fail = JvmRunError(3, ["java"], "Exception: no", tmp_path / "e.log")
    other = [str(p) for p in _bundles(tmp_path, "gpf-3_y", "gpf-4_y")]
    assert main(["massspec", "quantify", "--runs", runs[0], "--fractions", *other,
                 "--library", str(inputs["library"]), *common]) == 3  # fmt: skip
    assert "gpf-merged_y" in capsys.readouterr().err



def test_a_searchs_sidecars_and_quantity_arrays_become_experiment_tables(tmp_path, jar, inputs) -> None:
    """Phase 7: Percolator's scores make the precursor scores complete
    (every precursor scored, content-keyed); the calibration fits and the
    features are tables of their own."""
    from constellation.massspec.library.ids import precursor_id

    experiment = tmp_path / "experiment"
    (result,) = search_acquisitions(_bundles(tmp_path, "run_a"), experiment_dir=experiment, **inputs).results
    scores = read_experiment_table(experiment, "precursor_score", unit=result.unit)
    beyond = scores.filter(pc.equal(scores.column("modified_sequence"), "ELVISK")).to_pylist()
    assert len(beyond) == 1 and beyond[0]["qvalue"] == 0.4
    assert beyond[0]["precursor_id"] == precursor_id("ELVISK", 2)  # in no table of the library
    precursors = set(read_experiment_table(experiment, "precursor", unit=result.unit).column("precursor_id").to_pylist())
    within = scores.filter(pc.less_equal(scores.column("qvalue"), 0.01))
    targets = within.filter(pc.invert(within.column("is_decoy")))
    assert set(targets.column("precursor_id").to_pylist()) == precursors
    assert scores.filter(scores.column("is_decoy")).num_rows == 1
    alignment = read_experiment_table(experiment, "rt_alignment", unit=result.unit)
    assert alignment.num_rows == len(_PEPTIDES) and set(alignment.column("pass").to_pylist()) == {"final"}
    assert set(alignment.column("acquisition_id").to_pylist()) == {0}
    features = read_experiment_table(experiment, "percolator_feature", unit=result.unit)
    assert features.num_rows == len(_PEPTIDES) + 1 and "primary" in features.column_names
    assert result.manifest.runtime["table_rows"]["rt_alignment"] == len(_PEPTIDES)
