# ruff: noqa: F811  (the stand-in jar's fixtures are imported, then named as arguments)
"""``massspec summarize``: what an experiment directory, or one search in it, holds.

Searches are named ``<run>-<digest>``, which keeps them apart and says
nothing else; the summary is what says the rest. The experiments here
are written by the real workflow around the stand-in jar of
``test_massspec_search_workflow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from constellation.core.manifests import read_manifest, write_manifest
from constellation.massspec.experiment import PASSED, RECORDED, format_summary, summarize
from constellation.massspec.manifest import SearchManifest
from constellation.massspec.search.encyclopedia.params import SearchParameters, Tolerance
from constellation.massspec.workflows.encyclopedia import search_acquisition
from constellation.thirdparty.jvm import JvmRunError

from test_massspec_search_workflow import _PEPTIDES, _bundle, inputs, jar  # noqa: F401


def _flat(text: str) -> str:
    """The text with its line wrapping undone."""
    return " ".join(text.split())


def _by_unit(summary: dict) -> dict[str, dict]:
    return {s["unit"]: s for s in summary["searches"]}


def test_a_search_says_what_it_is_while_it_runs_and_after(tmp_path: Path, jar, inputs) -> None:
    seen = {}

    def look(unit_dir: Path) -> None:
        seen["record"] = read_manifest(unit_dir, expect=SearchManifest)
        seen["done"] = (unit_dir / "_SUCCESS").exists()
        seen["summary"] = summarize(unit_dir)

    jar.while_running = look
    params = SearchParameters(precursor_tolerance=Tolerance(0.35, "Da"), extra={"-fixed": ""})
    result = search_acquisition(
        _bundle(tmp_path, "run_01"), experiment_dir=tmp_path / "experiment", parameters=params, **inputs
    )

    running = seen["record"]
    assert running.status == "running" and not seen["done"]
    assert running.engine_flags["-ptol"] == "0.35" and running.engine_flags["-ptolunits"] == "AMU"
    assert running.engine_flags["-fixed"] == "" and running.engine_settings == {}
    assert running.inputs["fasta"]["sha256"] and running.argv[:2] == ["-i", running.engine_flags["-i"]]
    # Mid-search, the summary already says what it is.
    (now,) = seen["summary"]["searches"]
    assert now["status"] == "not finished" and now["settings_source"] == PASSED
    assert ["precursor tolerance", "0.35 Da"] in now["settings"]
    # The jar's documented default stands in for what was not given.
    assert ["fragment tolerance", "10 ppm"] in now["settings"]

    done = result.manifest
    assert done.status == "finished" and done.error == ""
    assert done.created_at == running.created_at and done.runtime["finished_at"]
    # What the jar recorded: defaults included, not only what was passed.
    assert done.engine_settings["-ptol"] == "0.35"
    assert done.engine_settings["-minNumOfQuantitativePeaks"] == "3"
    assert done.engine_flags == running.engine_flags
    assert done.argv[0] == "java"


def test_failed_and_cancelled_searches_say_what_stopped_them(tmp_path: Path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    bundle = _bundle(tmp_path, "run_01")
    jar.fail = JvmRunError(1, ["java", "-jar", "x.jar"], "a line\nException: out of memory", tmp_path / "e.log")
    with pytest.raises(JvmRunError):
        search_acquisition(bundle, experiment_dir=experiment, **inputs)
    jar.fail = SystemExit(143)  # what jvm.cancellation_scope raises on SIGTERM
    with pytest.raises(SystemExit):
        search_acquisition(
            bundle, experiment_dir=experiment, parameters=SearchParameters(threads=2,
            precursor_tolerance=Tolerance(5.0)), **inputs,
        )  # fmt: skip

    searches = summarize(experiment)["searches"]
    failed, cancelled = sorted(searches, key=lambda s: s["status"], reverse=True)
    assert failed["status"] == "failed"
    assert failed["error"].startswith("the jar exited 1; the end of its stderr:")
    assert "Exception: out of memory" in failed["error"]
    assert cancelled["status"] == "cancelled"
    assert cancelled["error"] == "stopped by a signal (exit status 143)"
    for s in searches:
        assert s["finished_at"] and s["found"] is None and s["settings_source"] == PASSED
        assert not (experiment / s["directory"] / "_SUCCESS").exists()
    # Neither got into the experiment's manifest, and the summary finds them anyway.
    assert {s["in_experiment_manifest"] for s in searches} == {False}
    text = _flat(format_summary(summarize(experiment)))
    assert "failed" in text and "the jar exited 1" in text and "cancelled" in text
    assert f"logs {failed['logs']}" in text


def test_an_experiment_summary_tells_its_searches_apart(tmp_path: Path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    run_01 = _bundle(tmp_path, "run_01")
    wide = search_acquisition(
        run_01, experiment_dir=experiment,
        parameters=SearchParameters(precursor_tolerance=Tolerance(0.35, "Da")), **inputs,
    )  # fmt: skip
    jar.peptides = _PEPTIDES[:2]
    narrow = search_acquisition(
        run_01, experiment_dir=experiment,
        parameters=SearchParameters(precursor_tolerance=Tolerance(10.0)), **inputs,
    )  # fmt: skip
    (inputs["library"].parent / "other.dlib").write_bytes(b"another library")
    other_library = search_acquisition(
        run_01, experiment_dir=experiment, fasta=inputs["fasta"],
        library=inputs["library"].parent / "other.dlib",
        parameters=SearchParameters(precursor_tolerance=Tolerance(0.35, "Da")),
    )  # fmt: skip
    alone = search_acquisition(_bundle(tmp_path, "run_02"), experiment_dir=experiment, **inputs)
    (experiment / "units" / "stray").mkdir()  # a directory with nothing to say

    summary = summarize(experiment)
    assert summary["focus"] is None and summary["name"] == "experiment"
    assert [(a["acquisition_id"], a["name"]) for a in summary["acquisitions"]] == [
        (0, "run_01.raw"), (1, "run_02.raw"),
    ]  # fmt: skip
    searches = _by_unit(summary)
    assert set(searches) == {wide.unit, narrow.unit, other_library.unit, alone.unit, "stray"}
    assert searches["stray"]["status"] == "no record"

    w, n, o = searches[wide.unit], searches[narrow.unit], searches[other_library.unit]
    for s in (w, n, o):
        assert s["status"] == "finished" and s["settings_source"] == RECORDED
        assert s["acquisition_ids"] == [0] and s["in_experiment_manifest"]
    assert w["found"] == {"precursors": 3, "peptides": 3, "proteins": 1, "fdr": "1%"}
    assert n["found"]["precursors"] == 2
    assert sorted(w["same_acquisitions_as"]) == sorted([narrow.unit, other_library.unit])
    # What sets each apart, in Constellation's spelling, and only that.
    assert w["differs_by"] == [["library", "lib.dlib"], ["precursor tolerance", "0.35 Da"]]
    # The jar's metadata table gives "10.0" back as "10" (numeric affinity);
    # the summary shows what the result says.
    assert n["differs_by"] == [["library", "lib.dlib"], ["precursor tolerance", "10 ppm"]]
    assert o["differs_by"] == [["library", "other.dlib"], ["precursor tolerance", "0.35 Da"]]
    a = searches[alone.unit]
    assert a["same_acquisitions_as"] == [] and a["differs_by"] == []

    # Plain data all the way down.
    assert json.loads(json.dumps(summary)) == summary

    text = format_summary(summary)
    assert "acquisitions (2)" in text and "searches (5: 4 finished, 1 no record)" in text
    assert "found 3 precursors, 3 peptides, 1 proteins at 1% FDR" in _flat(text)
    assert "differs library other.dlib; precursor tolerance 0.35 Da" in _flat(text)
    assert "precursor tolerance 10 ppm" in _flat(text)
    for unit in (wide.unit, narrow.unit, other_library.unit, alone.unit):
        assert f"\n  {unit}\n" in text


def test_one_search_in_full(tmp_path: Path, jar, inputs) -> None:
    experiment = tmp_path / "experiment"
    run_01 = _bundle(tmp_path, "run_01")
    first = search_acquisition(
        run_01, experiment_dir=experiment,
        parameters=SearchParameters(precursor_tolerance=Tolerance(0.35, "Da"), extra={"-fixed": ""}),
        **inputs,
    )  # fmt: skip
    second = search_acquisition(run_01, experiment_dir=experiment, **inputs)

    summary = summarize(first.unit_dir)
    assert summary["focus"] == first.unit
    text = format_summary(summary)
    assert f"search {first.unit}" in text and second.unit not in text.split("other searches")[0]
    assert "settings (as the engine recorded them in its result)" in text
    # Every recorded setting is listed, the headline ones in Constellation's spelling.
    for flag in first.manifest.engine_settings:
        if flag not in ("-ptol", "-ptolunits", "-ftol", "-ftolunits", "-lftol", "-lftolunits",
                        "-enzyme", "-frag", "-acquisition", "-fixed", "-percolatorVersion",
                        "-percolatorThreshold"):  # fmt: skip
            assert f"      {flag} " in text, flag
    assert "precursor tolerance" in text and "0.35 Da" in text and "none" in text
    assert "given     " in text and "-ptol 0.35" in text and "-fixed ''" in text
    assert "command   java -jar" in text
    assert "spectra   spectra/run_01/run_01.dia  (written by Constellation" in text
    assert f"result    units/{first.unit}/jar/run_01.dia.elib" in text
    assert "other searches of the same acquisition in this experiment" in text
    assert second.unit in text and "this one  precursor tolerance 0.35 Da" in text


def test_a_search_stopped_without_a_record(tmp_path: Path, jar, inputs) -> None:
    """A search killed outright (SIGKILL, a power cut) cannot say so: its
    manifest still reads ``running`` and it has no marker."""
    result = search_acquisition(_bundle(tmp_path, "run_01"), experiment_dir=tmp_path / "e", **inputs)
    (result.unit_dir / "_SUCCESS").unlink()
    record = read_manifest(result.unit_dir, expect=SearchManifest)
    write_manifest(result.unit_dir, type(record)(**{**record.__dict__, "status": "running"}))
    (s,) = summarize(tmp_path / "e")["searches"]
    assert s["status"] == "not finished" and s["last_activity"] and s["found"] is None
    assert "still running, or stopped without a chance to say so" in _flat(
        format_summary(summarize(tmp_path / "e"))
    )


def test_the_command(tmp_path: Path, jar, inputs, capsys) -> None:
    from constellation.cli.__main__ import main

    result = search_acquisition(_bundle(tmp_path, "run_01"), experiment_dir=tmp_path / "e", **inputs)
    assert main(["massspec", "summarize", str(tmp_path / "e")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("experiment e  (") and result.unit in out
    assert main(["massspec", "summarize", str(result.unit_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["focus"] == result.unit
    (tmp_path / "plain").mkdir()
    assert main(["massspec", "summarize", str(tmp_path / "plain")]) == 2
    assert "is not an experiment directory" in capsys.readouterr().err
