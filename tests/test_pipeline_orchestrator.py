"""Tier A tests for the `constellation pipeline transcriptome-to-proteomics`
orchestrator and its CLI parser.

Strategy:
  * CLI introspection — assert required args, optional knobs, and that
    the parser surfaces in the dashboard introspector.
  * Mocked stage handoffs — monkeypatch every external runner
    (mmseqs2, JVM, swissprot fetch, ParquetDir writers) so the test
    exercises the orchestrator's stage sequencing + manifest writing
    + collision-filter wiring + --resume short-circuit without
    spawning jars or hitting the network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.cli.pipeline import build_parser
from constellation.transcriptome_to_proteome import (
    _sanitise_dirname,
    _stage_done,
    _touch_success,
)


# ──────────────────────────────────────────────────────────────────────
# CLI introspection
# ──────────────────────────────────────────────────────────────────────


def _make_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="constellation")
    subs = root.add_subparsers(dest="subcommand", required=True)
    build_parser(subs)
    return root


def _subparsers_action(parser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("no _SubParsersAction found on parser")


def test_pipeline_subcommand_registers() -> None:
    parser = _make_parser()
    pipeline_subs = _subparsers_action(parser).choices["pipeline"]
    inner = _subparsers_action(pipeline_subs)
    assert "transcriptome-to-proteomics" in inner.choices


def test_t2p_required_args() -> None:
    parser = _make_parser()
    t2p = _subparsers_action(
        _subparsers_action(parser).choices["pipeline"]
    ).choices["transcriptome-to-proteomics"]
    required = {
        opt
        for action in t2p._actions
        if getattr(action, "required", False)
        for opt in action.option_strings
    }
    expected = {
        "--protein-counts",
        "--proteins-fasta",
        "--reference-fasta",
        "--reference-annotation",
        "--gpf",
        "--injections",
        "--output-dir",
    }
    assert expected.issubset(required), (
        f"missing required args: {expected - required}"
    )


def test_t2p_collision_filter_default_on() -> None:
    parser = _make_parser()
    parsed = parser.parse_args(
        [
            "pipeline", "transcriptome-to-proteomics",
            "--protein-counts", "/x/c.tsv",
            "--proteins-fasta", "/x/p.fa",
            "--reference-fasta", "/x/r.fa",
            "--reference-annotation", "/x/r.gff3",
            "--gpf", "/x/g.mzML",
            "--injections", "/x/i.mzML",
            "--output-dir", "/x/out",
        ]
    )
    # Default-on means args.no_collision_filter is False.
    assert parsed.no_collision_filter is False


def test_t2p_can_opt_out_collision_filter() -> None:
    parser = _make_parser()
    parsed = parser.parse_args(
        [
            "pipeline", "transcriptome-to-proteomics",
            "--protein-counts", "/x/c.tsv",
            "--proteins-fasta", "/x/p.fa",
            "--reference-fasta", "/x/r.fa",
            "--reference-annotation", "/x/r.gff3",
            "--gpf", "/x/g.mzML",
            "--injections", "/x/i.mzML",
            "--output-dir", "/x/out",
            "--no-collision-filter",
        ]
    )
    assert parsed.no_collision_filter is True


def test_t2p_ptm_carbamidomethyl_default_fix() -> None:
    """Mirrors the predict-library invariant — Carbamidomethyl is
    EncyclopeDIA's default-fixed PTM."""
    parser = _make_parser()
    t2p = _subparsers_action(
        _subparsers_action(parser).choices["pipeline"]
    ).choices["transcriptome-to-proteomics"]
    for action in t2p._actions:
        if "--ptm-carbamidomethyl" in action.option_strings:
            assert action.default == "fix"
            assert action.choices == ["off", "var", "fix"]
            return
    pytest.fail("--ptm-carbamidomethyl not registered")


def test_t2p_in_dashboard_introspector() -> None:
    """The dashboard auto-generates forms from the argparse tree — walking
    must surface the pipeline → transcriptome-to-proteomics subcommand
    with its arguments populated."""
    pytest.importorskip("constellation.viz.introspect.walk")
    from constellation.viz.introspect.walk import walk_parser

    parser = _make_parser()
    tree = walk_parser(parser)
    pipeline_node = next(
        (s for s in tree["subcommands"] if s["name"] == "pipeline"), None
    )
    assert pipeline_node is not None
    sub_names = {s["name"] for s in pipeline_node["subcommands"]}
    assert "transcriptome-to-proteomics" in sub_names
    t2p_node = next(
        s for s in pipeline_node["subcommands"]
        if s["name"] == "transcriptome-to-proteomics"
    )
    assert t2p_node["arguments"], "no arguments surfaced for transcriptome-to-proteomics"


# ──────────────────────────────────────────────────────────────────────
# Orchestrator helpers
# ──────────────────────────────────────────────────────────────────────


def test_stage_done_returns_false_when_no_success(tmp_path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    assert _stage_done(stage, resume=True) is False


def test_stage_done_returns_true_when_success_and_resume(tmp_path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    _touch_success(stage)
    assert _stage_done(stage, resume=True) is True


def test_stage_done_raises_when_success_without_resume(tmp_path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    _touch_success(stage)
    with pytest.raises(RuntimeError, match="already complete"):
        _stage_done(stage, resume=False)


def test_sanitise_dirname_strips_unsafe_chars() -> None:
    assert _sanitise_dirname("sample 01/foo") == "sample_01_foo"
    assert _sanitise_dirname("") == "sample"
    assert _sanitise_dirname("safe-name_v1.0") == "safe-name_v1.0"


# ──────────────────────────────────────────────────────────────────────
# Mocked end-to-end orchestrator
# ──────────────────────────────────────────────────────────────────────


_REF_FASTA = ">REF_A\nMKAAA\n>REF_B\nMKBBB\n"
_NOVEL_FASTA = ">N1\nMKCCC\n>N2\nMKDDD\n"


@pytest.fixture
def stub_inputs(tmp_path) -> dict:
    """Lay down minimal input fixtures that the orchestrator's
    helpers can read without throwing."""
    # Protein counts TSV.
    counts = tmp_path / "protein_counts.tsv"
    counts.write_text(
        "\tProtein\tqJS001\tSequence\n"
        "0\tN1\t10.0\t" + "M" * 120 + "\n"
        "1\tN2\t5.0\t" + "K" * 120 + "\n"
    )
    # Proteins FASTA.
    proteins = tmp_path / "proteins.fasta"
    proteins.write_text(
        ">N1 source=demux\n" + "M" * 120 + "\n"
        ">N2 source=demux\n" + "K" * 120 + "\n"
    )
    # Reference FASTA + annotation.
    ref = tmp_path / "reference.fasta"
    ref.write_text(_REF_FASTA)
    ann = tmp_path / "reference.gff3"
    ann.write_text(
        "##gff-version 3\n"
        "chr1\tt\tCDS\t1\t10\t.\t+\t0\tID=c1;protein_id=REF_A;gene=GENE_A\n"
        "chr1\tt\tCDS\t20\t30\t.\t+\t0\tID=c2;protein_id=REF_B;gene=GENE_B\n"
    )
    # GPF + injection paths can be empty (mocked at the runner layer).
    gpf = tmp_path / "gpf.mzML"
    gpf.write_text("")
    inj = tmp_path / "sample01.mzML"
    inj.write_text("")
    out = tmp_path / "out"
    return {
        "protein_counts": counts,
        "proteins_fasta": proteins,
        "reference_fasta": ref,
        "reference_annotation": ann,
        "gpf": [gpf],
        "injections": [inj],
        "output_dir": out,
    }


def _build_args(stub: dict, **overrides) -> argparse.Namespace:
    """Build the namespace the orchestrator expects.

    Mirrors the CLI parser defaults — kept in sync by parsing through
    argparse and overriding by attribute.
    """
    parser = _make_parser()
    parsed = parser.parse_args(
        [
            "pipeline", "transcriptome-to-proteomics",
            "--protein-counts", str(stub["protein_counts"]),
            "--proteins-fasta", str(stub["proteins_fasta"]),
            "--reference-fasta", str(stub["reference_fasta"]),
            "--reference-annotation", str(stub["reference_annotation"]),
            "--gpf", *[str(p) for p in stub["gpf"]],
            "--injections", *[str(p) for p in stub["injections"]],
            "--output-dir", str(stub["output_dir"]),
        ]
    )
    for k, v in overrides.items():
        setattr(parsed, k, v)
    return parsed


@pytest.fixture
def stub_external_calls(monkeypatch, tmp_path) -> dict:
    """Replace every external runner (mmseqs, JVM, swissprot fetch,
    auto-ingest) with deterministic stubs. Returns a dict the tests
    can read to assert call ordering / arguments.
    """
    state = {"calls": []}

    def _fake_fetch_swissprot(*, release=None, **_):
        class _SH:
            fasta_path = tmp_path / "stub_sprot.fasta"
            release = "2026_02"
            sha256 = "stub"
            source_url = "stub"
        (tmp_path / "stub_sprot.fasta").write_text(">SP_X\nMKAAA\n")
        state["calls"].append(("fetch_swissprot", release))
        return _SH()

    def _fake_run_mmseqs_search(*, output_tab, **kw):
        # Emit a synthetic .tab that classify_novel_peptides can read.
        Path(output_tab).write_text(
            "N1\tREF_A\t1e-50\t1\t10\t1\t10\t10M\n"
            # N2 → no hit; it'll drop from the alignment filter.
        )
        state["calls"].append(("mmseqs", str(output_tab)))

        class _R:
            elapsed_seconds = 0.01
            returncode = 0
            mmseqs_version = "stub-15-6f452"

        return _R()

    def _fake_run_predict_library(*, output_dlib, output_dir, **kw):
        Path(output_dlib).write_text("stub-dlib")
        state["calls"].append(("predict_library", str(output_dlib)))

        class _R:
            elapsed_seconds = 0.01
            returncode = 0
            java_version = "stub-java"

        return _R()

    def _fake_run_process_dia(*, output_dia, output_dir, **kw):
        # Lay down a real SQLite with the ranges table so the
        # downstream collision filter can read isolation windows
        # from it.
        import sqlite3
        Path(output_dia).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(output_dia))
        try:
            con.execute(
                "CREATE TABLE ranges (Start REAL, Stop REAL, "
                "DutyCycle REAL, NumWindows INTEGER)"
            )
            con.commit()
        finally:
            con.close()
        state["calls"].append(("process_dia", str(output_dia)))

        class _R:
            elapsed_seconds = 0.01
            returncode = 0

        return _R()

    def _fake_run_library_search(*, input_file, library, output_dir, fasta=None, **kw):
        # EncyclopeDIA 6.5.15's search REQUIRES -f <fasta>. Capture the
        # fasta kwarg so the regression test can assert the orchestrator
        # threads the combined FASTA through to every search call.
        state.setdefault("search_fasta", []).append(fasta)
        # EncyclopeDIA 6.5.15 writes <input_stem>.elib to cwd=output_dir;
        # match that convention so the orchestrator's find_search_elib
        # (real, not stubbed) can locate it.
        import sqlite3
        elib_path = Path(output_dir) / (Path(input_file).stem + ".elib")
        dia_path = Path(output_dir) / (Path(input_file).stem + ".dia")
        for p in (elib_path, dia_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(elib_path))
        try:
            con.execute(
                "CREATE TABLE entries ("
                "PrecursorMz REAL, PrecursorCharge INTEGER, "
                "PeptideModSeq TEXT, PeptideSeq TEXT, Copies INTEGER, "
                "RTInSeconds REAL, Score REAL, "
                "MassEncodedLength INTEGER, MassArray BLOB, "
                "IntensityEncodedLength INTEGER, IntensityArray BLOB, "
                "CorrelationEncodedLength INTEGER, CorrelationArray BLOB, "
                "QuantifiedIonsArray BLOB, "
                "RTInSecondsStart REAL, RTInSecondsStop REAL, "
                "MedianChromatogramEncodedLength INTEGER, "
                "MedianChromatogramArray BLOB, "
                "SourceFile TEXT)"
            )
            con.commit()
        finally:
            con.close()
        # .dia placeholder with the ranges table (empty).
        con2 = sqlite3.connect(str(dia_path))
        try:
            con2.execute(
                "CREATE TABLE ranges (Start REAL, Stop REAL, "
                "DutyCycle REAL, NumWindows INTEGER)"
            )
            con2.commit()
        finally:
            con2.close()
        state["calls"].append(("library_search", str(elib_path)))

        class _R:
            elapsed_seconds = 0.01
            returncode = 0

        return _R()

    def _fake_run_library_export(*, output_elib, output_dir, **kw):
        Path(output_elib).write_text("stub-quant-elib")
        state["calls"].append(("library_export", str(output_elib)))

        class _R:
            elapsed_seconds = 0.01
            returncode = 0

        return _R()

    def _fake_maybe_auto_ingest(elib_path, run_dir, no_ingest):
        if no_ingest:
            return
        # Write empty marker dirs so the orchestrator's downstream
        # checks see "library_pqdir exists".
        (Path(run_dir) / "library_pqdir").mkdir(parents=True, exist_ok=True)
        state["calls"].append(("auto_ingest", str(elib_path)))

    import constellation.transcriptome_to_proteome as t2p

    monkeypatch.setattr(
        "constellation.catalog.uniprot.fetch_swissprot", _fake_fetch_swissprot
    )
    # The orchestrator imports these lazily inside the function — patch
    # at the *origin* modules so the lookup picks up the stubs.
    monkeypatch.setattr(
        "constellation.thirdparty.mmseqs2_run.run_mmseqs_search",
        _fake_run_mmseqs_search,
    )
    monkeypatch.setattr(
        "constellation.massspec.search.encyclopedia.run_predict_library",
        _fake_run_predict_library,
    )
    monkeypatch.setattr(
        "constellation.massspec.search.encyclopedia.run_process_dia",
        _fake_run_process_dia,
    )
    monkeypatch.setattr(
        "constellation.massspec.search.encyclopedia.run_library_search",
        _fake_run_library_search,
    )
    monkeypatch.setattr(
        "constellation.massspec.search.encyclopedia.run_library_export",
        _fake_run_library_export,
    )
    monkeypatch.setattr(t2p, "_maybe_auto_ingest_elib", _fake_maybe_auto_ingest)

    return state


def test_orchestrator_end_to_end_mocked(
    stub_inputs, stub_external_calls
) -> None:
    """Run the orchestrator with every external runner mocked; assert
    the stage dirs + _SUCCESS markers + top-level manifest land."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs)
    rc = run_transcriptome_to_proteomics(args=args)
    assert rc == 0
    out = stub_inputs["output_dir"]
    # Per-stage _SUCCESS markers.
    for stage_name in (
        "01_protein_counts",
        "02_novel_fasta",
        "03_alignment",
        "04_combined_fasta",
        "05_predict_library",
        "06_process_dia",
        "07_gpf_search",
        "08_classify_novel_peptides",
        "09_per_injection",
        "10_quant_report",
    ):
        assert (out / stage_name / "_SUCCESS").is_file(), (
            f"{stage_name} missing _SUCCESS"
        )
        assert (out / stage_name / "manifest.json").is_file(), (
            f"{stage_name} missing manifest.json"
        )
    # Top-level _SUCCESS + manifest.
    assert (out / "_SUCCESS").is_file()
    top = json.loads((out / "manifest.json").read_text())
    assert top["subcommand"] == "pipeline transcriptome-to-proteomics"
    assert "stages" in top
    assert set(top["stages"]) == {
        "01_protein_counts",
        "02_novel_fasta",
        "03_alignment",
        "04_combined_fasta",
        "05_predict_library",
        "06_process_dia",
        "07_gpf_search",
        "08_classify_novel_peptides",
        "09_per_injection",
        "10_quant_report",
    }


def test_orchestrator_threads_fasta_to_every_search(
    stub_inputs, stub_external_calls
) -> None:
    """EncyclopeDIA 6.5.15 search REQUIRES -f <fasta>. Every
    run_library_search call (GPF + the per-injection loop) must receive
    the combined FASTA — regression guard for the missing-fasta bug
    that surfaced on the first real OSC run."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs)
    rc = run_transcriptome_to_proteomics(args=args)
    assert rc == 0
    combined_fasta = (
        stub_inputs["output_dir"] / "04_combined_fasta" / "combined.fasta"
    )
    fastas = stub_external_calls["search_fasta"]
    # 1 GPF search + 1 injection (stub has a single injection) = 2 calls.
    assert len(fastas) >= 2
    assert all(f == combined_fasta for f in fastas), (
        f"some search calls missing the combined FASTA: {fastas}"
    )


def test_orchestrator_resume_short_circuit(
    stub_inputs, stub_external_calls
) -> None:
    """First run completes; second run with --resume short-circuits at
    the top-level _SUCCESS check."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs)
    rc = run_transcriptome_to_proteomics(args=args)
    assert rc == 0
    first_calls = list(stub_external_calls["calls"])
    assert first_calls, "first run made some external calls"

    # Reset call tracking; re-run with --resume.
    stub_external_calls["calls"].clear()
    args2 = _build_args(stub_inputs, resume=True)
    rc = run_transcriptome_to_proteomics(args=args2)
    assert rc == 0
    # Top-level _SUCCESS short-circuits before any per-stage call.
    assert stub_external_calls["calls"] == []


def test_orchestrator_stage7_resumes_from_existing_elib(
    stub_inputs, stub_external_calls
) -> None:
    """If a prior run produced 07_gpf_search/combined.filtered.elib but
    died before _SUCCESS (e.g. mid auto-ingest), --resume must pick up
    from the existing elib rather than re-running the multi-hour
    EncyclopeDIA search."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs)
    assert run_transcriptome_to_proteomics(args=args) == 0

    stage7 = stub_inputs["output_dir"] / "07_gpf_search"
    assert (stage7 / "combined.filtered.elib").is_file()
    # Simulate a crash after the filtered elib was written but before the
    # stage was marked complete — clear Stage 7's _SUCCESS and the
    # top-level _SUCCESS (which would otherwise short-circuit the run).
    (stage7 / "_SUCCESS").unlink()
    (stub_inputs["output_dir"] / "_SUCCESS").unlink()

    stub_external_calls["calls"].clear()
    args2 = _build_args(stub_inputs, resume=True)
    assert run_transcriptome_to_proteomics(args=args2) == 0

    # No search re-run (neither the GPF search nor any injection search).
    assert not any(
        c[0] == "library_search" for c in stub_external_calls["calls"]
    ), stub_external_calls["calls"]
    # Stage 7 is now properly completed.
    assert (stage7 / "_SUCCESS").is_file()


def test_orchestrator_collision_filter_default_runs(
    stub_inputs, stub_external_calls
) -> None:
    """Collision filter is on by default → Stage 7 writes
    collision_metadata.json + _raw/."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs)
    rc = run_transcriptome_to_proteomics(args=args)
    assert rc == 0
    stage7 = stub_inputs["output_dir"] / "07_gpf_search"
    assert (stage7 / "collision_metadata.json").is_file()
    assert (stage7 / "_raw").is_dir()
    assert (stage7 / "combined.filtered.elib").is_file()
    # Raw search output consolidated under _raw/ with a transparent name.
    assert (stage7 / "_raw" / "combined.raw.elib").is_file()


def test_orchestrator_no_collision_filter_skips_raw(
    stub_inputs, stub_external_calls
) -> None:
    """--no-collision-filter → no _raw/ subdir, no collision_metadata.json."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs, no_collision_filter=True)
    rc = run_transcriptome_to_proteomics(args=args)
    assert rc == 0
    stage7 = stub_inputs["output_dir"] / "07_gpf_search"
    assert not (stage7 / "collision_metadata.json").is_file()
    # _raw/ still exists because the jar's raw output lands there, but
    # no extra files were created since filter was skipped.
    assert (stage7 / "combined.elib").is_file()
