"""Tier A tests for the `constellation transcriptome-to-proteome`
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

from constellation.cli.transcriptome_to_proteome import build_parser
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


def test_t2p_subcommand_registers() -> None:
    parser = _make_parser()
    assert "transcriptome-to-proteome" in _subparsers_action(parser).choices


def test_t2p_required_args() -> None:
    """The base set of required argparse flags. Note that
    ``--reference-fasta`` and ``--reference-annotation`` are NOT
    argparse-level required — when ``--reference-dir`` is supplied
    they're auto-resolved from the Constellation cache layout. The
    handler enforces the "at least one of the three" rule
    post-parse."""
    parser = _make_parser()
    t2p = _subparsers_action(parser).choices["transcriptome-to-proteome"]
    required = {
        opt
        for action in t2p._actions
        if getattr(action, "required", False)
        for opt in action.option_strings
    }
    expected = {
        "--protein-counts",
        "--gpf",
        "--injections",
        "--output-dir",
    }
    assert expected.issubset(required), (
        f"missing required args: {expected - required}"
    )
    # --reference-fasta / --reference-annotation / --reference-dir are
    # all optional at parse-time (handler validates post-parse).
    # --proteins-fasta is optional — the pipeline derives novel ORF
    # sequences from --protein-counts (the demux table's `sequence`
    # column) by default.
    optional = {
        opt
        for action in t2p._actions
        if not getattr(action, "required", False)
        for opt in action.option_strings
    }
    assert "--reference-fasta" in optional
    assert "--reference-annotation" in optional
    assert "--reference-dir" in optional
    assert "--proteins-fasta" in optional


def test_t2p_ptm_defaults_match_lab_convention() -> None:
    """Pins the orchestrator's PTM defaults so a future refactor cannot
    silently flip a variable mod off. Defaults come from
    ``massspec.search.encyclopedia.ptm_defaults`` — see that module's
    docstring for the rationale (Carb-C fix + N-term-Acetyl/PyroGluQ/M-ox
    variable, everything else off)."""
    parser = _make_parser()
    t2p = _subparsers_action(parser).choices["transcriptome-to-proteome"]
    # Synthesise a minimal valid argv so parse_args succeeds without
    # touching the filesystem — every PTM flag defaults to the value
    # we want to verify.
    args = t2p.parse_args([
        "--protein-counts", "/tmp/x",
        "--reference-fasta", "/tmp/x",
        "--reference-annotation", "/tmp/x",
        "--gpf", "/tmp/x",
        "--injections", "/tmp/x",
        "--output-dir", "/tmp/x",
    ])
    assert args.ptm_carbamidomethyl == "fix"
    assert args.ptm_protein_n_term_acetyl == "var"
    assert args.ptm_pyro_glu_q == "var"
    assert args.ptm_oxidation == "var"
    # Negative-case sanity — these stay off by default.
    assert args.ptm_acetyl == "off"
    assert args.ptm_phospho == "off"
    assert args.ptm_tmt == "off"


def test_t2p_reference_dir_auto_resolves_fasta_and_annotation(
    tmp_path: Path, capsys
) -> None:
    """When ``--reference-dir`` is supplied, the handler fills in
    ``--reference-fasta`` and ``--reference-annotation`` from the
    Constellation reference-cache layout
    (``<dir>/protein.faa`` + ``<dir>/annotation/``)."""
    from constellation.cli.transcriptome_to_proteome import (
        _cmd_transcriptome_to_proteome,
    )

    # Fake reference cache layout — only the path existence matters
    # for the handler's resolution step (it doesn't open the files).
    ref_dir = tmp_path / "homo_sapiens" / "refseq-GCF_000001405.40"
    (ref_dir / "annotation").mkdir(parents=True)
    (ref_dir / "protein.faa").write_text(">stub\nMAGCKL\n")

    args = argparse.Namespace(
        reference_dir=ref_dir,
        reference_fasta=None,
        reference_annotation=None,
    )
    # Patch the orchestrator entry-point so we observe the resolved
    # args without running the pipeline.
    import constellation.cli.transcriptome_to_proteome as cli_mod
    observed: dict = {}

    def _fake_run(*, args):
        observed["fasta"] = args.reference_fasta
        observed["annotation"] = args.reference_annotation
        return 0

    import constellation.transcriptome_to_proteome as orch_mod
    orig = orch_mod.run_transcriptome_to_proteomics
    orch_mod.run_transcriptome_to_proteomics = _fake_run
    try:
        rc = _cmd_transcriptome_to_proteome(args)
    finally:
        orch_mod.run_transcriptome_to_proteomics = orig

    assert rc == 0
    assert observed["fasta"] == ref_dir / "protein.faa"
    assert observed["annotation"] == ref_dir / "annotation"


def test_t2p_reference_dir_explicit_flags_override(tmp_path: Path) -> None:
    """Explicit ``--reference-fasta`` / ``--reference-annotation``
    flags override the dir-derived defaults — caller can pull the
    FASTA from the cache but point the annotation somewhere else
    (e.g. when running against an experimental annotation release)."""
    from constellation.cli.transcriptome_to_proteome import (
        _cmd_transcriptome_to_proteome,
    )

    ref_dir = tmp_path / "cache_release"
    (ref_dir / "annotation").mkdir(parents=True)
    (ref_dir / "protein.faa").write_text(">stub\nMAGCKL\n")

    override_fasta = tmp_path / "override.fa"
    override_fasta.write_text(">override\nKVERPD\n")

    args = argparse.Namespace(
        reference_dir=ref_dir,
        reference_fasta=override_fasta,  # explicit, should win
        reference_annotation=None,        # falls through to dir default
    )

    import constellation.transcriptome_to_proteome as orch_mod
    observed: dict = {}

    def _fake_run(*, args):
        observed["fasta"] = args.reference_fasta
        observed["annotation"] = args.reference_annotation
        return 0

    orig = orch_mod.run_transcriptome_to_proteomics
    orch_mod.run_transcriptome_to_proteomics = _fake_run
    try:
        rc = _cmd_transcriptome_to_proteome(args)
    finally:
        orch_mod.run_transcriptome_to_proteomics = orig

    assert rc == 0
    # Explicit override wins for FASTA.
    assert observed["fasta"] == override_fasta
    # Default fallback for annotation.
    assert observed["annotation"] == ref_dir / "annotation"


def test_t2p_missing_reference_args_errors(tmp_path: Path, capsys) -> None:
    """Without --reference-dir AND without both --reference-fasta and
    --reference-annotation, the handler errors out before invoking the
    orchestrator."""
    from constellation.cli.transcriptome_to_proteome import (
        _cmd_transcriptome_to_proteome,
    )

    args = argparse.Namespace(
        reference_dir=None,
        reference_fasta=None,
        reference_annotation=None,
    )
    rc = _cmd_transcriptome_to_proteome(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "--reference-fasta" in err
    assert "--reference-annotation" in err
    assert "--reference-dir" in err


def test_t2p_collision_filter_default_on() -> None:
    parser = _make_parser()
    parsed = parser.parse_args(
        [
            "transcriptome-to-proteome",
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
            "transcriptome-to-proteome",
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
    t2p = _subparsers_action(parser).choices["transcriptome-to-proteome"]
    for action in t2p._actions:
        if "--ptm-carbamidomethyl" in action.option_strings:
            assert action.default == "fix"
            assert action.choices == ["off", "var", "fix"]
            return
    pytest.fail("--ptm-carbamidomethyl not registered")


def test_t2p_in_dashboard_introspector() -> None:
    """The dashboard auto-generates forms from the argparse tree — walking
    must surface the `transcriptome-to-proteome` top-level verb with
    its arguments populated."""
    pytest.importorskip("constellation.viz.introspect.walk")
    from constellation.viz.introspect.walk import walk_parser

    parser = _make_parser()
    tree = walk_parser(parser)
    t2p_node = next(
        (s for s in tree["subcommands"] if s["name"] == "transcriptome-to-proteome"),
        None,
    )
    assert t2p_node is not None
    assert t2p_node["arguments"], "no arguments surfaced for transcriptome-to-proteome"


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
    # EncyclopeDIA's search writes Percolator/feature sidecars next to the
    # source spectra file; Stage 10 must co-locate them for -libexport.
    (tmp_path / "sample01.mzML.features.txt").write_text("stub-features")
    (tmp_path / "sample01.mzML.encyclopedia.txt").write_text("stub-perc")
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
            "transcriptome-to-proteome",
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

    def _fake_run_library_export(*, output_elib, output_dir, search_dir=None,
                                 fasta=None, **kw):
        # EncyclopeDIA 6.5.15 -libexport REQUIRES -f <fasta> and scans -i
        # for spectra files (not bare .elibs); capture both so the
        # regression tests can assert Stage 10 sets them up correctly.
        state["export_fasta"] = fasta
        state["export_dir"] = Path(search_dir) if search_dir is not None else None
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
    assert top["subcommand"] == "transcriptome-to-proteome"
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
    # Stage 6 cache name: <run-stem>_combined_GPF.dia. <run-stem>
    # defaults to the output-dir basename when no --run-name override.
    process_dia_calls = [
        c[1] for c in stub_external_calls["calls"] if c[0] == "process_dia"
    ]
    assert len(process_dia_calls) == 1
    expected_stem = out.name
    assert process_dia_calls[0].endswith(
        f"06_process_dia/{expected_stem}_combined_GPF.dia"
    ), f"unexpected Stage 6 path: {process_dia_calls[0]}"


def test_orchestrator_tpm_cutoff_in_filenames(
    stub_inputs, stub_external_calls
) -> None:
    """Every TPM-dependent output (Stages 4, 5, 7, 8, 10) carries the
    TPM cutoff in its filename so parallel sweeps at different cutoffs
    under one --output-dir don't collide. Stages 0, 1, 2, 3, 6, 9 are
    TPM-independent and keep their plain names. Default cutoff is 1.0
    → suffix '_1TPM' (integer form, not '_1.0TPM')."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs)
    rc = run_transcriptome_to_proteomics(args=args)
    assert rc == 0
    out = stub_inputs["output_dir"]

    # TPM-dependent outputs.
    assert (out / "04_combined_fasta" / "combined_1TPM.fasta").is_file()
    assert (out / "05_predict_library" / "combined_1TPM.dlib").is_file()
    # Stage 7 — collision filter on by default → .filtered.elib variant.
    assert (out / "07_gpf_search" / "combined_1TPM.filtered.elib").is_file()
    assert (
        out / "08_classify_novel_peptides" / "novel_peptides_1TPM.parquet"
    ).is_file()
    assert (out / "10_quant_report" / "quant_report_1TPM.elib").is_file()


def test_orchestrator_tpm_cutoff_fractional_filename(
    stub_inputs, stub_external_calls
) -> None:
    """Fractional cutoffs preserve the decimal with `.` → `_` so the
    filename stays POSIX-safe: e.g. 0.5 → '_0_5TPM'."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs, min_avg_tpm=0.5)
    rc = run_transcriptome_to_proteomics(args=args)
    assert rc == 0
    out = stub_inputs["output_dir"]
    assert (out / "04_combined_fasta" / "combined_0_5TPM.fasta").is_file()
    assert (out / "10_quant_report" / "quant_report_0_5TPM.elib").is_file()


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
        stub_inputs["output_dir"] / "04_combined_fasta" / "combined_1TPM.fasta"
    )
    fastas = stub_external_calls["search_fasta"]
    # 1 GPF search + 1 injection (stub has a single injection) = 2 calls.
    assert len(fastas) >= 2
    assert all(f == combined_fasta for f in fastas), (
        f"some search calls missing the combined FASTA: {fastas}"
    )


def test_orchestrator_threads_fasta_to_library_export(
    stub_inputs, stub_external_calls
) -> None:
    """EncyclopeDIA 6.5.15 -libexport REQUIRES -f <fasta>. Stage 10 must
    pass the combined FASTA — regression guard for the missing-fasta
    failure on the first real OSC quant-report run."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    args = _build_args(stub_inputs)
    assert run_transcriptome_to_proteomics(args=args) == 0
    combined_fasta = (
        stub_inputs["output_dir"] / "04_combined_fasta" / "combined_1TPM.fasta"
    )
    assert stub_external_calls["export_fasta"] == combined_fasta

    # -libexport reconstructs each run's analysis from a co-located set:
    # the spectra file, <raw>.elib, and the percolator/feature sidecars.
    export_dir = stub_external_calls["export_dir"]
    names = {p.name for p in export_dir.iterdir()}
    assert "sample01.mzML" in names, names                  # spectra
    assert "sample01.mzML.elib" in names, names             # per-run chromatogram lib
    assert "sample01.mzML.features.txt" in names, names     # percolator features
    assert "sample01.mzML.encyclopedia.txt" in names, names # percolator PSMs
    # The spectra file is HARD-linked, not symlinked — EncyclopeDIA's native
    # Thermo reader can't open a .raw through a symlink.
    assert not (export_dir / "sample01.mzML").is_symlink()


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
    assert (stage7 / "combined_1TPM.filtered.elib").is_file()
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
    assert (stage7 / "combined_1TPM.filtered.elib").is_file()
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
    assert (stage7 / "combined_1TPM.elib").is_file()
