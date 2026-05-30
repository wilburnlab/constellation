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
    """The base set of required argparse flags. ``--reference`` and
    ``--reference-from`` are NOT argparse-level required — the handler
    enforces the "exactly one of them" rule post-parse so the
    actionable error message includes the recommended action."""
    parser = _make_parser()
    t2p = _subparsers_action(parser).choices["transcriptome-to-proteome"]
    required = {
        opt
        for action in t2p._actions
        if getattr(action, "required", False)
        for opt in action.option_strings
    }
    expected = {
        "--demux-dir",
        "--gpf",
        "--injections",
        "--output-dir",
    }
    assert expected.issubset(required), (
        f"missing required args: {expected - required}"
    )
    # --protein-fasta is optional — only needed when overriding the
    # ORF sequences the orchestrator derives from --demux-dir.
    optional = {
        opt
        for action in t2p._actions
        if not getattr(action, "required", False)
        for opt in action.option_strings
    }
    assert "--reference" in optional
    assert "--reference-from" in optional
    assert "--swissprot-reference" in optional
    assert "--protein-fasta" in optional


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
        "--demux-dir", "/tmp/x",
        "--reference", "homo_sapiens",
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


def _write_fake_reference(root: Path, organism: str, release_slug: str) -> Path:
    """Build a minimal portal-layout reference dir for handler tests.

    Creates the directory + a meta.toml v2 with the [contents] block so
    ``Reference.open()`` resolves it without needing real parquet
    bundles. Returns the release dir Path.
    """
    from constellation.sequencing.reference.handle import Handle, write_meta_toml

    release_dir = root / organism / release_slug
    (release_dir / "annotation").mkdir(parents=True)
    (release_dir / "annotation" / "features.parquet").write_text("stub")
    (release_dir / "annotation" / "manifest.json").write_text("{}")
    (release_dir / "protein.faa").write_text(">stub\nMAGCKL\n")
    handle = Handle(
        organism=organism,
        source=release_slug.split("-", 1)[0],
        release=release_slug.split("-", 1)[1],
    )
    write_meta_toml(
        release_dir,
        handle=handle,
        assembly_accession=None,
        assembly_name=None,
        annotation_release=None,
        constellation_version="test",
        urls={},
        sha256={},
        source_checksum_verified=False,
        has_genome=False,  # tests don't need a real genome bundle
        has_annotation=True,
        has_proteome=True,
    )
    return release_dir


def test_t2p_reference_handle_resolves_to_reference_object(
    tmp_path: Path, monkeypatch
) -> None:
    """``--reference <handle>`` opens a ``Reference`` whose
    ``protein_fasta_path`` + ``annotation_dir`` flow into the
    orchestrator."""
    from constellation.cli.transcriptome_to_proteome import (
        _resolve_reference_from_args,
    )

    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    release_dir = _write_fake_reference(tmp_path, "homo_sapiens", "refseq-test")

    args = argparse.Namespace(
        reference="homo_sapiens@refseq-test",
        reference_from=None,
    )
    ref = _resolve_reference_from_args(args)
    assert ref.protein_fasta_path == release_dir / "protein.faa"
    assert ref.annotation_dir == release_dir / "annotation"
    assert ref.has_proteome
    assert ref.has_annotation


def test_t2p_reference_missing_errors(tmp_path: Path, monkeypatch) -> None:
    """Neither --reference nor --reference-from supplied → exit 2 with
    the actionable hint."""
    from constellation.cli.transcriptome_to_proteome import (
        _cmd_transcriptome_to_proteome,
    )

    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    args = argparse.Namespace(reference=None, reference_from=None)
    rc = _cmd_transcriptome_to_proteome(args)
    assert rc == 2


def test_t2p_reference_from_demux_dir_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    """--reference-from pointing at a demux dir errors with a clear
    "demux is reference-free" message."""
    from constellation.cli.transcriptome_to_proteome import (
        _resolve_reference_from_args,
        _ResolutionError,
    )

    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()
    (demux_dir / "manifest.json").write_text(
        '{"schema_version": 4, "kind": "demux", "output_dir": "."}'
    )

    args = argparse.Namespace(reference=None, reference_from=demux_dir)
    with pytest.raises(_ResolutionError, match="demux is reference-free"):
        _resolve_reference_from_args(args)


def test_t2p_reference_and_reference_from_disagreement_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    """Explicit --reference plus --reference-from with disagreeing
    handles → ResolutionError."""
    from constellation.cli.transcriptome_to_proteome import (
        _resolve_reference_from_args,
        _ResolutionError,
    )

    align_dir = tmp_path / "align"
    align_dir.mkdir()
    (align_dir / "manifest.json").write_text(
        '{"schema_version": 4, "kind": "align", "output_dir": ".", '
        '"input_demux_dir": "x", "input_bam_paths": [], "samples": [], '
        '"reference_handle": "mus_musculus@refseq-other", '
        '"reference_path": "/tmp/x", "assembly_accession": null}'
    )

    args = argparse.Namespace(
        reference="homo_sapiens@refseq-X",
        reference_from=align_dir,
    )
    with pytest.raises(_ResolutionError, match="disagrees"):
        _resolve_reference_from_args(args)


# ──────────────────────────────────────────────────────────────────────
# Spectra input expansion (--gpf / --injections)
# ──────────────────────────────────────────────────────────────────────


def test_expand_spectra_passes_files_through(tmp_path: Path) -> None:
    """Plain file inputs are resolved and returned in sorted order."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
    )

    a = tmp_path / "a.raw"
    b = tmp_path / "b.raw"
    a.write_bytes(b"")
    b.write_bytes(b"")

    result = _expand_spectra_inputs([b, a], flag="--gpf")
    assert result == [a.resolve(), b.resolve()]


def test_expand_spectra_scans_directories(tmp_path: Path) -> None:
    """A directory is scanned non-recursively for spectra files."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
    )

    d = tmp_path / "runs"
    d.mkdir()
    (d / "sample01.raw").write_bytes(b"")
    (d / "sample02.raw").write_bytes(b"")
    (d / "notes.txt").write_text("ignored")
    # Nested directory should NOT be descended into.
    nested = d / "old_runs"
    nested.mkdir()
    (nested / "sample99.raw").write_bytes(b"")

    result = _expand_spectra_inputs([d], flag="--gpf")
    stems = sorted(p.stem for p in result)
    assert stems == ["sample01", "sample02"]


def test_expand_spectra_mixed_files_and_dirs(tmp_path: Path) -> None:
    """Mixing dir + explicit file inputs collects both."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
    )

    d = tmp_path / "runs"
    d.mkdir()
    (d / "sample01.raw").write_bytes(b"")
    standalone = tmp_path / "standalone.raw"
    standalone.write_bytes(b"")

    result = _expand_spectra_inputs([d, standalone], flag="--gpf")
    stems = sorted(p.stem for p in result)
    assert stems == ["sample01", "standalone"]


def test_expand_spectra_dia_wins_over_raw(
    tmp_path: Path, capsys
) -> None:
    """When the same stem has both .dia and .raw, .dia wins (cache
    already preprocessed; Stage 6 won't re-convert)."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
    )

    d = tmp_path / "runs"
    d.mkdir()
    raw = d / "sample01.raw"
    dia = d / "sample01.dia"
    raw.write_bytes(b"")
    dia.write_bytes(b"")

    result = _expand_spectra_inputs([d], flag="--gpf")
    assert result == [dia.resolve()]
    err = capsys.readouterr().err
    assert "sample01" in err
    assert "sample01.raw" in err


def test_expand_spectra_raw_wins_over_mzml(tmp_path: Path) -> None:
    """When the same stem has both .raw and .mzML, .raw wins (canonical
    Thermo input; mzML is the legacy intermediate)."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
    )

    d = tmp_path / "runs"
    d.mkdir()
    raw = d / "sample01.raw"
    mzml = d / "sample01.mzML"
    raw.write_bytes(b"")
    mzml.write_bytes(b"")

    result = _expand_spectra_inputs([d], flag="--gpf")
    assert result == [raw.resolve()]


def test_expand_spectra_accepts_bruker_d_directory(tmp_path: Path) -> None:
    """A ``.d`` directory bundle is passed through verbatim (not scanned
    for inner spectra files)."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
    )

    bundle = tmp_path / "sample01.d"
    bundle.mkdir()
    (bundle / "analysis.tdf").write_bytes(b"stub")

    result = _expand_spectra_inputs([bundle], flag="--gpf")
    assert result == [bundle.resolve()]


def test_expand_spectra_errors_on_empty_directory(tmp_path: Path) -> None:
    """A directory with no spectra files raises with the actionable
    error message the CLI surfaces as exit 2."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
        _ResolutionError,
    )

    d = tmp_path / "empty"
    d.mkdir()
    (d / "notes.txt").write_text("nothing to see")

    with pytest.raises(_ResolutionError, match="matched no spectra files"):
        _expand_spectra_inputs([d], flag="--gpf")


def test_expand_spectra_errors_on_unknown_extension(tmp_path: Path) -> None:
    """A bare path with an unsupported extension errors out instead of
    silently passing through."""
    from constellation.cli.transcriptome_to_proteome import (
        _expand_spectra_inputs,
        _ResolutionError,
    )

    p = tmp_path / "sample01.txt"
    p.write_text("")
    with pytest.raises(_ResolutionError, match="not a recognised spectra"):
        _expand_spectra_inputs([p], flag="--gpf")


def test_t2p_collision_filter_default_on() -> None:
    parser = _make_parser()
    parsed = parser.parse_args(
        [
            "transcriptome-to-proteome",
            "--demux-dir", "/x/c",
            "--reference", "homo_sapiens",
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
            "--demux-dir", "/x/c",
            "--reference", "homo_sapiens",
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
    """Lay down minimal input fixtures + a portal-layout fake reference
    cache that the orchestrator's helpers can read without throwing."""
    from constellation.sequencing.reference import Reference
    from constellation.sequencing.reference.handle import Handle, write_meta_toml

    # Reference cache root.
    cache_root = tmp_path / "refs"

    # Reference release dir with a real GFF3 annotation file
    # (Reference.gene_map() resolves it via _gene_map_from_dir → GFF3
    # path when no parquet bundle is present).
    ref_release = cache_root / "test_org" / "refseq-stub"
    (ref_release / "annotation").mkdir(parents=True)
    (ref_release / "annotation" / "annotation.gff3").write_text(
        "##gff-version 3\n"
        "chr1\tt\tCDS\t1\t10\t.\t+\t0\tID=c1;protein_id=REF_A;gene=GENE_A\n"
        "chr1\tt\tCDS\t20\t30\t.\t+\t0\tID=c2;protein_id=REF_B;gene=GENE_B\n"
    )
    (ref_release / "protein.faa").write_text(_REF_FASTA)
    write_meta_toml(
        ref_release,
        handle=Handle(organism="test_org", source="refseq", release="stub"),
        assembly_accession=None,
        assembly_name=None,
        annotation_release=None,
        constellation_version="test",
        urls={},
        sha256={},
        source_checksum_verified=False,
        has_genome=False,
        has_annotation=True,
        has_proteome=True,
    )

    # Fake SwissProt install in the same cache.
    sp_release = cache_root / "swissprot" / "uniprot-2026_02"
    sp_release.mkdir(parents=True)
    (sp_release / "protein.faa").write_text(">SP_X\nMKAAA\n")
    write_meta_toml(
        sp_release,
        handle=Handle(organism="swissprot", source="uniprot", release="2026_02"),
        assembly_accession=None,
        assembly_name=None,
        annotation_release=None,
        constellation_version="test",
        urls={},
        sha256={},
        source_checksum_verified=False,
        has_genome=False,
        has_annotation=False,
        has_proteome=True,
    )

    reference = Reference.open("test_org@refseq-stub", root=cache_root)
    swissprot_reference = Reference.open(
        "swissprot@uniprot-2026_02", root=cache_root
    )

    # Protein counts TSV (still the orchestrator's --demux-dir input).
    counts = tmp_path / "protein_counts.tsv"
    counts.write_text(
        "\tProtein\tqJS001\tSequence\n"
        "0\tN1\t10.0\t" + "M" * 120 + "\n"
        "1\tN2\t5.0\t" + "K" * 120 + "\n"
    )
    # ORF FASTA override (the --protein-fasta input).
    proteins = tmp_path / "proteins.fasta"
    proteins.write_text(
        ">N1 source=demux\n" + "M" * 120 + "\n"
        ">N2 source=demux\n" + "K" * 120 + "\n"
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
        "demux_dir": counts,
        "protein_fasta": proteins,
        "reference": reference,
        "swissprot_reference": swissprot_reference,
        "gpf": [gpf],
        "injections": [inj],
        "output_dir": out,
    }


def _build_args(stub: dict, **overrides) -> argparse.Namespace:
    """Build the namespace the orchestrator expects.

    Mirrors the CLI parser defaults — kept in sync by parsing through
    argparse and overriding by attribute. Does NOT carry the resolved
    Reference objects; the orchestrator takes those as separate kwargs.
    """
    parser = _make_parser()
    parsed = parser.parse_args(
        [
            "transcriptome-to-proteome",
            "--demux-dir", str(stub["demux_dir"]),
            "--protein-fasta", str(stub["protein_fasta"]),
            "--reference", str(stub["reference"].handle),
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

    def _fake_find(name):
        # The orchestrator's pre-Stage-0 preflight resolves the encyclopedia
        # jar and version-checks it. No real jar in the test env, so hand
        # back a fake >= 6.5.15 handle.
        from constellation.thirdparty.registry import ToolHandle, ToolSpec

        assert name == "encyclopedia"
        spec = ToolSpec(
            name="encyclopedia", env_var="CONSTELLATION_ENCYCLOPEDIA_HOME"
        )
        return ToolHandle(
            spec, tmp_path / "encyclopedia-6.5.15.jar", "env", "6.5.15"
        )

    import constellation.transcriptome_to_proteome as t2p

    monkeypatch.setattr("constellation.thirdparty.registry.find", _fake_find)
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
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
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
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
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
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
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
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
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
    assert run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"]) == 0
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
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
    assert rc == 0
    first_calls = list(stub_external_calls["calls"])
    assert first_calls, "first run made some external calls"

    # Reset call tracking; re-run with --resume.
    stub_external_calls["calls"].clear()
    args2 = _build_args(stub_inputs, resume=True)
    rc = run_transcriptome_to_proteomics(args=args2, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
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
    assert run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"]) == 0

    stage7 = stub_inputs["output_dir"] / "07_gpf_search"
    assert (stage7 / "combined_1TPM.filtered.elib").is_file()
    # Simulate a crash after the filtered elib was written but before the
    # stage was marked complete — clear Stage 7's _SUCCESS and the
    # top-level _SUCCESS (which would otherwise short-circuit the run).
    (stage7 / "_SUCCESS").unlink()
    (stub_inputs["output_dir"] / "_SUCCESS").unlink()

    stub_external_calls["calls"].clear()
    args2 = _build_args(stub_inputs, resume=True)
    assert run_transcriptome_to_proteomics(args=args2, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"]) == 0

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
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
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
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
    assert rc == 0
    stage7 = stub_inputs["output_dir"] / "07_gpf_search"
    assert not (stage7 / "collision_metadata.json").is_file()
    # _raw/ still exists because the jar's raw output lands there, but
    # no extra files were created since filter was skipped.
    assert (stage7 / "combined_1TPM.elib").is_file()


def test_orchestrator_preflight_rejects_old_encyclopedia(
    stub_inputs, monkeypatch, capsys
) -> None:
    """A sub-6.5.15 jar hard-errors at the preflight, before Stage 0 runs."""
    from constellation.thirdparty.registry import ToolHandle, ToolSpec
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )

    def _old_find(name):
        assert name == "encyclopedia"
        spec = ToolSpec(
            name="encyclopedia", env_var="CONSTELLATION_ENCYCLOPEDIA_HOME"
        )
        return ToolHandle(
            spec,
            Path("/nonexistent/encyclopedia-2.12.30-executable.jar"),
            "env",
            "2.12.30",
        )

    monkeypatch.setattr("constellation.thirdparty.registry.find", _old_find)

    args = _build_args(stub_inputs)
    rc = run_transcriptome_to_proteomics(args=args, reference=stub_inputs["reference"], swissprot_reference=stub_inputs["swissprot_reference"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "2.12.30" in err and "6.5.15" in err
    # Failed at preflight — Stage 0 never created its output dir.
    assert not (stub_inputs["output_dir"] / "00_deduped_refseq").exists()
