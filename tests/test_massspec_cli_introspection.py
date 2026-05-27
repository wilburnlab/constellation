"""Tier A tests for :mod:`constellation.massspec.cli`.

Verifies that all four ``constellation massspec <subcommand>`` parsers
register with the right arg surface and that the dashboard's argparse
walker sees them. Catches dropped ``required=True``, missing args, or
broken type hints before the implementation PRs run end-to-end.

No subprocess invocation — pure argparse introspection.
"""

from __future__ import annotations

import argparse

import pytest

from constellation.massspec.cli import build_parser


def _build_massspec_only_parser() -> argparse.ArgumentParser:
    """Mount the massspec subtree on a fresh root parser for testing."""
    root = argparse.ArgumentParser(prog="constellation")
    subs = root.add_subparsers(dest="subcommand", required=True)
    build_parser(subs)
    return root


# ── subcommand presence ────────────────────────────────────────────────


def test_all_four_subcommands_register() -> None:
    parser = _build_massspec_only_parser()
    # Walk to the massspec subparser
    ms_parser = _massspec_subparser(parser)
    ms_subs = _subparsers_action(ms_parser)
    assert set(ms_subs.choices) == {
        "search",
        "predict-library",
        "process-dia",
        "library-export",
        "classify-novel-peptides",
        "collision-filter",
    }


# ── per-subcommand arg surface ─────────────────────────────────────────


@pytest.mark.parametrize(
    "subcommand, required_args",
    [
        # --fasta is optional for search per EncyclopeDIA 6.5.15 — the
        # default-mode library search doesn't require it when the
        # library already carries decoys (e.g. predict-library output).
        ("search", {"--mzml", "--library", "--output-dir"}),
        ("predict-library", {"--fasta", "--output-dlib", "--output-dir"}),
        ("process-dia", {"--inputs", "--output-dia", "--output-dir"}),
        (
            "library-export",
            {"--search-dir", "--library", "--output-elib", "--output-dir"},
        ),
        (
            "classify-novel-peptides",
            {
                "--search-results", "--library", "--alignment-hits",
                "--reference-fasta", "--novel-fasta", "--output-dir",
            },
        ),
        (
            "collision-filter",
            {"--elib", "--dia", "--output-elib", "--output-dir"},
        ),
    ],
)
def test_subcommand_required_args(
    subcommand: str, required_args: set[str]
) -> None:
    parser = _build_massspec_only_parser()
    ms_parser = _massspec_subparser(parser)
    ms_subs = _subparsers_action(ms_parser)
    sub = ms_subs.choices[subcommand]
    seen_required = {
        opt
        for action in sub._actions
        if getattr(action, "required", False)
        for opt in action.option_strings
    }
    assert required_args.issubset(seen_required), (
        f"{subcommand}: expected required {required_args}, got {seen_required}"
    )


@pytest.mark.parametrize(
    "subcommand",
    ["search", "predict-library", "process-dia", "library-export"],
)
def test_subcommand_has_jvm_args(subcommand: str) -> None:
    """Every subcommand surfaces ``--jvm-heap``, ``--jvm-heap-min``,
    ``--jvm-tmpdir``, ``--encyclopedia-arg`` so users can tune the
    runner uniformly."""
    parser = _build_massspec_only_parser()
    sub = _subparsers_action(_massspec_subparser(parser)).choices[subcommand]
    all_opts = {
        opt for action in sub._actions for opt in action.option_strings
    }
    for expected in (
        "--jvm-heap",
        "--jvm-heap-min",
        "--jvm-tmpdir",
        "--encyclopedia-arg",
    ):
        assert expected in all_opts, f"{subcommand}: missing {expected}"


def test_predict_library_has_per_ptm_flags() -> None:
    """All 14 known PTMs surface as ``--ptm-<kebab>`` flags."""
    parser = _build_massspec_only_parser()
    pl = _subparsers_action(_massspec_subparser(parser)).choices[
        "predict-library"
    ]
    all_opts = {
        opt for action in pl._actions for opt in action.option_strings
    }
    expected_ptms = {
        "--ptm-acetyl",
        "--ptm-protein-n-term-acetyl",
        "--ptm-carbamidomethyl",
        "--ptm-deamidation",
        "--ptm-dimethyl",
        "--ptm-gly-gly",
        "--ptm-hex-n-ac",
        "--ptm-methyl",
        "--ptm-oxidation",
        "--ptm-phospho",
        "--ptm-pyro-glu-q",
        "--ptm-succinyl",
        "--ptm-trimethyl",
        "--ptm-tmt",
    }
    missing = expected_ptms - all_opts
    assert not missing, f"missing PTM flags: {missing}"


def test_predict_library_carbamidomethyl_default_fix() -> None:
    """Carbamidomethyl default is ``fix`` (EncyclopeDIA's default)."""
    parser = _build_massspec_only_parser()
    pl = _subparsers_action(_massspec_subparser(parser)).choices[
        "predict-library"
    ]
    for action in pl._actions:
        if "--ptm-carbamidomethyl" in action.option_strings:
            assert action.default == "fix"
            assert action.choices == ["off", "var", "fix"]
            return
    pytest.fail("--ptm-carbamidomethyl not registered")


# ── dashboard introspection ────────────────────────────────────────────


def test_dashboard_introspector_sees_all_subcommands() -> None:
    """The viz dashboard auto-generates forms from the argparse tree.
    Walking the parser must surface massspec → all subcommands with
    arguments populated.
    """
    pytest.importorskip("constellation.viz.introspect.walk")
    from constellation.viz.introspect.walk import walk_parser

    parser = _build_massspec_only_parser()
    tree = walk_parser(parser)
    # Top-level should have 'massspec' as a subcommand
    massspec_node = next(
        (s for s in tree["subcommands"] if s["name"] == "massspec"), None
    )
    assert massspec_node is not None, "massspec not in top-level subcommands"
    sub_names = {s["name"] for s in massspec_node["subcommands"]}
    assert sub_names == {
        "search",
        "predict-library",
        "process-dia",
        "library-export",
        "classify-novel-peptides",
        "collision-filter",
    }
    # Each subcommand has a non-empty argument list
    for sub in massspec_node["subcommands"]:
        assert sub["arguments"], f"{sub['name']}: no arguments surfaced"


# ── stub handlers (subcommands still pending implementation) ───────────


def test_no_stub_handlers_remain() -> None:
    """All five massspec subcommands now have wired handlers — no stubs
    left. This guards against regressions where a subcommand's handler
    gets reverted to ``_not_yet_wired`` without updating the parser.
    """
    parser = _build_massspec_only_parser()
    ms_subs = _subparsers_action(_massspec_subparser(parser))
    for subcommand, sub in ms_subs.choices.items():
        handler = sub.get_default("func")
        assert handler is not None, f"{subcommand}: no func set"
        # Stub handlers are named ``_not_yet_wired`` — fail if any survive.
        assert handler.__name__ != "_not_yet_wired", (
            f"{subcommand}: still wired to the _not_yet_wired stub"
        )


# ── --jvm-heap suffix validation ──────────────────────────────────────


@pytest.mark.parametrize(
    "subcommand,minimal_args",
    [
        (
            "predict-library",
            ["--fasta", "x.fa", "--output-dlib", "x.dlib",
             "--output-dir", "/tmp/x"],
        ),
        (
            "search",
            ["--mzml", "x.mzML", "--library", "x.elib", "--fasta", "x.fa",
             "--output-dir", "/tmp/x"],
        ),
        (
            "process-dia",
            ["--inputs", "a.mzML", "--output-dia", "x.dia",
             "--output-dir", "/tmp/x"],
        ),
        (
            "library-export",
            ["--search-dir", "/tmp/sr", "--library", "x.dlib",
             "--output-elib", "x.elib", "--output-dir", "/tmp/x"],
        ),
    ],
)
def test_jvm_heap_bare_number_rejected(
    subcommand: str,
    minimal_args: list[str],
    capsys: pytest.CaptureFixture,
) -> None:
    """``--jvm-heap 24`` (no suffix) used to be accepted and crash the
    JVM with the cryptic 'Too small maximum heap' message — the bare
    number was interpreted as 24 BYTES. The parser now rejects it at
    parse time with a clear error pointing at the missing unit
    suffix."""
    parser = _build_massspec_only_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["massspec", subcommand, *minimal_args, "--jvm-heap", "24"]
        )
    err = capsys.readouterr().err
    assert "invalid JVM heap value" in err or "must be" in err


@pytest.mark.parametrize(
    "valid_value",
    ["24g", "8g", "2048m", "512m", "1024k", "1G", "2M"],
)
def test_jvm_heap_valid_suffixes_accepted(valid_value: str) -> None:
    parser = _build_massspec_only_parser()
    parsed = parser.parse_args(
        [
            "massspec", "predict-library",
            "--fasta", "x.fa",
            "--output-dlib", "x.dlib",
            "--output-dir", "/tmp/x",
            "--jvm-heap", valid_value,
        ]
    )
    assert parsed.jvm_heap_max == valid_value


# ── collision-filter handler (end-to-end against synthetic .elib/.dia) ──


def _build_synthetic_elib(path, entries) -> None:
    import sqlite3

    import torch

    from constellation.massspec.io.encyclopedia._codec import compress_mz

    con = sqlite3.connect(str(path))
    try:
        # Full entries schema — apply_collision_filter reads every column
        # in massspec.io.encyclopedia._sql._ENTRIES_COLS.
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
            "MedianChromatogramArray BLOB, SourceFile TEXT)"
        )
        con.execute(
            "CREATE TABLE peptidescores ("
            "PrecursorCharge INTEGER, PeptideModSeq TEXT, PeptideSeq TEXT, "
            "SourceFile TEXT, QValue REAL, "
            "PosteriorErrorProbability REAL, IsDecoy INTEGER)"
        )
        for e in entries:
            mz_arr = torch.tensor(e["mz"], dtype=torch.float64)
            blob, n = compress_mz(mz_arr)
            con.execute(
                "INSERT INTO entries (PrecursorMz, PrecursorCharge, "
                "PeptideModSeq, PeptideSeq, Copies, RTInSeconds, Score, "
                "MassEncodedLength, MassArray, "
                "IntensityEncodedLength, IntensityArray, "
                "CorrelationEncodedLength, CorrelationArray, "
                "QuantifiedIonsArray, RTInSecondsStart, RTInSecondsStop, "
                "MedianChromatogramEncodedLength, MedianChromatogramArray, "
                "SourceFile) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    e["precursor_mz"], 2, e["modseq"], e["modseq"], 1,
                    e["rt"], e["score"], n, blob,
                    0, b"", 0, b"", None,
                    e["rt"] - 1.0, e["rt"] + 1.0, 0, b"", "run.mzML",
                ),
            )
            con.execute(
                "INSERT INTO peptidescores VALUES (?, ?, ?, ?, ?, ?, ?)",
                (2, e["modseq"], e["modseq"], "run.mzML", 0.01, 0.001, 0),
            )
        con.commit()
    finally:
        con.close()


def _build_synthetic_dia(path, windows) -> None:
    import sqlite3

    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE ranges ("
            "Start REAL, Stop REAL, DutyCycle REAL, NumWindows INTEGER)"
        )
        for lo, hi in windows:
            con.execute(
                "INSERT INTO ranges VALUES (?, ?, ?, ?)",
                (lo, hi, 1.0, len(windows)),
            )
        con.commit()
    finally:
        con.close()


def test_collision_filter_handler_writes_filtered_elib(tmp_path) -> None:
    """The collision-filter subcommand runs end-to-end: filters the
    .elib, writes the filtered .elib + collision_metadata.json +
    manifest.json + _SUCCESS."""
    import sqlite3

    elib = tmp_path / "search.elib"
    dia = tmp_path / "combined.dia"
    shared = [100.0, 200.0, 300.0, 400.0]
    _build_synthetic_elib(
        elib,
        [
            {"modseq": "AAAR", "precursor_mz": 500.0, "rt": 1000.0,
             "score": 0.01, "mz": shared},   # better → winner
            {"modseq": "BBBR", "precursor_mz": 500.0, "rt": 1002.0,
             "score": 0.10, "mz": shared},   # worse → loser
        ],
    )
    _build_synthetic_dia(dia, [(450.0, 550.0)])

    parser = _build_massspec_only_parser()
    out_elib = tmp_path / "filtered.elib"
    out_dir = tmp_path / "run"
    args = parser.parse_args(
        [
            "massspec", "collision-filter",
            "--elib", str(elib),
            "--dia", str(dia),
            "--output-elib", str(out_elib),
            "--output-dir", str(out_dir),
            "--no-ingest",
            "--no-progress",
        ]
    )
    rc = args.func(args)
    assert rc == 0
    # Filtered .elib written, with the loser dropped.
    assert out_elib.is_file()
    con = sqlite3.connect(str(out_elib))
    try:
        modseqs = {
            r[0] for r in con.execute("SELECT PeptideModSeq FROM entries")
        }
    finally:
        con.close()
    assert modseqs == {"AAAR"}   # BBBR dropped
    # Sidecars.
    assert (out_dir / "collision_metadata.json").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "_SUCCESS").is_file()


def test_collision_filter_handler_resume_short_circuits(tmp_path) -> None:
    elib = tmp_path / "search.elib"
    dia = tmp_path / "combined.dia"
    _build_synthetic_elib(
        elib,
        [{"modseq": "AAAR", "precursor_mz": 500.0, "rt": 1000.0,
          "score": 0.01, "mz": [100.0, 200.0]}],
    )
    _build_synthetic_dia(dia, [(450.0, 550.0)])
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "_SUCCESS").write_bytes(b"")

    parser = _build_massspec_only_parser()
    args = parser.parse_args(
        [
            "massspec", "collision-filter",
            "--elib", str(elib), "--dia", str(dia),
            "--output-elib", str(tmp_path / "filtered.elib"),
            "--output-dir", str(out_dir),
            "--resume", "--no-ingest", "--no-progress",
        ]
    )
    rc = args.func(args)
    assert rc == 0   # short-circuits on existing _SUCCESS


# ── helpers ────────────────────────────────────────────────────────────


def _massspec_subparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return _subparsers_action(parser).choices["massspec"]


def _subparsers_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("no _SubParsersAction found on parser")
