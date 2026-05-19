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
    }
    # Each subcommand has a non-empty argument list
    for sub in massspec_node["subcommands"]:
        assert sub["arguments"], f"{sub['name']}: no arguments surfaced"


# ── stub handlers (subcommands still pending implementation) ───────────


@pytest.mark.parametrize(
    "subcommand,minimal_args",
    [
        (
            "library-export",
            ["--search-dir", "/tmp/sr", "--library", "x.dlib",
             "--output-elib", "x.elib", "--output-dir", "/tmp/x"],
        ),
    ],
)
def test_stub_handlers_return_exit_2(
    subcommand: str, minimal_args: list[str], capsys: pytest.CaptureFixture
) -> None:
    """Subcommands whose runner isn't wired yet print a helpful error +
    return exit code 2. predict-library, process-dia, and search are
    excluded — all three are wired."""
    parser = _build_massspec_only_parser()
    args = parser.parse_args(["massspec", subcommand, *minimal_args])
    rc = args.func(args)
    assert rc == 2
    captured = capsys.readouterr()
    assert subcommand in captured.err
    assert "not yet implemented" in captured.err


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
