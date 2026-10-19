"""The captured EncyclopeDIA command-line surface.

Three layers, by what they need:

  * always: the text normaliser and help parser on small inputs, and the
    packaged JSON against the raw captures it is built from;
  * always: every flag the hand-written wrappers emit is one the pinned
    jar documents (this is the check that would have caught the stale
    ``-scribetwodda`` / ``-scribetwodia`` notes);
  * with the pinned jar installed: a fresh capture equals the snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia import cli_spec
from constellation.massspec.search.encyclopedia.cli_spec import (
    HOST_CORES,
    build_cli_spec,
    is_known_flag,
    known_flags,
    load_cli_spec,
    normalize_help,
    parse_help,
    program_argv,
    program_names,
    raw_help_dir,
    unknown_flags,
)
from constellation.thirdparty.encyclopedia import MINIMUM_VERSION


# ── normalize_help ─────────────────────────────────────────────────────


def test_normalize_strips_timestamps_and_thermo_server_line() -> None:
    raw = (
        "[19:06:28] ScribeTwo Help\n"
        "[19:06:28] Thermo server: extracting common runtime files...\n"
        "ScribeTwo is a search engine.\n"
        "Required Parameters: \n"
        "\t-i\tinput .DIA or .MZML file\n"
        "\n\n"
    )
    assert normalize_help(raw) == (
        "ScribeTwo Help\n"
        "ScribeTwo is a search engine.\n"
        "Required Parameters:\n"
        "\t-i\tinput .DIA or .MZML file\n"
    )


def test_normalize_is_independent_of_where_the_thermo_line_lands() -> None:
    # The line comes from another thread, so its position varies by run.
    a = "[01:00:00] Thermo server: extracting...\n[01:00:00] Title\nbody\n"
    b = "[02:30:59] Title\n[02:30:59] Thermo server: extracting...\nbody\n"
    assert normalize_help(a) == normalize_help(b) == "Title\nbody\n"


def test_normalize_survives_the_thermo_line_landing_mid_line() -> None:
    """The helper thread does not wait for a line break: seen about once
    in thirty captures of ``-batch -h``, where it cut the crash line in
    two. What stood before it must not be dropped with it."""
    clean = (
        "[23:21:49] EncyclopeDIA Batch Help\n"
        "[23:21:49] Thermo server: extracting common runtime files...\n"
        'Exception in thread "main" java.lang.NullPointerException\n'
        "\tat java.base/java.io.File.<init>(File.java:278)\n"
    )
    interrupted = (
        "[23:21:49] EncyclopeDIA Batch Help\n"
        'Exception in thread "main" [23:21:49] Thermo server: extracting common runtime files...\n'
        "java.lang.NullPointerException\n"
        "\tat java.base/java.io.File.<init>(File.java:278)\n"
    )
    assert normalize_help(interrupted) == normalize_help(clean)
    assert 'Exception in thread "main" java.lang.NullPointerException' in normalize_help(clean)
    # At the very end of the output, the fragment before it is still kept.
    assert normalize_help("Title\nlast words [01:02:03] Thermo server: bye\n") == (
        "Title\nlast words\n"
    )


def test_normalize_replaces_host_core_count_default() -> None:
    raw = "T\n\t-numberOfThreadsUsed                 (default: 96)\n\t-ptol  (default: 10)\n"
    out = normalize_help(raw)
    assert f"(default: {HOST_CORES})" in out
    assert "(default: 10)" in out  # other numeric defaults untouched


# ── parse_help ─────────────────────────────────────────────────────────


_HELP = """\
Demo Help
Demo is a demonstration.
Required Parameters:
\t-i\tinput file
\t-a\talign between files (default=true)
Other Programs:
\t-sub\trun Sub (use -sub -h for help)
Optional Parameters:
\t-o\toutput report file (default: [input file].demo.txt)
\t-searchType\tsearch type: auto, dia, or dda (default: auto)
\t-mode                 (default: off, values: off|var|fix)
\t-better\t(true/false, default: [true])
\t-mzid\twrite mzIdentML (DDA mode only, default: false)
\t-fasta\t(default: empty), use to reset annotations
\t-minLength/-maxLength\tlength range (default: 7-31)
\t-ptm...\tnamed PTM state (off|var|fix)
\t\tFixed TMT needs care.
\t-fixed\tfixed site masses
\t-fixed                (default: C=57.0214635)
\t-plain

Notes:
\tPeptide length is fixed at 7-31 residues.
"""


def test_parse_help_shapes() -> None:
    p = parse_help(_HELP)
    f = p["flags"]
    assert p["title"] == "Demo Help"
    assert p["summary"] == "Demo is a demonstration."
    assert p["subprograms"] == {"-sub": "run Sub (use -sub -h for help)"}
    assert "-sub" not in f

    assert f["-i"] == {"required": True, "section": "Required Parameters", "description": "input file"}
    # Listed under "Required" by the jar, but it has a default.
    assert f["-a"]["required"] is False and f["-a"]["default"] == "true"
    # Brackets inside a default are kept; only a fully bracketed one is unwrapped.
    assert f["-o"]["default"] == "[input file].demo.txt"
    assert f["-better"]["default"] == "true" and f["-better"]["values"] == ["true", "false"]
    assert f["-searchType"]["default"] == "auto"
    assert f["-mode"] == {
        "required": False,
        "section": "Optional Parameters",
        "default": "off",
        "values": ["off", "var", "fix"],
    }
    # A non-default part of the group stays in the description.
    assert f["-mzid"]["default"] == "false"
    assert f["-mzid"]["description"] == "write mzIdentML (DDA mode only)"
    assert f["-fasta"]["default"] == "empty"
    assert f["-fasta"]["description"] == "use to reset annotations"
    # Two flags on one line.
    assert f["-minLength"] == f["-maxLength"]
    assert f["-minLength"]["default"] == "7-31"
    # A flag printed twice keeps the first description and gains the default.
    assert f["-fixed"]["description"] == "fixed site masses"
    assert f["-fixed"]["default"] == "C=57.0214635"
    assert f["-plain"] == {"required": False, "section": "Optional Parameters"}
    # A family is not a flag.
    assert "-ptm..." not in f and "-ptm" not in f
    assert p["flag_patterns"] == {"-ptm*": "named PTM state (off|var|fix)"}
    # Unrecognised lines are kept, not dropped.
    assert "Fixed TMT needs care." in p["notes"]
    assert "Peptide length is fixed at 7-31 residues." in p["notes"]


# ── packaged JSON vs raw captures ──────────────────────────────────────


def test_packaged_json_is_in_step_with_raw_captures() -> None:
    """`scripts/build-encyclopedia-cli-json.py` was re-run after the last
    survey: rebuilding from the raw help gives exactly the packaged JSON."""
    rebuilt = build_cli_spec(raw_help_dir(MINIMUM_VERSION), version=MINIMUM_VERSION)
    assert json.loads(json.dumps(rebuilt)) == dict(load_cli_spec())


def test_snapshot_is_for_the_pinned_version() -> None:
    assert load_cli_spec()["encyclopedia_version"] == MINIMUM_VERSION
    index = json.loads((raw_help_dir(MINIMUM_VERSION).parent / "programs.json").read_text())
    assert index["encyclopedia_version"] == MINIMUM_VERSION
    assert {p["name"] for p in index["programs"]} == set(program_names())
    for name in program_names():
        assert (raw_help_dir(MINIMUM_VERSION) / f"{name}.txt").is_file()


def test_program_argv() -> None:
    assert program_argv("encyclopedia") == ()  # the default search has no program flag
    assert program_argv("scribetwo") == ("-scribetwo",)
    assert program_argv("convert.processDIA") == ("-convert", "-processDIA")
    with pytest.raises(KeyError, match="unknown EncyclopeDIA program"):
        program_argv("scribetwodda")


def test_search_programs_as_the_plan_assumes() -> None:
    """The facts the search-mode toggle is built on."""
    names = set(program_names())
    assert {"encyclopedia", "scribetwo", "scribe", "libexport"} <= names
    # 6.5.15's two ScribeTwo programs are one program with a search type.
    assert not {"scribetwodda", "scribetwodia"} & names
    search_type = load_cli_spec()["programs"]["scribetwo"]["flags"]["-searchType"]
    assert search_type["default"] == "auto"
    for mode in ("auto", "dia", "dda_chimeric", "dda"):
        assert mode in search_type["description"]
    # Spectrum-centric can run from a FASTA alone; peptide-centric cannot.
    assert not load_cli_spec()["programs"]["scribetwo"]["flags"]["-l"]["required"]
    assert load_cli_spec()["programs"]["encyclopedia"]["flags"]["-l"]["required"]
    assert is_known_flag("scribetwo", "-ptmOxidation")  # via the -ptm... family
    assert not is_known_flag("encyclopedia", "-searchType")


def test_truncated_help_name_matches_the_full_flag() -> None:
    """The jar's help table cuts long names off; the real flag still counts."""
    flags = known_flags("encyclopedia")
    assert "-numberOfExtraDecoyLibrariesSearche" in flags  # as printed
    assert "-numberOfExtraDecoyLibrariesSearched" not in flags
    assert is_known_flag("encyclopedia", "-numberOfExtraDecoyLibrariesSearched")
    assert not is_known_flag("encyclopedia", "-numberOfThreadsUsedX")  # not a truncated name


def test_undocumented_parameters_record_is_still_true() -> None:
    """The record lists only names the pinned jar's help does not print.

    If a later jar documents one, it shows up in the snapshot and this
    fails: delete it from ``UNDOCUMENTED_PARAMETERS``.
    """
    from constellation.massspec.search.encyclopedia.cli_spec import UNDOCUMENTED_PARAMETERS

    assert set(UNDOCUMENTED_PARAMETERS) <= set(program_names())
    for program, names in UNDOCUMENTED_PARAMETERS.items():
        assert len(set(names)) == len(names)
        documented = [n for n in names if is_known_flag(program, n)]
        assert documented == [], f"{program}: now documented: {documented}"


def test_unknown_flags_ignores_values_and_program_prefix() -> None:
    argv = ["-convert", "-processDIA", "-i", "a.raw:b.raw", "-o", "-1", "-bogus", "x"]
    assert unknown_flags("convert.processDIA", argv) == ["-bogus"]
    # ScribeTwo's "none" value is a bare dash, not a flag.
    assert unknown_flags("scribetwo", ["-fixed", "-", "-variable", "-"]) == []


# ── the hand-written wrappers only emit documented flags ───────────────


def test_library_search_wrapper_flags_are_documented(tmp_path: Path) -> None:
    from constellation.massspec.search.encyclopedia import build_library_search_args

    args = build_library_search_args(
        input_file=tmp_path / "a.dia",
        library=tmp_path / "lib.dlib",
        fasta=tmp_path / "bg.fasta",
        report_output=tmp_path / "a.encyclopedia.txt",
        precursor_tolerance=10.0,
        fragment_tolerance=0.35,
        fragment_tolerance_unit="Da",
        library_fragment_tolerance=10.0,
        acquisition="DIA",
        enzyme="Trypsin",
        fragmentation="HCD",
        percolator_version="v3-05",
        percolator_threshold=0.01,
        percolator_protein_threshold=0.01,
        threads=8,
    )
    assert unknown_flags("encyclopedia", args) == []
    assert {"-ptolunits", "-ftolunits", "-lftolunits"} <= set(args)


def test_predict_library_wrapper_flags_are_documented(tmp_path: Path) -> None:
    from constellation.massspec.search.encyclopedia import build_predict_library_args
    from constellation.massspec.search.encyclopedia.ptm_defaults import default_for

    names = ("Acetyl", "ProteinNTermAcetyl", "Carbamidomethyl", "Deamidation", "Dimethyl",
             "GlyGly", "HexNAc", "Methyl", "Oxidation", "Phospho", "PyroGluQ", "Succinyl",
             "Trimethyl", "TMT")
    args = build_predict_library_args(
        fasta=tmp_path / "x.fasta",
        output_dlib=tmp_path / "x.dlib",
        ptms={n: default_for(n) for n in names},
        prediction_cache=tmp_path / "cache",
    )
    assert unknown_flags("convert.fastaToJChronologerLibrary", args) == []
    # ... and the wrapper covers every flag the jar documents for it.
    assert known_flags("convert.fastaToJChronologerLibrary") <= set(args) | {"-h", "-help"}


def test_process_dia_wrapper_flags_are_documented(tmp_path: Path) -> None:
    from constellation.massspec.search.encyclopedia import build_process_dia_args

    args = build_process_dia_args(
        inputs=[tmp_path / "a.raw", tmp_path / "b.raw"], output_dia=tmp_path / "m.dia"
    )
    assert unknown_flags("convert.processDIA", args) == []


def test_library_export_wrapper_flags_are_documented(tmp_path: Path) -> None:
    from constellation.massspec.search.encyclopedia import build_library_export_args

    args = build_library_export_args(
        search_dir=tmp_path,
        library=tmp_path / "lib.elib",
        output_elib=tmp_path / "out.elib",
        fasta=tmp_path / "bg.fasta",
        write_blib=True,
    )
    assert unknown_flags("libexport", args) == []


# ── with the pinned jar installed ──────────────────────────────────────


def _pinned_jar_installed() -> bool:
    from constellation.thirdparty.registry import try_find

    handle = try_find("encyclopedia")
    return handle is not None and handle.path.is_file() and handle.version == MINIMUM_VERSION


@pytest.mark.skipif(
    not _pinned_jar_installed(),
    reason=f"needs EncyclopeDIA {MINIMUM_VERSION} installed (constellation doctor)",
)
def test_fresh_capture_equals_snapshot() -> None:
    """The checked-in help is what the installed jar prints today."""
    texts, probes = cli_spec.capture_help()
    snapshot = {
        p.stem: p.read_text() for p in sorted(raw_help_dir(MINIMUM_VERSION).glob("*.txt"))
    }
    assert sorted(texts) == sorted(snapshot)
    for name in sorted(texts):
        assert texts[name] == snapshot[name], f"help for {name!r} changed"
    assert sorted(p.name for p in probes) == list(program_names())
