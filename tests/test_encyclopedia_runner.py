"""The generic EncyclopeDIA runner, the search-mode resolver and the output locator.

All of it is checked without a JVM: the runner's job is to refuse a bad
command line *before* Java starts, and what the jar accepts comes from
the checked-in snapshot of its own help.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia import cli_spec
from constellation.massspec.search.encyclopedia.outputs import (
    MissingOutputError,
    expected_names,
    locate_outputs,
    output_inventory,
)
from constellation.massspec.search.encyclopedia.params import (
    PEPTIDE_CENTRIC,
    SPECTRUM_CENTRIC,
    SearchModeError,
    SearchParameters,
    Tolerance,
    resolve_search_mode,
)
from constellation.massspec.search.encyclopedia.runner import (
    UnknownFlagError,
    UnknownProgramError,
    UnsupportedVersionError,
    compose_argv,
    format_value,
    run_program,
)

# ── compose_argv ───────────────────────────────────────────────────────


def test_argv_is_the_program_selector_then_flags_in_order() -> None:
    assert compose_argv("encyclopedia", {"-i": "run.dia", "-l": "lib.dlib", "-f": "db.fasta"}) == [
        "-i", "run.dia", "-l", "lib.dlib", "-f", "db.fasta",
    ]  # fmt: skip
    assert compose_argv("scribetwo", [("-i", Path("run.dia")), ("-searchType", "dda")]) == [
        "-scribetwo", "-i", "run.dia", "-searchType", "dda",
    ]  # fmt: skip
    assert compose_argv("convert.fastaToJChronologerLibrary", {"-i": "db.fasta"})[:2] == [
        "-convert",
        "-fastaToJChronologerLibrary",
    ]


def test_values_are_formatted_as_the_jar_reads_them() -> None:
    assert format_value(True) == "true" and format_value(False) == "false"
    assert format_value(0.35) == "0.35" and format_value(12) == "12"
    assert format_value(Path("/a/b.dia")) == "/a/b.dia"
    # An empty value is a value: no fixed modification.
    assert compose_argv("encyclopedia", {"-fixed": ""}) == ["-fixed", ""]


def test_a_misspelt_flag_is_refused_with_the_near_misses() -> None:
    with pytest.raises(UnknownFlagError) as raised:
        compose_argv("encyclopedia", {"-i": "run.dia", "-ptoll": 10, "-bogus": 1})
    message = str(raised.value)
    assert "-ptoll  (did you mean -ptol" in message and "-bogus" in message
    assert "massspec encyclopedia flags encyclopedia" in message


def test_a_flag_of_another_program_is_refused() -> None:
    assert cli_spec.is_known_flag("scribetwo", "-searchType")
    with pytest.raises(UnknownFlagError, match="-searchType"):
        compose_argv("encyclopedia", {"-searchType": "dda"})
    # The programs the stale 6.5.15 notes listed do not exist.
    with pytest.raises(UnknownProgramError, match="no program 'scribetwodda'"):
        compose_argv("scribetwodda", {})


def test_flag_families_and_truncated_names_are_known() -> None:
    # ScribeTwo documents its PTM flags as a family, `-ptm...`.
    assert compose_argv("scribetwo", {"-ptmCarbamidomethylC": "fix"})[-2:] == [
        "-ptmCarbamidomethylC",
        "fix",
    ]
    # The help table cuts names at 35 characters.
    assert compose_argv("encyclopedia", {"-numberOfExtraDecoyLibrariesSearched": 0.0})


def test_an_undocumented_parameter_has_to_be_allowed_by_name() -> None:
    with pytest.raises(UnknownFlagError):
        compose_argv("encyclopedia", {"-instrument": "Iontrap/Iontrap"})
    assert compose_argv(
        "encyclopedia", {"-instrument": "Iontrap/Iontrap"}, allow_undocumented=["-instrument"]
    ) == ["-instrument", "Iontrap/Iontrap"]


def test_a_flag_needs_a_value_and_a_dash() -> None:
    with pytest.raises(UnknownFlagError, match="has no value"):
        compose_argv("encyclopedia", {"-i": None})  # type: ignore[dict-item]
    with pytest.raises(UnknownFlagError, match="is not a flag"):
        compose_argv("encyclopedia", {"i": "run.dia"})


def test_the_legacy_wrappers_emit_only_what_the_runner_would_accept(tmp_path: Path) -> None:
    """The hand-written search wrapper and this runner agree on the jar."""
    from constellation.massspec.search.encyclopedia import build_library_search_args

    legacy = build_library_search_args(
        input_file=tmp_path / "run.dia",
        library=tmp_path / "lib.dlib",
        fasta=tmp_path / "db.fasta",
        report_output=tmp_path / "run.encyclopedia.txt",
        precursor_tolerance=0.35,
        precursor_tolerance_unit="Da",
        threads=4,
        percolator_version="v3-05",
    )
    flags = list(zip(legacy[0::2], legacy[1::2], strict=True))
    assert compose_argv("encyclopedia", flags) == legacy


def test_run_program_validates_before_touching_the_jar(monkeypatch, tmp_path: Path) -> None:
    from constellation.massspec.search.encyclopedia import runner
    from constellation.thirdparty import registry

    def _no_jar(*_args, **_kwargs):
        raise AssertionError("the jar was looked up for an invalid command line")

    monkeypatch.setattr(runner, "find", _no_jar)
    with pytest.raises(UnknownFlagError):
        run_program("encyclopedia", {"-nope": 1}, log_dir=tmp_path)

    spec = next(s for s in registry.registered() if s.name == "encyclopedia")
    old = registry.ToolHandle(spec, tmp_path / "encyclopedia-6.5.15.jar", "env", "6.5.15")
    monkeypatch.setattr(runner, "find", lambda name: old)
    monkeypatch.setattr(runner, "run_jar", _no_jar)
    with pytest.raises(UnsupportedVersionError, match="6.5.15"):
        run_program("encyclopedia", {"-i": "run.dia"}, log_dir=tmp_path)


# ── search mode ────────────────────────────────────────────────────────


def test_the_default_mode_follows_the_data() -> None:
    dia = resolve_search_mode(None, acquisition_type="DIA")
    assert (dia.mode, dia.program, dia.flags, dia.library_required) == (
        PEPTIDE_CENTRIC, "encyclopedia", {}, True,
    )  # fmt: skip
    dda = resolve_search_mode(None, acquisition_type="DDA")
    assert (dda.mode, dda.program, dda.flags, dda.library_required) == (
        SPECTRUM_CENTRIC, "scribetwo", {"-searchType": "auto"}, False,
    )  # fmt: skip
    assert resolve_search_mode(None, acquisition_type=None).mode == PEPTIDE_CENTRIC


@pytest.mark.parametrize(
    ("search_type", "token"),
    [("auto", "auto"), ("dia", "dia"), ("dda", "dda"), ("dda-chimeric", "dda_chimeric")],
)
def test_spectrum_centric_search_types_map_to_the_jars_tokens(search_type: str, token: str) -> None:
    kind = "DIA" if search_type == "dia" else "DDA" if search_type.startswith("dda") else None
    resolved = resolve_search_mode(SPECTRUM_CENTRIC, acquisition_type=kind, search_type=search_type)
    assert resolved.flags == {"-searchType": token}
    # ...and the token is one the pinned jar documents.
    assert token in cli_spec._program("scribetwo")["flags"]["-searchType"]["description"]


def test_combinations_that_make_no_sense_are_refused() -> None:
    with pytest.raises(SearchModeError, match="no peptide-centric search of DDA"):
        resolve_search_mode(PEPTIDE_CENTRIC, acquisition_type="DDA")
    with pytest.raises(SearchModeError, match="belongs to the spectrum-centric engine"):
        resolve_search_mode(PEPTIDE_CENTRIC, acquisition_type="DIA", search_type="dia")
    with pytest.raises(SearchModeError, match="dia was given for a DDA"):
        resolve_search_mode(SPECTRUM_CENTRIC, acquisition_type="DDA", search_type="dia")
    with pytest.raises(SearchModeError, match="dda was given for a DIA"):
        resolve_search_mode(SPECTRUM_CENTRIC, acquisition_type="DIA", search_type="dda")
    with pytest.raises(SearchModeError, match="search mode must be one of"):
        resolve_search_mode("dia", acquisition_type="DIA")
    with pytest.raises(SearchModeError, match="search type must be one of"):
        resolve_search_mode(SPECTRUM_CENTRIC, acquisition_type=None, search_type="dda_chimeric")


# ── parameters ─────────────────────────────────────────────────────────


def test_parameters_become_flags_and_a_record() -> None:
    params = SearchParameters(
        precursor_tolerance=Tolerance(10.0),
        fragment_tolerance=Tolerance(0.35, "Da"),
        threads=8,
        percolator_version="v3-05",
        extra={"-fixed": ""},
    )
    flags = params.flags()
    assert flags == {
        "-ptol": 10.0,
        "-ptolunits": "PPM",
        "-ftol": 0.35,
        "-ftolunits": "AMU",  # the jar's spelling of Da
        "-percolatorVersion": "v3-05",
        "-numberOfThreadsUsed": 8,
        "-fixed": "",
    }
    compose_argv("encyclopedia", flags)  # every one is documented
    assert params.as_given() == {
        "precursor_tolerance": {"value": 10.0, "unit": "ppm"},
        "fragment_tolerance": {"value": 0.35, "unit": "Da"},
        "percolator_version": "v3-05",
        "threads": 8,
        "extra": {"-fixed": ""},
    }
    assert SearchParameters().flags() == {} and SearchParameters().as_given() == {}


def test_parameters_refuse_what_cannot_be_meant() -> None:
    with pytest.raises(ValueError, match="ppm or Da"):
        Tolerance(10.0, "mmu")
    with pytest.raises(ValueError, match="must be positive"):
        Tolerance(0.0)
    with pytest.raises(ValueError, match="set both"):
        SearchParameters(threads=4, extra={"-numberOfThreadsUsed": 8}).flags()


# ── outputs ────────────────────────────────────────────────────────────


def _fake_outputs(directory: Path, input_name: str, leave_out: tuple[str, ...] = ()) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, file_name in expected_names("encyclopedia", input_name).items():
        if name not in leave_out:
            (directory / file_name).write_bytes(b"")


def test_outputs_are_named_for_the_input_file() -> None:
    """The jar's own convention: everything beside the input, named for it,
    extension included."""
    names = expected_names("encyclopedia", "run_01.dia")
    assert names["<input>.elib"] == "run_01.dia.elib"
    assert names["<input>.encyclopedia.txt"] == "run_01.dia.encyclopedia.txt"
    assert names["<input>.features.txt"] == "run_01.dia.features.txt"
    assert all(name.startswith("run_01.dia.") for name in names.values())


def test_a_complete_run_is_located_by_the_captured_inventory(tmp_path: Path) -> None:
    _fake_outputs(tmp_path / "jar", "run_01.dia")
    (tmp_path / "jar" / "unrelated.txt").write_text("")
    found = locate_outputs(tmp_path / "jar", "run_01.dia")
    assert found.result_library == tmp_path / "jar" / "run_01.dia.elib"
    assert len(found.files) == len(output_inventory("encyclopedia")["files"]) == 24
    assert found.missing == ()
    assert found.path("<input>.features.txt").name == "run_01.dia.features.txt"
    assert found.relative(tmp_path)["<input>.elib"] == "jar/run_01.dia.elib"


def test_missing_sidecars_are_reported_and_a_missing_result_is_an_error(tmp_path: Path) -> None:
    _fake_outputs(
        tmp_path / "a", "run.dia", leave_out=("<input>.encyclopedia.txt.final.rt_fit.pdf",)
    )
    assert locate_outputs(tmp_path / "a", "run.dia").missing == (
        "<input>.encyclopedia.txt.final.rt_fit.pdf",
    )
    _fake_outputs(tmp_path / "b", "run.dia", leave_out=("<input>.elib",))
    with pytest.raises(MissingOutputError, match="wrote no result library: expected run.dia.elib"):
        locate_outputs(tmp_path / "b", "run.dia")
    with pytest.raises(KeyError, match="no output inventory was captured"):
        output_inventory("scribetwo")


# ── `massspec encyclopedia flags` / `run` ──────────────────────────────


def test_the_flags_verb_lists_programs_and_one_programs_flags(capsys) -> None:
    from constellation.cli.__main__ import main

    assert main(["massspec", "encyclopedia", "flags"]) == 0
    listing = capsys.readouterr().out
    for name in cli_spec.load_cli_spec()["programs"]:
        assert f"  {name} " in listing
    assert "-scribetwo" in listing and "(default)" in listing

    assert main(["massspec", "encyclopedia", "flags", "encyclopedia"]) == 0
    out = capsys.readouterr().out
    assert "selected with: (none: the default program)" in out
    for flag in ("-i", "-ptolunits", "-percolatorVersion"):
        assert f"  {flag} " in out
    # Required flags come first.
    assert out.index("  -i ") < out.index("  -acquisition ")

    assert main(["massspec", "encyclopedia", "flags", "nonesuch"]) == 2
    assert "no program 'nonesuch'" in capsys.readouterr().err


def test_the_run_verb_checks_the_command_line_before_java(monkeypatch, tmp_path: Path, capsys) -> None:
    from constellation.cli.__main__ import main
    from constellation.massspec.search.encyclopedia import runner

    calls = []
    monkeypatch.setattr(runner, "run_program", lambda *a, **k: calls.append((a, k)))
    run = ["massspec", "encyclopedia", "run"]
    out_dir = ["--output-dir", str(tmp_path / "work")]

    # --dry-run prints the jar arguments and starts nothing.
    assert main([*run, "scribetwo", *out_dir, "--dry-run", "--", "-i", "x.dia", "-f", "x.fasta"]) == 0
    assert capsys.readouterr().out.split() == [
        "-scribetwo", "-i", "x.dia", "-f", "x.fasta",
    ]  # fmt: skip
    assert not (tmp_path / "work").exists()

    assert main([*run, "encyclopedia", *out_dir, "--", "-i", "x.dia", "-ptoll", "10"]) == 2
    assert "-ptoll" in capsys.readouterr().err
    assert main([*run, "nonesuch", *out_dir, "--", "-i", "x.dia"]) == 2
    assert "nonesuch" in capsys.readouterr().err
    assert main([*run, "encyclopedia", *out_dir, "--", "-i"]) == 2
    assert "`-flag value` pairs" in capsys.readouterr().err
    assert calls == []

    assert main([*run, "encyclopedia", *out_dir, "--no-progress", "--", "-i", "x.dia", "-ptol", "10"]) == 0
    ((args, kwargs),) = calls
    assert args == ("encyclopedia", [("-i", "x.dia"), ("-ptol", "10")])
    assert kwargs["cwd"] == kwargs["log_dir"].parent == (tmp_path / "work").resolve()
    assert kwargs["stream_to_stderr"] is False
