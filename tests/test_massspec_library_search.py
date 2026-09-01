"""Tests for ``constellation massspec search``.

Mostly Tier A: pure-Python arg-builder coverage, no JVM required. Full
end-to-end testing against a real .dia + .dlib + .fasta is left to the
user since we don't carry sample spectra fixtures.

One Tier B test (gated on a registry-resolved encyclopedia install)
pins the tolerance-unit translation table against the real jar. It
needs no fixture files: EncyclopeDIA parses ``-ftolunits`` and friends
in ``SearchParameterParser.parseParameters`` *before* it opens any
input, so a run with nonexistent paths still exercises unit validation
and then fails on the library read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia import (
    build_library_search_args,
    find_search_elib,
)


# ── arg builder ────────────────────────────────────────────────────────


def test_build_args_minimal(tmp_path: Path) -> None:
    inp = tmp_path / "sample.dia"
    lib = tmp_path / "predicted.dlib"
    args = build_library_search_args(input_file=inp, library=lib)
    # Required flags present, in -i / -l order
    assert args[:4] == ["-i", str(inp), "-l", str(lib)]
    # No optional flags emitted when caller doesn't set them
    assert "-f" not in args
    assert "-o" not in args
    assert "-ftol" not in args
    assert "-ptol" not in args
    assert "-lftol" not in args
    assert "-percolatorVersion" not in args


def test_build_args_with_fasta_and_report(tmp_path: Path) -> None:
    args = build_library_search_args(
        input_file=tmp_path / "x.dia",
        library=tmp_path / "y.dlib",
        fasta=tmp_path / "z.fasta",
        report_output=tmp_path / "report.txt",
    )
    assert "-f" in args
    assert str(tmp_path / "z.fasta") in args
    assert "-o" in args
    assert str(tmp_path / "report.txt") in args


def test_build_args_with_tolerances(tmp_path: Path) -> None:
    args = build_library_search_args(
        input_file=tmp_path / "x.dia",
        library=tmp_path / "y.dlib",
        fragment_tolerance=20.0,
        precursor_tolerance=10.0,
    )
    # ftol gets value + unit pair; ppm is the default unit
    ftol_idx = args.index("-ftol")
    assert args[ftol_idx + 1] == "20.0"
    assert args[args.index("-ftolunits") + 1] == "PPM"
    ptol_idx = args.index("-ptol")
    assert args[ptol_idx + 1] == "10.0"
    assert args[args.index("-ptolunits") + 1] == "PPM"
    # lftol untouched when not asked for
    assert "-lftol" not in args


def test_build_args_dalton_tolerances_translate_to_amu(tmp_path: Path) -> None:
    """Ion-trap spelling: the jar rejects the literal ``Da`` and wants ``AMU``.

    Verified against encyclopedia-6.5.15: ``-ftolunits Da`` raises
    ``EncyclopediaException: Error parsing fragment mass error unit type
    from [Da]``, so this translation is load-bearing.
    """
    args = build_library_search_args(
        input_file=tmp_path / "x.dia",
        library=tmp_path / "y.dlib",
        precursor_tolerance=0.3,
        precursor_tolerance_unit="Da",
        fragment_tolerance=0.8,
        fragment_tolerance_unit="Da",
    )
    assert args[args.index("-ptol") + 1] == "0.3"
    assert args[args.index("-ptolunits") + 1] == "AMU"
    assert args[args.index("-ftol") + 1] == "0.8"
    assert args[args.index("-ftolunits") + 1] == "AMU"


def test_build_args_library_fragment_tolerance_is_independent(tmp_path: Path) -> None:
    """``-lftol`` is not coupled to ``-ftol``.

    A predicted library's m/z are exact theoretical values, so it stays
    tight in ppm even when the acquired data is ion-trap.
    """
    args = build_library_search_args(
        input_file=tmp_path / "x.dia",
        library=tmp_path / "y.dlib",
        fragment_tolerance=0.8,
        fragment_tolerance_unit="Da",
        library_fragment_tolerance=10.0,
    )
    assert args[args.index("-ftolunits") + 1] == "AMU"
    assert args[args.index("-lftol") + 1] == "10.0"
    assert args[args.index("-lftolunits") + 1] == "PPM"


def test_build_args_unit_is_case_insensitive(tmp_path: Path) -> None:
    """The jar matches its enum with ``equalsIgnoreCase``; so do we."""
    args = build_library_search_args(
        input_file=tmp_path / "x.dia",
        library=tmp_path / "y.dlib",
        fragment_tolerance=0.8,
        fragment_tolerance_unit="da",
    )
    assert args[args.index("-ftolunits") + 1] == "AMU"


def test_build_args_rejects_unknown_unit(tmp_path: Path) -> None:
    """Catch a bad unit here rather than in a JVM stack trace."""
    with pytest.raises(ValueError, match="tolerance unit must be one of"):
        build_library_search_args(
            input_file=tmp_path / "x.dia",
            library=tmp_path / "y.dlib",
            fragment_tolerance=0.8,
            fragment_tolerance_unit="daltons",
        )


def test_build_args_with_full_scoring_config(tmp_path: Path) -> None:
    args = build_library_search_args(
        input_file=tmp_path / "x.dia",
        library=tmp_path / "y.dlib",
        acquisition="DIA",
        enzyme="LysC",
        fragmentation="HCD",
        percolator_version="v3-05",
        percolator_threshold=0.01,
        percolator_protein_threshold=0.05,
        threads=16,
    )
    flag_value = dict(zip(args[::2], args[1::2]))
    assert flag_value["-acquisition"] == "DIA"
    assert flag_value["-enzyme"] == "LysC"
    assert flag_value["-frag"] == "HCD"
    assert flag_value["-percolatorVersion"] == "v3-05"
    assert flag_value["-percolatorThreshold"] == "0.01"
    assert flag_value["-percolatorProteinThreshold"] == "0.05"
    assert flag_value["-numberOfThreadsUsed"] == "16"


def test_build_args_extra_args_appended_last(tmp_path: Path) -> None:
    args = build_library_search_args(
        input_file=tmp_path / "x.dia",
        library=tmp_path / "y.dlib",
        extra_args=["-experimentalFlag", "1.0"],
    )
    assert args[-2:] == ["-experimentalFlag", "1.0"]


# ── find_search_elib ──────────────────────────────────────────────────


def test_find_search_elib_canonical(tmp_path: Path) -> None:
    """Older EncyclopeDIA versions write <input>.elib next to the input."""
    inp = tmp_path / "sample.dia"
    elib = tmp_path / "sample.dia.elib"
    inp.write_bytes(b"")
    elib.write_bytes(b"x")
    found = find_search_elib(inp)
    assert found == elib


def test_find_search_elib_stem_fallback(tmp_path: Path) -> None:
    """Some jar versions write <input_stem>.elib next to the input
    instead of <input>.elib. The wrapper tolerates both."""
    inp = tmp_path / "sample.dia"
    elib = tmp_path / "sample.elib"
    inp.write_bytes(b"")
    elib.write_bytes(b"x")
    found = find_search_elib(inp)
    assert found == elib


def test_find_search_elib_cwd_stem(tmp_path: Path) -> None:
    """EncyclopeDIA 6.5.15 writes <input_stem>.elib to the runner's cwd
    (verified against a real GPF search). Must be checked BEFORE the
    next-to-input candidates so cwd wins when both exist."""
    inp_dir = tmp_path / "input_dir"
    inp_dir.mkdir()
    cwd_dir = tmp_path / "search_run"
    cwd_dir.mkdir()
    inp = inp_dir / "GPF_combined.dia"
    inp.write_bytes(b"")
    cwd_elib = cwd_dir / "GPF_combined.elib"
    cwd_elib.write_bytes(b"x")
    found = find_search_elib(inp, cwd=cwd_dir)
    assert found == cwd_elib


def test_find_search_elib_cwd_precedence_over_input_dir(
    tmp_path: Path,
) -> None:
    """If both cwd-stem and input-dir-name exist, cwd wins — the cwd
    file is the one the current EncyclopeDIA version is writing right
    now, so it's the freshest output."""
    inp_dir = tmp_path / "input_dir"
    inp_dir.mkdir()
    cwd_dir = tmp_path / "search_run"
    cwd_dir.mkdir()
    inp = inp_dir / "sample.dia"
    inp.write_bytes(b"")
    # Both candidates exist — stale and fresh
    stale = inp_dir / "sample.dia.elib"
    stale.write_bytes(b"stale")
    fresh = cwd_dir / "sample.elib"
    fresh.write_bytes(b"fresh")
    found = find_search_elib(inp, cwd=cwd_dir)
    assert found == fresh
    assert found.read_bytes() == b"fresh"


def test_find_search_elib_missing(tmp_path: Path) -> None:
    inp = tmp_path / "sample.dia"
    inp.write_bytes(b"")
    assert find_search_elib(inp) is None
    assert find_search_elib(inp, cwd=tmp_path / "doesnt_exist") is None


# ── Tier B: tolerance units against the real jar ───────────────────────


def _have_encyclopedia_install() -> bool:
    from constellation.massspec.search.encyclopedia import is_supported_version
    from constellation.thirdparty.registry import try_find

    handle = try_find("encyclopedia")
    return (
        handle is not None
        and handle.path.is_file()
        and is_supported_version(handle.version)
    )


_HAVE_ENC = _have_encyclopedia_install()

_UNIT_PARSE_ERROR = "Error parsing"
_PAST_UNIT_PARSING = "Can't read library file"


def _run_jar_for_units(tmp_path: Path, extra: list[str]) -> str:
    """Invoke the jar with nonexistent inputs; return its combined output.

    Unit parsing precedes file I/O, so reaching "Can't read library file"
    proves every tolerance token was accepted.
    """
    import subprocess

    from constellation.thirdparty.jvm import _resolve_java
    from constellation.thirdparty.registry import find

    handle = find("encyclopedia")
    java_path, _source, _version = _resolve_java(handle)
    argv = [
        str(java_path),
        "-Djava.awt.headless=true",
        "-jar",
        str(handle.path),
        "-i",
        str(tmp_path / "nope.mzML"),
        "-l",
        str(tmp_path / "nope.dlib"),
        "-f",
        str(tmp_path / "nope.fasta"),
        *extra,
    ]
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=300, cwd=tmp_path
    )
    return proc.stdout + proc.stderr


@pytest.mark.skipif(
    not _HAVE_ENC,
    reason="needs $CONSTELLATION_ENCYCLOPEDIA_HOME with the encyclopedia jar",
)
def test_jar_accepts_emitted_tolerance_tokens(tmp_path: Path) -> None:
    """Every token ``build_library_search_args`` emits must parse.

    Guards the ``Da`` -> ``AMU`` translation table against jar upgrades.
    """
    args = build_library_search_args(
        input_file=tmp_path / "nope.mzML",
        library=tmp_path / "nope.dlib",
        precursor_tolerance=0.3,
        precursor_tolerance_unit="Da",
        fragment_tolerance=0.8,
        fragment_tolerance_unit="Da",
        library_fragment_tolerance=10.0,
        library_fragment_tolerance_unit="ppm",
    )
    tolerance_args = args[4:]  # drop the -i/-l pair; the helper supplies its own
    out = _run_jar_for_units(tmp_path, tolerance_args)
    assert _UNIT_PARSE_ERROR not in out, out
    assert _PAST_UNIT_PARSING in out, out


@pytest.mark.skipif(
    not _HAVE_ENC,
    reason="needs $CONSTELLATION_ENCYCLOPEDIA_HOME with the encyclopedia jar",
)
def test_jar_rejects_literal_da(tmp_path: Path) -> None:
    """The negative half: without translation the jar hard-errors.

    If this ever starts passing, the jar has learned to spell ``Da`` and
    the translation table could be simplified.
    """
    out = _run_jar_for_units(tmp_path, ["-ftol", "0.8", "-ftolunits", "Da"])
    assert "Error parsing fragment mass error unit type" in out, out
