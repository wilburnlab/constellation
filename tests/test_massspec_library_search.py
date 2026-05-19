"""Tier A tests for ``constellation massspec search``.

Pure-Python arg-builder coverage — no JVM required. End-to-end testing
against a real .dia + .dlib + .fasta is left to the user since we
don't carry sample spectra fixtures.
"""

from __future__ import annotations

from pathlib import Path

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
        fragment_tolerance_ppm=20.0,
        precursor_tolerance_ppm=10.0,
    )
    # ftol gets value + unit pair
    ftol_idx = args.index("-ftol")
    assert args[ftol_idx + 1] == "20.0"
    assert args[args.index("-ftolunits") + 1] == "ppm"
    ptol_idx = args.index("-ptol")
    assert args[ptol_idx + 1] == "10.0"
    assert args[args.index("-ptolunits") + 1] == "ppm"


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
