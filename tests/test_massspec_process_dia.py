"""Tests for ``constellation massspec process-dia``.

Tier A only — the arg builder is a thin function that joins paths with
``:`` and adds the ``-convert -processDIA -i ... -o ...`` shape. End-
to-end testing against a real ``.raw`` / ``.mzML`` fixture is gated on
``$CONSTELLATION_ENCYCLOPEDIA_SAMPLES`` (not in this file — the lab
provides the fixture out-of-tree) since we don't carry sample spectra
in the repo.

The jar invocation path is already exercised end-to-end by
``test_massspec_predict_library.py``; what process-dia adds on top is
just the arg-list shape, which is what these tests verify.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia import (
    build_process_dia_args,
)


# ── arg builder ────────────────────────────────────────────────────────


def test_build_args_single_input(tmp_path: Path) -> None:
    inp = tmp_path / "a.mzML"
    args = build_process_dia_args(inputs=[inp])
    assert args == [
        "-convert",
        "-processDIA",
        "-i",
        str(inp),
    ]


def test_build_args_single_input_never_emits_output_flag(tmp_path: Path) -> None:
    """``-o`` must be suppressed when there is exactly one input.

    EncyclopeDIA 6.5.15 rejects the combination outright — "When using
    one input, do not specify an output file!" — and exits 1 before
    doing any work. This test previously asserted the opposite, which
    is how a Slurm sweep ended up losing six array tasks in the first
    four minutes.

    ``output_dia`` is still accepted (callers want a uniform interface
    regardless of input count); the CLI handler relocates the
    jar-produced ``<input_stem>.dia`` afterwards.
    """
    inp = tmp_path / "a.mzML"
    out = tmp_path / "out.dia"
    args = build_process_dia_args(inputs=[inp], output_dia=out)
    assert args == [
        "-convert",
        "-processDIA",
        "-i",
        str(inp),
    ]
    assert "-o" not in args


def test_single_input_dia_path_finds_both_conventions(tmp_path: Path) -> None:
    """Locate the .dia the jar drops beside a single input."""
    from constellation.massspec.search.encyclopedia import single_input_dia_path

    inp = tmp_path / "a.raw"
    inp.write_bytes(b"")
    assert single_input_dia_path(inp) is None

    stem_form = tmp_path / "a.dia"
    stem_form.write_bytes(b"x")
    assert single_input_dia_path(inp) == stem_form
    stem_form.unlink()

    name_form = tmp_path / "a.raw.dia"
    name_form.write_bytes(b"x")
    assert single_input_dia_path(inp) == name_form


def test_build_args_multi_input_colon_joined(tmp_path: Path) -> None:
    """Six GPF fractions are colon-delimited per EncyclopeDIA's merge syntax."""
    inputs = [tmp_path / f"gpf_{i}.raw" for i in range(6)]
    out = tmp_path / "merged.dia"
    args = build_process_dia_args(inputs=inputs, output_dia=out)
    assert args[:3] == ["-convert", "-processDIA", "-i"]
    joined = args[3]
    assert ":" in joined
    parts = joined.split(":")
    assert len(parts) == 6
    assert all(parts[i] == str(inputs[i]) for i in range(6))
    assert args[-2:] == ["-o", str(out)]


def test_build_args_mixed_formats(tmp_path: Path) -> None:
    """EncyclopeDIA's processDIA accepts heterogeneous input formats
    in a single merge — useful when one fraction had to be re-converted."""
    inputs = [
        tmp_path / "a.mzML",
        tmp_path / "b.raw",
        tmp_path / "c.d",
        tmp_path / "d.DIA",
    ]
    out = tmp_path / "merged.dia"
    args = build_process_dia_args(inputs=inputs, output_dia=out)
    joined = args[args.index("-i") + 1]
    assert joined == ":".join(str(p) for p in inputs)


def test_build_args_extra_args_appended(tmp_path: Path) -> None:
    inp = tmp_path / "a.mzML"
    out = tmp_path / "out.dia"
    args = build_process_dia_args(
        inputs=[inp],
        output_dia=out,
        extra_args=["-someFutureFlag", "value"],
    )
    assert args[-2:] == ["-someFutureFlag", "value"]


def test_build_args_empty_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="at least one input"):
        build_process_dia_args(inputs=[])
