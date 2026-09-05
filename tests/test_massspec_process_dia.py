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


def test_single_input_dia_path_prefers_cwd(tmp_path: Path) -> None:
    """cwd wins over next-to-input, mirroring find_search_elib.

    6.5.15 writes the single-input .dia into the process's working
    directory, not beside the input. The runner pins cwd=output_dir, so
    that candidate must be checked first — otherwise a stale .dia
    sitting next to the raw file would be picked up in preference to the
    one this run just produced.
    """
    from constellation.massspec.search.encyclopedia import single_input_dia_path

    raw_dir = tmp_path / "MS_Data"
    raw_dir.mkdir()
    inp = raw_dir / "sample.raw"
    inp.write_bytes(b"")

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Nothing anywhere yet.
    assert single_input_dia_path(inp, cwd=run_dir) is None

    # Only next-to-input: still found (older convention).
    beside = raw_dir / "sample.dia"
    beside.write_bytes(b"x")
    assert single_input_dia_path(inp, cwd=run_dir) == beside

    # cwd copy appears -> it wins.
    in_cwd = run_dir / "sample.dia"
    in_cwd.write_bytes(b"x")
    assert single_input_dia_path(inp, cwd=run_dir) == in_cwd


# ── runner: single-input relocation ────────────────────────────────────
#
# The relocation lives in run_process_dia rather than in each caller.
# It used to live only in the CLI handler, so the transcriptome→proteome
# orchestrator — which calls the runner directly — marked its Stage 6
# complete while the cache still sat under the jar-chosen name, and
# Stage 7 then searched a path that did not exist.


class _FakeJvmResult:
    """Stand-in for JvmResult; run_process_dia returns it untouched."""

    returncode = 0
    elapsed_seconds = 0.01


def _fake_jar_writing(stem_source: str | None):
    """Build a run_jar double that writes ``<stem>.dia`` into its cwd.

    Mirrors 6.5.15 single-input behaviour: the jar ignores ``-o`` and
    names the cache itself, relative to the working directory.

    ``stem_source=None`` models the jar exiting 0 having produced no new
    cache — what happens when it is handed an existing ``.DIA`` and
    reuses it. Every kwarg is captured, not just args/cwd, so tests can
    assert on what the JVM was actually handed.
    """
    captured: dict[str, object] = {}

    def _fake_run_jar(tool, *, args, cwd, **kw):
        captured["tool"] = tool
        captured["args"] = list(args)
        captured["cwd"] = Path(cwd)
        captured.update(kw)
        Path(cwd).mkdir(parents=True, exist_ok=True)
        if stem_source is not None:
            produced = Path(cwd) / f"{stem_source}.dia"
            produced.write_bytes(b"fresh")
        return _FakeJvmResult()

    return _fake_run_jar, captured


def test_run_process_dia_relocates_single_input_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single input: the runner moves the jar-named cache onto output_dia.

    Direct callers (the orchestrator) get the same ``--output-dia``
    contract the CLI does, without special-casing input count.
    """
    from constellation.massspec.search.encyclopedia import process_dia as pd

    inp = tmp_path / "MS_Data" / "sample.raw"
    inp.parent.mkdir(parents=True)
    inp.write_bytes(b"")
    out_dir = tmp_path / "06_process_dia"
    out_dia = out_dir / "run_combined_GPF.dia"

    fake, _ = _fake_jar_writing("sample")
    monkeypatch.setattr(pd, "run_jar", fake)

    pd.run_process_dia(
        inputs=[inp], output_dia=out_dia, output_dir=out_dir,
        stream_to_stderr=False,
    )

    assert out_dia.is_file()
    assert out_dia.read_bytes() == b"fresh"
    assert not (out_dir / "sample.dia").exists()


def test_run_process_dia_replaces_stale_single_input_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale destination must not shadow what this JVM just produced.

    Reruns of an incomplete output dir (an ``--output-dia`` present but
    no ``_SUCCESS`` — exactly what the manifest-kwarg TypeError left
    behind) previously skipped the move on ``not output_dia.is_file()``,
    then accepted the stale file and wrote a manifest over it, binding
    old spectra to the new run.
    """
    from constellation.massspec.search.encyclopedia import process_dia as pd

    inp = tmp_path / "MS_Data" / "sample.raw"
    inp.parent.mkdir(parents=True)
    inp.write_bytes(b"")
    out_dir = tmp_path / "06_process_dia"
    out_dir.mkdir()
    out_dia = out_dir / "run_combined_GPF.dia"
    out_dia.write_bytes(b"stale")

    fake, _ = _fake_jar_writing("sample")
    monkeypatch.setattr(pd, "run_jar", fake)

    pd.run_process_dia(
        inputs=[inp], output_dia=out_dia, output_dir=out_dir,
        stream_to_stderr=False,
    )

    assert out_dia.read_bytes() == b"fresh"


def test_run_process_dia_no_move_when_already_at_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """output_dia may itself BE the jar's chosen path — must not self-move."""
    from constellation.massspec.search.encyclopedia import process_dia as pd

    inp = tmp_path / "sample.raw"
    inp.write_bytes(b"")
    out_dir = tmp_path / "run"
    out_dia = out_dir / "sample.dia"  # exactly what the jar will write

    fake, _ = _fake_jar_writing("sample")
    monkeypatch.setattr(pd, "run_jar", fake)

    pd.run_process_dia(
        inputs=[inp], output_dia=out_dia, output_dir=out_dir,
        stream_to_stderr=False,
    )

    assert out_dia.read_bytes() == b"fresh"


def test_run_process_dia_resolves_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative paths resolve against the CALLER's cwd, not output_dir.

    run_jar runs with ``cwd=output_dir``, so an unresolved relative
    ``-i data/sample.raw`` would be looked up beneath output_dir and a
    relative output_dia would land somewhere nested. Both in-tree
    callers resolve first; this pins it for the public wrapper.
    """
    from constellation.massspec.search.encyclopedia import process_dia as pd

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample.raw").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    fake, captured = _fake_jar_writing("sample")
    monkeypatch.setattr(pd, "run_jar", fake)

    pd.run_process_dia(
        inputs=[Path("data/sample.raw")],
        output_dia=Path("out/combined.dia"),
        output_dir=Path("rundir"),
        stream_to_stderr=False,
    )

    i_value = captured["args"][captured["args"].index("-i") + 1]
    assert Path(i_value).is_absolute()
    assert Path(i_value) == (tmp_path / "data" / "sample.raw").resolve()
    assert captured["cwd"] == (tmp_path / "rundir").resolve()
    # And the cache still lands where the caller asked, not under rundir.
    assert (tmp_path / "out" / "combined.dia").is_file()


# ── runner: the input is never the product ─────────────────────────────


def test_run_process_dia_never_moves_a_dia_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.DIA`` input must survive the run byte-for-byte.

    ``.DIA`` is a documented input format — the jar reuses the cache and
    emits nothing new. But ``sample.dia``.with_suffix(".dia") IS
    ``sample.dia``, so the lookup returned the input itself and the
    relocation carried the caller's source file off to output_dia. The
    CLI then raised FileNotFoundError hashing an input that no longer
    existed, and the user's cache was gone.
    """
    from constellation.massspec.search.encyclopedia import process_dia as pd

    inp = tmp_path / "MS_Data" / "sample.dia"
    inp.parent.mkdir(parents=True)
    inp.write_bytes(b"original-cache")
    out_dir = tmp_path / "run"
    out_dia = out_dir / "combined.dia"

    fake, _ = _fake_jar_writing(None)  # exits 0, produces nothing
    monkeypatch.setattr(pd, "run_jar", fake)

    pd.run_process_dia(
        inputs=[inp], output_dia=out_dia, output_dir=out_dir,
        stream_to_stderr=False,
    )

    assert inp.is_file(), "the .DIA input was moved away"
    assert inp.read_bytes() == b"original-cache"
    # Reuse still honours --output-dia — by copy, not by consuming it.
    assert out_dia.read_bytes() == b"original-cache"


def test_run_process_dia_leaves_non_dia_input_alone_when_nothing_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Jar exits 0 but emits no cache: report nothing, touch nothing.

    The caller owns the error message (it names every path searched);
    the runner must not invent an output or disturb the input.
    """
    from constellation.massspec.search.encyclopedia import process_dia as pd

    inp = tmp_path / "sample.raw"
    inp.write_bytes(b"raw-bytes")
    out_dir = tmp_path / "run"
    out_dia = out_dir / "combined.dia"

    fake, _ = _fake_jar_writing(None)
    monkeypatch.setattr(pd, "run_jar", fake)

    pd.run_process_dia(
        inputs=[inp], output_dia=out_dia, output_dir=out_dir,
        stream_to_stderr=False,
    )

    assert not out_dia.exists()
    assert inp.read_bytes() == b"raw-bytes"


def test_single_input_dia_path_excludes_the_inputs(tmp_path: Path) -> None:
    """``exclude`` keeps a .DIA input from being reported as the product."""
    from constellation.massspec.search.encyclopedia import single_input_dia_path

    inp = tmp_path / "sample.dia"
    inp.write_bytes(b"cache")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Without the exclusion the input itself matches.
    assert single_input_dia_path(inp, cwd=run_dir) == inp
    assert single_input_dia_path(inp, cwd=run_dir, exclude=[inp]) is None

    # A genuinely produced cache still wins.
    produced = run_dir / "sample.dia"
    produced.write_bytes(b"fresh")
    assert single_input_dia_path(inp, cwd=run_dir, exclude=[inp]) == produced


def test_run_process_dia_resolves_jvm_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative --jvm-tmpdir must not be reinterpreted under output_dir.

    Java resolves ``-Djava.io.tmpdir`` against its own cwd, which run_jar
    sets to output_dir — so ``./scratch`` would silently become
    ``<output_dir>/scratch``, spilling onto whatever filesystem holds the
    results rather than the fast scratch the caller picked.
    """
    from constellation.massspec.search.encyclopedia import process_dia as pd

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample.raw").write_bytes(b"")
    (tmp_path / "scratch").mkdir()
    monkeypatch.chdir(tmp_path)

    fake, captured = _fake_jar_writing("sample")
    monkeypatch.setattr(pd, "run_jar", fake)

    pd.run_process_dia(
        inputs=[Path("data/sample.raw")],
        output_dia=Path("out/combined.dia"),
        output_dir=Path("rundir"),
        jvm_tmpdir=Path("scratch"),
        stream_to_stderr=False,
    )

    assert captured["jvm_tmpdir"] == (tmp_path / "scratch").resolve()
