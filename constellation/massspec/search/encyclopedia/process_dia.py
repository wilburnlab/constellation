"""Wrapper for ``-convert -processDIA`` — mzML / .raw / .d / .DIA preprocessing
and gas-phase-fraction merging.

The jar's CLI takes colon-delimited input paths (``-i a.mzML:b.mzML:c.mzML``)
when merging multiple files into a single ``.DIA`` cache; this wrapper
takes a Python ``list[Path]`` and joins for the user. The merge step
preserves per-MS2 isolation-window metadata so overlapping / staggered
DIA schemes flow through to the downstream search.

Unlike :mod:`predict_library`, no auto-ingest happens after the jar
exits: ``.DIA`` is a spectra-cache format meant for EncyclopeDIA's own
consumption (search / library export), not a Constellation-native
analysis artifact. The handler writes a ``manifest.json`` for
reproducibility and touches ``_SUCCESS``; the merged ``.DIA`` itself
sits at ``--output-dia``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from constellation.thirdparty.jvm import JvmResult, run_jar


def build_process_dia_args(
    *,
    inputs: Sequence[Path],
    output_dia: Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Translate typed kwargs to the EncyclopeDIA CLI argv (no JVM call).

    Multi-input mode (``len(inputs) > 1``) requires ``output_dia``;
    EncyclopeDIA writes the consolidated cache there. Single-input mode
    can omit ``output_dia`` — the jar writes a ``.DIA`` cache next to
    the input file automatically.

    Pure function — exists so the Tier A test can exercise the flag
    layout without spawning Java.
    """
    if not inputs:
        raise ValueError("process-dia requires at least one input file")
    joined = ":".join(str(p) for p in inputs)
    args: list[str] = [
        "-convert",
        "-processDIA",
        "-i",
        joined,
    ]
    if output_dia is not None:
        args.extend(["-o", str(output_dia)])
    args.extend(str(a) for a in extra_args)
    return args


def run_process_dia(
    *,
    inputs: Sequence[Path],
    output_dia: Path | None,
    output_dir: Path,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_args: Sequence[str] = (),
    extra_jvm_args: Sequence[str] = (),
    stream_to_stderr: bool = True,
) -> JvmResult:
    """Preprocess one or more spectra files into a combined ``.DIA`` cache.

    Single-input mode preprocesses one acquisition (decode + index +
    cache; ``.DIA`` lands next to the input). Multi-input mode merges
    gas-phase fractions into one ``.DIA`` at ``output_dia`` — the
    intended GPF workflow.

    Inputs may be ``.mzML``, ``.raw``, ``.d``, or ``.DIA`` — vendor-raw
    formats decode via the bundled MSRawJava (no external msconvert
    dependency).

    Streams the jar's stdout/stderr to ``<output_dir>/logs/`` and
    returns a :class:`JvmResult`. Raises :class:`JvmRunError` on
    non-zero exit.
    """
    args = build_process_dia_args(
        inputs=inputs,
        output_dia=output_dia,
        extra_args=extra_args,
    )
    return run_jar(
        "encyclopedia",
        args=args,
        jvm_heap_max=jvm_heap_max,
        jvm_heap_min=jvm_heap_min,
        jvm_tmpdir=jvm_tmpdir,
        extra_jvm_args=extra_jvm_args,
        log_dir=output_dir / "logs",
        stream_to_stderr=stream_to_stderr,
    )


__all__ = ["build_process_dia_args", "run_process_dia"]
