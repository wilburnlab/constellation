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

import shutil
import sys
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
    EncyclopeDIA writes the consolidated cache there.

    **Single-input mode must NOT emit ``-o``.** The jar hard-errors with
    "When using one input, do not specify an output file!" and exits 1 —
    it always writes the cache next to the input as
    ``<input_stem>.dia``. So ``output_dia`` is *ignored* here when there
    is exactly one input; :func:`run_process_dia` relocates the
    produced file to ``output_dia`` afterwards.

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
    if output_dia is not None and len(inputs) > 1:
        args.extend(["-o", str(output_dia)])
    args.extend(str(a) for a in extra_args)
    return args


def _same_file(a: Path, b: Path) -> bool:
    """True when two paths denote the same file.

    ``Path.samefile`` is the honest check — it sees through symlinks and
    case-insensitive filesystems (macOS is best-effort-supported) — but
    it raises when either side is absent, so fall back to comparing
    resolved paths.
    """
    try:
        return a.samefile(b)
    except OSError:
        return a.resolve() == b.resolve()


def single_input_dia_path(
    input_file: Path,
    *,
    cwd: Path | None = None,
    exclude: Sequence[Path] = (),
) -> Path | None:
    """Locate the ``.dia`` the jar produced from a single input.

    Single-input mode ignores ``-o``, so the output lands by convention
    rather than by request — and the convention has the same two axes of
    drift :func:`library_search.find_search_elib` already handles for the
    ``.elib``:

      * **where** — 6.5.15 writes to the process's *current working
        directory*, not next to the input. Older behaviour was
        next-to-input. Both are checked, cwd first.
      * **what** — ``<stem>.dia`` vs ``<name>.dia``.

    ``cwd`` should be the same path the runner handed :func:`run_jar`.
    Returns ``None`` when nothing matches so the caller can surface a
    clear error naming where it looked.

    ``exclude`` lists paths that must never be reported as the produced
    cache — pass the run's inputs. A ``.DIA`` input is legal (the jar
    reuses an existing cache and emits nothing new), and for such an
    input ``with_suffix(".dia")`` is the input itself. Without the
    exclusion the caller would relocate the user's source file, and
    every later step reading that input fails on a path that no longer
    exists.
    """
    candidates: list[Path] = []
    if cwd is not None:
        cwd = Path(cwd)
        candidates.extend(
            [
                cwd / f"{input_file.stem}.dia",
                cwd / f"{input_file.name}.dia",
            ]
        )
    candidates.extend(
        [
            input_file.with_suffix(".dia"),
            input_file.parent / f"{input_file.name}.dia",
        ]
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if any(_same_file(candidate, Path(x)) for x in exclude):
            continue
        return candidate
    return None


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

    Single-input mode preprocesses one acquisition. The jar refuses
    ``-o`` with one input and names the cache itself, so this function
    locates the produced ``<input_stem>.dia`` and moves it to
    ``output_dia`` afterwards. ``output_dia`` therefore means the same
    thing regardless of input count, and no caller has to special-case
    the runs that happen to have exactly one input. Multi-input mode
    merges gas-phase fractions into one ``.DIA`` at ``output_dia`` —
    the intended GPF workflow.

    Inputs may be ``.mzML``, ``.raw``, ``.d``, or ``.DIA`` — vendor-raw
    formats decode via the bundled MSRawJava (no external msconvert
    dependency).

    Streams the jar's stdout/stderr to ``<output_dir>/logs/`` and
    returns a :class:`JvmResult`. Raises :class:`JvmRunError` on
    non-zero exit.

    Runs with ``cwd=output_dir``, matching
    :func:`library_search.run_library_search`. This matters for more
    than tidiness: in single-input mode the jar writes ``<stem>.dia``
    into its working directory, so without this it would land in
    whatever shell (or Slurm submit) directory launched the run.

    Path arguments are resolved against the *caller's* working
    directory before that switch, so relative paths mean what the
    caller meant.
    """
    # Resolve before run_jar changes directory. A relative
    # ``-i data/sample.raw`` would otherwise be looked up beneath
    # output_dir, and a relative output_dia would be written into an
    # unintended nested location. Both in-tree callers already resolve;
    # this is for the public wrapper's own callers.
    inputs = [Path(p).resolve() for p in inputs]
    output_dir = Path(output_dir).resolve()
    if output_dia is not None:
        output_dia = Path(output_dia).resolve()
    if jvm_tmpdir is not None:
        # Java resolves -Djava.io.tmpdir against ITS cwd, which is
        # output_dir — so a relative ./scratch would silently become
        # <output_dir>/scratch rather than the caller's scratch, sending
        # spill files to the wrong (possibly much smaller) filesystem.
        jvm_tmpdir = Path(jvm_tmpdir).resolve()

    args = build_process_dia_args(
        inputs=inputs,
        output_dia=output_dia,
        extra_args=extra_args,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_jar(
        "encyclopedia",
        args=args,
        jvm_heap_max=jvm_heap_max,
        jvm_heap_min=jvm_heap_min,
        jvm_tmpdir=jvm_tmpdir,
        extra_jvm_args=extra_jvm_args,
        log_dir=output_dir / "logs",
        stream_to_stderr=stream_to_stderr,
        cwd=output_dir,
    )

    # Single-input mode ignores -o and names the cache itself, so the
    # relocation belongs here rather than in each caller — the
    # orchestrator called this directly and marked its stage complete
    # while the cache still sat under the input-derived name.
    #
    # Unconditional, deliberately: an earlier interrupted run can leave
    # a stale file at output_dia, and skipping the move when the
    # destination already exists would silently bind those old spectra
    # to this run's manifest.
    if len(inputs) == 1 and output_dia is not None:
        # exclude=inputs: never mistake the source for the product. A
        # .DIA input is legal and the jar then emits nothing new, so the
        # naive lookup returns the input itself and the move would carry
        # the user's own file away.
        produced = single_input_dia_path(
            inputs[0], cwd=output_dir, exclude=inputs
        )
        if produced is not None and not _same_file(produced, output_dia):
            output_dia.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), str(output_dia))
            if stream_to_stderr:
                print(
                    f"process-dia: single input — moved {produced.name} "
                    f"→ {output_dia}",
                    file=sys.stderr,
                )
        elif (
            produced is None
            and inputs[0].suffix.lower() == ".dia"
            and inputs[0].is_file()
            and not _same_file(inputs[0], output_dia)
        ):
            # Cache reuse: the jar was handed a .DIA and produced no new
            # file. COPY, never move — output_dia still has to mean what
            # it means for every other input count, but not at the cost
            # of consuming the caller's source.
            output_dia.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(inputs[0]), str(output_dia))
            if stream_to_stderr:
                print(
                    f"process-dia: single .DIA input reused — copied "
                    f"{inputs[0].name} → {output_dia}",
                    file=sys.stderr,
                )
    return result


__all__ = [
    "build_process_dia_args",
    "run_process_dia",
    "single_input_dia_path",
]
