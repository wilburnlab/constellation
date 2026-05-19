"""Wrapper for ``-libexport`` — combine per-acquisition search results
into a quant-report ``.elib`` (or ``.blib``).

This is EncyclopeDIA's quant-report pathway: takes a directory of
search outputs (``.elib`` per acquisition), aligns retention times
across them (``-a`` default ``true``), and writes a consolidated
library file. The transcriptome→proteomics pipeline's Stage 10
consumes this — per-injection ``.elibs`` from Stage 9 (each
collision-filtered) feed in; one quant-report ``.elib`` comes out.

Nested sub-modes (``-pecan`` / ``-xcordia`` / ``-phospho``) target
non-default scoring pipelines and remain deferred — they earn wrappers
when a concrete lab use case appears. The ``--encyclopedia-arg
FLAG=VALUE`` escape hatch on the CLI lets users reach unwrapped flags
without waiting for a code change.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from constellation.thirdparty.jvm import JvmResult, run_jar


def build_library_export_args(
    *,
    search_dir: Path,
    library: Path,
    output_elib: Path,
    align: bool = True,
    write_blib: bool = False,
    fasta: Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Translate typed kwargs into the EncyclopeDIA ``-libexport`` argv.

    Pure function — Tier A tests exercise this without spawning Java.

    Output shape (per ``java -jar encyclopedia.jar -libexport --help``)::

        -libexport
        -i <search_dir>          # dir of per-acquisition .elib files (or one file)
        -l <library>             # the .dlib / .elib originally searched
        -o <output_elib>         # consolidated quant report
        -a true|false            # RT-align across input files (default true)
        [-blib]                  # write BLIB instead of ELIB
        [-f <fasta>]             # required for Pecan / XCorDIA sub-modes only
    """
    args: list[str] = [
        "-libexport",
        "-i",
        str(search_dir),
        "-l",
        str(library),
        "-o",
        str(output_elib),
        "-a",
        _bool_str(align),
    ]
    if write_blib:
        args.append("-blib")
    if fasta is not None:
        args.extend(["-f", str(fasta)])
    args.extend(str(a) for a in extra_args)
    return args


def run_library_export(
    *,
    search_dir: Path,
    library: Path,
    output_elib: Path,
    output_dir: Path,
    align: bool = True,
    write_blib: bool = False,
    fasta: Path | None = None,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_args: Sequence[str] = (),
    extra_jvm_args: Sequence[str] = (),
    stream_to_stderr: bool = True,
) -> JvmResult:
    """Build a quant-report library from a directory of acquisition searches.

    ``search_dir`` may be a directory of per-acquisition ``.elib`` files
    OR a single file (EncyclopeDIA's ``-i`` accepts either). ``library``
    is the original ``.dlib`` / ``.elib`` that was searched against
    (default mode requires it). When ``write_blib=True``, the output is
    written in BLIB format instead of ELIB — useful for downstream tools
    that consume BLIBs natively.

    Streams the jar's stdout/stderr to ``<output_dir>/logs/`` and
    returns a :class:`JvmResult`. Raises :class:`JvmRunError` on
    non-zero exit.
    """
    args = build_library_export_args(
        search_dir=search_dir,
        library=library,
        output_elib=output_elib,
        align=align,
        write_blib=write_blib,
        fasta=fasta,
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


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "build_library_export_args",
    "run_library_export",
]
