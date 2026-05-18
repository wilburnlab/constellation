"""Wrapper for ``-libexport`` — combine per-acquisition search results into
a quant-report ``.elib`` (or ``.blib``).

This is EncyclopeDIA's quant-report pathway: takes a directory of search
outputs (``.elib`` per acquisition), aligns retention times across them
(``-a`` default true), and writes a consolidated library file.

Nested sub-modes (``-pecan`` / ``-xcordia`` / ``-phospho``) target
non-default scoring pipelines and are deferred to PR 5+.

Filled in PR 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from constellation.thirdparty.jvm import JvmResult


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

    ``search_dir`` may be a directory or a single file (per
    EncyclopeDIA's ``-i`` semantics). ``library`` is the original
    library that was searched against (required by the default mode).

    Filled in PR 4.
    """
    raise NotImplementedError(
        "run_library_export lands in PR 4 — wraps `-libexport`. PR 0 "
        "ships the architecture only."
    )


__all__ = ["run_library_export"]
