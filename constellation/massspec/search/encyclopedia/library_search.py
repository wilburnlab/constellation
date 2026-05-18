"""Wrapper for EncyclopeDIA library search (default-mode invocation).

Maps to the jar's default entry point::

    java -jar encyclopedia-<ver>.jar -i <mzml|dia> -l <library.elib> [...]

Filled in PR 1. The signature below is the contract the CLI handler in
:mod:`constellation.massspec.cli` is wired to call; the body is a stub
so importing the package does not require the survey-doc work to be
complete.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from constellation.thirdparty.jvm import JvmResult


def run_library_search(
    *,
    mzml: Path,
    library: Path,
    fasta: Path | None = None,
    output: Path,
    output_dir: Path,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_args: Sequence[str] = (),
    extra_jvm_args: Sequence[str] = (),
    stream_to_stderr: bool = True,
) -> JvmResult:
    """Run an EncyclopeDIA DIA library search.

    Required: ``mzml`` (or ``.dia`` / ``.raw`` / ``.d``), ``library``
    (``.dlib`` or ``.elib``), ``output`` (path to write the resulting
    ``.elib``). Optional ``fasta`` is the background proteome for
    decoy generation when the library lacks decoys. The runner streams
    the jar's stdout/stderr to ``output_dir/logs/`` and returns a
    :class:`JvmResult`.

    Filled in PR 1.
    """
    raise NotImplementedError(
        "run_library_search lands in PR 1 — see "
        "docs/plans/encyclopedia-6.5.15-utilities.md for the surveyed "
        "flag table. PR 0 ships the architecture (jvm runner + CLI "
        "parser + manifest helper) only."
    )


__all__ = ["run_library_search"]
