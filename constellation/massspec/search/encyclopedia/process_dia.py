"""Wrapper for ``-convert -processDIA`` — mzML / .raw / .d / .DIA preprocessing
and gas-phase-fraction merging.

The jar's CLI takes colon-delimited input paths (``-i a.mzML:b.mzML:c.mzML``)
when merging multiple files into a single ``.dia`` cache; this wrapper
takes a Python ``list[Path]`` and joins for the user.

Filled in PR 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from constellation.thirdparty.jvm import JvmResult


def run_process_dia(
    *,
    inputs: Sequence[Path],
    output_dia: Path,
    output_dir: Path,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_args: Sequence[str] = (),
    extra_jvm_args: Sequence[str] = (),
    stream_to_stderr: bool = True,
) -> JvmResult:
    """Preprocess one or more spectra files into a combined ``.dia`` cache.

    Single-input mode = preprocess (decode + index) one acquisition.
    Multi-input mode = merge gas-phase fractions into one ``.dia``.

    Inputs may be ``.mzML``, ``.raw``, ``.d``, or ``.DIA`` — vendor-raw
    formats are handled directly via the bundled MSRawJava (no
    msconvert dependency).

    Filled in PR 3.
    """
    raise NotImplementedError(
        "run_process_dia lands in PR 3 — wraps `-convert -processDIA`. "
        "PR 0 ships the architecture only."
    )


__all__ = ["run_process_dia"]
