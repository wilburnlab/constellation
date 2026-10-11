"""Reproduce a reference EncyclopeDIA search, when one is available locally.

Opt-in: a real search needs the jar, a spectra file, a library, a FASTA
and minutes of CPU, so nothing here runs unless the environment names
them. It is the check behind "a search of a Constellation-written `.dia`
equals a search of the reference file" (first run 2026-10-11, recorded
in ``docs/plans/encyclopedia-6.9.23-survey.md``), kept so the search
path can be rewritten later against a fixed target.

Set all of::

    CONSTELLATION_SEARCH_REFERENCE_INPUT    the file to search (.dia or .raw), in a
                                            directory the jar may write beside
    CONSTELLATION_SEARCH_REFERENCE_LIBRARY  the library it was searched against
    CONSTELLATION_SEARCH_REFERENCE_FASTA    the FASTA
    CONSTELLATION_SEARCH_REFERENCE_RESULT   the result library to reproduce

and optionally ``CONSTELLATION_SEARCH_REFERENCE_ARGS``: further
``massspec search`` arguments, shell-quoted, for whatever the reference
run used that differs from the defaults (tolerances, ``-fixed``), and
``--percolator-version`` on a kernel where the bundled build cannot run.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from constellation.massspec.io.encyclopedia import compare_results

_ENV = {
    name: os.environ.get(f"CONSTELLATION_SEARCH_REFERENCE_{name}")
    for name in ("INPUT", "LIBRARY", "FASTA", "RESULT")
}

#: Two searches of one file differ by a few precursors at the FDR line
#: (11 of 31,833 at worst in the first measurements, Jaccard 0.99965).
MIN_OVERLAP = 0.999


@pytest.mark.slow
@pytest.mark.skipif(
    not all(_ENV.values()),
    reason="set CONSTELLATION_SEARCH_REFERENCE_{INPUT,LIBRARY,FASTA,RESULT} (module docstring)",
)
def test_search_reproduces_the_reference_result(tmp_path: Path) -> None:
    from constellation.cli.__main__ import main

    out = tmp_path / "out"
    argv = [
        "massspec",
        "search",
        "--mzml",
        _ENV["INPUT"],
        "--library",
        _ENV["LIBRARY"],
        "--fasta",
        _ENV["FASTA"],
        "--output-dir",
        str(out),
        "--no-ingest",
        "--no-progress",
        *shlex.split(os.environ.get("CONSTELLATION_SEARCH_REFERENCE_ARGS", "")),
    ]
    assert main(argv) == 0, f"search failed; see {out / 'logs'}"
    (result,) = out.glob("*.elib")

    comparison = compare_results(_ENV["RESULT"], result)
    report = comparison.report()
    assert comparison.precursors.jaccard >= MIN_OVERLAP, report
    assert comparison.proteins.jaccard >= MIN_OVERLAP, report
    # What both runs identified, they place and quantify identically.
    assert comparison.rt_abs_s["max"] < 1e-6, report
    assert comparison.intensity["frac_within_1pct"] == 1.0, report
