"""Reproduce a reference EncyclopeDIA search, when one is available locally.

Opt-in: a real search needs the jar, a spectra file, a library, a FASTA
and minutes of CPU, so nothing here runs unless the environment names
them. It is the check behind "a search of a Constellation-written `.dia`
equals a search of the reference file" (first run 2026-10-11, recorded
in ``docs/plans/encyclopedia-6.9.23-survey.md``), kept so the search
path can be rewritten later against a fixed target.

Set all of::

    CONSTELLATION_SEARCH_REFERENCE_INPUT    the acquisition to search: a converted
                                            bundle, a Thermo .raw, or a single-run
                                            EncyclopeDIA spectra file
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

from constellation.core.manifests import read_manifest
from constellation.massspec.io.encyclopedia import compare_results
from constellation.massspec.io.encyclopedia._mirror import count_rows, open_library
from constellation.massspec.manifest import SearchManifest
from constellation.massspec.workflows.encyclopedia import read_experiment_table

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

    experiment = tmp_path / "experiment"
    argv = [
        "massspec",
        "search",
        "--input",
        _ENV["INPUT"],
        "--library",
        _ENV["LIBRARY"],
        "--fasta",
        _ENV["FASTA"],
        "--output-dir",
        str(experiment),
        "--no-progress",
        *shlex.split(os.environ.get("CONSTELLATION_SEARCH_REFERENCE_ARGS", "")),
    ]
    assert main(argv) == 0, f"search failed; see {experiment / 'units'}"
    (unit,) = (experiment / "units").iterdir()
    manifest = read_manifest(unit, expect=SearchManifest)
    result = unit / manifest.outputs["result_library"]

    comparison = compare_results(_ENV["RESULT"], result)
    report = comparison.report()
    assert comparison.precursors.jaccard >= MIN_OVERLAP, report
    assert comparison.proteins.jaccard >= MIN_OVERLAP, report
    # What both runs identified, they place and quantify identically.
    assert comparison.rt_abs_s["max"] < 1e-6, report
    assert comparison.intensity["frac_within_1pct"] == 1.0, report

    # And the tables hold what the result library holds: every precursor,
    # every quantity, every score row, bound to the one acquisition.
    con = open_library(result)
    try:
        stored = {name: count_rows(con, name) for name in ("entries", "peptidequants", "peptidescores")}
        total = con.execute("SELECT SUM(TotalIntensity) FROM peptidequants").fetchone()[0]
    finally:
        con.close()
    quant = read_experiment_table(experiment, "precursor_quant")
    assert read_experiment_table(experiment, "precursor").num_rows == stored["entries"]
    assert quant.num_rows == stored["peptidequants"]
    assert read_experiment_table(experiment, "precursor_score").num_rows == stored["peptidescores"]
    assert sum(quant.column("intensity").to_pylist()) == pytest.approx(total, rel=1e-9)
    assert set(quant.column("acquisition_id").to_pylist()) == set(manifest.acquisition_ids)
    assert manifest.caveats == []
