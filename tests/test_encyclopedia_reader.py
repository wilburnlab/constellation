"""End-to-end reader tests against the bench's sample dlib / elib files.

Skipped if the sample files aren't present (so the suite stays green on
machines without the lab's data tree). Sample-file location is
configurable via ``$CONSTELLATION_ENCYCLOPEDIA_SAMPLES`` — keep
lab-specific filenames out of the test source.

Subsets the dlib + elib down to 200 entries each so reads finish in
seconds (whole-file reads take minutes due to per-entry fragment
re-annotation), and scrubs run-identifying metadata before the test
exercises the file.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from constellation.massspec.io.encyclopedia import read_encyclopedia
from constellation.massspec.library.io import load_library
from constellation.massspec.quant.io import load_quant
from constellation.massspec.search.io import load_search


# Default to the lab's local layout when the env var isn't set; tests
# skip cleanly if the directory doesn't exist on the host.
SAMPLES = Path(
    os.environ.get(
        "CONSTELLATION_ENCYCLOPEDIA_SAMPLES",
        "~/WilburnLab/MS_Quant/test_encyclopedia_files",
    )
).expanduser()


def _first_match(suffix: str) -> Path | None:
    """First sample file with the given suffix, or ``None`` if absent."""
    if not SAMPLES.exists():
        return None
    matches = sorted(SAMPLES.glob(f"*{suffix}"))
    return matches[0] if matches else None


# Generic sentinels we replace lab-identifying values with.
_SCRUBBED_SOURCE_FILE = "test-acquisition.mzML"
_SCRUBBED_RUN_NAME = "test-acquisition"


def _subset_dlib_or_elib(src: Path, dst: Path, limit: int) -> Path:
    """Copy ``src`` to ``dst``, trim to ``limit`` entries, and scrub
    run-identifying metadata.

    DLIB and ELIB share the same SQLite schema, so the subsetting query
    is identical for both. Scrub:
      * ``entries.SourceFile`` and ``peptidequants/peptidescores/proteinscores.SourceFile``
        → replaced with a generic placeholder so test assertions don't
        depend on lab-side filenames.
      * The ``metadata`` table is wiped and replaced with a minimal
        synthetic set — encyclopedia stamps run-name-prefixed keys
        (``TIC_<file>.mzML`` etc.) that would otherwise leak the run id.
    """
    shutil.copy(src, dst)
    con = sqlite3.connect(dst)
    try:
        con.executescript(f"""
DELETE FROM entries WHERE rowid NOT IN (SELECT rowid FROM entries ORDER BY rowid LIMIT {limit});
DELETE FROM peptidequants WHERE PeptideSeq NOT IN (SELECT PeptideSeq FROM entries);
DELETE FROM peptidescores WHERE PeptideSeq NOT IN (SELECT PeptideSeq FROM entries);
DELETE FROM peptidetoprotein WHERE PeptideSeq NOT IN (SELECT PeptideSeq FROM entries);
DELETE FROM proteinscores WHERE ProteinAccession NOT IN (SELECT ProteinAccession FROM peptidetoprotein);

UPDATE entries SET SourceFile = '{_SCRUBBED_SOURCE_FILE}';
UPDATE peptidequants SET SourceFile = '{_SCRUBBED_SOURCE_FILE}';
UPDATE peptidescores SET SourceFile = '{_SCRUBBED_SOURCE_FILE}';
UPDATE proteinscores SET SourceFile = '{_SCRUBBED_SOURCE_FILE}';

DELETE FROM metadata;
INSERT INTO metadata (Key, Value) VALUES ('version', '0.1.15');
INSERT INTO metadata (Key, Value) VALUES ('-acquisition', 'DIA');
INSERT INTO metadata (Key, Value) VALUES ('-enzyme', 'Trypsin');
INSERT INTO metadata (Key, Value) VALUES ('source-file', '{_SCRUBBED_SOURCE_FILE}');

VACUUM;
""")
        con.commit()
    finally:
        con.close()
    return dst


@pytest.fixture(scope="session")
def dlib_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = _first_match(".dlib")
    if src is None:
        pytest.skip(
            "no .dlib sample present; "
            "set $CONSTELLATION_ENCYCLOPEDIA_SAMPLES to a directory containing one"
        )
    out = tmp_path_factory.mktemp("encyc-dlib") / "subset.dlib"
    return _subset_dlib_or_elib(src, out, limit=200)


@pytest.fixture(scope="session")
def elib_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = _first_match(".elib")
    if src is None:
        pytest.skip(
            "no .elib sample present; "
            "set $CONSTELLATION_ENCYCLOPEDIA_SAMPLES to a directory containing one"
        )
    out = tmp_path_factory.mktemp("encyc-elib") / "subset.elib"
    return _subset_dlib_or_elib(src, out, limit=200)


# ──────────────────────────────────────────────────────────────────────
# DLIB — small subset (fast)
# ──────────────────────────────────────────────────────────────────────


def test_dlib_subset_full_read_returns_three_projections(dlib_subset: Path):
    """A real-world dlib carries entries + chromatograms + peptidescores
    → all three (Library, Quant, Search) populate."""
    r = read_encyclopedia(dlib_subset)
    # 200 entries → at most 200 unique peptides
    assert r.library.n_peptides <= 200
    assert r.library.n_precursors <= 200
    assert r.library.n_fragments > 0
    # Quant invariants — single SourceFile after scrubbing
    assert r.quant is not None
    assert r.quant.acquisitions.table.num_rows == 1
    assert r.quant.precursor_quant.num_rows > 0
    # Search invariants — peptidescores has rows
    assert r.search is not None
    assert r.search.peptide_scores.num_rows > 0


def test_dlib_subset_metadata_round_trips(dlib_subset: Path):
    """The encyclopedia metadata table populates Library.metadata_extras
    via the ``x.encyclopedia.metadata`` namespace."""
    r = read_encyclopedia(dlib_subset)
    md = r.library.metadata_extras["x.encyclopedia.metadata"]
    # Synthetic keys we wrote in _subset_dlib_or_elib.
    assert md["-acquisition"] == "DIA"
    assert md["-enzyme"] == "Trypsin"


def test_dlib_subset_modified_sequences_are_proforma(dlib_subset: Path):
    """Every modified_sequence in PEPTIDE_TABLE must parse cleanly via
    parse_proforma — the modseq translation produced valid ProForma 2.0."""
    from constellation.core.sequence.proforma import parse_proforma

    r = read_encyclopedia(dlib_subset)
    for ms in r.library.peptides.column("modified_sequence").to_pylist():
        parse_proforma(ms)  # must not raise


def test_dlib_subset_scrubbed_source_file(dlib_subset: Path):
    """The fixture replaces SourceFile with a generic placeholder; the
    Acquisitions table should pick that up, not a lab-side filename."""
    r = read_encyclopedia(dlib_subset)
    assert r.quant is not None
    sources = r.quant.acquisitions.table.column("source_file").to_pylist()
    assert sources == [_SCRUBBED_SOURCE_FILE]


def test_dlib_subset_load_library_via_suffix(dlib_subset: Path):
    """Suffix-based dispatch (no explicit format=) must resolve to the
    encyclopedia adapter and return just the Library projection."""
    lib = load_library(dlib_subset)
    assert lib.n_peptides > 0


def test_dlib_subset_load_quant_via_suffix(dlib_subset: Path):
    quant = load_quant(dlib_subset)
    assert quant.precursor_quant.num_rows > 0


def test_dlib_subset_load_search_via_suffix(dlib_subset: Path):
    search = load_search(dlib_subset)
    assert search.peptide_scores.num_rows > 0


# ──────────────────────────────────────────────────────────────────────
# ELIB — same SQLite schema; subset path mirrors dlib's
# ──────────────────────────────────────────────────────────────────────


def test_elib_subset_returns_three_projections(elib_subset: Path):
    r = read_encyclopedia(elib_subset)
    assert r.library.n_peptides <= 200
    assert r.quant is not None
    assert r.quant.precursor_quant.num_rows > 0
    assert r.search is not None
    assert r.search.peptide_scores.num_rows > 0


def test_elib_subset_load_library_via_suffix(elib_subset: Path):
    """``.elib`` resolves to the same encyclopedia adapter as ``.dlib``
    — same on-disk format, different file extension."""
    lib = load_library(elib_subset)
    assert lib.n_peptides > 0
