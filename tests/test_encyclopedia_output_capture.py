"""The checked-in capture of what EncyclopeDIA 6.9.23 writes.

``scripts/survey-encyclopedia-outputs.py`` records, from a real run, the
table definitions of a result library and of a spectra file and the list
of files a search leaves behind. These tests hold Constellation's own
schemas to that capture, so the difference between "what the pinned jar
writes" and "what our reader and writers assume" is stated in one place
and a version bump shows up as a failing assertion with a diff.
"""

from __future__ import annotations

import json
import sqlite3
from importlib import resources

import pytest

from constellation.massspec.io.encyclopedia import _schema
from constellation.thirdparty.encyclopedia import MINIMUM_VERSION


def _capture(*parts: str):
    return resources.files("constellation.data").joinpath(
        "_raw", "encyclopedia", MINIMUM_VERSION, *parts
    )


def _build(script: str) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(script)
    return con


def _columns(con: sqlite3.Connection) -> dict[str, list[str]]:
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return {t: [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}


def _indexes(con: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
    }


@pytest.fixture(scope="module")
def captured_library() -> sqlite3.Connection:
    return _build(_capture("schema", "result_library.sql").read_text())


def test_result_library_capture_is_container_0_1_18(captured_library) -> None:
    header = _capture("schema", "result_library.sql").read_text().splitlines()[0]
    assert MINIMUM_VERSION in header and "container 0.1.18" in header
    assert len(_columns(captured_library)) == 10
    assert len(_indexes(captured_library)) == 22


def test_writer_schema_differs_from_the_capture_only_in_known_ways(captured_library) -> None:
    """Constellation still writes the 0.1.15 layout. Everything it writes
    exists in 0.1.18 in the same column order; what 0.1.18 adds is listed
    here, and is what the writer has to grow to match (plan phase 7)."""
    ours = sqlite3.connect(":memory:")
    for _name, ddl in _schema.LIBRARY_TABLES:
        ours.execute(ddl)
    for statement in _schema.LIBRARY_INDEXES:
        ours.execute(statement)
    theirs, mine = _columns(captured_library), _columns(ours)

    assert set(theirs) - set(mine) == {"decoyentries"}
    assert set(mine) <= set(theirs)
    added = {}
    for table, columns in mine.items():
        assert [c for c in theirs[table] if c in columns] == columns, table
        extra = [c for c in theirs[table] if c not in columns]
        if extra:
            added[table] = extra
    assert added == {
        "entries": ["IonMobility"],
        "peptidequants": ["QuantIonCorrelationLength", "QuantIonCorrelationArray"],
    }
    assert _indexes(ours) <= _indexes(captured_library)
    assert len(_indexes(captured_library) - _indexes(ours)) == 16


def test_spectra_file_writer_matches_the_capture_exactly() -> None:
    """The `.dia` writer's tables, column order and indexes are the
    reference writer's."""
    script = _capture("schema", "spectra_file.sql").read_text()
    assert f"format {_schema.DIA_FORMAT_VERSION}" in script.splitlines()[0]
    theirs = _build(script)
    ours = sqlite3.connect(":memory:")
    for _name, ddl in _schema.DIA_TABLES_V08:
        ours.execute(ddl)
    for statement in _schema.DIA_INDEXES_V08:
        ours.execute(statement)
    assert _columns(ours) == _columns(theirs)
    assert _indexes(ours) == _indexes(theirs)


def test_peptide_centric_search_output_inventory() -> None:
    doc = json.loads(_capture("outputs", "encyclopedia.json").read_text())
    assert doc["encyclopedia_version"] == MINIMUM_VERSION
    assert doc["container_version"] == "0.1.18"
    files = {f["name"]: f for f in doc["files"]}
    assert all(name.startswith("<input>") for name in files)

    # The result library, the Percolator input and both Percolator outputs.
    assert files["<input>.elib"]["tables_with_rows"] == [
        "entries",
        "metadata",
        "peptidequants",
        "peptidescores",
        "peptidetoprotein",
        "proteinscores",
    ]
    assert len(files["<input>.features.txt"]["columns"]) == 35
    assert files["<input>.encyclopedia.txt"]["columns"] == files[
        "<input>.encyclopedia.decoy.txt"
    ]["columns"]
    # Two calibration passes, three fitted relations each, as a table and a plot.
    for stage in ("first", "final"):
        for fit in ("rt_fit", "ms1_fit", "ms2_fit"):
            assert f"<input>.encyclopedia.txt.{stage}.{fit}.txt" in files
            assert f"<input>.encyclopedia.txt.{stage}.{fit}.pdf" in files

    # The metadata table names every effective parameter, including ones
    # the help does not print; the keys named for the run are normalised,
    # so nothing in the capture says which run it was taken from.
    keys = doc["result_metadata_keys"]
    assert {"<source file> search type", "<source file> library", "TIC_<source file>"} <= set(keys)
    assert not any(".raw" in key for key in keys)
    assert {"-percolatorTrainingIterations", "-percolatorVersion", "version"} <= set(keys)
