"""Tests for :func:`massspec.search.cross_validate_against_scan_metadata`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.schemas import SCAN_METADATA_TABLE
from constellation.massspec.search import (
    Search,
    assemble_search,
    cross_validate_against_scan_metadata,
)


def _psm(**over) -> dict:
    row = {
        "psm_id": 0,
        "raw_file": "runX",
        "acquisition_id": None,
        "scan": 10,
        "sequence": "PEPTIDEK",
        "modified_sequence": "PEPTIDEK",
        "charge": 2,
        "fragmentation": "HCD",
        "mass_analyzer": "FTMS",
        "is_decoy": False,
        "is_contaminant": False,
        "engine": "maxquant",
    }
    row.update(over)
    return row


def _scan_metadata(rows: list[dict]) -> pa.Table:
    full = [
        {
            "scan": r["scan"],
            "level": 2,
            "rt": 1.0,
            "analyzer": r.get("analyzer"),
            "activation_type": r.get("activation_type"),
        }
        for r in rows
    ]
    return pa.Table.from_pylist(full, schema=SCAN_METADATA_TABLE)


def _search(psm_rows: list[dict]) -> Search:
    return assemble_search(acquisitions=Acquisitions.empty(), psms=psm_rows)


def test_all_status_categories():
    search = _search(
        [
            _psm(psm_id=0, scan=10, mass_analyzer="FTMS", fragmentation="HCD"),
            _psm(psm_id=1, scan=11, mass_analyzer="FTMS", fragmentation="CID"),
            _psm(psm_id=2, scan=12, mass_analyzer="ITMS", fragmentation="HCD"),
            _psm(psm_id=3, scan=13, mass_analyzer="FTMS", fragmentation="HCD"),
            _psm(psm_id=4, scan=14, mass_analyzer="ITMS", fragmentation="ETD"),
        ]
    )
    scan_meta = _scan_metadata(
        [
            {"scan": 10, "analyzer": "FTMS", "activation_type": "hcd"},  # matched
            {"scan": 11, "analyzer": "FTMS", "activation_type": "hcd"},  # activation_mismatch
            {"scan": 12, "analyzer": "FTMS", "activation_type": "hcd"},  # analyzer_mismatch
            # scan 13 absent → scan_absent
            {"scan": 14, "analyzer": "FTMS", "activation_type": "hcd"},  # both_mismatch
        ]
    )
    result = cross_validate_against_scan_metadata(search, scan_meta)
    status = dict(
        zip(
            result.column("psm_id").to_pylist(),
            result.column("status").to_pylist(),
            strict=True,
        )
    )
    assert status == {
        0: "matched",
        1: "activation_mismatch",  # UPPER "CID" vs lowercase "hcd" handled
        2: "analyzer_mismatch",
        3: "scan_absent",
        4: "both_mismatch",
    }


def test_activation_casing_handled():
    # PSM stores UPPER fragmentation; scan_metadata stores lowercase.
    search = _search([_psm(psm_id=0, scan=10, fragmentation="HCD", mass_analyzer="FTMS")])
    scan_meta = _scan_metadata([{"scan": 10, "analyzer": "FTMS", "activation_type": "hcd"}])
    result = cross_validate_against_scan_metadata(search, scan_meta)
    assert result.column("status").to_pylist() == ["matched"]


def test_raw_file_scoping():
    search = _search(
        [
            _psm(psm_id=0, raw_file="runA", scan=10),
            _psm(psm_id=0, raw_file="runB", scan=10),  # same psm_id, different run
        ]
    )
    scan_meta = _scan_metadata([{"scan": 10, "analyzer": "FTMS", "activation_type": "hcd"}])
    result = cross_validate_against_scan_metadata(search, scan_meta, raw_file="runA")
    assert result.num_rows == 1
    assert result.column("raw_file").to_pylist() == ["runA"]


def test_multi_acquisition_without_raw_file_raises():
    search = _search(
        [
            _psm(psm_id=0, raw_file="runA", scan=10),
            _psm(psm_id=1, raw_file="runB", scan=10),
        ]
    )
    scan_meta = _scan_metadata([{"scan": 10, "analyzer": "FTMS", "activation_type": "hcd"}])
    with pytest.raises(ValueError, match="per-acquisition"):
        cross_validate_against_scan_metadata(search, scan_meta)


def test_accepts_psms_table_directly():
    search = _search([_psm(psm_id=0, scan=10)])
    scan_meta = _scan_metadata([{"scan": 10, "analyzer": "FTMS", "activation_type": "hcd"}])
    # passing the bare psms table works the same as passing the Search
    result = cross_validate_against_scan_metadata(search.psms, scan_meta)
    assert result.column("status").to_pylist() == ["matched"]
