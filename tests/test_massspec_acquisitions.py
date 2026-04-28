"""Tests for ``constellation.massspec.acquisitions``."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.core.io.schemas import get_schema
from constellation.massspec.acquisitions import (
    ACQUISITION_TABLE,
    Acquisitions,
    validate_acquisitions,
)


def _records():
    return [
        {
            "acquisition_id": 0,
            "source_file": "run_01.raw",
            "source_kind": "thermo_raw",
            "acquisition_datetime": "2026-04-28T10:00:00Z",
        },
        {
            "acquisition_id": 1,
            "source_file": "run_02.mzML",
            "source_kind": "mzml",
            "acquisition_datetime": None,
        },
    ]


def test_schema_registered() -> None:
    assert get_schema("AcquisitionTable") is ACQUISITION_TABLE


def test_from_records_round_trip() -> None:
    acq = Acquisitions.from_records(_records())
    assert acq.ids == [0, 1]
    assert acq.table.column("source_kind").to_pylist() == ["thermo_raw", "mzml"]


def test_empty() -> None:
    acq = Acquisitions.empty()
    assert len(acq) == 0
    assert acq.ids == []


def test_duplicate_acquisition_id_rejected() -> None:
    rows = _records()
    rows[1]["acquisition_id"] = 0
    with pytest.raises(ValueError, match="duplicates"):
        Acquisitions.from_records(rows)


def test_extra_columns_dropped_on_cast() -> None:
    rows = _records()
    rows[0]["extra_column"] = "ignored"
    acq = Acquisitions.from_records(rows)
    assert "extra_column" not in acq.table.column_names


def test_with_metadata_round_trip() -> None:
    acq = Acquisitions.from_records(_records()).with_metadata(
        {"x.library.source": "test_set"}
    )
    assert acq.metadata["x.library.source"] == "test_set"


def test_validate_acquisitions_passes_for_known_ids() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": [0, 1, 0]})
    validate_acquisitions(table, acq)


def test_validate_acquisitions_rejects_unknown_id() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": [0, 99]})
    with pytest.raises(ValueError, match="unknown acquisition_ids"):
        validate_acquisitions(table, acq)


def test_validate_acquisitions_rejects_null_when_not_nullable() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": pa.array([0, None], type=pa.int64())})
    with pytest.raises(ValueError, match="nullable=False"):
        validate_acquisitions(table, acq)


def test_validate_acquisitions_allows_null_when_nullable() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": pa.array([0, None], type=pa.int64())})
    validate_acquisitions(table, acq, nullable=True)


def test_validate_acquisitions_missing_column_raises() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"other_column": [0, 1]})
    with pytest.raises(ValueError, match="missing FK column"):
        validate_acquisitions(table, acq)
