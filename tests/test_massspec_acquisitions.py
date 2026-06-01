"""Tests for ``constellation.massspec.acquisitions``."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.core.io.schemas import get_schema
from constellation.massspec.acquisitions import (
    ACQUISITION_SCHEMA_VERSION,
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


# ──────────────────────────────────────────────────────────────────────
# Schema v1: instrument identity + per-instrument chronological order
# ──────────────────────────────────────────────────────────────────────


def _instrument_records():
    """Two instruments interleaved in input order; a tie; a null datetime;
    a null-serial row. ``acquisition_id`` is the stable key for assertions."""
    return [
        # instrument A, later run (input first to prove ordering, not input pos)
        {"acquisition_id": 10, "source_file": "b.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-02T00:00:00",
         "instrument_serial": "A", "instrument_model": "Lumos"},
        # instrument A, earlier run
        {"acquisition_id": 11, "source_file": "a.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": "A", "instrument_model": "Lumos"},
        # instrument B, tie on datetime with id 13 → id breaks (12 < 13)
        {"acquisition_id": 12, "source_file": "c.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": "B", "instrument_model": "Lumos"},
        {"acquisition_id": 13, "source_file": "d.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": "B", "instrument_model": "Lumos"},
        # instrument A, null datetime → sorts last in A but still ranked
        {"acquisition_id": 14, "source_file": "e.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": None,
         "instrument_serial": "A", "instrument_model": "Lumos"},
        # null instrument_serial → null order
        {"acquisition_id": 15, "source_file": "f.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": None, "instrument_model": None},
    ]


def _order_by_id(acq: Acquisitions) -> dict[int, int | None]:
    rows = acq.table.to_pylist()
    return {r["acquisition_id"]: r["acquisition_order"] for r in rows}


def test_schema_has_version_and_instrument_fields() -> None:
    assert ACQUISITION_TABLE.metadata[b"schema_version"] == b"1"
    assert ACQUISITION_SCHEMA_VERSION == 1
    names = set(ACQUISITION_TABLE.names)
    assert {"instrument_serial", "instrument_model", "acquisition_order"} <= names
    assert ACQUISITION_TABLE.field("instrument_serial").type == pa.string()
    assert ACQUISITION_TABLE.field("acquisition_order").type == pa.int32()


def test_instrument_fields_round_trip() -> None:
    acq = Acquisitions.from_records(_instrument_records())
    by_id = {r["acquisition_id"]: r for r in acq.table.to_pylist()}
    assert by_id[10]["instrument_serial"] == "A"
    assert by_id[10]["instrument_model"] == "Lumos"


def test_backward_compat_four_column_table() -> None:
    # The literal old-parquet shape: only the original 4 columns, no metadata.
    old = pa.table(
        {
            "acquisition_id": pa.array([0, 1], type=pa.int64()),
            "source_file": ["run_01.raw", "run_02.mzML"],
            "source_kind": ["thermo_raw", "mzml"],
            "acquisition_datetime": ["2016-01-01T00:00:00", None],
        }
    )
    acq = Acquisitions(old)
    assert acq.table.column("instrument_serial").to_pylist() == [None, None]
    assert acq.table.column("instrument_model").to_pylist() == [None, None]
    assert acq.table.column("acquisition_order").to_pylist() == [None, None]


def test_with_acquisition_order_basic() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    # instrument A: id 11 earliest → 1, id 10 → 2
    assert by_id[11] == 1
    assert by_id[10] == 2


def test_with_acquisition_order_tie_break_by_id() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    # instrument B: 12 and 13 share a datetime → lower id ranks first
    assert by_id[12] == 1
    assert by_id[13] == 2


def test_with_acquisition_order_null_datetime_sorts_last_but_ranked() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    # instrument A has 3 rows (11, 10, 14-null-dt) → null-dt gets the last rank
    assert by_id[14] == 3


def test_with_acquisition_order_null_serial_gets_null_order() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    assert by_id[15] is None


def test_with_acquisition_order_idempotent() -> None:
    once = Acquisitions.from_records(_instrument_records()).with_acquisition_order()
    twice = once.with_acquisition_order()
    assert _order_by_id(once) == _order_by_id(twice)


def test_with_acquisition_order_empty() -> None:
    acq = Acquisitions.empty().with_acquisition_order()
    assert len(acq) == 0


def test_with_acquisition_order_preserves_uniqueness() -> None:
    acq = Acquisitions.from_records(_instrument_records()).with_acquisition_order()
    assert len(set(acq.ids)) == len(acq.ids)
