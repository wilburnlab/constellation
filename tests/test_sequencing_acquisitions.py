"""Tests for ``constellation.sequencing.acquisitions``.

Smoke-level container behavior: PK uniqueness on construction, cast-on-
construct dropping extras, FK cross-check via ``validate_acquisitions``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.acquisitions import (
    SEQUENCING_ACQUISITION_TABLE,
    Acquisitions,
    validate_acquisitions,
)


def _row(**kwargs):
    """Build a SEQUENCING_ACQUISITION_TABLE-shaped row, defaulting unset
    fields to None."""
    base = {
        "source_path": "/tmp/x.sam",
        "source_kind": "sam",
        "acquisition_datetime": None,
        "instrument_id": None,
        "flow_cell_id": None,
        "flow_cell_type": None,
        "sample_kit": None,
        "basecaller_model": None,
        "experiment_type": None,
    }
    base.update(kwargs)
    return base


def test_empty_constructor() -> None:
    acq = Acquisitions.empty()
    assert len(acq) == 0
    assert acq.ids == []
    assert acq.table.schema.equals(SEQUENCING_ACQUISITION_TABLE)


def test_from_records_basic() -> None:
    acq = Acquisitions.from_records(
        [
            _row(acquisition_id=1, source_path="/data/run1.sam"),
            _row(acquisition_id=2, source_path="/data/run2.sam"),
        ]
    )
    assert acq.ids == [1, 2]
    assert len(acq) == 2


def test_pk_uniqueness_raises() -> None:
    with pytest.raises(ValueError, match="acquisition_id contains duplicates"):
        Acquisitions.from_records(
            [
                _row(acquisition_id=1),
                _row(acquisition_id=1),
            ]
        )


def test_cast_drops_extra_columns() -> None:
    """Wider input table with extra columns should be projected."""
    wider = pa.table(
        {
            "acquisition_id": pa.array([1, 2], pa.int64()),
            "source_path": pa.array(["/a.sam", "/b.sam"], pa.string()),
            "source_kind": pa.array(["sam", "sam"], pa.string()),
            # Extra column not in schema:
            "extra_diagnostic": pa.array(["foo", "bar"], pa.string()),
        }
    )
    acq = Acquisitions(wider)
    assert "extra_diagnostic" not in acq.table.column_names
    assert acq.ids == [1, 2]


def test_with_metadata_round_trip() -> None:
    acq = Acquisitions.from_records([_row(acquisition_id=1)])
    acq2 = acq.with_metadata({"x.lab.run_label": "qJS00x"})
    assert acq2.metadata.get("x.lab.run_label") == "qJS00x"


def test_validate_acquisitions_ok() -> None:
    acq = Acquisitions.from_records(
        [_row(acquisition_id=1), _row(acquisition_id=2)]
    )
    fk_table = pa.table({"acquisition_id": pa.array([1, 2, 1], pa.int64())})
    validate_acquisitions(fk_table, acq)  # no raise


def test_validate_acquisitions_unknown_id_raises() -> None:
    acq = Acquisitions.from_records([_row(acquisition_id=1)])
    fk_table = pa.table({"acquisition_id": pa.array([1, 99], pa.int64())})
    with pytest.raises(ValueError, match="unknown acquisition_ids: \\[99\\]"):
        validate_acquisitions(fk_table, acq)


def test_validate_acquisitions_null_handling() -> None:
    acq = Acquisitions.from_records([_row(acquisition_id=1)])
    fk_table = pa.table({"acquisition_id": pa.array([1, None], pa.int64())})
    # Default nullable=False raises on nulls
    with pytest.raises(ValueError, match="contains nulls but nullable=False"):
        validate_acquisitions(fk_table, acq)
    # nullable=True allows them
    validate_acquisitions(fk_table, acq, nullable=True)
