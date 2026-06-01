"""Tests for the XIC target adapters (sources → XIC_TARGET_TABLE)."""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.quant.targets import (
    targets_from_library,
    targets_from_precursor_quant,
    targets_from_search,
    targets_from_table,
)


def _library(rt_predicted=(1800.0, -1.0)):
    precursors = pa.table(
        {
            "precursor_id": pa.array([10, 11], pa.int64()),
            "peptide_id": pa.array([1, 1], pa.int64()),
            "charge": pa.array([2, 3], pa.int32()),
            "precursor_mz": pa.array([500.25, 333.84], pa.float64()),
            "rt_predicted": pa.array(list(rt_predicted), pa.float64()),
            "ccs_predicted": pa.array([-1.0, -1.0], pa.float64()),
        }
    )
    peptides = pa.table(
        {
            "peptide_id": pa.array([1], pa.int64()),
            "sequence": pa.array(["PEPTIDE"]),
            "modified_sequence": pa.array(["PEPTIDE"]),
        }
    )
    return SimpleNamespace(precursors=precursors, peptides=peptides)


def test_from_library_schema_and_rt_sentinel():
    t = targets_from_library(_library())
    assert t.schema.equals(XIC_TARGET_TABLE)
    # -1.0 rt_predicted → null rt_center
    assert t.column("rt_center").to_pylist() == [1800.0, None]
    assert t.column("modified_sequence").to_pylist() == ["PEPTIDE", "PEPTIDE"]
    # library source carries no scan
    assert t.column("scan").null_count == t.num_rows


def test_from_search_carries_scan():
    psms = pa.table(
        {
            "psm_id": pa.array([7], pa.int64()),
            "scan": pa.array([1234], pa.int32()),
            "modified_sequence": pa.array(["PEPK"]),
            "charge": pa.array([2], pa.int8()),
            "mz": pa.array([400.7], pa.float64()),
            "retention_time_s": pa.array([900.0], pa.float64()),
        }
    )
    t = targets_from_search(SimpleNamespace(psms=psms))
    assert t.schema.equals(XIC_TARGET_TABLE)
    assert t.column("scan").to_pylist() == [1234]
    assert t.column("target_id").to_pylist() == [7]
    assert t.column("rt_center").to_pylist() == [900.0]


def test_from_precursor_quant_uses_observed_rt():
    lib = _library()
    pq_tbl = pa.table(
        {
            "precursor_id": pa.array([10], pa.int64()),
            "acquisition_id": pa.array([0], pa.int64()),
            "intensity": pa.array([1e6], pa.float64()),
            "rt_observed": pa.array([1805.0], pa.float64()),
            "ccs_observed": pa.array([-1.0], pa.float64()),
        }
    )
    t = targets_from_precursor_quant(SimpleNamespace(precursor_quant=pq_tbl), lib)
    assert t.column("rt_center").to_pylist() == [1805.0]
    assert t.column("precursor_mz").to_pylist() == [500.25]


def test_from_table_fills_optional_columns():
    bare = pa.table(
        {
            "target_id": pa.array([1, 2], pa.int64()),
            "precursor_mz": pa.array([700.1, 800.2], pa.float64()),
        }
    )
    t = targets_from_table(bare)
    assert t.schema.equals(XIC_TARGET_TABLE)
    assert t.column("modified_sequence").null_count == 2
    assert t.column("rt_center").null_count == 2


def test_from_table_rejects_missing_target_id():
    bad = pa.table({"precursor_mz": pa.array([700.1], pa.float64())})
    with pytest.raises(ValueError):
        targets_from_table(bad)
