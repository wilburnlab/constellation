"""Tests for ``constellation.sequencing.samples``.

PK uniqueness on samples table, edge-side FK closure, and the M:N
lookup methods that demux uses to translate ``(acquisition_id,
barcode_id)`` to ``sample_id``.
"""

from __future__ import annotations

import pytest

from constellation.sequencing.samples import (
    SAMPLE_ACQUISITION_EDGE,
    SAMPLE_TABLE,
    Samples,
)


def _build_basic() -> Samples:
    """A small mixed panel: BC01/BC02 multiplexed on acq 1; an
    unbarcoded genomic pool on acq 2."""
    return Samples.from_records(
        samples=[
            {"sample_id": 100, "sample_name": "test_simplex_BC01", "description": None},
            {"sample_id": 101, "sample_name": "test_simplex_BC02", "description": None},
            {"sample_id": 200, "sample_name": "genomic_pool", "description": None},
        ],
        edges=[
            {"sample_id": 100, "acquisition_id": 1, "barcode_id": 1},
            {"sample_id": 101, "acquisition_id": 1, "barcode_id": 2},
            {"sample_id": 200, "acquisition_id": 2, "barcode_id": None},
        ],
    )


def test_empty_constructor() -> None:
    s = Samples.empty()
    assert len(s) == 0
    assert s.ids == []
    assert s.samples.schema.equals(SAMPLE_TABLE)
    assert s.edges.schema.equals(SAMPLE_ACQUISITION_EDGE)


def test_from_records_basic() -> None:
    s = _build_basic()
    assert s.ids == [100, 101, 200]
    assert len(s) == 3


def test_pk_uniqueness_raises() -> None:
    with pytest.raises(ValueError, match="sample_id contains duplicates"):
        Samples.from_records(
            samples=[
                {"sample_id": 1, "sample_name": "a", "description": None},
                {"sample_id": 1, "sample_name": "b", "description": None},
            ],
            edges=[],
        )


def test_edge_fk_closure_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="unknown sample_ids: \\[99\\]"):
        Samples.from_records(
            samples=[
                {"sample_id": 1, "sample_name": "a", "description": None},
            ],
            edges=[
                {"sample_id": 99, "acquisition_id": 1, "barcode_id": None},
            ],
        )


def test_samples_for_with_barcode() -> None:
    s = _build_basic()
    assert s.samples_for(1, 1) == [100]
    assert s.samples_for(1, 2) == [101]
    # Acquisition 1 with barcode_id=99 has no sample
    assert s.samples_for(1, 99) == []


def test_samples_for_unbarcoded() -> None:
    s = _build_basic()
    # Genomic pool on acquisition 2 (barcode_id is null)
    assert s.samples_for(2, None) == [200]
    # Acquisition 1 with barcode_id=None has no unbarcoded sample
    assert s.samples_for(1, None) == []


def test_acquisitions_for() -> None:
    s = _build_basic()
    assert s.acquisitions_for(100) == [1]
    assert s.acquisitions_for(200) == [2]
    # Sample with no edges
    assert s.acquisitions_for(99999) == []


def test_multiple_acquisitions_per_sample() -> None:
    """Genomic ultra-HMW: one sample pooled across multiple flowcells."""
    s = Samples.from_records(
        samples=[
            {"sample_id": 1, "sample_name": "hmw_pool", "description": None},
        ],
        edges=[
            {"sample_id": 1, "acquisition_id": 10, "barcode_id": None},
            {"sample_id": 1, "acquisition_id": 11, "barcode_id": None},
            {"sample_id": 1, "acquisition_id": 12, "barcode_id": None},
        ],
    )
    assert sorted(s.acquisitions_for(1)) == [10, 11, 12]
