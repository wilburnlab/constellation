"""Tests for ``constellation.sequencing.samples``.

PK uniqueness on samples table, edge-side FK closure, the M:N
lookup methods that demux uses to translate ``(acquisition_id,
barcode_id)`` to ``sample_id``, and the ParquetDir persistence
that demux writes for downstream stages to consume without
re-parsing the user's TSV.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.samples import (
    SAMPLE_ACQUISITION_EDGE,
    SAMPLE_TABLE,
    Samples,
    load_samples,
    save_samples,
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


def test_sample_name_uniqueness_raises() -> None:
    """Two SAMPLE_TABLE rows with the same sample_name (but different
    sample_ids) describe two distinct biological samples that happen to
    share a name — downstream code that keys on sample_name (gene-matrix
    columns, fastq filenames) silently corrupts in that state, so
    Samples refuses the construction."""
    with pytest.raises(ValueError, match="appears on multiple sample rows"):
        Samples.from_records(
            samples=[
                {"sample_id": 7, "sample_name": "20M+", "description": None},
                {"sample_id": 9, "sample_name": "20M+", "description": None},
            ],
            edges=[
                {"sample_id": 7, "acquisition_id": 1, "barcode_id": 8},
                {"sample_id": 9, "acquisition_id": 1, "barcode_id": 11},
            ],
        )


def test_sample_name_uniqueness_does_not_reject_multi_edge() -> None:
    """The check is on SAMPLE_TABLE only — a sample legitimately
    contributed to by N barcodes shares ONE sample_id and produces N
    SAMPLE_ACQUISITION_EDGE rows, which must continue to validate."""
    s = Samples.from_records(
        samples=[
            {"sample_id": 1, "sample_name": "20M+", "description": None},
        ],
        edges=[
            {"sample_id": 1, "acquisition_id": 1, "barcode_id": 8},
            {"sample_id": 1, "acquisition_id": 1, "barcode_id": 11},
            {"sample_id": 1, "acquisition_id": 2, "barcode_id": 8},
        ],
    )
    assert s.samples.num_rows == 1
    assert s.edges.num_rows == 3


def test_parquet_dir_roundtrip(tmp_path: Path) -> None:
    """``save_samples`` + ``load_samples`` round-trip both tables byte-
    for-byte so downstream stages get the same container demux
    produced."""
    s = _build_basic()
    samples_dir = tmp_path / "samples"
    save_samples(s, samples_dir)

    assert (samples_dir / "samples.parquet").exists()
    assert (samples_dir / "edges.parquet").exists()
    assert (samples_dir / "manifest.json").exists()

    loaded = load_samples(samples_dir)
    assert loaded.samples.equals(s.samples)
    assert loaded.edges.equals(s.edges)


def test_parse_samples_tsv_auto_assigns_sample_ids(tmp_path: Path) -> None:
    """The TSV loader is the source of auto-assignment: users supply
    sample_name + barcode_id only; sample_ids are assigned 1..N in
    sample_name sort order, and duplicate sample_name rows aggregate
    into one sample with multiple edges."""
    from constellation.cli.__main__ import _parse_samples_tsv

    tsv = tmp_path / "samples.tsv"
    tsv.write_text(
        "sample_name\tbarcode_id\n"
        "20M+\t8\n"
        "10G+\t6\n"
        "20M+\t11\n"  # second row sharing the 20M+ sample_name
    )
    sample_rows, edge_rows, _ = _parse_samples_tsv(
        str(tsv), input_files=[], default_acquisition_id=1
    )
    # Two unique sample_names → two SAMPLE_TABLE rows, sample_ids in
    # sample_name sort order ('10G+' < '20M+').
    assert sample_rows == [
        {"sample_id": 1, "sample_name": "10G+", "description": None},
        {"sample_id": 2, "sample_name": "20M+", "description": None},
    ]
    # The duplicate sample_name produces two edges pointing at the same
    # auto-assigned sample_id — that's the aggregation semantic.
    assert sorted(
        (e["sample_id"], e["barcode_id"]) for e in edge_rows
    ) == [(1, 6), (2, 8), (2, 11)]


def test_parse_samples_tsv_rejects_legacy_sample_id_column(
    tmp_path: Path,
) -> None:
    """The pre-refactor TSV format carried a user-supplied sample_id
    column; that path is gone and the loader points the user at the
    new format."""
    from constellation.cli.__main__ import _parse_samples_tsv

    tsv = tmp_path / "samples.tsv"
    tsv.write_text(
        "sample_id\tsample_name\tbarcode_id\n"
        "7\t20M+\t8\n"
        "9\t20M+\t11\n"
    )
    with pytest.raises(ValueError, match="'sample_id' column"):
        _parse_samples_tsv(str(tsv), input_files=[], default_acquisition_id=1)


def test_parse_samples_tsv_dedups_identical_edges(tmp_path: Path) -> None:
    """An exact duplicate (sample_name, acquisition_id, barcode_id)
    row is dropped with a warning rather than producing two
    indistinguishable edges."""
    from constellation.cli.__main__ import _parse_samples_tsv

    tsv = tmp_path / "samples.tsv"
    tsv.write_text(
        "sample_name\tbarcode_id\n"
        "20M+\t8\n"
        "20M+\t8\n"  # exact duplicate of the row above
    )
    _, edge_rows, _ = _parse_samples_tsv(
        str(tsv), input_files=[], default_acquisition_id=1
    )
    assert len(edge_rows) == 1
