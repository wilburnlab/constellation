"""Tests for ``constellation.sequencing.readers.sam_bam.SamReader``.

Covers eager + streaming reads against synthetic SAMs (tight invariants)
and a smoke check against the lab's `test_simplex.sam` fixture (loose
invariants — schema match, row count, first-row sanity).
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.readers.sam_bam import BamReader, SamReader
from constellation.sequencing.schemas.reads import READ_TABLE


# Synthetic minimal SAM: 1 header line + 3 records. Two carry full
# Dorado-style optional tags (ch/st/du); one is bare (just the 11
# fixed fields).
_SAMPLE_SAM = (
    "@HD\tVN:1.6\tSO:unknown\n"
    "@PG\tID:test\tPN:synthetic\n"
    "read-001\t4\t*\t0\t0\t*\t*\t0\t0\tACGTACGTAC\t!!!!!!!!!!"
    "\tch:i:14\tst:Z:2024-06-01T12:00:00.000000+00:00\tdu:f:0.5\n"
    "read-002\t4\t*\t0\t0\t*\t*\t0\t0\tNNNNN\t#####"
    "\tch:i:7\tdu:f:0.25\n"
    "read-003\t4\t*\t0\t0\t*\t*\t0\t0\tA\t*\n"  # quality '*' = missing
)


@pytest.fixture
def sam_path(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic.sam"
    p.write_text(_SAMPLE_SAM)
    return p


# ──────────────────────────────────────────────────────────────────────
# Eager read
# ──────────────────────────────────────────────────────────────────────


def test_read_returns_read_table_schema(sam_path: Path) -> None:
    reader = SamReader()
    result = reader.read(sam_path, acquisition_id=42)
    table = result.primary
    assert table.schema.equals(READ_TABLE)


def test_read_row_count(sam_path: Path) -> None:
    reader = SamReader()
    result = reader.read(sam_path, acquisition_id=42)
    assert result.primary.num_rows == 3


def test_read_acquisition_id_injected(sam_path: Path) -> None:
    reader = SamReader()
    result = reader.read(sam_path, acquisition_id=42)
    assert result.primary.column("acquisition_id").to_pylist() == [42, 42, 42]


def test_read_fields_populated(sam_path: Path) -> None:
    reader = SamReader()
    rows = reader.read(sam_path, acquisition_id=1).primary.to_pylist()

    r0 = rows[0]
    assert r0["read_id"] == "read-001"
    assert r0["sequence"] == "ACGTACGTAC"
    assert r0["quality"] == "!!!!!!!!!!"
    assert r0["length"] == 10
    # mean(ord('!')) - 33 == 33 - 33 == 0
    assert r0["mean_quality"] == pytest.approx(0.0)
    assert r0["channel"] == 14
    assert r0["duration_s"] == pytest.approx(0.5)
    # st:Z parses to a unix timestamp (>0); exact value depends on TZ
    # math, but we know the prefix is 2024 → roughly 1.7e9
    assert r0["start_time_s"] is not None
    assert 1.7e9 < r0["start_time_s"] < 1.8e9

    r1 = rows[1]
    assert r1["channel"] == 7
    assert r1["start_time_s"] is None  # no st:Z tag
    assert r1["duration_s"] == pytest.approx(0.25)
    # mean(ord('#')) - 33 == 35 - 33 == 2
    assert r1["mean_quality"] == pytest.approx(2.0)

    r2 = rows[2]
    assert r2["quality"] is None  # SAM '*' translates to NULL
    assert r2["mean_quality"] is None
    assert r2["channel"] is None
    assert r2["duration_s"] is None


def test_read_skips_header_lines(tmp_path: Path) -> None:
    """Many header lines, no body — should yield empty table."""
    p = tmp_path / "headers_only.sam"
    p.write_text("@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:100\n@PG\tID:x\n")
    table = SamReader().read(p).primary
    assert table.num_rows == 0
    assert table.schema.equals(READ_TABLE)


def test_read_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.sam"
    p.write_text("")
    table = SamReader().read(p).primary
    assert table.num_rows == 0


def test_read_run_metadata_populated(sam_path: Path) -> None:
    result = SamReader().read(sam_path)
    assert result.run_metadata["source_kind"] == "sam"
    assert result.run_metadata["source_path"] == str(sam_path)


# ──────────────────────────────────────────────────────────────────────
# iter_batches
# ──────────────────────────────────────────────────────────────────────


def test_iter_batches_chunk_boundary(sam_path: Path) -> None:
    """3 records, batch_size=2 → batches of [2, 1]."""
    reader = SamReader()
    batches = list(reader.iter_batches(sam_path, batch_size=2, acquisition_id=1))
    assert [b.num_rows for b in batches] == [2, 1]
    assert all(b.schema.equals(READ_TABLE) for b in batches)


def test_iter_batches_total_matches_eager(sam_path: Path) -> None:
    reader = SamReader()
    eager = reader.read(sam_path, acquisition_id=1).primary
    streamed = pa.concat_tables(
        list(reader.iter_batches(sam_path, batch_size=2, acquisition_id=1))
    )
    assert eager.num_rows == streamed.num_rows
    # Row-by-row equality (acquisition_id, read_id, length should all match)
    assert eager.column("read_id").to_pylist() == streamed.column("read_id").to_pylist()
    assert eager.column("length").to_pylist() == streamed.column("length").to_pylist()


def test_iter_batches_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.sam"
    p.write_text("")
    batches = list(SamReader().iter_batches(p, batch_size=10))
    assert batches == []


def test_iter_batches_invalid_size(sam_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        list(SamReader().iter_batches(sam_path, batch_size=0))


def test_iter_batches_exact_multiple(tmp_path: Path) -> None:
    """4 records, batch_size=2 → batches of [2, 2]; no trailing empty."""
    p = tmp_path / "four.sam"
    body = "\n".join(
        f"r{i}\t4\t*\t0\t0\t*\t*\t0\t0\tA\tI" for i in range(4)
    )
    p.write_text("@HD\tVN:1.6\n" + body + "\n")
    batches = list(SamReader().iter_batches(p, batch_size=2))
    assert [b.num_rows for b in batches] == [2, 2]


# ──────────────────────────────────────────────────────────────────────
# Real fixture
# ──────────────────────────────────────────────────────────────────────


_FIXTURE_SAM = Path(
    "/home/dbwilburn/WilburnLab/Coding/Constellation/test_nanopore_data/"
    "pod5_cDNA/Constellation_test_data/test_simplex.sam"
)


@pytest.mark.skipif(
    not _FIXTURE_SAM.exists(), reason="lab test_simplex.sam fixture not present"
)
def test_real_fixture_row_count_matches_baseline() -> None:
    """test_simplex.sam should produce 4009 reads — the same row count
    as the supplied NanoporeAnalysis baseline parquet."""
    table = SamReader().read(_FIXTURE_SAM, acquisition_id=1).primary
    assert table.num_rows == 4009


@pytest.mark.skipif(
    not _FIXTURE_SAM.exists(), reason="lab test_simplex.sam fixture not present"
)
def test_real_fixture_first_read_sanity() -> None:
    """First record should have a UUID-like read_id, non-empty sequence,
    and matching quality length."""
    table = SamReader().read(_FIXTURE_SAM, acquisition_id=1).primary
    first = table.slice(0, 1).to_pylist()[0]
    assert len(first["read_id"]) == 36  # ONT UUID
    assert "-" in first["read_id"]
    assert len(first["sequence"]) == first["length"]
    assert len(first["quality"]) == first["length"]
    assert first["channel"] is not None
    assert first["mean_quality"] > 0


# ──────────────────────────────────────────────────────────────────────
# BamReader
# ──────────────────────────────────────────────────────────────────────


def test_bam_reader_stub_raises(tmp_path: Path) -> None:
    """BamReader is not in S1; calling .read() should raise
    NotImplementedError clearly."""
    p = tmp_path / "fake.bam"
    p.write_bytes(b"")
    with pytest.raises(NotImplementedError, match="BamReader is not yet implemented"):
        BamReader().read(p)
