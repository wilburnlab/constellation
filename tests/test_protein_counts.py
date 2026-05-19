"""Tier A tests for :mod:`constellation.sequencing.quant.protein_counts`.

Synthetic wide TSV + long parquet fixtures. Covers:

  * ``read_protein_counts_tab`` on wide TSV (cartographer-compat)
  * ``read_protein_counts_tab`` on long parquet (Constellation-native
    PROTEIN_COUNT_TABLE)
  * ``read_protein_counts_tab`` on a directory containing either layout
  * ``tpm_normalize`` math against hand-computed values
  * ``min_sequence_length`` filter behaviour (rows dropped, denominator
    excludes them)
  * Edge cases: empty matrix, single-sample, all-short-sequences
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.quant.protein_counts import (
    PROTEIN_COUNTS_LONG_SCHEMA,
    read_protein_counts_tab,
    tpm_normalize,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


_WIDE_TSV = (
    "\tProtein\tqJS001\tqJS002\tqJS003\tSequence\n"
    "0\tP0\t5.0\t3.0\t0.0\tMKLIGHTPEPTIDESEQUENCEAA\n"     # len 23
    "1\tP1\t2.0\t10.0\t1.0\t" + "ACDEFGHIKLMNPQRSTVWY" * 6 + "\n"  # 120 AA
    "2\tP2\t0.0\t0.0\t7.0\tSHORTSEQ\n"                    # 8 AA (short)
)


@pytest.fixture
def wide_tsv(tmp_path) -> Path:
    p = tmp_path / "protein_counts.tsv"
    p.write_text(_WIDE_TSV)
    return p


@pytest.fixture
def long_parquet(tmp_path) -> Path:
    """PROTEIN_COUNT_TABLE-shaped parquet (the demux-output native form)."""
    table = pa.table(
        {
            "protein_label": ["P0", "P0", "P1", "P1", "P2"],
            "protein_sequence": [
                "MKLIGHTPEPTIDESEQUENCEAA",
                "MKLIGHTPEPTIDESEQUENCEAA",
                "ACDEFGHIKLMNPQRSTVWY" * 6,
                "ACDEFGHIKLMNPQRSTVWY" * 6,
                "SHORTSEQ",
            ],
            "sample_id": [0, 1, 0, 1, 2],
            "sample_name": ["qJS001", "qJS002", "qJS001", "qJS002", "qJS003"],
            "count": [5, 3, 2, 10, 7],
        }
    )
    p = tmp_path / "feature_quant.parquet"
    pq.write_table(table, p)
    return p


# ──────────────────────────────────────────────────────────────────────
# read_protein_counts_tab — wide TSV path
# ──────────────────────────────────────────────────────────────────────


def test_read_wide_tsv_long_shape(wide_tsv) -> None:
    table = read_protein_counts_tab(wide_tsv)
    assert table.schema == PROTEIN_COUNTS_LONG_SCHEMA
    # 3 proteins × 3 samples = 9 rows (zeros included).
    assert table.num_rows == 9
    rows = {
        (r["protein_id"], r["sample_name"]): r["count"]
        for r in table.to_pylist()
    }
    assert rows[("P0", "qJS001")] == 5
    assert rows[("P0", "qJS002")] == 3
    assert rows[("P0", "qJS003")] == 0
    assert rows[("P1", "qJS002")] == 10
    assert rows[("P2", "qJS003")] == 7


def test_read_wide_tsv_preserves_sequence(wide_tsv) -> None:
    table = read_protein_counts_tab(wide_tsv)
    by_id = {r["protein_id"]: r["sequence"] for r in table.to_pylist()}
    assert by_id["P0"] == "MKLIGHTPEPTIDESEQUENCEAA"
    assert by_id["P2"] == "SHORTSEQ"


def test_read_wide_tsv_no_index_column(tmp_path) -> None:
    """Some non-pandas writers omit the leading unnamed index column."""
    p = tmp_path / "no_index.tsv"
    p.write_text(
        "Protein\tqJS001\tSequence\n"
        "P0\t5.0\tMKLIGHTPEPTIDESEQUENCEAA\n"
        "P1\t2.0\tACDEFGHIK\n"
    )
    table = read_protein_counts_tab(p)
    assert table.num_rows == 2


def test_read_wide_tsv_missing_required_column(tmp_path) -> None:
    p = tmp_path / "bad.tsv"
    p.write_text("Protein\tqJS001\n" "P0\t5.0\n")  # no Sequence column
    with pytest.raises(ValueError, match="missing required columns"):
        read_protein_counts_tab(p)


# ──────────────────────────────────────────────────────────────────────
# read_protein_counts_tab — long parquet path
# ──────────────────────────────────────────────────────────────────────


def test_read_long_parquet_long_shape(long_parquet) -> None:
    table = read_protein_counts_tab(long_parquet)
    assert table.schema == PROTEIN_COUNTS_LONG_SCHEMA
    assert table.num_rows == 5  # only non-zero pairs in the long parquet
    rows = {
        (r["protein_id"], r["sample_name"]): r["count"]
        for r in table.to_pylist()
    }
    assert rows[("P0", "qJS001")] == 5
    assert rows[("P0", "qJS002")] == 3
    assert rows[("P1", "qJS001")] == 2
    assert rows[("P1", "qJS002")] == 10
    assert rows[("P2", "qJS003")] == 7


# ──────────────────────────────────────────────────────────────────────
# read_protein_counts_tab — directory sniff
# ──────────────────────────────────────────────────────────────────────


def test_read_directory_prefers_parquet(tmp_path, wide_tsv, long_parquet) -> None:
    """When both feature_quant.parquet and protein_counts.tsv exist,
    parquet wins (long format is more authoritative)."""
    d = tmp_path / "demux_out"
    d.mkdir()
    # Stage both: copy parquet + TSV under d/.
    import shutil
    shutil.copy(long_parquet, d / "feature_quant.parquet")
    shutil.copy(wide_tsv, d / "protein_counts.tsv")
    table = read_protein_counts_tab(d)
    # Parquet has 5 non-zero rows; TSV expands to 9 rows (zeros included).
    assert table.num_rows == 5


def test_read_directory_falls_back_to_tsv(tmp_path, wide_tsv) -> None:
    d = tmp_path / "demux_out_tsv_only"
    d.mkdir()
    import shutil
    shutil.copy(wide_tsv, d / "protein_counts.tsv")
    table = read_protein_counts_tab(d)
    assert table.num_rows == 9


def test_read_directory_no_artefact(tmp_path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="no protein-counts artefact"):
        read_protein_counts_tab(d)


def test_read_unsupported_extension(tmp_path) -> None:
    p = tmp_path / "bogus.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="unsupported protein-counts input"):
        read_protein_counts_tab(p)


# ──────────────────────────────────────────────────────────────────────
# tpm_normalize
# ──────────────────────────────────────────────────────────────────────


def test_tpm_normalize_basic_math() -> None:
    """Hand-computed values: per-sample TPM = count * 1e6 / sample_total."""
    table = pa.table(
        {
            "protein_id": ["P0", "P0", "P1", "P1"],
            "sequence": ["A" * 120, "A" * 120, "B" * 120, "B" * 120],
            "sample_name": ["s1", "s2", "s1", "s2"],
            "count": [3, 1, 7, 9],
        },
        schema=PROTEIN_COUNTS_LONG_SCHEMA,
    )
    out = tpm_normalize(table, min_sequence_length=None)
    assert "tpm" in out.column_names
    rows = {
        (r["protein_id"], r["sample_name"]): r["tpm"]
        for r in out.to_pylist()
    }
    # s1 total = 3 + 7 = 10
    assert rows[("P0", "s1")] == pytest.approx(3 * 1e6 / 10)
    assert rows[("P1", "s1")] == pytest.approx(7 * 1e6 / 10)
    # s2 total = 1 + 9 = 10
    assert rows[("P0", "s2")] == pytest.approx(1 * 1e6 / 10)
    assert rows[("P1", "s2")] == pytest.approx(9 * 1e6 / 10)


def test_tpm_normalize_min_sequence_length_filter() -> None:
    """Rows below ``min_sequence_length`` are dropped AND not in the
    per-sample sum denominator."""
    table = pa.table(
        {
            "protein_id": ["P0", "P1"],
            "sequence": ["A" * 120, "B" * 50],   # P1 below threshold (100)
            "sample_name": ["s1", "s1"],
            "count": [10, 90],
        },
        schema=PROTEIN_COUNTS_LONG_SCHEMA,
    )
    out = tpm_normalize(table, min_sequence_length=100)
    assert out.num_rows == 1
    row = out.to_pylist()[0]
    assert row["protein_id"] == "P0"
    # Denominator excludes P1, so total = 10; TPM = 10 * 1e6 / 10 = 1e6.
    assert row["tpm"] == pytest.approx(1e6)


def test_tpm_normalize_empty_input() -> None:
    table = PROTEIN_COUNTS_LONG_SCHEMA.empty_table()
    out = tpm_normalize(table)
    assert out.num_rows == 0
    assert "tpm" in out.column_names


def test_tpm_normalize_all_filtered_out() -> None:
    """If every row is below ``min_sequence_length``, output is empty
    with the tpm column present."""
    table = pa.table(
        {
            "protein_id": ["P0", "P1"],
            "sequence": ["short", "tiny"],
            "sample_name": ["s1", "s2"],
            "count": [5, 3],
        },
        schema=PROTEIN_COUNTS_LONG_SCHEMA,
    )
    out = tpm_normalize(table, min_sequence_length=100)
    assert out.num_rows == 0
    assert "tpm" in out.column_names


def test_tpm_normalize_single_sample() -> None:
    table = pa.table(
        {
            "protein_id": ["P0", "P1", "P2"],
            "sequence": ["A" * 120, "B" * 120, "C" * 120],
            "sample_name": ["s1", "s1", "s1"],
            "count": [2, 3, 5],
        },
        schema=PROTEIN_COUNTS_LONG_SCHEMA,
    )
    out = tpm_normalize(table, min_sequence_length=None)
    rows = {r["protein_id"]: r["tpm"] for r in out.to_pylist()}
    # Total = 10
    assert rows["P0"] == pytest.approx(2 * 1e6 / 10)
    assert rows["P1"] == pytest.approx(3 * 1e6 / 10)
    assert rows["P2"] == pytest.approx(5 * 1e6 / 10)


# ──────────────────────────────────────────────────────────────────────
# End-to-end: read → normalize
# ──────────────────────────────────────────────────────────────────────


def test_read_wide_then_tpm(wide_tsv) -> None:
    """Wide TSV → read → TPM. Verifies the full Stage 1 surface in one
    pass: short ORFs (P2) drop, the rest contribute to per-sample
    denominators."""
    table = read_protein_counts_tab(wide_tsv)
    out = tpm_normalize(table, min_sequence_length=100)
    # Only P1 (120 AA) survives the length filter; P0 (23 AA) + P2 (8 AA) drop.
    assert set(r["protein_id"] for r in out.to_pylist()) == {"P1"}
    rows = {r["sample_name"]: r["tpm"] for r in out.to_pylist()}
    # P1 is the only row per sample; TPM = 1e6 for every non-zero sample.
    # s1: count 2 → 1e6;  s2: count 10 → 1e6;  s3: count 1 → 1e6.
    assert rows["qJS001"] == pytest.approx(1e6)
    assert rows["qJS002"] == pytest.approx(1e6)
    assert rows["qJS003"] == pytest.approx(1e6)
