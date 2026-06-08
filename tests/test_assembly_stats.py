"""Assembly summary statistics — n50 / gc_content / assembly_stats."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.stats import (
    assembly_stats,
    gc_content,
    n50,
)


# ── n50 / l50 ─────────────────────────────────────────────────────────


def test_n50_basic():
    # 6,5,4,3,2 (total 20, half 10); cumsum 6,11 -> first >=10 is 5, l50=2
    assert n50([2, 3, 4, 5, 6]) == (5, 2)


def test_n50_single_contig():
    assert n50([100]) == (100, 1)


def test_n50_all_equal():
    # four 10s (total 40, half 20); cumsum 10,20 -> first >=20 is 10, l50=2
    assert n50([10, 10, 10, 10]) == (10, 2)


def test_n50_empty():
    assert n50([]) == (0, 0)


def test_n50_order_independent():
    assert n50([6, 2, 5, 3, 4]) == n50([2, 3, 4, 5, 6])


# ── gc_content ────────────────────────────────────────────────────────


def _seqs(rows: list[tuple[int, str]]) -> pa.Table:
    return pa.table(
        {
            "contig_id": pa.array([r[0] for r in rows], type=pa.int64()),
            "sequence": [r[1] for r in rows],
        }
    )


def test_gc_content_excludes_n_and_counts_lowercase():
    # c0: G2 C2 N4   c1: A4 T4 (mixed case)  -> gc=4, acgt=12
    seqs = _seqs([(0, "GGCCNNNN"), (1, "ATATatat")])
    assert gc_content(seqs) == pytest.approx(4 / 12)


def test_gc_content_all_n_is_zero():
    assert gc_content(_seqs([(0, "NNNN")])) == 0.0


def test_gc_content_empty_is_zero():
    assert gc_content(_seqs([(0, "")])) == 0.0


# ── assembly_stats ────────────────────────────────────────────────────


def _toy_assembly() -> Assembly:
    contigs = pa.table(
        {
            "contig_id": pa.array([0, 1, 2], type=pa.int64()),
            "name": ["c0", "c1", "c2"],
            "length": pa.array([6, 5, 4], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0, 1, 2], type=pa.int64()),
            "sequence": ["GGCCCC", "ATATA", "ACGT"],
        }
    )
    return Assembly.from_tables(contigs, sequences)


def test_assembly_stats_one_row():
    stats = assembly_stats(_toy_assembly())
    assert stats.num_rows == 1
    row = stats.to_pylist()[0]
    assert row["n_contigs"] == 3
    assert row["total_length"] == 15
    assert row["largest_contig"] == 6
    # total 15, half 7.5; sorted 6,5,4 cumsum 6,11 -> n50=5, l50=2
    assert row["n50"] == 5
    assert row["l50"] == 2
    assert row["n_scaffolds"] is None
    # busco columns null until busco_stats runs
    assert row["busco_complete"] is None
    assert row["busco_lineage"] is None


def test_assembly_stats_gc_matches_helper():
    a = _toy_assembly()
    row = assembly_stats(a).to_pylist()[0]
    assert row["gc_content"] == pytest.approx(gc_content(a.sequences), rel=1e-5)
