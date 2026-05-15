"""Unit tests for the streaming resolve-stage aggregator.

``aggregate_partial_quants`` folds the per-worker ``feature_quant/``
shards into the global Protein × Sample matrix. It used to materialise
the whole partitioned dataset via ``ds.dataset(...).to_table()``, which
segfaults under pyarrow 22.0.0 (``to_pylist()`` on a large string
column). It now streams the shards one at a time. These tests pin the
behaviour that matters across that change: counts sum across shards,
ordering / P-labels are count-descending, and the singleton filter
applies on the *global* per-protein total.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.samples import (
    SAMPLE_ACQUISITION_EDGE,
    SAMPLE_TABLE,
    Samples,
)
from constellation.sequencing.transcriptome.quant import (
    PARTIAL_QUANT_TABLE,
    aggregate_partial_quants,
)


def _samples(names: dict[int, str]) -> Samples:
    """A minimal Samples container — only sample_id/sample_name are read
    by the aggregator; edges stay empty (FK closure trivially holds)."""
    sample_tbl = pa.table(
        {
            "sample_id": list(names.keys()),
            "sample_name": list(names.values()),
            "description": [None] * len(names),
        },
        schema=SAMPLE_TABLE,
    )
    return Samples(sample_tbl, SAMPLE_ACQUISITION_EDGE.empty_table())


def _write_shard(path, rows: list[tuple[str, int, int]]) -> None:
    """Write one PARTIAL_QUANT_TABLE shard from (protein, sample, count)."""
    tbl = pa.table(
        {
            "protein_sequence": [r[0] for r in rows],
            "sample_id": [r[1] for r in rows],
            "count": [r[2] for r in rows],
        },
        schema=PARTIAL_QUANT_TABLE,
    )
    pq.write_table(tbl, path)


def test_counts_sum_across_shards(tmp_path):
    """The same (protein, sample) pair split across shards is summed."""
    fq = tmp_path / "feature_quant"
    fq.mkdir()
    # "AAA"/sample 1 appears in both shards: 3 + 2 -> 5.
    _write_shard(fq / "part-00000.parquet", [("AAA", 1, 3), ("BBB", 1, 1)])
    _write_shard(fq / "part-00001.parquet", [("AAA", 1, 2), ("AAA", 2, 5)])

    samples = _samples({1: "sample_1", 2: "sample_2"})
    quant, fasta, tsv = aggregate_partial_quants(
        fq, samples, min_protein_count=0
    )

    cells = {
        (r["protein_sequence"], r["sample_id"]): r["count"]
        for r in quant.to_pylist()
    }
    assert cells[("AAA", 1)] == 5
    assert cells[("AAA", 2)] == 5
    assert cells[("BBB", 1)] == 1


def test_singleton_filter_uses_global_total(tmp_path):
    """min_protein_count drops a protein whose per-shard counts are each
    below threshold but whose global total clears it — and keeps nothing
    that doesn't."""
    fq = tmp_path / "feature_quant"
    fq.mkdir()
    # "AAA" total = 1 + 1 = 2 (kept at min=2); "BBB" total = 1 (dropped).
    _write_shard(fq / "part-00000.parquet", [("AAA", 1, 1), ("BBB", 1, 1)])
    _write_shard(fq / "part-00001.parquet", [("AAA", 2, 1)])

    samples = _samples({1: "sample_1", 2: "sample_2"})
    quant, fasta, _ = aggregate_partial_quants(fq, samples, min_protein_count=2)

    kept = {r["protein_sequence"] for r in quant.to_pylist()}
    assert kept == {"AAA"}
    assert [f.sequence for f in fasta] == ["AAA"]


def test_protein_labels_are_count_descending(tmp_path):
    """P0..PN labels track global count, descending."""
    fq = tmp_path / "feature_quant"
    fq.mkdir()
    _write_shard(
        fq / "part-00000.parquet",
        [("LOW", 1, 1), ("HIGH", 1, 10), ("MID", 1, 5)],
    )
    _write_shard(fq / "part-00001.parquet", [("LOW", 2, 1), ("MID", 2, 1)])

    samples = _samples({1: "sample_1", 2: "sample_2"})
    quant, fasta, _ = aggregate_partial_quants(fq, samples, min_protein_count=0)

    label_by_seq = {
        r["protein_sequence"]: r["protein_label"] for r in quant.to_pylist()
    }
    assert label_by_seq["HIGH"] == "P0"  # total 10
    assert label_by_seq["MID"] == "P1"   # total 6
    assert label_by_seq["LOW"] == "P2"   # total 2
    assert [f.label for f in fasta] == ["P0", "P1", "P2"]


def test_empty_feature_quant_dir(tmp_path):
    """An empty feature_quant/ dir yields empty outputs, no crash."""
    fq = tmp_path / "feature_quant"
    fq.mkdir()
    samples = _samples({1: "sample_1"})
    quant, fasta, tsv = aggregate_partial_quants(fq, samples)
    assert quant.num_rows == 0
    assert fasta == []


def test_sample_name_resolution(tmp_path):
    """sample_id -> sample_name comes from the Samples container."""
    fq = tmp_path / "feature_quant"
    fq.mkdir()
    _write_shard(fq / "part-00000.parquet", [("AAA", 1, 2), ("AAA", 7, 3)])

    samples = _samples({1: "HeLa_rep1", 7: "HEK_rep2"})
    quant, _, _ = aggregate_partial_quants(fq, samples, min_protein_count=0)

    name_by_id = {
        r["sample_id"]: r["sample_name"] for r in quant.to_pylist()
    }
    assert name_by_id == {1: "HeLa_rep1", 7: "HEK_rep2"}
