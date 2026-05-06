"""Tests for ``constellation.sequencing.quant``.

Covers:
  - apply_filter_predicates kernel (audit, edge cases)
  - compute_gene_overlap kernel (sense/antisense, no-overlap, two-gene
    span, gene-boundary cases)
  - count_reads_per_gene aggregator (hash-join, group_by-sum, FK validation)
  - serialise_gene_set + fused_decode_filter_overlap_worker IPC contract
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.quant import (
    apply_filter_predicates,
    compute_gene_overlap,
    count_reads_per_gene,
    gene_set_from_annotation,
    serialise_gene_set,
)
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.samples import Samples


def _alignment_row(**overrides):
    base = {
        "alignment_id": 0,
        "read_id": "r0",
        "acquisition_id": 1,
        "ref_name": "chr1",
        "ref_start": 100,
        "ref_end": 200,
        "strand": "+",
        "mapq": 60,
        "flag": 0,
        "cigar_string": "100M",
        "nm_tag": None,
        "as_tag": None,
        "read_group": None,
        "is_secondary": False,
        "is_supplementary": False,
    }
    base.update(overrides)
    return base


def _alignment_table(rows):
    return pa.Table.from_pylist(rows)


# ──────────────────────────────────────────────────────────────────────
# apply_filter_predicates
# ──────────────────────────────────────────────────────────────────────


def test_filter_audit_records_each_stage() -> None:
    rows = [
        _alignment_row(alignment_id=0, mapq=60, ref_start=100, ref_end=500),
        _alignment_row(alignment_id=1, mapq=10, ref_start=100, ref_end=500),
        _alignment_row(alignment_id=2, mapq=60, ref_start=100, ref_end=110),
    ]
    out, audit = apply_filter_predicates(
        _alignment_table(rows), min_length=100, min_mapq=30
    )
    assert out.num_rows == 1
    assert audit == [
        {"stage": "min_length>=100", "kept": 2, "dropped": 1},
        {"stage": "min_mapq>=30", "kept": 1, "dropped": 1},
    ]


def test_filter_no_predicates_passes_all() -> None:
    rows = [_alignment_row(alignment_id=i) for i in range(5)]
    out, audit = apply_filter_predicates(_alignment_table(rows))
    assert out.num_rows == 5
    assert audit == []


# ──────────────────────────────────────────────────────────────────────
# compute_gene_overlap
# ──────────────────────────────────────────────────────────────────────


def _gene_set(rows):
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("gene_id", pa.int64(), nullable=False),
                pa.field("contig_name", pa.string(), nullable=False),
                pa.field("start", pa.int64(), nullable=False),
                pa.field("end", pa.int64(), nullable=False),
                pa.field("strand", pa.string(), nullable=False),
            ]
        ),
    )


def test_overlap_assigns_best_gene() -> None:
    # Alignment 100..200 overlaps gene 100 (start 50..250) more than gene 200 (start 180..300).
    rows = [_alignment_row(alignment_id=0, ref_start=100, ref_end=200, strand="+")]
    genes = _gene_set(
        [
            {"gene_id": 100, "contig_name": "chr1", "start": 50, "end": 250, "strand": "+"},
            {"gene_id": 200, "contig_name": "chr1", "start": 180, "end": 300, "strand": "+"},
        ]
    )
    out = compute_gene_overlap(_alignment_table(rows), genes)
    assert out.column("gene_id").to_pylist() == [100]
    assert out.column("overlap_fraction").to_pylist()[0] == pytest.approx(1.0)


def test_overlap_drops_below_threshold() -> None:
    # Alignment 0..100 overlaps gene only at the last 10 bp → fraction=0.1
    rows = [_alignment_row(alignment_id=0, ref_start=0, ref_end=100, strand="+")]
    genes = _gene_set(
        [{"gene_id": 100, "contig_name": "chr1", "start": 90, "end": 200, "strand": "+"}]
    )
    out = compute_gene_overlap(_alignment_table(rows), genes, min_overlap_fraction=0.5)
    assert out.num_rows == 0
    out = compute_gene_overlap(_alignment_table(rows), genes, min_overlap_fraction=0.05)
    assert out.num_rows == 1


def test_overlap_strand_filter() -> None:
    rows = [_alignment_row(alignment_id=0, strand="-")]
    genes = _gene_set(
        [{"gene_id": 100, "contig_name": "chr1", "start": 50, "end": 250, "strand": "+"}]
    )
    out = compute_gene_overlap(_alignment_table(rows), genes, allow_antisense=False)
    assert out.num_rows == 0
    out = compute_gene_overlap(_alignment_table(rows), genes, allow_antisense=True)
    assert out.num_rows == 1


def test_overlap_skips_alignments_with_no_gene_on_contig() -> None:
    rows = [_alignment_row(alignment_id=0, ref_name="chrZ")]
    genes = _gene_set(
        [{"gene_id": 100, "contig_name": "chr1", "start": 50, "end": 250, "strand": "+"}]
    )
    out = compute_gene_overlap(_alignment_table(rows), genes)
    assert out.num_rows == 0


def test_overlap_gene_boundary_fully_inside() -> None:
    # Alignment 100..200 fully inside gene 50..250 → fraction = 1.0
    rows = [_alignment_row(alignment_id=0, ref_start=100, ref_end=200, strand="+")]
    genes = _gene_set(
        [{"gene_id": 100, "contig_name": "chr1", "start": 50, "end": 250, "strand": "+"}]
    )
    out = compute_gene_overlap(_alignment_table(rows), genes)
    assert out.column("overlap_fraction").to_pylist()[0] == pytest.approx(1.0)


def test_overlap_unstranded_gene_passes_either_strand() -> None:
    rows = [
        _alignment_row(alignment_id=0, strand="+"),
        _alignment_row(alignment_id=1, strand="-"),
    ]
    genes = _gene_set(
        [{"gene_id": 100, "contig_name": "chr1", "start": 50, "end": 250, "strand": "."}]
    )
    out = compute_gene_overlap(_alignment_table(rows), genes, allow_antisense=False)
    assert out.num_rows == 2


# ──────────────────────────────────────────────────────────────────────
# count_reads_per_gene
# ──────────────────────────────────────────────────────────────────────


def _samples():
    return Samples.from_records(
        samples=[
            {"sample_id": 1, "sample_name": "sA", "description": None},
            {"sample_id": 2, "sample_name": "sB", "description": None},
        ],
        edges=[{"sample_id": 1, "acquisition_id": 1, "barcode_id": None}],
    )


def test_count_reads_per_gene_basic() -> None:
    gene_assignments = pa.Table.from_pylist(
        [
            {"alignment_id": 0, "read_id": "r0", "gene_id": 100, "overlap_fraction": 1.0},
            {"alignment_id": 1, "read_id": "r1", "gene_id": 100, "overlap_fraction": 0.9},
            {"alignment_id": 2, "read_id": "r2", "gene_id": 100, "overlap_fraction": 0.8},
            {"alignment_id": 3, "read_id": "r3", "gene_id": 200, "overlap_fraction": 1.0},
        ]
    )
    read_demux = pa.Table.from_pylist(
        [
            {"read_id": "r0", "sample_id": 1},
            {"read_id": "r1", "sample_id": 1},
            {"read_id": "r2", "sample_id": 2},
            {"read_id": "r3", "sample_id": 1},
        ]
    )
    quant, stats = count_reads_per_gene(gene_assignments, read_demux, _samples())
    assert stats["reads_with_sample"] == 4
    assert stats["reads_without_sample"] == 0
    assert stats["unique_(gene,sample)_pairs"] == 3
    assert stats["total_count"] == 4

    rows = sorted(
        quant.to_pylist(),
        key=lambda r: (r["feature_id"], r["sample_id"]),
    )
    assert [(r["feature_id"], r["sample_id"], r["count"]) for r in rows] == [
        (100, 1, 2.0),
        (100, 2, 1.0),
        (200, 1, 1.0),
    ]
    assert all(r["feature_origin"] == "gene_id" for r in rows)
    assert all(r["engine"] == "constellation_overlap" for r in rows)


def test_count_reads_per_gene_drops_unassigned_reads() -> None:
    gene_assignments = pa.Table.from_pylist(
        [
            {"alignment_id": 0, "read_id": "r0", "gene_id": 100, "overlap_fraction": 1.0},
            {"alignment_id": 1, "read_id": "r99", "gene_id": 200, "overlap_fraction": 1.0},
        ]
    )
    read_demux = pa.Table.from_pylist(
        [{"read_id": "r0", "sample_id": 1}]
    )
    _, stats = count_reads_per_gene(gene_assignments, read_demux, _samples())
    assert stats["reads_with_sample"] == 1
    assert stats["reads_without_sample"] == 1


def test_count_reads_per_gene_rejects_unknown_sample_id() -> None:
    gene_assignments = pa.Table.from_pylist(
        [
            {"alignment_id": 0, "read_id": "r0", "gene_id": 100, "overlap_fraction": 1.0},
        ]
    )
    read_demux = pa.Table.from_pylist(
        [{"read_id": "r0", "sample_id": 999}]
    )
    with pytest.raises(ValueError, match="absent from Samples"):
        count_reads_per_gene(gene_assignments, read_demux, _samples())


def test_count_reads_per_gene_populates_tpm_per_sample_sums_to_1e6() -> None:
    """Each sample's TPM column sums to 1e6 (long-read depth-only formula)."""
    gene_assignments = pa.Table.from_pylist(
        [
            {"alignment_id": i, "read_id": f"r{i}", "gene_id": gid, "overlap_fraction": 1.0}
            for i, gid in enumerate(
                # sample 1: 10 reads on gene 100, 30 reads on gene 200, 60 reads on gene 300
                # sample 2: 70 reads on gene 100, 30 reads on gene 200
                [100] * 10 + [200] * 30 + [300] * 60 + [100] * 70 + [200] * 30
            )
        ]
    )
    read_demux = pa.Table.from_pylist(
        [{"read_id": f"r{i}", "sample_id": (1 if i < 100 else 2)} for i in range(200)]
    )
    quant, stats = count_reads_per_gene(gene_assignments, read_demux, _samples())

    assert stats["samples_normalised"] == 2
    by_sample: dict[int, float] = {}
    for row in quant.to_pylist():
        by_sample[row["sample_id"]] = by_sample.get(row["sample_id"], 0.0) + row["tpm"]
    for sid, total in by_sample.items():
        assert total == pytest.approx(1e6, rel=1e-6), (
            f"sample {sid} TPMs sum to {total}, expected 1e6"
        )

    # Verify a specific TPM: gene 100 in sample 1 has 10/100 reads → 100_000 TPM
    gene_100_sample_1 = next(
        r for r in quant.to_pylist() if r["feature_id"] == 100 and r["sample_id"] == 1
    )
    assert gene_100_sample_1["tpm"] == pytest.approx(1e5)


def test_count_reads_per_gene_tpm_no_length_dependence() -> None:
    """Equal counts on different genes produce equal TPM regardless of any
    notional gene length. Documents the long-read formula choice — the
    function deliberately does not consult an Annotation for length."""
    gene_assignments = pa.Table.from_pylist(
        [
            {"alignment_id": 0, "read_id": "r0", "gene_id": 100, "overlap_fraction": 1.0},
            {"alignment_id": 1, "read_id": "r1", "gene_id": 200, "overlap_fraction": 1.0},
        ]
    )
    read_demux = pa.Table.from_pylist(
        [{"read_id": "r0", "sample_id": 1}, {"read_id": "r1", "sample_id": 1}]
    )
    quant, _ = count_reads_per_gene(gene_assignments, read_demux, _samples())
    tpms = sorted(r["tpm"] for r in quant.to_pylist())
    # Both genes have count=1, sample total=2 → TPM = 5e5 each
    assert tpms == pytest.approx([5e5, 5e5])


def test_count_reads_per_gene_cpm_left_null() -> None:
    """`cpm` stays null per the long-read regime (CPM ≡ TPM here)."""
    gene_assignments = pa.Table.from_pylist(
        [{"alignment_id": 0, "read_id": "r0", "gene_id": 100, "overlap_fraction": 1.0}]
    )
    read_demux = pa.Table.from_pylist([{"read_id": "r0", "sample_id": 1}])
    quant, _ = count_reads_per_gene(gene_assignments, read_demux, _samples())
    assert quant.to_pylist()[0]["cpm"] is None


def test_count_reads_per_gene_handles_empty_inputs() -> None:
    empty_assignments = pa.Table.from_pylist(
        [],
        schema=pa.schema(
            [
                pa.field("alignment_id", pa.int64(), nullable=False),
                pa.field("read_id", pa.string(), nullable=False),
                pa.field("gene_id", pa.int64(), nullable=False),
                pa.field("overlap_fraction", pa.float32(), nullable=False),
            ]
        ),
    )
    empty_demux = pa.Table.from_pylist(
        [],
        schema=pa.schema(
            [
                pa.field("read_id", pa.string(), nullable=False),
                pa.field("sample_id", pa.int64(), nullable=True),
            ]
        ),
    )
    quant, stats = count_reads_per_gene(empty_assignments, empty_demux, _samples())
    assert quant.num_rows == 0
    assert stats["total_count"] == 0


# ──────────────────────────────────────────────────────────────────────
# gene_set_from_annotation + serialise_gene_set
# ──────────────────────────────────────────────────────────────────────


def _make_genome_annotation():
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
            {"contig_id": 1, "name": "chr2", "length": 500, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist(
        [
            {"contig_id": 0, "sequence": "A" * 1000},
            {"contig_id": 1, "sequence": "C" * 500},
        ]
    )
    genome = GenomeReference(contigs=contigs, sequences=sequences)

    features = pa.Table.from_pylist(
        [
            {
                "feature_id": 100,
                "contig_id": 0,
                "start": 50,
                "end": 250,
                "strand": "+",
                "type": "gene",
                "name": "geneA",
                "parent_id": None,
                "source": "test",
                "score": None,
                "phase": None,
                "attributes_json": None,
            },
            {
                "feature_id": 101,
                "contig_id": 1,
                "start": 100,
                "end": 200,
                "strand": "-",
                "type": "gene",
                "name": "geneB",
                "parent_id": None,
                "source": "test",
                "score": None,
                "phase": None,
                "attributes_json": None,
            },
        ]
    )
    annotation = Annotation(features=features)
    annotation.validate_against(genome)
    return genome, annotation


def test_gene_set_from_annotation_projects_contig_names() -> None:
    genome, annotation = _make_genome_annotation()
    gs = gene_set_from_annotation(annotation, genome)
    rows = sorted(gs.to_pylist(), key=lambda r: r["gene_id"])
    assert [r["contig_name"] for r in rows] == ["chr1", "chr2"]
    assert [r["gene_id"] for r in rows] == [100, 101]


def test_serialise_gene_set_round_trip() -> None:
    genome, annotation = _make_genome_annotation()
    gs = gene_set_from_annotation(annotation, genome)
    raw = serialise_gene_set(gs)
    with pa.ipc.open_stream(pa.py_buffer(raw)) as reader:
        gs2 = reader.read_all()
    assert gs2.equals(gs)


# ──────────────────────────────────────────────────────────────────────
# fused worker (without BAM — direct table feed)
# ──────────────────────────────────────────────────────────────────────


def test_fused_worker_end_to_end_path_via_kernels() -> None:
    """Smoke test the kernel composition mirroring the fused worker."""
    rows = [
        _alignment_row(alignment_id=0, ref_start=100, ref_end=200, strand="+"),
        _alignment_row(alignment_id=1, ref_start=200, ref_end=300, strand="-"),
        _alignment_row(alignment_id=2, ref_start=5000, ref_end=5100, strand="+"),
    ]
    table = _alignment_table(rows)
    filtered, audit = apply_filter_predicates(table, min_length=50)
    assert filtered.num_rows == 3

    genes = _gene_set(
        [{"gene_id": 100, "contig_name": "chr1", "start": 50, "end": 250, "strand": "+"}]
    )
    out = compute_gene_overlap(filtered, genes, allow_antisense=False)
    # Only the + strand alignment that overlaps the gene matches.
    assert out.column("alignment_id").to_pylist() == [0]
