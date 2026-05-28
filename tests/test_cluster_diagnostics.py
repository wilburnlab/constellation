"""Tests for ``constellation.sequencing.transcriptome.cluster.diagnostics``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.transcriptome.cluster.diagnostics import (
    _count_gene_overlaps_per_cluster,
    build_cluster_diagnostics_report,
    section_cluster_size_distribution,
    section_cluster_span_vs_annotated_genes,
    section_consensus_quality,
    section_drift_filter_stats,
    section_layer0_derep,
    section_multi_gene_cluster_overlap,
)
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)


# ── Fixture builders ──────────────────────────────────────────────────


def _cluster_row(
    cid: int,
    *,
    rep: str | None = None,
    n_reads: int = 5,
    n_unique: int = 3,
    contig_id: int | None = 0,
    strand: str | None = "+",
    span_start: int | None = 100,
    span_end: int | None = 1000,
    fingerprint_hash: int = 42,
    sample_id: int | None = None,
) -> dict:
    return {
        "cluster_id": cid,
        "representative_read_id": rep or f"r{cid}",
        "n_reads": n_reads,
        "identity_threshold": None,
        "consensus_sequence": None,
        "predicted_protein": None,
        "orf_start": None,
        "orf_end": None,
        "orf_strand": None,
        "codon_table": None,
        "mode": "genome-guided",
        "contig_id": contig_id,
        "strand": strand,
        "span_start": span_start,
        "span_end": span_end,
        "fingerprint_hash": fingerprint_hash,
        "n_unique_sequences": n_unique,
        "sample_id": sample_id,
    }


def _clusters_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=TRANSCRIPT_CLUSTER_TABLE)


def _member_row(
    cid: int, rid: str, *, role: str = "member",
    drift_5p: int | None = 0, drift_3p: int | None = 0,
    match_rate: float | None = 0.95, indel_rate: float | None = 0.02,
    n_aligned_bp: int = 500,
) -> dict:
    return {
        "cluster_id": cid,
        "read_id": rid,
        "role": role,
        "drift_5p_bp": drift_5p,
        "drift_3p_bp": drift_3p,
        "match_rate": match_rate,
        "indel_rate": indel_rate,
        "n_aligned_bp": n_aligned_bp,
    }


def _membership_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CLUSTER_MEMBERSHIP_TABLE)


def _annotation_with_genes(
    gene_specs: list[tuple[int, int, int, int]],
):
    from constellation.sequencing.annotation.annotation import Annotation
    from constellation.sequencing.schemas.reference import FEATURE_TABLE
    rows = []
    for fid, cid, start, end in gene_specs:
        rows.append({
            "feature_id": fid, "contig_id": cid, "start": start, "end": end,
            "strand": "+", "type": "gene", "name": f"gene{fid}",
            "parent_id": None, "source": "test",
            "score": None, "phase": None, "attributes_json": None,
        })
    return Annotation(features=pa.Table.from_pylist(rows, schema=FEATURE_TABLE))


# ── section_cluster_size_distribution ────────────────────────────────


def test_cluster_size_distribution_basic(tmp_path: Path):
    # 100 clusters of 5 reads each — top is 1% of total reads (under the
    # 5% mega-cluster threshold), no singletons (under the 80% threshold).
    clusters = _clusters_table([_cluster_row(i, n_reads=5) for i in range(100)])
    section = section_cluster_size_distribution(clusters, figures_dir=tmp_path)
    assert "100 clusters covering 500 reads" in section.body
    assert section.figures and section.figures[0].is_file()
    assert section.flags == []


def test_cluster_size_distribution_flags_high_singleton_fraction(tmp_path: Path):
    # 9 singletons + 1 cluster of 2 = 90% singletons
    clusters = _clusters_table(
        [_cluster_row(i, n_reads=1) for i in range(9)]
        + [_cluster_row(9, n_reads=2)]
    )
    section = section_cluster_size_distribution(clusters, figures_dir=tmp_path)
    assert any("cluster_size" in f and "singleton" in f for f in section.flags)


def test_cluster_size_distribution_flags_giant_top_cluster(tmp_path: Path):
    # 1 cluster of 1000 reads + 4 of 1 read = top holds ~99.6% of reads
    clusters = _clusters_table(
        [_cluster_row(0, n_reads=1000)]
        + [_cluster_row(i, n_reads=1) for i in range(1, 5)]
    )
    section = section_cluster_size_distribution(clusters, figures_dir=tmp_path)
    assert any("top cluster" in f for f in section.flags)


# ── section_cluster_span_vs_annotated_genes ──────────────────────────


def test_cluster_span_vs_genes_flags_huge_spans(tmp_path: Path):
    # 10 clusters with spans 50 kb each; genes are 500-1000 bp → 2× p99 = ~2 kb
    clusters = _clusters_table([
        _cluster_row(i, span_start=0, span_end=50_000) for i in range(10)
    ])
    ann = _annotation_with_genes([
        (i, 0, i * 2000, i * 2000 + 500) for i in range(10)
    ])
    section = section_cluster_span_vs_annotated_genes(
        clusters, ann, figures_dir=tmp_path,
    )
    assert section.figures and section.figures[0].is_file()
    assert any("cluster_span" in f for f in section.flags)


def test_cluster_span_vs_genes_no_annotation_skips(tmp_path: Path):
    clusters = _clusters_table([_cluster_row(0)])
    section = section_cluster_span_vs_annotated_genes(
        clusters, None, figures_dir=tmp_path,
    )
    assert "No annotation available" in section.body
    assert section.figures == []


# ── section_multi_gene_cluster_overlap ───────────────────────────────


def test_multi_gene_cluster_helper_counts_correctly():
    # 3 cluster spans, 2 genes
    clusters = _clusters_table([
        _cluster_row(0, span_start=500, span_end=2500, n_reads=5),  # spans both
        _cluster_row(1, span_start=100, span_end=900, n_reads=3),   # one
        _cluster_row(2, span_start=10_000, span_end=11_000, n_reads=2),  # zero
    ])
    ann = _annotation_with_genes([(1, 0, 0, 1000), (2, 0, 2000, 3000)])
    counts = _count_gene_overlaps_per_cluster(
        clusters.select(["cluster_id", "contig_id", "span_start", "span_end", "n_reads"]),
        ann.features_of_type("gene"),
    )
    assert list(counts) == [2, 1, 0]


def test_multi_gene_cluster_section_flags_fusion(tmp_path: Path):
    # 5 fusion + 5 clean clusters → 50% > 2% → flag
    fusion = [
        _cluster_row(i, span_start=500, span_end=2500, n_reads=10)
        for i in range(5)
    ]
    clean = [
        _cluster_row(5 + i, span_start=100, span_end=900, n_reads=10)
        for i in range(5)
    ]
    ann = _annotation_with_genes([(1, 0, 0, 1000), (2, 0, 2000, 3000)])
    section = section_multi_gene_cluster_overlap(
        _clusters_table(fusion + clean), ann,
        contig_name_to_id={"chr1": 0},
        figures_dir=tmp_path,
    )
    assert any("multi_gene_cluster" in f for f in section.flags)
    assert section.figures and section.figures[0].is_file()


# ── section_drift_filter_stats ───────────────────────────────────────


def test_drift_filter_stats_flags_excessive_drift(tmp_path: Path):
    # 4 representatives + 30 drift_filtered + 6 members = 30 / 40 = 75% drift
    members = (
        [_member_row(i, f"rep{i}", role="representative") for i in range(4)]
        + [
            _member_row(0, f"dr{i}", role="drift_filtered",
                        drift_5p=50, drift_3p=200)
            for i in range(30)
        ]
        + [_member_row(0, f"m{i}", role="member") for i in range(6)]
    )
    section = section_drift_filter_stats(
        _membership_table(members), figures_dir=tmp_path,
    )
    assert any("drift_filter" in f for f in section.flags)
    assert section.figures and section.figures[0].is_file()


def test_drift_filter_stats_clean_run_no_flag(tmp_path: Path):
    members = [
        _member_row(i, f"r{i}", role="representative")
        for i in range(10)
    ]
    section = section_drift_filter_stats(
        _membership_table(members), figures_dir=tmp_path,
    )
    assert section.flags == []


# ── section_layer0_derep ─────────────────────────────────────────────


def test_layer0_derep_section_emits_distribution(tmp_path: Path):
    clusters = _clusters_table(
        [_cluster_row(i, n_reads=10, n_unique=3) for i in range(20)]
    )
    section = section_layer0_derep(clusters, figures_dir=tmp_path)
    assert "n_unique_sequences / n_reads ratio" in section.body
    assert section.figures and section.figures[0].is_file()


# ── section_consensus_quality ────────────────────────────────────────


def test_consensus_quality_flags_low_match_rate(tmp_path: Path):
    members = [
        _member_row(0, f"r{i}", match_rate=0.80, indel_rate=0.05)
        for i in range(20)
    ]
    section = section_consensus_quality(
        _membership_table(members), figures_dir=tmp_path,
    )
    assert any("consensus_quality" in f for f in section.flags)


def test_consensus_quality_skips_when_all_null(tmp_path: Path):
    members = [
        _member_row(0, f"r{i}", match_rate=None, indel_rate=None)
        for i in range(5)
    ]
    section = section_consensus_quality(
        _membership_table(members), figures_dir=tmp_path,
    )
    assert "null for every member" in section.body
    assert section.figures == []


# ── Orchestrator end-to-end ──────────────────────────────────────────


def _write_minimal_cluster_dir(tmp_path: Path) -> Path:
    """Build a synthetic align + cluster output pair with manifests."""
    from constellation.sequencing.transcriptome.manifest import (
        write_align_manifest, write_cluster_manifest,
    )

    align_dir = tmp_path / "align_out"
    align_dir.mkdir()
    write_align_manifest(
        align_dir / "manifest.json",
        reference_handle=None,
        reference_path=str(tmp_path / "missing_ref"),
        assembly_accession=None,
        demux_dir=str(tmp_path / "fake_demux"),
        input_files=[],
        parameters={},
        stages={},
        outputs={},
    )

    cluster_dir = tmp_path / "cluster_out"
    cluster_dir.mkdir()
    clusters = _clusters_table(
        [_cluster_row(i, n_reads=5) for i in range(5)]
    )
    pq.write_table(clusters, cluster_dir / "clusters.parquet")
    members = [
        _member_row(i, f"r{i}", role="representative") for i in range(5)
    ]
    pq.write_table(_membership_table(members), cluster_dir / "cluster_membership.parquet")

    write_cluster_manifest(
        cluster_dir / "manifest.json",
        reference_handle=None,
        reference_path=str(tmp_path / "missing_ref"),
        assembly_accession=None,
        align_dir=str(align_dir),
        demux_dir=str(tmp_path / "fake_demux"),
        parameters={
            "mode": "genome-guided",
            "max_5p_drift": 25,
            "max_3p_drift": 75,
        },
        stages={},
        outputs={},
    )
    return cluster_dir


def test_build_cluster_diagnostics_report_orchestrator(tmp_path: Path):
    cluster_dir = _write_minimal_cluster_dir(tmp_path)
    report_path = build_cluster_diagnostics_report(cluster_dir)
    assert report_path.is_file()
    content = report_path.read_text()
    assert "# Transcriptome cluster — diagnostics" in content
    assert "Run parameters" in content
    assert "Cluster size distribution" in content
    # No annotation loaded — annotation-dependent sections degrade
    assert "No annotation available" in content
    figures_dir = cluster_dir / "diagnostics" / "figures"
    assert figures_dir.is_dir()
    assert any(figures_dir.glob("*.svg"))


def test_build_cluster_diagnostics_report_missing_outputs_raises(tmp_path: Path):
    from constellation.sequencing.transcriptome.manifest import (
        write_cluster_manifest,
    )

    cluster_dir = tmp_path / "empty_cluster"
    cluster_dir.mkdir()
    write_cluster_manifest(
        cluster_dir / "manifest.json",
        reference_handle=None,
        reference_path=None,
        assembly_accession=None,
        align_dir="/tmp/nope",
        demux_dir="/tmp/nope",
        parameters={},
        stages={},
        outputs={},
    )
    with pytest.raises(FileNotFoundError, match="clusters.parquet"):
        build_cluster_diagnostics_report(cluster_dir)
