"""Tests for ``constellation transcriptome diagnose`` — the standalone
re-runs-no-pipeline-stages diagnostic regenerator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.cli.__main__ import (
    _build_transcriptome_parser,
    _cmd_transcriptome_diagnose,
)
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="constellation")
    subs = parser.add_subparsers(dest="subcommand")
    _build_transcriptome_parser(subs)
    return parser


def test_diagnose_parser_accepts_align_dir(tmp_path: Path):
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "diagnose",
        "--align-dir", str(tmp_path),
    ])
    assert args.align_dir == str(tmp_path)
    assert args.cluster_dir is None


def test_diagnose_parser_accepts_both_dirs(tmp_path: Path):
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "diagnose",
        "--align-dir", str(tmp_path / "a"),
        "--cluster-dir", str(tmp_path / "c"),
    ])
    assert args.align_dir == str(tmp_path / "a")
    assert args.cluster_dir == str(tmp_path / "c")


def test_diagnose_requires_at_least_one_dir(capsys):
    parser = _build_parser()
    args = parser.parse_args(["transcriptome", "diagnose"])
    rc = _cmd_transcriptome_diagnose(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "at least one of --align-dir / --cluster-dir" in err


def test_diagnose_mutually_exclusive_reference_args(tmp_path: Path, capsys):
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "diagnose",
        "--align-dir", str(tmp_path),
        "--reference", "x",
        "--reference-dir", str(tmp_path),
    ])
    rc = _cmd_transcriptome_diagnose(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_diagnose_align_dir_not_found(tmp_path: Path, capsys):
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "diagnose",
        "--align-dir", str(tmp_path / "does_not_exist"),
    ])
    rc = _cmd_transcriptome_diagnose(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


def _build_synthetic_align_and_cluster(tmp_path: Path) -> tuple[Path, Path]:
    """Synthesise minimal align + cluster output dirs with manifests."""
    from constellation.sequencing.schemas.alignment import (
        ALIGNMENT_BLOCK_TABLE, ALIGNMENT_TABLE, INTRON_TABLE,
    )
    from constellation.sequencing.transcriptome.manifest import (
        write_align_manifest, write_cluster_manifest,
    )

    align_dir = tmp_path / "align"
    align_dir.mkdir()
    introns = pa.Table.from_pylist(
        [{
            "intron_id": i, "contig_id": 0, "strand": "+",
            "donor_pos": i * 100, "acceptor_pos": i * 100 + 150,
            "read_count": 10, "motif": "GT-AG",
            "is_intron_seed": True, "annotated": True,
        } for i in range(3)],
        schema=INTRON_TABLE,
    )
    pq.write_table(introns, align_dir / "introns.parquet")
    alignments = pa.Table.from_pylist(
        [{
            "alignment_id": i, "read_id": f"r{i}", "acquisition_id": 0,
            "ref_name": "chr1", "ref_start": i * 1000, "ref_end": i * 1000 + 800,
            "strand": "+", "mapq": 60, "flag": 0, "cigar_string": "800M",
            "nm_tag": None, "as_tag": None, "read_group": None,
            "is_secondary": False, "is_supplementary": False,
        } for i in range(3)],
        schema=ALIGNMENT_TABLE,
    )
    aln_d = align_dir / "alignments"
    aln_d.mkdir()
    pq.write_table(alignments, aln_d / "part-00000.parquet")
    blocks = pa.Table.from_pylist(
        [{
            "alignment_id": i, "block_index": 0,
            "ref_start": i * 1000, "ref_end": i * 1000 + 800,
            "query_start": 0, "query_end": 800,
            "n_match": 800, "n_mismatch": 0, "n_insert": 0, "n_delete": 0,
        } for i in range(3)],
        schema=ALIGNMENT_BLOCK_TABLE,
    )
    blocks_d = align_dir / "alignment_blocks"
    blocks_d.mkdir()
    pq.write_table(blocks, blocks_d / "part-00000.parquet")
    write_align_manifest(
        align_dir / "manifest.json",
        reference_handle=None,
        reference_path=str(tmp_path / "missing_ref"),
        assembly_accession=None,
        demux_dir=str(tmp_path / "fake_demux"),
        input_files=[], parameters={}, stages={}, outputs={},
    )

    cluster_dir = tmp_path / "cluster"
    cluster_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist([
            {
                "cluster_id": i, "representative_read_id": f"r{i}",
                "n_reads": 5, "identity_threshold": None,
                "consensus_sequence": None, "predicted_protein": None,
                "orf_start": None, "orf_end": None, "orf_strand": None,
                "codon_table": None, "mode": "genome-guided",
                "contig_id": 0, "strand": "+",
                "span_start": i * 1000, "span_end": i * 1000 + 800,
                "fingerprint_hash": 42, "n_unique_sequences": 3,
                "sample_id": None,
            } for i in range(5)
        ], schema=TRANSCRIPT_CLUSTER_TABLE),
        cluster_dir / "clusters.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([
            {
                "cluster_id": i, "read_id": f"r{i}", "role": "representative",
                "drift_5p_bp": None, "drift_3p_bp": None,
                "match_rate": 0.95, "indel_rate": 0.02, "n_aligned_bp": 800,
            } for i in range(5)
        ], schema=CLUSTER_MEMBERSHIP_TABLE),
        cluster_dir / "cluster_membership.parquet",
    )
    write_cluster_manifest(
        cluster_dir / "manifest.json",
        reference_handle=None,
        reference_path=str(tmp_path / "missing_ref"),
        assembly_accession=None,
        align_dir=str(align_dir),
        demux_dir=str(tmp_path / "fake_demux"),
        parameters={}, stages={}, outputs={},
    )
    return align_dir, cluster_dir


def test_diagnose_runs_against_existing_align_and_cluster_dirs(
    tmp_path: Path, capsys,
):
    align_dir, cluster_dir = _build_synthetic_align_and_cluster(tmp_path)
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "diagnose",
        "--align-dir", str(align_dir),
        "--cluster-dir", str(cluster_dir),
    ])
    rc = _cmd_transcriptome_diagnose(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "align report:" in captured.out
    assert "cluster report:" in captured.out
    # Reports were written
    assert (align_dir / "diagnostics" / "report.md").is_file()
    assert (cluster_dir / "diagnostics" / "report.md").is_file()


def test_diagnose_only_cluster_dir_skips_align(tmp_path: Path, capsys):
    _, cluster_dir = _build_synthetic_align_and_cluster(tmp_path)
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "diagnose",
        "--cluster-dir", str(cluster_dir),
    ])
    rc = _cmd_transcriptome_diagnose(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "cluster report:" in captured.out
    assert "align report:" not in captured.out
