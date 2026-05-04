"""Tests for the BAM-cross-tier adapter (``constellation.sequencing.io.sam_bam``).

Builds a tiny synthetic BAM via pysam at fixture time, exercises the
full-file decoder + chunked decoder + Alignments wrap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pysam = pytest.importorskip("pysam")  # skip the file when [sequencing] extra is missing

import pyarrow as pa  # noqa: E402

from constellation.sequencing.alignments import Alignments  # noqa: E402
from constellation.sequencing.io.sam_bam import read_bam  # noqa: E402
from constellation.sequencing.readers.sam_bam import (  # noqa: E402
    _bam_alignment_chunks,
    read_bam_alignments,
    read_bam_alignments_chunk,
)
from constellation.sequencing.reference.reference import GenomeReference  # noqa: E402


@pytest.fixture
def tiny_bam(tmp_path: Path) -> Path:
    """Write a minimal SAM with 4 records (3 mapped, 1 unmapped, 1 secondary)
    and convert to BAM via pysam."""
    sam_path = tmp_path / "test.sam"
    bam_path = tmp_path / "test.bam"
    seq = "ACGT" * 25  # 100 bp
    qual = "I" * 100
    sam_text = "\n".join(
        [
            "@HD\tVN:1.6\tSO:coordinate",
            "@SQ\tSN:chr1\tLN:1000",
            "@SQ\tSN:chr2\tLN:500",
            f"r0\t0\tchr1\t100\t60\t100M\t*\t0\t0\t{seq}\t{qual}\tNM:i:0\tAS:i:100",
            f"r1\t16\tchr1\t150\t30\t50M10I40M\t*\t0\t0\t{seq}\t{qual}\tNM:i:5",
            f"r0\t256\tchr2\t50\t10\t100M\t*\t0\t0\t{seq}\t{qual}",
            "r2\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\t!!!!",
        ]
    )
    sam_path.write_text(sam_text + "\n")
    pysam.view("-b", "-o", str(bam_path), str(sam_path), catch_stdout=False)
    return bam_path


def _make_genome():
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
    return GenomeReference(contigs=contigs, sequences=sequences)


def test_read_bam_alignments_skips_unmapped(tiny_bam: Path) -> None:
    a, t = read_bam_alignments(tiny_bam, acquisition_id=1, tags_to_keep=("NM",))
    assert a.num_rows == 3  # 3 mapped (incl. secondary); unmapped dropped
    secondaries = a.column("is_secondary").to_pylist()
    assert secondaries == [False, False, True]
    # NM tag rows materialise only for the alignments that carry NM.
    assert t.num_rows == 2
    assert all(t.column("tag").to_pylist()[i] == "NM" for i in range(2))


def test_read_bam_alignments_promoted_columns(tiny_bam: Path) -> None:
    a, _ = read_bam_alignments(tiny_bam, acquisition_id=1)
    nm = a.column("nm_tag").to_pylist()
    as_ = a.column("as_tag").to_pylist()
    assert nm == [0, 5, None]
    assert as_ == [100.0, None, None]


def test_read_bam_alignments_strand_inference(tiny_bam: Path) -> None:
    a, _ = read_bam_alignments(tiny_bam, acquisition_id=1)
    assert a.column("strand").to_pylist() == ["+", "-", "+"]


def test_read_bam_alignments_zero_based_coords(tiny_bam: Path) -> None:
    a, _ = read_bam_alignments(tiny_bam, acquisition_id=1)
    # SAM POS is 1-based; we convert to 0-based half-open.
    assert a.column("ref_start").to_pylist() == [99, 149, 49]
    # 100M alignment ⇒ ref_end = ref_start + 100
    assert a.column("ref_end").to_pylist()[0] == 199


def test_chunked_decoder_packs_worker_idx(tiny_bam: Path) -> None:
    chunks = _bam_alignment_chunks(tiny_bam, chunk_size=2)
    assert len(chunks) == 2  # 4 records → 2 chunks of size 2
    a0, _ = read_bam_alignments_chunk(
        tiny_bam, vo_start=chunks[0][0], n_records=chunks[0][1], worker_idx=0
    )
    a1, _ = read_bam_alignments_chunk(
        tiny_bam, vo_start=chunks[1][0], n_records=chunks[1][1], worker_idx=1
    )
    ids0 = a0.column("alignment_id").to_pylist()
    ids1 = a1.column("alignment_id").to_pylist()
    # Worker 0 starts at 0; worker 1 starts at (1 << 32) = 4294967296.
    assert all(i < (1 << 32) for i in ids0)
    assert all(i >= (1 << 32) for i in ids1)
    # Combined cover all 3 mapped records.
    assert a0.num_rows + a1.num_rows == 3


def test_read_bam_returns_alignments_container(tiny_bam: Path) -> None:
    genome = _make_genome()
    aln = read_bam(tiny_bam, genome=genome, acquisition_id=1)
    assert isinstance(aln, Alignments)
    assert aln.n_alignments == 3
    # validate_against should have run cleanly inside read_bam.
    aln.validate_against(genome)


def test_read_bam_validates_against_genome(tiny_bam: Path) -> None:
    # Build a genome that doesn't include chr2 → the secondary alignment
    # row's ref_name should trip validate_against.
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist([{"contig_id": 0, "sequence": "A" * 1000}])
    genome = GenomeReference(contigs=contigs, sequences=sequences)
    with pytest.raises(ValueError, match="absent from GenomeReference"):
        read_bam(tiny_bam, genome=genome, acquisition_id=1)
