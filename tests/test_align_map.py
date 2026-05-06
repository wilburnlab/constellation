"""Tests for ``constellation.sequencing.align.map``.

Two test groups:
  * Pure-Python tests for ``_stream_demux_fastq`` — run anywhere with
    pyarrow.
  * minimap2 / samtools end-to-end tests — gated on tool availability.

The full pipeline test exercises ``map_to_genome`` against a tiny
synthetic genome FASTA + a synthetic S1 demux output dir.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.align.map import _stream_demux_fastq
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.reads import READ_TABLE
from constellation.sequencing.schemas.transcriptome import READ_DEMUX_TABLE


_TOOLS_AVAILABLE = bool(shutil.which("minimap2") and shutil.which("samtools"))
_skip_no_tools = pytest.mark.skipif(
    not _TOOLS_AVAILABLE,
    reason=(
        "minimap2 / samtools not on PATH; install via "
        "`conda install -c bioconda minimap2 samtools`"
    ),
)


def _make_genome():
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist(
        [
            {"contig_id": 0, "sequence": "ACGTACGTAC" * 100},
        ]
    )
    return GenomeReference(contigs=contigs, sequences=sequences)


def _make_demux_dir(
    tmp_path: Path,
    *,
    reads: list[dict],
    demux: list[dict],
) -> Path:
    """Stand up a minimal S1 demux output: ``reads/`` + ``read_demux/``
    partitioned datasets, each with one parquet shard."""
    demux_dir = tmp_path / "demux"
    (demux_dir / "reads").mkdir(parents=True)
    (demux_dir / "read_demux").mkdir(parents=True)

    reads_table = pa.Table.from_pylist(reads, schema=READ_TABLE)
    pq.write_table(reads_table, demux_dir / "reads" / "part-00000.parquet")

    demux_table = pa.Table.from_pylist(demux, schema=READ_DEMUX_TABLE)
    pq.write_table(demux_table, demux_dir / "read_demux" / "part-00000.parquet")
    return demux_dir


def test_stream_demux_fastq_filters_and_slices(tmp_path: Path) -> None:
    """Filter set + transcript-window slicing produce the right FASTQ.

    Synthetic dataset:
      r0: status='Complete', sample_id=1, transcript window [10:50] → emit
      r1: status='Unknown'   → drop (status filter)
      r2: status='Complete', sample_id=null → drop (sample_id filter)
      r3: status='Complete', sample_id=1, is_fragment=True → drop
    """
    seq = "ACGT" * 100  # 400 bp marker sequence
    qual = "I" * 400
    reads = [
        {"read_id": f"r{i}", "acquisition_id": 1, "sequence": seq, "quality": qual,
         "length": 400, "mean_quality": 40.0, "channel": None, "start_time_s": None,
         "duration_s": None, "dorado_quality": None, "read_group": None,
         "duplex_class": None}
        for i in range(4)
    ]

    def _demux_row(read_id: str, *, status: str, sample_id, is_fragment: bool):
        return {
            "read_id": read_id,
            "transcript_segment_index": 0,
            "sample_id": sample_id,
            "transcript_start": 10,
            "transcript_end": 50,
            "score": 0.0,
            "is_chimera": False,
            "status": status,
            "is_fragment": is_fragment,
            "artifact": "none",
        }

    demux = [
        _demux_row("r0", status="Complete", sample_id=1, is_fragment=False),
        _demux_row("r1", status="Unknown", sample_id=1, is_fragment=False),
        _demux_row("r2", status="Complete", sample_id=None, is_fragment=False),
        _demux_row("r3", status="Complete", sample_id=1, is_fragment=True),
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)

    chunks = list(_stream_demux_fastq(demux_dir))
    assert len(chunks) == 1
    fastq_bytes, n_reads = chunks[0]
    assert n_reads == 1
    text = fastq_bytes.decode("ascii")
    # One @-prefixed record, the trimmed window only
    assert text.startswith("@r0\n")
    lines = text.strip().split("\n")
    assert len(lines) == 4
    assert lines[1] == seq[10:50]
    assert lines[2] == "+"
    assert lines[3] == qual[10:50]


def test_stream_demux_fastq_skips_invalid_window(tmp_path: Path) -> None:
    """transcript_start=-1 / transcript_end<=transcript_start are dropped."""
    seq = "ACGT" * 50
    reads = [
        {"read_id": "r0", "acquisition_id": 1, "sequence": seq, "quality": "I" * 200,
         "length": 200, "mean_quality": 40.0, "channel": None, "start_time_s": None,
         "duration_s": None, "dorado_quality": None, "read_group": None,
         "duplex_class": None},
        {"read_id": "r1", "acquisition_id": 1, "sequence": seq, "quality": "I" * 200,
         "length": 200, "mean_quality": 40.0, "channel": None, "start_time_s": None,
         "duration_s": None, "dorado_quality": None, "read_group": None,
         "duplex_class": None},
    ]
    demux = [
        {"read_id": "r0", "transcript_segment_index": 0, "sample_id": 1,
         "transcript_start": -1, "transcript_end": -1, "score": 0.0,
         "is_chimera": False, "status": "Complete", "is_fragment": False,
         "artifact": "none"},
        {"read_id": "r1", "transcript_segment_index": 0, "sample_id": 1,
         "transcript_start": 50, "transcript_end": 50, "score": 0.0,
         "is_chimera": False, "status": "Complete", "is_fragment": False,
         "artifact": "none"},
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)
    chunks = list(_stream_demux_fastq(demux_dir))
    assert chunks == []


@_skip_no_tools
def test_map_to_genome_end_to_end(tmp_path: Path) -> None:
    """Full pipeline: synthetic demux dir → minimap2 → sorted+indexed BAM.

    The read carries a 30 bp 'adapter' prefix + a 100 bp transcript
    window matching the genome + a 30 bp 'adapter' suffix. The demux
    table locates the transcript window at [30, 130). Streaming should
    feed only the 100 bp window to minimap2 — if it fed the full read
    (or sliced wrongly), the alignment would either soft-clip the
    adapters or fail to map cleanly.
    """
    genome = _make_genome()
    transcript = "ACGTACGTAC" * 10  # 100 bp matches the genome
    full_read = "N" * 30 + transcript + "N" * 30  # 160 bp total
    full_qual = "I" * len(full_read)

    reads = [
        {
            "read_id": "r0",
            "acquisition_id": 1,
            "sequence": full_read,
            "quality": full_qual,
            "length": len(full_read),
            "mean_quality": 40.0,
            "channel": None,
            "start_time_s": None,
            "duration_s": None,
            "dorado_quality": None,
            "read_group": None,
            "duplex_class": None,
        },
    ]
    demux = [
        {
            "read_id": "r0",
            "transcript_segment_index": 0,
            "sample_id": 1,
            "transcript_start": 30,
            "transcript_end": 130,
            "score": 0.0,
            "is_chimera": False,
            "status": "Complete",
            "is_fragment": False,
            "artifact": "none",
        },
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)

    from constellation.sequencing.align.map import map_to_genome
    from constellation.sequencing.io.sam_bam import read_bam

    output_dir = tmp_path / "out"
    bam = map_to_genome(demux_dir, genome, output_dir=output_dir, threads=1)
    assert bam.exists()
    assert (output_dir / "genome.fa").exists()
    assert (output_dir / "genome.mmi").exists()
    assert (output_dir / "bam" / "aligned.bam.bai").exists()

    aln = read_bam(bam, genome=genome, acquisition_id=1)
    assert aln.n_alignments >= 1
    assert "chr1" in set(aln.alignments.column("ref_name").to_pylist())
    # If we fed minimap2 the trimmed window, the CIGAR is clip-free
    # (e.g. "100M"). If we'd fed the full raw read, we'd see "30S100M30S"
    # — direct proof that the demux slicing took effect.
    cigars = aln.alignments.column("cigar_string").to_pylist()
    assert all("S" not in c for c in cigars), (
        f"expected no soft-clipping in CIGAR (window already trimmed); got {cigars}"
    )


@_skip_no_tools
def test_map_to_genome_caches_fasta(tmp_path: Path) -> None:
    """Genome FASTA is regenerated only on contig-count change."""
    from constellation.sequencing.align.map import map_to_genome

    genome = _make_genome()
    transcript = "ACGTACGTAC" * 10
    reads = [
        {
            "read_id": "r0", "acquisition_id": 1, "sequence": transcript,
            "quality": "I" * len(transcript), "length": len(transcript),
            "mean_quality": 40.0, "channel": None, "start_time_s": None,
            "duration_s": None, "dorado_quality": None, "read_group": None,
            "duplex_class": None,
        },
    ]
    demux = [
        {
            "read_id": "r0", "transcript_segment_index": 0, "sample_id": 1,
            "transcript_start": 0, "transcript_end": len(transcript),
            "score": 0.0, "is_chimera": False, "status": "Complete",
            "is_fragment": False, "artifact": "none",
        },
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)
    output_dir = tmp_path / "out"
    map_to_genome(demux_dir, genome, output_dir=output_dir, threads=1)
    fa_mtime = (output_dir / "genome.fa").stat().st_mtime
    map_to_genome(demux_dir, genome, output_dir=output_dir, threads=1)
    assert (output_dir / "genome.fa").stat().st_mtime == fa_mtime


@_skip_no_tools
def test_minimap2_build_index_skips_when_fresh(tmp_path: Path) -> None:
    from constellation.sequencing.align.minimap2 import minimap2_build_index

    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "ACGTACGTAC" * 100 + "\n")
    mmi = tmp_path / "g.mmi"
    minimap2_build_index(fasta, mmi, threads=1)
    first = mmi.stat().st_mtime
    minimap2_build_index(fasta, mmi, threads=1)
    assert mmi.stat().st_mtime == first


@_skip_no_tools
def test_minimap2_run_accepts_arbitrary_args(tmp_path: Path) -> None:
    """Generic runner stays use-case-agnostic — verify by passing
    map-ont (DNA) flags rather than splice."""
    from constellation.sequencing.align.minimap2 import (
        minimap2_build_index,
        minimap2_run,
    )

    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "ACGTACGTAC" * 100 + "\n")
    mmi = tmp_path / "g.mmi"
    minimap2_build_index(fasta, mmi, threads=1)
    fastq = tmp_path / "r.fq"
    fastq.write_text(f"@r0\n{'ACGTACGTAC' * 10}\n+\n{'I' * 100}\n")
    sam = tmp_path / "out.sam"
    out = minimap2_run(
        target=mmi,
        queries=[fastq],
        output_path=sam,
        args=("-ax", "map-ont"),
        threads=1,
    )
    assert out == sam
    assert sam.exists()
    assert sam.stat().st_size > 0
