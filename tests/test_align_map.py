"""Tests for ``constellation.sequencing.align.map`` — minimap2 wrapper.

Skip the file if minimap2 isn't on $PATH (test environment without
[sequencing] extra). The full test exercises ``map_to_genome`` against
a tiny synthetic genome FASTA + simulator-derived FASTQ.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

if not shutil.which("minimap2") or not shutil.which("samtools"):
    pytest.skip(
        "minimap2 / samtools not on PATH; install via `conda install -c bioconda minimap2 samtools`",
        allow_module_level=True,
    )

import pyarrow as pa  # noqa: E402

from constellation.sequencing.align.map import map_to_genome  # noqa: E402
from constellation.sequencing.align.minimap2 import (  # noqa: E402
    minimap2_build_index,
    minimap2_run,
)
from constellation.sequencing.io.sam_bam import read_bam  # noqa: E402
from constellation.sequencing.reference.reference import GenomeReference  # noqa: E402


def _make_genome():
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist(
        [
            # Random-ish sequence to give minimap2 something to map against
            {"contig_id": 0, "sequence": "ACGTACGTAC" * 100},
        ]
    )
    return GenomeReference(contigs=contigs, sequences=sequences)


def test_map_to_genome_end_to_end(tmp_path: Path) -> None:
    genome = _make_genome()
    fastq = tmp_path / "reads.fastq"
    # One synthetic read identical to a slice of the genome
    seq = "ACGTACGTAC" * 10  # 100 bp
    fastq.write_text(f"@r0\n{seq}\n+\n{'I' * 100}\n")

    output_dir = tmp_path / "out"
    bam = map_to_genome(
        [fastq],
        genome,
        output_dir=output_dir,
        threads=1,
    )
    assert bam.exists()
    assert (output_dir / "genome.fa").exists()
    assert (output_dir / "genome.mmi").exists()
    assert (output_dir / "bam" / "aligned.bam.bai").exists()

    aln = read_bam(bam, genome=genome, acquisition_id=1)
    assert aln.n_alignments >= 1
    assert "chr1" in set(aln.alignments.column("ref_name").to_pylist())


def test_map_to_genome_caches_fasta(tmp_path: Path) -> None:
    genome = _make_genome()
    fastq = tmp_path / "reads.fastq"
    fastq.write_text(f"@r0\n{'ACGTACGTAC' * 10}\n+\n{'I' * 100}\n")
    output_dir = tmp_path / "out"
    map_to_genome([fastq], genome, output_dir=output_dir, threads=1)
    fa_mtime = (output_dir / "genome.fa").stat().st_mtime
    map_to_genome([fastq], genome, output_dir=output_dir, threads=1)
    # Cached FASTA shouldn't be rewritten on the second call
    assert (output_dir / "genome.fa").stat().st_mtime == fa_mtime


def test_minimap2_build_index_skips_when_fresh(tmp_path: Path) -> None:
    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "ACGTACGTAC" * 100 + "\n")
    mmi = tmp_path / "g.mmi"
    minimap2_build_index(fasta, mmi, threads=1)
    first = mmi.stat().st_mtime
    minimap2_build_index(fasta, mmi, threads=1)
    assert mmi.stat().st_mtime == first


def test_minimap2_run_accepts_arbitrary_args(tmp_path: Path) -> None:
    """Generic runner stays use-case-agnostic — verify by passing
    map-ont (DNA) flags rather than splice."""
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
