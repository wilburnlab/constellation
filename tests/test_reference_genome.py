"""``GenomeReference`` container + ParquetDir round-trip + FASTA reader."""

from __future__ import annotations

import gzip
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.readers.fastx import read_fasta_genome
from constellation.sequencing.reference.io import (
    load_genome_reference,
    save_genome_reference,
)
from constellation.sequencing.reference.reference import GenomeReference


def _make_genome() -> GenomeReference:
    contigs = pa.table(
        {
            "contig_id": pa.array([0, 1], type=pa.int64()),
            "name": ["chrA", "chrB"],
            "length": pa.array([12, 8], type=pa.int64()),
            "topology": pa.array(["chromosome", "scaffold"]),
            "circular": pa.array([False, False]),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0, 1], type=pa.int64()),
            "sequence": ["ACGTACGTACGT", "TTTTGGGG"],
        }
    )
    return GenomeReference(contigs=contigs, sequences=sequences)


def test_construct_validates_pk_and_fk():
    g = _make_genome()
    assert g.n_contigs == 2
    assert g.total_length == 20


def test_duplicate_contig_id_rejected():
    contigs = pa.table(
        {
            "contig_id": pa.array([0, 0], type=pa.int64()),
            "name": ["a", "b"],
            "length": pa.array([1, 1], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0, 0], type=pa.int64()),
            "sequence": ["A", "C"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        GenomeReference(contigs=contigs, sequences=sequences)


def test_orphan_sequence_rejected():
    contigs = pa.table(
        {
            "contig_id": pa.array([0], type=pa.int64()),
            "name": ["x"],
            "length": pa.array([1], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0, 99], type=pa.int64()),
            "sequence": ["A", "C"],
        }
    )
    with pytest.raises(ValueError, match="references ids"):
        GenomeReference(contigs=contigs, sequences=sequences)


def test_sequence_of_lookup():
    g = _make_genome()
    assert g.sequence_of(0) == "ACGTACGTACGT"
    assert g.sequence_of(1) == "TTTTGGGG"
    with pytest.raises(KeyError):
        g.sequence_of(99)


def test_with_metadata_roundtrip():
    g = _make_genome().with_metadata({"organism": "test"})
    assert g.metadata_extras["organism"] == "test"


def test_parquetdir_roundtrip(tmp_path: Path):
    g = _make_genome().with_metadata({"organism": "test"})
    save_genome_reference(g, tmp_path / "genome")
    g2 = load_genome_reference(tmp_path / "genome")
    assert g2.n_contigs == g.n_contigs
    assert g2.total_length == g.total_length
    assert g2.sequence_of(0) == g.sequence_of(0)
    assert g2.metadata_extras["organism"] == "test"


def test_sequences_parquet_uses_one_row_group_per_contig(tmp_path: Path):
    """A filter on `contig_id` must be able to skip whole-genome
    decode — required so the viz layer's reference_sequence cache
    miss path doesn't materialize every contig's bytes at first use.
    """
    import pyarrow.parquet as pq

    g = _make_genome()
    save_genome_reference(g, tmp_path / "genome")
    pf = pq.ParquetFile(tmp_path / "genome" / "sequences.parquet")
    assert pf.num_row_groups == g.n_contigs


def test_read_fasta_genome(tmp_path: Path):
    fa = tmp_path / "g.fa"
    fa.write_text(">chr1 some desc\nACGTACGT\nACGT\n>chr2\nTTTT\n")
    g = read_fasta_genome(fa)
    assert g.n_contigs == 2
    assert g.sequence_of(0) == "ACGTACGTACGT"
    assert g.sequence_of(1) == "TTTT"
    assert "contig_descriptions" in g.metadata_extras


def test_read_fasta_genome_gzip(tmp_path: Path):
    fa_gz = tmp_path / "g.fa.gz"
    with gzip.open(fa_gz, "wt", encoding="utf-8") as fh:
        fh.write(">chrA\nACGT\n>chrB\nTTTT\n")
    g = read_fasta_genome(fa_gz)
    assert g.n_contigs == 2
    assert g.sequence_of(0) == "ACGT"


def test_read_fasta_skips_blank_and_comment_lines(tmp_path: Path):
    fa = tmp_path / "g.fa"
    fa.write_text(
        "; a legacy comment\n>chrA\n\nACGT\nACGT\n; another comment\n>chrB\n\nTTTT\n"
    )
    g = read_fasta_genome(fa)
    assert g.n_contigs == 2
    assert g.sequence_of(0) == "ACGTACGT"
    assert g.sequence_of(1) == "TTTT"
