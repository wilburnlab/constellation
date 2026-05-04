"""``TranscriptReference`` container + from_annotation materialiser."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.readers.fastx import read_fasta_transcriptome
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.transcripts.io import (
    load_transcript_reference,
    save_transcript_reference,
)
from constellation.sequencing.transcripts.transcripts import TranscriptReference


def _genome_with_known_sequence() -> GenomeReference:
    # 60 bp contig: positions 0-59
    seq = "AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGG"
    assert len(seq) == 60
    return GenomeReference(
        contigs=pa.table(
            {
                "contig_id": pa.array([0], type=pa.int64()),
                "name": ["chr1"],
                "length": pa.array([60], type=pa.int64()),
            }
        ),
        sequences=pa.table(
            {
                "contig_id": pa.array([0], type=pa.int64()),
                "sequence": [seq],
            }
        ),
    )


def test_construct_validates_pk():
    table = pa.table(
        {
            "transcript_id": pa.array([0, 0], type=pa.int64()),
            "name": ["t1", "t2"],
            "gene_id": pa.array([None, None], type=pa.int64()),
            "sequence": ["ACGT", "TTTT"],
            "length": pa.array([4, 4], type=pa.int32()),
            "source": ["fasta", "fasta"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        TranscriptReference(transcripts=table)


def test_length_mismatch_rejected():
    table = pa.table(
        {
            "transcript_id": pa.array([0], type=pa.int64()),
            "name": ["t1"],
            "gene_id": pa.array([None], type=pa.int64()),
            "sequence": ["ACGT"],
            "length": pa.array([99], type=pa.int32()),  # wrong
            "source": ["fasta"],
        }
    )
    with pytest.raises(ValueError, match="length column"):
        TranscriptReference(transcripts=table)


def test_from_annotation_single_exon_plus_strand():
    g = _genome_with_known_sequence()
    seq = g.sequence_of(0)

    features = pa.table(
        {
            "feature_id": pa.array([0, 1, 2], type=pa.int64()),
            "contig_id": pa.array([0, 0, 0], type=pa.int64()),
            "start": pa.array([10, 10, 10], type=pa.int64()),
            "end": pa.array([20, 20, 20], type=pa.int64()),
            "strand": ["+", "+", "+"],
            "type": ["gene", "mRNA", "exon"],
            "name": ["G1", "G1.1", None],
            "parent_id": pa.array([None, 0, 1], type=pa.int64()),
        }
    )
    a = Annotation(features=features)
    t = TranscriptReference.from_annotation(g, a)
    assert t.n_transcripts == 1
    expected = seq[10:20]
    assert t.sequence_of(1) == expected


def test_from_annotation_multi_exon_plus_strand():
    g = _genome_with_known_sequence()
    seq = g.sequence_of(0)

    features = pa.table(
        {
            "feature_id": pa.array([0, 1, 2, 3], type=pa.int64()),
            "contig_id": pa.array([0, 0, 0, 0], type=pa.int64()),
            "start": pa.array([0, 0, 4, 16], type=pa.int64()),
            "end": pa.array([24, 24, 8, 24], type=pa.int64()),
            "strand": ["+", "+", "+", "+"],
            "type": ["gene", "mRNA", "exon", "exon"],
            "name": ["G", "G.1", None, None],
            "parent_id": pa.array([None, 0, 1, 1], type=pa.int64()),
        }
    )
    a = Annotation(features=features)
    t = TranscriptReference.from_annotation(g, a)
    expected = seq[4:8] + seq[16:24]
    assert t.sequence_of(1) == expected
    # gene_id linkage walked from mRNA → gene
    rows = t.transcripts.to_pylist()
    assert rows[0]["gene_id"] == 0


def test_from_annotation_minus_strand_reverse_complements():
    g = _genome_with_known_sequence()
    seq = g.sequence_of(0)

    # mRNA on '-' strand spanning [4..12)
    features = pa.table(
        {
            "feature_id": pa.array([0, 1, 2], type=pa.int64()),
            "contig_id": pa.array([0, 0, 0], type=pa.int64()),
            "start": pa.array([4, 4, 4], type=pa.int64()),
            "end": pa.array([12, 12, 12], type=pa.int64()),
            "strand": ["-", "-", "-"],
            "type": ["gene", "mRNA", "exon"],
            "name": ["G", "G.1", None],
            "parent_id": pa.array([None, 0, 1], type=pa.int64()),
        }
    )
    a = Annotation(features=features)
    t = TranscriptReference.from_annotation(g, a)
    expected_plus = seq[4:12]
    # reverse-complement: reverse the string and complement bases
    complement = str.maketrans("ACGT", "TGCA")
    expected_minus = expected_plus.translate(complement)[::-1]
    assert t.sequence_of(1) == expected_minus


def test_from_annotation_skips_transcripts_without_exons():
    g = _genome_with_known_sequence()
    features = pa.table(
        {
            "feature_id": pa.array([0, 1], type=pa.int64()),
            "contig_id": pa.array([0, 0], type=pa.int64()),
            "start": pa.array([0, 0], type=pa.int64()),
            "end": pa.array([10, 10], type=pa.int64()),
            "strand": ["+", "+"],
            "type": ["gene", "mRNA"],
            "name": ["G", "G.1"],
            "parent_id": pa.array([None, 0], type=pa.int64()),
        }
    )
    a = Annotation(features=features)
    t = TranscriptReference.from_annotation(g, a)
    assert t.n_transcripts == 0
    assert "skipped_transcripts_without_exons" in t.metadata_extras


def test_parquetdir_roundtrip(tmp_path: Path):
    table = pa.table(
        {
            "transcript_id": pa.array([0, 1], type=pa.int64()),
            "name": ["t1", "t2"],
            "gene_id": pa.array([None, None], type=pa.int64()),
            "sequence": ["ACGT", "TTTTGGGG"],
            "length": pa.array([4, 8], type=pa.int32()),
            "source": ["fasta", "fasta"],
        }
    )
    t = TranscriptReference(transcripts=table)
    save_transcript_reference(t, tmp_path / "tx")
    t2 = load_transcript_reference(tmp_path / "tx")
    assert t2.n_transcripts == 2


def test_read_fasta_transcriptome(tmp_path: Path):
    p = tmp_path / "tx.fa"
    p.write_text(">ENST001 desc1\nACGTACGT\n>ENST002\nTTTT\n")
    t = read_fasta_transcriptome(p)
    assert t.n_transcripts == 2
    assert t.transcripts.column("name").to_pylist() == ["ENST001", "ENST002"]
    assert t.sequence_of(0) == "ACGTACGT"
