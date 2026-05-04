"""``Annotation`` container + GFF3 reader/writer + BED reader."""

from __future__ import annotations

import gzip
import json
import textwrap
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.annotation.io import (
    load_annotation,
    save_annotation,
)
from constellation.sequencing.readers.bed import read_bed
from constellation.sequencing.readers.gff import iter_gff3_lines, read_gff3, write_gff3
from constellation.sequencing.reference.reference import GenomeReference


def _make_genome() -> GenomeReference:
    return GenomeReference(
        contigs=pa.table(
            {
                "contig_id": pa.array([0], type=pa.int64()),
                "name": ["chr1"],
                "length": pa.array([100], type=pa.int64()),
            }
        ),
        sequences=pa.table(
            {
                "contig_id": pa.array([0], type=pa.int64()),
                "sequence": ["A" * 100],
            }
        ),
    )


def _make_annotation() -> Annotation:
    features = pa.table(
        {
            "feature_id": pa.array([0, 1, 2, 3], type=pa.int64()),
            "contig_id": pa.array([0, 0, 0, 0], type=pa.int64()),
            "start": pa.array([0, 0, 0, 50], type=pa.int64()),
            "end": pa.array([100, 100, 30, 100], type=pa.int64()),
            "strand": ["+", "+", "+", "+"],
            "type": ["gene", "mRNA", "exon", "exon"],
            "name": ["G1", "G1.1", None, None],
            "parent_id": pa.array([None, 0, 1, 1], type=pa.int64()),
        }
    )
    return Annotation(features=features)


def test_construct_validates():
    a = _make_annotation()
    assert a.n_features == 4


def test_duplicate_feature_id_rejected():
    features = pa.table(
        {
            "feature_id": pa.array([0, 0], type=pa.int64()),
            "contig_id": pa.array([0, 0], type=pa.int64()),
            "start": pa.array([0, 0], type=pa.int64()),
            "end": pa.array([1, 1], type=pa.int64()),
            "strand": ["+", "+"],
            "type": ["x", "x"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        Annotation(features=features)


def test_dangling_parent_id_rejected():
    features = pa.table(
        {
            "feature_id": pa.array([0], type=pa.int64()),
            "contig_id": pa.array([0], type=pa.int64()),
            "start": pa.array([0], type=pa.int64()),
            "end": pa.array([1], type=pa.int64()),
            "strand": ["+"],
            "type": ["exon"],
            "parent_id": pa.array([99], type=pa.int64()),
        }
    )
    with pytest.raises(ValueError, match="parent_id"):
        Annotation(features=features)


def test_validate_against_genome():
    g = _make_genome()
    a = _make_annotation()
    a.validate_against(g)


def test_validate_against_genome_missing_contig():
    g = _make_genome()
    bad = pa.table(
        {
            "feature_id": pa.array([0], type=pa.int64()),
            "contig_id": pa.array([99], type=pa.int64()),
            "start": pa.array([0], type=pa.int64()),
            "end": pa.array([1], type=pa.int64()),
            "strand": ["+"],
            "type": ["gene"],
        }
    )
    a = Annotation(features=bad)
    with pytest.raises(ValueError, match="absent from GenomeReference"):
        a.validate_against(g)


def test_features_on_and_features_of_type():
    a = _make_annotation()
    assert a.features_on(0).num_rows == 4
    assert a.features_of_type("gene").num_rows == 1
    assert a.features_of_type("exon").num_rows == 2


def test_parquetdir_roundtrip(tmp_path: Path):
    a = _make_annotation()
    save_annotation(a, tmp_path / "ann")
    a2 = load_annotation(tmp_path / "ann")
    assert a2.n_features == a.n_features


# ──────────────────────────────────────────────────────────────────────
# GFF3 reader
# ──────────────────────────────────────────────────────────────────────


def _gff3_text() -> str:
    return textwrap.dedent(
        """\
        ##gff-version 3
        ##sequence-region chr1 1 100
        # plain comment
        chr1\ttest\tgene\t1\t100\t.\t+\t.\tID=g1;Name=Gene1
        chr1\ttest\tmRNA\t1\t100\t0.99\t+\t.\tID=t1;Parent=g1;Name=Gene1.1
        chr1\ttest\texon\t1\t30\t.\t+\t0\tID=e1;Parent=t1
        chr1\ttest\texon\t51\t100\t.\t+\t0\tID=e2;Parent=t1;Note=second%20exon
        """
    )


def test_read_gff3_basic(tmp_path: Path):
    p = tmp_path / "x.gff3"
    p.write_text(_gff3_text())
    a = read_gff3(p)
    assert a.n_features == 4
    rows = a.features.to_pylist()
    # 1-based-incl → 0-based-half-open conversion
    gene_row = next(r for r in rows if r["type"] == "gene")
    assert gene_row["start"] == 0
    assert gene_row["end"] == 100


def test_read_gff3_resolves_parent_ids(tmp_path: Path):
    p = tmp_path / "x.gff3"
    p.write_text(_gff3_text())
    a = read_gff3(p)
    rows = a.features.to_pylist()
    gene_row = next(r for r in rows if r["type"] == "gene")
    mrna_row = next(r for r in rows if r["type"] == "mRNA")
    exon_rows = [r for r in rows if r["type"] == "exon"]
    assert gene_row["parent_id"] is None
    assert mrna_row["parent_id"] == gene_row["feature_id"]
    for er in exon_rows:
        assert er["parent_id"] == mrna_row["feature_id"]


def test_read_gff3_url_decode_attribute(tmp_path: Path):
    p = tmp_path / "x.gff3"
    p.write_text(_gff3_text())
    a = read_gff3(p)
    second_exon = next(
        r
        for r in a.features.to_pylist()
        if r["type"] == "exon" and r["attributes_json"]
    )
    payload = json.loads(second_exon["attributes_json"])
    assert payload["Note"] == "second exon"


def test_read_gff3_gzip(tmp_path: Path):
    p = tmp_path / "x.gff3.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(_gff3_text())
    a = read_gff3(p)
    assert a.n_features == 4


def test_read_gff3_fasta_directive_terminates(tmp_path: Path):
    text = (
        _gff3_text()
        + "##FASTA\n>chr1\nACGTACGTACGT\n"
    )
    p = tmp_path / "x.gff3"
    p.write_text(text)
    a = read_gff3(p)
    assert a.n_features == 4
    assert a.metadata_extras.get("saw_fasta_directive") is True


def test_read_gff3_empty(tmp_path: Path):
    p = tmp_path / "empty.gff3"
    p.write_text("##gff-version 3\n# only directives + a comment\n")
    a = read_gff3(p)
    assert a.n_features == 0


def test_write_gff3_roundtrip(tmp_path: Path):
    g = _make_genome()
    a = read_gff3_with_chr_map(_gff3_text(), tmp_path)
    out = tmp_path / "out.gff3"
    write_gff3(a, out, genome=g)
    a2 = read_gff3(out)
    assert a2.n_features == a.n_features
    # Coordinates round-trip cleanly (gff3 1-based-incl → 0-based-half-open
    # → 1-based-incl on write → 0-based-half-open on re-read).
    types1 = sorted(a.features.column("type").to_pylist())
    types2 = sorted(a2.features.column("type").to_pylist())
    assert types1 == types2


def read_gff3_with_chr_map(text: str, tmp_path: Path) -> Annotation:
    p = tmp_path / "in.gff3"
    p.write_text(text)
    return read_gff3(p, contig_name_to_id={"chr1": 0})


def test_iter_gff3_lines_skips_directives(tmp_path: Path):
    p = tmp_path / "x.gff3"
    p.write_text(_gff3_text())
    lines = list(iter_gff3_lines(p))
    # 4 data lines + 0 directive/comment lines
    assert len(lines) == 4


# ──────────────────────────────────────────────────────────────────────
# BED reader
# ──────────────────────────────────────────────────────────────────────


def test_read_bed_basic(tmp_path: Path):
    p = tmp_path / "x.bed"
    p.write_text("chr1\t0\t100\tregion1\t500\t+\nchr1\t200\t300\t.\t.\t-\n")
    a = read_bed(p, feature_type="enhancer", contig_name_to_id={"chr1": 0})
    assert a.n_features == 2
    rows = a.features.to_pylist()
    assert rows[0]["start"] == 0
    assert rows[0]["end"] == 100
    assert rows[0]["type"] == "enhancer"
    assert rows[0]["name"] == "region1"
    assert rows[0]["score"] == 500.0
    assert rows[1]["strand"] == "-"
    assert rows[1]["name"] is None
