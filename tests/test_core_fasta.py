"""``constellation.core.io.fasta`` and ``core.io.text``: the package's one
FASTA reader and writer, held to the private copies it replaced."""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.core.io.fasta import (
    FASTA_SCHEMA,
    FastaRecord,
    count_fasta_records,
    format_fasta,
    iter_fasta,
    iter_fasta_batches,
    parse_header,
    read_fasta,
    split_header,
    write_fasta,
)
from constellation.core.io.text import is_gzip, open_text

# A file with every rule in it: CRLF, a comment, blank lines, padding and
# tabs inside a sequence, trailing header whitespace, a lowercase residue, an
# empty header, sequence text before the first header, and a last record
# with no trailing newline.
TRICKY = (
    "stray sequence before any header\n"
    "; a comment\r\n"
    ">sp|P02769|ALBU_BOVIN Albumin OS=Bos taurus OX=9913 GN=ALB PE=1 SV=4  \r\n"
    "MKWVT FISLL\r\n"
    "\tLFSSAYS\r\n"
    "\r\n"
    "; another comment\n"
    ">second\n"
    "acgt\n"
    ">\n"
    "PEP\n"
    ">last record\twith tab\n"
    "XYZ"
)


def _old_fastx_iter(text: str):
    """``sequencing.readers.fastx._iter_fasta`` as it was before
    2026-11-05, over a string: the rules the shared parser must keep."""
    header = None
    chunks: list[str] = []
    out = []
    for raw in io.StringIO(text):
        line = raw.rstrip("\r\n")
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(chunks).replace(" ", "").replace("\t", "")))
            header = line[1:].rstrip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header is not None:
        out.append((header, "".join(chunks).replace(" ", "").replace("\t", "")))
    return out


@pytest.fixture
def tricky(tmp_path: Path) -> Path:
    p = tmp_path / "tricky.fasta"
    p.write_bytes(TRICKY.encode())
    return p


# ── reading ───────────────────────────────────────────────────────────


def test_iter_fasta_keeps_the_old_parser_rules(tricky: Path) -> None:
    got = [(r.header, r.sequence) for r in iter_fasta(tricky)]
    assert got == _old_fastx_iter(TRICKY)
    assert got[0] == (
        "sp|P02769|ALBU_BOVIN Albumin OS=Bos taurus OX=9913 GN=ALB PE=1 SV=4",
        "MKWVTFISLLLFSSAYS",
    )
    assert got[1] == ("second", "acgt")  # case is the caller's business
    assert got[2] == ("", "PEP")
    assert got[3] == ("last record\twith tab", "XYZ")


def test_record_id_and_description() -> None:
    r = FastaRecord("sp|P1|X_Y  Some  protein", "AAA")
    assert r.id == "sp|P1|X_Y"
    assert r.description == "Some  protein"
    assert split_header("") == ("", "")
    assert split_header("only") == ("only", "")


@pytest.mark.parametrize("name", ["plain.fasta.gz", "no_suffix.fasta"])
def test_gzip_is_recognised_by_its_bytes(tmp_path: Path, name: str) -> None:
    p = tmp_path / name
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(TRICKY)
    assert is_gzip(p)
    assert [(r.header, r.sequence) for r in iter_fasta(p)] == _old_fastx_iter(TRICKY)
    with open_text(p) as fh:  # text mode: universal newlines, as for a plain file
        assert fh.read() == TRICKY.replace("\r\n", "\n")


def test_uncompressed_file_named_gz_still_reads(tmp_path: Path) -> None:
    p = tmp_path / "mislabelled.fa.gz"
    p.write_text(">a\nAC\n")
    assert not is_gzip(p)
    assert [r.sequence for r in iter_fasta(p)] == ["AC"]


def test_read_fasta_and_batches(tricky: Path) -> None:
    table = read_fasta(tricky)
    assert table.schema == FASTA_SCHEMA
    assert table.column("id").to_pylist() == ["sp|P02769|ALBU_BOVIN", "second", "", "last"]
    assert table.column("description").to_pylist()[3] == "record\twith tab"
    batches = list(iter_fasta_batches(tricky, batch_rows=3))
    assert [b.num_rows for b in batches] == [3, 1]
    assert pa.Table.from_batches(batches).equals(table)
    with pytest.raises(ValueError):
        list(iter_fasta_batches(tricky, batch_rows=0))


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.fa"
    p.write_text("")
    assert list(iter_fasta(p)) == []
    assert read_fasta(p).num_rows == 0
    assert count_fasta_records(p) == 0


def test_count_fasta_records(tricky: Path) -> None:
    assert count_fasta_records(tricky) == 4


# ── writing ───────────────────────────────────────────────────────────


def test_named_records_skip_empty_like_the_cluster_writers(tmp_path: Path) -> None:
    # `_io._write_fasta` / `em/outputs._write_fasta` before 2026-11-05.
    ids, seqs = ["cluster_0", "cluster_1", "cluster_2"], ["ACGT", "", None]
    p = tmp_path / "cluster.fa"
    assert write_fasta(p, seqs, ids, skip_empty=True) == 1
    assert p.read_text() == ">cluster_0\nACGT\n"


def test_row_named_arrow_column_like_the_corpus(tmp_path: Path) -> None:
    # `em/corpus._write_fasta_rows` / `em/templates`: `>{row}`, every record,
    # in chunked slices of an Arrow column.
    seqs = [f"{'ACGT'[i % 4]}" * (i + 1) for i in range(23)]
    col = pa.chunked_array([pa.array(seqs[:10], pa.large_string()), pa.array(seqs[10:], pa.large_string())])
    fh = io.StringIO()
    n = write_fasta(fh, col, start_row=5, chunk_rows=4)
    n += write_fasta(fh, pa.array(["TT"]), start_row=5 + n)
    expected = "".join(f">{5 + i}\n{s}\n" for i, s in enumerate(seqs)) + ">28\nTT\n"
    assert (n, fh.getvalue()) == (24, expected)


def test_descriptions_like_the_genome_guided_cluster_fa() -> None:
    text = format_fasta(
        ["AC", "", "GT"],
        ["cluster_1", "cluster_2", "cluster_3"],
        descriptions=["representative=r1", "representative=r2", ""],
        skip_empty=True,
    )
    assert text == ">cluster_1 representative=r1\nAC\n>cluster_3\nGT\n"


def test_format_fasta_like_the_demux_protein_records() -> None:
    # `demux/quant.fasta_records_to_text`: every record, empty ones included.
    assert format_fasta(["MK", ""], ["P0", "P1"]) == ">P0\nMK\n>P1\n\n"


def test_long_sequence_written_unchanged(tmp_path: Path) -> None:
    contig = "ACGT" * (1 << 19) + "A"  # past the 1 MiB single-write threshold
    p = tmp_path / "genome.fa"
    write_fasta(p, (s for s in [contig, "GG"]), ["chr1", "chr2"])
    assert p.read_text() == f">chr1\n{contig}\n>chr2\nGG\n"


def test_lengths_must_match() -> None:
    with pytest.raises(ValueError):
        format_fasta(["A", "C"], ["only_one"])
    with pytest.raises(ValueError):
        format_fasta(["A", "C"], ["a", "c"], descriptions=["d"])


def test_atomic_write(tmp_path: Path) -> None:
    p = tmp_path / "out.fa"
    write_fasta(p, ["A"], ["a"], atomic=True)
    assert p.read_text() == ">a\nA\n"
    assert not (tmp_path / "out.fa.tmp").exists()

    def boom():
        yield "C"
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        write_fasta(p, boom(), atomic=True)
    assert p.read_text() == ">a\nA\n"  # the old file is untouched
    assert not (tmp_path / "out.fa.tmp").exists()
    with pytest.raises(ValueError):
        write_fasta(io.StringIO(), ["A"], atomic=True)


def test_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "rt.fa"
    write_fasta(p, ["MKV", "PEPTIDE"], ["a", "b"], descriptions=["first one", None])
    assert [(r.id, r.description, r.sequence) for r in iter_fasta(p)] == [
        ("a", "first one", "MKV"),
        ("b", "", "PEPTIDE"),
    ]


# ── headers ───────────────────────────────────────────────────────────


def test_uniprot_header() -> None:
    h = parse_header(">sp|P02769|ALBU_BOVIN Albumin OS=Bos taurus OX=9913 GN=ALB PE=1 SV=4")
    assert (h.database, h.accession, h.entry_name) == ("sp", "P02769", "ALBU_BOVIN")
    assert h.tags == (("OS", "Bos taurus"), ("OX", "9913"), ("GN", "ALB"), ("PE", "1"), ("SV", "4"))
    assert h.gene == "ALB"
    assert h.tag("OX") == "9913"
    assert h.tag("missing") is None


def test_ncbi_and_bare_tags_in_order() -> None:
    h = parse_header("XP_1.1 avg_tpm=1.5 [gene=ALB] [organism=Homo sapiens] source=refseq")
    assert h.accession == "XP_1.1" and h.database is None and h.entry_name is None
    assert h.tags == (
        ("avg_tpm", "1.5"),
        ("gene", "ALB"),
        ("organism", "Homo sapiens"),
        ("source", "refseq"),
    )
    assert h.gene == "ALB"


def test_gene_is_one_symbol() -> None:
    assert parse_header("x GN=ABC extra words").gene == "ABC"
    assert parse_header("x gene=").gene is None
    assert parse_header("x no tags here").gene is None


# ── the gene maps, against their old implementations ──────────────────


def _old_gene_map(text: str) -> dict[str, str]:
    """``novel.build_gene_map_from_fasta_headers`` before 2026-11-05."""
    out: dict[str, str] = {}
    for line in io.StringIO(text):
        if not line.startswith(">"):
            continue
        tokens = line[1:].rstrip("\n").split()
        if not tokens:
            continue
        accession, gene = tokens[0], None
        for token in tokens[1:]:
            bare = token.strip("[]")
            if bare.startswith("gene="):
                gene = bare[len("gene="):]
                break
            if bare.startswith("GN="):
                gene = bare[len("GN="):]
                break
        if gene and gene.lower() != "unknown":
            out[accession] = gene
            if "|" in accession:
                parts = accession.split("|")
                if len(parts) >= 2 and parts[1]:
                    out.setdefault(parts[1], gene)
    return out


def _old_swissprot_map(text: str) -> dict[str, str]:
    """``novel.protein_to_gene_from_swissprot`` before 2026-11-05."""
    mapping: dict[str, str] = {}
    for line in io.StringIO(text):
        if not line.startswith(">"):
            continue
        header = line[1:].rstrip("\n")
        parts = header.split("|")
        key = parts[1] if len(parts) >= 2 and parts[1] else (header.split()[0] if header.split() else "")
        if not key:
            continue
        m = re.search(r"\bGN=(\S+)", header)
        if m:
            mapping[key] = m.group(1)
    return mapping


GENE_HEADERS = (
    ">sp|P02769|ALBU_BOVIN Albumin OS=Bos taurus OX=9913 GN=ALB PE=1 SV=4\nMK\n"
    ">tr|A0A0|A0A0_HUMAN Uncharacterized OS=Homo sapiens OX=9606 PE=4 SV=1\nMK\n"
    ">novel_1 avg_tpm=3.0 [gene=Akap4] source=transcriptome\nMK\n"
    ">novel_2 [gene=unknown] source=transcriptome\nMK\n"
    ">legacy_3 gene=Prm1\nMK\n"
    ">\nMK\n"
)


def test_gene_maps_match_their_old_implementations(tmp_path: Path) -> None:
    from constellation.massspec.search.novel import (
        build_gene_map_from_fasta_headers,
        protein_to_gene_from_swissprot,
    )

    p = tmp_path / "genes.fasta"
    p.write_text(GENE_HEADERS)
    assert build_gene_map_from_fasta_headers([p]) == _old_gene_map(GENE_HEADERS)
    assert protein_to_gene_from_swissprot(p) == _old_swissprot_map(GENE_HEADERS)
    # and now gzip and CRLF, which the old ones could not read
    gz = tmp_path / "genes.fasta.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write(GENE_HEADERS.replace("\n", "\r\n"))
    assert build_gene_map_from_fasta_headers([gz]) == _old_gene_map(GENE_HEADERS)
    assert protein_to_gene_from_swissprot(gz) == _old_swissprot_map(GENE_HEADERS)


def test_read_fasta_proteins_strips_crlf(tmp_path: Path) -> None:
    from constellation.massspec.search.novel import read_fasta_proteins

    p = tmp_path / "crlf.fasta"
    p.write_bytes(b">sp|P1|A_B desc\r\nmkv\r\nPEP\r\n>P2\r\nAAA\r\n")
    t = read_fasta_proteins(p)
    assert t.column("protein_id").to_pylist() == ["sp|P1|A_B", "P2"]
    assert t.column("sequence").to_pylist() == ["MKVPEP", "AAA"]  # no '\r', upper-cased
