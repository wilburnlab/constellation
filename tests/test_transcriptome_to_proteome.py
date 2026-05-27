"""Tier A tests for the transcriptome→proteomics pipeline helpers.

Covers:
  * filter_and_write_novel_fasta — TPM filter + reference dedup
  * apply_alignment_filter — best-hit-per-query + threshold + semi-join
  * read_reference_gene_map — GFF3 + GBFF parsers
  * build_combined_fasta — header annotation correctness, dedup behaviour
  * write_competitive_target_fasta — record concatenation + count
  * collect_protein_ids — FASTA header parsing
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.core.io.schemas import ALIGNMENT_HIT_TABLE
from constellation.sequencing.quant.protein_counts import (
    PROTEIN_COUNTS_LONG_SCHEMA,
)
from constellation.transcriptome_to_proteome import (
    _tag_aligned_to,
    apply_alignment_filter,
    build_combined_fasta,
    collect_protein_ids,
    filter_and_write_novel_fasta,
    read_reference_gene_map,
    write_competitive_target_fasta,
)


# ──────────────────────────────────────────────────────────────────────
# _tag_aligned_to — source tagging from reference membership
# ──────────────────────────────────────────────────────────────────────


def test_tag_aligned_to_by_membership(tmp_path: Path) -> None:
    """Targets in the reference proteome → 'refseq'; everything else
    (i.e. SwissProt) → 'swissprot'."""
    import pyarrow as pa

    ref = tmp_path / "ref.fasta"
    ref.write_text(">NP_001\nMKAAA\n>NP_002\nMKBBB\n")
    hits = pa.table(
        {
            "query": pa.array(["N1", "N2", "N3"]),
            "target": pa.array(["NP_001", "SP_Q9Y6K9", "NP_002"]),
            "evalue": pa.array([1e-50, 1e-30, 1e-40]),
        }
    )
    tagged = _tag_aligned_to(hits, ref)
    assert tagged.column("aligned_to").to_pylist() == [
        "refseq",
        "swissprot",
        "refseq",
    ]


def test_tag_aligned_to_replaces_existing(tmp_path: Path) -> None:
    """Any pre-existing aligned_to column is overwritten by the
    membership-derived value."""
    import pyarrow as pa

    ref = tmp_path / "ref.fasta"
    ref.write_text(">NP_001\nMKAAA\n")
    hits = pa.table(
        {
            "query": pa.array(["N1"]),
            "target": pa.array(["NP_001"]),
            "evalue": pa.array([1e-50]),
            "aligned_to": pa.array(["stale"]),
        }
    )
    tagged = _tag_aligned_to(hits, ref)
    assert tagged.column("aligned_to").to_pylist() == ["refseq"]


# ──────────────────────────────────────────────────────────────────────
# filter_and_write_novel_fasta
# ──────────────────────────────────────────────────────────────────────


def _make_proteins_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as fh:
        for pid, seq in records:
            fh.write(f">{pid}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60])
                fh.write("\n")


def test_filter_and_write_novel_fasta_basic(tmp_path: Path) -> None:
    """P0 + P1 above min_avg_tpm; P2 below; P0 hash matches reference."""
    counts = pa.table(
        {
            "protein_id": ["P0", "P0", "P1", "P1", "P2"],
            "sequence": [
                "AAAAAAAAAA",
                "AAAAAAAAAA",
                "BBBBBBBBBB",
                "BBBBBBBBBB",
                "CCCCCCCCCC",
            ],
            "sample_name": ["s1", "s2", "s1", "s2", "s1"],
            "count": [10, 10, 5, 5, 100],
            "tpm": [500000.0, 500000.0, 250000.0, 250000.0, 500000.0],
        }
    )
    proteins = tmp_path / "proteins.fasta"
    _make_proteins_fasta(
        proteins,
        [("P0", "AAAAAAAAAA"), ("P1", "BBBBBBBBBB"), ("P2", "CCCCCCCCCC")],
    )
    # Reference contains P0's sequence — P0 should be deduped out as novel.
    reference = tmp_path / "reference.fasta"
    _make_proteins_fasta(reference, [("REF_A", "AAAAAAAAAA")])
    out = tmp_path / "novel.fasta"

    novel_table, n_written = filter_and_write_novel_fasta(
        counts_tpm=counts,
        proteins_fasta=proteins,
        reference_fasta=reference,
        output_path=out,
        min_avg_tpm=1.0,
    )
    # P0 dedup'd (matches reference); P2 below 1.0 TPM (avg = 500000…
    # actually wait, P2 has 100 count → tpm 500000). Let me reconsider:
    # min_avg_tpm=1.0 is very permissive; only P0 should drop via dedup.
    assert n_written == 2
    ids = set(novel_table.column("protein_id").to_pylist())
    assert "P0" not in ids   # deduped
    assert ids == {"P1", "P2"}
    # FASTA file on disk has 2 records.
    content = out.read_text()
    assert content.count(">") == 2


def test_filter_and_write_novel_fasta_tpm_threshold(tmp_path: Path) -> None:
    """min_avg_tpm above one protein's value → that protein drops."""
    counts = pa.table(
        {
            "protein_id": ["P0", "P1"],
            "sequence": ["AAA", "BBB"],
            "sample_name": ["s1", "s1"],
            "count": [9, 1],
            "tpm": [900000.0, 100000.0],
        }
    )
    proteins = tmp_path / "proteins.fasta"
    _make_proteins_fasta(proteins, [("P0", "AAAAAAAAAA"), ("P1", "BBBBBBBBBB")])
    reference = tmp_path / "reference.fasta"
    reference.write_text("")  # empty reference
    out = tmp_path / "novel.fasta"
    novel_table, n_written = filter_and_write_novel_fasta(
        counts_tpm=counts,
        proteins_fasta=proteins,
        reference_fasta=reference,
        output_path=out,
        min_avg_tpm=500000.0,
    )
    assert n_written == 1
    assert novel_table.column("protein_id").to_pylist() == ["P0"]


def test_filter_and_write_novel_fasta_all_filtered_empty(tmp_path: Path) -> None:
    counts = pa.table(
        {
            "protein_id": ["P0"],
            "sequence": ["AAA"],
            "sample_name": ["s1"],
            "count": [0],
            "tpm": [0.0],
        }
    )
    proteins = tmp_path / "proteins.fasta"
    _make_proteins_fasta(proteins, [("P0", "AAAAAAAAAA")])
    reference = tmp_path / "reference.fasta"
    reference.write_text("")
    out = tmp_path / "novel.fasta"
    novel_table, n_written = filter_and_write_novel_fasta(
        counts_tpm=counts,
        proteins_fasta=proteins,
        reference_fasta=reference,
        output_path=out,
        min_avg_tpm=1.0,
    )
    assert n_written == 0
    assert novel_table.num_rows == 0


# ──────────────────────────────────────────────────────────────────────
# apply_alignment_filter
# ──────────────────────────────────────────────────────────────────────


def _make_alignment_hits(rows: list[dict]) -> pa.Table:
    cols = {f.name: [] for f in ALIGNMENT_HIT_TABLE}
    for row in rows:
        for name in cols:
            cols[name].append(row.get(name))
    return pa.table(
        {
            name: pa.array(values, type=ALIGNMENT_HIT_TABLE.field(name).type)
            for name, values in cols.items()
        },
        schema=ALIGNMENT_HIT_TABLE,
    )


def _make_novel_table(rows: list[tuple[str, str]]) -> pa.Table:
    return pa.table(
        {
            "protein_id": pa.array([r[0] for r in rows], type=pa.string()),
            "sequence": pa.array([r[1] for r in rows], type=pa.string()),
        }
    )


def test_apply_alignment_filter_keeps_passing(tmp_path: Path) -> None:
    novel = _make_novel_table([("N1", "AAA"), ("N2", "BBB"), ("N3", "CCC")])
    hits = _make_alignment_hits(
        [
            {"query": "N1", "target": "REF_A", "evalue": 1e-30,
             "qstart": 1, "qend": 10, "tstart": 1, "tend": 10,
             "cigar": "10M", "aligned_to": "reference"},
            # N2 has only a poor hit (above the 1e-20 threshold).
            {"query": "N2", "target": "REF_B", "evalue": 1e-5,
             "qstart": 1, "qend": 10, "tstart": 1, "tend": 10,
             "cigar": "10M", "aligned_to": "reference"},
            # N3 has no hits.
        ]
    )
    out = apply_alignment_filter(
        novel_table=novel, alignment_hits=hits, evalue_threshold=1e-20,
    )
    assert out.column("protein_id").to_pylist() == ["N1"]


def test_apply_alignment_filter_best_hit_per_query(tmp_path: Path) -> None:
    """Multiple hits per query: best (min evalue) is used for the
    threshold check."""
    novel = _make_novel_table([("N1", "AAA")])
    hits = _make_alignment_hits(
        [
            # Poor hit + good hit for N1 → best is 1e-30, passes.
            {"query": "N1", "target": "REF_A", "evalue": 1e-5,
             "qstart": 1, "qend": 10, "tstart": 1, "tend": 10,
             "cigar": "10M", "aligned_to": "reference"},
            {"query": "N1", "target": "REF_B", "evalue": 1e-30,
             "qstart": 1, "qend": 10, "tstart": 1, "tend": 10,
             "cigar": "10M", "aligned_to": "reference"},
        ]
    )
    out = apply_alignment_filter(
        novel_table=novel, alignment_hits=hits, evalue_threshold=1e-20,
    )
    assert out.num_rows == 1


def test_apply_alignment_filter_empty_hits(tmp_path: Path) -> None:
    """No hits → no novel proteins survive."""
    novel = _make_novel_table([("N1", "AAA")])
    hits = ALIGNMENT_HIT_TABLE.empty_table()
    out = apply_alignment_filter(
        novel_table=novel, alignment_hits=hits, evalue_threshold=1e-20,
    )
    assert out.num_rows == 0


# ──────────────────────────────────────────────────────────────────────
# read_reference_gene_map
# ──────────────────────────────────────────────────────────────────────


_GFF3_FIXTURE = (
    "##gff-version 3\n"
    "chr1\ttest\tgene\t1\t100\t.\t+\t.\tID=gene1;Name=ACTB\n"
    "chr1\ttest\tCDS\t1\t100\t.\t+\t0\tID=cds1;Parent=mrna1;protein_id=NP_001185763.1;gene=ACTB\n"
    "chr1\ttest\tCDS\t200\t300\t.\t+\t0\tID=cds2;Parent=mrna2;protein_id=NP_002.1;gene=GAPDH\n"
    # CDS with only protein_id (no gene) — skipped.
    "chr1\ttest\tCDS\t400\t500\t.\t+\t0\tID=cds3;Parent=mrna3;protein_id=NP_003.1\n"
    # Non-CDS feature with protein_id — skipped.
    "chr1\ttest\texon\t1\t100\t.\t+\t.\tID=exon1;Parent=mrna1;protein_id=NP_001185763.1;gene=ACTB\n"
)


def test_read_reference_gene_map_gff3(tmp_path: Path) -> None:
    p = tmp_path / "ref.gff3"
    p.write_text(_GFF3_FIXTURE)
    out = read_reference_gene_map(p)
    assert out == {
        "NP_001185763.1": "ACTB",
        "NP_002.1": "GAPDH",
    }


def test_read_reference_gene_map_gff3_gz(tmp_path: Path) -> None:
    import gzip as _gzip
    p = tmp_path / "ref.gff3.gz"
    p.write_bytes(_gzip.compress(_GFF3_FIXTURE.encode()))
    out = read_reference_gene_map(p)
    assert out["NP_001185763.1"] == "ACTB"


_GBFF_FIXTURE = """LOCUS       NM_001 1500 bp linear
FEATURES             Location/Qualifiers
     source          1..1500
                     /organism="Homo sapiens"
     gene            1..1500
                     /gene="ACTB"
     CDS             1..345
                     /gene="ACTB"
                     /protein_id="NP_001185763.1"
                     /product="actin beta"
     CDS             400..600
                     /gene="GAPDH"
                     /protein_id="NP_002.1"
                     /product="GAPDH"
ORIGIN
        1 atgaaa
//
LOCUS       NM_002 1000 bp linear
FEATURES             Location/Qualifiers
     CDS             1..200
                     /gene="HBB"
                     /protein_id="NP_003.1"
ORIGIN
//
"""


def test_read_reference_gene_map_gbff(tmp_path: Path) -> None:
    p = tmp_path / "ref.gbff"
    p.write_text(_GBFF_FIXTURE)
    out = read_reference_gene_map(p)
    assert out == {
        "NP_001185763.1": "ACTB",
        "NP_002.1": "GAPDH",
        "NP_003.1": "HBB",
    }


def test_read_reference_gene_map_unsupported_ext(tmp_path: Path) -> None:
    p = tmp_path / "bogus.txt"
    p.write_text("")
    with pytest.raises(ValueError, match="unsupported annotation path"):
        read_reference_gene_map(p)


# ──────────────────────────────────────────────────────────────────────
# build_combined_fasta
# ──────────────────────────────────────────────────────────────────────


def test_build_combined_fasta_headers(tmp_path: Path) -> None:
    """Every emitted record carries [gene=...]/[accession=...] tags
    (when available) + source=refseq|transcriptome."""
    ref = tmp_path / "reference.fasta"
    _make_proteins_fasta(
        ref,
        [("REF_A", "AAAAAAAAAA"), ("REF_B", "BBBBBBBBBB")],
    )
    novel = _make_novel_table([("N1", "CCCCCCCCCC")])
    hits = _make_alignment_hits(
        [
            {"query": "N1", "target": "REF_A", "evalue": 1e-30,
             "qstart": 1, "qend": 10, "tstart": 1, "tend": 10,
             "cigar": "10M", "aligned_to": "reference"},
        ]
    )
    gene_map = {"REF_A": "ACTB", "REF_B": "GAPDH"}
    out = tmp_path / "combined.fasta"
    n_written, stats = build_combined_fasta(
        reference_fasta=ref,
        filtered_novel=novel,
        alignment_hits=hits,
        reference_gene_map=gene_map,
        output_path=out,
    )
    assert n_written == 3
    assert stats["reference"] == 2
    assert stats["novel"] == 1
    text = out.read_text()
    # Reference records.
    assert ">REF_A [gene=ACTB] [accession=REF_A] source=refseq" in text
    assert ">REF_B [gene=GAPDH] [accession=REF_B] source=refseq" in text
    # Novel record inherits ACTB from its REF_A target.
    assert (
        ">N1 [gene=ACTB] [accession=REF_A] [aligned_to=reference] "
        "source=transcriptome"
    ) in text


def test_build_combined_fasta_dedup_against_self(tmp_path: Path) -> None:
    """Duplicate sequences within the reference FASTA are dropped."""
    ref = tmp_path / "reference.fasta"
    _make_proteins_fasta(
        ref,
        [
            ("REF_A", "AAAAAAAAAA"),    # unique
            ("REF_A_iso", "AAAAAAAAAA"),  # isoform with identical sequence
            ("REF_B", "BBBBBBBBBB"),    # unique
        ],
    )
    novel = pa.table(
        {
            "protein_id": pa.array([], type=pa.string()),
            "sequence": pa.array([], type=pa.string()),
        }
    )
    hits = ALIGNMENT_HIT_TABLE.empty_table()
    out = tmp_path / "combined.fasta"
    n_written, stats = build_combined_fasta(
        reference_fasta=ref,
        filtered_novel=novel,
        alignment_hits=hits,
        reference_gene_map={},
        output_path=out,
    )
    assert n_written == 2  # 3 → 2 after dedup
    assert stats["duplicate_dropped"] == 1


def test_build_combined_fasta_no_gene_count(tmp_path: Path) -> None:
    """Reference records without a gene mapping get counted as
    no_gene + emitted without the [gene=...] tag."""
    ref = tmp_path / "reference.fasta"
    _make_proteins_fasta(ref, [("REF_A", "AAA")])
    novel = _make_novel_table([])
    hits = ALIGNMENT_HIT_TABLE.empty_table()
    out = tmp_path / "combined.fasta"
    n_written, stats = build_combined_fasta(
        reference_fasta=ref,
        filtered_novel=novel,
        alignment_hits=hits,
        reference_gene_map={},   # empty map
        output_path=out,
    )
    assert stats["no_gene"] == 1
    text = out.read_text()
    assert "[accession=REF_A]" in text
    assert "[gene=" not in text
    assert "source=refseq" in text


# ──────────────────────────────────────────────────────────────────────
# write_competitive_target_fasta + collect_protein_ids
# ──────────────────────────────────────────────────────────────────────


def test_write_competitive_target_fasta(tmp_path: Path) -> None:
    ref = tmp_path / "reference.fasta"
    _make_proteins_fasta(ref, [("REF_A", "AAA"), ("REF_B", "BBB")])
    sprot = tmp_path / "sprot.fasta"
    _make_proteins_fasta(sprot, [("SP_A", "CCC"), ("SP_B", "DDD"), ("SP_C", "EEE")])
    out = tmp_path / "target.fasta"
    n = write_competitive_target_fasta(
        reference_fasta=ref,
        swissprot_fasta=sprot,
        output_path=out,
    )
    assert n == 5
    assert collect_protein_ids(out) == {"REF_A", "REF_B", "SP_A", "SP_B", "SP_C"}


def test_collect_protein_ids_from_complex_headers(tmp_path: Path) -> None:
    """First-token-of-header is the protein_id."""
    p = tmp_path / "fasta_with_desc.fasta"
    p.write_text(">sp|P12345|TEST extra info\nMKLIGHT\n>NP_001 [organism=Homo sapiens]\nACDEFG\n")
    assert collect_protein_ids(p) == {"sp|P12345|TEST", "NP_001"}


# ──────────────────────────────────────────────────────────────────────
# deduplicate_fasta + swissprot-aligned novel gene tagging
# ──────────────────────────────────────────────────────────────────────


def test_deduplicate_fasta_first_occurrence_wins(tmp_path: Path) -> None:
    """deduplicate_fasta drops identical-sequence entries (case-insensitive),
    keeps the first occurrence's header verbatim."""
    from constellation.transcriptome_to_proteome import deduplicate_fasta

    inp = tmp_path / "in.fasta"
    out = tmp_path / "out.fasta"
    inp.write_text(
        ">NP_001.1 protein A\nMAGCKLPLS\n"
        ">XP_001.1 predicted dup of A\nMAGCKLPLS\n"
        ">NP_002.1 protein B\nMAGCKLPLT\n"
        ">NP_003.1 same seq B lowercase\nmagcklplt\n"
    )
    n = deduplicate_fasta(inp, out)
    assert n == 2
    text = out.read_text()
    # First occurrence preserved with its full header
    assert ">NP_001.1 protein A" in text
    assert ">NP_002.1 protein B" in text
    # Duplicates dropped
    assert ">XP_001.1" not in text
    assert ">NP_003.1" not in text


def test_build_combined_fasta_swissprot_aligned_novel_gets_gene(tmp_path: Path) -> None:
    """When a novel ORF aligns to a SwissProt accession, build_combined_fasta
    must tag its header with the SwissProt gene symbol — using the merged
    gene_map (.gbff + GN= entries). This is the regression that drove the
    cartographer-style architecture: previously the gene_map was refseq-only
    and SwissProt-aligned novels had no [gene=X] tag."""
    from constellation.transcriptome_to_proteome import build_combined_fasta

    # One reference protein + two novel ORFs (one aligns to refseq, one to swissprot)
    ref = tmp_path / "reference.fasta"
    _make_proteins_fasta(ref, [("NP_REF1", "MAAAAA")])

    novel_table = pa.table(
        {
            "protein_id": pa.array(["NOV_REFHIT", "NOV_SPHIT"]),
            "sequence":   pa.array(["MBBBBB", "MCCCCC"]),
        }
    )

    alignment_hits = pa.table(
        {
            "query":  pa.array(["NOV_REFHIT", "NOV_SPHIT"]),
            "target": pa.array(["NP_REF1",   "P04483"]),  # bare UniProt acc
            "evalue": pa.array([1e-30, 1e-30]),
            "qstart": pa.array([1, 1]),
            "qend":   pa.array([6, 6]),
            "tstart": pa.array([1, 1]),
            "tend":   pa.array([6, 6]),
            "cigar":  pa.array(["6M", "6M"]),
            "aligned_to": pa.array(["reference", "swissprot"]),
        }
    )

    # Merged gene_map: refseq + swissprot. Keys match what mmseqs2 reports.
    gene_map = {"NP_REF1": "REFGENE", "P04483": "TETR2"}

    out = tmp_path / "combined.fasta"
    n_written, stats = build_combined_fasta(
        reference_fasta=ref,
        filtered_novel=novel_table,
        alignment_hits=alignment_hits,
        reference_gene_map=gene_map,
        output_path=out,
    )
    text = out.read_text()
    assert n_written == 3   # one ref + two novel
    assert "[gene=REFGENE]" in text       # reference protein tagged
    assert "[gene=TETR2]" in text         # SwissProt-aligned novel tagged
    assert stats["no_gene"] == 0          # nothing missed

    # Verify the SwissProt-aligned novel ORF specifically:
    sp_novel_lines = [l for l in text.splitlines()
                       if l.startswith(">NOV_SPHIT")]
    assert len(sp_novel_lines) == 1
    assert "[gene=TETR2]" in sp_novel_lines[0]
    assert "[accession=P04483]" in sp_novel_lines[0]
    assert "[aligned_to=swissprot]" in sp_novel_lines[0]


# ──────────────────────────────────────────────────────────────────────
# read_reference_gene_map — Constellation parquet bundle dispatch
# ──────────────────────────────────────────────────────────────────────


def test_read_reference_gene_map_dispatches_to_parquet(tmp_path: Path) -> None:
    """A directory containing ``features.parquet`` + ``manifest.json``
    is read via the Arrow path; the gene map matches what the GFF3
    parser would produce against the same source."""
    from constellation.sequencing.annotation.io import save_annotation
    from constellation.sequencing.readers.gff import read_gff3

    # Hand-built GFF3 with two protein-coding CDS records — one with a
    # ``gene=`` tag, one without (skipped by the parser).
    gff3 = tmp_path / "src.gff3"
    gff3.write_text(
        "##gff-version 3\n"
        "chr1\trefseq\tgene\t1\t300\t.\t+\t.\tID=g1;Name=ACTB\n"
        "chr1\trefseq\tmRNA\t1\t300\t.\t+\t.\tID=m1;Parent=g1\n"
        "chr1\trefseq\tCDS\t1\t300\t.\t+\t0\tID=c1;Parent=m1;protein_id=NP_0001;gene=ACTB\n"
        "chr2\trefseq\tgene\t1\t300\t.\t+\t.\tID=g2;Name=ORPHAN\n"
        "chr2\trefseq\tmRNA\t1\t300\t.\t+\t.\tID=m2;Parent=g2\n"
        "chr2\trefseq\tCDS\t1\t300\t.\t+\t0\tID=c2;Parent=m2;protein_id=NP_0002\n"
    )
    # GFF3 reader needs a contig-name → contig_id mapping (chr1/chr2 → 0/1)
    # to build a valid Annotation; we supply a fake one since the gene-map
    # extractor doesn't care about contig assignments.
    annotation = read_gff3(gff3, contig_name_to_id={"chr1": 0, "chr2": 1})

    annot_dir = tmp_path / "annotation"
    save_annotation(annotation, annot_dir)
    assert (annot_dir / "features.parquet").is_file()
    assert (annot_dir / "manifest.json").is_file()

    # Parquet-path gene map.
    parquet_map = read_reference_gene_map(annot_dir)
    assert parquet_map == {"NP_0001": "ACTB"}

    # GFF3-path gene map for cross-check — same source, same result.
    gff3_map = read_reference_gene_map(gff3)
    assert gff3_map == parquet_map


def test_read_reference_gene_map_rejects_unknown_path(tmp_path: Path) -> None:
    """Bare file with no recognised suffix and not a parquet bundle → ValueError."""
    weird = tmp_path / "annotation.dat"
    weird.write_text("not a real annotation")
    with pytest.raises(ValueError, match="unsupported annotation path"):
        read_reference_gene_map(weird)


def test_read_reference_gene_map_rejects_directory_without_features(
    tmp_path: Path,
) -> None:
    """A directory without features.parquet is not a Constellation bundle;
    falls through to the suffix dispatch which then fails."""
    empty_dir = tmp_path / "no-bundle"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="unsupported annotation path"):
        read_reference_gene_map(empty_dir)
