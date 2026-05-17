"""Unit tests for ``constellation.catalog``.

Per-source parsers are exercised against synthetic fixtures so the
suite runs offline; integration tests against the live endpoints are
deferred to a future ``CONSTELLATION_NETWORK_TESTS``-gated suite.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.catalog import (
    ASSEMBLY_CATALOG_TABLE,
    CATALOG_SOURCE_VOCAB,
    CatalogBundle,
    CatalogResolver,
    CatalogRow,
    catalogs_root,
    ensembl,
    ensembl_genomes,
    list_installed,
    read_catalog,
    refseq,
    release_dir,
    uniprot,
    write_catalog,
)
from constellation.core.io.schemas import get_schema


# ──────────────────────────────────────────────────────────────────────
# Schema registration
# ──────────────────────────────────────────────────────────────────────


def test_assembly_catalog_schema_registers():
    assert get_schema("AssemblyCatalogTable") is ASSEMBLY_CATALOG_TABLE


def test_catalog_source_vocab():
    assert "ensembl" in CATALOG_SOURCE_VOCAB
    assert "refseq" in CATALOG_SOURCE_VOCAB
    assert "uniprot" in CATALOG_SOURCE_VOCAB


# ──────────────────────────────────────────────────────────────────────
# Ensembl parser
# ──────────────────────────────────────────────────────────────────────


_ENSEMBL_FIXTURE = """\
#name\tspecies\tdivision\ttaxonomy_id\tassembly\tassembly_accession\tgenebuild\tvariation\tmicroarray\tpan_compara\tpeptide_compara\tgenome_alignments\tother_alignments\tcore_db\tspecies_id
Human\thomo_sapiens\tEnsemblVertebrates\t9606\tGRCh38.p14\tGCA_000001405.29\t2024-03\tY\tY\tY\tY\tY\tY\thomo_sapiens_core_111_38\t1
Mouse\tmus_musculus\tEnsemblVertebrates\t10090\tGRCm39\tGCA_000001635.9\t2024-02\tY\tY\tY\tY\tY\tY\tmus_musculus_core_111_39\t1
Zebrafish\tdanio_rerio\tEnsemblVertebrates\t7955\tGRCz11\tGCA_000002035.4\t2024-01\tY\tY\tY\tY\tY\tY\tdanio_rerio_core_111_11\t1
"""


def test_ensembl_parse_species_txt():
    tbl = ensembl.parse_species_txt(_ENSEMBL_FIXTURE, release=111)
    assert tbl.num_rows == 3
    species = tbl.column("species_name").to_pylist()
    assert "Homo sapiens" in species
    assert "Mus musculus" in species
    assert "Danio rerio" in species
    # Taxid round-trips.
    taxids = tbl.column("taxid").to_pylist()
    assert 9606 in taxids and 10090 in taxids and 7955 in taxids
    # URLs derived correctly.
    fasta_urls = tbl.column("fasta_url").to_pylist()
    assert any("homo_sapiens" in u and "GRCh38.p14" in u for u in fasta_urls)
    # Protein + cDNA URLs populated.
    assert all(u for u in tbl.column("protein_url").to_pylist())
    assert all(u for u in tbl.column("cdna_url").to_pylist())


def test_ensembl_schema_matches_canonical():
    tbl = ensembl.parse_species_txt(_ENSEMBL_FIXTURE, release=111)
    assert tbl.schema == ASSEMBLY_CATALOG_TABLE


# ──────────────────────────────────────────────────────────────────────
# Ensembl Genomes parser
# ──────────────────────────────────────────────────────────────────────


_EG_FIXTURE = """\
#name\tspecies\tdivision\ttaxonomy_id\tassembly\tassembly_accession\tgenebuild\tvariation\tmicroarray\tpan_compara\tpeptide_compara\tgenome_alignments\tother_alignments\tcore_db\tspecies_id
Saccharomyces cerevisiae\tsaccharomyces_cerevisiae\tEnsemblFungi\t4932\tR64-1-1\tGCA_000146045.2\t2024\t\t\t\t\t\t\tsaccharomyces_cerevisiae_core_57_4\t1
"""


def test_ensembl_genomes_parse_species_txt():
    tbl = ensembl_genomes.parse_species_txt(_EG_FIXTURE, division="fungi", release=57)
    assert tbl.num_rows == 1
    assert tbl.column("species_name").to_pylist() == ["Saccharomyces cerevisiae"]
    assert tbl.column("division").to_pylist() == ["fungi"]
    fasta = tbl.column("fasta_url").to_pylist()[0]
    assert "ensemblgenomes.org/pub/fungi/release-57" in fasta
    assert "saccharomyces_cerevisiae" in fasta


# ──────────────────────────────────────────────────────────────────────
# RefSeq parser
# ──────────────────────────────────────────────────────────────────────


_REFSEQ_FIXTURE = """\
##  See ftp://ftp.ncbi.nlm.nih.gov/genomes/all/README.txt for documentation.
#assembly_accession\tbioproject\tbiosample\twgs_master\trefseq_category\ttaxid\tspecies_taxid\torganism_name\tinfraspecific_name\tisolate\tversion_status\tassembly_level\trelease_type\tgenome_rep\tseq_rel_date\tasm_name\tsubmitter\tgbrs_paired_asm\tpaired_asm_comp\tftp_path\texcluded_from_refseq\trelation_to_type_material\tasm_not_live_date\tassembly_type\tgroup
GCF_000001405.40\tPRJNA168\tSAMN12121739\t\treference genome\t9606\t9606\tHomo sapiens\t\t\tlatest\tChromosome\tPatch\tFull\t2022/02/03\tGRCh38.p14\tGenome Reference Consortium\tGCA_000001405.29\tidentical\thttps://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14\t\t\tna\thaploid\tvertebrate_mammalian
GCF_000001635.27\tPRJNA169\tSAMN03457050\t\treference genome\t10090\t10090\tMus musculus\t\t\tlatest\tChromosome\tPatch\tFull\t2020/06/24\tGRCm39\tGenome Reference Consortium\tGCA_000001635.9\tidentical\thttps://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/GCF_000001635.27_GRCm39\t\t\tna\thaploid\tvertebrate_mammalian
GCF_000146045.2\tPRJNA128\tSAMEA3138177\t\treference genome\t559292\t4932\tSaccharomyces cerevisiae S288C\t\t\tlatest\tComplete Genome\tMajor\tFull\t2014/12/17\tR64\tSaccharomyces Genome Database\tGCA_000146045.2\tidentical\thttps://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64\t\t\tna\thaploid\tfungi
"""


def test_refseq_parse_summary():
    tbl = refseq.parse_summary(_REFSEQ_FIXTURE)
    assert tbl.num_rows == 3
    accessions = tbl.column("assembly_accession").to_pylist()
    assert "GCF_000001405.40" in accessions
    # FTP path → genomic.fna.gz URL.
    fastas = tbl.column("fasta_url").to_pylist()
    assert any(u.endswith("_genomic.fna.gz") for u in fastas)
    # Protein FASTA URL populated.
    proteins = tbl.column("protein_url").to_pylist()
    assert all(u.endswith("_protein.faa.gz") for u in proteins)
    # Slug derives from organism_name.
    slugs = tbl.column("organism_slug").to_pylist()
    assert "homo_sapiens" in slugs and "mus_musculus" in slugs


def test_refseq_schema_matches_canonical():
    tbl = refseq.parse_summary(_REFSEQ_FIXTURE)
    assert tbl.schema == ASSEMBLY_CATALOG_TABLE


# ──────────────────────────────────────────────────────────────────────
# UniProt parser
# ──────────────────────────────────────────────────────────────────────


_UNIPROT_FIXTURE = """\
UniProt Reference Proteomes

Some preamble text we should ignore.

Proteome_ID\tTax_ID\tOSCODE\tSUPERREGNUM\t#(1)\t#(2)\t#(3)\tSpecies Name
UP000005640\t9606\tHUMAN\tEukaryota\t20420\t0\t0\tHomo sapiens
UP000000589\t10090\tMOUSE\tEukaryota\t22293\t0\t0\tMus musculus
UP000002311\t559292\tYEAST\tEukaryota\t6060\t0\t0\tSaccharomyces cerevisiae

(trailing footer text after blank line)
"""


def test_uniprot_parse_index():
    tbl = uniprot.parse_index(_UNIPROT_FIXTURE, release="2024_02")
    assert tbl.num_rows == 3
    proteomes = tbl.column("assembly_accession").to_pylist()
    assert "UP000005640" in proteomes
    # taxid populated.
    taxids = tbl.column("taxid").to_pylist()
    assert 9606 in taxids and 10090 in taxids
    # gff_url null (proteome-only).
    assert all(g is None for g in tbl.column("gff_url").to_pylist())
    # protein_url populated.
    assert all(p for p in tbl.column("protein_url").to_pylist())
    # source field correct.
    assert all(s == "uniprot" for s in tbl.column("source").to_pylist())


# ──────────────────────────────────────────────────────────────────────
# Resolver — precedence + search
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def populated_resolver(tmp_path: Path, monkeypatch) -> CatalogResolver:
    """Install Ensembl + RefSeq + UniProt catalogs into a temp dir."""
    monkeypatch.setenv("CONSTELLATION_CATALOGS_HOME", str(tmp_path))

    ens_tbl = ensembl.parse_species_txt(_ENSEMBL_FIXTURE, release=111)
    rs_tbl = refseq.parse_summary(_REFSEQ_FIXTURE)
    up_tbl = uniprot.parse_index(_UNIPROT_FIXTURE, release="2024_02")

    write_catalog(
        release_dir("ensembl", "111", root=tmp_path),
        table=ens_tbl,
        meta={"source": "ensembl", "release": "111", "n_rows": ens_tbl.num_rows},
    )
    write_catalog(
        release_dir("refseq", "20260517", root=tmp_path),
        table=rs_tbl,
        meta={"source": "refseq", "release": "20260517", "n_rows": rs_tbl.num_rows},
    )
    write_catalog(
        release_dir("uniprot", "2024_02", root=tmp_path),
        table=up_tbl,
        meta={"source": "uniprot", "release": "2024_02", "n_rows": up_tbl.num_rows},
    )
    return CatalogResolver.from_cache(root=tmp_path)


def test_resolver_lists_installed_sources(populated_resolver):
    sources = {s for s, _ in populated_resolver.installed_sources()}
    assert sources == {"ensembl", "refseq", "uniprot"}


def test_resolver_best_for_refseq_first(populated_resolver):
    row = populated_resolver.best_for(9606)
    assert row is not None
    # RefSeq has reference_genome — should win.
    assert row.source == "refseq"
    assert row.refseq_category == "reference genome"


def test_resolver_best_for_with_source_override(populated_resolver):
    row = populated_resolver.best_for(9606, source="ensembl")
    assert row is not None
    assert row.source == "ensembl"
    assert row.assembly_name == "GRCh38.p14"


def test_resolver_best_for_uniprot_only_when_requested(populated_resolver):
    # UniProt's UP000002311 is keyed on strain taxid 559292; RefSeq's
    # row for S288C collapses to species_taxid 4932. The implicit
    # fallback chain excludes UniProt, so best_for(559292) returns None
    # without --source uniprot.
    assert populated_resolver.best_for(559292) is None
    explicit = populated_resolver.best_for(559292, source="uniprot")
    assert explicit is not None and explicit.source == "uniprot"
    # But best_for(4932) hits RefSeq's species-level row.
    sc_row = populated_resolver.best_for(4932)
    assert sc_row is not None and sc_row.source == "refseq"


def test_resolver_all_for_taxid_lists_every_source(populated_resolver):
    rows = populated_resolver.all_for(9606)
    sources = {r.source for r in rows}
    # Ensembl + RefSeq both have 9606. UniProt also has 9606.
    assert "refseq" in sources and "ensembl" in sources and "uniprot" in sources


def test_resolver_search_by_name(populated_resolver):
    hits = populated_resolver.search("homo")
    assert hits
    assert all("Homo" in r.species_name or "homo" in r.organism_slug for r in hits)


def test_resolver_search_with_source_filter(populated_resolver):
    hits = populated_resolver.search("homo", source="ensembl")
    assert hits
    assert all(r.source == "ensembl" for r in hits)


def test_resolver_returns_none_for_unknown_taxid(populated_resolver):
    assert populated_resolver.best_for(999999999) is None


def test_resolver_empty_is_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_CATALOGS_HOME", str(tmp_path))
    r = CatalogResolver.from_cache(root=tmp_path)
    assert r.is_empty()
    assert r.best_for(9606) is None
    assert r.search("anything") == []


# ──────────────────────────────────────────────────────────────────────
# Store round-trip
# ──────────────────────────────────────────────────────────────────────


def test_store_round_trip(tmp_path: Path):
    tbl = ensembl.parse_species_txt(_ENSEMBL_FIXTURE, release=111)
    bundle_dir = release_dir("ensembl", "111", root=tmp_path)
    write_catalog(bundle_dir, table=tbl, meta={"source": "ensembl", "release": "111"})
    loaded = read_catalog(bundle_dir)
    assert loaded.table.num_rows == tbl.num_rows
    assert loaded.source == "ensembl" and loaded.release == "111"


def test_list_installed(tmp_path: Path):
    ens = ensembl.parse_species_txt(_ENSEMBL_FIXTURE, release=111)
    rs = refseq.parse_summary(_REFSEQ_FIXTURE)
    write_catalog(
        release_dir("ensembl", "111", root=tmp_path),
        table=ens,
        meta={"source": "ensembl", "release": "111"},
    )
    write_catalog(
        release_dir("refseq", "20260517", root=tmp_path),
        table=rs,
        meta={"source": "refseq", "release": "20260517"},
    )
    bundles = list_installed(root=tmp_path)
    assert {b.source for b in bundles} == {"ensembl", "refseq"}
