"""End-to-end test for ``reference fetch <species_name>`` dispatch.

Uses the bundled taxonomy starter + a hand-built catalog cache (no
network) to verify that species-name and taxid queries route through
taxonomy → catalog → ``_ResolvedSpec`` correctly.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.catalog import (
    ensembl,
    refseq,
    release_dir,
    uniprot,
    write_catalog,
)
from constellation.sequencing.reference.fetch import (
    _resolve_species_query,
    _resolve_spec,
)


_REFSEQ_FIXTURE = """\
##  See README.txt
#assembly_accession\tbioproject\tbiosample\twgs_master\trefseq_category\ttaxid\tspecies_taxid\torganism_name\tinfraspecific_name\tisolate\tversion_status\tassembly_level\trelease_type\tgenome_rep\tseq_rel_date\tasm_name\tsubmitter\tgbrs_paired_asm\tpaired_asm_comp\tftp_path\texcluded_from_refseq\trelation_to_type_material\tasm_not_live_date\tassembly_type\tgroup
GCF_000001405.40\tPRJNA168\tSAMN12121739\t\treference genome\t9606\t9606\tHomo sapiens\t\t\tlatest\tChromosome\tPatch\tFull\t2022/02/03\tGRCh38.p14\tGRC\tGCA_000001405.29\tidentical\thttps://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14\t\t\tna\thaploid\tvertebrate_mammalian
GCF_000003025.6\tPRJNA13184\tSAMN05451087\t\treference genome\t6454\t6454\tHaliotis rufescens\t\t\tlatest\tScaffold\tMajor\tFull\t2021/01/15\tHRufescensV1\tUniversity of California\tGCA_000003025.5\tidentical\thttps://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/003/025/GCF_000003025.6_HRufescensV1\t\t\tna\thaploid\tinvertebrate
"""


_ENSEMBL_FIXTURE = """\
#name\tspecies\tdivision\ttaxonomy_id\tassembly\tassembly_accession\tgenebuild\tvariation\tmicroarray\tpan_compara\tpeptide_compara\tgenome_alignments\tother_alignments\tcore_db\tspecies_id
Human\thomo_sapiens\tEnsemblVertebrates\t9606\tGRCh38.p14\tGCA_000001405.29\t2024-03\tY\tY\tY\tY\tY\tY\thomo_sapiens_core_111_38\t1
"""


_UNIPROT_FIXTURE = """\
Proteome_ID\tTax_ID\tOSCODE\tSUPERREGNUM\t#(1)\t#(2)\t#(3)\tSpecies Name
UP000005640\t9606\tHUMAN\tEukaryota\t20420\t0\t0\tHomo sapiens

"""


@pytest.fixture
def catalogs_installed(tmp_path: Path, monkeypatch):
    """Install Ensembl + RefSeq + UniProt catalogs into a temp dir."""
    monkeypatch.setenv("CONSTELLATION_CATALOGS_HOME", str(tmp_path))
    write_catalog(
        release_dir("refseq", "20260517", root=tmp_path),
        table=refseq.parse_summary(_REFSEQ_FIXTURE),
        meta={"source": "refseq", "release": "20260517"},
    )
    write_catalog(
        release_dir("ensembl", "111", root=tmp_path),
        table=ensembl.parse_species_txt(_ENSEMBL_FIXTURE, release=111),
        meta={"source": "ensembl", "release": "111"},
    )
    write_catalog(
        release_dir("uniprot", "2024_02", root=tmp_path),
        table=uniprot.parse_index(_UNIPROT_FIXTURE, release="2024_02"),
        meta={"source": "uniprot", "release": "2024_02"},
    )
    return tmp_path


# ──────────────────────────────────────────────────────────────────────
# Dispatch by species name
# ──────────────────────────────────────────────────────────────────────


def test_resolve_human_by_common_name(catalogs_installed):
    spec = _resolve_species_query("human", source=None, release=None)
    # RefSeq-first: should return the RefSeq reference_genome.
    assert spec.handle.source == "refseq"
    assert spec.handle.organism == "homo_sapiens"
    assert spec.assembly_accession == "GCF_000001405.40"
    assert spec.taxid == 9606
    assert spec.scientific_name == "Homo sapiens"
    assert spec.fasta_url.endswith("_genomic.fna.gz")
    assert spec.gff_url and spec.gff_url.endswith("_genomic.gff.gz")


def test_resolve_human_by_scientific_name(catalogs_installed):
    spec = _resolve_species_query("Homo sapiens", source=None, release=None)
    assert spec.handle.organism == "homo_sapiens"
    assert spec.taxid == 9606


def test_resolve_human_by_taxid_string(catalogs_installed):
    spec = _resolve_species_query("9606", source=None, release=None)
    assert spec.handle.organism == "homo_sapiens"
    assert spec.taxid == 9606


def test_resolve_with_source_override_to_ensembl(catalogs_installed):
    spec = _resolve_species_query("human", source="ensembl", release=None)
    assert spec.handle.source == "ensembl"
    assert spec.assembly_name == "GRCh38.p14"


def test_resolve_red_abalone_routes_through_refseq(catalogs_installed):
    spec = _resolve_species_query("red abalone", source=None, release=None)
    assert spec.handle.source == "refseq"
    assert spec.handle.organism == "haliotis_rufescens"
    assert spec.taxid == 6454
    assert spec.scientific_name == "Haliotis rufescens"


def test_resolve_uniprot_explicit_source_rejected_for_fetch(catalogs_installed):
    """UniProt rows are catalogued but ``reference fetch`` doesn't materialise
    them in this PR — the route raises ValueError pointing at PR-C."""
    with pytest.raises(ValueError, match="UniProt"):
        _resolve_species_query("human", source="uniprot", release=None)


def test_resolve_unknown_species_raises(catalogs_installed):
    with pytest.raises(ValueError, match="unknown species"):
        _resolve_species_query("totally fake species name", source=None, release=None)


def test_resolve_known_species_but_no_catalog_hit(catalogs_installed, tmp_path: Path, monkeypatch):
    # Point catalogs at an empty dir — taxonomy resolves but catalog is empty.
    monkeypatch.setenv("CONSTELLATION_CATALOGS_HOME", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="no catalogs installed"):
        _resolve_species_query("human", source=None, release=None)


def test_resolve_spec_routes_no_colon_to_species_path(catalogs_installed):
    """The public ``_resolve_spec`` dispatcher should send no-colon specs
    through the species-name path."""
    spec = _resolve_spec("human", release=None, source=None)
    assert spec.handle.organism == "homo_sapiens"
    assert spec.taxid == 9606


def test_resolve_spec_preserves_legacy_source_id_form(catalogs_installed):
    """``ensembl:human`` continues to route through the legacy URL builder,
    not the catalog path."""
    spec = _resolve_spec("ensembl:human", release=111, source=None)
    assert spec.handle.source == "ensembl"
    assert spec.handle.organism == "homo_sapiens"
    # Legacy path doesn't populate taxid.
    assert spec.taxid is None
