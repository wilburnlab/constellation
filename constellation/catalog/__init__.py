"""Genome / proteome catalog framework.

Top-level peer module to ``sequencing``, ``massspec``, ``structure``,
etc. The catalog content (taxid → assembly/proteome URLs across
Ensembl / Ensembl Genomes / RefSeq / UniProt) is registry-of-external-
databases — useful for sequencing analysis but equally useful for
evolutionary work, data-mining, and future PDB / ChEMBL / mzCloud
catalogs that will land in the same module without a rename.

Public surface:

* ``CatalogResolver`` — query installed catalogs by taxid / name /
  substring; ``best_for(taxid, source=None)`` implements the
  RefSeq-first precedence ordering.
* ``CatalogRow`` — immutable record (mirror of one row of
  ``ASSEMBLY_CATALOG_TABLE``).
* ``ASSEMBLY_CATALOG_TABLE`` — Arrow schema (self-registered).
* Per-source modules: ``ensembl``, ``ensembl_genomes``, ``refseq``,
  ``uniprot`` — each ships ``fetch_catalog(...)`` + ``parse_*``
  helpers usable from tests with synthetic fixtures.
* ``catalogs_root()``, ``list_installed()``, ``write_catalog`` /
  ``read_catalog`` — on-disk bundle management.
"""

from constellation.catalog import (
    ensembl,
    ensembl_genomes,
    refseq,
    uniprot,
)
from constellation.catalog.resolver import CatalogResolver
from constellation.catalog.schemas import (
    ASSEMBLY_CATALOG_TABLE,
    CATALOG_SOURCE_VOCAB,
)
from constellation.catalog.store import (
    CatalogBundle,
    CatalogNotInstalled,
    catalogs_root,
    latest_release,
    list_installed,
    read_catalog,
    release_dir,
    source_root,
    write_catalog,
)
from constellation.catalog.types import CatalogRow


__all__ = [
    "ASSEMBLY_CATALOG_TABLE",
    "CATALOG_SOURCE_VOCAB",
    "CatalogBundle",
    "CatalogNotInstalled",
    "CatalogResolver",
    "CatalogRow",
    "catalogs_root",
    "ensembl",
    "ensembl_genomes",
    "latest_release",
    "list_installed",
    "read_catalog",
    "refseq",
    "release_dir",
    "source_root",
    "uniprot",
    "write_catalog",
]
