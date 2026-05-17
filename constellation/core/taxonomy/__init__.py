"""NCBI taxonomy layer.

Sibling to ``core.ontology`` — both are named-entity controlled
vocabularies with hierarchy — but distinct in scale (~2.5M taxa vs
~1.5K UNIMOD), source format (NCBI .dmp tarball vs OBO/XML), and query
patterns (heavy tree traversal — LCA, lineage, descendants).

Public surface:

* ``TaxonomyResolver`` — name/taxid lookup + tree traversal
* ``TaxonNode`` — immutable record returned by lookups
* ``UnknownTaxonError`` — raised by ``lookup_strict``
* ``TAXONOMY_NODES_TABLE`` / ``TAXONOMY_NAMES_TABLE`` /
  ``TAXONOMY_MERGED_TABLE`` — Arrow schemas (self-registered)
* ``fetch_taxdump`` — entry point for ``constellation taxonomy update``

The resolver is layered: ``from_bundled()`` reads the small starter
shipped with the package; ``from_cache()`` reads a full lazy-fetched
dump under ``~/.constellation/taxonomy/``; ``auto()`` prefers the
cache and falls back to the bundled starter.
"""

from constellation.core.taxonomy.fetch_ncbi import (
    NCBI_TAXDUMP_URL,
    fetch_taxdump,
    parse_taxdump_archive,
)
from constellation.core.taxonomy.resolver import TaxonomyResolver
from constellation.core.taxonomy.schemas import (
    NAME_CLASS_VOCAB,
    RANK_VOCAB,
    TAXONOMY_MERGED_TABLE,
    TAXONOMY_NAMES_TABLE,
    TAXONOMY_NODES_TABLE,
)
from constellation.core.taxonomy.store import (
    CachedTaxonomyMissing,
    list_installed,
    resolve_current,
    taxonomy_root,
)
from constellation.core.taxonomy.types import TaxonNode, UnknownTaxonError


__all__ = [
    "CachedTaxonomyMissing",
    "NAME_CLASS_VOCAB",
    "NCBI_TAXDUMP_URL",
    "RANK_VOCAB",
    "TAXONOMY_MERGED_TABLE",
    "TAXONOMY_NAMES_TABLE",
    "TAXONOMY_NODES_TABLE",
    "TaxonNode",
    "TaxonomyResolver",
    "UnknownTaxonError",
    "fetch_taxdump",
    "list_installed",
    "parse_taxdump_archive",
    "resolve_current",
    "taxonomy_root",
]
