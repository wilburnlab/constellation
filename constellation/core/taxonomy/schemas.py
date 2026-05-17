"""Arrow schemas for the taxonomy layer.

Three tables ride together:

* ``TAXONOMY_NODES_TABLE`` — one row per taxon, with the parent_taxid
  pointer that ``core.graph.Tree`` walks.
* ``TAXONOMY_NAMES_TABLE`` — one row per (taxon, name, name_class)
  tuple. A given taxid has multiple rows (scientific name + common
  names + synonyms + ...).
* ``TAXONOMY_MERGED_TABLE`` — NCBI's ``merged.dmp`` view: deprecated
  taxids and the taxids they were merged into. Resolver consults this
  to transparently redirect requests for retired IDs.

Mirrors ``constellation.core.io.schemas``'s self-registration pattern:
schemas register themselves with the central registry at import time.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema, registered_schemas


# ──────────────────────────────────────────────────────────────────────
# Vocabularies (frozen — drift means an upstream NCBI format change)
# ──────────────────────────────────────────────────────────────────────


RANK_VOCAB: frozenset[str] = frozenset(
    {
        "no rank",
        "superkingdom",
        "kingdom",
        "subkingdom",
        "superphylum",
        "phylum",
        "subphylum",
        "superclass",
        "class",
        "subclass",
        "infraclass",
        "superorder",
        "order",
        "suborder",
        "infraorder",
        "parvorder",
        "superfamily",
        "family",
        "subfamily",
        "tribe",
        "subtribe",
        "genus",
        "subgenus",
        "species group",
        "species subgroup",
        "species",
        "subspecies",
        "varietas",
        "forma",
        "strain",
        "serotype",
        "serogroup",
        "biotype",
        "genotype",
        "isolate",
        "clade",
        "cohort",
        "subcohort",
        "section",
        "subsection",
        "series",
        "subseries",
        "morph",
        "pathogroup",
        "forma specialis",
        "infraorder",
        "infraclass",
        "infrakingdom",
        "subvariety",
        "subterclass",
    }
)


NAME_CLASS_VOCAB: frozenset[str] = frozenset(
    {
        "scientific name",
        "common name",
        "genbank common name",
        "synonym",
        "equivalent name",
        "blast name",
        "authority",
        "type material",
        "in-part",
        "includes",
        "acronym",
        "genbank acronym",
    }
)


# ──────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────


TAXONOMY_NODES_TABLE: pa.Schema = pa.schema(
    [
        pa.field("taxid", pa.int64(), nullable=False),
        pa.field("parent_taxid", pa.int64(), nullable=False),
        pa.field("rank", pa.string(), nullable=False),
        pa.field("division_id", pa.int16(), nullable=True),
        pa.field("genetic_code_id", pa.int16(), nullable=True),
        pa.field("mito_genetic_code_id", pa.int16(), nullable=True),
        pa.field("scientific_name", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"TaxonomyNodesTable"},
)


TAXONOMY_NAMES_TABLE: pa.Schema = pa.schema(
    [
        pa.field("taxid", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("name_lower", pa.string(), nullable=False),
        pa.field("name_class", pa.string(), nullable=False),
        pa.field("unique_name", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"TaxonomyNamesTable"},
)


TAXONOMY_MERGED_TABLE: pa.Schema = pa.schema(
    [
        pa.field("old_taxid", pa.int64(), nullable=False),
        pa.field("new_taxid", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"TaxonomyMergedTable"},
)


# Self-registration. Idempotent — guards re-registration on test reload.
for _name, _schema in (
    ("TaxonomyNodesTable", TAXONOMY_NODES_TABLE),
    ("TaxonomyNamesTable", TAXONOMY_NAMES_TABLE),
    ("TaxonomyMergedTable", TAXONOMY_MERGED_TABLE),
):
    if _name not in registered_schemas():
        register_schema(_name, _schema)


__all__ = [
    "NAME_CLASS_VOCAB",
    "RANK_VOCAB",
    "TAXONOMY_MERGED_TABLE",
    "TAXONOMY_NAMES_TABLE",
    "TAXONOMY_NODES_TABLE",
]
