"""Arrow schemas for the assembly / proteome catalog layer.

One schema covers genomic and proteome catalogs alike — UniProt rows
set ``source="uniprot"``, ``gff_url=null``, ``protein_url`` populated,
and ``assembly_accession`` carries the proteome ID (e.g. ``UP000005640``);
Ensembl / Ensembl Genomes / RefSeq / GenBank rows populate
``gff_url`` / ``fasta_url`` / ``protein_url`` as available.

Mirrors the self-registration pattern in
``constellation.core.io.schemas``.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema, registered_schemas


CATALOG_SOURCE_VOCAB: frozenset[str] = frozenset(
    {
        "ensembl",
        "ensembl_genomes",
        "refseq",
        "genbank",
        "uniprot",
    }
)


ASSEMBLY_CATALOG_TABLE: pa.Schema = pa.schema(
    [
        pa.field("catalog_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("release", pa.string(), nullable=False),
        pa.field("taxid", pa.int64(), nullable=True),
        pa.field("species_name", pa.string(), nullable=False),
        pa.field("organism_slug", pa.string(), nullable=False),
        pa.field("assembly_accession", pa.string(), nullable=True),
        pa.field("assembly_name", pa.string(), nullable=True),
        pa.field("assembly_level", pa.string(), nullable=True),
        pa.field("refseq_category", pa.string(), nullable=True),
        pa.field("annotation_release", pa.string(), nullable=True),
        pa.field("fasta_url", pa.string(), nullable=False),
        pa.field("gff_url", pa.string(), nullable=True),
        pa.field("cdna_url", pa.string(), nullable=True),
        pa.field("protein_url", pa.string(), nullable=True),
        pa.field("checksums_url", pa.string(), nullable=True),
        pa.field("checksums_kind", pa.string(), nullable=True),
        pa.field("division", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"AssemblyCatalogTable"},
)


if "AssemblyCatalogTable" not in registered_schemas():
    register_schema("AssemblyCatalogTable", ASSEMBLY_CATALOG_TABLE)


__all__ = ["ASSEMBLY_CATALOG_TABLE", "CATALOG_SOURCE_VOCAB"]
