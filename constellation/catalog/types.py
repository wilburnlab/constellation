"""Immutable record types for the catalog layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CatalogRow:
    """A single assembly / proteome entry — mirrors a row of
    ``ASSEMBLY_CATALOG_TABLE``.

    ``taxid`` is nullable only for legacy rows the source itself couldn't
    resolve; downstream consumers should treat ``taxid is None`` as a
    bug-class warning, not a silent fall-through.
    """

    catalog_id: int
    source: str
    release: str
    taxid: int | None
    species_name: str
    organism_slug: str
    assembly_accession: str | None
    assembly_name: str | None
    assembly_level: str | None
    refseq_category: str | None
    annotation_release: str | None
    fasta_url: str
    gff_url: str | None
    cdna_url: str | None
    protein_url: str | None
    checksums_url: str | None
    checksums_kind: str | None
    division: str | None


__all__ = ["CatalogRow"]
