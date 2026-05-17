"""Immutable record types for the taxonomy layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaxonNode:
    """A single NCBI taxon — the shape ``TaxonomyResolver`` returns.

    Mirrors a row of ``TAXONOMY_NODES_TABLE``, with the parent reference
    normalised so the root carries ``parent_taxid=None`` (NCBI encodes
    the root as ``taxid=1, parent_taxid=1`` — Tree's root-normalisation
    rewrites that).
    """

    taxid: int
    scientific_name: str
    rank: str
    parent_taxid: int | None
    division: str | None = None
    genetic_code: int | None = None
    mito_genetic_code: int | None = None


class UnknownTaxonError(KeyError):
    """Raised by ``TaxonomyResolver.lookup_strict`` when a query
    (taxid, scientific name, common name, synonym, ...) doesn't resolve
    against the active taxonomy source."""


__all__ = ["TaxonNode", "UnknownTaxonError"]
