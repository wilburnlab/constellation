"""``CatalogResolver`` — query installed catalogs by taxid or species name.

Loads every installed ``(source, release)`` bundle under
``catalogs_root()``, exposes a unified search/lookup surface across
them, and implements the RefSeq-first precedence ordering for
``best_for(taxid)``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from constellation.catalog.store import (
    CatalogBundle,
    list_installed,
)
from constellation.catalog.types import CatalogRow


# RefSeq-first precedence, with UniProt excluded from the implicit
# fallback (proteome ≠ genome).
_DEFAULT_PRECEDENCE: tuple[str, ...] = (
    "refseq",
    "genbank",
    "ensembl",
    "ensembl_genomes",
)


# RefSeq-category ordering inside the "refseq" source.
_REFSEQ_CATEGORY_RANK = {
    "reference genome": 0,
    "representative genome": 1,
}


class CatalogResolver:
    """Aggregator over every installed catalog bundle."""

    __slots__ = ("_bundles", "_combined", "_by_taxid", "_by_slug")

    def __init__(self, bundles: list[CatalogBundle]) -> None:
        self._bundles: list[CatalogBundle] = list(bundles)
        self._combined: pa.Table | None = None
        self._by_taxid: dict[int, list[int]] | None = None  # taxid -> row indices
        self._by_slug: dict[str, list[int]] | None = None

    @classmethod
    def from_cache(cls, *, root: Path | None = None) -> "CatalogResolver":
        bundles = list_installed(root=root)
        return cls(bundles)

    # ── views ─────────────────────────────────────────────────────
    def installed_sources(self) -> list[tuple[str, str]]:
        return [(b.source, b.release) for b in self._bundles]

    def n_rows(self) -> int:
        return self._combined_table().num_rows

    def is_empty(self) -> bool:
        return not self._bundles

    # ── lookup / search ───────────────────────────────────────────
    def best_for(
        self,
        taxid: int,
        *,
        source: str | None = None,
    ) -> CatalogRow | None:
        """Return the highest-precedence catalog row for ``taxid``.

        With ``source`` provided, restrict to that source (and pick the
        best within it: RefSeq prefers ``reference_genome`` then
        ``representative_genome``; other sources pick the lowest
        ``catalog_id``). Without ``source``, walk the default precedence
        chain (RefSeq → Ensembl → Ensembl Genomes; UniProt excluded
        from the implicit fallback).
        """
        if self.is_empty():
            return None
        if source is not None:
            return self._best_within_source(taxid, source)
        for src in _DEFAULT_PRECEDENCE:
            hit = self._best_within_source(taxid, src)
            if hit is not None:
                return hit
        return None

    def all_for(
        self,
        taxid: int,
        *,
        source: str | None = None,
    ) -> list[CatalogRow]:
        """Every catalog row matching ``taxid`` (across sources by default)."""
        idx = self._build_taxid_index()
        row_ids = idx.get(taxid, [])
        rows = [self._materialise_row(i) for i in row_ids]
        if source is not None:
            rows = [r for r in rows if r.source == source]
        rows.sort(key=_sort_key)
        return rows

    def search(
        self,
        query: str,
        *,
        source: str | None = None,
        limit: int = 25,
    ) -> list[CatalogRow]:
        """Case-insensitive substring search over species_name + organism_slug."""
        if not query:
            return []
        tbl = self._combined_table()
        if tbl.num_rows == 0:
            return []
        q = query.lower()
        mask_name = pc.match_substring(
            pc.utf8_lower(tbl.column("species_name")), q
        )
        mask_slug = pc.match_substring(tbl.column("organism_slug"), q)
        mask = pc.or_(mask_name, mask_slug)
        if source is not None:
            mask = pc.and_(mask, pc.equal(tbl.column("source"), source))
        filtered = tbl.filter(mask)
        out: list[CatalogRow] = []
        for i in range(min(limit, filtered.num_rows)):
            out.append(_row_from_table(filtered, i))
        return out

    # ── internals ─────────────────────────────────────────────────
    def _combined_table(self) -> pa.Table:
        if self._combined is not None:
            return self._combined
        if not self._bundles:

            self._combined = _empty_catalog_table()
            return self._combined
        self._combined = pa.concat_tables([b.table for b in self._bundles])
        return self._combined

    def _build_taxid_index(self) -> dict[int, list[int]]:
        if self._by_taxid is not None:
            return self._by_taxid
        tbl = self._combined_table()
        idx: dict[int, list[int]] = {}
        if tbl.num_rows > 0:
            taxids = tbl.column("taxid").to_pylist()
            for i, t in enumerate(taxids):
                if t is None:
                    continue
                idx.setdefault(t, []).append(i)
        self._by_taxid = idx
        return idx

    def _best_within_source(self, taxid: int, source: str) -> CatalogRow | None:
        rows = self.all_for(taxid, source=source)
        return rows[0] if rows else None

    def _materialise_row(self, row_index: int) -> CatalogRow:
        return _row_from_table(self._combined_table(), row_index)


# ──────────────────────────────────────────────────────────────────────
# Row materialisation + sorting
# ──────────────────────────────────────────────────────────────────────


def _row_from_table(tbl: pa.Table, i: int) -> CatalogRow:
    def at(col: str):
        return tbl.column(col)[i].as_py()

    return CatalogRow(
        catalog_id=at("catalog_id"),
        source=at("source"),
        release=at("release"),
        taxid=at("taxid"),
        species_name=at("species_name"),
        organism_slug=at("organism_slug"),
        assembly_accession=at("assembly_accession"),
        assembly_name=at("assembly_name"),
        assembly_level=at("assembly_level"),
        refseq_category=at("refseq_category"),
        annotation_release=at("annotation_release"),
        fasta_url=at("fasta_url"),
        gff_url=at("gff_url"),
        cdna_url=at("cdna_url"),
        protein_url=at("protein_url"),
        checksums_url=at("checksums_url"),
        checksums_kind=at("checksums_kind"),
        division=at("division"),
    )


def _sort_key(row: CatalogRow) -> tuple[int, int]:
    """RefSeq rows ranked by refseq_category; everything else by catalog_id."""
    if row.source == "refseq":
        cat_rank = _REFSEQ_CATEGORY_RANK.get(row.refseq_category or "", 9)
        return (cat_rank, row.catalog_id)
    return (5, row.catalog_id)


def _empty_catalog_table() -> pa.Table:
    from constellation.catalog.schemas import ASSEMBLY_CATALOG_TABLE

    cols = {
        f.name: pa.array([], type=f.type) for f in ASSEMBLY_CATALOG_TABLE
    }
    return pa.table(cols, schema=ASSEMBLY_CATALOG_TABLE)


__all__ = ["CatalogResolver"]
