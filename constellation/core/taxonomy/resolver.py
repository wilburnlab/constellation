"""``TaxonomyResolver`` — name/taxid lookup over a taxonomy bundle.

A bundle is the triple of Arrow tables defined in ``schemas.py``:

* ``TAXONOMY_NODES_TABLE`` — drives the underlying ``core.graph.Tree``
* ``TAXONOMY_NAMES_TABLE`` — multi-name lookup (scientific + common +
  synonyms + ...)
* ``TAXONOMY_MERGED_TABLE`` — deprecated→current taxid redirects

``TaxonomyResolver.from_bundled()`` reads the small starter shipped
with the package; ``TaxonomyResolver.from_cache()`` reads a full lazy-
fetched dump under ``~/.constellation/taxonomy/``; ``TaxonomyResolver
.auto()`` prefers the cache and falls back to the bundled starter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.graph import Tree
from constellation.core.taxonomy.types import TaxonNode, UnknownTaxonError


# NCBI division id → human-readable name. Lifted from division.dmp; the
# small lookup avoids carrying a fourth table just for this.
_DIVISION_NAMES: dict[int, str] = {
    0: "Bacteria",
    1: "Invertebrates",
    2: "Mammals",
    3: "Phages",
    4: "Plants and Fungi",
    5: "Primates",
    6: "Rodents",
    7: "Synthetic and Chimeric",
    8: "Unassigned",
    9: "Viruses",
    10: "Vertebrates",
    11: "Environmental samples",
}


class TaxonomyResolver:
    """Lookup engine for an Arrow-backed taxonomy bundle."""

    __slots__ = (
        "_nodes",
        "_names",
        "_merged",
        "_tree",
        "_node_index",
        "_name_index",
        "_merged_index",
        "_source_meta",
    )

    def __init__(
        self,
        nodes: pa.Table,
        names: pa.Table,
        merged: pa.Table | None = None,
        *,
        source_meta: dict[str, Any] | None = None,
    ) -> None:
        self._nodes: pa.Table = nodes
        self._names: pa.Table = names
        self._merged: pa.Table = merged if merged is not None else _empty_merged()
        self._tree: Tree[int] = Tree(
            nodes,
            node_id_col="taxid",
            parent_id_col="parent_taxid",
            root_sentinel=None,
        )
        # Lazy caches built on first access.
        self._node_index: dict[int, int] | None = None
        self._name_index: dict[str, list[int]] | None = None
        self._merged_index: dict[int, int] | None = None
        self._source_meta: dict[str, Any] = dict(source_meta or {})

    # ── factories ────────────────────────────────────────────────────
    @classmethod
    def from_bundled(cls) -> "TaxonomyResolver":
        from constellation.core.taxonomy.bundled import load_bundled_taxonomy

        nodes, names, merged, meta = load_bundled_taxonomy()
        return cls(nodes, names, merged, source_meta=meta)

    @classmethod
    def from_cache(cls, *, root: Path | None = None) -> "TaxonomyResolver":
        from constellation.core.taxonomy.store import (
            load_cached_taxonomy,
        )

        nodes, names, merged, meta = load_cached_taxonomy(root=root)
        return cls(nodes, names, merged, source_meta=meta)

    @classmethod
    def auto(cls) -> "TaxonomyResolver":
        """Prefer the lazy-fetched cache; fall back to the bundled starter."""
        from constellation.core.taxonomy.store import CachedTaxonomyMissing

        try:
            return cls.from_cache()
        except CachedTaxonomyMissing:
            return cls.from_bundled()

    # ── indexing ─────────────────────────────────────────────────────
    def _build_node_index(self) -> dict[int, int]:
        if self._node_index is not None:
            return self._node_index
        taxids = self._nodes.column("taxid").to_pylist()
        self._node_index = {tid: i for i, tid in enumerate(taxids)}
        return self._node_index

    def _build_name_index(self) -> dict[str, list[int]]:
        if self._name_index is not None:
            return self._name_index
        idx: dict[str, list[int]] = {}
        if self._names.num_rows > 0:
            taxids = self._names.column("taxid").to_pylist()
            lows = self._names.column("name_lower").to_pylist()
            for tid, lo in zip(taxids, lows):
                idx.setdefault(lo, []).append(tid)
        self._name_index = idx
        return idx

    def _build_merged_index(self) -> dict[int, int]:
        if self._merged_index is not None:
            return self._merged_index
        idx: dict[int, int] = {}
        if self._merged.num_rows > 0:
            old = self._merged.column("old_taxid").to_pylist()
            new = self._merged.column("new_taxid").to_pylist()
            idx = dict(zip(old, new))
        self._merged_index = idx
        return idx

    # ── node materialisation ────────────────────────────────────────
    def _row_to_node(self, row_index: int) -> TaxonNode:
        nodes = self._nodes
        div_id = nodes.column("division_id")[row_index].as_py()
        return TaxonNode(
            taxid=nodes.column("taxid")[row_index].as_py(),
            scientific_name=nodes.column("scientific_name")[row_index].as_py(),
            rank=nodes.column("rank")[row_index].as_py(),
            parent_taxid=self._tree.parent_of(
                nodes.column("taxid")[row_index].as_py()
            ),
            division=_DIVISION_NAMES.get(div_id) if div_id is not None else None,
            genetic_code=nodes.column("genetic_code_id")[row_index].as_py(),
            mito_genetic_code=nodes.column("mito_genetic_code_id")[row_index].as_py(),
        )

    def _resolve_taxid(self, taxid: int) -> int | None:
        """Apply merged-taxid redirects; return ``None`` if taxid is unknown."""
        node_index = self._build_node_index()
        if taxid in node_index:
            return taxid
        merged = self._build_merged_index()
        return merged.get(taxid)

    # ── public lookup ───────────────────────────────────────────────
    def lookup(self, name_or_taxid: str | int) -> TaxonNode | None:
        """Exact lookup. Strings tried as int-coerced taxid first, then
        as case-insensitive name across all ``NAME_CLASS_VOCAB`` rows.
        Merged taxids are resolved transparently.
        """
        if isinstance(name_or_taxid, int):
            return self._lookup_taxid(name_or_taxid)
        s = name_or_taxid.strip()
        if not s:
            return None
        # Try numeric first.
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return self._lookup_taxid(int(s))
            except ValueError:
                pass
        # Name lookup.
        idx = self._build_name_index()
        hits = idx.get(s.lower())
        if not hits:
            return None
        # Multiple taxids can share a name (homonym across kingdoms);
        # pick the lowest taxid for determinism — caller wanting all hits
        # should use search().
        taxid = min(hits)
        return self._lookup_taxid(taxid)

    def lookup_strict(self, name_or_taxid: str | int) -> TaxonNode:
        node = self.lookup(name_or_taxid)
        if node is None:
            raise UnknownTaxonError(name_or_taxid)
        return node

    def _lookup_taxid(self, taxid: int) -> TaxonNode | None:
        resolved = self._resolve_taxid(taxid)
        if resolved is None:
            return None
        return self._row_to_node(self._build_node_index()[resolved])

    # ── search ──────────────────────────────────────────────────────
    def search(
        self,
        substring: str,
        *,
        limit: int = 25,
        rank: str | None = None,
        name_class: str | None = None,
    ) -> list[tuple[TaxonNode, str]]:
        """Case-insensitive substring search across all name rows.

        Returns up to ``limit`` ``(node, matched_name)`` tuples in ascending
        taxid order. ``rank`` filters to a specific node rank;
        ``name_class`` filters to a specific name kind (``"scientific
        name"``, ``"common name"``, ...).
        """
        if not substring:
            return []
        q = substring.lower()
        names = self._names
        if names.num_rows == 0:
            return []
        mask = pc.match_substring(names.column("name_lower"), q)
        if name_class is not None:
            mask = pc.and_(mask, pc.equal(names.column("name_class"), name_class))
        filtered = names.filter(mask)
        if filtered.num_rows == 0:
            return []
        # Walk hits in order, dedup by (taxid, first-matched-name).
        node_index = self._build_node_index()
        out: list[tuple[TaxonNode, str]] = []
        seen: set[int] = set()
        taxids = filtered.column("taxid").to_pylist()
        names_col = filtered.column("name").to_pylist()
        for tid, nm in zip(taxids, names_col):
            if tid in seen:
                continue
            row = node_index.get(tid)
            if row is None:
                continue
            node = self._row_to_node(row)
            if rank is not None and node.rank != rank:
                continue
            seen.add(tid)
            out.append((node, nm))
            if len(out) >= limit:
                break
        return out

    # ── tree operations (delegate to the underlying Tree) ──────────
    def lineage(self, taxid: int) -> list[TaxonNode]:
        resolved = self._resolve_taxid(taxid)
        if resolved is None:
            raise UnknownTaxonError(taxid)
        node_index = self._build_node_index()
        return [self._row_to_node(node_index[t]) for t in self._tree.lineage(resolved)]

    def ancestors(self, taxid: int) -> list[TaxonNode]:
        resolved = self._resolve_taxid(taxid)
        if resolved is None:
            raise UnknownTaxonError(taxid)
        node_index = self._build_node_index()
        return [
            self._row_to_node(node_index[t]) for t in self._tree.ancestors_of(resolved)
        ]

    def descendants(
        self,
        taxid: int,
        *,
        rank: str | None = None,
        max_depth: int | None = None,
    ) -> list[TaxonNode]:
        resolved = self._resolve_taxid(taxid)
        if resolved is None:
            raise UnknownTaxonError(taxid)
        node_index = self._build_node_index()
        out: list[TaxonNode] = []
        for t in self._tree.descendants_of(resolved, max_depth=max_depth):
            node = self._row_to_node(node_index[t])
            if rank is not None and node.rank != rank:
                continue
            out.append(node)
        return out

    def lca(self, taxid_a: int, taxid_b: int) -> TaxonNode | None:
        ra = self._resolve_taxid(taxid_a)
        rb = self._resolve_taxid(taxid_b)
        if ra is None or rb is None:
            return None
        result = self._tree.lca(ra, rb)
        if result is None:
            return None
        return self._row_to_node(self._build_node_index()[result])

    # ── slugs (replaces the deleted hardcoded slug tables) ─────────
    def organism_slug(self, taxid: int) -> str:
        """Stable lowercase slug for cache directory naming.

        ``Homo sapiens`` → ``"homo_sapiens"``; spaces and any non
        ``[a-z0-9_]`` character collapse to underscore; consecutive
        underscores collapse; trailing underscores stripped.
        """
        node = self._lookup_taxid(taxid)
        if node is None:
            raise UnknownTaxonError(taxid)
        return _slugify(node.scientific_name)

    # ── views ───────────────────────────────────────────────────────
    @property
    def tree(self) -> Tree[int]:
        return self._tree

    def source_meta(self) -> dict[str, Any]:
        return dict(self._source_meta)

    def n_taxa(self) -> int:
        return self._nodes.num_rows


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _empty_merged() -> pa.Table:
    from constellation.core.taxonomy.schemas import TAXONOMY_MERGED_TABLE

    return pa.table(
        {
            "old_taxid": pa.array([], type=pa.int64()),
            "new_taxid": pa.array([], type=pa.int64()),
        },
        schema=TAXONOMY_MERGED_TABLE,
    )


def _slugify(name: str) -> str:
    """Lowercase name; collapse runs of non-alphanumerics to a single ``_``."""
    out: list[str] = []
    prev_underscore = False
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    slug = "".join(out).strip("_")
    return slug or "unknown"


__all__ = ["TaxonomyResolver"]
