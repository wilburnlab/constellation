"""Unit tests for ``constellation.core.taxonomy``.

Drives the bundled starter parquet shipped with the package — the
hand-curated v1 set covers lab targets + canonical model organisms.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.core.taxonomy import (
    NAME_CLASS_VOCAB,
    RANK_VOCAB,
    TAXONOMY_MERGED_TABLE,
    TAXONOMY_NAMES_TABLE,
    TAXONOMY_NODES_TABLE,
    TaxonNode,
    TaxonomyResolver,
    UnknownTaxonError,
)
from constellation.core.io.schemas import get_schema


# ──────────────────────────────────────────────────────────────────────
# Schema registration
# ──────────────────────────────────────────────────────────────────────


def test_schemas_self_register():
    assert get_schema("TaxonomyNodesTable") is TAXONOMY_NODES_TABLE
    assert get_schema("TaxonomyNamesTable") is TAXONOMY_NAMES_TABLE
    assert get_schema("TaxonomyMergedTable") is TAXONOMY_MERGED_TABLE


def test_vocabs_are_frozen_sets():
    assert isinstance(RANK_VOCAB, frozenset)
    assert isinstance(NAME_CLASS_VOCAB, frozenset)
    assert "species" in RANK_VOCAB
    assert "scientific name" in NAME_CLASS_VOCAB


# ──────────────────────────────────────────────────────────────────────
# Bundled starter — lookup
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def resolver() -> TaxonomyResolver:
    return TaxonomyResolver.from_bundled()


def test_bundled_starter_non_empty(resolver):
    assert resolver.n_taxa() > 100
    assert resolver.source_meta()["source"] == "constellation-v1-hand-curated"


def test_lookup_scientific_name(resolver):
    n = resolver.lookup("Homo sapiens")
    assert n is not None
    assert n.taxid == 9606
    assert n.rank == "species"


def test_lookup_common_name(resolver):
    n = resolver.lookup("human")
    assert n is not None and n.taxid == 9606


def test_lookup_synonym_via_alt_class(resolver):
    n = resolver.lookup("man")
    assert n is not None and n.taxid == 9606


def test_lookup_is_case_insensitive(resolver):
    a = resolver.lookup("HUMAN")
    b = resolver.lookup("Human")
    c = resolver.lookup("human")
    assert a is not None and a.taxid == b.taxid == c.taxid == 9606


def test_lookup_int_taxid(resolver):
    assert resolver.lookup(9606).taxid == 9606
    assert resolver.lookup("9606").taxid == 9606


def test_lookup_lab_priority_species(resolver):
    # Abalone (red abalone — the lab's named target)
    n = resolver.lookup("red abalone")
    assert n is not None and n.scientific_name == "Haliotis rufescens"

    # Lungless salamander common name
    n = resolver.lookup("eastern red-backed salamander")
    assert n is not None and n.scientific_name == "Plethodon cinereus"


def test_lookup_unknown_returns_none(resolver):
    assert resolver.lookup("not a real species name") is None
    assert resolver.lookup(999999999) is None


def test_lookup_strict_raises_on_unknown(resolver):
    with pytest.raises(UnknownTaxonError):
        resolver.lookup_strict("not a real species name")


# ──────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────


def test_search_returns_node_matched_name_pairs(resolver):
    hits = resolver.search("abalone")
    assert hits, "expected at least one match for 'abalone'"
    # Every result should be a Haliotis species in v1 starter.
    for node, _ in hits:
        assert "Haliotis" in node.scientific_name


def test_search_respects_limit(resolver):
    hits = resolver.search("a", limit=3)
    assert len(hits) <= 3


def test_search_filters_by_rank(resolver):
    # 'abalone' is a common name only on species nodes — no genus hit.
    hits = resolver.search("abalone", rank="genus")
    assert hits == []


def test_search_empty_query_returns_empty(resolver):
    assert resolver.search("") == []


# ──────────────────────────────────────────────────────────────────────
# Tree operations
# ──────────────────────────────────────────────────────────────────────


def test_lineage_root_to_species(resolver):
    line = resolver.lineage(9606)
    assert line[0].scientific_name == "root"
    assert line[-1].scientific_name == "Homo sapiens"
    sci_names = [n.scientific_name for n in line]
    for expected in (
        "Eukaryota",
        "Metazoa",
        "Chordata",
        "Mammalia",
        "Primates",
        "Hominidae",
        "Homo",
    ):
        assert expected in sci_names


def test_ancestors_of_excludes_self(resolver):
    anc = resolver.ancestors(9606)
    assert all(n.taxid != 9606 for n in anc)
    # leaf->root order
    assert anc[0].scientific_name == "Homo"
    assert anc[-1].scientific_name == "root"


def test_descendants_haliotis_species(resolver):
    sp = resolver.descendants(216500, rank="species")
    names = {n.scientific_name for n in sp}
    assert "Haliotis rufescens" in names
    assert "Haliotis sorenseni" in names
    assert "Haliotis cracherodii" in names


def test_descendants_max_depth(resolver):
    # depth=1 below genus Haliotis = direct species children only
    sp_depth1 = resolver.descendants(216500, max_depth=1)
    sp_unbounded = resolver.descendants(216500)
    assert len(sp_depth1) == len(sp_unbounded)  # genus has no grand-children in starter


def test_lca_within_mollusca(resolver):
    # Haliotis rufescens vs Tegula funebralis — both Vetigastropoda
    lca = resolver.lca(6454, 231224)
    assert lca is not None
    assert lca.scientific_name == "Vetigastropoda"


def test_lca_human_vs_drosophila_is_bilateria(resolver):
    lca = resolver.lca(9606, 7227)
    assert lca is not None and lca.scientific_name == "Bilateria"


def test_lca_human_vs_ecoli_is_cellular_organisms(resolver):
    lca = resolver.lca(9606, 562)
    assert lca is not None and lca.scientific_name == "cellular organisms"


# ──────────────────────────────────────────────────────────────────────
# organism_slug — replaces the deleted hardcoded slug tables
# ──────────────────────────────────────────────────────────────────────


def test_organism_slug_canonical_form(resolver):
    assert resolver.organism_slug(9606) == "homo_sapiens"
    assert resolver.organism_slug(10090) == "mus_musculus"
    assert resolver.organism_slug(7227) == "drosophila_melanogaster"
    assert resolver.organism_slug(6454) == "haliotis_rufescens"
    assert resolver.organism_slug(4932) == "saccharomyces_cerevisiae"


def test_organism_slug_unknown_raises(resolver):
    with pytest.raises(UnknownTaxonError):
        resolver.organism_slug(999999999)


# ──────────────────────────────────────────────────────────────────────
# Tree handle
# ──────────────────────────────────────────────────────────────────────


def test_tree_handle_exposes_underlying_tree(resolver):
    tree = resolver.tree
    assert tree.root() == 1
    assert tree.has_node(9606)
    assert tree.parent_of(9606) == 9605


# ──────────────────────────────────────────────────────────────────────
# TaxonNode dataclass
# ──────────────────────────────────────────────────────────────────────


def test_taxon_node_is_frozen():
    n = TaxonNode(
        taxid=1, scientific_name="root", rank="no rank", parent_taxid=None
    )
    with pytest.raises(AttributeError):
        n.taxid = 2  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Merged-taxid redirect
# ──────────────────────────────────────────────────────────────────────


def test_merged_redirect_resolves_to_canonical():
    # Build a tiny resolver with a merged.dmp-style redirect.
    nodes = pa.table(
        {
            "taxid": pa.array([1, 100], type=pa.int64()),
            "parent_taxid": pa.array([1, 1], type=pa.int64()),
            "rank": pa.array(["no rank", "species"], type=pa.string()),
            "division_id": pa.array([None, 1], type=pa.int16()),
            "genetic_code_id": pa.array([None, None], type=pa.int16()),
            "mito_genetic_code_id": pa.array([None, None], type=pa.int16()),
            "scientific_name": pa.array(["root", "Examplus exemplum"], type=pa.string()),
        },
        schema=TAXONOMY_NODES_TABLE,
    )
    names = pa.table(
        {
            "taxid": pa.array([100], type=pa.int64()),
            "name": pa.array(["Examplus exemplum"], type=pa.string()),
            "name_lower": pa.array(["examplus exemplum"], type=pa.string()),
            "name_class": pa.array(["scientific name"], type=pa.string()),
            "unique_name": pa.array([None], type=pa.string()),
        },
        schema=TAXONOMY_NAMES_TABLE,
    )
    merged = pa.table(
        {
            "old_taxid": pa.array([42], type=pa.int64()),
            "new_taxid": pa.array([100], type=pa.int64()),
        },
        schema=TAXONOMY_MERGED_TABLE,
    )
    r = TaxonomyResolver(nodes, names, merged)
    redirected = r.lookup(42)
    assert redirected is not None and redirected.taxid == 100
