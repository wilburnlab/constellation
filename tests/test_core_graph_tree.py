"""Unit tests for ``constellation.core.graph.tree.Tree``."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.core.graph import Tree


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


def _toy_tree() -> Tree[str]:
    """A tiny taxonomy-shaped tree:

        root
        ├── eukaryota
        │   ├── metazoa
        │   │   └── homo_sapiens
        │   └── fungi
        │       └── saccharomyces
        └── bacteria
            └── ecoli
    """
    pairs = [
        ("root", None),
        ("eukaryota", "root"),
        ("metazoa", "eukaryota"),
        ("homo_sapiens", "metazoa"),
        ("fungi", "eukaryota"),
        ("saccharomyces", "fungi"),
        ("bacteria", "root"),
        ("ecoli", "bacteria"),
    ]
    return Tree.from_parent_pairs(pairs)


def test_from_parent_pairs_smoke():
    t = _toy_tree()
    assert t.n_nodes() == 8
    assert t.root() == "root"


def test_constructor_rejects_duplicate_ids():
    tbl = pa.table({"id": ["A", "A"], "parent_id": [None, "A"]})
    with pytest.raises(ValueError, match="duplicate"):
        Tree(tbl)


def test_constructor_rejects_missing_id_column():
    tbl = pa.table({"name": ["A"], "parent_id": [None]})
    with pytest.raises(ValueError, match="missing id column"):
        Tree(tbl)


def test_constructor_rejects_missing_parent_column():
    tbl = pa.table({"id": ["A"], "parent_name": [None]})
    with pytest.raises(ValueError, match="missing parent column"):
        Tree(tbl)


def test_constructor_rejects_dangling_parent():
    tbl = pa.table({"id": ["A", "B"], "parent_id": [None, "ghost"]})
    with pytest.raises(ValueError, match="not in the nodes table"):
        Tree(tbl)


def test_constructor_detects_cycle():
    tbl = pa.table({"id": ["A", "B", "C"], "parent_id": ["B", "C", "A"]})
    with pytest.raises(ValueError, match="cycle"):
        Tree(tbl)


def test_constructor_handles_self_referential_root():
    """NCBI taxonomy stores the root as taxid=1, parent_taxid=1."""
    tbl = pa.table({"id": [1, 2, 3], "parent_id": [1, 1, 2]})
    t = Tree(tbl)
    assert t.root() == 1
    assert t.parent_of(1) is None


def test_constructor_accepts_explicit_root_sentinel():
    tbl = pa.table({"id": [1, 2, 3], "parent_id": [0, 1, 2]})
    t = Tree(tbl, root_sentinel=0)
    assert t.root() == 1
    assert t.parent_of(1) is None


def test_empty_tree_constructor():
    t = Tree.from_parent_pairs([])
    assert t.n_nodes() == 0
    with pytest.raises(ValueError, match="empty"):
        t.root()


# ──────────────────────────────────────────────────────────────────────
# Forest support
# ──────────────────────────────────────────────────────────────────────


def test_forest_with_multiple_roots():
    pairs = [
        ("a_root", None),
        ("a_child", "a_root"),
        ("b_root", None),
        ("b_child", "b_root"),
    ]
    t = Tree.from_parent_pairs(pairs)
    assert sorted(t.roots()) == ["a_root", "b_root"]
    with pytest.raises(ValueError, match="forest"):
        t.root()


# ──────────────────────────────────────────────────────────────────────
# Lookups
# ──────────────────────────────────────────────────────────────────────


def test_has_node_and_parent_of():
    t = _toy_tree()
    assert t.has_node("homo_sapiens")
    assert not t.has_node("ghost")
    assert t.parent_of("homo_sapiens") == "metazoa"
    assert t.parent_of("root") is None


def test_children_of():
    t = _toy_tree()
    assert sorted(t.children_of("eukaryota")) == ["fungi", "metazoa"]
    assert t.children_of("ecoli") == []


def test_depth_of():
    t = _toy_tree()
    assert t.depth_of("root") == 0
    assert t.depth_of("eukaryota") == 1
    assert t.depth_of("metazoa") == 2
    assert t.depth_of("homo_sapiens") == 3


def test_parent_of_unknown_raises():
    t = _toy_tree()
    with pytest.raises(KeyError):
        t.parent_of("ghost")


# ──────────────────────────────────────────────────────────────────────
# Tree-shaped queries
# ──────────────────────────────────────────────────────────────────────


def test_ancestors_of_in_leaf_to_root_order():
    t = _toy_tree()
    assert t.ancestors_of("homo_sapiens") == ["metazoa", "eukaryota", "root"]
    assert t.ancestors_of("root") == []


def test_lineage_in_root_to_node_order():
    t = _toy_tree()
    assert t.lineage("homo_sapiens") == ["root", "eukaryota", "metazoa", "homo_sapiens"]
    assert t.lineage("root") == ["root"]


def test_descendants_of_includes_grandchildren():
    t = _toy_tree()
    assert sorted(t.descendants_of("eukaryota")) == [
        "fungi",
        "homo_sapiens",
        "metazoa",
        "saccharomyces",
    ]
    assert t.descendants_of("homo_sapiens") == []
    assert t.descendants_of("root") == [
        d for d in t.descendants_of("root")  # just verify non-empty
    ]
    assert len(t.descendants_of("root")) == 7


def test_descendants_of_respects_max_depth():
    t = _toy_tree()
    assert sorted(t.descendants_of("eukaryota", max_depth=1)) == ["fungi", "metazoa"]
    assert len(t.descendants_of("eukaryota", max_depth=2)) == 4


def test_lca_self():
    t = _toy_tree()
    assert t.lca("homo_sapiens", "homo_sapiens") == "homo_sapiens"


def test_lca_one_is_ancestor():
    t = _toy_tree()
    assert t.lca("homo_sapiens", "eukaryota") == "eukaryota"
    assert t.lca("eukaryota", "homo_sapiens") == "eukaryota"


def test_lca_siblings():
    t = _toy_tree()
    assert t.lca("homo_sapiens", "saccharomyces") == "eukaryota"


def test_lca_across_clades():
    t = _toy_tree()
    assert t.lca("homo_sapiens", "ecoli") == "root"


def test_lca_across_disjoint_forest_is_none():
    pairs = [
        ("a_root", None),
        ("a_child", "a_root"),
        ("b_root", None),
        ("b_child", "b_root"),
    ]
    t = Tree.from_parent_pairs(pairs)
    assert t.lca("a_child", "b_child") is None


# ──────────────────────────────────────────────────────────────────────
# Subtree extraction
# ──────────────────────────────────────────────────────────────────────


def test_subtree_at_root_returns_full_tree():
    t = _toy_tree()
    sub = t.subtree_at("root")
    assert sub.n_nodes() == t.n_nodes()


def test_subtree_at_internal_node():
    t = _toy_tree()
    sub = t.subtree_at("eukaryota")
    # eukaryota + metazoa + homo_sapiens + fungi + saccharomyces
    assert sub.n_nodes() == 5
    assert sub.root() == "eukaryota"
    assert sub.parent_of("eukaryota") is None
    assert sub.parent_of("metazoa") == "eukaryota"
    assert sub.lineage("homo_sapiens") == ["eukaryota", "metazoa", "homo_sapiens"]


def test_subtree_at_leaf_is_single_node():
    t = _toy_tree()
    sub = t.subtree_at("homo_sapiens")
    assert sub.n_nodes() == 1
    assert sub.root() == "homo_sapiens"


# ──────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────


def test_with_metadata_round_trips():
    t = _toy_tree()
    decorated = t.with_metadata({"source": "test", "version": 3})
    assert decorated.metadata == {"source": "test", "version": 3}
    # Original is untouched.
    assert t.metadata == {}


# ──────────────────────────────────────────────────────────────────────
# Repr
# ──────────────────────────────────────────────────────────────────────


def test_repr_distinguishes_tree_from_forest():
    t = _toy_tree()
    assert "tree" in repr(t)
    forest = Tree.from_parent_pairs([("a", None), ("b", None)])
    assert "forest" in repr(forest)
