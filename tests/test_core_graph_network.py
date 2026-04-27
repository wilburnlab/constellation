"""Unit tests for ``constellation.core.graph.network.Network``."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.core.graph import Network


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


def test_from_edges_with_implicit_node_ids():
    g = Network.from_edges([("A", "B"), ("B", "C"), ("C", "A")])
    assert g.n_nodes() == 3
    assert g.n_edges() == 3
    assert not g.directed


def test_from_edges_with_explicit_node_ids_includes_isolated():
    g = Network.from_edges([("A", "B")], node_ids=["A", "B", "C"])
    assert g.n_nodes() == 3
    assert g.has_node("C")
    assert g.degree("C") == 0


def test_from_edges_empty_returns_empty_graph():
    g = Network.from_edges([], node_ids=["A", "B"])
    assert g.n_nodes() == 2 and g.n_edges() == 0


def test_constructor_validates_edge_endpoints():
    nodes = pa.table({"id": ["A", "B"]})
    edges = pa.table({"src": ["A"], "dst": ["Z"]})
    with pytest.raises(ValueError, match="not in nodes table"):
        Network(nodes, edges)


def test_constructor_rejects_duplicate_node_ids():
    nodes = pa.table({"id": ["A", "A", "B"]})
    edges = pa.table({"src": ["A"], "dst": ["B"]})
    with pytest.raises(ValueError, match="duplicate"):
        Network(nodes, edges)


def test_constructor_validates_id_columns_present():
    nodes = pa.table({"name": ["A", "B"]})
    edges = pa.table({"src": ["A"], "dst": ["B"]})
    with pytest.raises(ValueError, match="missing id column"):
        Network(nodes, edges, node_id_col="id")


# ──────────────────────────────────────────────────────────────────────
# Neighbors / degree / has_edge
# ──────────────────────────────────────────────────────────────────────


def test_undirected_triangle_neighbors_symmetric():
    g = Network.from_edges([("A", "B"), ("B", "C"), ("C", "A")])
    assert sorted(g.neighbors("A")) == ["B", "C"]
    assert sorted(g.neighbors("B")) == ["A", "C"]
    assert sorted(g.neighbors("C")) == ["A", "B"]
    assert g.degree("A") == 2


def test_directed_triangle_neighbors_directional():
    g = Network.from_edges([("A", "B"), ("B", "C"), ("C", "A")], directed=True)
    assert g.neighbors("A") == ["B"]
    assert g.neighbors("B") == ["C"]
    assert g.neighbors("C") == ["A"]


def test_has_edge_undirected():
    g = Network.from_edges([("A", "B")])
    assert g.has_edge("A", "B")
    assert g.has_edge("B", "A")  # symmetric
    assert not g.has_edge("A", "Z")  # silent False for unknown


def test_has_edge_directed():
    g = Network.from_edges([("A", "B")], directed=True)
    assert g.has_edge("A", "B")
    assert not g.has_edge("B", "A")


def test_neighbors_unknown_node_raises():
    g = Network.from_edges([("A", "B")])
    with pytest.raises(KeyError):
        g.neighbors("Z")


# ──────────────────────────────────────────────────────────────────────
# Induced subgraph
# ──────────────────────────────────────────────────────────────────────


def test_induced_subgraph_drops_dangling_edges():
    g = Network.from_edges([("A", "B"), ("B", "C"), ("C", "D")])
    sub = g.induced_subgraph(["A", "B", "C"])
    assert sub.n_nodes() == 3
    # Edge B-C survives; C-D drops because D was excluded.
    assert sub.has_edge("A", "B") and sub.has_edge("B", "C")
    assert not sub.has_node("D")


def test_induced_subgraph_preserves_edge_features():
    nodes = pa.table({"id": ["A", "B", "C"]})
    edges = pa.table({"src": ["A", "B"], "dst": ["B", "C"], "weight": [1.5, 2.5]})
    g = Network(nodes, edges)
    sub = g.induced_subgraph(["A", "B"])
    assert "weight" in sub.edges.column_names
    assert sub.edges.column("weight").to_pylist() == [1.5]


# ──────────────────────────────────────────────────────────────────────
# Connected components
# ──────────────────────────────────────────────────────────────────────


def test_connected_components_disconnected():
    g = Network.from_edges(
        [("A", "B"), ("C", "D"), ("D", "E")],
        node_ids=["A", "B", "C", "D", "E", "F"],
    )
    comps = g.connected_components()
    # Sorted by descending size: {C, D, E}, {A, B}, {F}.
    assert [len(c) for c in comps] == [3, 2, 1]
    assert set(comps[0]) == {"C", "D", "E"}
    assert set(comps[1]) == {"A", "B"}
    assert comps[2] == ["F"]


def test_connected_components_directed_raises():
    g = Network.from_edges([("A", "B")], directed=True)
    with pytest.raises(NotImplementedError):
        g.connected_components()


def test_connected_components_singleton():
    g = Network.from_edges([], node_ids=["A"])
    assert g.connected_components() == [["A"]]


# ──────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────


def test_with_metadata_round_trip():
    g = Network.from_edges([("A", "B")])
    g2 = g.with_metadata({"name": "edge_one", "version": 3})
    assert g2.metadata["name"] == "edge_one"
    assert g2.metadata["version"] == 3
    # Original is unchanged.
    assert g.metadata == {}


# ──────────────────────────────────────────────────────────────────────
# Repr
# ──────────────────────────────────────────────────────────────────────


def test_repr_includes_counts_and_kind():
    g = Network.from_edges([("A", "B")], directed=True)
    s = repr(g)
    assert "directed" in s and "n_nodes=2" in s and "n_edges=1" in s
