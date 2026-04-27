"""``Network[NodeT, EdgeT]`` — generic Arrow-backed graph container.

Scope-reduced to exactly what ``core.structure.topology.Topology`` needs:
an Arrow ``nodes`` table, an Arrow ``edges`` table, neighbor lookup,
induced subgraphs, connected-components, basic predicates. Graph
algorithms beyond connected-components (BFS/DFS iterators, shortest
paths, adjacency tensors, spectral views) are deferred until a second
consumer arrives — likely massspec spectral-library nearest-neighbor
graphs or phylogeny / Tree work.

Edges are stored in an Arrow table with two integer / string columns
(``edge_src_col`` / ``edge_dst_col``) holding node ids that must appear
in ``nodes[node_id_col]``. Any number of additional edge-feature columns
may ride along (``order``, ``kind``, ``weight``, ...) — they survive
``induced_subgraph`` unchanged on the surviving rows.

Schema metadata uses the same ``pack_metadata`` / ``unpack_metadata``
helpers as ``core.io.schemas`` — re-imported, never duplicated.
"""

from __future__ import annotations

from typing import Any, Generic, Iterable, TypeVar

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import pack_metadata, unpack_metadata

NodeT = TypeVar("NodeT")
EdgeT = TypeVar("EdgeT")


# ──────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────


class Network(Generic[NodeT, EdgeT]):
    """Generic graph: an Arrow ``nodes`` table + an Arrow ``edges`` table.

    Node identity lives in ``nodes[node_id_col]`` (default ``"id"``);
    every value in ``edges[edge_src_col]`` / ``edges[edge_dst_col]``
    must appear there. Both directed and undirected modes are supported
    — the undirected case treats each edge row as an unordered pair, so
    callers should not list ``(a, b)`` and ``(b, a)`` as separate rows.

    The neighbor index is built lazily on first ``neighbors()`` /
    ``degree()`` / ``connected_components()`` call by sorting edges by
    src and prefix-scanning offsets (CSR-style). Any operation that
    returns a new ``Network`` (``induced_subgraph``, ``with_metadata``)
    rebuilds it — instances are otherwise immutable.
    """

    __slots__ = (
        "_nodes",
        "_edges",
        "_node_id_col",
        "_edge_src_col",
        "_edge_dst_col",
        "_directed",
        "_node_index",
        "_neighbors",
    )

    def __init__(
        self,
        nodes: pa.Table,
        edges: pa.Table,
        *,
        node_id_col: str = "id",
        edge_src_col: str = "src",
        edge_dst_col: str = "dst",
        directed: bool = False,
    ) -> None:
        if node_id_col not in nodes.column_names:
            raise ValueError(
                f"nodes table missing id column {node_id_col!r}; "
                f"have {nodes.column_names}"
            )
        for col in (edge_src_col, edge_dst_col):
            if col not in edges.column_names:
                raise ValueError(
                    f"edges table missing column {col!r}; have {edges.column_names}"
                )

        # Validate that every edge endpoint is a known node id.
        node_ids = nodes.column(node_id_col)
        node_id_set = set(node_ids.to_pylist())
        # Empty-table fast paths
        if edges.num_rows > 0:
            for col in (edge_src_col, edge_dst_col):
                vals = edges.column(col).to_pylist()
                missing = {v for v in vals if v not in node_id_set}
                if missing:
                    sample = sorted(missing, key=str)[:5]
                    raise ValueError(
                        f"edges.{col} references node ids not in nodes table: "
                        f"{sample}{'...' if len(missing) > 5 else ''}"
                    )

        # Reject duplicate node ids — a single-id-per-row contract.
        if len(node_id_set) != nodes.num_rows:
            raise ValueError(
                f"nodes.{node_id_col} contains duplicate ids "
                f"({nodes.num_rows - len(node_id_set)} duplicates)"
            )

        self._nodes: pa.Table = nodes
        self._edges: pa.Table = edges
        self._node_id_col: str = node_id_col
        self._edge_src_col: str = edge_src_col
        self._edge_dst_col: str = edge_dst_col
        self._directed: bool = directed
        # Lazy caches
        self._node_index: dict[Any, int] | None = None
        self._neighbors: dict[Any, list[Any]] | None = None

    # ── views ────────────────────────────────────────────────────────
    @property
    def nodes(self) -> pa.Table:
        return self._nodes

    @property
    def edges(self) -> pa.Table:
        return self._edges

    @property
    def directed(self) -> bool:
        return self._directed

    @property
    def node_id_col(self) -> str:
        return self._node_id_col

    @property
    def edge_src_col(self) -> str:
        return self._edge_src_col

    @property
    def edge_dst_col(self) -> str:
        return self._edge_dst_col

    def n_nodes(self) -> int:
        return self._nodes.num_rows

    def n_edges(self) -> int:
        return self._edges.num_rows

    def __len__(self) -> int:
        return self._nodes.num_rows

    # ── lazy neighbor index ─────────────────────────────────────────
    def _build_neighbor_index(self) -> dict[Any, list[Any]]:
        if self._neighbors is not None:
            return self._neighbors
        idx: dict[Any, list[Any]] = {
            nid: [] for nid in self._nodes.column(self._node_id_col).to_pylist()
        }
        if self._edges.num_rows > 0:
            srcs = self._edges.column(self._edge_src_col).to_pylist()
            dsts = self._edges.column(self._edge_dst_col).to_pylist()
            for s, d in zip(srcs, dsts):
                idx[s].append(d)
                if not self._directed and s != d:
                    idx[d].append(s)
        self._neighbors = idx
        return idx

    # ── lookups ──────────────────────────────────────────────────────
    def has_node(self, node_id: NodeT) -> bool:
        return node_id in self._build_neighbor_index()

    def neighbors(self, node_id: NodeT) -> list[NodeT]:
        idx = self._build_neighbor_index()
        if node_id not in idx:
            raise KeyError(f"unknown node id: {node_id!r}")
        return list(idx[node_id])

    def degree(self, node_id: NodeT) -> int:
        return len(self.neighbors(node_id))

    def has_edge(self, src: NodeT, dst: NodeT) -> bool:
        idx = self._build_neighbor_index()
        if src not in idx:
            return False
        return dst in idx[src]

    # ── derived graphs ──────────────────────────────────────────────
    def induced_subgraph(self, node_ids: Iterable[NodeT]) -> Network[NodeT, EdgeT]:
        """Return the subgraph induced by ``node_ids``.

        Surviving nodes are listed in their original order; edges are
        kept iff both endpoints survive. Edge feature columns are
        preserved; metadata is preserved.
        """
        keep = set(node_ids)
        # Filter nodes preserving original order.
        node_ids_col = self._nodes.column(self._node_id_col)
        node_mask = pc.is_in(node_ids_col, value_set=pa.array(list(keep)))
        new_nodes = self._nodes.filter(node_mask)

        # Filter edges where both endpoints survive.
        if self._edges.num_rows > 0:
            keep_arr = pa.array(list(keep))
            src_mask = pc.is_in(
                self._edges.column(self._edge_src_col), value_set=keep_arr
            )
            dst_mask = pc.is_in(
                self._edges.column(self._edge_dst_col), value_set=keep_arr
            )
            new_edges = self._edges.filter(pc.and_(src_mask, dst_mask))
        else:
            new_edges = self._edges

        return Network(
            new_nodes,
            new_edges,
            node_id_col=self._node_id_col,
            edge_src_col=self._edge_src_col,
            edge_dst_col=self._edge_dst_col,
            directed=self._directed,
        )

    def connected_components(self) -> list[list[NodeT]]:
        """Connected components (undirected only).

        Returns components sorted by descending size, with node ids
        within each component preserving their order in ``nodes``.
        """
        if self._directed:
            raise NotImplementedError(
                "connected_components requires an undirected graph; "
                "use induced_subgraph or rebuild without directed=True"
            )
        idx = self._build_neighbor_index()
        order = self._nodes.column(self._node_id_col).to_pylist()
        order_pos = {nid: i for i, nid in enumerate(order)}
        seen: set[Any] = set()
        components: list[list[Any]] = []
        for nid in order:
            if nid in seen:
                continue
            stack = [nid]
            comp: list[Any] = []
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                for nbr in idx[cur]:
                    if nbr not in seen:
                        stack.append(nbr)
            comp.sort(key=lambda n: order_pos[n])
            components.append(comp)
        components.sort(key=lambda c: -len(c))
        return components

    # ── metadata ────────────────────────────────────────────────────
    def with_metadata(self, extras: dict[str, Any]) -> Network[NodeT, EdgeT]:
        existing = unpack_metadata(self._nodes.schema.metadata)
        existing.update(extras)
        new_nodes = self._nodes.replace_schema_metadata(pack_metadata(existing))
        return Network(
            new_nodes,
            self._edges,
            node_id_col=self._node_id_col,
            edge_src_col=self._edge_src_col,
            edge_dst_col=self._edge_dst_col,
            directed=self._directed,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return unpack_metadata(self._nodes.schema.metadata)

    # ── factories ───────────────────────────────────────────────────
    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[NodeT, NodeT]],
        *,
        node_ids: Iterable[NodeT] | None = None,
        directed: bool = False,
    ) -> Network[NodeT, EdgeT]:
        """Build a ``Network`` from an iterable of ``(src, dst)`` pairs.

        If ``node_ids`` is omitted, the node set is the union of all
        endpoints in ``edges``, in first-encounter order.
        """
        edge_pairs = list(edges)
        if node_ids is None:
            seen: dict[Any, None] = {}
            for s, d in edge_pairs:
                seen.setdefault(s, None)
                seen.setdefault(d, None)
            ids = list(seen.keys())
        else:
            ids = list(node_ids)
        nodes_tbl = pa.table({"id": ids})
        if edge_pairs:
            srcs = [s for s, _ in edge_pairs]
            dsts = [d for _, d in edge_pairs]
            edges_tbl = pa.table({"src": srcs, "dst": dsts})
        else:
            # Empty-edges table needs typed columns; infer from node ids.
            node_type = nodes_tbl.column("id").type
            edges_tbl = pa.table(
                {
                    "src": pa.array([], type=node_type),
                    "dst": pa.array([], type=node_type),
                }
            )
        return cls(nodes_tbl, edges_tbl, directed=directed)

    # ── repr ────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        kind = "directed" if self._directed else "undirected"
        return f"Network({kind}, n_nodes={self.n_nodes()}, n_edges={self.n_edges()})"


__all__ = ["Network"]
