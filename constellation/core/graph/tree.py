"""``Tree[T]`` — Arrow-backed single-parent tree primitive.

Mirrors ``Network``'s contract — Arrow nodes table, lazy CSR-style
children index, immutable, ``with_metadata`` factory. The single-parent
invariant is enforced in ``__init__``: every node appears exactly once
in the nodes table and references at most one parent that is itself in
the table (or the ``root_sentinel``). Cycles are caught by a one-pass
depth walk during validation.

First consumer is ``core.taxonomy`` (NCBI taxonomy tree); phylogeny work
later (branch lengths, Newick I/O, support values) layers on top.
"""

from __future__ import annotations

from typing import Any, Generic, Iterable, TypeVar

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import pack_metadata, unpack_metadata

T = TypeVar("T")


# Sentinel for "root has no parent in the nodes table". Stored on the
# instance so callers asking for ``parent_of(root)`` get back ``None``
# regardless of how the input table encoded rootness.
_NO_PARENT = object()


class Tree(Generic[T]):
    """Generic tree: an Arrow ``nodes`` table with a parent-id column.

    Every node id in ``nodes[node_id_col]`` is unique. Every value in
    ``nodes[parent_id_col]`` is either (a) another id in the same
    column or (b) the ``root_sentinel`` (often ``None`` or a self-
    reference like NCBI's ``taxid=1 -> parent=1``). Forests — multiple
    roots — are supported via ``roots()``.

    The children index is built lazily on first ``children_of()`` /
    ``descendants_of()`` / ``lca()`` call by single-pass grouping over
    the parent column. Operations that return a new ``Tree`` rebuild
    the index — instances are otherwise immutable.
    """

    __slots__ = (
        "_nodes",
        "_node_id_col",
        "_parent_id_col",
        "_root_sentinel",
        "_parent",
        "_children",
        "_depth",
        "_roots",
    )

    def __init__(
        self,
        nodes: pa.Table,
        *,
        node_id_col: str = "id",
        parent_id_col: str = "parent_id",
        root_sentinel: Any | None = None,
    ) -> None:
        if node_id_col not in nodes.column_names:
            raise ValueError(
                f"nodes table missing id column {node_id_col!r}; "
                f"have {nodes.column_names}"
            )
        if parent_id_col not in nodes.column_names:
            raise ValueError(
                f"nodes table missing parent column {parent_id_col!r}; "
                f"have {nodes.column_names}"
            )

        ids = nodes.column(node_id_col).to_pylist()
        parents = nodes.column(parent_id_col).to_pylist()
        id_set = set(ids)
        if len(id_set) != len(ids):
            n_dup = len(ids) - len(id_set)
            raise ValueError(
                f"nodes.{node_id_col} contains {n_dup} duplicate id(s)"
            )

        # Build the validated parent map. A node is a root iff its
        # encoded parent is the sentinel, missing, or self-referential.
        parent: dict[Any, Any] = {}
        roots: list[Any] = []
        for nid, pid in zip(ids, parents):
            if pid == root_sentinel or pid is None or pid == nid:
                parent[nid] = _NO_PARENT
                roots.append(nid)
                continue
            if pid not in id_set:
                raise ValueError(
                    f"node {nid!r} references parent {pid!r} which is not in "
                    f"the nodes table; pass root_sentinel={pid!r} if it is "
                    "meant to mark the root"
                )
            parent[nid] = pid

        if not roots and ids:
            raise ValueError(
                "tree has no roots — every node references an in-table "
                "parent. This implies a cycle; double-check parent_id_col."
            )

        # Cycle detection + depth via memoised walk to root.
        depth: dict[Any, int] = {}
        for nid in ids:
            if nid in depth:
                continue
            chain: list[Any] = []
            cur = nid
            visiting: set[Any] = set()
            while cur not in depth:
                if cur in visiting:
                    raise ValueError(
                        f"cycle detected involving node {cur!r}; "
                        f"path: {chain + [cur]}"
                    )
                visiting.add(cur)
                chain.append(cur)
                p = parent[cur]
                if p is _NO_PARENT:
                    depth[cur] = 0
                    chain.pop()
                    break
                cur = p
            base = depth[cur]
            for offset, n in enumerate(reversed(chain), start=1):
                depth[n] = base + offset

        self._nodes: pa.Table = nodes
        self._node_id_col: str = node_id_col
        self._parent_id_col: str = parent_id_col
        self._root_sentinel: Any | None = root_sentinel
        self._parent: dict[Any, Any] = parent
        self._depth: dict[Any, int] = depth
        self._roots: list[Any] = roots
        # Children index is built on first query.
        self._children: dict[Any, list[Any]] | None = None

    # ── views ────────────────────────────────────────────────────────
    @property
    def nodes(self) -> pa.Table:
        return self._nodes

    @property
    def node_id_col(self) -> str:
        return self._node_id_col

    @property
    def parent_id_col(self) -> str:
        return self._parent_id_col

    def n_nodes(self) -> int:
        return self._nodes.num_rows

    def __len__(self) -> int:
        return self._nodes.num_rows

    def root(self) -> T:
        """The single root. Raises if the tree is empty or is a forest."""
        if not self._roots:
            raise ValueError("tree is empty — no root")
        if len(self._roots) > 1:
            raise ValueError(
                f"tree has {len(self._roots)} roots (forest); use .roots() instead"
            )
        return self._roots[0]

    def roots(self) -> list[T]:
        """All roots — supports forests."""
        return list(self._roots)

    # ── lazy children index ─────────────────────────────────────────
    def _build_children_index(self) -> dict[Any, list[Any]]:
        if self._children is not None:
            return self._children
        idx: dict[Any, list[Any]] = {
            nid: [] for nid in self._nodes.column(self._node_id_col).to_pylist()
        }
        for nid, pid in self._parent.items():
            if pid is _NO_PARENT:
                continue
            idx[pid].append(nid)
        self._children = idx
        return idx

    # ── lookups ──────────────────────────────────────────────────────
    def has_node(self, node_id: T) -> bool:
        return node_id in self._parent

    def parent_of(self, node_id: T) -> T | None:
        if node_id not in self._parent:
            raise KeyError(f"unknown node id: {node_id!r}")
        pid = self._parent[node_id]
        return None if pid is _NO_PARENT else pid

    def children_of(self, node_id: T) -> list[T]:
        idx = self._build_children_index()
        if node_id not in idx:
            raise KeyError(f"unknown node id: {node_id!r}")
        return list(idx[node_id])

    def depth_of(self, node_id: T) -> int:
        if node_id not in self._depth:
            raise KeyError(f"unknown node id: {node_id!r}")
        return self._depth[node_id]

    # ── tree-shaped queries ─────────────────────────────────────────
    def ancestors_of(self, node_id: T) -> list[T]:
        """Ancestors in leaf → root order (excludes ``node_id`` itself)."""
        if node_id not in self._parent:
            raise KeyError(f"unknown node id: {node_id!r}")
        out: list[Any] = []
        cur = self._parent[node_id]
        while cur is not _NO_PARENT:
            out.append(cur)
            cur = self._parent[cur]
        return out

    def lineage(self, node_id: T) -> list[T]:
        """Lineage in root → ``node_id`` order (includes both endpoints)."""
        anc = self.ancestors_of(node_id)
        return list(reversed(anc)) + [node_id]

    def descendants_of(
        self,
        node_id: T,
        *,
        max_depth: int | None = None,
    ) -> list[T]:
        """All descendants below ``node_id`` (excludes ``node_id``).

        ``max_depth`` caps the relative depth from ``node_id`` (1 → only
        direct children, 2 → children + grandchildren, ...).
        """
        if node_id not in self._parent:
            raise KeyError(f"unknown node id: {node_id!r}")
        idx = self._build_children_index()
        out: list[Any] = []
        stack: list[tuple[Any, int]] = [(c, 1) for c in idx[node_id]]
        while stack:
            cur, d = stack.pop()
            out.append(cur)
            if max_depth is None or d < max_depth:
                for c in idx[cur]:
                    stack.append((c, d + 1))
        return out

    def lca(self, a: T, b: T) -> T | None:
        """Lowest common ancestor of ``a`` and ``b``.

        Returns ``None`` if the two nodes lie in different trees of a
        forest. Returns ``a`` (or ``b``) if one is an ancestor of the
        other. When ``a == b``, returns ``a``.
        """
        if a not in self._parent:
            raise KeyError(f"unknown node id: {a!r}")
        if b not in self._parent:
            raise KeyError(f"unknown node id: {b!r}")
        if a == b:
            return a
        # Walk a to root collecting ancestor set.
        a_chain: set[Any] = {a}
        cur = self._parent[a]
        while cur is not _NO_PARENT:
            a_chain.add(cur)
            cur = self._parent[cur]
        # Walk b upward; first hit in a_chain is the LCA.
        if b in a_chain:
            return b
        cur = self._parent[b]
        while cur is not _NO_PARENT:
            if cur in a_chain:
                return cur
            cur = self._parent[cur]
        return None

    def subtree_at(self, node_id: T) -> "Tree[T]":
        """Return the subtree rooted at ``node_id`` as a new ``Tree``.

        ``node_id`` becomes a root in the returned tree (its parent
        column is set to the sentinel). Metadata is preserved.
        """
        if node_id not in self._parent:
            raise KeyError(f"unknown node id: {node_id!r}")
        keep = {node_id, *self.descendants_of(node_id)}
        ids_col = self._nodes.column(self._node_id_col)
        mask = pc.is_in(ids_col, value_set=pa.array(list(keep)))
        new_nodes = self._nodes.filter(mask)

        # Rewrite parent column so ``node_id`` becomes a root.
        parent_dtype = self._nodes.schema.field(self._parent_id_col).type
        parent_col = new_nodes.column(self._parent_id_col).to_pylist()
        id_col = new_nodes.column(self._node_id_col).to_pylist()
        rewritten = [
            self._root_sentinel if nid == node_id else pid
            for nid, pid in zip(id_col, parent_col)
        ]
        new_nodes = new_nodes.set_column(
            new_nodes.schema.get_field_index(self._parent_id_col),
            self._parent_id_col,
            pa.array(rewritten, type=parent_dtype),
        )
        return Tree(
            new_nodes,
            node_id_col=self._node_id_col,
            parent_id_col=self._parent_id_col,
            root_sentinel=self._root_sentinel,
        )

    # ── metadata ────────────────────────────────────────────────────
    def with_metadata(self, extras: dict[str, Any]) -> "Tree[T]":
        existing = unpack_metadata(self._nodes.schema.metadata)
        existing.update(extras)
        new_nodes = self._nodes.replace_schema_metadata(pack_metadata(existing))
        return Tree(
            new_nodes,
            node_id_col=self._node_id_col,
            parent_id_col=self._parent_id_col,
            root_sentinel=self._root_sentinel,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return unpack_metadata(self._nodes.schema.metadata)

    # ── factories ───────────────────────────────────────────────────
    @classmethod
    def from_parent_pairs(
        cls,
        pairs: Iterable[tuple[T, T | None]],
        *,
        root_sentinel: Any | None = None,
    ) -> "Tree[T]":
        """Build a ``Tree`` from an iterable of ``(node_id, parent_id)`` pairs.

        ``parent_id == root_sentinel`` (default ``None``) marks the root.
        """
        pair_list = list(pairs)
        if not pair_list:
            return cls(
                pa.table({"id": pa.array([], type=pa.int64()), "parent_id": pa.array([], type=pa.int64())}),
                root_sentinel=root_sentinel,
            )
        ids = [p[0] for p in pair_list]
        parents = [p[1] for p in pair_list]
        tbl = pa.table({"id": ids, "parent_id": parents})
        return cls(tbl, root_sentinel=root_sentinel)

    # ── repr ────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        n_roots = len(self._roots)
        kind = "tree" if n_roots == 1 else f"forest({n_roots} roots)"
        return f"Tree({kind}, n_nodes={self.n_nodes()})"


__all__ = ["Tree"]
