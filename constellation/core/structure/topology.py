"""Topology — atoms + bonds / angles / dihedrals + connectivity views.

A ``Topology`` wraps an Arrow ``atoms`` table (conforming to
``ATOM_TABLE``) plus optional Arrow tables for bonds, angles, and
dihedrals. Bond-graph operations route through a lazy ``Network`` view
over the bond table.

Index-integrity contract (load-bearing).
========================================

``bonds.i`` / ``bonds.j`` (and the analogous columns in ``angles`` /
``dihedrals``) are **row indices into the atoms table**, NOT
``atoms.serial`` values. PDB serial numbers are arbitrary tokens
chosen by the writer; row indices are the only stable references
under filtering / subsetting. ``Topology.subset(indices)`` remaps
indices through a translation array so bonds in the returned
sub-topology refer to the new atom positions.

This convention is what every structural toolkit gets wrong at least
once. Surface it loudly in docstrings and verify it in tests.
"""

from __future__ import annotations

from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import torch

from constellation.core.chem.elements import ELEMENTS
from constellation.core.graph.network import Network


# ──────────────────────────────────────────────────────────────────────
# Empty-table schemas for bonds / angles / dihedrals
# ──────────────────────────────────────────────────────────────────────

_EMPTY_BONDS = pa.schema(
    [
        pa.field("i", pa.int32(), nullable=False),
        pa.field("j", pa.int32(), nullable=False),
        pa.field("order", pa.float32(), nullable=True),
        pa.field("kind", pa.string(), nullable=True),
    ]
)

_EMPTY_ANGLES = pa.schema(
    [
        pa.field("i", pa.int32(), nullable=False),
        pa.field("j", pa.int32(), nullable=False),
        pa.field("k", pa.int32(), nullable=False),
    ]
)

_EMPTY_DIHEDRALS = pa.schema(
    [
        pa.field("i", pa.int32(), nullable=False),
        pa.field("j", pa.int32(), nullable=False),
        pa.field("k", pa.int32(), nullable=False),
        pa.field("l", pa.int32(), nullable=False),
        pa.field("kind", pa.string(), nullable=True),
    ]
)


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)


# ──────────────────────────────────────────────────────────────────────
# Topology
# ──────────────────────────────────────────────────────────────────────


class Topology:
    """Atom table + optional bond / angle / dihedral tables.

    See module docstring for the index-integrity contract — every
    integer column in the connectivity tables is a row index into
    ``atoms``, not a serial / chain-residue identifier.
    """

    __slots__ = (
        "_atoms",
        "_bonds",
        "_angles",
        "_dihedrals",
        "_metadata",
        "_bond_network_cache",
    )

    def __init__(
        self,
        atoms: pa.Table,
        *,
        bonds: pa.Table | None = None,
        angles: pa.Table | None = None,
        dihedrals: pa.Table | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        n_atoms = atoms.num_rows
        self._atoms: pa.Table = atoms
        self._bonds: pa.Table = (
            self._validate_indices(bonds, ("i", "j"), n_atoms, "bonds")
            if bonds is not None
            else _empty_table(_EMPTY_BONDS)
        )
        self._angles: pa.Table = (
            self._validate_indices(angles, ("i", "j", "k"), n_atoms, "angles")
            if angles is not None
            else _empty_table(_EMPTY_ANGLES)
        )
        self._dihedrals: pa.Table = (
            self._validate_indices(
                dihedrals, ("i", "j", "k", "l"), n_atoms, "dihedrals"
            )
            if dihedrals is not None
            else _empty_table(_EMPTY_DIHEDRALS)
        )
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self._bond_network_cache: Network[int, dict] | None = None

    @staticmethod
    def _validate_indices(
        table: pa.Table,
        cols: tuple[str, ...],
        n_atoms: int,
        kind: str,
    ) -> pa.Table:
        for c in cols:
            if c not in table.column_names:
                raise ValueError(f"{kind} table missing required column {c!r}")
            arr = table.column(c).to_numpy(zero_copy_only=False)
            if len(arr) == 0:
                continue
            mn = int(arr.min())
            mx = int(arr.max())
            if mn < 0 or mx >= n_atoms:
                raise ValueError(
                    f"{kind}.{c} out of range: index {mn}..{mx} not in [0, {n_atoms})"
                )
        return table

    # ── views ────────────────────────────────────────────────────────
    @property
    def atoms(self) -> pa.Table:
        return self._atoms

    @property
    def bonds(self) -> pa.Table:
        return self._bonds

    @property
    def angles(self) -> pa.Table:
        return self._angles

    @property
    def dihedrals(self) -> pa.Table:
        return self._dihedrals

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def n_atoms(self) -> int:
        return self._atoms.num_rows

    def n_bonds(self) -> int:
        return self._bonds.num_rows

    def chain_ids(self) -> list[str]:
        """Distinct chain ids in first-encounter order."""
        seen: dict[str, None] = {}
        for cid in self._atoms.column("chain_id").to_pylist():
            seen.setdefault(cid, None)
        return list(seen.keys())

    def n_chains(self) -> int:
        return len(self.chain_ids())

    def n_residues(self) -> int:
        return self.residue_groups().num_rows

    # ── residue grouping ─────────────────────────────────────────────
    def residue_groups(self) -> pa.Table:
        """One row per ``(chain_id, res_seq, ins_code)`` group, in the
        order each group first appears in ``atoms``.

        Columns: ``chain_id``, ``res_seq``, ``ins_code``, ``res_name``,
        ``first_index`` (row index of the first atom in the group),
        ``n_atoms`` (number of atoms in the group).

        Assumes atoms within a residue are contiguous in the table —
        the standard PDB / mmCIF / MD topology convention. Disjoint
        residues (same identity, multiple disjoint runs of rows) are
        reported as a single group with the first run's start index
        and the total count; callers needing per-run indexing should
        use ``select_indices`` directly.
        """
        n = self._atoms.num_rows
        if n == 0:
            return pa.table(
                {
                    "chain_id": pa.array([], type=pa.string()),
                    "res_seq": pa.array([], type=pa.int32()),
                    "ins_code": pa.array([], type=pa.string()),
                    "res_name": pa.array([], type=pa.string()),
                    "first_index": pa.array([], type=pa.int32()),
                    "n_atoms": pa.array([], type=pa.int32()),
                }
            )
        chain_ids = self._atoms.column("chain_id").to_pylist()
        res_seqs = self._atoms.column("res_seq").to_pylist()
        ins_codes = self._atoms.column("ins_code").to_pylist()
        res_names = self._atoms.column("res_name").to_pylist()

        groups: dict[tuple, dict[str, Any]] = {}
        for idx in range(n):
            key = (chain_ids[idx], res_seqs[idx], ins_codes[idx])
            if key in groups:
                groups[key]["n_atoms"] += 1
            else:
                groups[key] = {
                    "chain_id": chain_ids[idx],
                    "res_seq": res_seqs[idx],
                    "ins_code": ins_codes[idx],
                    "res_name": res_names[idx],
                    "first_index": idx,
                    "n_atoms": 1,
                }
        return pa.table(
            {
                "chain_id": pa.array(
                    [g["chain_id"] for g in groups.values()], type=pa.string()
                ),
                "res_seq": pa.array(
                    [g["res_seq"] for g in groups.values()], type=pa.int32()
                ),
                "ins_code": pa.array(
                    [g["ins_code"] for g in groups.values()], type=pa.string()
                ),
                "res_name": pa.array(
                    [g["res_name"] for g in groups.values()], type=pa.string()
                ),
                "first_index": pa.array(
                    [g["first_index"] for g in groups.values()], type=pa.int32()
                ),
                "n_atoms": pa.array(
                    [g["n_atoms"] for g in groups.values()], type=pa.int32()
                ),
            }
        )

    # ── selection / subsetting ──────────────────────────────────────
    def select_indices(self, expr: pc.Expression) -> torch.Tensor:
        """Evaluate a selection expression against ``atoms`` and return
        the matching row indices as a ``torch.int64`` tensor.

        The expression must reference only ``ATOM_TABLE`` columns
        — see ``constellation.core.structure.selection`` for the
        canonical predicate helpers.
        """
        idx_col = pa.array(range(self._atoms.num_rows), type=pa.int64())
        with_idx = self._atoms.append_column("__row_idx", idx_col)
        kept = (
            with_idx.filter(expr)
            .column("__row_idx")
            .to_numpy(zero_copy_only=False)
            .copy()
        )
        return torch.from_numpy(kept).to(torch.int64)

    def subset(self, indices: torch.Tensor | list[int] | Iterable[int]) -> Topology:
        """Induced sub-topology over the given atom row indices.

        Bonds, angles, and dihedrals are filtered to those whose every
        endpoint is in ``indices``, and their integer columns are
        **remapped** to the new row positions in the kept atom table.
        Indices are taken in the given order — the resulting atom
        table preserves that ordering, not the original ``atoms``
        order.
        """
        if isinstance(indices, torch.Tensor):
            idx_list = indices.to(torch.int64).tolist()
        else:
            idx_list = list(indices)
        n = self._atoms.num_rows
        for i in idx_list:
            if i < 0 or i >= n:
                raise IndexError(f"atom index {i} out of range [0, {n})")

        new_atoms = self._atoms.take(pa.array(idx_list, type=pa.int64()))
        # Translation: old row -> new row position. -1 for dropped atoms.
        translation = [-1] * n
        for new_pos, old in enumerate(idx_list):
            translation[old] = new_pos

        new_bonds = _filter_and_remap(self._bonds, ("i", "j"), translation)
        new_angles = _filter_and_remap(self._angles, ("i", "j", "k"), translation)
        new_dihedrals = _filter_and_remap(
            self._dihedrals, ("i", "j", "k", "l"), translation
        )

        return Topology(
            new_atoms,
            bonds=new_bonds,
            angles=new_angles,
            dihedrals=new_dihedrals,
            metadata=self._metadata,
        )

    # ── connectivity ────────────────────────────────────────────────
    @property
    def bond_network(self) -> Network[int, dict]:
        """Lazy ``Network`` view over the bond table.

        Nodes are atom row indices (``0..n_atoms``); edges carry any
        feature columns (``order``, ``kind``) that ride on the bond
        table. Undirected. Cached after first access.
        """
        if self._bond_network_cache is not None:
            return self._bond_network_cache
        node_ids = pa.array(range(self._atoms.num_rows), type=pa.int32())
        nodes_tbl = pa.table({"id": node_ids})
        edges_tbl = self._bonds.rename_columns(
            [
                "src" if c == "i" else "dst" if c == "j" else c
                for c in self._bonds.column_names
            ]
        )
        net: Network[int, dict] = Network(nodes_tbl, edges_tbl, directed=False)
        self._bond_network_cache = net
        return net

    # ── metadata ────────────────────────────────────────────────────
    def with_metadata(self, extras: dict[str, Any]) -> Topology:
        merged = dict(self._metadata)
        merged.update(extras)
        return Topology(
            self._atoms,
            bonds=self._bonds,
            angles=self._angles,
            dihedrals=self._dihedrals,
            metadata=merged,
        )

    # ── repr ────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"Topology(n_atoms={self.n_atoms()}, n_bonds={self.n_bonds()}, "
            f"n_residues={self.n_residues()}, n_chains={self.n_chains()})"
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers — internal
# ──────────────────────────────────────────────────────────────────────


def _filter_and_remap(
    table: pa.Table,
    cols: tuple[str, ...],
    translation: list[int],
) -> pa.Table:
    """Filter rows where every column is in the kept set, then remap
    each integer column through the translation array."""
    if table.num_rows == 0:
        return table

    # Gather column arrays; build keep mask from "all endpoints survive".
    arrs = {c: table.column(c).to_pylist() for c in cols}
    keep: list[bool] = []
    n = table.num_rows
    for r in range(n):
        survives = all(translation[arrs[c][r]] >= 0 for c in cols)
        keep.append(survives)
    keep_arr = pa.array(keep)
    filtered = table.filter(keep_arr)
    if filtered.num_rows == 0:
        return filtered

    # Remap integer columns.
    new_columns: list[pa.Array | pa.ChunkedArray] = []
    new_fields: list[pa.Field] = []
    for f in filtered.schema:
        if f.name in cols:
            old_vals = filtered.column(f.name).to_pylist()
            new_vals = [translation[v] for v in old_vals]
            new_columns.append(pa.array(new_vals, type=f.type))
        else:
            new_columns.append(filtered.column(f.name))
        new_fields.append(f)
    return pa.table(new_columns, schema=pa.schema(new_fields))


# ──────────────────────────────────────────────────────────────────────
# Bond inference
# ──────────────────────────────────────────────────────────────────────


def infer_bonds(
    atoms: pa.Table,
    coords: torch.Tensor,
    *,
    tolerance: float = 0.4,
    exclude_hetatm_to_atom: bool = False,
) -> pa.Table:
    """Distance-based bond inference using covalent radii.

    For every atom pair ``(i, j)`` with ``i < j``, a bond is recorded
    iff ``d_ij <= r_i + r_j + tolerance``, where ``r_x`` is the
    covalent radius of element ``x`` from
    ``constellation.core.chem.ELEMENTS`` (picometers, converted to
    Ångströms). The default ``tolerance`` of 0.4 Å matches the
    OpenBabel / RDKit convention.

    Parameters
    ----------
    atoms : pa.Table
        Conforming to ``ATOM_TABLE``; must have ``element`` and
        ``is_hetatm`` columns.
    coords : torch.Tensor
        Single-frame Cartesian coordinates in Ångströms, shape
        ``(N_atoms, 3)``. Trajectory-wide bond inference is rarely
        meaningful — pick a representative frame.
    tolerance : float, default 0.4
        Slack added to the sum of covalent radii (Ångströms).
    exclude_hetatm_to_atom : bool, default False
        When True, suppress bonds between an ``ATOM`` row and a
        ``HETATM`` row. Useful for skipping spurious protein-ligand
        contacts before manual curation.

    Returns
    -------
    pa.Table
        Bond table with columns ``i`` (int32), ``j`` (int32, ``i < j``),
        ``order`` (float32, all 1.0 — distance-based inference can't
        recover bond order), ``kind`` (string, ``"covalent"``).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3); got {tuple(coords.shape)}")
    n = atoms.num_rows
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) != atoms rows ({n})")
    if n < 2:
        return _empty_table(_EMPTY_BONDS)

    elements = atoms.column("element").to_pylist()
    hetatm_flags = atoms.column("is_hetatm").to_pylist()

    # Per-atom covalent radius in Ångströms.
    radii = torch.empty(n, dtype=torch.float64)
    for k, sym in enumerate(elements):
        try:
            r_pm = ELEMENTS[sym].covalent_radius_pm
        except KeyError:
            r_pm = None
        if r_pm is None:
            # Unknown element / no data → don't infer bonds for this atom.
            radii[k] = -1.0
        else:
            radii[k] = r_pm / 100.0  # pm → Å

    # Pairwise distance matrix (float64 accumulator).
    coords64 = coords.to(torch.float64)
    diff = coords64.unsqueeze(0) - coords64.unsqueeze(1)  # (N, N, 3)
    dist = torch.linalg.norm(diff, dim=2)  # (N, N)

    # Pairwise threshold = r_i + r_j + tolerance; -inf if either radius < 0.
    thresh = radii.unsqueeze(0) + radii.unsqueeze(1) + tolerance
    bad = (radii.unsqueeze(0) < 0) | (radii.unsqueeze(1) < 0)
    thresh = torch.where(bad, torch.tensor(-1.0, dtype=torch.float64), thresh)

    # Upper-triangular mask (i < j) and threshold cut.
    iu, ju = torch.triu_indices(n, n, offset=1)
    pair_dist = dist[iu, ju]
    pair_thresh = thresh[iu, ju]
    mask = (pair_dist <= pair_thresh) & (pair_thresh > 0)

    if exclude_hetatm_to_atom:
        het_arr = torch.tensor(hetatm_flags, dtype=torch.bool)
        # Suppress pairs that mix an ATOM (false) with a HETATM (true).
        het_i = het_arr[iu]
        het_j = het_arr[ju]
        mixed = het_i ^ het_j
        mask = mask & ~mixed

    keep = mask.nonzero(as_tuple=False).squeeze(-1)
    if keep.numel() == 0:
        return _empty_table(_EMPTY_BONDS)

    i_vals = iu[keep].to(torch.int32).cpu().numpy()
    j_vals = ju[keep].to(torch.int32).cpu().numpy()
    n_bonds = int(keep.numel())
    return pa.table(
        {
            "i": pa.array(i_vals, type=pa.int32()),
            "j": pa.array(j_vals, type=pa.int32()),
            "order": pa.array([1.0] * n_bonds, type=pa.float32()),
            "kind": pa.array(["covalent"] * n_bonds, type=pa.string()),
        },
        schema=_EMPTY_BONDS,
    )


# ──────────────────────────────────────────────────────────────────────
# Re-exports — empty-schema factories useful for tests / readers
# ──────────────────────────────────────────────────────────────────────


def empty_bonds_table() -> pa.Table:
    return _empty_table(_EMPTY_BONDS)


__all__ = [
    "Topology",
    "infer_bonds",
    "empty_bonds_table",
]
