"""Unit tests for ``constellation.core.structure.topology``."""

from __future__ import annotations

import pyarrow as pa
import pytest
import torch

from constellation.core.structure import (
    ATOM_TABLE,
    Topology,
    infer_bonds,
    select_chain,
    select_protein,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures: water, benzene, dipeptide
# ──────────────────────────────────────────────────────────────────────


def _atoms_table(rows: list[dict]) -> pa.Table:
    """Build a ATOM_TABLE-conformant atoms table from row dicts."""
    cols: dict = {f.name: [] for f in ATOM_TABLE}
    for r in rows:
        for f in ATOM_TABLE:
            cols[f.name].append(r.get(f.name, None))
    arrays = {f.name: pa.array(cols[f.name], type=f.type) for f in ATOM_TABLE}
    return pa.table(arrays, schema=ATOM_TABLE)


def _water_atoms() -> pa.Table:
    return _atoms_table(
        [
            {
                "serial": 1,
                "name": "O",
                "res_name": "HOH",
                "chain_id": "W",
                "res_seq": 1,
                "element": "O",
                "is_hetatm": True,
                "model_id": 1,
            },
            {
                "serial": 2,
                "name": "H1",
                "res_name": "HOH",
                "chain_id": "W",
                "res_seq": 1,
                "element": "H",
                "is_hetatm": True,
                "model_id": 1,
            },
            {
                "serial": 3,
                "name": "H2",
                "res_name": "HOH",
                "chain_id": "W",
                "res_seq": 1,
                "element": "H",
                "is_hetatm": True,
                "model_id": 1,
            },
        ]
    )


def _water_coords() -> torch.Tensor:
    # Approx geometry: O at origin, H atoms at ~0.96 Å with ~104.5° H-O-H.
    return torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.7570, 0.5860, 0.0000],
            [-0.7570, 0.5860, 0.0000],
        ],
        dtype=torch.float32,
    )


def _benzene_atoms() -> pa.Table:
    rows = []
    for i in range(6):
        rows.append(
            {
                "serial": i + 1,
                "name": f"C{i + 1}",
                "res_name": "BEN",
                "chain_id": "L",
                "res_seq": 1,
                "element": "C",
                "is_hetatm": True,
                "model_id": 1,
            }
        )
    for i in range(6):
        rows.append(
            {
                "serial": i + 7,
                "name": f"H{i + 1}",
                "res_name": "BEN",
                "chain_id": "L",
                "res_seq": 1,
                "element": "H",
                "is_hetatm": True,
                "model_id": 1,
            }
        )
    return _atoms_table(rows)


def _benzene_coords() -> torch.Tensor:
    """Planar regular hexagon, C-C = 1.40 Å, C-H = 1.09 Å."""
    import math

    cc = 1.40
    ch = 1.09
    coords = []
    for i in range(6):
        a = i * math.pi / 3.0
        coords.append([cc * math.cos(a), cc * math.sin(a), 0.0])
    for i in range(6):
        a = i * math.pi / 3.0
        r = cc + ch
        coords.append([r * math.cos(a), r * math.sin(a), 0.0])
    return torch.tensor(coords, dtype=torch.float32)


def _dipeptide_atoms() -> pa.Table:
    """Two-chain ALA-GLY dipeptide, four atoms each (N, CA, C, O)."""
    rows = []
    serial = 1
    for chain in ("A", "B"):
        for resi, rn in enumerate(("ALA", "GLY"), start=1):
            for nm in ("N", "CA", "C", "O"):
                el = nm[0]  # N -> N, CA -> C, C -> C, O -> O
                rows.append(
                    {
                        "serial": serial,
                        "name": nm,
                        "res_name": rn,
                        "chain_id": chain,
                        "res_seq": resi,
                        "element": el,
                        "is_hetatm": False,
                        "model_id": 1,
                    }
                )
                serial += 1
    return _atoms_table(rows)


# ──────────────────────────────────────────────────────────────────────
# Topology constructor + accessors
# ──────────────────────────────────────────────────────────────────────


def test_topology_atoms_only_has_empty_connectivity():
    t = Topology(_water_atoms())
    assert t.n_atoms() == 3
    assert t.n_bonds() == 0
    assert t.bonds.num_rows == 0
    assert t.angles.num_rows == 0
    assert t.dihedrals.num_rows == 0


def test_topology_rejects_out_of_range_bond_indices():
    bonds = pa.table(
        {"i": pa.array([0], type=pa.int32()), "j": pa.array([99], type=pa.int32())}
    )
    with pytest.raises(ValueError, match="out of range"):
        Topology(_water_atoms(), bonds=bonds)


def test_topology_rejects_missing_bond_columns():
    bonds = pa.table({"foo": pa.array([0], type=pa.int32())})
    with pytest.raises(ValueError, match="missing required column"):
        Topology(_water_atoms(), bonds=bonds)


# ──────────────────────────────────────────────────────────────────────
# infer_bonds
# ──────────────────────────────────────────────────────────────────────


def test_infer_bonds_water_two_bonds():
    bonds = infer_bonds(_water_atoms(), _water_coords())
    # Two O-H bonds, no H-H.
    assert bonds.num_rows == 2
    pairs = sorted(zip(bonds.column("i").to_pylist(), bonds.column("j").to_pylist()))
    assert pairs == [(0, 1), (0, 2)]


def test_infer_bonds_benzene_full_connectivity():
    bonds = infer_bonds(_benzene_atoms(), _benzene_coords())
    # 6 ring C-C bonds + 6 C-H bonds = 12 total.
    assert bonds.num_rows == 12


def test_infer_bonds_orders_i_lt_j():
    bonds = infer_bonds(_water_atoms(), _water_coords())
    for i, j in zip(bonds.column("i").to_pylist(), bonds.column("j").to_pylist()):
        assert i < j


def test_infer_bonds_tolerance_scaling():
    # With strict tolerance (-0.5), no bonds satisfy the threshold
    # because the H-O distance ~ 0.96 Å is already at the radii sum.
    bonds_loose = infer_bonds(_water_atoms(), _water_coords(), tolerance=0.5)
    bonds_strict = infer_bonds(_water_atoms(), _water_coords(), tolerance=-0.6)
    assert bonds_loose.num_rows >= bonds_strict.num_rows


def test_infer_bonds_empty_for_single_atom():
    atoms = _atoms_table(
        [
            {
                "serial": 1,
                "name": "O",
                "res_name": "HOH",
                "chain_id": "W",
                "res_seq": 1,
                "element": "O",
                "is_hetatm": True,
                "model_id": 1,
            }
        ]
    )
    coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    assert infer_bonds(atoms, coords).num_rows == 0


def test_infer_bonds_emits_canonical_schema():
    bonds = infer_bonds(_water_atoms(), _water_coords())
    names = bonds.column_names
    assert names == ["i", "j", "order", "kind"]
    assert bonds.column("kind").to_pylist() == ["covalent", "covalent"]


def test_infer_bonds_mismatched_coords_raises():
    with pytest.raises(ValueError):
        infer_bonds(_water_atoms(), torch.zeros(2, 3))


def test_infer_bonds_rejects_3d_coords():
    with pytest.raises(ValueError):
        infer_bonds(_water_atoms(), torch.zeros(1, 3, 3))


# ──────────────────────────────────────────────────────────────────────
# Residue grouping / counts
# ──────────────────────────────────────────────────────────────────────


def test_residue_groups_one_row_per_residue():
    t = Topology(_dipeptide_atoms())
    groups = t.residue_groups()
    assert groups.num_rows == 4  # 2 chains × 2 residues
    assert "first_index" in groups.column_names
    assert "n_atoms" in groups.column_names
    assert all(n == 4 for n in groups.column("n_atoms").to_pylist())


def test_n_chains_n_residues():
    t = Topology(_dipeptide_atoms())
    assert t.n_chains() == 2
    assert t.n_residues() == 4


def test_chain_ids_first_encounter_order():
    t = Topology(_dipeptide_atoms())
    assert t.chain_ids() == ["A", "B"]


# ──────────────────────────────────────────────────────────────────────
# Selection / subsetting + index integrity
# ──────────────────────────────────────────────────────────────────────


def test_select_indices_chain_filter():
    t = Topology(_dipeptide_atoms())
    idx = t.select_indices(select_chain("A"))
    assert idx.tolist() == list(range(0, 8))


def test_select_indices_returns_int64():
    t = Topology(_dipeptide_atoms())
    idx = t.select_indices(select_protein())
    assert idx.dtype == torch.int64


def test_subset_remaps_bond_indices():
    """Index integrity contract: bonds.i / bonds.j survive subsetting
    only when both endpoints survive, and the values get remapped to
    the new row positions in the kept atoms table."""
    atoms = _water_atoms()
    bonds = pa.table(
        {
            "i": pa.array([0, 0], type=pa.int32()),
            "j": pa.array([1, 2], type=pa.int32()),
            "order": pa.array([1.0, 1.0], type=pa.float32()),
            "kind": pa.array(["covalent", "covalent"], type=pa.string()),
        }
    )
    t = Topology(atoms, bonds=bonds)
    # Drop H1 (index 1). Surviving atoms: O (was 0) and H2 (was 2).
    sub = t.subset([0, 2])
    assert sub.n_atoms() == 2
    assert sub.n_bonds() == 1
    new_bonds = sub.bonds
    assert new_bonds.column("i").to_pylist() == [0]
    # H2 was at original index 2; now at new index 1.
    assert new_bonds.column("j").to_pylist() == [1]


def test_subset_preserves_order_features_on_bonds():
    atoms = _water_atoms()
    bonds = pa.table(
        {
            "i": pa.array([0], type=pa.int32()),
            "j": pa.array([1], type=pa.int32()),
            "order": pa.array([2.5], type=pa.float32()),
            "kind": pa.array(["double"], type=pa.string()),
        }
    )
    t = Topology(atoms, bonds=bonds)
    sub = t.subset([0, 1, 2])
    assert sub.bonds.column("order").to_pylist() == [2.5]
    assert sub.bonds.column("kind").to_pylist() == ["double"]


def test_subset_out_of_range_raises():
    t = Topology(_water_atoms())
    with pytest.raises(IndexError):
        t.subset([0, 99])


# ──────────────────────────────────────────────────────────────────────
# bond_network
# ──────────────────────────────────────────────────────────────────────


def test_bond_network_n_edges_matches_bonds():
    atoms = _water_atoms()
    bonds = infer_bonds(atoms, _water_coords())
    t = Topology(atoms, bonds=bonds)
    net = t.bond_network
    assert net.n_edges() == bonds.num_rows
    assert net.n_nodes() == atoms.num_rows


def test_bond_network_neighbors_match():
    atoms = _water_atoms()
    bonds = infer_bonds(atoms, _water_coords())
    t = Topology(atoms, bonds=bonds)
    net = t.bond_network
    # Atom 0 (O) is neighbors with both H atoms.
    assert sorted(net.neighbors(0)) == [1, 2]


# ──────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────


def test_with_metadata_round_trip():
    t = Topology(_water_atoms(), metadata={"source": "test"})
    t2 = t.with_metadata({"version": 2})
    assert t2.metadata == {"source": "test", "version": 2}
    # Original unchanged.
    assert t.metadata == {"source": "test"}
