"""Macromolecular-structure data structures and tensor-native geometry.

Replaces the half-transitioned pandas/list-of-class atom tables in
Contour with PyArrow ``StructureTable`` Arrow tables for atom identity
plus separate ``torch.float32`` coord tensors for numerical math
(Principle 3). Ensembles (NMR / cryo-EM multi-model / MD frames) are
first-class: a single static structure is the ``n_frames == 1``
degenerate case.

Modules:
    coords       - ``STRUCTURE_TABLE`` Arrow schema, ``CoordinateFrame``
                   (units, periodic boundaries), tensor bridge.
    topology     - ``Topology`` (atoms + bonds/angles/dihedrals,
                   ``Network``-backed bond view), ``infer_bonds``
                   (covalent-radii distance inference), residue grouping.
    selection    - Predicate helpers returning ``pa.compute.Expression``;
                   compose with ``&`` / ``|`` / ``~``.
    ensemble     - ``Ensemble`` container, ``FRAME_METADATA`` schema,
                   ``frame_to_table`` (coord-augmented Arrow projection).
    geometry     - Tensor-native ``translate`` / ``rotate`` / ``centroid``
                   / ``mass_centroid`` / ``radius_of_gyration`` /
                   ``principal_axes`` over ``(N, 3)`` and ``(F, N, 3)``.

Explicit non-goals this session: Kabsch / RMSD / superposition /
alignment — deferred until ``core.stats`` + ``core.optim`` ship.
"""

from constellation.core.structure.coords import (
    STRUCTURE_TABLE,
    CoordinateFrame,
    structure_table_to_tensors,
)
from constellation.core.structure.ensemble import (
    FRAME_METADATA,
    Ensemble,
    frame_to_table,
)
from constellation.core.structure.geometry import (
    centroid,
    mass_centroid,
    principal_axes,
    radius_of_gyration,
    rotate,
    translate,
)
from constellation.core.structure.selection import (
    select_atom_names,
    select_backbone,
    select_chain,
    select_element,
    select_hetatm,
    select_protein,
    select_resname,
    select_residues,
    select_sidechain,
    select_water,
)
from constellation.core.structure.topology import (
    Topology,
    empty_bonds_table,
    infer_bonds,
)

__all__ = [
    # coords
    "STRUCTURE_TABLE",
    "CoordinateFrame",
    "structure_table_to_tensors",
    # topology
    "Topology",
    "infer_bonds",
    "empty_bonds_table",
    # selection
    "select_chain",
    "select_residues",
    "select_resname",
    "select_atom_names",
    "select_backbone",
    "select_sidechain",
    "select_protein",
    "select_water",
    "select_hetatm",
    "select_element",
    # ensemble
    "Ensemble",
    "FRAME_METADATA",
    "frame_to_table",
    # geometry
    "translate",
    "rotate",
    "centroid",
    "mass_centroid",
    "radius_of_gyration",
    "principal_axes",
]
