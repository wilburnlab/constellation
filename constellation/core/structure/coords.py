"""Atom-table schema, coordinate-frame metadata, tensor bridges.

This module defines the universal Arrow column layout for an atom
table — ``STRUCTURE_TABLE`` — and the ``CoordinateFrame`` dataclass
holding coordinate-system metadata (units, periodic boundaries, origin)
that rides alongside coord tensors.

The schema deliberately omits ``x`` / ``y`` / ``z`` columns. Coordinates
live as a separate ``torch.float32`` tensor of shape
``(N_frames, N_atoms, 3)`` on ``Ensemble`` — see
``constellation.core.structure.ensemble``. Identity columns stay in
Arrow (selection / filter / groupby), numerical math runs on torch
tensors (Principle 3).

The schema is registered with the ``core.io`` schema registry under
the name ``"StructureTable"`` at import time, so domain-module code
can ``get_schema("StructureTable")`` symmetrically with ``Trace1D`` /
``PeakTable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pyarrow as pa
import torch

from constellation.core.io.schemas import register_schema, registered_schemas


# ──────────────────────────────────────────────────────────────────────
# StructureTable schema
# ──────────────────────────────────────────────────────────────────────

# Per-atom identity / metadata columns. PDB-derived but PyArrow-typed.
# Coordinates are NOT here — they live as a separate torch tensor on
# ``Ensemble``. Reasoning:
#   * ensembles (NMR / cryo-EM multi-model / MD frames) are first-class;
#     a single static structure is just the (1, N, 3) degenerate case.
#   * geometry math (translate, rotate, principal axes) runs on torch
#     tensors directly without an Arrow → numpy round-trip every call.
#   * selection still drives an index list; geometry consumes the tensor.
STRUCTURE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("serial", pa.int32(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("alt_loc", pa.string(), nullable=True),
        pa.field("res_name", pa.string(), nullable=False),
        pa.field("chain_id", pa.string(), nullable=False),
        pa.field("res_seq", pa.int32(), nullable=False),
        pa.field("ins_code", pa.string(), nullable=True),
        pa.field("segment_id", pa.string(), nullable=True),
        pa.field("element", pa.string(), nullable=False),
        pa.field("occupancy", pa.float32(), nullable=True),
        pa.field("b_factor", pa.float32(), nullable=True),
        pa.field("formal_charge", pa.int8(), nullable=True),
        pa.field("is_hetatm", pa.bool_(), nullable=False),
        pa.field("model_id", pa.int32(), nullable=True),
    ],
    metadata={b"schema_name": b"StructureTable"},
)


# Self-register with the core.io schema registry. Idempotent against
# duplicate imports — ``register_schema`` raises on duplicates, so we
# guard with the snapshot.
if "StructureTable" not in registered_schemas():
    register_schema("StructureTable", STRUCTURE_TABLE)


# ──────────────────────────────────────────────────────────────────────
# CoordinateFrame
# ──────────────────────────────────────────────────────────────────────

UnitName = Literal["angstrom", "nanometer", "bohr"]

# Conversion factors: multiply native coords by this to get Ångströms.
_TO_ANGSTROM: dict[str, float] = {
    "angstrom": 1.0,
    "nanometer": 10.0,
    "bohr": 0.529177210903,  # CODATA 2018
}


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Coordinate-system metadata that rides alongside a coord tensor.

    A single frame can be non-periodic (``cell=None``) or periodic
    (``cell`` a ``(3, 3)`` torch tensor of lattice vectors as rows,
    ``pbc`` flagging which axes wrap). The frame does not own the coord
    tensor itself — ``Ensemble`` does that. Frames are immutable.
    """

    units: UnitName = "angstrom"
    cell: torch.Tensor | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.units not in _TO_ANGSTROM:
            raise ValueError(
                f"unknown units {self.units!r}; supported: {sorted(_TO_ANGSTROM)}"
            )
        if self.cell is not None:
            if self.cell.shape != (3, 3):
                raise ValueError(
                    f"cell must have shape (3, 3); got {tuple(self.cell.shape)}"
                )
        if len(self.pbc) != 3:
            raise ValueError(f"pbc must have length 3; got {self.pbc!r}")
        if len(self.origin) != 3:
            raise ValueError(f"origin must have length 3; got {self.origin!r}")

    def to_angstrom(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert coords from this frame's units to Ångströms.

        Returns a new tensor; the input is unmodified. Identity when
        ``self.units == "angstrom"``.
        """
        factor = _TO_ANGSTROM[self.units]
        if factor == 1.0:
            return coords
        return coords * factor

    def is_periodic(self) -> bool:
        return any(self.pbc) and self.cell is not None

    def cell_volume(self) -> float | None:
        """Triple-product volume of the unit cell in the frame's units³.

        Returns ``None`` for non-periodic frames or when ``cell`` is
        unset. Negative determinants are returned with their sign — a
        left-handed cell is suspicious but representable.
        """
        if self.cell is None:
            return None
        det = float(torch.det(self.cell.to(torch.float64)).item())
        return det


# ──────────────────────────────────────────────────────────────────────
# Tensor bridges
# ──────────────────────────────────────────────────────────────────────


_NUMERIC_FIELDS: tuple[tuple[str, torch.dtype], ...] = (
    ("serial", torch.int32),
    ("res_seq", torch.int32),
    ("occupancy", torch.float32),
    ("b_factor", torch.float32),
    ("formal_charge", torch.int8),
    ("model_id", torch.int32),
)


def structure_table_to_tensors(table: pa.Table) -> dict[str, torch.Tensor]:
    """Pull the numerical columns of a ``STRUCTURE_TABLE``-conforming
    Arrow table into a flat dict of ``torch.Tensor``s.

    Only columns that are actually present in ``table`` are returned —
    missing nullable fields are skipped silently (caller can default
    them). Strings, booleans, and the residue-identity columns stay in
    Arrow on the source table; this helper exists for the geometry /
    selection layers, not for round-tripping every column.
    """
    out: dict[str, torch.Tensor] = {}
    for name, dtype in _NUMERIC_FIELDS:
        if name not in table.column_names:
            continue
        # ``zero_copy_only=False`` because nullable columns may not be
        # contiguous; we accept the one-allocation cost. ``.copy()``
        # to get a writable ndarray (PyArrow's view is read-only).
        arr = table.column(name).to_numpy(zero_copy_only=False).copy()
        out[name] = torch.from_numpy(arr).to(dtype)
    if "is_hetatm" in table.column_names:
        arr = table.column("is_hetatm").to_numpy(zero_copy_only=False).copy()
        out["is_hetatm"] = torch.from_numpy(arr.astype("bool"))
    return out


__all__ = [
    "STRUCTURE_TABLE",
    "CoordinateFrame",
    "structure_table_to_tensors",
]
