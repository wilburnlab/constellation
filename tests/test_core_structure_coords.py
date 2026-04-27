"""Unit tests for ``constellation.core.structure.coords``."""

from __future__ import annotations

import pyarrow as pa
import pytest
import torch

from constellation.core.io.schemas import (
    cast_to_schema,
    get_schema,
    unpack_metadata,
)
from constellation.core.structure import (
    STRUCTURE_TABLE,
    CoordinateFrame,
    structure_table_to_tensors,
)


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


def test_schema_required_columns():
    required = {
        "serial",
        "name",
        "res_name",
        "chain_id",
        "res_seq",
        "element",
        "is_hetatm",
    }
    names = set(f.name for f in STRUCTURE_TABLE)
    assert required <= names


def test_schema_dtypes():
    assert STRUCTURE_TABLE.field("serial").type == pa.int32()
    assert STRUCTURE_TABLE.field("res_seq").type == pa.int32()
    assert STRUCTURE_TABLE.field("occupancy").type == pa.float32()
    assert STRUCTURE_TABLE.field("b_factor").type == pa.float32()
    assert STRUCTURE_TABLE.field("formal_charge").type == pa.int8()
    assert STRUCTURE_TABLE.field("is_hetatm").type == pa.bool_()


def test_schema_nullability():
    # Required identity columns are non-null.
    for c in (
        "serial",
        "name",
        "res_name",
        "chain_id",
        "res_seq",
        "element",
        "is_hetatm",
    ):
        assert not STRUCTURE_TABLE.field(c).nullable, c
    # Optional columns are nullable.
    for c in (
        "alt_loc",
        "ins_code",
        "segment_id",
        "occupancy",
        "b_factor",
        "formal_charge",
        "model_id",
    ):
        assert STRUCTURE_TABLE.field(c).nullable, c


def test_schema_metadata_stamped():
    meta = unpack_metadata(STRUCTURE_TABLE.metadata)
    assert meta["schema_name"] == "StructureTable"


def test_schema_self_registers_with_core_io_registry():
    assert get_schema("StructureTable") is STRUCTURE_TABLE


# ──────────────────────────────────────────────────────────────────────
# CoordinateFrame
# ──────────────────────────────────────────────────────────────────────


def test_default_frame_is_angstrom_non_periodic():
    frame = CoordinateFrame()
    assert frame.units == "angstrom"
    assert frame.cell is None
    assert frame.pbc == (False, False, False)
    assert not frame.is_periodic()


def test_to_angstrom_identity_for_angstrom():
    frame = CoordinateFrame()
    coords = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(frame.to_angstrom(coords), coords)


def test_to_angstrom_converts_nm():
    frame = CoordinateFrame(units="nanometer")
    coords = torch.tensor([[1.0, 2.0, 3.0]])
    out = frame.to_angstrom(coords)
    assert torch.allclose(out, coords * 10.0)


def test_to_angstrom_converts_bohr():
    frame = CoordinateFrame(units="bohr")
    coords = torch.tensor([[1.0]])
    out = frame.to_angstrom(coords)
    # Bohr → Å factor is roughly 0.529.
    assert 0.5 < out.item() < 0.55


def test_unknown_units_rejected():
    with pytest.raises(ValueError, match="unknown units"):
        CoordinateFrame(units="furlong")  # type: ignore[arg-type]


def test_cell_volume_unit_cube():
    cell = torch.eye(3, dtype=torch.float64) * 5.0
    frame = CoordinateFrame(cell=cell, pbc=(True, True, True))
    assert frame.is_periodic()
    assert abs(frame.cell_volume() - 125.0) < 1e-9


def test_cell_volume_none_when_no_cell():
    assert CoordinateFrame().cell_volume() is None


def test_cell_must_be_3x3():
    with pytest.raises(ValueError, match="shape"):
        CoordinateFrame(cell=torch.eye(2))


# ──────────────────────────────────────────────────────────────────────
# Tensor bridge
# ──────────────────────────────────────────────────────────────────────


def _mini_table():
    return pa.table(
        {
            "serial": pa.array([1, 2, 3], type=pa.int32()),
            "name": pa.array(["N", "CA", "C"], type=pa.string()),
            "alt_loc": pa.array([None, None, None], type=pa.string()),
            "res_name": pa.array(["ALA", "ALA", "ALA"], type=pa.string()),
            "chain_id": pa.array(["A", "A", "A"], type=pa.string()),
            "res_seq": pa.array([1, 1, 1], type=pa.int32()),
            "ins_code": pa.array([None, None, None], type=pa.string()),
            "segment_id": pa.array([None, None, None], type=pa.string()),
            "element": pa.array(["N", "C", "C"], type=pa.string()),
            "occupancy": pa.array([1.0, 1.0, 1.0], type=pa.float32()),
            "b_factor": pa.array([10.0, 12.5, 15.0], type=pa.float32()),
            "formal_charge": pa.array([0, 0, 0], type=pa.int8()),
            "is_hetatm": pa.array([False, False, False], type=pa.bool_()),
            "model_id": pa.array([1, 1, 1], type=pa.int32()),
        },
        schema=STRUCTURE_TABLE,
    )


def test_structure_table_to_tensors_keys():
    t = _mini_table()
    out = structure_table_to_tensors(t)
    assert "serial" in out and "res_seq" in out
    assert "occupancy" in out and "b_factor" in out
    assert "is_hetatm" in out
    assert out["serial"].dtype == torch.int32
    assert out["b_factor"].dtype == torch.float32
    assert out["is_hetatm"].dtype == torch.bool


def test_structure_table_to_tensors_values():
    t = _mini_table()
    out = structure_table_to_tensors(t)
    assert out["serial"].tolist() == [1, 2, 3]
    assert torch.allclose(out["b_factor"], torch.tensor([10.0, 12.5, 15.0]))


def test_cast_to_schema_fills_missing_optional_columns():
    """A foreign atom table missing optional columns is castable."""
    minimal = pa.table(
        {
            "serial": pa.array([1, 2], type=pa.int32()),
            "name": pa.array(["N", "CA"], type=pa.string()),
            "res_name": pa.array(["ALA", "ALA"], type=pa.string()),
            "chain_id": pa.array(["A", "A"], type=pa.string()),
            "res_seq": pa.array([1, 1], type=pa.int32()),
            "element": pa.array(["N", "C"], type=pa.string()),
            "is_hetatm": pa.array([False, False], type=pa.bool_()),
        }
    )
    casted = cast_to_schema(minimal, STRUCTURE_TABLE)
    assert casted.schema == STRUCTURE_TABLE
    # Optional columns are filled with nulls.
    assert casted.column("occupancy").null_count == 2
    assert casted.column("b_factor").null_count == 2
