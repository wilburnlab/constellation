"""Unit tests for ``constellation.core.structure.ensemble``."""

from __future__ import annotations

import pyarrow as pa
import pytest
import torch

from constellation.core.io.schemas import cast_to_schema
from constellation.core.structure import (
    FRAME_METADATA,
    ATOM_TABLE,
    CoordinateFrame,
    Ensemble,
    Topology,
    frame_to_table,
    select_chain,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _two_chain_atoms() -> pa.Table:
    rows = []
    serial = 1
    for chain in ("A", "B"):
        for nm, el in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
            rows.append(
                {
                    "serial": serial,
                    "name": nm,
                    "alt_loc": None,
                    "res_name": "ALA",
                    "chain_id": chain,
                    "res_seq": 1,
                    "ins_code": None,
                    "segment_id": None,
                    "element": el,
                    "occupancy": 1.0,
                    "b_factor": 10.0,
                    "formal_charge": 0,
                    "is_hetatm": False,
                    "model_id": 1,
                }
            )
            serial += 1
    cols = {
        f.name: pa.array([r[f.name] for r in rows], type=f.type)
        for f in ATOM_TABLE
    }
    return pa.table(cols, schema=ATOM_TABLE)


def _topology() -> Topology:
    return Topology(_two_chain_atoms())


def _coords(n_frames: int) -> torch.Tensor:
    """Synthetic trajectory: each frame is a small rigid translation."""
    base = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    return torch.stack([base + i * 0.1 for i in range(n_frames)])


# ──────────────────────────────────────────────────────────────────────
# Construction + shape normalization
# ──────────────────────────────────────────────────────────────────────


def test_single_accepts_2d_coords():
    e = Ensemble.single(_topology(), _coords(1)[0])
    assert e.n_frames() == 1 and e.n_atoms() == 8
    assert e.coords.shape == (1, 8, 3)


def test_single_accepts_3d_coords():
    e = Ensemble.single(_topology(), _coords(1))
    assert e.n_frames() == 1 and e.n_atoms() == 8


def test_constructor_accepts_multi_frame():
    e = Ensemble(_topology(), _coords(5))
    assert e.n_frames() == 5
    assert e.coords.dtype == torch.float32


def test_constructor_rejects_atom_mismatch():
    bad = torch.zeros(7, 3)  # topology has 8 atoms
    with pytest.raises(ValueError, match="topology n_atoms"):
        Ensemble(_topology(), bad)


def test_constructor_rejects_bad_ndim():
    with pytest.raises(ValueError):
        Ensemble(_topology(), torch.zeros(3))


def test_constructor_upcasts_to_float32():
    coords = torch.zeros(8, 3, dtype=torch.float64)
    e = Ensemble(_topology(), coords)
    assert e.coords.dtype == torch.float32


def test_frame_metadata_default_one_row_per_frame():
    e = Ensemble(_topology(), _coords(4))
    fm = e.frame_metadata
    assert fm.num_rows == 4
    assert fm.column("frame_id").to_pylist() == [0, 1, 2, 3]
    assert fm.column("time_ps").null_count == 4


def test_frame_metadata_length_must_match_n_frames():
    fields = list(FRAME_METADATA)
    arrays = [pa.array([0, 1], type=pa.int32())]
    arrays += [pa.nulls(2, type=f.type) for f in fields[1:]]
    bad = pa.table(arrays, schema=FRAME_METADATA)
    with pytest.raises(ValueError, match="frame_metadata"):
        Ensemble(_topology(), _coords(3), frame_metadata=bad)


# ──────────────────────────────────────────────────────────────────────
# Iteration / indexing
# ──────────────────────────────────────────────────────────────────────


def test_len_matches_n_frames():
    e = Ensemble(_topology(), _coords(7))
    assert len(e) == 7


def test_iter_yields_frame_idx_pairs():
    e = Ensemble(_topology(), _coords(3))
    seen = list(e)
    assert [s[0] for s in seen] == [0, 1, 2]
    assert all(s[1].shape == (8, 3) for s in seen)


def test_get_frame_returns_2d_view():
    e = Ensemble(_topology(), _coords(3))
    assert e.get_frame(1).shape == (8, 3)


def test_get_frame_out_of_range_raises():
    e = Ensemble(_topology(), _coords(2))
    with pytest.raises(IndexError):
        e.get_frame(5)


# ──────────────────────────────────────────────────────────────────────
# Selection / slicing
# ──────────────────────────────────────────────────────────────────────


def test_select_reduces_topology_and_coords():
    e = Ensemble(_topology(), _coords(3))
    sub = e.select(select_chain("A"))
    assert sub.n_atoms() == 4
    assert sub.coords.shape == (3, 4, 3)
    # First atom of selection corresponds to the original chain-A row 0.
    assert torch.allclose(sub.coords[0, 0], e.coords[0, 0])


def test_select_no_match_raises():
    e = Ensemble(_topology(), _coords(2))
    with pytest.raises(ValueError, match="zero atoms"):
        e.select(select_chain("Z"))


def test_slice_frames_with_slice():
    e = Ensemble(_topology(), _coords(5))
    sub = e.slice_frames(slice(1, 4))
    assert sub.n_frames() == 3
    assert sub.frame_metadata.num_rows == 3
    assert sub.frame_metadata.column("frame_id").to_pylist() == [1, 2, 3]


def test_slice_frames_with_index_list():
    e = Ensemble(_topology(), _coords(5))
    sub = e.slice_frames([0, 2, 4])
    assert sub.n_frames() == 3
    assert sub.frame_metadata.column("frame_id").to_pylist() == [0, 2, 4]


def test_slice_frames_out_of_range_raises():
    e = Ensemble(_topology(), _coords(2))
    with pytest.raises(IndexError):
        e.slice_frames([0, 5])


# ──────────────────────────────────────────────────────────────────────
# frame_to_table
# ──────────────────────────────────────────────────────────────────────


def test_frame_to_table_appends_xyz():
    e = Ensemble(_topology(), _coords(2))
    t = frame_to_table(e, 0)
    for c in ("x", "y", "z"):
        assert c in t.column_names
    assert t.num_rows == 8
    # Numeric values match the underlying tensor.
    expected = e.coords[0]
    assert t.column("x").to_pylist() == expected[:, 0].tolist()


def test_frame_to_table_round_trips_through_cast():
    """Coord-augmented projection still casts cleanly to ATOM_TABLE
    (xyz columns get dropped on the way back)."""
    e = Ensemble(_topology(), _coords(1))
    t = frame_to_table(e, 0)
    casted = cast_to_schema(t, ATOM_TABLE)
    assert casted.schema == ATOM_TABLE
    assert casted.num_rows == 8


# ──────────────────────────────────────────────────────────────────────
# CoordinateFrame attached
# ──────────────────────────────────────────────────────────────────────


def test_frame_defaults_to_angstrom():
    e = Ensemble(_topology(), _coords(1))
    assert e.frame.units == "angstrom"


def test_custom_frame_preserved():
    custom = CoordinateFrame(units="nanometer")
    e = Ensemble(_topology(), _coords(1), frame=custom)
    assert e.frame.units == "nanometer"


# ──────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────


def test_with_metadata_round_trip():
    e = Ensemble(_topology(), _coords(1), metadata={"source": "synth"})
    e2 = e.with_metadata({"version": 1})
    assert e2.metadata == {"source": "synth", "version": 1}
    assert e.metadata == {"source": "synth"}
