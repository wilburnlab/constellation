"""Unit tests for constellation.core.io.schemas."""

from __future__ import annotations

import pyarrow as pa
import pytest
import torch

from constellation.core.io.schemas import (
    PEAK_TABLE,
    TRACE_1D,
    cast_to_schema,
    pack_metadata,
    ragged_to_tensors,
    ragged_trace_1d,
    register_schema,
    registered_schemas,
    spectra_matrix_2d,
    spectra_to_tensor,
    tensor_to_spectra,
    trace_to_tensor,
    unpack_metadata,
    with_metadata,
)


# ----------------------------------------------------------------------
# Static schemas
# ----------------------------------------------------------------------


def test_trace_1d_schema_columns():
    assert TRACE_1D.field("time_s").type == pa.float64()
    assert TRACE_1D.field("intensity").type == pa.float64()
    assert TRACE_1D.field("channel_id").type == pa.string()
    # channel_id nullable so single-channel tables can omit it.
    assert TRACE_1D.field("channel_id").nullable
    # Schema name stamped in metadata.
    assert unpack_metadata(TRACE_1D.metadata)["schema_name"] == "Trace1D"


def test_peak_table_schema():
    names = [f.name for f in PEAK_TABLE]
    assert "peak_id" in names and "retention_time_s" in names
    assert PEAK_TABLE.field("peak_id").type == pa.int32()
    assert PEAK_TABLE.field("calibrated_value").nullable
    assert PEAK_TABLE.field("calibrated_unit").type == pa.string()


def test_spectra_matrix_2d_factory():
    s = spectra_matrix_2d(226)
    assert s.field("time_ms").type == pa.int64()
    inner = s.field("spectrum").type
    assert pa.types.is_fixed_size_list(inner)
    assert inner.list_size == 226
    assert inner.value_type == pa.float64()


def test_spectra_matrix_2d_rejects_zero():
    with pytest.raises(ValueError):
        spectra_matrix_2d(0)


def test_ragged_trace_1d_minimal():
    s = ragged_trace_1d(pa.int16(), value_name="signal")
    names = [f.name for f in s]
    assert names == ["signal"]
    assert s.field("signal").type == pa.list_(pa.int16())
    assert unpack_metadata(s.metadata)["schema_name"] == "RaggedTrace1D"


def test_ragged_trace_1d_with_x_and_extras():
    s = ragged_trace_1d(
        pa.float64(),
        value_name="intensity",
        x_name="mz",
        extra_columns=(
            ("scan_id", pa.int64()),
            ("rt_s", pa.float64()),
        ),
    )
    names = [f.name for f in s]
    # Per-row metadata first, then x, then bulk-array column.
    assert names == ["scan_id", "rt_s", "mz", "intensity"]
    assert s.field("mz").type == pa.list_(pa.float64())
    assert s.field("intensity").type == pa.list_(pa.float64())


def test_ragged_trace_1d_pod5_shape():
    """The shape POD5 raw signal will instantiate."""
    s = ragged_trace_1d(
        pa.int16(),
        value_name="signal",
        x_name=None,
        extra_columns=(
            ("read_id", pa.string()),
            ("channel", pa.int32()),
            ("sampling_rate_hz", pa.float32()),
        ),
    )
    assert s.field("signal").type == pa.list_(pa.int16())
    assert "read_id" in [f.name for f in s]
    # No implicit x column.
    assert "x" not in [f.name for f in s]


def test_ragged_to_tensors_no_x():
    s = ragged_trace_1d(
        pa.int16(), value_name="signal", extra_columns=(("read_id", pa.string()),)
    )
    table = pa.table(
        {
            "read_id": pa.array(["a", "b"]),
            "signal": pa.array([[1, 2, 3], [10, 20]], type=pa.list_(pa.int16())),
        },
        schema=s,
    )
    values, xs = ragged_to_tensors(table, value_column="signal")
    assert xs is None
    assert len(values) == 2
    assert values[0].dtype == torch.int16
    assert torch.equal(values[0], torch.tensor([1, 2, 3], dtype=torch.int16))
    assert torch.equal(values[1], torch.tensor([10, 20], dtype=torch.int16))


def test_ragged_to_tensors_with_x():
    s = ragged_trace_1d(pa.float64(), value_name="intensity", x_name="mz")
    table = pa.table(
        {
            "mz": pa.array([[100.0, 200.0], [300.0]], type=pa.list_(pa.float64())),
            "intensity": pa.array([[1.0, 2.0], [9.0]], type=pa.list_(pa.float64())),
        },
        schema=s,
    )
    values, xs = ragged_to_tensors(table, value_column="intensity", x_column="mz")
    assert xs is not None
    assert len(values) == 2 and len(xs) == 2
    assert torch.equal(xs[0], torch.tensor([100.0, 200.0], dtype=torch.float64))
    assert torch.equal(values[1], torch.tensor([9.0], dtype=torch.float64))


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_default_registry_has_known_schemas():
    snap = registered_schemas()
    assert "Trace1D" in snap and "PeakTable" in snap


def test_register_schema_rejects_dupes():
    with pytest.raises(ValueError):
        register_schema("Trace1D", TRACE_1D)


# ----------------------------------------------------------------------
# Metadata round-trip
# ----------------------------------------------------------------------


def test_metadata_pack_unpack_roundtrip():
    d = {
        "run_id": "abc-123",
        "x.ce.capillary_index": 12,
        "x.hplc.wavelength_nm": 280.0,
        "x.hplc.signal_description": "DAD1A,Sig=220,4 Ref=360,100",
        "axis_values": [192.0, 194.0, 196.0],
    }
    packed = pack_metadata(d)
    assert all(isinstance(k, bytes) and isinstance(v, bytes) for k, v in packed.items())
    unpacked = unpack_metadata(packed)
    # JSON-encoded values round-trip; the description string survives as plain str.
    assert unpacked["run_id"] == "abc-123"
    assert unpacked["x.ce.capillary_index"] == 12
    assert unpacked["x.hplc.wavelength_nm"] == 280.0
    assert unpacked["axis_values"] == [192.0, 194.0, 196.0]


def test_unpack_empty():
    assert unpack_metadata(None) == {}
    assert unpack_metadata({}) == {}


def test_with_metadata_merges():
    t = pa.table({"time_s": pa.array([0.0, 1.0]), "intensity": pa.array([0.0, 1.0])})
    t = with_metadata(t, {"run_id": "first", "device": "DAD"})
    t = with_metadata(t, {"run_id": "second"})  # overwrites
    meta = unpack_metadata(t.schema.metadata)
    assert meta["run_id"] == "second"
    assert meta["device"] == "DAD"


# ----------------------------------------------------------------------
# Tensor bridges
# ----------------------------------------------------------------------


def test_trace_to_tensor_basic():
    t = pa.table(
        {
            "time_s": pa.array([0.0, 1.0, 2.0]),
            "intensity": pa.array([10.0, 20.0, 30.0]),
        }
    )
    times, ys = trace_to_tensor(t)
    assert times.dtype == torch.float64
    assert ys.dtype == torch.float64
    assert torch.allclose(times, torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(ys, torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))


def test_trace_to_tensor_filters_channel():
    t = pa.table(
        {
            "time_s": pa.array([0.0, 0.0, 1.0, 1.0]),
            "intensity": pa.array([1.0, 100.0, 2.0, 200.0]),
            "channel_id": pa.array(["A", "B", "A", "B"]),
        }
    )
    times, ys = trace_to_tensor(t, channel="B")
    assert torch.allclose(ys, torch.tensor([100.0, 200.0], dtype=torch.float64))


def test_spectra_tensor_roundtrip():
    n_time, n_axis = 5, 4
    times = torch.arange(n_time, dtype=torch.int64) * 100
    matrix = torch.arange(n_time * n_axis, dtype=torch.float64).reshape(n_time, n_axis)
    table = tensor_to_spectra(
        times,
        matrix,
        axis_name="wavelength_nm",
        axis_values=[200.0, 220.0, 240.0, 260.0],
        units="mAU",
    )
    # Schema correct.
    inner = table.schema.field("spectrum").type
    assert inner.list_size == n_axis
    # Metadata round-trips.
    meta = unpack_metadata(table.schema.metadata)
    assert meta["axis_name"] == "wavelength_nm"
    assert meta["axis_values"] == [200.0, 220.0, 240.0, 260.0]
    assert meta["units"] == "mAU"
    # Tensor round-trips bit-exact.
    rebuilt = spectra_to_tensor(table)
    assert rebuilt.shape == (n_time, n_axis)
    assert torch.equal(rebuilt, matrix)


def test_tensor_to_spectra_validates_shapes():
    times = torch.arange(3, dtype=torch.int64)
    bad_shape = torch.zeros(3)
    with pytest.raises(ValueError, match="must be 2D"):
        tensor_to_spectra(times, bad_shape, axis_name="x", axis_values=[0.0], units="")

    matrix = torch.zeros((3, 2))
    with pytest.raises(ValueError, match="doesn't match"):
        tensor_to_spectra(
            torch.arange(2, dtype=torch.int64),
            matrix,
            axis_name="x",
            axis_values=[0.0, 1.0],
            units="",
        )

    matrix = torch.zeros((3, 4))
    with pytest.raises(ValueError, match="length"):
        tensor_to_spectra(
            times,
            matrix,
            axis_name="x",
            axis_values=[0.0, 1.0, 2.0],  # wrong length
            units="",
        )


# ----------------------------------------------------------------------
# cast_to_schema
# ----------------------------------------------------------------------


def test_cast_drops_extra_columns_and_fills_nullable():
    t = pa.table(
        {
            "time_s": pa.array([0.0, 1.0]),
            "intensity": pa.array([1.0, 2.0]),
            "extra": pa.array(["x", "y"]),
        }
    )
    out = cast_to_schema(t, TRACE_1D)
    assert "extra" not in out.column_names
    assert "channel_id" in out.column_names  # nullable; filled with nulls
    assert out.column("channel_id").null_count == 2


def test_cast_raises_when_required_column_missing():
    t = pa.table({"intensity": pa.array([1.0])})
    # ``time_s`` is non-nullable in TRACE_1D.
    with pytest.raises(ValueError, match="missing"):
        cast_to_schema(t, TRACE_1D)


def test_cast_preserves_metadata():
    t = pa.table({"time_s": pa.array([0.0]), "intensity": pa.array([1.0])})
    t = with_metadata(t, {"run_id": "incoming"})
    out = cast_to_schema(t, TRACE_1D)
    meta = unpack_metadata(out.schema.metadata)
    assert meta["run_id"] == "incoming"
    assert meta["schema_name"] == "Trace1D"  # target metadata merged in
