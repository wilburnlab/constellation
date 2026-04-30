"""Canonical Arrow schemas + tensor / metadata helpers.

These schemas are *generic data shapes* — column layouts that recur
across modalities (HPLC chromatograms, CE electropherograms, MS XICs,
POD5 raw signal traces, ...). A schema earns a slot here only if its
column structure is universal across ≥2 modalities; modality-specific
information rides as **namespaced schema metadata**, not as columns.

    Trace1D          — (time_s, intensity) with optional channel_id;
                       row-per-sample, fine for short benchtop traces.
    SpectraMatrix2D  — (time_ms, spectrum: list<float64>[K])  for any
                       regular second axis (DAD wavelength, 2D NMR
                       chemical shift); fixed inner dim.
    RaggedTrace1D    — variable-length 1D blob per row, for POD5 raw
                       signal (~10⁶ samples/read), MS scan blobs
                       (~10⁵ peaks/scan), NMR FIDs, anywhere a row
                       holds an irregular 1D array. Built via the
                       ``ragged_trace_1d(value_dtype, ...)`` factory.
    PeakTable        — universal peak record across modalities.

Metadata convention. Schema-level metadata is JSON-encoded UTF-8
strings under byte keys. Universal keys (no prefix): ``run_id``,
``run_datetime``, ``instrument_id``, ``device``, ``units``,
``sampling_rate_hz``, ``channel_name``, ``slope``, ``schema_name``.
Modality-specific keys are namespaced ``x.<domain>.<key>``:

    x.ce.capillary_index, x.ce.ladder_name, x.ce.calibration_model
    x.hplc.wavelength_nm, x.hplc.signal_description
    x.pod5.read_id, x.pod5.channel_id
    x.ms.precursor_mz, x.ms.charge

``pack_metadata`` / ``unpack_metadata`` round-trip the dict <-> bytes
form; consumers should always go through these.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import torch


# ----------------------------------------------------------------------
# Universal column layouts
# ----------------------------------------------------------------------

# Trace1D — a single 1D time-series. ``channel_id`` is nullable so
# single-channel tables can omit it; stacked multi-channel tables fill
# it (CE capillary index as string, HPLC channel name like "DAD1A",
# POD5 read id, ...).
TRACE_1D = pa.schema(
    [
        pa.field("time_s", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        pa.field("channel_id", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"Trace1D"},
)


def spectra_matrix_2d(n_axis: int) -> pa.Schema:
    """Build a SpectraMatrix2D schema for a fixed inner-axis size.

    The inner axis size is part of the type (``fixed_size_list``), so
    the schema is parameterised. The axis values themselves (e.g.
    wavelength grid in nm) live in schema metadata under
    ``axis_values`` as JSON.
    """
    if n_axis <= 0:
        raise ValueError(f"n_axis must be positive, got {n_axis}")
    return pa.schema(
        [
            pa.field("time_ms", pa.int64(), nullable=False),
            pa.field(
                "spectrum",
                pa.list_(pa.float64(), n_axis),
                nullable=False,
            ),
        ],
        metadata={b"schema_name": b"SpectraMatrix2D"},
    )


def ragged_trace_1d(
    value_dtype: pa.DataType,
    *,
    value_name: str = "y",
    x_name: str | None = None,
    extra_columns: Sequence[tuple[str, pa.DataType]] = (),
) -> pa.Schema:
    """Schema for variable-length 1D arrays per row.

    Distinct from ``Trace1D`` (one row per sample — explodes for POD5)
    and ``SpectraMatrix2D`` (fixed inner dim — only works on regular
    grids like DAD wavelength bins). Used for:

        POD5 raw signal     value_name='signal', value_dtype=int16,
                            x implicit (t = start_sample / sampling_rate)
        MS scan blob        value_name='intensity', value_dtype=float64,
                            x_name='mz'  (when massspec scan readers ship)
        NMR FID             value_name='fid', value_dtype=float64,
                            x_name='time_us'

    The bulk-array column comes last in the field order so per-row
    metadata reads first. ``extra_columns`` carry whatever per-row
    metadata the modality needs (read_id, channel, scale, sampling_rate
    for POD5; rt, ms_level, polarity for MS scans).
    """
    cols = [pa.field(name, dtype, nullable=False) for name, dtype in extra_columns]
    if x_name is not None:
        cols.append(pa.field(x_name, pa.list_(pa.float64()), nullable=False))
    cols.append(pa.field(value_name, pa.list_(value_dtype), nullable=False))
    return pa.schema(cols, metadata={b"schema_name": b"RaggedTrace1D"})


PEAK_TABLE = pa.schema(
    [
        pa.field("peak_id", pa.int32(), nullable=False),
        pa.field("channel", pa.string(), nullable=True),
        pa.field("retention_time_s", pa.float64(), nullable=False),
        pa.field("area", pa.float64(), nullable=False),
        pa.field("height", pa.float64(), nullable=False),
        pa.field("fwhm_s", pa.float64(), nullable=True),
        pa.field("asymmetry", pa.float64(), nullable=True),
        pa.field("snr", pa.float64(), nullable=True),
        pa.field("baseline", pa.float64(), nullable=True),
        pa.field("calibrated_value", pa.float64(), nullable=True),
        pa.field("calibrated_unit", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"PeakTable"},
)


# ----------------------------------------------------------------------
# Schema registry
# ----------------------------------------------------------------------

_REGISTRY: dict[str, pa.Schema] = {
    "Trace1D": TRACE_1D,
    "PeakTable": PEAK_TABLE,
    # SpectraMatrix2D is parametric (n_axis) — register the canonical
    # *factory* under a distinct key so consumers can ask "is this a
    # SpectraMatrix2D?" without committing to a specific axis size.
}


def register_schema(name: str, schema: pa.Schema) -> None:
    """Add a schema to the registry. Domain modules call this for any
    genuinely cross-modality schema they introduce."""
    if name in _REGISTRY:
        raise ValueError(f"schema {name!r} already registered")
    _REGISTRY[name] = schema


def get_schema(name: str) -> pa.Schema:
    if name not in _REGISTRY:
        raise KeyError(f"schema {name!r} is not registered")
    return _REGISTRY[name]


def registered_schemas() -> dict[str, pa.Schema]:
    return dict(_REGISTRY)


# ----------------------------------------------------------------------
# Metadata pack / unpack
# ----------------------------------------------------------------------


def pack_metadata(d: dict[str, Any]) -> dict[bytes, bytes]:
    """Encode a flat ``str -> Any`` dict to Arrow's ``bytes -> bytes``.

    Values that are not already strings get ``json.dumps``'d so round-
    tripping through Arrow is loss-free. Keys are UTF-8 encoded as-is.
    """
    out: dict[bytes, bytes] = {}
    for k, v in d.items():
        kb = k.encode("utf-8")
        if isinstance(v, str):
            out[kb] = v.encode("utf-8")
        elif isinstance(v, bytes):
            out[kb] = v
        else:
            out[kb] = json.dumps(v).encode("utf-8")
    return out


def unpack_metadata(meta: dict[bytes, bytes] | None) -> dict[str, Any]:
    """Inverse of ``pack_metadata``. Strings that don't round-trip
    through ``json.loads`` come back as plain ``str``."""
    if not meta:
        return {}
    out: dict[str, Any] = {}
    for kb, vb in meta.items():
        k = kb.decode("utf-8") if isinstance(kb, bytes) else str(kb)
        v = vb.decode("utf-8") if isinstance(vb, bytes) else vb
        try:
            out[k] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            out[k] = v
    return out


def with_metadata(table: pa.Table, extras: dict[str, Any]) -> pa.Table:
    """Return ``table`` with its schema metadata merged with ``extras``."""
    existing = unpack_metadata(table.schema.metadata)
    existing.update(extras)
    return table.replace_schema_metadata(pack_metadata(existing))


# ----------------------------------------------------------------------
# Tensor bridges
# ----------------------------------------------------------------------


def trace_to_tensor(
    table: pa.Table,
    channel: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull a ``Trace1D`` table into ``(time_s, intensity)`` tensors.

    If the table has a ``channel_id`` column and ``channel`` is given,
    rows are filtered to that channel first. Returns float64 tensors
    on CPU; move with ``.to(device)``.
    """
    if channel is not None and "channel_id" in table.column_names:
        mask = pa.compute.equal(table["channel_id"], channel)
        table = table.filter(mask)
    t = torch.from_numpy(table.column("time_s").to_numpy(zero_copy_only=False))
    y = torch.from_numpy(table.column("intensity").to_numpy(zero_copy_only=False))
    return t, y


def spectra_to_tensor(table: pa.Table) -> torch.Tensor:
    """Reconstruct a ``SpectraMatrix2D`` table into a (T, K) tensor.

    Routes through Arrow's native ``to_numpy_ndarray`` for the
    fixed-size-list column (zero-copy when contiguous, one allocation
    otherwise) and hands off to torch via ``torch.from_numpy``. No
    Python-level row iteration.
    """
    col = table.column("spectrum")
    # ChunkedArray of FixedSizeListArray — combine into one ndarray of
    # shape (n_rows, list_size). PyArrow exposes this as a single
    # to_numpy() call on the combined array.
    if hasattr(col, "combine_chunks"):
        arr = col.combine_chunks()
    else:
        arr = col
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.ndim == 1:
        # to_numpy on a FixedSizeListArray sometimes returns an object
        # array of per-row 1D ndarrays — coerce to 2D.
        import numpy as np

        np_arr = np.stack(np_arr)
    return torch.from_numpy(np_arr)


def ragged_to_tensors(
    table: pa.Table,
    *,
    value_column: str = "y",
    x_column: str | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Project ragged ``list<numeric>`` columns to per-row torch tensors.

    Returns ``(values, xs)`` where ``values[i]`` is the 1D tensor from
    row ``i``'s ``value_column``, and ``xs[i]`` is the 1D tensor from
    ``x_column`` (or ``None`` when ``x_column`` is omitted). Each row
    may have a different length — caller pads / batches as needed.
    """
    value_arr = table.column(value_column)
    if hasattr(value_arr, "combine_chunks"):
        value_arr = value_arr.combine_chunks()
    values = [torch.from_numpy(value_arr[i].values.to_numpy(zero_copy_only=False).copy())
              for i in range(len(value_arr))]
    if x_column is None:
        return values, None
    x_arr = table.column(x_column)
    if hasattr(x_arr, "combine_chunks"):
        x_arr = x_arr.combine_chunks()
    xs = [torch.from_numpy(x_arr[i].values.to_numpy(zero_copy_only=False).copy())
          for i in range(len(x_arr))]
    return values, xs


def tensor_to_spectra(
    times_ms: torch.Tensor,
    matrix: torch.Tensor,
    *,
    axis_name: str,
    axis_values: list[float],
    units: str,
    extra_metadata: dict[str, Any] | None = None,
) -> pa.Table:
    """Inverse of ``spectra_to_tensor``. Wraps a ``(T, K)`` torch
    tensor into a ``SpectraMatrix2D`` Arrow table with axis values
    stamped into metadata."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D (T, K), got shape {tuple(matrix.shape)}")
    n_rows, n_axis = matrix.shape
    if times_ms.shape != (n_rows,):
        raise ValueError(
            f"times_ms shape {tuple(times_ms.shape)} doesn't match matrix rows {n_rows}"
        )
    if len(axis_values) != n_axis:
        raise ValueError(
            f"axis_values length {len(axis_values)} != matrix inner dim {n_axis}"
        )
    schema = spectra_matrix_2d(n_axis)
    # Flatten the matrix into a 1D float64 buffer for FixedSizeListArray.
    flat = matrix.contiguous().to(torch.float64).reshape(-1).cpu().numpy()
    values = pa.array(flat, type=pa.float64())
    spectra = pa.FixedSizeListArray.from_arrays(values, n_axis)
    times = pa.array(times_ms.to(torch.int64).cpu().numpy(), type=pa.int64())
    table = pa.table([times, spectra], schema=schema)
    metadata: dict[str, Any] = {
        "axis_name": axis_name,
        "axis_values": axis_values,
        "units": units,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return with_metadata(table, metadata)


# ----------------------------------------------------------------------
# Forward-compatibility cast
# ----------------------------------------------------------------------


def cast_to_schema(table: pa.Table, target: pa.Schema) -> pa.Table:
    """Project ``table`` onto ``target``: keep matching columns, drop
    extras, fill missing nullable columns with nulls. Metadata of the
    incoming table is preserved (``target``'s metadata is merged in,
    not replaced)."""
    new_cols: list[pa.Array | pa.ChunkedArray] = []
    new_fields: list[pa.Field] = []
    for field in target:
        if field.name in table.column_names:
            col = table.column(field.name)
            if col.type != field.type:
                col = col.cast(field.type)
            new_cols.append(col)
        elif field.nullable:
            new_cols.append(pa.nulls(table.num_rows, type=field.type))
        else:
            raise ValueError(
                f"cannot cast: required column {field.name!r} ({field.type}) missing"
            )
        new_fields.append(field)
    incoming_meta = unpack_metadata(table.schema.metadata)
    target_meta = unpack_metadata(target.metadata)
    incoming_meta.update(target_meta)
    out_schema = pa.schema(new_fields, metadata=pack_metadata(incoming_meta))
    return pa.table(new_cols, schema=out_schema)
