"""``NMR_FID_TABLE`` — generic n-dimensional NMR FID, ragged 1D per row.

An NMR FID — whether 1D, 2D HSQC, 3D triple-resonance, or higher —
is stored as **two rows in a RaggedTrace1D-shaped table**, one row per
``component`` ('real' / 'imag'), each row holding the *complete*
multidimensional FID as a flat row-major array. The dimensional shape
travels in schema metadata under ``x.nmr.shape``.

This shape-in-metadata convention generalizes across dimensionalities
without schema changes:

    1D:   x.nmr.shape = [N_t2]                          (single FID)
    2D:   x.nmr.shape = [N_t1, N_t2]                    (HSQC, NOESY, ...)
    3D:   x.nmr.shape = [N_t1, N_t2, N_t3]              (triple-resonance)
    nD:   x.nmr.shape = [N_d, ..., N_2, N_1]            (high-dim)

Layout convention: row-major (numpy / torch C order), with the
**innermost dimension last in `shape` = the direct dimension** (t2 for
2D, t3 for 3D), and indirect dimensions listed outer-first. This
matches nmrglue's convention and the standard Bruker `ser`/`fid` byte
ordering — the on-disk file is read straight through and the resulting
flat array is `tensor.reshape(shape)` to recover the nD structure.

Built on :func:`core.io.schemas.ragged_trace_1d`. Time is *implicit* —
dwell spacing per dimension is reconstructable from per-dimension
sweep-width metadata (``x.nmr.sw_hz`` for direct, ``x.nmr.sw_hz_t1``
etc. for indirect dimensions when 2D+ readers land). Storing dense
time arrays in the table would be O(N) bytes wasted on uniformly spaced
data the metadata fully determines.

Why two rows (real / imag) rather than one row of complex or
interleaved samples: the universal ``ragged_trace_1d`` factory ships a
single value-dtype slot, which doesn't accommodate complex numbers
directly. Splitting into 'real' / 'imag' rows mirrors the ``Trace1D``
channel-stacking convention and keeps the table flat and column-typed.
Downstream code recovers the complex tensor with :func:`fid_to_complex`
below, which handles reshape + complex assembly in one step.

Open extension points for the future 2D port:
    - ``x.nmr.acquisition_mode_t1`` — TPPI / States / States-TPPI /
      echo-antiecho; determines how hypercomplex transposition combines
      cosine- and sine-modulated FIDs in the indirect dimension.
    - ``x.nmr.sw_hz_t1`` (and ``..._t2`` for 3D direct) — per-dimension
      sweep widths.
    - ``x.nmr.sfo1_mhz_t1`` etc. — per-dimension carrier frequencies.

None of those are populated by the 1D reader; they appear once the 2D
reader lands.
"""

from __future__ import annotations

import json

import pyarrow as pa
import torch

from typing import Any

from constellation.core.io.schemas import (
    pack_metadata,
    ragged_to_tensors,
    ragged_trace_1d,
    register_schema,
    unpack_metadata,
    with_metadata,
)


NMR_FID_TABLE: pa.Schema = ragged_trace_1d(
    value_dtype=pa.float64(),
    value_name="fid",
    x_name=None,  # t implicit; derive from dwell_s = 1 / x.nmr.sw_hz
    extra_columns=(
        ("component", pa.string()),  # "real" or "imag"
    ),
)

# Tag with a concrete schema name so doctor / cast_to_schema consumers
# can distinguish an NMR FID from a generic ragged 1D blob.
NMR_FID_TABLE = NMR_FID_TABLE.with_metadata({b"schema_name": b"NmrFidTable"})

register_schema("NmrFidTable", NMR_FID_TABLE)


def fid_to_complex(table: pa.Table) -> torch.Tensor:
    """Reconstruct an nD complex FID tensor from an ``NMR_FID_TABLE``.

    Reads the flat ``real`` and ``imag`` rows, assembles them into a
    complex tensor, and reshapes to the dimensional shape recorded in
    ``x.nmr.shape`` metadata. For 1D data the result is a 1D tensor of
    length ``N``; for 2D it's ``(N_t1, N_t2)``; for nD it's the full
    n-tensor in row-major (C) order matching the on-disk byte layout.

    Parameters
    ----------
    table : pa.Table
        Table built against ``NMR_FID_TABLE``. Must contain exactly two
        rows with ``component`` values 'real' and 'imag'; the
        ``x.nmr.shape`` metadata key must be present.

    Returns
    -------
    torch.Tensor
        Complex128 tensor in the dimensional shape from metadata.

    Raises
    ------
    ValueError
        If the table is malformed (wrong row count, missing components,
        missing shape metadata, or array length doesn't match shape).
    """
    if table.num_rows != 2:
        raise ValueError(
            f"NMR_FID_TABLE must contain exactly 2 rows (real, imag); "
            f"got {table.num_rows}"
        )

    components = table.column("component").to_pylist()
    if set(components) != {"real", "imag"}:
        raise ValueError(
            f"NMR_FID_TABLE rows must have components {{'real', 'imag'}}; "
            f"got {sorted(components)}"
        )

    metadata = unpack_metadata(table.schema.metadata)
    if "x.nmr.shape" not in metadata:
        raise ValueError(
            "NMR_FID_TABLE is missing required 'x.nmr.shape' metadata key"
        )
    shape = metadata["x.nmr.shape"]
    if isinstance(shape, str):
        # `unpack_metadata` json-decodes when possible; defensive fallback.
        shape = json.loads(shape)
    shape = tuple(int(n) for n in shape)

    values, _ = ragged_to_tensors(table, value_column="fid")
    # Order the (real, imag) pair by component to be robust to row ordering.
    by_component = dict(zip(components, values))
    real_flat = by_component["real"]
    imag_flat = by_component["imag"]

    expected_n = 1
    for n in shape:
        expected_n *= n
    if real_flat.numel() != expected_n or imag_flat.numel() != expected_n:
        raise ValueError(
            f"FID array length mismatch: shape {shape} requires "
            f"{expected_n} samples per component; got real={real_flat.numel()}, "
            f"imag={imag_flat.numel()}"
        )

    real = real_flat.to(torch.float64).reshape(shape)
    imag = imag_flat.to(torch.float64).reshape(shape)
    return torch.complex(real, imag)


def replace_fid_data(table: pa.Table, new_data: torch.Tensor) -> pa.Table:
    """Return a new ``NMR_FID_TABLE`` with new complex data, preserving metadata.

    Every NMR processing wrapper that transforms FID/spectrum data
    (apodization, zero-fill, FFT, phase correction) routes through here
    to keep the metadata-handling logic in one place. The function:

    1. Flattens ``new_data`` in row-major order and splits into ``real``
       / ``imag`` rows.
    2. Inherits every metadata key from ``table`` *except*
       ``x.nmr.shape``, which is rewritten from ``new_data.shape`` so
       length-changing operations (zero-fill) stay consistent.
    3. Preserves the schema-name marker (``NmrFidTable``) so the new
       table is still recognised by ``cast_to_schema`` consumers.

    Callers that need to update *other* metadata fields (e.g. FFT
    transitions ``x.nmr.domain`` from ``"fid"`` to ``"spectrum"``)
    follow this call with :func:`core.io.schemas.with_metadata`.

    Parameters
    ----------
    table : pa.Table
        Original ``NMR_FID_TABLE`` whose metadata is inherited.
    new_data : torch.Tensor
        Complex-valued tensor of any shape. Will be cast to
        ``torch.complex128`` for serialization.

    Returns
    -------
    pa.Table
        New ``NMR_FID_TABLE`` with the transformed data + inherited and
        shape-updated metadata.

    Raises
    ------
    ValueError
        If ``new_data`` is not a complex-valued tensor.
    """
    if new_data.dtype not in (torch.complex64, torch.complex128):
        raise ValueError(
            f"replace_fid_data requires a complex tensor; got dtype {new_data.dtype}"
        )

    new_data = new_data.detach().to(torch.complex128).cpu()
    real_flat = new_data.real.reshape(-1).tolist()
    imag_flat = new_data.imag.reshape(-1).tolist()

    components = pa.array(["real", "imag"], type=pa.string())
    fid_lists = pa.array(
        [real_flat, imag_flat], type=pa.list_(pa.float64())
    )
    out = pa.Table.from_arrays(
        [components, fid_lists], schema=NMR_FID_TABLE
    )

    inherited = unpack_metadata(table.schema.metadata)
    inherited["x.nmr.shape"] = list(new_data.shape)
    # Preserve the schema-name marker from the canonical schema header.
    schema_marker = unpack_metadata(NMR_FID_TABLE.metadata)
    return out.replace_schema_metadata(pack_metadata({**schema_marker, **inherited}))


def log_step(table: pa.Table, step_name: str, **params: Any) -> pa.Table:
    """Append a processing-history entry to ``x.nmr.processing_history``.

    Each NMR processing wrapper (apodization, zero-fill, FFT, phase
    correction, ...) calls this once after producing its output so the
    full lineage of transformations applied to a table is recoverable
    from the table itself. The history lives at
    ``x.nmr.processing_history`` as a JSON list of records, each
    record a dict ``{"step": str, "params": dict}``.

    Use this for:
        - Reproducibility — re-run a pipeline by walking the history.
        - Debugging — spot double-applied steps or missing steps.
        - Audit / publication — exportable provenance for processing
          choices that affect downstream peak heights / volumes.

    Parameters
    ----------
    table : pa.Table
        Table to annotate. Typically the output of
        :func:`replace_fid_data` from a processing wrapper.
    step_name : str
        Identifier for the processing step. Convention: ``"module.function"``
        (e.g. ``"apodization.em"``, ``"fourier.fft"``) so the record is
        unambiguous across modules that share function names.
    **params : Any
        Keyword arguments describing the step's parameters. Values
        must be JSON-serializable (float / int / str / bool / None /
        list / dict of same).

    Returns
    -------
    pa.Table
        Table with an updated ``x.nmr.processing_history`` list (the
        new entry appended to whatever was there).
    """
    existing = unpack_metadata(table.schema.metadata)
    history = existing.get("x.nmr.processing_history", [])
    if not isinstance(history, list):
        # Defensive: if upstream corruption produced a non-list value,
        # start a fresh history rather than failing here.
        history = []
    history.append({"step": step_name, "params": dict(params)})
    return with_metadata(table, {"x.nmr.processing_history": history})


__all__ = ["NMR_FID_TABLE", "fid_to_complex", "log_step", "replace_fid_data"]
