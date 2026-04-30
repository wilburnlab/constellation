"""``RAW_SIGNAL_TABLE`` — POD5 raw signal, ragged 1D per row.

Each row is one nanopore read's full electrical-current trace as the
strand translocated through the pore: variable-length ``list<int16>``
of raw ADC counts. Reads range from ~1k samples (short fragments) to
5M+ samples (ultra-long), which makes a flat row-per-sample layout
(``Trace1D``) untenable — one PromethION flow cell would explode to
~10¹³ rows. The ragged layout keeps it at one row per read with the
signal as a list cell.

Built on :func:`core.io.schemas.ragged_trace_1d`. The same factory
will instantiate ``MS_SCAN_TABLE`` when massspec scan readers ship
(value_dtype=float64, x_name='mz', extras=rt/ms_level/polarity).

Time is *implicit* — derive ``t_s = (start_sample + i) /
sampling_rate_hz`` rather than materializing an x-column. ``scale``
and ``offset`` convert int16 ADC counts to picoamperes:
``current_pA = scale * signal[i] + offset``.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import ragged_trace_1d, register_schema


RAW_SIGNAL_TABLE: pa.Schema = ragged_trace_1d(
    value_dtype=pa.int16(),
    value_name="signal",
    x_name=None,  # t implicit; derive from start_sample / sampling_rate_hz
    extra_columns=(
        ("read_id", pa.string()),
        ("acquisition_id", pa.int64()),
        ("channel", pa.int32()),
        ("well", pa.int32()),
        ("start_sample", pa.int64()),
        ("sampling_rate_hz", pa.float32()),
        # int16 → pA: current_pA = scale * raw + offset
        ("scale", pa.float32()),
        ("offset", pa.float32()),
    ),
)


# Replace the generic RaggedTrace1D schema_name with our concrete one;
# downstream registry consumers (cast_to_schema, doctor) want a
# distinct identity for POD5 rows even though the shape primitive is
# generic.
RAW_SIGNAL_TABLE = RAW_SIGNAL_TABLE.with_metadata({b"schema_name": b"RawSignalTable"})


register_schema("RawSignalTable", RAW_SIGNAL_TABLE)


__all__ = ["RAW_SIGNAL_TABLE"]
