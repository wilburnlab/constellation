"""``READ_TABLE`` — basecalled long-read records.

FASTA/FASTQ-shaped row layout. ``read_id`` is the canonical ONT UUID
("00000000-0000-0000-0000-000000000000"-style) and serves as the PK
that ``ALIGNMENT_TABLE`` rows FK into. ``acquisition_id`` ties each
read back to its source flow cell via :class:`Acquisitions`.

Quality is stored as the raw offset-33 ASCII string (BAM/FASTQ native);
:mod:`sequencing.quality.phred` exposes decode helpers when downstream
math needs Q-scores as floats. ``mean_quality`` is the precomputed
per-read aggregate so coarse filters skip the per-base decode.

``channel`` / ``start_time_s`` / ``duration_s`` are nullable — they
travel through Dorado output but are absent from FASTA inputs.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


READ_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        # offset-33 ASCII Phred quality string; same length as sequence
        pa.field("quality", pa.string(), nullable=True),
        pa.field("length", pa.int32(), nullable=False),
        pa.field("mean_quality", pa.float32(), nullable=True),
        pa.field("channel", pa.int32(), nullable=True),
        pa.field("start_time_s", pa.float64(), nullable=True),
        pa.field("duration_s", pa.float32(), nullable=True),
        # Dorado-reported per-read quality (``qs:f``). Distinct from
        # ``mean_quality``: Dorado computes qs in log-space across
        # chunked segments while we recompute mean_quality as the
        # plain arithmetic mean over the quality string. Both kept so
        # downstream filters can pick whichever signal matches their
        # expectations.
        pa.field("dorado_quality", pa.float32(), nullable=True),
        # SAM read-group identifier (``RG:Z``). Ties each read back
        # to a specific run / flowcell / chemistry — required when
        # joining multi-flowcell datasets at the read level.
        pa.field("read_group", pa.string(), nullable=True),
        # Duplex class (``dx:i``). Dorado emits 0=simplex, 1=duplex,
        # -1=simplex-with-duplex-offspring. Drives the S3 duplex-
        # artifact validation work; captured from S1 so the column
        # doesn't appear later as a backwards-incompatible schema bump.
        pa.field("duplex_class", pa.int8(), nullable=True),
    ],
    metadata={b"schema_name": b"ReadTable"},
)


register_schema("ReadTable", READ_TABLE)


__all__ = ["READ_TABLE"]
