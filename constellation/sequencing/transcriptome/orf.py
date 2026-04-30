"""Open-reading-frame prediction over read tables.

Thin wrapper over :func:`core.sequence.nucleic.find_orfs` that takes a
``READ_DEMUX_TABLE`` (one row per resolved transcript window) and emits
per-read ORF rows. The actual ORF logic lives in core; this module
just adapts it to the Arrow-table interface.

Status: STUB. Pending Phase 6.
"""

from __future__ import annotations

import pyarrow as pa


_PHASE = "Phase 6 (transcriptome/{orf, cluster, consensus, network})"


# Output schema — one row per ORF found per read. read_id may repeat
# (multiple ORFs per read are common). Defined here rather than in
# schemas/transcriptome.py because it's a transient pipeline artifact
# rather than a stored container shape.
ORF_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("frame", pa.int8(), nullable=False),         # +1, +2, +3, -1, -2, -3
        pa.field("start", pa.int32(), nullable=False),         # 0-based half-open on read sense
        pa.field("end", pa.int32(), nullable=False),
        pa.field("length_aa", pa.int32(), nullable=False),
        pa.field("protein_sequence", pa.string(), nullable=False),
        pa.field("has_start_codon", pa.bool_(), nullable=False),
        pa.field("has_stop_codon", pa.bool_(), nullable=False),
        pa.field("codon_table", pa.int32(), nullable=False),   # NCBI transl_table
    ],
    metadata={b"schema_name": b"OrfTable"},
)


def predict_orfs(
    demuxed_reads: pa.Table,         # READ_DEMUX_TABLE joined to READ_TABLE
    *,
    min_length_aa: int = 50,
    require_start_codon: bool = False,
    codon_table: int = 1,            # NCBI transl_table (1=Standard)
    keep_top_n_per_read: int | None = 1,
) -> pa.Table:                       # ORF_TABLE
    """Run :func:`core.sequence.nucleic.find_orfs` per read and emit
    ``ORF_TABLE`` rows.

    ``keep_top_n_per_read`` defaults to 1 (longest ORF) — the lab's
    naive clustering uses this single-ORF-per-read assumption. Set
    to ``None`` to emit all ORFs above ``min_length_aa``.
    """
    raise NotImplementedError(f"predict_orfs pending {_PHASE}")


__all__ = [
    "ORF_TABLE",
    "predict_orfs",
]
