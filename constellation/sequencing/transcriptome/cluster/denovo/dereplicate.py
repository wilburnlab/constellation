"""Layer-0 exact dereplication.

Collapse byte-identical transcript windows to unique sequences with an
abundance count, via a C++-speed Arrow ``group_by("sequence")``. This
bounds *all* downstream work (minimizers, candidates, verify, consensus)
to the number of unique sequences rather than the raw read count.

Returns two tables:

* ``uniq_table`` — one row per unique sequence: ``uniq_id`` (dense
  ``0..U-1``), ``sequence``, ``abundance`` (raw read multiplicity),
  ``representative_read_id`` (lexicographically-first read), ``seq_len``.
* ``read_map`` — one row per input read: ``read_id`` → ``uniq_id`` +
  ``sample_id``. Drives per-(cluster, sample) quant and the
  per-(cluster, read) membership edges.

``uniq_id`` is the row index in ``uniq_table``; ``read_map.uniq_id`` is
resolved with a single ``pc.index_in`` hash-table build over the unique
sequences (no second string join).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


UNIQ_TABLE_SCHEMA = pa.schema(
    [
        pa.field("uniq_id", pa.int64(), nullable=False),
        pa.field("sequence", pa.large_string(), nullable=False),
        pa.field("abundance", pa.int64(), nullable=False),
        pa.field("representative_read_id", pa.string(), nullable=False),
        pa.field("seq_len", pa.int32(), nullable=False),
    ]
)

READ_MAP_SCHEMA = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("uniq_id", pa.int64(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=True),
    ]
)


def dereplicate(reads: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Exact-dereplicate ``reads`` → ``(uniq_table, read_map)``.

    ``reads`` carries ``read_id`` (string), ``sequence`` (the trimmed
    transcript window), and ``sample_id`` (int64, nullable).
    """
    if reads.num_rows == 0:
        return UNIQ_TABLE_SCHEMA.empty_table(), READ_MAP_SCHEMA.empty_table()

    seq = reads.column("sequence")
    if not pa.types.is_large_string(seq.type):
        seq = pc.cast(seq, pa.large_string())
    reads = reads.set_column(reads.schema.get_field_index("sequence"), "sequence", seq)

    grouped = reads.group_by("sequence").aggregate(
        [("read_id", "count"), ("read_id", "min")]
    )
    uniq_seq = grouped.column("sequence").combine_chunks()
    n_uniq = grouped.num_rows

    uniq_table = pa.table(
        {
            "uniq_id": pa.array(np.arange(n_uniq, dtype=np.int64)),
            "sequence": uniq_seq,
            "abundance": pc.cast(grouped.column("read_id_count"), pa.int64()),
            "representative_read_id": pc.cast(
                grouped.column("read_id_min"), pa.string()
            ),
            "seq_len": pc.cast(pc.utf8_length(uniq_seq), pa.int32()),
        },
        schema=UNIQ_TABLE_SCHEMA,
    )

    # uniq_id per read: position of each read's sequence in the unique set.
    uid_per_read = pc.cast(
        pc.index_in(reads.column("sequence"), value_set=uniq_seq), pa.int64()
    )
    read_map = pa.table(
        {
            "read_id": pc.cast(reads.column("read_id"), pa.string()),
            "uniq_id": uid_per_read,
            "sample_id": pc.cast(reads.column("sample_id"), pa.int64()),
        },
        schema=READ_MAP_SCHEMA,
    )
    return uniq_table, read_map


__all__ = ["dereplicate", "UNIQ_TABLE_SCHEMA", "READ_MAP_SCHEMA"]
