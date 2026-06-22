"""Layer-0 exact dereplication.

Collapse byte-identical transcript windows to unique sequences with an
abundance count. This bounds *all* downstream work (minimizers,
candidates, verify, consensus) to the number of unique sequences rather
than the raw read count.

We do **not** ``group_by`` the raw sequence string: at PromethION scale
the distinct grouping keys total tens of GB (≈18M unique × ~1 kb), which
overflows the int32 offset in Acero's distinct-key storage and aborts the
process with ``std::length_error: vector::_M_default_append``. Instead we
hash each window to a fixed-width 128-bit key (xxh3-128, fixed_size_binary
— collision probability ≈ n²/2¹²⁹ ≈ 1e-22 at 19M reads, i.e. never) and
group on that fixed-width column; the representative sequence is recovered
with a ``pc.take`` at each group's first row.

Returns two tables:

* ``uniq_table`` — one row per unique sequence: ``uniq_id`` (dense
  ``0..U-1``), ``sequence``, ``abundance`` (raw read multiplicity),
  ``representative_read_id`` (the group's first-occurring read), ``seq_len``.
* ``read_map`` — one row per input read: ``read_id`` → ``uniq_id`` +
  ``sample_id``. Drives per-(cluster, sample) quant and the
  per-(cluster, read) membership edges.
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


def _hash_sequences(seq_col: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Per-read 128-bit sequence hash as a ``fixed_size_binary(16)`` array.

    Hashing to a fixed-width key lets the dedup ``group_by`` avoid Acero's
    variable-length distinct-key storage, whose int32 offset overflows once
    the unique keys exceed ~2 GB. Walks the Arrow value buffers directly
    (offset-aware, chunk-aware) — no per-row Python string materialisation
    beyond the unavoidable hash call.
    """
    import xxhash

    digest = xxhash.xxh3_128_digest
    n_total = len(seq_col)
    out = bytearray(16 * n_total)
    chunks = (
        seq_col.chunks if isinstance(seq_col, pa.ChunkedArray) else [seq_col]
    )
    pos = 0
    for ch in chunks:
        nch = len(ch)
        if nch == 0:
            continue
        if not pa.types.is_large_string(ch.type):
            ch = pc.cast(ch, pa.large_string())
        bufs = ch.buffers()
        offsets = np.frombuffer(bufs[1], dtype=np.int64)
        base = ch.offset
        data = memoryview(bufs[2]) if bufs[2] is not None else memoryview(b"")
        for i in range(nch):
            a = int(offsets[base + i])
            b = int(offsets[base + i + 1])
            out[pos : pos + 16] = digest(data[a:b])
            pos += 16
    return pa.Array.from_buffers(
        pa.binary(16), n_total, [None, pa.py_buffer(bytes(out))]
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
    n = reads.num_rows

    seq_hash = _hash_sequences(seq)
    work = pa.table(
        {"h": seq_hash, "row_idx": pa.array(np.arange(n, dtype=np.int64))}
    )
    grouped = work.group_by("h").aggregate([("row_idx", "min"), ("row_idx", "count")])
    group_hash = grouped.column("h")
    rep_row = grouped.column("row_idx_min")
    n_uniq = grouped.num_rows

    # Representative sequence + read_id = the group's first-occurring row.
    uniq_seq = pc.take(seq, rep_row)
    rep_read = pc.cast(pc.take(reads.column("read_id"), rep_row), pa.string())

    uniq_table = pa.table(
        {
            "uniq_id": pa.array(np.arange(n_uniq, dtype=np.int64)),
            "sequence": uniq_seq,
            "abundance": pc.cast(grouped.column("row_idx_count"), pa.int64()),
            "representative_read_id": rep_read,
            "seq_len": pc.cast(pc.utf8_length(uniq_seq), pa.int32()),
        },
        schema=UNIQ_TABLE_SCHEMA,
    )

    # uniq_id per read = position of each read's hash in the distinct-hash
    # set (group_by output order, which is the uniq_id order).
    uid_per_read = pc.cast(
        pc.index_in(seq_hash, value_set=group_hash), pa.int64()
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


__all__ = ["dereplicate", "_hash_sequences", "UNIQ_TABLE_SCHEMA", "READ_MAP_SCHEMA"]
