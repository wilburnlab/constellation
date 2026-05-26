"""Shared helpers for kernels that render per-alignment block geometry.

The read pile-up track (and, after PR 5, the cluster_pileup track's
``members`` view) attach two derived columns to each alignment row:

- ``blocks``: list-of-struct ``[{ref_start, ref_end, n_match, n_mismatch}]``
  built by joining the per-alignment ``ALIGNMENT_BLOCK_TABLE`` rows back
  into a single Arrow list-of-struct cell per alignment. Drives the
  solid exon-segment + dotted intron-connector rendering.
- ``mismatch_positions``: list of reference positions where cs:long
  recorded a substitution. Drives the per-base X glyph overlay.

Both helpers are kept small and pure — no caching, no side effects — so
they can be unit-tested with hand-built Arrow tables. The cs:long parse
walks the shared
:func:`constellation.sequencing.align.cigar.parse_cs_long_mismatch_positions`
helper; this is the **first viz → sequencing import** at runtime (the
precedent set by ``align.consensus`` for cluster building). The cs
grammar is non-trivial and lives in one place by design — duplicating
the regex walk in viz would be the wrong trade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from pathlib import Path


# Wire-side struct fields for `blocks: list<struct{...}>`. Kept here so
# the read_pileup and cluster_pileup (member-view, PR 5) schemas use the
# same per-block layout.
BLOCK_STRUCT_TYPE: pa.DataType = pa.struct(
    [
        ("ref_start", pa.int64()),
        ("ref_end", pa.int64()),
        ("n_match", pa.int32()),
        ("n_mismatch", pa.int32()),
    ]
)
BLOCKS_LIST_TYPE: pa.DataType = pa.list_(BLOCK_STRUCT_TYPE)
MISMATCH_POSITIONS_TYPE: pa.DataType = pa.list_(pa.int64())


def attach_blocks(
    alignments: pa.Table,
    blocks_table: pa.Table,
) -> pa.Table:
    """Append a ``blocks`` list-of-struct column to ``alignments``.

    For each alignment_id in ``alignments``, the column holds the list
    of its blocks (``ref_start``, ``ref_end``, ``n_match``,
    ``n_mismatch``) sorted by ``block_index``. Alignments without a
    matching row in ``blocks_table`` get a single fallback block
    spanning the alignment's own ``(ref_start, ref_end)`` with null
    match/mismatch counts — keeps the renderer's per-row loop uniform
    in shape (no special-case branch for "block-less" alignments).

    Both inputs must carry ``alignment_id``. ``alignments`` additionally
    needs ``ref_start`` and ``ref_end`` for the fallback path.

    The implementation is Arrow-native: it sorts blocks once on
    ``(alignment_id, block_index)``, then for each alignment row pulls
    its slice via ``np.searchsorted``-style index resolution against
    the sorted ``alignment_id`` column. No ``to_pylist()`` on
    block-cardinality data — sticks to the resolve-stage rule.
    """
    if alignments.num_rows == 0:
        return alignments.append_column(
            "blocks", pa.array([], type=BLOCKS_LIST_TYPE)
        )

    aln_ids = alignments.column("alignment_id")
    aln_starts = alignments.column("ref_start")
    aln_ends = alignments.column("ref_end")

    # No blocks at all → fall back to one-block-per-alignment for every row.
    if blocks_table.num_rows == 0:
        fallback = _build_fallback_blocks(aln_starts, aln_ends)
        return alignments.append_column("blocks", fallback)

    # Sort blocks once by (alignment_id, block_index) so each
    # alignment's block run is contiguous and ordered.
    sorted_blocks = blocks_table.sort_by(
        [("alignment_id", "ascending"), ("block_index", "ascending")]
    )
    sb_ids = sorted_blocks.column("alignment_id").combine_chunks()
    sb_starts = sorted_blocks.column("ref_start").combine_chunks()
    sb_ends = sorted_blocks.column("ref_end").combine_chunks()
    sb_match = sorted_blocks.column("n_match").combine_chunks()
    sb_mm = sorted_blocks.column("n_mismatch").combine_chunks()

    sb_ids_np = sb_ids.to_numpy(zero_copy_only=False)
    # Per-alignment slice bounds: lo[i]..hi[i] index into sb_*.
    # np.searchsorted on the sorted alignment_id column gives O(n log m)
    # total work where n=alignments.num_rows, m=blocks rows.
    import numpy as np

    aln_ids_np = aln_ids.to_numpy(zero_copy_only=False)
    lo = np.searchsorted(sb_ids_np, aln_ids_np, side="left")
    hi = np.searchsorted(sb_ids_np, aln_ids_np, side="right")

    # Build the flat-value buffers of the list<struct> column. Empty
    # slices fall back to a single-block synthetic from the alignment's
    # own (ref_start, ref_end). We coalesce the synthetic and real
    # blocks into one struct array, then assemble the list offsets.
    aln_starts_np = aln_starts.to_numpy(zero_copy_only=False)
    aln_ends_np = aln_ends.to_numpy(zero_copy_only=False)
    sb_starts_np = sb_starts.to_numpy(zero_copy_only=False)
    sb_ends_np = sb_ends.to_numpy(zero_copy_only=False)
    sb_match_np = _nullable_int32_to_object(sb_match)
    sb_mm_np = _nullable_int32_to_object(sb_mm)

    # Pre-size the flat-struct arrays. Each alignment contributes either
    # `hi-lo` real blocks (when >0) or 1 fallback block.
    counts = np.maximum(hi - lo, 1)
    total = int(counts.sum())
    flat_start = np.empty(total, dtype=np.int64)
    flat_end = np.empty(total, dtype=np.int64)
    flat_match: list[int | None] = [None] * total
    flat_mm: list[int | None] = [None] * total

    offsets = np.empty(alignments.num_rows + 1, dtype=np.int32)
    offsets[0] = 0
    cursor = 0
    for i in range(alignments.num_rows):
        slice_lo = int(lo[i])
        slice_hi = int(hi[i])
        if slice_hi > slice_lo:
            n = slice_hi - slice_lo
            flat_start[cursor : cursor + n] = sb_starts_np[slice_lo:slice_hi]
            flat_end[cursor : cursor + n] = sb_ends_np[slice_lo:slice_hi]
            for j in range(n):
                flat_match[cursor + j] = sb_match_np[slice_lo + j]
                flat_mm[cursor + j] = sb_mm_np[slice_lo + j]
            cursor += n
        else:
            # Fallback: one synthetic block spanning the alignment.
            flat_start[cursor] = int(aln_starts_np[i])
            flat_end[cursor] = int(aln_ends_np[i])
            flat_match[cursor] = None
            flat_mm[cursor] = None
            cursor += 1
        offsets[i + 1] = cursor

    struct_arr = pa.StructArray.from_arrays(
        [
            pa.array(flat_start, type=pa.int64()),
            pa.array(flat_end, type=pa.int64()),
            pa.array(flat_match, type=pa.int32()),
            pa.array(flat_mm, type=pa.int32()),
        ],
        fields=list(BLOCK_STRUCT_TYPE),
    )
    list_arr = pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), struct_arr)
    return alignments.append_column("blocks", list_arr)


def attach_mismatch_positions(
    alignments: pa.Table,
    cs_path: "Path | None",
    *,
    skip: bool = False,
) -> pa.Table:
    """Append a ``mismatch_positions`` list<int64> column to ``alignments``.

    When ``skip`` is True (zoom too coarse to render per-base glyphs)
    or ``cs_path`` is None / does not exist, every row gets an empty
    list. Otherwise the per-alignment ``cs_string`` is parsed via
    :func:`parse_cs_long_mismatch_positions` and the resulting positions
    populate each row's list.

    The cs path is opened with predicate pushdown — only rows whose
    ``alignment_id`` is in scope are decoded — so the cost is bounded
    by the visible-window alignment count (already capped at the
    kernel's ``vector_glyph_limit``).
    """
    if alignments.num_rows == 0:
        return alignments.append_column(
            "mismatch_positions", pa.array([], type=MISMATCH_POSITIONS_TYPE)
        )

    if skip or cs_path is None or not cs_path.exists():
        empty = pa.array(
            [[] for _ in range(alignments.num_rows)],
            type=MISMATCH_POSITIONS_TYPE,
        )
        return alignments.append_column("mismatch_positions", empty)

    from constellation.sequencing.align.cigar import (
        parse_cs_long_mismatch_positions,
    )

    aln_ids = alignments.column("alignment_id").to_pylist()
    aln_starts = alignments.column("ref_start").to_pylist()
    id_set = set(int(x) for x in aln_ids)
    if not id_set:
        empty = pa.array(
            [[] for _ in range(alignments.num_rows)],
            type=MISMATCH_POSITIONS_TYPE,
        )
        return alignments.append_column("mismatch_positions", empty)

    # Read cs_path with predicate pushdown so we don't decode the entire
    # alignment_cs/ partitioned dataset just to look at the visible window.
    import pyarrow.dataset as pa_ds

    dataset = pa_ds.dataset(str(cs_path), format="parquet")
    filtered = dataset.to_table(
        columns=["alignment_id", "cs_string"],
        filter=pc.field("alignment_id").isin(pa.array(list(id_set), pa.int64())),
    )
    cs_by_aid: dict[int, str] = {
        int(aid): str(cs)
        for aid, cs in zip(
            filtered.column("alignment_id").to_pylist(),
            filtered.column("cs_string").to_pylist(),
            strict=True,
        )
        if cs is not None
    }

    positions_per_row: list[list[int]] = []
    for aid, ref_start in zip(aln_ids, aln_starts, strict=True):
        cs = cs_by_aid.get(int(aid))
        if cs is None:
            positions_per_row.append([])
            continue
        try:
            positions_per_row.append(
                parse_cs_long_mismatch_positions(cs, int(ref_start))
            )
        except ValueError:
            # Malformed cs in one row should never crash the whole
            # request — silently drop just that alignment's glyphs.
            positions_per_row.append([])

    arr = pa.array(positions_per_row, type=MISMATCH_POSITIONS_TYPE)
    return alignments.append_column("mismatch_positions", arr)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _build_fallback_blocks(
    starts: pa.ChunkedArray | pa.Array,
    ends: pa.ChunkedArray | pa.Array,
) -> pa.ListArray:
    """One-block-per-alignment fallback when no per-CIGAR blocks ship.

    Each alignment becomes a single synthetic block spanning the
    alignment's own ``(ref_start, ref_end)`` with null match/mismatch
    counts (we have no way to distinguish them from CIGAR alone here).
    """
    import numpy as np

    starts_np = starts.to_numpy(zero_copy_only=False).astype(np.int64)
    ends_np = ends.to_numpy(zero_copy_only=False).astype(np.int64)
    n = len(starts_np)
    null_i32 = pa.array([None] * n, type=pa.int32())
    struct_arr = pa.StructArray.from_arrays(
        [
            pa.array(starts_np, type=pa.int64()),
            pa.array(ends_np, type=pa.int64()),
            null_i32,
            null_i32,
        ],
        fields=list(BLOCK_STRUCT_TYPE),
    )
    offsets = pa.array(list(range(n + 1)), type=pa.int32())
    return pa.ListArray.from_arrays(offsets, struct_arr)


def _nullable_int32_to_object(arr: pa.Array) -> list[int | None]:
    """Convert a nullable int32 Arrow array into a Python list that
    preserves null cells as ``None``.

    We can't use ``to_numpy`` on a nullable column with default zero-fill
    semantics because the renderer needs to distinguish "no data" from
    zero. ``to_pylist`` is acceptable here because the cardinality is
    bounded by the visible-window block count (a few thousand at most
    at the kernel's glyph cap), not by alignment-block cardinality at
    PromethION scale.
    """
    return arr.to_pylist()


__all__ = [
    "BLOCK_STRUCT_TYPE",
    "BLOCKS_LIST_TYPE",
    "MISMATCH_POSITIONS_TYPE",
    "attach_blocks",
    "attach_mismatch_positions",
]
