"""Pile-up producer — RLE-encoded depth tracks for ``COVERAGE_TABLE``.

Sweep-line over the per-block intervals in ``ALIGNMENT_BLOCK_TABLE``,
emitting one row per interval of constant depth on the reference.
Splices (the gaps between blocks) contribute zero depth as expected,
so a spliced read covers only its exonic blocks — not the intron — in
the coverage track.

Per-sample stratification via a ``read_id → sample_id`` Arrow table.
When no mapping is supplied, the function emits unstratified depth with
a sentinel ``sample_id = -1`` so the schema (which requires non-null
``sample_id``) stays satisfied without lying about which sample
contributed the depth. Downstream consumers either filter on
``sample_id != -1`` (per-sample queries) or accept the aggregate
form.

Output modes:
- ``output_path`` provided → stream `(contig, sample)`-partitioned
  RecordBatches directly to Parquet; return a summary stats dict.
  This is the production hot path at mouse-scale (200M+ alignments)
  where the aggregate COVERAGE_TABLE would otherwise pin tens of GB
  resident.
- ``output_path`` omitted → assemble the full table in memory and
  return it. Used by tests and small-scale Jupyter callers.

Implementation: Arrow joins + sort_by (multithreaded C++); per-partition
sweep-line RLE in PyTorch CPU tensors (`torch.unique`, `scatter_add_`,
`cumsum` — all CPU intra-op parallelised).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from constellation.sequencing.schemas.quant import COVERAGE_TABLE


# Sentinel sample_id when no read_to_sample mapping is supplied.
# COVERAGE_TABLE.sample_id is non-nullable int64; -1 is a documented
# "depth not stratified by sample" marker, distinct from any real
# Samples row id (which start from 0 / positive).
_UNSTRATIFIED_SAMPLE_ID = -1


def build_pileup(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    contigs: pa.Table,
    *,
    output_path: Path | None = None,
    read_to_sample: pa.Table | None = None,
    primary_only: bool = True,
) -> pa.Table | dict[str, int]:
    """Compute per-position depth from per-block intervals.

    Parameters
    ----------
    alignment_blocks : pa.Table
        ``ALIGNMENT_BLOCK_TABLE``-shaped. Each row contributes a
        ``[ref_start, ref_end)`` interval of +1 depth on its
        alignment's contig.
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped — supplies ``read_id`` (for the
        sample mapping) and ``ref_name`` (for contig_id resolution).
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — supplies the ``ref_name → contig_id``
        resolution.
    output_path : Path | None, default None
        If provided, stream partitioned RecordBatches to a Parquet
        file at this path and return a summary stats dict
        ``{"n_partitions": int, "n_rows_written": int}``. Production
        callers always pass a path; tests + Jupyter callers omit it.
    read_to_sample : pa.Table | None, default None
        Optional 2-column Arrow table with ``read_id`` (string) +
        ``sample_id`` (int64) columns. When provided, the join drops
        reads without a sample mapping (matching the prior dict
        ``.get is None: continue`` semantics) and the output is
        stratified per sample. When ``None``, all reads contribute to
        a single ``sample_id = -1`` partition.
    primary_only : bool, default True
        Drop secondary / supplementary alignments before pile-up.
        Keeps depth interpretable (each read counts once per
        position) at the cost of dropping the chimeric tail.

    Returns
    -------
    pa.Table conforming to ``COVERAGE_TABLE`` (when ``output_path`` is
    None), or ``dict[str, int]`` summary stats (when streaming). Output
    is sorted by ``(contig_id, sample_id, start)``. Zero-depth intervals
    between runs are NOT emitted (callers compute them as the gaps
    between rows on the same contig+sample).
    """
    if alignment_blocks.num_rows == 0 or alignments.num_rows == 0:
        return _finalise_empty(output_path)

    if primary_only:
        primary_mask = pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
        alignments = alignments.filter(primary_mask)
        if alignments.num_rows == 0:
            return _finalise_empty(output_path)

    # Arrow-native join chain: blocks × primary → contig_id → sample_id.
    # All three steps are multithreaded C++ kernels; the prior version's
    # 5-column to_pylist + Python row-loop at 1.6B-block scale was the
    # dominant single-threaded hang.
    primary = alignments.select(["alignment_id", "read_id", "ref_name"])
    joined = alignment_blocks.join(
        primary, keys="alignment_id", join_type="inner"
    )
    if joined.num_rows == 0:
        return _finalise_empty(output_path)

    contig_lookup = contigs.select(["contig_id", "name"]).rename_columns(
        ["contig_id", "ref_name"]
    )
    joined = joined.join(contig_lookup, keys="ref_name", join_type="inner")
    if joined.num_rows == 0:
        return _finalise_empty(output_path)

    if read_to_sample is not None:
        # Inner join drops reads with no sample mapping (matches the prior
        # `read_to_sample.get(...) is None: continue` semantics).
        joined = joined.join(
            read_to_sample, keys="read_id", join_type="inner"
        )
    else:
        joined = joined.append_column(
            "sample_id",
            pa.array(
                np.full(joined.num_rows, _UNSTRATIFIED_SAMPLE_ID, dtype=np.int64),
                type=pa.int64(),
            ),
        )
    if joined.num_rows == 0:
        return _finalise_empty(output_path)

    # Sort by (contig_id, sample_id, ref_start) — keeps partition slices
    # contiguous and gives downstream sweep-line a sorted starts array
    # at zero extra cost.
    joined = joined.sort_by(
        [
            ("contig_id", "ascending"),
            ("sample_id", "ascending"),
            ("ref_start", "ascending"),
        ]
    ).combine_chunks()

    contig_arr = joined.column("contig_id").chunk(0).to_numpy(zero_copy_only=True)
    sample_arr = joined.column("sample_id").chunk(0).to_numpy(zero_copy_only=True)
    starts_arr = joined.column("ref_start").chunk(0).to_numpy(zero_copy_only=True)
    ends_arr = joined.column("ref_end").chunk(0).to_numpy(zero_copy_only=True)

    # Build a per-partition slice iterator using sorted-key transitions.
    # `combined_key` packs (contig_id, sample_id) into one int64 so the
    # boundary scan is one tensor diff. We need the two halves separately
    # later, so we also keep the per-partition (contig, sample) lookups.
    partition_boundaries = _partition_boundaries(contig_arr, sample_arr)

    writer: pq.ParquetWriter | None = None
    batches: list[pa.RecordBatch] = []
    n_rows_written = 0
    n_partitions = 0
    try:
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(output_path), schema=COVERAGE_TABLE)

        for start, size in partition_boundaries:
            cid = int(contig_arr[start])
            sid = int(sample_arr[start])
            part_starts = torch.from_numpy(starts_arr[start:start + size])
            part_ends = torch.from_numpy(ends_arr[start:start + size])

            seg_starts, seg_ends, seg_depths = _sweep_line_rle_torch(
                part_starts, part_ends
            )
            n = int(seg_starts.numel())
            if n == 0:
                continue
            n_partitions += 1
            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(np.full(n, cid, dtype=np.int64), type=pa.int64()),
                    pa.array(np.full(n, sid, dtype=np.int64), type=pa.int64()),
                    pa.array(seg_starts.numpy(), type=pa.int64()),
                    pa.array(seg_ends.numpy(), type=pa.int64()),
                    pa.array(seg_depths.numpy(), type=pa.int32()),
                ],
                schema=COVERAGE_TABLE,
            )
            n_rows_written += batch.num_rows
            if writer is not None:
                writer.write_batch(batch)
            else:
                batches.append(batch)
    finally:
        if writer is not None:
            writer.close()

    if output_path is not None:
        return {"n_partitions": n_partitions, "n_rows_written": n_rows_written}
    if not batches:
        return COVERAGE_TABLE.empty_table()
    return pa.Table.from_batches(batches, schema=COVERAGE_TABLE)


def _finalise_empty(output_path: Path | None) -> pa.Table | dict[str, int]:
    """Common empty-output handling: write an empty Parquet (so the
    callers' `_SUCCESS`-style markers still see a file) or return an
    empty table."""
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        empty = COVERAGE_TABLE.empty_table()
        pq.write_table(empty, str(output_path))
        return {"n_partitions": 0, "n_rows_written": 0}
    return COVERAGE_TABLE.empty_table()


def _partition_boundaries(
    contig_arr: np.ndarray, sample_arr: np.ndarray
) -> list[tuple[int, int]]:
    """Return (start, size) slices per unique (contig_id, sample_id) run.

    Both inputs must already be sorted by (contig_id, sample_id) so the
    runs are contiguous. Uses torch on CPU tensors for the diff +
    nonzero scan — single multithreaded pass.
    """
    n = contig_arr.shape[0]
    if n == 0:
        return []
    contig_t = torch.from_numpy(contig_arr)
    sample_t = torch.from_numpy(sample_arr)
    change_mask = torch.zeros(n, dtype=torch.bool)
    change_mask[0] = True
    if n > 1:
        change_mask[1:] = (contig_t[1:] != contig_t[:-1]) | (
            sample_t[1:] != sample_t[:-1]
        )
    starts = torch.nonzero(change_mask, as_tuple=False).squeeze(1).tolist()
    # Append a sentinel so size = next_start - this_start works uniformly.
    starts.append(n)
    return [(starts[i], starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]


def _sweep_line_rle_torch(
    starts: torch.Tensor, ends: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised sweep-line RLE: ``[start_i, end_i)`` intervals →
    ``(out_starts, out_ends, depths)`` non-zero-depth runs.

    All PyTorch CPU intra-op parallel. Algorithm:

    1. Concatenate start+end positions; tag deltas (+1 for starts, -1
       for ends).
    2. ``torch.unique`` on positions (sorted, with inverse) coalesces
       same-position events.
    3. ``scatter_add_`` sums deltas at each unique position.
    4. ``cumsum`` gives the depth across each
       ``[unique_pos[i], unique_pos[i+1])`` gap.
    5. Run-length-encode the per-gap depths to merge adjacent same-depth
       segments (the abutment-coalescing rule: read A ``[100, 200)`` +
       read B ``[200, 300)`` emit one ``[100, 300)`` depth-1 row).
    6. Filter to ``depth > 0`` runs (zero-depth gaps inside introns or
       between regions are implicit).
    """
    if starts.numel() == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64)
        empty_i32 = torch.empty(0, dtype=torch.int32)
        return empty_i64, empty_i64, empty_i32

    # Drop zero-length intervals (matches the original `if end <= start:
    # continue` guard inside the per-row Python loop).
    valid = ends > starts
    if not bool(valid.any()):
        empty_i64 = torch.empty(0, dtype=torch.int64)
        empty_i32 = torch.empty(0, dtype=torch.int32)
        return empty_i64, empty_i64, empty_i32
    starts = starts[valid]
    ends = ends[valid]
    n = starts.numel()

    events_pos = torch.cat([starts, ends])
    events_delta = torch.cat(
        [
            torch.ones(n, dtype=torch.int32),
            -torch.ones(n, dtype=torch.int32),
        ]
    )

    unique_pos, inverse = torch.unique(
        events_pos, return_inverse=True, sorted=True
    )
    n_unique = unique_pos.numel()
    coalesced = torch.zeros(n_unique, dtype=torch.int32)
    coalesced.scatter_add_(0, inverse, events_delta)
    depths = torch.cumsum(coalesced, dim=0).to(torch.int32)

    # Number of gap segments between consecutive unique positions.
    n_seg = n_unique - 1
    if n_seg <= 0:
        empty_i64 = torch.empty(0, dtype=torch.int64)
        empty_i32 = torch.empty(0, dtype=torch.int32)
        return empty_i64, empty_i64, empty_i32
    seg_depths = depths[:n_seg]

    # Run-length-encode seg_depths so adjacent same-depth gaps merge.
    # `is_run_start[i] = True` iff seg_depths[i] differs from seg_depths[i-1]
    # (with a virtual sentinel before index 0).
    is_run_start = torch.empty(n_seg, dtype=torch.bool)
    is_run_start[0] = True
    if n_seg > 1:
        is_run_start[1:] = seg_depths[1:] != seg_depths[:-1]
    run_start_idx = torch.nonzero(is_run_start, as_tuple=False).squeeze(1)
    # Each run ends where the next run starts; the final run ends at n_seg.
    next_start_idx = torch.cat(
        [run_start_idx[1:], torch.tensor([n_seg], dtype=run_start_idx.dtype)]
    )

    out_starts = unique_pos[run_start_idx]
    out_ends = unique_pos[next_start_idx]
    out_depths = seg_depths[run_start_idx]

    # Drop zero-depth runs (the "intron-internal depth is 0 and not
    # emitted" invariant).
    nonzero_mask = out_depths > 0
    return (
        out_starts[nonzero_mask].to(torch.int64),
        out_ends[nonzero_mask].to(torch.int64),
        out_depths[nonzero_mask].to(torch.int32),
    )


__all__ = ["build_pileup"]
