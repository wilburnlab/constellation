"""Pile-up producer — RLE-encoded depth tracks for ``COVERAGE_TABLE``.

Sweep-line over the per-block intervals in ``ALIGNMENT_BLOCK_TABLE``,
emitting one row per interval of constant depth on the reference.
Splices (the gaps between blocks) contribute zero depth as expected,
so a spliced read covers only its exonic blocks — not the intron — in
the coverage track.

Per-sample stratification via a ``read_id → sample_id`` mapping. When
no mapping is supplied, the function emits unstratified depth with a
sentinel ``sample_id = -1`` so the schema (which requires non-null
``sample_id``) stays satisfied without lying about which sample
contributed the depth. Downstream consumers either filter on
``sample_id != -1`` (per-sample queries) or accept the aggregate
form.

Bounded-output: ≤ contig_length × n_samples rows in the worst case
(every-base depth change), but realistic long-read RNA-seq exhibits
~10–100× compression because most positions sit inside exons where
depth is locally constant.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.schemas.quant import COVERAGE_TABLE


# Sentinel sample_id when no read_to_sample mapping is supplied.
# COVERAGE_TABLE.sample_id is non-nullable int64; -1 is a documented
# "depth not stratified by sample" marker, distinct from any real
# Samples row id (which start from 0 / positive).
_UNSTRATIFIED_SAMPLE_ID = -1


def _contig_id_lookup(contigs: pa.Table) -> dict[str, int]:
    return {
        str(name): int(cid)
        for name, cid in zip(
            contigs.column("name").to_pylist(),
            contigs.column("contig_id").to_pylist(),
            strict=True,
        )
    }


def build_pileup(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    contigs: pa.Table,
    *,
    read_to_sample: dict[str, int] | None = None,
    primary_only: bool = True,
) -> pa.Table:
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
    read_to_sample : dict[str, int] | None
        Optional ``read_id → sample_id`` mapping. When provided, the
        output is stratified per sample. When ``None``, all reads
        contribute to a single ``sample_id = -1`` partition.
    primary_only : bool, default True
        Drop secondary / supplementary alignments before pile-up.
        Keeps depth interpretable (each read counts once per
        position) at the cost of dropping the chimeric tail.

    Returns
    -------
    pa.Table conforming to ``COVERAGE_TABLE``. Sorted by
    ``(contig_id, sample_id, start)``. Zero-depth intervals between
    runs are NOT emitted (callers compute them as the gaps between
    rows on the same contig+sample).
    """
    if alignment_blocks.num_rows == 0 or alignments.num_rows == 0:
        return COVERAGE_TABLE.empty_table()

    if primary_only:
        primary_mask = pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
        alignments = alignments.filter(primary_mask)
        if alignments.num_rows == 0:
            return COVERAGE_TABLE.empty_table()

    # Hash-join alignment_blocks × alignments on alignment_id; secondary
    # / supplementary blocks fall away.
    primary = alignments.select(["alignment_id", "read_id", "ref_name"])
    joined = alignment_blocks.join(primary, keys="alignment_id", join_type="inner")
    if joined.num_rows == 0:
        return COVERAGE_TABLE.empty_table()

    name_to_id = _contig_id_lookup(contigs)

    aids_aid = joined.column("alignment_id").to_pylist()
    rids = joined.column("read_id").to_pylist()
    rnames = joined.column("ref_name").to_pylist()
    rstarts = joined.column("ref_start").to_pylist()
    rends = joined.column("ref_end").to_pylist()

    # Bucket intervals per (contig_id, sample_id). For the unstratified
    # path, every read maps to _UNSTRATIFIED_SAMPLE_ID.
    by_partition: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for _, rid, rname, rs, re_ in zip(
        aids_aid, rids, rnames, rstarts, rends, strict=True
    ):
        contig_id = name_to_id.get(str(rname))
        if contig_id is None:
            continue
        if read_to_sample is None:
            sid = _UNSTRATIFIED_SAMPLE_ID
        else:
            sid = read_to_sample.get(str(rid))
            if sid is None:
                # Read has no resolved sample (e.g. demux dropped it
                # as Unknown). Skip — depth shouldn't be attributed
                # to a non-existent sample partition.
                continue
        by_partition.setdefault((int(contig_id), int(sid)), []).append(
            (int(rs), int(re_))
        )

    if not by_partition:
        return COVERAGE_TABLE.empty_table()

    rows: list[dict[str, int]] = []
    for (contig_id, sample_id) in sorted(by_partition.keys()):
        intervals = by_partition[(contig_id, sample_id)]
        rows.extend(
            _sweep_line_rle(
                intervals, contig_id=contig_id, sample_id=sample_id
            )
        )

    if not rows:
        return COVERAGE_TABLE.empty_table()
    return pa.Table.from_pylist(rows, schema=COVERAGE_TABLE)


def _sweep_line_rle(
    intervals: list[tuple[int, int]],
    *,
    contig_id: int,
    sample_id: int,
) -> list[dict[str, int]]:
    """Sweep-line over a single (contig, sample) bucket.

    Produces minimal RLE: one row per maximal interval of constant
    non-zero depth. Only emits when depth *changes*, so half-open
    abutments — read A ``[100, 200)`` immediately followed by read B
    ``[200, 300)``, with +1/-1 cancelling at pos=200 — coalesce into
    a single ``[100, 300)`` depth-1 row.

    Algorithm: convert each ``[start, end)`` interval to events
    ``(start, +1)`` + ``(end, -1)``, sort by position, walk while
    summing deltas at each unique position, and emit only on
    transitions where ``new_depth != prev_depth``.
    """
    if not intervals:
        return []

    events: list[tuple[int, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        events.append((start, +1))
        events.append((end, -1))
    if not events:
        return []
    events.sort()

    rows: list[dict[str, int]] = []
    depth = 0
    run_start: int | None = None  # start of current non-zero run
    i = 0
    n = len(events)
    while i < n:
        pos = events[i][0]
        # Sum all deltas at this exact position before moving on
        delta = 0
        while i < n and events[i][0] == pos:
            delta += events[i][1]
            i += 1
        new_depth = depth + delta
        if new_depth == depth:
            # No transition (deltas cancelled). Continue current run.
            depth = new_depth
            continue
        # Depth transition. Close the current run if one's open at
        # the prior depth, then maybe open a new run at the new depth.
        if depth > 0 and run_start is not None:
            rows.append(
                {
                    "contig_id": contig_id,
                    "sample_id": sample_id,
                    "start": run_start,
                    "end": pos,
                    "depth": depth,
                }
            )
        if new_depth > 0:
            run_start = pos
        else:
            run_start = None
        depth = new_depth
    return rows


__all__ = ["build_pileup"]
