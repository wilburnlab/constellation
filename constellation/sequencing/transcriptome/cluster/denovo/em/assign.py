"""The E-step's reducer: PAF blocks in, bounded assignment batches out.

This is where :mod:`.paf_scan`, :mod:`.scheduler` and :mod:`.likelihood` meet.
Per block:

1. group boundaries from ``np.diff`` on the integer ``read_row`` — no string
   comparison, because the corpus FASTA names reads by row index;
2. admit on ``n_match / aln_len``, both eager in the scan;
3. rank — round 1 on ORF replication and seed quality, round 2+ on the
   likelihood over differing positions;
4. decode the output fields **for the winners only**, which is the whole
   reason the scanner defers nine of its twelve columns.

Nothing read-cardinality is accumulated in Python: batches are emitted as
they are reduced. The rule it replaces built ``a_rows`` / ``c_rows`` as
Python tuple lists until one final Arrow conversion, which the last review
measured at +70.2 MiB / 3.16 s for a 20,000-read x 10-template fixture and
which is unbounded in the read count.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import register_schema
from constellation.sequencing.transcriptome.cluster.denovo.em import scheduler as sched
from constellation.sequencing.transcriptome.cluster.denovo.em.likelihood import (
    read_template_loglik,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.paf_scan import HitBlock
from constellation.sequencing.transcriptome.cluster.denovo.variants import ErrorModel


EM_ASSIGNMENT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        # The corpus row. Every downstream join keys on this rather than on
        # read_id: it is an int32 the PAF already carried, so the M-step gets
        # a read's sequence by `pc.take` instead of a dict lookup.
        pa.field("read_row", pa.int32(), nullable=False),
        # -1 when no candidate cleared p_floor. Emitted rather than dropped so
        # the rejection rate is visible per round.
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("template_row", pa.int32(), nullable=False),
        pa.field("round", pa.int32(), nullable=False),
        pa.field("weight", pa.float32(), nullable=False),
        pa.field("as_score", pa.int32(), nullable=False),
        pa.field("as_delta", pa.int32(), nullable=False),
        pa.field("logl", pa.float32(), nullable=True),
        pa.field("logl_delta", pa.float32(), nullable=True),
        pa.field("n_hits", pa.int32(), nullable=False),
        pa.field("n_admitted", pa.int32(), nullable=False),
        # True when this read's candidate list reached minimap2's -N cap, so
        # the pool was TRUNCATED and the ranking arbitrated over an arbitrary
        # subset. A non-trivial rate makes the round's assignments unsound —
        # see the saturation flag in the diagnostics.
        pa.field("candidate_cap_hit", pa.bool_(), nullable=False),
        pa.field("offset_5p", pa.int32(), nullable=False),
        pa.field("q_start", pa.int32(), nullable=False),
        pa.field("q_end", pa.int32(), nullable=False),
        pa.field("t_start", pa.int32(), nullable=False),
        pa.field("t_end", pa.int32(), nullable=False),
        pa.field("cigar", pa.large_string(), nullable=True),
        pa.field("sample_id", pa.int64(), nullable=True),
    ],
    metadata={b"schema_name": b"EmAssignmentTable"},
)

register_schema("EmAssignmentTable", EM_ASSIGNMENT_TABLE)

_WINNER_FIELDS = ("q_start", "q_end", "t_start", "t_end", "q_len", "t_len")


def _group_bounds(read_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(group_ptr, read_of_group)`` from a contiguously-grouped array."""
    if read_row.size == 0:
        return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64)
    changes = np.flatnonzero(read_row[1:] != read_row[:-1]) + 1
    starts = np.concatenate([[0], changes]).astype(np.int64)
    ptr = np.concatenate([starts, [read_row.size]]).astype(np.int64)
    return ptr, read_row[starts]


def assign_block(
    hb: HitBlock,
    *,
    store,
    reads,
    round_index: int,
    p_floor: float = 0.97,
    delta_logl: float = 5.0,
    support_ratio: float = 20.0,
    near_tie_z: float = 2.0,
    error_rate: float = 0.01,
    error_model: ErrorModel | None = None,
    minimap2_n: int = 500,
    keep_cigars: bool = True,
) -> pa.RecordBatch:
    """Reduce one :class:`HitBlock` to one assignment row per read."""
    ptr, read_of_group = _group_bounds(hb.read_row)
    n_groups = read_of_group.size
    if n_groups == 0:
        return pa.RecordBatch.from_pylist([], schema=EM_ASSIGNMENT_TABLE)

    sizes = np.diff(ptr)
    admitted, n_admitted = sched.admit_candidates(
        hb.n_match, hb.aln_len, ptr, p_floor=p_floor
    )

    logl = np.full(hb.read_row.size, -np.inf, dtype=np.float64)
    logl_delta = np.full(hb.read_row.size, np.inf, dtype=np.float64)

    if round_index <= 1:
        winner = sched.rank_round1(
            admitted,
            hb.template_row,
            ptr,
            orf_replication=store.orf_replication,
            seed_read_quality=store.seed_read_quality,
        )
    else:
        # AS shortlists; the likelihood decides. Confining the CIGAR decode to
        # the shortlist is what keeps this affordable: AS and logL agree except
        # where the error model disagrees with uniform weighting, which can
        # only flip the answer when the AS gap is small.
        short = sched.shortlist_for_likelihood(
            admitted,
            hb.as_score,
            ptr,
            span=hb.aln_len,
            z=near_tie_z,
            error_rate=error_rate,
        )
        sel = np.flatnonzero(short)
        if sel.size:
            fields = hb.int_fields(sel, _WINNER_FIELDS)
            logl[sel] = read_template_loglik(
                hb.cigars(sel),
                t_start=fields["t_start"],
                q_start=fields["q_start"],
                q_end=fields["q_end"],
                q_len=fields["q_len"],
                template_row=hb.template_row[sel],
                store=store,
                model=error_model,
            )
        read_len_h = hb.aln_len.astype(np.float64)
        winner, logl_delta = sched.rank_likelihood(
            short,
            hb.template_row,
            ptr,
            logl=logl,
            support=store.node_weight,
            read_len=read_len_h,
            template_len=store.lengths().astype(np.float64),
            delta_logl=delta_logl,
            support_ratio=support_ratio,
        )

    has = winner != sched.UNASSIGNED
    slot = np.where(has, winner, 0)

    # as_delta against the group's best ADMITTED score, for continuity with
    # the bench diagnostics.
    masked_as = np.where(admitted, hb.as_score.astype(np.float64), -np.inf)
    best_as = np.maximum.reduceat(masked_as, ptr[:-1])
    best_as = np.where((sizes > 0) & np.isfinite(best_as), best_as, 0.0)

    fields = hb.int_fields(slot, _WINNER_FIELDS)
    cig = hb.cigars(slot) if keep_cigars else pa.nulls(n_groups, pa.large_string())
    if not has.all():
        cig = pa.array(
            [c if h else None for c, h in zip(cig.to_pylist(), has.tolist())],
            pa.large_string(),
        )

    t_row = np.where(has, hb.template_row[slot], -1).astype(np.int32)
    t_id = np.where(has, store.template_id[hb.template_row[slot]], -1).astype(np.int64)

    return pa.RecordBatch.from_arrays(
        [
            # read_id is materialised exactly once per round, on the winners —
            # never decoded over the ~51x larger hit stream. And taken
            # CHUNK-LOCALLY: `pc.take` on a chunked column concatenates the
            # whole column first, so this would pull the corpus's entire
            # read_id array resident once per block.
            reads.take_read_ids(read_of_group),
            pa.array(read_of_group.astype(np.int32)),
            pa.array(t_id),
            pa.array(t_row),
            pa.array(np.full(n_groups, int(round_index), dtype=np.int32)),
            pa.array(np.where(has, 1.0, 0.0).astype(np.float32)),
            pa.array(np.where(has, hb.as_score[slot], 0).astype(np.int32)),
            pa.array(np.where(has, best_as - hb.as_score[slot], 0).astype(np.int32)),
            pa.array(
                np.where(has & np.isfinite(logl[slot]), logl[slot], np.nan).astype(
                    np.float32
                )
            ),
            pa.array(
                np.where(
                    has & np.isfinite(logl_delta[slot]), logl_delta[slot], np.nan
                ).astype(np.float32)
            ),
            pa.array(sizes.astype(np.int32)),
            pa.array(n_admitted.astype(np.int32)),
            pa.array(sizes >= int(minimap2_n) + 1),
            pa.array(
                np.where(has, fields["t_start"] - fields["q_start"], 0).astype(np.int32)
            ),
            pa.array(np.where(has, fields["q_start"], 0).astype(np.int32)),
            pa.array(np.where(has, fields["q_end"], 0).astype(np.int32)),
            pa.array(np.where(has, fields["t_start"], 0).astype(np.int32)),
            pa.array(np.where(has, fields["t_end"], 0).astype(np.int32)),
            cig,
            pa.array(reads.sample_id[read_of_group].astype(np.int64))
            if reads.sample_id.size
            else pa.nulls(n_groups, pa.int64()),
        ],
        schema=EM_ASSIGNMENT_TABLE,
    )


def assign_blocks(blocks: Iterable[HitBlock], **kwargs) -> Iterator[pa.RecordBatch]:
    """Reduce a stream of blocks, emitting one bounded batch per block."""
    for hb in blocks:
        batch = assign_block(hb, **kwargs)
        if batch.num_rows:
            yield batch


# minimap2 flags for aligning reads to templates.
#
# `-p 0.05` and a large `-N` are load-bearing, and for different reasons.
# `-p`: the stock 0.8 suppresses any hit scoring below 80% of the best, which
# is structurally what a nested or 5'-truncated proteoform of a longer
# template IS, so earlier E-steps ran blind to them.
# `-N`: the pool must contain EVERY template within p_floor of the read,
# because both rankers assume it — round 1 picks the most-replicated
# candidate and round 2+ the most likely one, and minimap2 truncates by
# SCORE. Among near-clone templates score differences are 1-2 units, so a
# truncated pool is close to an arbitrary sample of the set being ranked.
# That is not a degraded answer; it is an answer to a different question.
# `--eqx` is required too: without it cg:Z is all `M`, mismatches fold into
# matches, and both the identity gate and the likelihood read every alignment
# as perfect.
TEMPLATE_MINIMAP2_ARGS: tuple[str, ...] = (
    "-x",
    "map-ont",
    "-c",
    "--eqx",
    "--secondary=yes",
    "-p",
    "0.05",
)


def run_em_estep(
    templates_fasta,
    reads_fasta,
    *,
    store,
    reads,
    output_dir,
    round_index: int,
    threads: int = 8,
    minimap2_n: int = 500,
    index_batch_size: str = "16G",
    block_bytes: int = 128 << 20,
    extra_minimap2_args: tuple[str, ...] = (),
    progress=None,
    **assign_kwargs,
) -> dict:
    """Align every read to every template and stream assignments to Parquet.

    Writes ``output_dir/part-NNNNN.parquet`` per PAF block and returns the
    round's counters. Nothing is written to disk by minimap2 and nothing
    read-cardinality is held in RAM: the hit stream is reduced as it arrives.
    """
    from pathlib import Path

    import pyarrow.parquet as pq

    from constellation.sequencing.align.minimap2 import minimap2_stream
    from constellation.sequencing.transcriptome.cluster.denovo.em.paf_scan import (
        iter_hit_blocks,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _m: None)

    _check_single_index_part(store, index_batch_size)

    args = (
        *TEMPLATE_MINIMAP2_ARGS,
        "-N",
        str(int(minimap2_n)),
        "-I",
        index_batch_size,
        *extra_minimap2_args,
    )
    log(f"round {round_index}: minimap2 {' '.join(args)}")
    stream = minimap2_stream(
        Path(templates_fasta), [Path(reads_fasta)], args=args, threads=threads
    )
    blocks = iter_hit_blocks(
        stream, n_templates=store.n_templates, block_bytes=block_bytes
    )

    # Which corpus rows minimap2 actually reported on. `n_reads` is None when
    # the caller did not supply a corpus size (tests on hand-built stores).
    n_reads = getattr(reads, "n_reads", None)
    seen = np.zeros(int(n_reads), dtype=bool) if n_reads else None

    stats = {
        "n_reads_seen": 0,
        "n_assigned": 0,
        "n_unassigned": 0,
        "n_cap_hit": 0,
        "n_hits": 0,
        "n_dropped_strand": 0,
        "n_dropped_template": 0,
    }
    shard = 0
    for hb in blocks:
        stats["n_hits"] += len(hb)
        stats["n_dropped_strand"] += hb.n_dropped_strand
        stats["n_dropped_template"] += hb.n_dropped_template
        batch = assign_block(
            hb,
            store=store,
            reads=reads,
            round_index=round_index,
            minimap2_n=minimap2_n,
            **assign_kwargs,
        )
        if not batch.num_rows:
            continue
        stats["n_reads_seen"] += batch.num_rows
        assigned = pc.sum(pc.greater_equal(batch.column("template_id"), 0)).as_py() or 0
        stats["n_assigned"] += int(assigned)
        stats["n_unassigned"] += batch.num_rows - int(assigned)
        stats["n_cap_hit"] += int(
            pc.sum(batch.column("candidate_cap_hit")).as_py() or 0
        )
        if seen is not None:
            seen[batch.column("read_row").to_numpy(zero_copy_only=False)] = True
        pq.write_table(
            pa.Table.from_batches([batch], schema=EM_ASSIGNMENT_TABLE),
            output_dir / f"part-{shard:05d}.parquet",
        )
        shard += 1

    # Reads minimap2 never reported at all — no hit above its chaining
    # threshold — produce no PAF line and would otherwise disappear from the
    # accounting entirely: not assigned, not unassigned, no row. That silently
    # understates the rejection rate, which is the one number the admission
    # floor exists to make visible. Emit them explicitly.
    n_unmapped = 0
    if seen is not None:
        missing = np.flatnonzero(~seen)
        if missing.size:
            n_unmapped = int(missing.size)
            pq.write_table(
                pa.Table.from_batches(
                    [_unassigned_batch(missing, reads, round_index)],
                    schema=EM_ASSIGNMENT_TABLE,
                ),
                output_dir / f"part-{shard:05d}.parquet",
            )
            shard += 1

    stats["n_unmapped"] = n_unmapped
    stats["n_reads_seen"] += n_unmapped
    stats["n_unassigned"] += n_unmapped
    stats["n_shards"] = shard
    stats["cap_hit_fraction"] = (
        stats["n_cap_hit"] / stats["n_reads_seen"] if stats["n_reads_seen"] else 0.0
    )
    return stats


def _unassigned_batch(rows: np.ndarray, reads, round_index: int) -> pa.RecordBatch:
    """One ``template_id = -1`` row per corpus read minimap2 never reported."""
    n = rows.size
    zero32 = pa.array(np.zeros(n, dtype=np.int32))
    return pa.RecordBatch.from_arrays(
        [
            reads.take_read_ids(rows),
            pa.array(rows.astype(np.int32)),
            pa.array(np.full(n, -1, dtype=np.int64)),
            pa.array(np.full(n, -1, dtype=np.int32)),
            pa.array(np.full(n, int(round_index), dtype=np.int32)),
            pa.array(np.zeros(n, dtype=np.float32)),
            zero32,
            zero32,
            pa.array(np.full(n, np.nan, dtype=np.float32)),
            pa.array(np.full(n, np.nan, dtype=np.float32)),
            zero32,
            zero32,
            pa.array(np.zeros(n, dtype=bool)),
            zero32,
            zero32,
            zero32,
            zero32,
            zero32,
            pa.nulls(n, pa.large_string()),
            pa.array(reads.sample_id[rows].astype(np.int64))
            if reads.sample_id.size
            else pa.nulls(n, pa.int64()),
        ],
        schema=EM_ASSIGNMENT_TABLE,
    )


def _check_single_index_part(store, index_batch_size: str) -> None:
    """Refuse a multi-part minimap2 index. Two separate things depend on it.

    minimap2 with a multi-part index runs the queries once per part and
    applies ``-p``, ``-N`` and the primary/secondary call *within* each part,
    so (a) a read is emitted once per part and its mass counted more than
    once, and (b) a query's hits are no longer contiguous in the output —
    which is the assumption the block reducer's group detection rests on.
    Merging parts correctly needs a two-pass shuffle; refusing is the honest
    alternative.

    Note this is now the ONLY ceiling. The separate hard-coded 3 Gb
    `max_template_bases` guard is gone: it was a second, unstated scoping
    number that fired ~5x below where this correctness argument bites, and at
    9.4M reads it forced `--min-seed-reads 2`, which erases minority
    proteoforms before they can be tested at all.
    """
    total = int(store.lengths().sum())
    limit = _parse_size(index_batch_size)
    if limit < total:
        raise ValueError(
            f"-I is {index_batch_size} ({limit / 1e9:.3f} Gb) but the template "
            f"set is {total / 1e9:.3f} Gb, so minimap2 would build a multi-part "
            "index. It then runs the queries once per part and applies -p, -N "
            "and the primary/secondary call WITHIN each part, so a read is "
            "emitted once per part (its mass counted more than once) and its "
            "hits are no longer contiguous in the output, which the block "
            "reducer's grouping depends on. Raise --index-batch-size above the "
            "template total."
        )


def _parse_size(value: str | int) -> int:
    """minimap2's ``-I`` size grammar: a number with an optional K/M/G suffix."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("empty -I value")
    mult = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}.get(text[-1].upper())
    return int(float(text[:-1]) * mult) if mult else int(float(text))


__all__ = [
    "EM_ASSIGNMENT_TABLE",
    "TEMPLATE_MINIMAP2_ARGS",
    "assign_block",
    "assign_blocks",
    "run_em_estep",
]
