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
    group_idx = pa.array(read_of_group)

    return pa.RecordBatch.from_arrays(
        [
            # read_id is materialised exactly once per round, on the winners —
            # never decoded over the ~51x larger hit stream.
            pc.take(reads.read_id, group_idx).combine_chunks().cast(pa.string()),
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


__all__ = ["EM_ASSIGNMENT_TABLE", "assign_block", "assign_blocks"]
