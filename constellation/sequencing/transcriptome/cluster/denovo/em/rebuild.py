"""The final merge's consensus rebuild: one pooled PWM per survivor.

Between rounds a merge keeps the deepest member and the next M-step rebuilds
its consensus from the union of the reads. The final merge has no M-step
after it, so for a while it was held to EXACT pairs and kept the longest
member's own consensus and ORF. Measured at 9.4M reads that left 7-13k
round-6 splits unmerged, and the longest member was the wrong form more
often than the deepest (its start overshot the annotated TSS in 41% of 914
groups against 24%; ledger #52, #55). So the final merge now follows the
run's edit cap, keeps the deepest member, and does here what the next
M-step would have done: every read of every node in the group is aligned to
the survivor's consensus and one consensus is built from the pool — the
M-step's own kernel, without the split — with the ORF called on that.

The reads are RE-ALIGNED rather than carried over by offset. Over the shared
span the members agree to within ``merge_max_edits``, so an absorbed node's
alignments would transfer to the survivor almost everywhere; "almost" is an
alignment that is wrong at exactly the columns the merge tolerated, and the
E-step's edlib aligner (:func:`~.realign.align_finalist`) gives the exact
answer at the cost of one infix alignment per read, over a few thousand
groups.

Workers open the corpus themselves by path, as the M-step's do; what crosses
the fork is one survivor's consensus and its read rows.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    MemberSpec,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
    pooled_node,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
    MStepParams,
    _reads,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.realign import (
    align_finalist,
)

#: The node columns a rebuild rewrites. ``n_reads`` and ``node_weight`` are
#: the merge's (summed over the group) and are not touched.
REBUILT_COLUMNS: tuple[str, ...] = (
    "consensus",
    "protein",
    "orf_start",
    "orf_end",
    "n_inserted_columns",
    "n_extended_5p",
    "n_extended_3p",
    "n_trimmed_5p",
    "n_trimmed_3p",
    "n_members_used",
    "subsample_fraction",
)

#: Rebuilds submitted and not yet collected, per worker.
_IN_FLIGHT = 4

#: A placement must cover this fraction of whichever is shorter, the read or
#: the survivor. In the E-step a read reaches the aligner through a minimap2
#: chain, which is its proof of existence; here the whole read is placed
#: infix with no chain, and a read from the wrong group can place a handful
#: of bases somewhere by chance at identity 1.0.
_MIN_PLACED_FRACTION = 0.5


def align_to_survivor(read: str, consensus: str, *, identity_floor: float):
    """``(cigar, t_start, q_start)`` of ``read`` on ``consensus``, or ``None``
    below the floor, placed over less than half of the shorter of the two,
    or not placed at all. The E-step's aligner, run with no chain to guide
    it: the whole read is placed infix, anchor-trimmed, extended at both
    ends and score-trimmed, exactly as a shortlisted read is in a
    ``--estep-aligner edlib`` round."""
    hit = align_finalist(
        read,
        consensus,
        q_start=0,
        q_end=len(read),
        t_start=0,
        t_end=len(consensus),
    )
    if hit is None or hit.aln_len <= 0:
        return None
    if hit.n_match / hit.aln_len < identity_floor:
        return None
    placed = min(hit.q_end - hit.q_start, hit.t_end - hit.t_start)
    if placed < _MIN_PLACED_FRACTION * min(len(read), len(consensus)):
        return None
    return hit.cigar, int(hit.t_start), int(hit.q_start)


def rebuild_one(
    row: int,
    consensus: str,
    read_rows: np.ndarray,
    *,
    template_id: int,
    corpus_path: str,
    params: MStepParams,
    identity_floor: float,
) -> tuple[int, dict | None, int, float]:
    """One survivor. Returns ``(row, rebuilt or None, n_skipped, fraction)``.

    ``rebuilt`` is the new value of every :data:`REBUILT_COLUMNS` column;
    ``None`` when no read could be placed. Module-level so it pickles by
    name; numpy, edlib and the consensus kernel only below the fork.
    """
    reads = _reads(corpus_path)
    rows = np.asarray(read_rows, dtype=np.int64)
    fraction = 1.0
    cap = int(params.max_members_per_template)
    if cap and rows.shape[0] > cap:
        # The M-step's own rule: a uniform sample, seeded by the template so
        # a re-run agrees, never a prefix.
        rng = np.random.default_rng(int(template_id) & 0xFFFFFFFF)
        rows = np.sort(rng.choice(rows, size=cap, replace=False))
        fraction = cap / read_rows.shape[0]
    members: list[MemberSpec] = []
    n_skipped = 0
    for i, seq in enumerate(reads.take_sequences(rows)):
        placed = align_to_survivor(seq, consensus, identity_floor=identity_floor)
        if placed is None:
            n_skipped += 1
            continue
        cigar, t_start, q_start = placed
        members.append(
            MemberSpec(
                member_seq=seq,
                weight=1.0,
                cigar=cigar,
                centroid_is_query=False,
                ref_start=t_start,
                member_start=q_start,
                member_id=i,
            )
        )
    node = pooled_node(
        consensus,
        members,
        template_id=int(template_id),
        min_aa_length=params.min_aa_length,
        fold_insertions=params.fold_insertions,
        min_extension_support=params.min_extension_support,
    )
    if node is None:
        return row, None, n_skipped, fraction
    return (
        row,
        {
            "consensus": node.consensus,
            "protein": node.protein,
            "orf_start": int(node.orf_start),
            "orf_end": int(node.orf_end),
            "n_inserted_columns": int(node.n_inserted_columns),
            "n_extended_5p": int(node.n_extended_5p),
            "n_extended_3p": int(node.n_extended_3p),
            "n_trimmed_5p": int(node.n_trimmed_5p),
            "n_trimmed_3p": int(node.n_trimmed_3p),
            "n_members_used": len(members),
            "subsample_fraction": float(fraction),
        },
        n_skipped,
        fraction,
    )


def _pair_key(parent: np.ndarray, hap: np.ndarray) -> np.ndarray:
    """``(parent_template_id, haplotype_id)`` as one sortable structured key."""
    key = np.empty(parent.shape[0], dtype=[("p", np.int64), ("h", np.int64)])
    key["p"] = parent
    key["h"] = hap
    return key


def _read_rows_of(
    nodes: pa.Table, membership: pa.Table, rows: np.ndarray
) -> list[np.ndarray]:
    """The membership read rows of each node in ``rows``."""
    pid = nodes.column("parent_template_id").to_numpy(zero_copy_only=False)
    hap = nodes.column("haplotype_id").to_numpy(zero_copy_only=False).astype(np.int64)
    m_pid = membership.column("parent_template_id").to_numpy(zero_copy_only=False)
    m_hap = membership.column("haplotype_id").to_numpy(zero_copy_only=False)
    m_row = membership.column("read_row").to_numpy(zero_copy_only=False).astype(np.int64)
    m_key = _pair_key(m_pid, m_hap.astype(np.int64))
    order = np.argsort(m_key, kind="stable")
    sorted_key, sorted_row = m_key[order], m_row[order]
    want = _pair_key(pid[rows], hap[rows])
    lo = np.searchsorted(sorted_key, want, side="left")
    hi = np.searchsorted(sorted_key, want, side="right")
    return [sorted_row[a:b] for a, b in zip(lo.tolist(), hi.tolist())]


def rebuild_survivors(
    nodes: pa.Table,
    membership: pa.Table,
    rows: np.ndarray,
    corpus_path: str,
    *,
    params: MStepParams,
    identity_floor: float,
    threads: int = 1,
    log=None,
) -> tuple[pa.Table, dict]:
    """Rebuild the consensus of the nodes at ``rows`` from their membership.

    ``nodes`` and ``membership`` are the final merge's outputs — the
    absorbed nodes gone and their membership re-keyed to the survivors — and
    ``rows`` the survivors that absorbed something. Returns the node table
    with those rows rewritten, and counts: ``n_rebuilt``, ``n_rebuild_failed``
    (no read could be placed; the survivor's own consensus stands) and
    ``n_rebuild_reads_skipped``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    counts = {"n_rebuilt": 0, "n_rebuild_failed": 0, "n_rebuild_reads_skipped": 0}
    if rows.shape[0] == 0 or nodes.num_rows == 0:
        return nodes, counts
    say = log or (lambda _m: None)
    read_rows = _read_rows_of(nodes, membership, rows)
    template_id = nodes.column("parent_template_id").to_numpy(zero_copy_only=False)
    consensus = nodes.column("consensus")
    kwargs = {
        "corpus_path": str(corpus_path),
        "params": params,
        "identity_floor": float(identity_floor),
    }
    tasks = [
        (int(r), consensus[int(r)].as_py(), read_rows[i], int(template_id[r]))
        for i, r in enumerate(rows.tolist())
    ]

    results: dict[int, dict] = {}

    def take(outcome) -> None:
        row, rebuilt, n_skipped, _fraction = outcome
        counts["n_rebuild_reads_skipped"] += int(n_skipped)
        if rebuilt is None:
            counts["n_rebuild_failed"] += 1
        else:
            counts["n_rebuilt"] += 1
            results[row] = rebuilt

    workers = max(1, min(int(threads), len(tasks)))
    if workers == 1:
        for row, seq, member_rows, tid in tasks:
            take(rebuild_one(row, seq, member_rows, template_id=tid, **kwargs))
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            flying: deque = deque()
            try:
                for row, seq, member_rows, tid in tasks:
                    flying.append(
                        pool.submit(
                            rebuild_one, row, seq, member_rows, template_id=tid, **kwargs
                        )
                    )
                    if len(flying) >= workers * _IN_FLIGHT:
                        take(flying.popleft().result())
                while flying:
                    take(flying.popleft().result())
            finally:
                for future in flying:
                    future.cancel()
    say(
        f"final output: rebuilt {counts['n_rebuilt']:,} merged consensus"
        f"{'es' if counts['n_rebuilt'] != 1 else ''} from pooled reads"
        + (
            f" ({counts['n_rebuild_failed']:,} kept their own)"
            if counts["n_rebuild_failed"]
            else ""
        )
    )
    if not results:
        return nodes, counts

    # Replace by mask rather than through a Python list of every row's
    # value: the table is every final node, and only a few thousand of them
    # were rebuilt.
    rebuilt_rows = np.array(sorted(results), dtype=np.int64)
    mask = np.zeros(nodes.num_rows, dtype=bool)
    mask[rebuilt_rows] = True
    out = nodes
    for name in REBUILT_COLUMNS:
        field = nodes.schema.field(name)
        values = pa.array([results[int(r)][name] for r in rebuilt_rows], type=field.type)
        column = nodes.column(name)
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        out = out.set_column(
            nodes.schema.get_field_index(name),
            name,
            pc.replace_with_mask(column, pa.array(mask), values),
        )
    return out.cast(nodes.schema), counts


__all__ = ["REBUILT_COLUMNS", "align_to_survivor", "rebuild_one", "rebuild_survivors"]
