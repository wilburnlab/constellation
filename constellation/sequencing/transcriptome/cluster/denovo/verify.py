"""Alignment verification of candidate pairs (edlib HW) + CIGAR cache.

Each candidate pair is aligned exactly once, **shorter → longer** (query
= shorter sequence, ref = longer), in edlib HW (infix) mode so the ref
ends are free — the longer sequence's 5'/3' flanks beyond the shorter
are the length overhang. A pair is accepted iff

    edit_distance ≤ (1 − identity) · len(shorter)      (≈ identity gate)
    5' overhang ≤ max_5p   AND   3' overhang ≤ max_3p  (length gate)

The CIGAR is emitted here (``task="path"`` is ~free over score-only) and
cached in the returned table so the consensus stage never re-aligns —
edlib runs exactly once per edge. The cache is stored canonically as
``(uniq_short, uniq_long, ref_start, cigar)``; consensus normalizes the
orientation onto whichever sequence is the cluster centroid.

Parallelized with a fork-based process pool: the unique-sequence list is
shared via copy-on-write (set as a module global before the pool forks),
so worker tasks pass only integer pair indices — no per-task sequence
pickling.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa

from constellation.sequencing.transcriptome.cluster.denovo._cigar import cigar_stats


ACCEPTED_ALIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("uniq_short", pa.int64(), nullable=False),
        pa.field("uniq_long", pa.int64(), nullable=False),
        pa.field("ref_start", pa.int32(), nullable=False),  # short's start on long
        pa.field("cigar", pa.string(), nullable=False),  # query=short → ref=long
        pa.field("edit_distance", pa.int32(), nullable=False),
        pa.field("n_match", pa.int32(), nullable=False),
        pa.field("n_mismatch", pa.int32(), nullable=False),
        pa.field("n_insert", pa.int32(), nullable=False),
        pa.field("n_delete", pa.int32(), nullable=False),
        pa.field("overhang_5p", pa.int32(), nullable=False),
        pa.field("overhang_3p", pa.int32(), nullable=False),
    ]
)

# Set in the parent before the pool forks; workers read it copy-on-write.
_VERIFY_SEQS: list[str] | None = None


def _order_pair(a: int, b: int, seqs: list[str]) -> tuple[int, int]:
    """Deterministic (short, long) ordering: shorter sequence is query;
    ties (equal length) broken by uniq_id so verify + consensus agree."""
    la, lb = len(seqs[a]), len(seqs[b])
    if la < lb or (la == lb and a < b):
        return a, b
    return b, a


def _verify_one(
    a: int, b: int, *, identity: float, max_5p: int, max_3p: int
) -> tuple | None:
    """Align one candidate pair; return an accepted-record tuple or None."""
    import edlib

    seqs = _VERIFY_SEQS
    assert seqs is not None
    short, long = _order_pair(a, b, seqs)
    q, r = seqs[short], seqs[long]
    if not q or not r:
        return None
    budget = int((1.0 - identity) * len(q))
    res = edlib.align(q, r, mode="HW", task="path", k=budget)
    ed = res["editDistance"]
    if ed < 0:
        return None
    locs = res["locations"]
    if not locs:
        return None
    start, end_incl = locs[0]
    ref_start = int(start)
    ref_end = int(end_incl) + 1
    overhang_5p = ref_start
    overhang_3p = len(r) - ref_end
    if overhang_5p > max_5p or overhang_3p > max_3p:
        return None
    cigar = res["cigar"] or ""
    nm, nx, ni, nd = cigar_stats(cigar)
    return (
        short,
        long,
        ref_start,
        cigar,
        int(ed),
        nm,
        nx,
        ni,
        nd,
        overhang_5p,
        overhang_3p,
    )


def _verify_chunk(
    pairs: np.ndarray, identity: float, max_5p: int, max_3p: int
) -> list[tuple]:
    out: list[tuple] = []
    for i in range(pairs.shape[0]):
        rec = _verify_one(
            int(pairs[i, 0]),
            int(pairs[i, 1]),
            identity=identity,
            max_5p=max_5p,
            max_3p=max_3p,
        )
        if rec is not None:
            out.append(rec)
    return out


def verify_candidates(
    candidates: pa.Table,
    seqs: list[str],
    *,
    identity: float = 0.98,
    max_5p: int = 30,
    max_3p: int = 30,
    threads: int = 1,
    chunk_size: int = 20_000,
) -> pa.Table:
    """Verify candidate pairs; return accepted alignments (with CIGARs).

    ``seqs`` is the unique-sequence list indexed by ``uniq_id``.
    """
    global _VERIFY_SEQS
    n = candidates.num_rows
    if n == 0:
        return ACCEPTED_ALIGNMENT_SCHEMA.empty_table()

    pairs = np.column_stack(
        [
            candidates.column("uniq_a").to_numpy(zero_copy_only=False),
            candidates.column("uniq_b").to_numpy(zero_copy_only=False),
        ]
    ).astype(np.int64)

    _VERIFY_SEQS = seqs
    records: list[tuple] = []
    try:
        if threads <= 1:
            records = _verify_chunk(pairs, identity, max_5p, max_3p)
        else:
            chunks = [pairs[i : i + chunk_size] for i in range(0, n, chunk_size)]
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as ex:
                futs = [
                    ex.submit(_verify_chunk, c, identity, max_5p, max_3p)
                    for c in chunks
                ]
                for fut in futs:
                    records.extend(fut.result())
    finally:
        _VERIFY_SEQS = None

    if not records:
        return ACCEPTED_ALIGNMENT_SCHEMA.empty_table()

    cols = list(zip(*records))
    return pa.table(
        {
            "uniq_short": pa.array(cols[0], type=pa.int64()),
            "uniq_long": pa.array(cols[1], type=pa.int64()),
            "ref_start": pa.array(cols[2], type=pa.int32()),
            "cigar": pa.array(cols[3], type=pa.string()),
            "edit_distance": pa.array(cols[4], type=pa.int32()),
            "n_match": pa.array(cols[5], type=pa.int32()),
            "n_mismatch": pa.array(cols[6], type=pa.int32()),
            "n_insert": pa.array(cols[7], type=pa.int32()),
            "n_delete": pa.array(cols[8], type=pa.int32()),
            "overhang_5p": pa.array(cols[9], type=pa.int32()),
            "overhang_3p": pa.array(cols[10], type=pa.int32()),
        },
        schema=ACCEPTED_ALIGNMENT_SCHEMA,
    )


__all__ = ["verify_candidates", "ACCEPTED_ALIGNMENT_SCHEMA"]
