"""Candidate-pair generation from the sorted minimizer index.

Walk the hash-sorted minimizer index bucket by bucket (a bucket = all
occurrences of one minimizer hash). Suppress high-frequency
"stop-word" minimizers, then within each surviving bucket connect the
**highest-abundance** entry (the anchor — Bayesian: high abundance ⇒
most likely error-free, so it makes the natural cluster centroid) to
every other entry. Each such (anchor, other) sharing emits a diagonal
``d = anchor_pos − other_pos``; two sequences are a candidate pair iff
they share ``≥ min_shared`` minimizers whose diagonals are tightly
clustered (``diag_max − diag_min ≤ diag_span_max``) — the diagonal
consistency check that distinguishes a real co-linear match from
sequences sharing a few minimizers by chance.

The candidate graph is the union of these anchor-centered stars: a
low-abundance error variant connects to the high-abundance true
sequence it derived from, which is exactly the edge greedy set-cover
needs to collapse it.

All vectorized; the only per-element work is numpy index manipulation
over the minimizer occurrences (the same O(M) scale as the index sort).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
    MinimizerIndex,
)


CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("uniq_a", pa.int64(), nullable=False),  # uniq_a < uniq_b
        pa.field("uniq_b", pa.int64(), nullable=False),
        pa.field("n_shared", pa.int32(), nullable=False),
    ]
)

_POS_BITS = 31


def generate_candidates(
    index: MinimizerIndex,
    abundance: np.ndarray,
    *,
    min_shared: int = 2,
    diag_span_max: int = 20,
    max_bucket_frac: float = 0.01,
    min_bucket_floor: int = 50_000,
) -> pa.Table:
    """Generate candidate pairs from a sorted minimizer index.

    ``abundance`` is an int64 array indexed by ``uniq_id`` (i.e.
    ``uniq_table.abundance`` in row order). Returns a ``CANDIDATE_SCHEMA``
    table of distinct ``(uniq_a, uniq_b)`` pairs with their shared-
    minimizer count.

    The stop-word cap (``max(max_bucket_frac · U, min_bucket_floor)``)
    skips only genuinely ubiquitous (low-complexity) minimizers — the
    anchor-star is already linear in bucket size, and the verify gate
    supplies precision, so the floor is set high enough that small or
    expression-concentrated datasets are never pruned.
    """
    mh = index.mini_hash.numpy()
    uq = index.uniq_id.numpy().astype(np.int64, copy=False)
    ps = index.pos.numpy().astype(np.int64, copy=False)
    n = mh.shape[0]
    if n < 2:
        return CANDIDATE_SCHEMA.empty_table()

    n_uniq = abundance.shape[0]
    max_bucket = max(int(max_bucket_frac * n_uniq), int(min_bucket_floor))

    # Bucket id per minimizer occurrence (index is hash-sorted).
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = mh[1:] != mh[:-1]
    bucket_id = np.cumsum(change) - 1
    n_buckets = int(bucket_id[-1]) + 1
    bucket_size = np.bincount(bucket_id, minlength=n_buckets)

    ent_bsize = bucket_size[bucket_id]
    keep = (ent_bsize >= 2) & (ent_bsize <= max_bucket)
    if not keep.any():
        return CANDIDATE_SCHEMA.empty_table()

    b = bucket_id[keep]
    u = uq[keep]
    p = ps[keep]
    ab = abundance[u]

    # Order entries by (bucket asc, abundance desc, uniq asc, pos asc) so
    # the first row of each bucket is its anchor (deterministic tie-break).
    order = np.lexsort((p, u, -ab, b))
    bs = b[order]
    us = u[order]
    pps = p[order]

    first = np.empty(bs.shape[0], dtype=bool)
    first[0] = True
    first[1:] = bs[1:] != bs[:-1]
    anchor_row = np.maximum.accumulate(np.where(first, np.arange(bs.shape[0]), 0))
    anc_uniq = us[anchor_row]
    anc_pos = pps[anchor_row]

    # Non-anchor entries whose sequence differs from the anchor's.
    other = (~first) & (us != anc_uniq)
    if not other.any():
        return CANDIDATE_SCHEMA.empty_table()
    a_uniq = anc_uniq[other]
    o_uniq = us[other]
    diag = anc_pos[other] - pps[other]

    # Canonicalize to uniq_a < uniq_b (flip diagonal sign with the swap).
    swap = a_uniq > o_uniq
    ua = np.where(swap, o_uniq, a_uniq)
    ub = np.where(swap, a_uniq, o_uniq)
    dg = np.where(swap, -diag, diag)

    pairkey = (ua << _POS_BITS) + ub  # ub < 2**31
    po = np.argsort(pairkey, kind="stable")
    pk = pairkey[po]
    dgs = dg[po]

    seg_first = np.empty(pk.shape[0], dtype=bool)
    seg_first[0] = True
    seg_first[1:] = pk[1:] != pk[:-1]
    seg_starts = np.flatnonzero(seg_first)
    counts = np.diff(np.append(seg_starts, pk.shape[0]))
    dmin = np.minimum.reduceat(dgs, seg_starts)
    dmax = np.maximum.reduceat(dgs, seg_starts)

    sel = (counts >= int(min_shared)) & ((dmax - dmin) <= int(diag_span_max))
    sel_pk = pk[seg_starts][sel]
    out_a = (sel_pk >> _POS_BITS).astype(np.int64)
    out_b = (sel_pk & ((1 << _POS_BITS) - 1)).astype(np.int64)

    return pa.table(
        {
            "uniq_a": pa.array(out_a),
            "uniq_b": pa.array(out_b),
            "n_shared": pa.array(counts[sel].astype(np.int32)),
        },
        schema=CANDIDATE_SCHEMA,
    )


__all__ = ["generate_candidates", "CANDIDATE_SCHEMA"]
