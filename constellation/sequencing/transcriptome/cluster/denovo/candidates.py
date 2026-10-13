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

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

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


# ── containment candidates: the probe join ────────────────────────────
#
# `generate_candidates` above answers "which sequences are near-duplicates of
# a bucket's most abundant member". The template graph asks a different
# question — "which sequences does this one sit INSIDE" — and the anchor-star
# cannot answer it, for two reasons that are both structural:
#
# * A star pairs each bucket's anchor with the rest and nothing else, so a
#   family of N near-identical sequences yields ~N edges, not N(N-1)/2.
#   Measured: 299 of a 300-member family's 44,850 pairs, and a fragment shared
#   by two isoforms got an edge to only one of them.
# * It is run on a bottom-m sketch, which preserves Jaccard similarity. A
#   600-nt fragment of a 5 kb template shares 12% of the union however exact
#   the match; containment needs the sketch UNCAPPED.
#
# So the join here is driven from the SHORTER sequence. Every window of a
# contained sequence is a window of its container, so every one of its
# minimizers is one of the container's: a handful of them — "probes", spread
# along it — find every container without consulting the longer sequence's
# sketch at all. Each unordered pair is produced exactly once, shorter first,
# from the shorter one's probes, which is also what makes the reduction
# embarrassingly parallel: all rows of a pair belong to one owner, so blocks
# cut at owner boundaries never need merging.
#
# What that buys is CONTAINMENT recall. A staggered pair is found only while
# its overlap spans two of the shorter sequence's probe strata (len / 16
# each), so its recall falls with length: measured among 3-5 kb sequences and
# their 150-600 nt fragments, 7,813 of 7,813 nested pairs against 0.82 of the
# staggered pairs overlapping by 200-300 nt.
#
# Three choices that measurement made, each the opposite of the obvious one:
#
# * Probes are the SMALLEST-HASH minimizer of each position stratum, not the
#   rarest bucket. Rarest-first selects a sequence's private variants — the
#   k-mers carrying its own errors, shared with its one or two closest kin —
#   and measured recall 0.955.
# * Diagonal consistency is a WINDOW COUNT, not `dmax - dmin`. One k-mer
#   repeated elsewhere in a sequence puts a second, distant diagonal on every
#   pair it takes part in, and a global span then vetoes them all: 19 true
#   pairs of a 14-member family lost, against 0 of 39,508 with the window.
# * The caps report rather than bound. `bucket_cap` and `max_candidates` are
#   sized so that a real family never reaches them; when one does, the rows it
#   touched are named (`overflow_rows`, `truncated_rows`) so an incomplete
#   edge list is a recorded fact about a template, never a silent one.
#
# And one the window does NOT do by itself, although it looks as if it should:
# reject antisense pairs. The hash is canonical, so a sequence and its reverse
# complement share every minimizer, along an ANTI-diagonal — `pos_q + pos_x`
# is constant and the diagonal moves by twice the distance between two hits.
# Two probes within `diag_band / 2` of each other therefore share a window.
# Measured with the window alone, 16 probes and a 64-nt band: 200 of 200
# antisense pairs produced a candidate at 600 nt, 128 of 200 at 1.5 kb, 43 of
# 200 at 3 kb. So a window is also asked which of the two lines its hits
# agree on, and one that fits the anti-diagonal better is not a candidate
# (0 of 200 at every length; 0 of 214,200 sense pairs lost to it, at up to 1%
# indels) — UNLESS `min_shared` of the pair's probes sit on one diagonal,
# which is a candidate whatever else the window holds. Without that clause a
# fragment inside a tandem repeat was lost: each probe also hits the copies
# one unit either side, so the window (398, 428, 428, 458) spreads 60 along
# the diagonal axis from two probes 14 nt apart, the anti-diagonal spread is
# 32, and a pair with two probes agreeing on 428 was called antisense (47 of
# 2,000 exact 40-60 nt fragments of a 3-copy repeat). An antisense pair
# cannot meet the clause by being antisense: two hits on one diagonal AND one
# anti-diagonal are the same hit. Two limits, both left to the alignment: one
# hit lies on both lines, so a sequence short enough to own a single probe is
# not judged and its antisense copy is a candidate; and only a pair's BEST
# window is judged, so a pair with as many hits on an anti-diagonal as on its
# true diagonal and no two on one (a fold-back chimera against a fragment
# holding two probes) can be lost.


CONTAINMENT_CANDIDATE_SCHEMA = pa.schema(
    [
        # The SHORTER sequence of the pair (ties: the lower row).
        pa.field("src_row", pa.int64(), nullable=False),
        pa.field("dst_row", pa.int64(), nullable=False),
        # Hits inside the best diagonal window.
        pa.field("n_shared", pa.int32(), nullable=False),
        # Median diagonal of that window = expected start of src on dst.
        pa.field("diag", pa.int32(), nullable=False),
        # Found only through the anchor fallback: no probe that joined its
        # whole bucket saw this pair.
        pa.field("overflow", pa.bool_(), nullable=False),
    ]
)

#: Usable bits of a packed sort key. int64 is signed: a key that reaches bit
#: 63 sorts as negative, and nothing downstream would notice. Every packed key
#: in this section states its fields and checks their sum against this.
_KEY_BITS = 63
#: Widest join row sorted as ONE integer; a wider row is lexsorted instead.
#: Separate from `_KEY_BITS` only so a test can force the lexsort path on a
#: fixture small enough to check by hand.
_PACKED_ROW_BITS = _KEY_BITS


@dataclass(frozen=True, slots=True)
class ContainmentCandidates:
    """Candidate pairs plus the record of where a cap bound.

    ``table`` is ``CONTAINMENT_CANDIDATE_SCHEMA`` sorted by ``(src_row,
    dst_row)``. ``overflow_rows`` / ``truncated_rows`` are sorted unique
    int64 rows: sequences that had a probe answered by the anchor fallback,
    and sequences whose candidate list was cut at ``max_candidates``. Either
    means that sequence's edge list may be incomplete.
    """

    table: pa.Table
    stats: dict
    overflow_rows: np.ndarray
    truncated_rows: np.ndarray


@dataclass(frozen=True, slots=True)
class _JoinState:
    """What a worker reads: numpy arrays taken in the parent, shared by fork.

    Probes are sorted by ``(owner, stratum)``. A probe's join targets are
    ``count`` consecutive index entries from ``lo`` — or, for an overflow
    probe, ``count`` consecutive slots of ``anchor_entry`` (which hold index
    entries), so both kinds expand with the same arithmetic.
    """

    uniq: np.ndarray  # int32 (M,) index column
    pos: np.ndarray  # int32 (M,) index column
    anchor_entry: np.ndarray  # int64 (A,) index entries, overflow bucket by bucket
    probe_owner: np.ndarray  # int64 (P,) ascending
    probe_stratum: np.ndarray  # int64 (P,)
    probe_pos: np.ndarray  # int64 (P,)
    probe_lo: np.ndarray  # int64 (P,)
    probe_count: np.ndarray  # int64 (P,)
    probe_overflow: np.ndarray  # bool (P,)
    length_rank: np.ndarray  # int32 (n,) rank under (length asc, row asc)
    cap_rank: np.ndarray  # int64 (n,) rank under (support desc, row asc)
    probes_per_seq: int
    min_shared: int
    diag_band: int
    max_candidates: int
    diag_offset: int  # added to a diagonal so it packs as a non-negative field
    bits_row: int
    bits_diag: int
    bits_stratum: int


# Set in the parent before the pool forks and cleared in its `finally`;
# workers read it copy-on-write. A task is two integers.
_JOIN_STATE: _JoinState | None = None


def _bits(n_values: int) -> int:
    """Width of a field holding every integer in ``[0, n_values)``; >= 1."""
    return max(int(n_values) - 1, 1).bit_length()


def _key_width(what: str, *fields: int) -> int:
    """Total width of a packed key. Raises rather than let one reach the sign
    bit — and as an exception, not an ``assert``, so ``-O`` cannot remove it."""
    total = sum(fields)
    if total > _KEY_BITS:
        raise OverflowError(
            f"{what} key needs {' + '.join(map(str, fields))} = {total} bits, "
            f"and an int64 key holds {_KEY_BITS}"
        )
    return total


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    """First index of each run of equal values in a sorted, non-empty array."""
    first = np.empty(sorted_keys.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=first[1:])
    return np.flatnonzero(first)


def _descending_rank(n: int, *keys: np.ndarray) -> np.ndarray:
    """Rank of each row under ``keys`` descending (first key primary), row
    ascending as the final tie-break. Rank 0 is best."""
    columns = [np.arange(n, dtype=np.int64)]
    for key in reversed(keys):
        columns.append(-key)
    order = np.lexsort(tuple(columns))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return rank


def _orderable(values: np.ndarray, name: str) -> np.ndarray:
    """``values`` as int64 or float64, so that negating it cannot wrap."""
    arr = np.asarray(values)
    if arr.dtype.kind in "bui":
        return arr.astype(np.int64)
    if arr.dtype.kind == "f":
        out = arr.astype(np.float64)
        if np.isnan(out).any():
            raise ValueError(f"{name} contains NaN, which has no rank")
        return out
    raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")


def _empty_candidates(stats: dict) -> ContainmentCandidates:
    return ContainmentCandidates(
        table=CONTAINMENT_CANDIDATE_SCHEMA.empty_table(),
        stats=stats,
        overflow_rows=np.empty(0, dtype=np.int64),
        truncated_rows=np.empty(0, dtype=np.int64),
    )


def _plan_probes(
    index: MinimizerIndex,
    seq_len: np.ndarray,
    support: np.ndarray,
    *,
    k: int,
    probes_per_seq: int,
    bucket_cap: int,
    overflow_anchors: int,
    max_candidates: int,
    min_shared: int,
    diag_band: int,
    max_rows: int,
) -> tuple[_JoinState | None, dict]:
    """Choose every probe and size its join — the whole cost, before any of it
    is paid. Returns ``(None, stats)`` when nothing can pair.

    Parent only. The projected row count here is exact, so this is also what
    a sketch-only measurement wants: it costs the index scan and one sort, and
    no join.
    """
    uq = index.uniq_id.numpy()
    ps = index.pos.numpy()
    mh = index.mini_hash.numpy()
    m = int(mh.shape[0])
    n = int(seq_len.shape[0])

    stats = {
        "n_sequences": n,
        "n_with_probes": 0,
        "n_probes": 0,
        "n_overflow_probes": 0,
        "n_rows_projected": 0,
        "n_rows": 0,
        "n_candidates": 0,
        "n_antisense": 0,
        "n_overflow_templates": 0,
        "n_truncated_templates": 0,
        "bucket_cap": int(bucket_cap),
        "smallest_demoted_bucket": 0,
        "bucket_size_max": 0,
        "bucket_size_mean": 0.0,
        "bucket_size_biased_mean": 0.0,
        "n_buckets": 0,
    }
    if m == 0:
        return None, stats
    if int(uq.min()) < 0 or int(uq.max()) >= n:
        raise ValueError(
            f"the index names row {int(uq.max())} but seq_len has {n} rows"
        )

    bucket_start = _group_starts(mh)
    n_buckets = int(bucket_start.shape[0])
    bucket_size = np.diff(np.append(bucket_start, m))
    shared_sizes = bucket_size[bucket_size >= 2].astype(np.float64)
    stats["n_buckets"] = n_buckets
    stats["bucket_size_max"] = int(bucket_size.max())
    stats["bucket_size_mean"] = float(m / n_buckets)
    if shared_sizes.shape[0]:
        stats["bucket_size_biased_mean"] = float(
            (shared_sizes * shared_sizes).sum() / shared_sizes.sum()
        )
    del shared_sizes

    # A bucket whose entries all sit in ONE sequence — a k-mer repeated inside
    # it and found nowhere else — has size >= 2 and joins nothing. Left
    # eligible it would win its stratum on hash and spend the probe on rows
    # that are all the owner itself.
    several = np.minimum.reduceat(uq, bucket_start) != np.maximum.reduceat(
        uq, bucket_start
    )
    # 0 = joined whole, 1 = over the cap (anchor fallback), 2 = joins nothing.
    bucket_class = np.full(n_buckets, 2, dtype=np.int8)
    bucket_class[several & (bucket_size <= bucket_cap)] = 0
    bucket_class[several & (bucket_size > bucket_cap)] = 1
    del several

    is_start = np.zeros(m, dtype=bool)
    is_start[bucket_start] = True
    bucket_of = np.cumsum(is_start) - 1
    del is_start
    entry_class = bucket_class[bucket_of]
    entry = np.flatnonzero(entry_class < 2)
    if entry.shape[0] == 0:
        return None, stats

    # One probe per (owner, stratum): the smallest hash among the buckets that
    # can be joined whole, else the smallest among those over the cap. The
    # index is hash-sorted, so a bucket's id orders as its hash does, and the
    # minimum of `choice` is that rule with the position breaking a tie
    # between two copies of one k-mer in one stratum.
    #   group  = [ owner : bits_row ][ stratum : bits_stratum ]
    #   choice = [ class : 1 ][ bucket : bits_bucket ][ pos : bits_pos ]
    # Built in place: at 200M entries each of these arrays is 1.6 GB.
    bits_row = _bits(n)
    bits_stratum = _bits(probes_per_seq)
    bits_bucket = _bits(n_buckets)
    bits_pos = _bits(int(ps.max()) + 1)
    _key_width("probe group", bits_row, bits_stratum)
    _key_width("probe choice", 1, bits_bucket, bits_pos)

    choice = bucket_of[entry]
    del bucket_of
    choice |= entry_class[entry].astype(np.int64) << bits_bucket
    del entry_class
    choice <<= bits_pos
    pos = ps[entry].astype(np.int64)
    choice |= pos
    group = uq[entry].astype(np.int64)  # the owner, until the stratum joins it
    del entry
    span = np.maximum(seq_len[group] - k + 1, 1)
    if bool((pos >= span).any()):
        raise ValueError(
            "a minimizer starts past len - k: seq_len or k does not describe "
            "the sequences this index was built from"
        )
    pos *= probes_per_seq
    pos //= span
    np.minimum(pos, probes_per_seq - 1, out=pos)  # now the stratum
    del span
    group <<= bits_stratum
    group |= pos
    del pos

    order = np.argsort(group)
    group = group[order]
    choice = choice[order]
    del order
    starts = _group_starts(group)
    chosen = np.minimum.reduceat(choice, starts)
    probe_key = group[starts]
    del group, choice, starts

    probe_owner = probe_key >> bits_stratum
    probe_stratum = probe_key & ((1 << bits_stratum) - 1)
    probe_overflow = (chosen >> (bits_bucket + bits_pos)) == 1
    probe_bucket = (chosen >> bits_pos) & ((1 << bits_bucket) - 1)
    probe_pos = chosen & ((1 << bits_pos) - 1)
    n_probes = int(probe_key.shape[0])
    del probe_key, chosen

    # The row budget. Every count is known here, so the budget is met by
    # construction rather than discovered by running out of memory.
    size = bucket_size[probe_bucket]
    anchored = np.minimum(size, overflow_anchors)
    projected = int(np.where(probe_overflow, anchored, size).sum())
    if projected > max_rows:
        # Largest first, and a whole size class at a time, so what was demoted
        # is one threshold ("every bucket of at least this many entries")
        # rather than an arbitrary subset of equal-sized buckets.
        can_demote = ~probe_overflow & (size > overflow_anchors)
        sizes, n_of_size = np.unique(size[can_demote], return_counts=True)
        if sizes.shape[0]:
            saved = np.cumsum((n_of_size * (sizes - overflow_anchors))[::-1])
            cut = int(np.searchsorted(saved, projected - max_rows, side="left"))
            threshold = int(sizes[::-1][min(cut, sizes.shape[0] - 1)])
            probe_overflow = probe_overflow | (can_demote & (size >= threshold))
            projected = int(np.where(probe_overflow, anchored, size).sum())
            stats["smallest_demoted_bucket"] = threshold

    # Anchor lists: the best `overflow_anchors` entries of each bucket that
    # some probe reaches through the fallback.
    probe_lo = bucket_start[probe_bucket]
    probe_count = np.where(probe_overflow, anchored, size)
    anchor_entry = np.empty(0, dtype=np.int64)
    if bool(probe_overflow.any()):
        anchor_rank = _descending_rank(n, support, seq_len)
        over_bucket = np.unique(probe_bucket[probe_overflow])
        over_size = bucket_size[over_bucket]
        first = np.cumsum(over_size) - over_size
        total = int(over_size.sum())
        member = np.repeat(bucket_start[over_bucket] - first, over_size)
        member += np.arange(total, dtype=np.int64)
        which = np.repeat(np.arange(over_bucket.shape[0]), over_size)
        order = np.lexsort((ps[member], anchor_rank[uq[member]], which))
        # `which` is the primary key and was already ascending, so each bucket
        # still occupies [first, first + size) after the sort.
        place = np.arange(total, dtype=np.int64) - np.repeat(first, over_size)
        anchor_entry = member[order][place < overflow_anchors]
        n_anchor = np.minimum(over_size, overflow_anchors)
        anchor_lo = np.cumsum(n_anchor) - n_anchor
        probe_lo[probe_overflow] = anchor_lo[
            np.searchsorted(over_bucket, probe_bucket[probe_overflow])
        ]

    max_len = int(seq_len.max())
    stats["n_probes"] = n_probes
    stats["n_with_probes"] = int(_group_starts(probe_owner).shape[0])
    stats["n_overflow_probes"] = int(probe_overflow.sum())
    stats["n_rows_projected"] = projected

    state = _JoinState(
        uniq=uq,
        pos=ps,
        anchor_entry=anchor_entry,
        probe_owner=probe_owner,
        probe_stratum=probe_stratum,
        probe_pos=probe_pos,
        probe_lo=probe_lo,
        probe_count=probe_count,
        probe_overflow=probe_overflow,
        # "Shorter, ties to the lower row" as ONE comparison per join row.
        length_rank=_descending_rank(n, -seq_len).astype(np.int32),
        cap_rank=_descending_rank(n, support),
        probes_per_seq=int(probes_per_seq),
        min_shared=int(min_shared),
        diag_band=int(diag_band),
        max_candidates=int(max_candidates),
        diag_offset=max_len,
        bits_row=bits_row,
        # A diagonal lies in (-max_len, max_len); offset by max_len it is a
        # non-negative field, and the band is added to it inside the window
        # search, so the field has to hold that sum without carrying into the
        # pair bits above it.
        bits_diag=_bits(2 * max_len + int(diag_band) + 1),
        bits_stratum=bits_stratum,
    )
    return state, stats


def _cut_blocks(
    probe_owner: np.ndarray, probe_count: np.ndarray, chunk_rows: int
) -> list[tuple[int, int]]:
    """Probe ranges, cut at owner boundaries, each expanding to about
    ``chunk_rows`` rows. An owner larger than that is a block by itself."""
    owner_first = _group_starts(probe_owner)
    n_owners = int(owner_first.shape[0])
    rows_before = np.concatenate(
        ([0], np.cumsum(np.add.reduceat(probe_count, owner_first)))
    )
    bound = np.append(owner_first, probe_owner.shape[0])
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n_owners:
        j = int(np.searchsorted(rows_before, rows_before[i] + chunk_rows, "right")) - 1
        j = min(max(j, i + 1), n_owners)
        blocks.append((int(bound[i]), int(bound[j])))
        i = j
    return blocks


def _reduce_block(p0: int, p1: int) -> tuple:
    """Expand probes ``[p0, p1)`` into join rows and reduce them to pairs.

    numpy only — this runs below the fork. Returns ``(src, dst, n_shared,
    diag, overflow, truncated_src, n_rows, n_antisense)`` with the pairs
    sorted by ``(src, dst)``.
    """
    st = _JOIN_STATE
    assert st is not None
    per_seq = st.probes_per_seq
    owner = st.probe_owner[p0:p1]
    count = st.probe_count[p0:p1]
    over = st.probe_overflow[p0:p1]
    stratum_of_probe = st.probe_stratum[p0:p1]

    # Everything per-row is built by `np.repeat` over the probes (a
    # sequential write) or gathered once from the index; the join is bound by
    # memory traffic, not arithmetic.
    first = np.cumsum(count) - count
    member = np.repeat(st.probe_lo[p0:p1] - first, count)
    member += np.arange(member.shape[0], dtype=np.int64)
    any_over = bool(over.any())
    if any_over:
        through_anchor = np.repeat(over, count)
        member[through_anchor] = st.anchor_entry[member[through_anchor]]
        del through_anchor
    dst = st.uniq[member]

    # Shorter first, ties to the lower row. The rows a probe finds in a
    # sequence shorter than its owner are that pair seen from the wrong side;
    # the shorter one's own probes produce it.
    keep = st.length_rank[dst] > np.repeat(st.length_rank[owner], count)
    # Rows kept per probe. `first` is strictly increasing: a probe joins at
    # least one entry, so no reduceat segment is empty.
    n_kept = np.add.reduceat(keep, first, dtype=np.int64)
    dst = dst[keep].astype(np.int64)
    dst_pos = st.pos[member[keep]].astype(np.int64)
    del member, keep
    if dst.shape[0] == 0:
        none = np.empty(0, dtype=np.int64)
        return (
            none,
            none,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=bool),
            none,
            0,
            0,
        )

    # Owners are renumbered within the block, which is what lets the whole
    # row fit one key at any realistic scale: 1.6M sequences of up to 15 kb,
    # 16 probes each and ~16k owners to an 8M-row block is 14 + 21 + 15 + 4
    # = 54 bits.
    owner_first = _group_starts(owner)
    owners = owner[owner_first]
    n_local = int(owners.shape[0])
    local_of_probe = np.repeat(
        np.arange(n_local, dtype=np.int64),
        np.diff(np.append(owner_first, p1 - p0)),
    )
    # A diagonal is dst_pos - pos_q; the probe's half of it, offset so the
    # field is never negative, is a per-probe constant.
    diag_of_probe = st.diag_offset - st.probe_pos[p0:p1]

    bits_local = _bits(n_local)
    _key_width("pair", bits_local, st.bits_row)
    mask_stratum = (1 << st.bits_stratum) - 1
    mask_diag = (1 << st.bits_diag) - 1
    # (owner, stratum) names the probe and so its position: deduplicating on
    # (src, dst, stratum, diag) IS deduplicating on (src, dst, pos_q, diag),
    # in 4 bits instead of 24.
    row_bits = bits_local + st.bits_row + st.bits_diag + st.bits_stratum
    if row_bits <= _PACKED_ROW_BITS:
        #   [ local ][ dst : bits_row ][ diag : bits_diag ][ stratum ]
        # The probe's fields are packed per probe and repeated; the row's two
        # are added into their own fields, and the diag field cannot carry
        # because the sum IS the offset diagonal.
        low = st.bits_diag + st.bits_stratum
        key = np.repeat(
            (local_of_probe << (st.bits_row + low))
            | (diag_of_probe << st.bits_stratum)
            | stratum_of_probe,
            n_kept,
        )
        dst <<= low
        key += dst
        dst_pos <<= st.bits_stratum
        key += dst_pos
        del dst, dst_pos
        key.sort()
        fresh = np.empty(key.shape[0], dtype=bool)
        fresh[0] = True
        np.not_equal(key[1:], key[:-1], out=fresh[1:])
        key = key[fresh]
        stratum = key & mask_stratum
        key >>= st.bits_stratum
        diag = key & mask_diag
        pair = key >> st.bits_diag
        del key, fresh
    else:
        # Too wide for one key (hundreds of millions of sequences, or
        # megabase diagonals): same order, by lexsort.
        pair = np.repeat(local_of_probe << st.bits_row, n_kept)
        pair |= dst
        diag = np.repeat(diag_of_probe, n_kept)
        diag += dst_pos
        stratum = np.repeat(stratum_of_probe, n_kept)
        del dst, dst_pos
        order = np.lexsort((stratum, diag, pair))
        pair, diag, stratum = pair[order], diag[order], stratum[order]
        del order
        fresh = np.empty(pair.shape[0], dtype=bool)
        fresh[0] = True
        fresh[1:] = (
            (pair[1:] != pair[:-1])
            | (diag[1:] != diag[:-1])
            | (stratum[1:] != stratum[:-1])
        )
        pair, diag, stratum = pair[fresh], diag[fresh], stratum[fresh]
        del fresh
    n_rows = int(pair.shape[0])

    # The best window of each pair: for every row, how many of the pair's
    # rows have a diagonal in [d, d + band]. Rows are sorted by (pair, diag),
    # so that is one binary search per row on
    #   [ pair index : bits_pair ][ diag : bits_diag ]
    # and the band cannot carry out of the diag field (see `bits_diag`).
    pair_first = _group_starts(pair)
    n_pairs = int(pair_first.shape[0])
    pair_size = np.diff(np.append(pair_first, n_rows))
    _key_width("window", _bits(n_pairs), st.bits_diag)
    window = np.repeat(np.arange(n_pairs, dtype=np.int64), pair_size)
    window <<= st.bits_diag
    window |= diag
    row = np.arange(n_rows, dtype=np.int64)
    hits = np.searchsorted(window, window + st.diag_band, side="right") - row
    del window
    n_shared = np.maximum.reduceat(hits, pair_first)
    # First window on ties; its median is the lower one, so `diag` is always
    # a diagonal that was observed.
    lead = np.minimum.reduceat(
        np.where(hits == np.repeat(n_shared, pair_size), row, n_rows), pair_first
    )
    del hits, row
    median = diag[lead + (n_shared - 1) // 2] - st.diag_offset

    pair_local = pair[pair_first] >> st.bits_row
    pair_dst = pair[pair_first] & ((1 << st.bits_row) - 1)
    # Which probe a row came from, as an index into per-probe tables.
    slot_of_probe = local_of_probe * per_seq + stratum_of_probe
    slot = (pair >> st.bits_row) * per_seq + stratum
    del pair, stratum

    # A sequence too short to own `min_shared` probes is held to what it has.
    owned = np.bincount(local_of_probe, minlength=n_local)
    needed = np.minimum(st.min_shared, owned[pair_local])

    # Sense or antisense. Only a window spanning more than one diagonal can
    # be antisense: rows are distinct in (probe, diag), so two on ONE diagonal
    # are two probes that agree on it.
    spread = diag[lead + n_shared - 1] - diag[lead]
    antisense = np.zeros(n_pairs, dtype=bool)
    unsure = np.flatnonzero(spread > 0)
    if unsure.shape[0]:
        # The most rows any one diagonal of the pair holds. Enough probes
        # agreeing on a diagonal is what a candidate IS, whatever else its
        # window holds: in a tandem repeat every probe also hits the
        # neighbouring copies, one unit either side, and those hits spread
        # the window along the diagonal axis further than the probes are
        # apart on the sequence — which is all "fits the anti-diagonal
        # better" measures.
        run_start = np.ones(n_rows, dtype=bool)
        np.not_equal(diag[1:], diag[:-1], out=run_start[1:])
        run_start[pair_first] = True
        run_first = np.flatnonzero(run_start)
        run_size = np.diff(np.append(run_first, n_rows))
        agreed = np.maximum.reduceat(np.repeat(run_size, run_size), pair_first)
        del run_start, run_first, run_size
        pos_of_slot = np.zeros(n_local * per_seq, dtype=np.int64)
        pos_of_slot[slot_of_probe] = st.probe_pos[p0:p1]
        # pos_q + pos_x, up to the constant offset `diag` carries. One spare
        # element so a window ending at the last row has a valid upper bound.
        anti = np.empty(n_rows + 1, dtype=np.int64)
        anti[:n_rows] = diag + 2 * pos_of_slot[slot]
        anti[n_rows] = 0
        bound = np.empty(2 * unsure.shape[0], dtype=np.int64)
        bound[0::2] = lead[unsure]
        bound[1::2] = lead[unsure] + n_shared[unsure]
        anti_spread = (
            np.maximum.reduceat(anti, bound)[0::2]
            - np.minimum.reduceat(anti, bound)[0::2]
        )
        antisense[unsure] = (anti_spread < spread[unsure]) & (
            agreed[unsure] < needed[unsure]
        )
        del anti, bound, agreed
    del diag

    if any_over:
        probe_is_over = np.zeros(n_local * per_seq, dtype=bool)
        probe_is_over[slot_of_probe] = over
        only_anchor = np.logical_and.reduceat(probe_is_over[slot], pair_first)
    else:
        only_anchor = np.zeros(n_pairs, dtype=bool)
    del slot

    passed = n_shared >= needed
    n_antisense = int((passed & antisense).sum())
    passed &= ~antisense
    pair_local = pair_local[passed]
    pair_dst = pair_dst[passed]
    n_shared = n_shared[passed]
    median = median[passed]
    only_anchor = only_anchor[passed]

    truncated = np.empty(0, dtype=np.int64)
    n_dst = np.bincount(pair_local, minlength=n_local)
    if bool((n_dst > st.max_candidates).any()):
        crowded = np.flatnonzero(n_dst[pair_local] > st.max_candidates)
        ranked = crowded[
            np.lexsort(
                (
                    st.cap_rank[pair_dst[crowded]],
                    -n_shared[crowded],
                    pair_local[crowded],
                )
            )
        ]
        run_first = _group_starts(pair_local[ranked])
        run_size = np.diff(np.append(run_first, ranked.shape[0]))
        place = np.arange(ranked.shape[0]) - np.repeat(run_first, run_size)
        kept = np.ones(pair_local.shape[0], dtype=bool)
        kept[ranked[place >= st.max_candidates]] = False
        truncated = owners[n_dst > st.max_candidates]
        pair_local = pair_local[kept]
        pair_dst = pair_dst[kept]
        n_shared = n_shared[kept]
        median = median[kept]
        only_anchor = only_anchor[kept]

    return (
        owners[pair_local],
        pair_dst,
        n_shared.astype(np.int32),
        median.astype(np.int32),
        only_anchor,
        truncated,
        n_rows,
        n_antisense,
    )


def generate_containment_candidates(
    index: MinimizerIndex,
    seq_len: np.ndarray,
    support: np.ndarray,
    *,
    k: int = 19,
    probes_per_seq: int = 16,
    bucket_cap: int = 20_480,
    overflow_anchors: int = 32,
    max_candidates: int = 20_480,
    min_shared: int = 2,
    diag_band: int = 64,
    chunk_rows: int = 8_000_000,
    max_rows: int = 32_000_000_000,
    threads: int = 1,
) -> ContainmentCandidates:
    """Candidate (shorter, longer) pairs by joining each sequence's probes
    against the whole index.

    ``index`` must be UNCAPPED (``extract_minimizers(..., max_per_seq=None)``)
    and built with this ``k``; ``seq_len`` and ``support`` are indexed by the
    row the index calls ``uniq_id``. ``support`` ranks the anchors a
    too-large bucket falls back to and breaks ties when a candidate list is
    cut.

    Per sequence, one probe per position stratum (``probes_per_seq`` of them):
    the smallest-hash minimizer whose bucket holds another sequence and at
    most ``bucket_cap`` entries. A stratum with only larger buckets probes the
    smallest-hash one of those against its ``overflow_anchors`` best entries
    by ``(support desc, length desc, row asc)`` instead. A pair is kept when
    ``min_shared`` of its hits — or as many probes as its shorter sequence
    owns, if fewer — fall inside one ``diag_band``-wide diagonal window, and
    either that many sit on ONE diagonal or the window's hits agree on the
    diagonal at least as well as on the anti-diagonal (an antisense pair
    shares every canonical minimizer too).
    At most ``max_candidates`` are kept per shorter sequence, by ``(n_shared
    desc, support desc, row asc)``.

    ``max_rows`` is a backstop, not a tuning knob: when the join would exceed
    it, whole bucket-size classes are demoted to the anchor fallback, largest
    first, and ``stats["smallest_demoted_bucket"]`` records where that
    stopped. If even that cannot fit, the join runs at what demotion reached
    and ``n_rows_projected`` says so.

    Output does not depend on ``chunk_rows`` or ``threads``.

    ``stats`` — plain ints and floats:

    * ``n_sequences``, ``n_with_probes``, ``n_probes``, ``n_overflow_probes``
    * ``n_rows_projected`` — join rows expanded, after any demotion
    * ``n_rows`` — of those, the distinct rows on the shorter sequence's side
    * ``n_candidates``, ``n_overflow_templates``, ``n_truncated_templates``
    * ``n_antisense`` — pairs with enough hits that were dropped as antisense
    * ``bucket_cap``, ``smallest_demoted_bucket`` (0 = nothing demoted)
    * ``n_buckets``, ``bucket_size_max``, ``bucket_size_mean`` — over every
      bucket, singletons included
    * ``bucket_size_biased_mean`` — ``sum(s^2) / sum(s)`` over buckets of at
      least two entries: the bucket size a random shared entry sits in, which
      is what a probe costs
    """
    global _JOIN_STATE

    for name, value, floor in (
        ("k", k, 1),
        ("probes_per_seq", probes_per_seq, 1),
        ("bucket_cap", bucket_cap, 2),
        ("overflow_anchors", overflow_anchors, 1),
        ("max_candidates", max_candidates, 1),
        ("min_shared", min_shared, 1),
        ("diag_band", diag_band, 0),
        ("chunk_rows", chunk_rows, 1),
        ("max_rows", max_rows, 1),
    ):
        if int(value) != value or value < floor:
            raise ValueError(f"{name} must be an integer >= {floor}, got {value!r}")
    seq_len = np.ascontiguousarray(seq_len, dtype=np.int64)
    support = _orderable(support, "support")
    if seq_len.ndim != 1 or support.shape != seq_len.shape:
        raise ValueError(
            f"seq_len and support must be 1-d and the same length, got "
            f"{seq_len.shape} and {support.shape}"
        )

    state, stats = _plan_probes(
        index,
        seq_len,
        support,
        k=int(k),
        probes_per_seq=int(probes_per_seq),
        bucket_cap=int(bucket_cap),
        overflow_anchors=int(overflow_anchors),
        max_candidates=int(max_candidates),
        min_shared=int(min_shared),
        diag_band=int(diag_band),
        max_rows=int(max_rows),
    )
    if state is None:
        return _empty_candidates(stats)

    blocks = _cut_blocks(state.probe_owner, state.probe_count, int(chunk_rows))
    _JOIN_STATE = state
    try:
        if threads <= 1 or len(blocks) <= 1:
            parts = [_reduce_block(p0, p1) for p0, p1 in blocks]
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=min(int(threads), len(blocks)), mp_context=ctx
            ) as ex:
                futs = [ex.submit(_reduce_block, p0, p1) for p0, p1 in blocks]
                try:
                    # Submission order, not completion order: blocks ascend
                    # by owner, so concatenating them in order IS the
                    # (src, dst) sort.
                    parts = [fut.result() for fut in futs]
                finally:
                    # Leaving the `with` waits for every block still queued.
                    # After a failure that is the whole join, run for a
                    # result nobody will read.
                    for fut in futs:
                        fut.cancel()
    finally:
        _JOIN_STATE = None

    src, dst, n_shared, diag, overflow, truncated = (
        np.concatenate([part[i] for part in parts]) for i in range(6)
    )
    overflow_rows = np.unique(state.probe_owner[state.probe_overflow])
    truncated_rows = np.unique(truncated)
    stats["n_rows"] = int(sum(part[6] for part in parts))
    stats["n_antisense"] = int(sum(part[7] for part in parts))
    stats["n_candidates"] = int(src.shape[0])
    stats["n_overflow_templates"] = int(overflow_rows.shape[0])
    stats["n_truncated_templates"] = int(truncated_rows.shape[0])

    table = pa.table(
        {
            "src_row": pa.array(src, type=pa.int64()),
            "dst_row": pa.array(dst, type=pa.int64()),
            "n_shared": pa.array(n_shared, type=pa.int32()),
            "diag": pa.array(diag, type=pa.int32()),
            "overflow": pa.array(overflow, type=pa.bool_()),
        },
        schema=CONTAINMENT_CANDIDATE_SCHEMA,
    )
    return ContainmentCandidates(
        table=table,
        stats=stats,
        overflow_rows=overflow_rows,
        truncated_rows=truncated_rows,
    )


__all__ = [
    "generate_candidates",
    "CANDIDATE_SCHEMA",
    "generate_containment_candidates",
    "ContainmentCandidates",
    "CONTAINMENT_CANDIDATE_SCHEMA",
]
