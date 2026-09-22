"""Electing one representative read per seed group — shared by both seeders.

Round 1 needs one template per group, and "group" means different things to
the two seeders: a fold group of distinct ORFs (:mod:`.seed`) or a connected
component of reads (:mod:`.seed_kmer`). The *election* is identical either
way — rank the group's distinct cDNAs by a pluggable policy, break ties
deterministically, take the first — so it lives here and neither seeder owns
it.

Keeping it in one place is what makes ``--seed-representative`` mean the same
thing under both modes. The alternative, a second copy inside the kmer
seeder, is how the two paths would drift into ranking on different keys while
reporting the same flag value.

One policy does not survive the move to kmer seeding: ``most-5p-flank`` ranks
on ``orf_start``, and the kmer path has no ORF before election — predicting
one per unique cDNA is the 9.15M-call cost that seeder exists to avoid.
:func:`elect_representatives` raises rather than substituting silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


@dataclass(frozen=True, slots=True)
class RepCandidates:
    """The distinct cDNAs of one seed group, as parallel arrays."""

    template_length: np.ndarray  # int64
    orf_start: np.ndarray  # int64 — 5' flank length
    abundance: np.ndarray  # int64 — reads on each distinct cDNA
    # Abundance-weighted median template_length of the row's own group,
    # broadcast per row so a policy is a pure elementwise function.
    median_length: np.ndarray  # int64
    # Best `dorado_quality` among the reads sharing this exact cDNA, or -1
    # where the demux dir predates the qs:f tag. Reads sharing an exact cDNA
    # are byte-identical, so the best of them is the right representative and
    # the choice costs the template sequence nothing.
    quality: np.ndarray  # float64


#: Default floor for the quality-aware policy. The knee of the measured
#: decile curve: median error by `dorado_quality` decile runs 0.02321,
#: 0.01152, 0.00796, 0.00607, then flattens — so everything above the third
#: decile is nearly equivalent and there is nothing to buy above ~Q22.
SEED_QUALITY_FLOOR = 22.0

#: Ranks are int64 lexsort keys. Quality-failing rows are offset past every
#: possible `-template_length` so they sort strictly after every clearing row.
_RANK_FAIL_BASE = 1 << 40


RepresentativePolicy = Callable[[RepCandidates], np.ndarray]


def _rank_median_length(c: RepCandidates) -> np.ndarray:
    return np.abs(c.template_length - c.median_length)


def _rank_longest_template(c: RepCandidates) -> np.ndarray:
    return -c.template_length


def _rank_most_5p_flank(c: RepCandidates) -> np.ndarray:
    return -c.orf_start


def _rank_most_replicated(c: RepCandidates) -> np.ndarray:
    return -c.abundance


def _rank_longest_above_quality(c: RepCandidates) -> np.ndarray:
    """Longest cDNA clearing the quality floor; else the highest-quality one.

    The trade this appears to require does not exist. Over 8.3M reads,
    Spearman(length, error) is **+0.015 to +0.049** in every length stratum —
    length and accuracy are independent — so constraining one and optimising
    the other is nearly free. As selectors of read accuracy they are not
    comparable at all: the top decile by ``dorado_quality`` has median error
    0.00114 (0.25x), by ``mean_quality`` 0.00166, and by ``length`` 0.00501,
    which is **1.10x** — slightly worse than picking at random.

    So quality is the floor and length is the objective, in that order,
    because quality's benefit saturates (the decile curve flattens after the
    third) while length's cost does not: a 500 nt seed for a 3 kb transcript
    can never represent it, and since the covariance M-step made extent
    representable a truncated seed emits a truncated ORF rather than being
    silently extended.

    Honest limit: even the top quality decile has *mean* error 0.0068 against
    *median* 0.00114, so a floor makes bad seeds rarer, not absent. A
    single-read seed is one draw from that tail.
    """
    clears = c.quality >= SEED_QUALITY_FLOOR
    # Failing rows rank by quality descending, after every clearing row — so
    # "fall back to the group's best read" needs no group-level branch.
    fail = _RANK_FAIL_BASE - np.rint(c.quality * 1000.0).astype(np.int64)
    return np.where(clears, -c.template_length, fail)


REPRESENTATIVE_POLICIES: dict[str, RepresentativePolicy] = {
    # The ORF path's original default. The longest cDNA is the most likely
    # chimera / concatemer / internal-priming artifact, so electing it makes
    # the template a low-support outlier; the central length is the robust
    # estimate of the real extent. Safe ONLY because the consensus kernel can
    # now extend past the frame's ends — before that a too-short template was
    # an unrecoverable ceiling, which is why "longest" would have been the
    # defensive choice.
    "median-length": _rank_median_length,
    # The EM path's default. See :func:`_rank_longest_above_quality`.
    "longest-above-quality": _rank_longest_above_quality,
    "longest-template": _rank_longest_template,
    # Maximises the 5' flank the M-step gets to extend an ORF into.
    "most-5p-flank": _rank_most_5p_flank,
    # Most-replicated exact cDNA; ties fall through to length.
    "most-replicated": _rank_most_replicated,
}

#: Policies that need an ORF coordinate and so cannot rank a group the kmer
#: seeder produced. Named here rather than inside the seeder so the two stay
#: in step when a policy is added.
ORF_DEPENDENT_POLICIES: frozenset[str] = frozenset({"most-5p-flank"})


def resolve_policy(
    representative: str | RepresentativePolicy,
    *,
    allow_orf_dependent: bool = True,
) -> RepresentativePolicy:
    """Look a policy up by name, refusing the ones the caller cannot supply."""
    if not isinstance(representative, str):
        return representative
    if representative not in REPRESENTATIVE_POLICIES:
        raise KeyError(
            f"unknown --seed-representative {representative!r}; expected one "
            f"of {sorted(REPRESENTATIVE_POLICIES)}"
        )
    if not allow_orf_dependent and representative in ORF_DEPENDENT_POLICIES:
        usable = sorted(set(REPRESENTATIVE_POLICIES) - ORF_DEPENDENT_POLICIES)
        raise ValueError(
            f"--seed-representative {representative!r} ranks on the ORF's "
            f"position in the read, which kmer seeding does not have before "
            f"election (predicting an ORF per unique cDNA is the cost it "
            f"exists to avoid). Use one of {usable}."
        )
    return REPRESENTATIVE_POLICIES[representative]


# ── group statistics ──────────────────────────────────────────────────


def group_bounds(keys: np.ndarray) -> np.ndarray:
    """Start offset of each run in a sorted key array, plus the end sentinel."""
    if keys.shape[0] == 0:
        return np.zeros(1, dtype=np.int64)
    change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    return np.concatenate(
        [np.zeros(1, dtype=np.int64), change, np.array([keys.shape[0]], np.int64)]
    )


def weighted_median_per_group(
    starts: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Abundance-weighted median of ``values`` per group.

    ``values`` must already be sorted ascending *within* each group. Uses a
    global cumulative-weight scan plus one searchsorted per group boundary —
    no per-group Python loop.
    """
    n_groups = starts.shape[0] - 1
    if n_groups <= 0:
        return np.empty(0, dtype=np.int64)
    cw = np.cumsum(weights)
    lo, hi = starts[:-1], starts[1:]
    base = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    total = cw[hi - 1] - base
    target = base + total / 2.0
    # side='left' lands on the first row whose cumulative weight reaches half
    # the group's mass — the weighted median row.
    idx = np.searchsorted(cw, target, side="left")
    idx = np.clip(idx, lo, hi - 1)
    return values[idx]


def best_quality_per_uniq(
    reads: pa.Table, read_map: pa.Table, n_uniq: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per distinct cDNA: the best ``dorado_quality``, and which read has it.

    Returns ``(quality, read_row)`` with ``-1`` where the column is absent or
    all-null, which makes the quality-aware policy degrade to "highest of
    nothing" — every row fails the floor, so the group falls through to its
    existing abundance / length tie-breaks rather than erroring.

    ``read_map`` is row-aligned with ``reads`` by construction: ``dereplicate``
    builds it column-wise from ``reads`` itself, so read *i* of the map is read
    *i* of the table. The row index is therefore positional, and the string
    join this used to do (``pc.index_in`` over 9.4M read ids per run) bought
    nothing. :func:`assert_read_map_alignment` is the guard on that invariant.
    """
    quality = np.full(n_uniq, -1.0, dtype=np.float64)
    best_row = np.full(n_uniq, -1, dtype=np.int64)
    if "dorado_quality" not in reads.schema.names or read_map.num_rows == 0:
        return quality, best_row

    q_col = reads.column("dorado_quality")
    if q_col.null_count == reads.num_rows:
        return quality, best_row

    uid = read_map.column("uniq_id").to_numpy(zero_copy_only=False).astype(np.int64)
    q = pc.fill_null(q_col, -1.0).to_numpy(zero_copy_only=False).astype(np.float64)
    rows = np.arange(read_map.num_rows, dtype=np.int64)
    if q.shape[0] != rows.shape[0]:
        # Defensive: a caller that filtered `reads` after dereplicating it
        # has broken the positional contract, and silently mis-attributing
        # quality is worse than not attributing any.
        return quality, best_row
    qv = q[rows]

    # Highest quality first, so the first occurrence of each uniq_id is its best.
    order = np.lexsort((-qv, uid))
    su, sq, sr = uid[order], qv[order], rows[order]
    first = np.flatnonzero(np.concatenate([[True], su[1:] != su[:-1]]))
    quality[su[first]] = sq[first]
    best_row[su[first]] = sr[first]
    return quality, best_row


def assert_read_map_alignment(reads: pa.Table, read_map: pa.Table) -> None:
    """Row *i* of ``read_map`` is row *i* of ``reads`` — verify it.

    Cheap enough to call from tests, too expensive to call per run at 9.4M
    reads; :func:`best_quality_per_uniq` depends on it.
    """
    if reads.num_rows != read_map.num_rows:
        raise AssertionError(
            f"read_map has {read_map.num_rows} rows against reads' {reads.num_rows}"
        )
    if not pc.all(
        pc.equal(
            pc.cast(reads.column("read_id"), pa.string()),
            read_map.column("read_id"),
        )
    ).as_py():
        raise AssertionError("read_map is not row-aligned with reads")


# ── the election ──────────────────────────────────────────────────────


def elect_representatives(
    group_of_uniq: np.ndarray,
    n_groups: int,
    *,
    template_length: np.ndarray,
    abundance: np.ndarray,
    quality: np.ndarray,
    orf_start: np.ndarray | None = None,
    policy: str | RepresentativePolicy = "longest-above-quality",
) -> tuple[np.ndarray, np.ndarray]:
    """One elected row per group, plus each group's total read count.

    All array arguments are parallel, indexed by the same row space as
    ``group_of_uniq`` (a uniq id for the kmer seeder, a seed-ORF row for the
    ORF seeder). ``group_of_uniq`` must be dense in ``[0, n_groups)``.

    Returns ``(elected_row_per_group, n_reads_per_group)``. Groups with no
    member get ``-1`` and ``0`` — which only happens if ``n_groups`` overstates
    the labelling, and is reported rather than hidden so the caller's own
    bookkeeping error does not become a mis-elected template.
    """
    fn = resolve_policy(policy, allow_orf_dependent=orf_start is not None)
    elected = np.full(max(int(n_groups), 0), -1, dtype=np.int64)
    n_reads = np.zeros(max(int(n_groups), 0), dtype=np.int64)
    if n_groups <= 0 or group_of_uniq.shape[0] == 0:
        return elected, n_reads

    g = np.asarray(group_of_uniq, dtype=np.int64)
    tlen = np.asarray(template_length, dtype=np.int64)
    abund = np.asarray(abundance, dtype=np.int64)
    qual = np.asarray(quality, dtype=np.float64)
    ostart = (
        np.zeros(g.shape[0], dtype=np.int64)
        if orf_start is None
        else np.asarray(orf_start, dtype=np.int64)
    )

    # Pass A: sort by (group, template_length) for the group statistics. The
    # length sort is what lets the weighted median be a searchsorted rather
    # than a per-group partition.
    order = np.lexsort((tlen, g))
    starts = group_bounds(g[order])
    lo, hi = starts[:-1], starts[1:]
    g_abund = abund[order].astype(np.float64)
    median_len = weighted_median_per_group(starts, tlen[order], g_abund)
    cw = np.cumsum(g_abund)
    base = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    present_reads = (cw[hi - 1] - base).astype(np.int64)

    # Pass B: rank inside each group by the policy, then take the first row.
    group_of_row = np.repeat(np.arange(starts.shape[0] - 1), hi - lo)
    cand = RepCandidates(
        template_length=tlen[order],
        orf_start=ostart[order],
        abundance=abund[order],
        median_length=median_len[group_of_row],
        quality=qual[order],
    )
    rank = np.asarray(fn(cand), dtype=np.int64)
    # Deterministic tie-breaks: more reads, then longer cDNA, then row order.
    pick_order = np.lexsort(
        (
            np.arange(rank.shape[0]),
            -cand.template_length,
            -cand.abundance,
            rank,
            group_of_row,
        )
    )
    n_present = starts.shape[0] - 1
    first_of_group = pick_order[
        np.searchsorted(group_of_row[pick_order], np.arange(n_present))
    ]
    # `group_of_row` is dense over the groups that HAVE members; map back to
    # the caller's group ids, which may be sparser.
    present_ids = g[order][lo]
    elected[present_ids] = order[first_of_group]
    n_reads[present_ids] = present_reads
    return elected, n_reads


__all__ = [
    "ORF_DEPENDENT_POLICIES",
    "REPRESENTATIVE_POLICIES",
    "SEED_QUALITY_FLOOR",
    "RepCandidates",
    "RepresentativePolicy",
    "assert_read_map_alignment",
    "best_quality_per_uniq",
    "elect_representatives",
    "group_bounds",
    "resolve_policy",
    "weighted_median_per_group",
]
