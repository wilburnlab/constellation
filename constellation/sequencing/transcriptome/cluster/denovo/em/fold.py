"""Stage 1b — folding: collapse error variants of the same ORF.

Runs Constellation's own Layer-0/1 kernels (``extract_minimizers`` →
``generate_candidates`` → ``verify_candidates``) on the **ORF nucleotides**
with unbounded end tolerance, classifies every accepted pair from the verify
output columns, and groups the rule-1 edges by abundance-ordered radius-1
greedy set cover.

Unbounded ends are the point: under edlib HW the whole (shorter) query always
aligns inside the ref, so a contained fragment ORF passes the *identity* gate
and has to be rejected by **classification**, not by an overhang threshold.
One verify pass therefore serves both rules.

Three things the design docs state that are not quite right, and which this
module implements as corrected:

1. An ORF string runs ATG→stop inclusive, so its length is **always** a
   multiple of 3 and ``Δlen % 3 == 0`` holds identically. Since
   ``len_long − len_short = oh5 + oh3 + n_delete − n_insert``, rule 1's
   ``(n_insert − n_delete) % 3 == 0`` clause reduces to a test on the
   *terminal overhangs*. It is kept — that test is meaningful — but the
   discriminating power is ``|Δlen| ≤ 9``, i.e. ±3 codons.
2. Rule 2's "shorter fully contained in longer" is **vacuous** under HW.
   The clause that actually discriminates is ``n_insert == 0``: the shorter
   contributes no bases the longer lacks.
3. ``verify._order_pair`` breaks length ties by ``uniq_id``, and under
   ``|Δlen| ≤ 9`` the ``Δlen == 0`` case is modal — so the tie-break fires
   constantly. Rule 1 is symmetric under the short/long swap; rule 2 is not,
   and is written against ``(uniq_short, uniq_long)`` explicitly with the
   abundance comparison done separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import numpy as np
import pyarrow as pa

from constellation.sequencing.transcriptome.cluster.denovo._cigar import (
    indel_runs,
    parse_cigar,
)
from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
    generate_candidates,
)
from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
    greedy_set_cover,
)
from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
    extract_minimizers,
)
from constellation.sequencing.transcriptome.cluster.denovo.verify import (
    verify_candidates,
)


class FoldRule(IntEnum):
    """How a verified ORF pair relates."""

    SEPARATE = 0
    ERROR_VARIANT = 1  # rule 1 — same proteoform, sequencing error
    FRAMESHIFT = 2  # rule 2 — one net-frameshifting indel in the shared span


@dataclass(frozen=True, slots=True)
class FoldResult:
    group_of_orf: np.ndarray  # (O,) int64 — dense group id per seed ORF
    group_rep_orf: np.ndarray  # (G,) int64 — the representative ORF per group
    group_n_reads: np.ndarray  # (G,) int64 — reads summed over the group
    rule_of_edge: np.ndarray  # (E,) int8 — FoldRule per accepted pair
    # Per group, variant columns declared on the representative's ORF
    # coordinates by rule-2 collapses. Empty unless fold_frameshifts is on.
    declared_variants: list[np.ndarray]
    n_separate_contained: int  # suffix-ORF rejections, for the report


def _first_indel_offsets(cigar: str, ref_start: int) -> tuple[int, int]:
    """``(offset_on_short, offset_on_long)`` of the first indel run.

    The CIGAR is ``query = short → ref = long`` per ``verify``: ``I`` consumes
    the short only, ``D`` the long only.
    """
    spos, lpos = 0, ref_start
    for length, op in parse_cigar(cigar):
        if op in ("=", "X", "M"):
            spos += length
            lpos += length
        elif op == "I":
            return spos, lpos
        elif op == "D":
            return spos, lpos
    return -1, -1


def classify_pair(
    *,
    len_short: int,
    len_long: int,
    n_mismatch: int,
    n_insert: int,
    n_delete: int,
    cigar: str,
    max_len_delta: int = 9,
    max_fs_mismatch: int = 3,
    allow_frameshift: bool = False,
) -> FoldRule:
    """Classify one verified ORF pair. Scalar reference implementation."""
    net = n_insert - n_delete
    if abs(len_long - len_short) <= max_len_delta and net % 3 == 0:
        return FoldRule.ERROR_VARIANT
    if (
        allow_frameshift
        and n_insert == 0
        and n_mismatch <= max_fs_mismatch
        and net % 3 != 0
        and indel_runs(cigar) == (0, 1)
    ):
        return FoldRule.FRAMESHIFT
    # Everything else stays separate — including a contained pair with no
    # indel, which is a suffix ORF from a downstream ATG. At the ORF level a
    # 5' truncation and an alternative start are indistinguishable; the
    # E-step decides them against the full cDNA (the Akap4 840/849 case).
    return FoldRule.SEPARATE


def classify_pairs(
    accepted: pa.Table,
    seq_len: np.ndarray,
    *,
    max_len_delta: int = 9,
    max_fs_mismatch: int = 3,
    allow_frameshift: bool = False,
) -> np.ndarray:
    """Vectorized :func:`classify_pair` over a verify-output table.

    Returns an ``(E,) int8`` array of :class:`FoldRule` values. The numeric
    predicates are evaluated on whole columns; the CIGAR-dependent
    ``indel_runs`` check runs only on the handful of rows that survive
    everything else.
    """
    n = accepted.num_rows
    if n == 0:
        return np.empty(0, dtype=np.int8)

    def col(name):
        return accepted.column(name).to_numpy(zero_copy_only=False)

    short, long = col("uniq_short"), col("uniq_long")
    n_mismatch, n_insert, n_delete = col("n_mismatch"), col("n_insert"), col("n_delete")
    ls, ll = seq_len[short], seq_len[long]
    net = (n_insert - n_delete).astype(np.int64)

    out = np.full(n, FoldRule.SEPARATE, dtype=np.int8)
    rule1 = (np.abs(ll - ls) <= max_len_delta) & (net % 3 == 0)
    out[rule1] = FoldRule.ERROR_VARIANT

    if allow_frameshift:
        maybe = (
            ~rule1
            & (n_insert == 0)
            & (n_mismatch <= max_fs_mismatch)
            & (net % 3 != 0)
        )
        idx = np.flatnonzero(maybe)
        if idx.size:
            cig = accepted.column("cigar").take(pa.array(idx)).to_pylist()
            ok = np.array([indel_runs(c) == (0, 1) for c in cig], dtype=bool)
            out[idx[ok]] = FoldRule.FRAMESHIFT
    return out



def _merge_by_representative(
    res,
    orf_nt: list[str],
    seq_len: np.ndarray,
    group_n_reads: np.ndarray,
    *,
    identity: float,
    max_len_delta: int,
    min_edit_budget: int,
    kmer: int,
    window: int,
    minimizers_per_seq: int | None,
    threads: int,
):
    """Second fold pass over the group representatives only.

    Candidate pairs come from abundance-anchored minimizer *stars*, so two
    ORFs are compared when they share an anchor. If an abundant suffix ORF
    becomes that anchor, both of its edges are rejected as suffix-containment
    and the two longer ORFs behind it may never be compared to each other —
    the anchor-star's coverage and the classifier's rejections interact.
    Re-verifying the surviving representatives closes that gap for a few
    thousand sequences instead of a few hundred thousand, and is the same
    rule-1 merge the round loop applies to consensus ORFs later.
    """
    reps = res.group_rep_orf
    if reps.shape[0] < 2:
        return res
    rep_seqs = [orf_nt[int(i)] for i in reps]
    rep_len = seq_len[reps]
    index = extract_minimizers(
        pa.array(rep_seqs, type=pa.large_string()),
        k=kmer,
        w=window,
        max_per_seq=minimizers_per_seq,
    )
    cands = generate_candidates(index, group_n_reads)
    if cands.num_rows == 0:
        return res
    accepted = verify_candidates(
        cands,
        rep_seqs,
        identity=identity,
        max_5p=2**31 - 1,
        max_3p=2**31 - 1,
        min_budget=min_edit_budget,
        threads=threads,
    )
    rules = classify_pairs(
        accepted, rep_len, max_len_delta=max_len_delta, allow_frameshift=False
    )
    keep = rules == FoldRule.ERROR_VARIANT
    if not keep.any():
        return res
    ea = accepted.column("uniq_short").to_numpy(zero_copy_only=False)[keep]
    eb = accepted.column("uniq_long").to_numpy(zero_copy_only=False)[keep]
    merged = greedy_set_cover(reps.shape[0], group_n_reads, rep_len, ea, eb)
    # Re-index the original ORFs through the merged representative groups.
    group_of_orf = merged.cluster_of[res.group_of_orf]
    new_reps = reps[merged.centroid_uniq]
    n_new = new_reps.shape[0]
    n_reads = np.zeros(n_new, dtype=np.int64)
    np.add.at(n_reads, merged.cluster_of, group_n_reads)
    return FoldResult(
        group_of_orf=group_of_orf,
        group_rep_orf=new_reps,
        group_n_reads=n_reads,
        rule_of_edge=res.rule_of_edge,
        declared_variants=[np.empty(0, dtype=np.int64) for _ in range(n_new)],
        n_separate_contained=res.n_separate_contained,
    )


def fold_orfs(
    seed: pa.Table,
    *,
    identity: float = 0.97,
    max_len_delta: int = 9,
    min_edit_budget: int = 3,
    fold_frameshifts: bool = False,
    max_fs_mismatch: int = 3,
    kmer: int = 15,
    window: int = 10,
    minimizers_per_seq: int | None = 50,
    min_shared: int = 2,
    diag_span_max: int = 20,
    merge_representatives: bool = True,
    threads: int = 1,
    progress: Callable[[str], None] | None = None,
) -> FoldResult:
    """Fold error variants of the same ORF into groups.

    ``seed`` is a ``SEED_ORF_TABLE``. Returns the grouping plus the per-edge
    classification, which the diagnostics report summarises.
    """
    log = progress or (lambda _m: None)
    n_orf = seed.num_rows
    if n_orf == 0:
        empty64 = np.empty(0, dtype=np.int64)
        return FoldResult(empty64, empty64, empty64, np.empty(0, np.int8), [], 0)

    orf_nt = seed.column("orf_nucleotide")
    seq_len = np.array([len(s) for s in orf_nt.to_pylist()], dtype=np.int64)
    n_reads = seed.column("n_reads").to_numpy(zero_copy_only=False).astype(np.int64)

    log(f"folding {n_orf:,} distinct ORFs at identity {identity}…")
    index = extract_minimizers(
        orf_nt, k=kmer, w=window, max_per_seq=minimizers_per_seq
    )
    cands = generate_candidates(
        index, n_reads, min_shared=min_shared, diag_span_max=diag_span_max
    )
    log(f"  {cands.num_rows:,} candidate ORF pairs")
    # Unbounded ends: containment must be judged by the classifier, not by an
    # overhang gate. min_edit_budget floors the per-pair budget so short ORFs
    # are not gated out by rounding.
    unbounded = 2**31 - 1
    accepted = verify_candidates(
        cands,
        orf_nt.to_pylist(),
        identity=identity,
        max_5p=unbounded,
        max_3p=unbounded,
        min_budget=min_edit_budget,
        threads=threads,
    )
    rules = classify_pairs(
        accepted,
        seq_len,
        max_len_delta=max_len_delta,
        max_fs_mismatch=max_fs_mismatch,
        allow_frameshift=fold_frameshifts,
    )
    n_sep = int((rules == FoldRule.SEPARATE).sum())
    log(
        f"  {accepted.num_rows:,} verified pairs: "
        f"{int((rules == FoldRule.ERROR_VARIANT).sum()):,} error-variant, "
        f"{int((rules == FoldRule.FRAMESHIFT).sum()):,} frameshift, "
        f"{n_sep:,} separate"
    )

    fold_mask = rules != FoldRule.SEPARATE
    if accepted.num_rows:
        ea = accepted.column("uniq_short").to_numpy(zero_copy_only=False)[fold_mask]
        eb = accepted.column("uniq_long").to_numpy(zero_copy_only=False)[fold_mask]
    else:
        ea = eb = np.empty(0, dtype=np.int64)

    res = greedy_set_cover(n_orf, n_reads, seq_len, ea, eb)
    n_groups = res.centroid_uniq.shape[0]
    group_n_reads = np.zeros(n_groups, dtype=np.int64)
    np.add.at(group_n_reads, res.cluster_of, n_reads)

    declared: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_groups)]
    if fold_frameshifts and accepted.num_rows:
        declared = _declare_frameshift_columns(
            accepted, rules, res.cluster_of, res.centroid_uniq, seed
        )

    result = FoldResult(
        group_of_orf=res.cluster_of,
        group_rep_orf=res.centroid_uniq,
        group_n_reads=group_n_reads,
        rule_of_edge=rules,
        declared_variants=declared,
        n_separate_contained=n_sep,
    )
    if merge_representatives and not fold_frameshifts:
        result = _merge_by_representative(
            result,
            orf_nt.to_pylist(),
            seq_len,
            result.group_n_reads,
            identity=identity,
            max_len_delta=max_len_delta,
            min_edit_budget=min_edit_budget,
            kmer=kmer,
            window=window,
            minimizers_per_seq=minimizers_per_seq,
            threads=threads,
        )
    n_groups = result.group_rep_orf.shape[0]
    log(f"  → {n_groups:,} ORF groups (compression {n_orf / max(n_groups, 1):.2f}×)")
    return result


def _declare_frameshift_columns(
    accepted: pa.Table,
    rules: np.ndarray,
    group_of_orf: np.ndarray,
    group_rep_orf: np.ndarray,
    seed: pa.Table,
) -> list[np.ndarray]:
    """Variant columns on each group representative's *template*, from the
    rule-2 edges that touch it.

    Only edges with the representative as an endpoint contribute — for the
    rest the indel's position on the representative is not directly known,
    and the M-step's variant caller finds those columns anyway (which is why
    rule 2 is a convenience rather than a requirement).
    """
    idx = np.flatnonzero(rules == FoldRule.FRAMESHIFT)
    n_groups = group_rep_orf.shape[0]
    out: list[list[int]] = [[] for _ in range(n_groups)]
    if idx.size == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_groups)]

    short = accepted.column("uniq_short").to_numpy(zero_copy_only=False)[idx]
    long = accepted.column("uniq_long").to_numpy(zero_copy_only=False)[idx]
    ref_start = accepted.column("ref_start").to_numpy(zero_copy_only=False)[idx]
    cig = accepted.column("cigar").take(pa.array(idx)).to_pylist()
    orf_off = (
        seed.column("orf_start_in_template").to_numpy(zero_copy_only=False).astype(int)
    )

    for k in range(idx.size):
        s, lg = int(short[k]), int(long[k])
        g = int(group_of_orf[s])
        if int(group_of_orf[lg]) != g:
            continue
        rep = int(group_rep_orf[g])
        off_s, off_l = _first_indel_offsets(cig[k], int(ref_start[k]))
        if rep == lg and off_l >= 0:
            out[g].append(int(orf_off[rep]) + off_l)
        elif rep == s and off_s >= 0:
            out[g].append(int(orf_off[rep]) + off_s)
    return [np.unique(np.asarray(v, dtype=np.int64)) for v in out]


__all__ = [
    "FoldRule",
    "FoldResult",
    "classify_pair",
    "classify_pairs",
    "fold_orfs",
]
