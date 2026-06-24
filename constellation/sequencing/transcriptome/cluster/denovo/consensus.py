"""Centroid-anchored consensus via a CIGAR-walk PWM (numpy scatter).

Each cluster member is projected onto the centroid coordinate frame
using the **cached** verify-stage CIGAR (no re-alignment), and its bases
are accumulated into an abundance-weighted ``(L, 5)`` position-weight
matrix (columns A/C/G/T/gap). The consensus is the per-position argmax;
gap-winning positions drop out, positions with no coverage fall back to
the centroid base.

The projection mirrors ``align/consensus.py``'s scatter kernel but reads
member bases from the CIGAR + member sequence (not a cs:long string) and
retains the full PWM for the variant-calling stage (Cut 2).

Insertions relative to the centroid are recorded as length-variation
events, not folded into the PWM (v1 — matches the genome-anchored
kernel's behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo._cigar import (
    base_codes,
    parse_cigar,
)


_IDX_TO_BASE_NP = np.array(
    [ord("A"), ord("C"), ord("G"), ord("T"), ord("-"), ord("N")], dtype=np.uint8
)
_GAP = 4


@dataclass(frozen=True, slots=True)
class MemberSpec:
    """One member's alignment onto the centroid frame (from the cache)."""

    member_seq: str
    weight: float
    cigar: str
    centroid_is_query: bool  # True iff the centroid was the alignment query
    ref_start: int  # short's start on long (the cached alignment's ref_start)


@dataclass(slots=True)
class ConsensusResult:
    """Per-cluster consensus + the PWM it was argmax'd from.

    ``pwm`` is the abundance-weighted ``(L, 5)`` matrix in centroid
    coordinates (used for variant calling). ``winner`` is the per-centroid-
    position winning base (0-3 = ACGT, 4 = gap/dropped) before gap positions
    are removed — the centroid↔consensus coordinate map. Per-member alleles
    for the haplotype matrix are recomputed by the caller via
    :func:`haplotypes.member_allele_row` (so the votes needn't be retained).
    """

    consensus: str
    pwm: np.ndarray  # (L, 5) float64 abundance-weighted PWM in centroid coords
    winner: np.ndarray = field(repr=False)


def project_member(
    cigar_ops: list[tuple[int, str]],
    member_codes: np.ndarray,
    *,
    centroid_is_query: bool,
    centroid_start: int,
    member_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a member onto the centroid frame.

    Returns ``(match_cpos, match_base, gap_cpos)``: centroid positions
    receiving a member base vote + the base codes, and centroid
    positions receiving a gap (deletion) vote. Insertions relative to
    the centroid are skipped (not projectable onto the frame).
    """
    # Which op consumes the centroid only (→ gap) vs the member only (→ ins).
    gap_op = "I" if centroid_is_query else "D"
    ins_op = "D" if centroid_is_query else "I"

    match_cpos: list[np.ndarray] = []
    match_base: list[np.ndarray] = []
    gap_cpos: list[np.ndarray] = []
    cpos = centroid_start
    mpos = member_start
    for length, op in cigar_ops:
        if op in ("=", "X", "M"):
            match_cpos.append(np.arange(cpos, cpos + length, dtype=np.int64))
            match_base.append(member_codes[mpos : mpos + length])
            cpos += length
            mpos += length
        elif op == gap_op:
            gap_cpos.append(np.arange(cpos, cpos + length, dtype=np.int64))
            cpos += length
        elif op == ins_op:
            mpos += length
    mc = np.concatenate(match_cpos) if match_cpos else np.empty(0, dtype=np.int64)
    mb = (
        np.concatenate(match_base).astype(np.int64)
        if match_base
        else np.empty(0, dtype=np.int64)
    )
    gc = np.concatenate(gap_cpos) if gap_cpos else np.empty(0, dtype=np.int64)
    return mc, mb, gc


def centroid_consensus(
    centroid_seq: str,
    centroid_weight: float,
    members: list[MemberSpec],
) -> ConsensusResult:
    """Build the abundance-weighted consensus for one cluster.

    ``members`` excludes the centroid itself (added as a self-vote with
    ``centroid_weight``).
    """
    L = len(centroid_seq)
    centroid_codes = base_codes(centroid_seq).astype(np.int64)

    flat_idx: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    # Centroid self-vote.
    cpos0 = np.arange(L, dtype=np.int64)
    valid0 = centroid_codes < 4
    flat_idx.append(cpos0[valid0] * 5 + centroid_codes[valid0])
    weights.append(np.full(int(valid0.sum()), centroid_weight, dtype=np.float64))

    for spec in members:
        ops = parse_cigar(spec.cigar)
        mcodes = base_codes(spec.member_seq).astype(np.int64)
        if spec.centroid_is_query:
            cstart, mstart = 0, spec.ref_start
        else:
            cstart, mstart = spec.ref_start, 0
        mc, mb, gc = project_member(
            ops,
            mcodes,
            centroid_is_query=spec.centroid_is_query,
            centroid_start=cstart,
            member_start=mstart,
        )
        # Keep votes inside the centroid window with an unambiguous base.
        good = (mc >= 0) & (mc < L) & (mb < 4)
        flat_idx.append(mc[good] * 5 + mb[good])
        weights.append(np.full(int(good.sum()), spec.weight, dtype=np.float64))
        gc_in = gc[(gc >= 0) & (gc < L)]
        if gc_in.shape[0]:
            flat_idx.append(gc_in * 5 + _GAP)
            weights.append(np.full(gc_in.shape[0], spec.weight, dtype=np.float64))

    # Scatter-add the votes into the (L, 5) PWM with numpy bincount — kept
    # torch-free so this runs inside fork()ed ProcessPool workers without the
    # torch/OpenMP-after-fork deadlock (by this stage the parent has already
    # spawned torch's thread pool during minimizer extraction).
    if flat_idx:
        idx_all = np.concatenate(flat_idx)
        w_all = np.concatenate(weights)
        flat_pwm = np.bincount(idx_all, weights=w_all, minlength=L * 5)
    else:
        flat_pwm = np.zeros(L * 5, dtype=np.float64)
    pwm = flat_pwm.reshape(L, 5)

    row_max = pwm.max(axis=1)
    best_np = pwm.argmax(axis=1)
    # No-coverage positions fall back to the centroid base; gap-winning
    # positions (and centroid Ns, encoded 4) drop out of the consensus.
    winner = np.where(row_max <= 0.0, centroid_codes, best_np)
    keep = winner < _GAP
    consensus = _IDX_TO_BASE_NP[winner[keep]].tobytes().decode("ascii")

    return ConsensusResult(consensus=consensus, pwm=pwm, winner=winner)


__all__ = ["MemberSpec", "ConsensusResult", "centroid_consensus", "project_member"]
