"""Within-cluster haplotypes + variant covariance (phasing).

Build the read × variant-position allele matrix (one row per **member** —
the caller aligns *every* member, not just the consensus-voting subset,
so haplotype abundances sum to the cluster's read count), collapse it to
distinct allele-combination **haplotypes** (abundance-weighted), and
compute the pairwise **r²** between variant positions. Phased variants
co-occur on the same haplotype — the signal that distinguishes "one
transcript with scattered errors" from "a mix of alleles / paralogs that
maybe should be split." Vectorized over the (small) set of variant
positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    project_member,
)


_IDX_TO_CHAR = np.array(
    [ord("A"), ord("C"), ord("G"), ord("T"), ord("-")], dtype=np.uint8
)
_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 4}


@dataclass(slots=True)
class HaplotypeResult:
    """Per-cluster haplotypes + per-variant max linkage r²."""

    # (allele_string, abundance, n_unique_sequences, is_complete) per haplotype,
    # abundance-descending.
    haplotypes: list[tuple[str, int, int, bool]]
    variant_positions: list[int]  # consensus positions, in column order
    max_r2: np.ndarray  # per variant position, max pairwise r² to any other


def member_allele_row(
    cigar_ops: list[tuple[int, str]],
    member_codes: np.ndarray,
    *,
    centroid_is_query: bool,
    centroid_start: int,
    member_start: int,
    var_sorted: np.ndarray,
) -> np.ndarray:
    """A member's allele (``int8``: 0-3 = ACGT, 4 = gap/deletion, -1 =
    uncovered) at each variant **centroid** position (``var_sorted`` ascending).

    Projects the member onto the centroid frame and reads off the alleles at
    the variant columns — the per-read assignment that places every member on
    a haplotype."""
    mc, mb, gc = project_member(
        cigar_ops,
        member_codes,
        centroid_is_query=centroid_is_query,
        centroid_start=centroid_start,
        member_start=member_start,
    )
    V = var_sorted.shape[0]
    row = np.full(V, -1, dtype=np.int8)
    if mc.shape[0]:
        good = mb < 4  # skip ambiguous / N (matches the PWM vote filter)
        mcg, mbg = mc[good], mb[good]
        idx = np.clip(np.searchsorted(var_sorted, mcg), 0, V - 1)
        inb = var_sorted[idx] == mcg
        row[idx[inb]] = mbg[inb].astype(np.int8)
    if gc.shape[0]:
        idxg = np.clip(np.searchsorted(var_sorted, gc), 0, V - 1)
        inbg = var_sorted[idxg] == gc
        row[idxg[inbg]] = 4
    return row


def build_haplotypes(
    A: np.ndarray,
    member_weights: np.ndarray,
    var_consensus_positions: list[int],
    minor_alleles: list[str],
    consensus_alleles: list[str],
) -> HaplotypeResult:
    """Collapse a member × variant-position allele matrix ``A`` to distinct
    haplotypes + compute variant covariance.

    ``A[i, v]`` is member ``i``'s allele (0-3 ACGT, 4 gap, -1 uncovered) at
    variant column ``v``; ``member_weights[i]`` is its read multiplicity.
    ``var_consensus_positions`` label the columns in the output.
    """
    M, V = A.shape
    if V == 0 or M == 0:
        return HaplotypeResult([], [], np.zeros(0, dtype=np.float32))

    # Distinct haplotypes (rows of A), abundance-weighted.
    uniq_rows, inverse, counts = np.unique(
        A, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    hap_abund = np.zeros(uniq_rows.shape[0], dtype=np.int64)
    np.add.at(hap_abund, inverse, member_weights.astype(np.int64))
    hap_nuniq = counts.astype(np.int64)
    sort_h = np.argsort(-hap_abund)

    haplotypes: list[tuple[str, int, int, bool]] = []
    for h in sort_h:
        row = uniq_rows[h]
        chars = np.where(row < 0, ord("."), _IDX_TO_CHAR[np.clip(row, 0, 4)]).astype(
            np.uint8
        )
        allele_string = chars.tobytes().decode("ascii")
        is_complete = bool((row >= 0).all())
        haplotypes.append(
            (allele_string, int(hap_abund[h]), int(hap_nuniq[h]), is_complete)
        )

    # Pairwise r² over variant positions (read-weighted, on co-covered members).
    major_idx = np.array([_BASE_TO_IDX.get(b, -1) for b in consensus_alleles])
    minor_idx = np.array([_BASE_TO_IDX.get(b, -1) for b in minor_alleles])
    B = np.full((M, V), np.nan)
    for v in range(V):
        colv = A[:, v]
        B[colv == major_idx[v], v] = 0.0
        B[colv == minor_idx[v], v] = 1.0
    max_r2 = np.zeros(V, dtype=np.float64)
    w = member_weights.astype(np.float64)
    for u in range(V):
        for v in range(u + 1, V):
            mask = ~np.isnan(B[:, u]) & ~np.isnan(B[:, v])
            if not mask.any():
                continue
            ww = w[mask]
            W = ww.sum()
            if W <= 0:
                continue
            bu = B[mask, u]
            bv = B[mask, v]
            pu = (ww * bu).sum() / W
            pv = (ww * bv).sum() / W
            puv = (ww * bu * bv).sum() / W
            denom = pu * (1 - pu) * pv * (1 - pv)
            if denom <= 0:
                continue
            r2 = (puv - pu * pv) ** 2 / denom
            max_r2[u] = max(max_r2[u], r2)
            max_r2[v] = max(max_r2[v], r2)

    return HaplotypeResult(
        haplotypes=haplotypes,
        variant_positions=[int(x) for x in var_consensus_positions],
        max_r2=max_r2.astype(np.float32),
    )


__all__ = ["build_haplotypes", "member_allele_row", "HaplotypeResult"]
