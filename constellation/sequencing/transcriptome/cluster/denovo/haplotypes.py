"""Within-cluster haplotypes + variant covariance (phasing).

From the consensus stage's retained per-member base votes, build the
read × variant-position allele matrix, collapse it to distinct
allele-combination **haplotypes** (abundance-weighted), and compute the
pairwise **r²** between variant positions. Phased variants co-occur on
the same haplotype — the signal that distinguishes "one transcript with
scattered errors" from "a mix of alleles / paralogs that maybe should be
split." Everything is vectorized over the (small) set of variant
positions; no per-read Python loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def build_haplotypes(
    cres,
    var_centroid_positions: np.ndarray,
    var_consensus_positions: np.ndarray,
    minor_alleles: list[str],
    consensus_alleles: list[str],
    member_weights: np.ndarray,
) -> HaplotypeResult:
    """Collapse members to haplotypes + compute variant covariance.

    ``var_centroid_positions`` are the variants' centroid-frame columns
    (to index the votes); ``var_consensus_positions`` label them in the
    output. ``member_weights[i]`` is member ``i``'s read multiplicity
    (member 0 = centroid).
    """
    V = int(var_centroid_positions.shape[0])
    M = int(member_weights.shape[0])
    if V == 0 or M == 0:
        return HaplotypeResult([], [], np.zeros(0, dtype=np.float32))

    # Allele matrix A[member, variant]; -1 = position not covered by member.
    A = np.full((M, V), -1, dtype=np.int8)
    vp = var_centroid_positions.astype(np.int64)
    order = np.argsort(vp)
    vp_sorted = vp[order]
    cpos = cres.votes_cpos
    if cpos.shape[0]:
        idx = np.searchsorted(vp_sorted, cpos)
        idx_c = np.clip(idx, 0, V - 1)
        inb = vp_sorted[idx_c] == cpos
        col = order[idx_c]
        sel = inb
        A[cres.votes_member[sel], col[sel]] = cres.votes_base[sel].astype(np.int8)

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


__all__ = ["build_haplotypes", "HaplotypeResult"]
