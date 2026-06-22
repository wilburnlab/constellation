"""Context-aware within-cluster variant calling.

At each consensus position carrying a minor allele, decide *real variant*
vs *basecaller / PCR / RT error* using read depth and the **local
sequence context** — not a flat minor-allele-frequency cutoff. Per the
hifiasm-paper insight, Dorado error is non-uniform: homopolymer indels
dominate and scale with run length, while isolated substitutions are far
rarer. So the null error rate ε is conditioned on the variant class
(homopolymer indel by run length / substitution / non-homopolymer
indel), and the minor-allele count is tested against a Binomial (or, with
``overdispersion > 0``, Beta-Binomial) upper-tail under that ε.

The binomial ``n`` is the **raw read multiplicity** covering the position
(the abundance-weighted PWM column sums are exactly read counts, since a
unique sequence's weight is its read count). Low-depth positions
(``n < n_min``) are forced to ``ambiguous`` — you cannot distinguish a
variant from error without coverage. Surviving calls are FDR-controlled
(Benjamini-Hochberg) within the cluster.

The defaults are explicit priors (architecture invariant #1); the
empirical estimator that re-fits ε from the data lands in Cut 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo._cigar import base_codes


_IDX_TO_BASE = "ACGT-"


@dataclass(frozen=True, slots=True)
class ErrorModel:
    """Context-conditional per-read error-rate priors.

    Defaults are grounded in R10.4.1 / Dorado-sup error characteristics
    and the lab's homopolymer-dwell observations (CLAUDE.md): homopolymer
    indels dominate and scale with run length; substitutions are rarest.
    """

    eps_sub: float = 0.003
    eps_indel: float = 0.004  # non-homopolymer single-base indel
    eps_hp0: float = 0.01  # homopolymer-indel rate at the reference run length
    hp_ref_len: int = 3
    hp_slope: float = 0.015  # per extra base beyond hp_ref_len
    hp_min: float = 0.005
    hp_max: float = 0.15

    def epsilon_homopolymer(self, run_len: int) -> float:
        eps = self.eps_hp0 * (1.0 + self.hp_slope * (run_len - self.hp_ref_len))
        return float(min(max(eps, self.hp_min), self.hp_max))


def _homopolymer_runs(codes: np.ndarray) -> np.ndarray:
    """Per-position length of the homopolymer run containing each position."""
    n = codes.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = codes[1:] != codes[:-1]
    grp = np.cumsum(change) - 1
    sizes = np.bincount(grp)
    return sizes[grp].astype(np.int32)


def _survival(a: np.ndarray, n: np.ndarray, eps: np.ndarray, overdispersion: float):
    """P(X ≥ a) under Binomial(n, eps) — or Beta-Binomial when ρ > 0."""
    from scipy import stats

    if overdispersion and overdispersion > 0.0:
        rho = float(min(max(overdispersion, 1e-6), 1.0 - 1e-6))
        conc = (1.0 - rho) / rho
        alpha = eps * conc
        beta = (1.0 - eps) * conc
        return stats.betabinom.sf(a - 1, n, alpha, beta)
    return stats.binom.sf(a - 1, n, eps)


def call_variants(
    cres,
    *,
    model: ErrorModel | None = None,
    n_min: int = 5,
    a_min: int = 2,
    q_real: float = 0.01,
    alpha_amb: float = 0.05,
    overdispersion: float = 0.0,
) -> list[tuple]:
    """Call within-cluster variants from a :class:`ConsensusResult`.

    Returns one tuple per tested consensus position::

        (consensus_pos, consensus_allele, minor_allele, variant_class,
         homopolymer_run, depth_total, depth_minor, minor_fraction,
         p_error, epsilon_class, call)

    ``cluster_id`` is attached by the caller.
    """
    model = model or ErrorModel()
    pwm = cres.pwm
    winner = cres.winner
    consensus = cres.consensus
    L = pwm.shape[0]
    if L == 0 or not consensus:
        return []

    keep = winner < 4
    cons_pos_of_centroid = np.cumsum(keep) - 1
    n = pwm.sum(axis=1)
    sorted_counts = np.sort(pwm, axis=1)
    minor_count = sorted_counts[:, -2]
    cand = keep & (minor_count >= a_min) & (n >= 1)
    cand_idx = np.flatnonzero(cand)
    if cand_idx.size == 0:
        return []

    major = np.argmax(pwm, axis=1)
    pwm_nm = pwm.copy()
    pwm_nm[np.arange(L), major] = -1.0
    minor = np.argmax(pwm_nm, axis=1)

    cons_codes = base_codes(consensus)
    runs = _homopolymer_runs(cons_codes)

    cpos = cons_pos_of_centroid[cand_idx]
    mj = major[cand_idx]
    mn = minor[cand_idx]
    a = np.rint(minor_count[cand_idx]).astype(np.int64)
    nn = np.rint(n[cand_idx]).astype(np.int64)
    hp_run_full = runs[np.clip(cpos, 0, len(runs) - 1)]

    is_gap = mn == 4
    is_hp = is_gap & (hp_run_full >= 2)
    eps = np.empty(cand_idx.shape[0], dtype=np.float64)
    eps[~is_gap] = model.eps_sub
    eps[is_gap & ~is_hp] = model.eps_indel
    for i in np.flatnonzero(is_hp):
        eps[i] = model.epsilon_homopolymer(int(hp_run_full[i]))

    pvals = _survival(a, nn, eps, overdispersion)

    # Benjamini-Hochberg threshold over the cluster's tested positions.
    m = pvals.shape[0]
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked <= (np.arange(1, m + 1) / m) * q_real
    bh_cut = ranked[bh].max() if bh.any() else -1.0

    rows: list[tuple] = []
    for i in range(m):
        depth_gate = nn[i] < n_min
        if depth_gate:
            call = "ambiguous"
        elif pvals[i] <= bh_cut and a[i] >= a_min:
            call = "real"
        elif pvals[i] > alpha_amb:
            call = "collapsed_error"
        else:
            call = "ambiguous"
        vclass = (
            "homopolymer_indel"
            if is_hp[i]
            else ("non_hp_indel" if is_gap[i] else "substitution")
        )
        hp_run = int(hp_run_full[i]) if is_hp[i] else None
        rows.append(
            (
                int(cpos[i]),
                _IDX_TO_BASE[int(mj[i])],
                _IDX_TO_BASE[int(mn[i])],
                vclass,
                hp_run,
                int(nn[i]),
                int(a[i]),
                float(a[i] / nn[i]) if nn[i] else 0.0,
                float(pvals[i]),
                float(eps[i]),
                call,
            )
        )
    rows.sort(key=lambda r: r[0])
    return rows


def disagreement_stats(
    cres, *, min_major_frac: float = 0.95, min_depth: int = 30
) -> dict[tuple[int, int], tuple[float, float]]:
    """Aggregate minor-allele mass at high-confidence positions per class.

    At positions where the consensus is unambiguous (major fraction ≥
    ``min_major_frac``) and deep (``≥ min_depth`` reads), the minor mass is
    dominated by error — the substrate for fitting ε empirically. Returns
    ``{(class_code, run_len): (minor_sum, total_sum)}`` where ``class_code``
    is 0=substitution, 1=homopolymer_indel, 2=non_hp_indel. Fully
    vectorized over the consensus.
    """
    pwm = cres.pwm
    winner = cres.winner
    consensus = cres.consensus
    L = pwm.shape[0]
    if L == 0 or not consensus:
        return {}
    keep = winner < 4
    n = pwm.sum(axis=1)
    major = np.sort(pwm, axis=1)[:, -1]
    minor_total = n - major
    safe_n = np.maximum(n, 1)
    hc = keep & (n >= min_depth) & ((major / safe_n) >= min_major_frac)
    idx = np.flatnonzero(hc)
    if idx.size == 0:
        return {}

    cons_pos = (np.cumsum(keep) - 1)[idx]
    major_idx = np.argmax(pwm, axis=1)
    pwm_nm = pwm.copy()
    pwm_nm[np.arange(L), major_idx] = -1.0
    minor_idx = np.argmax(pwm_nm, axis=1)[idx]
    runs = _homopolymer_runs(base_codes(consensus))
    hp_run = runs[np.clip(cons_pos, 0, len(runs) - 1)]
    is_gap = minor_idx == 4
    is_hp = is_gap & (hp_run >= 2)
    class_code = np.where(is_gap, np.where(is_hp, 1, 2), 0)
    run_key = np.where(is_hp, hp_run, 0)

    keys = class_code.astype(np.int64) * 1000 + run_key
    uk, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    ms = np.zeros(uk.shape[0])
    ts = np.zeros(uk.shape[0])
    np.add.at(ms, inv, minor_total[idx])
    np.add.at(ts, inv, n[idx])
    return {
        (int(k // 1000), int(k % 1000)): (float(ms[i]), float(ts[i]))
        for i, k in enumerate(uk)
    }


def merge_disagreements(
    acc: dict[tuple[int, int], tuple[float, float]],
    other: dict[tuple[int, int], tuple[float, float]],
) -> None:
    """In-place merge of two disagreement-stat dicts (summing)."""
    for key, (m, t) in other.items():
        pm, pt = acc.get(key, (0.0, 0.0))
        acc[key] = (pm + m, pt + t)


def estimate_error_rates(
    stats: dict[tuple[int, int], tuple[float, float]],
    *,
    base: ErrorModel | None = None,
    min_total: float = 500.0,
) -> ErrorModel:
    """Fit a context-conditional :class:`ErrorModel` from disagreement stats.

    Substitution / non-homopolymer-indel rates are pooled ratios; the
    homopolymer-indel rate is a weighted linear fit of ε against run length
    mapped onto the parametric ``eps_hp0 · (1 + slope·(L − ref))`` form.
    Classes with too little evidence fall back to the prior.
    """
    base = base or ErrorModel()
    sub_m = sum(m for (c, _), (m, _t) in stats.items() if c == 0)
    sub_t = sum(t for (c, _), (_m, t) in stats.items() if c == 0)
    ind_m = sum(m for (c, _), (m, _t) in stats.items() if c == 2)
    ind_t = sum(t for (c, _), (_m, t) in stats.items() if c == 2)
    eps_sub = sub_m / sub_t if sub_t >= min_total else base.eps_sub
    eps_indel = ind_m / ind_t if ind_t >= min_total else base.eps_indel

    hp = [
        (run, m, t)
        for (c, run), (m, t) in stats.items()
        if c == 1 and t > 0 and run >= 2
    ]
    eps_hp0, hp_slope = base.eps_hp0, base.hp_slope
    if len(hp) >= 2 and sum(t for _r, _m, t in hp) >= min_total:
        runs = np.array([r for r, _m, _t in hp], dtype=np.float64)
        eps_obs = np.array([m / t for _r, m, t in hp])
        wt = np.array([t for _r, _m, t in hp])
        x = runs - base.hp_ref_len
        # weighted least squares: eps = intercept + coef * x
        W = wt.sum()
        xb = (wt * x).sum() / W
        yb = (wt * eps_obs).sum() / W
        sxx = (wt * (x - xb) ** 2).sum()
        sxy = (wt * (x - xb) * (eps_obs - yb)).sum()
        coef = sxy / sxx if sxx > 0 else 0.0
        intercept = yb - coef * xb
        if intercept > 1e-6:
            eps_hp0 = float(intercept)
            hp_slope = float(coef / intercept)

    return ErrorModel(
        eps_sub=float(eps_sub),
        eps_indel=float(eps_indel),
        eps_hp0=float(eps_hp0),
        hp_ref_len=base.hp_ref_len,
        hp_slope=float(hp_slope),
        hp_min=base.hp_min,
        hp_max=base.hp_max,
    )


# Column indices into a full variant-result tuple
# (cluster_id, consensus_pos, consensus_allele, minor_allele, variant_class,
#  homopolymer_run, depth_total, depth_minor, minor_fraction, p_error,
#  epsilon_class, call, max_linkage_r2).
_VR_CLUSTER, _VR_CLASS, _VR_HPRUN = 0, 4, 5
_VR_DTOTAL, _VR_DMINOR, _VR_PERR, _VR_EPS, _VR_CALL = 6, 7, 9, 10, 11


def reclassify_variants(
    variant_res: list[tuple],
    model: ErrorModel,
    *,
    overdispersion: float = 0.0,
    n_min: int = 5,
    a_min: int = 2,
    q_real: float = 0.01,
    alpha_amb: float = 0.05,
) -> list[tuple]:
    """Recompute ε / p_error / call for each variant row under ``model``.

    Used by ``--error-model empirical`` to re-test the already-found
    variant positions with the fitted error rates — no second consensus
    pass. Benjamini-Hochberg is re-applied per cluster.
    """
    if not variant_res:
        return variant_res
    rows = [list(r) for r in variant_res]
    eps = np.empty(len(rows))
    a = np.empty(len(rows), dtype=np.int64)
    n = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        vclass = r[_VR_CLASS]
        hp_run = r[_VR_HPRUN]
        if vclass == "substitution":
            eps[i] = model.eps_sub
        elif vclass == "homopolymer_indel":
            eps[i] = model.epsilon_homopolymer(int(hp_run) if hp_run else 2)
        else:
            eps[i] = model.eps_indel
        a[i] = int(r[_VR_DMINOR])
        n[i] = int(r[_VR_DTOTAL])
    pvals = _survival(a, n, eps, overdispersion)

    by_cluster: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_cluster.setdefault(int(r[_VR_CLUSTER]), []).append(i)
    for idxs in by_cluster.values():
        m = len(idxs)
        cl_p = np.array([pvals[i] for i in idxs])
        order = np.argsort(cl_p)
        ranked = cl_p[order]
        bh = ranked <= (np.arange(1, m + 1) / m) * q_real
        bh_cut = ranked[bh].max() if bh.any() else -1.0
        for i in idxs:
            if n[i] < n_min:
                call = "ambiguous"
            elif pvals[i] <= bh_cut and a[i] >= a_min:
                call = "real"
            elif pvals[i] > alpha_amb:
                call = "collapsed_error"
            else:
                call = "ambiguous"
            rows[i][_VR_PERR] = float(pvals[i])
            rows[i][_VR_EPS] = float(eps[i])
            rows[i][_VR_CALL] = call
    return [tuple(r) for r in rows]


__all__ = [
    "call_variants",
    "ErrorModel",
    "disagreement_stats",
    "merge_disagreements",
    "estimate_error_rates",
    "reclassify_variants",
]
