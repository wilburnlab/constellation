"""Candidate column selection for the covariance-driven M-step.

The previous M-step chose haplotype columns by ranking ``call_variants``
output by p-value and keeping the top 64 inside a "core region". Every part
of that failed on the full run:

* ``_core_region`` assumes one unimodal coverage plateau. 43.4% of templates
  carry ≥2 separated 5′ start modes, so the thing it trims as terminal
  raggedness *is* the signal.
* the indel FDR bypass admitted every in-core homopolymer unconditionally,
  and the homopolymer null was ~20× too permissive, so
* ranking by p-value then filled the 64-column budget with exactly those.

This module replaces the selection, not the statistics. A column becomes a
**candidate** by one of two routes, and candidacy is only permission to be
*tested for covariance* (:mod:`.covariance`) — it is not retention. The
homopolymers still pass candidacy easily; they die one layer later, for
co-varying with nothing, which needs no special case.

Two routes, because there are two ways a latent transcript differs from its
template:

``allelic``
    the column carries a real minor allele — a substitution, an indel, a
    minority insertion (which is gap-major by construction, so nothing here
    may gate on "is the major allele a gap").

``coverage``
    the column's *covered read set* varies. An alternative transcription
    start is not an allele at all: reads of the short form simply are not
    there. Letting extent compete with allele on equal terms is what makes
    the M-step able to emit references of different *lengths*.

Everything is ``(F,)`` in **frame** (PWM column) coordinates and O(F) from
``cres.pwm`` plus one O(M) pass for the member spans — no new data pass over
the reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo._cigar import base_codes
from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    member_spans,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import (
    ErrorModel,
    benjamini_hochberg,
    homopolymer_runs,
    survival,
)


#: A column may be admitted by either route, or both; the flags are a mask
#: because the route is **load-bearing at assignment time**, not a label.
#: See :class:`CandidateSet.route`.
ROUTE_ALLELIC = 1
ROUTE_COVERAGE = 2

_GAP = 4


@dataclass(frozen=True, slots=True)
class ColumnStats:
    """Per-PWM-column summary, all ``(F,)`` in frame coordinates."""

    depth: np.ndarray  # float64 — total vote weight (gap votes ARE coverage)
    base_cov: np.ndarray  # float64 — A/C/G/T vote weight only
    major: np.ndarray  # int8 — 0-3 base, 4 gap
    minor: np.ndarray  # int8 — runner-up, same alphabet
    n_major: np.ndarray  # float64
    n_minor: np.ndarray  # float64
    minor_frac: np.ndarray  # float64 — n_minor / depth
    uncov_frac: np.ndarray  # float64 — 1 − depth / n_assigned
    uncov_frac_local: np.ndarray  # float64 — 1 − depth / local max depth
    hp_run: np.ndarray  # int32 — homopolymer run length in context (below)
    eps: np.ndarray  # float64 — context-conditional null, after the floor
    is_gap: np.ndarray  # bool — either allele is a gap ⇒ length change
    is_hp: np.ndarray  # bool — …and it sits in a homopolymer
    column_kind: np.ndarray  # int8 — DIAGNOSTICS ONLY, never a gate
    boundary: np.ndarray  # bool — a clustered span endpoint sits here
    boundary_mass: np.ndarray  # float64 — weight of spans ending/starting there
    n_assigned: float
    n_columns: int


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Columns admitted for covariance testing, ascending by PWM column."""

    columns: np.ndarray  # (V,) int64 — PWM columns
    route: np.ndarray  # (V,) int8 — ROUTE_ALLELIC | ROUTE_COVERAGE
    # (V,) int32 — how many admitted columns this representative stands for.
    # Run collapse is exact, so a representative with run_len ≥ 2 stands for a
    # set of columns that are *perfectly* correlated by construction: it is
    # already a clique, and the covariance pass must treat it as one rather
    # than as a lone column with no partner. Otherwise collapsing — an
    # optimization — would silently change which nodes exist, and an
    # alternative transcription start (the canonical run) could never be
    # retained under a rule that requires an edge.
    run_len: np.ndarray
    effect: np.ndarray  # (V,) float64 — minority mass of the split, ≤ 0.5
    eps: np.ndarray  # (V,) float64 — per-column mismatch rate for the LL
    major: np.ndarray  # (V,) int8 — the majority allele
    # (V,) float64 — weight of the span endpoints at this representative's
    # DELIMITING boundary, which is the evidence for its claim. Not the
    # cumulative uncovered mass: what a coverage representative asserts is
    # "these reads start/stop here", and a degradation ramp accumulates a
    # large uncovered fraction out of boundaries that each carry one read.
    boundary_mass: np.ndarray
    p_value: np.ndarray  # (V,) float64 — allelic route only; 1.0 elsewhere
    q_cut: float  # the BH cutoff that admitted the allelic family
    n_allelic: int
    n_coverage: int
    n_collapsed: int  # columns folded away by run collapse
    n_capped: int  # candidates dropped by max_columns
    n_tested: int  # size of the BH family


def _local_max(x: np.ndarray, window: int) -> np.ndarray:
    """Max of ``x`` over ``[c − window, c + window]``, edges clamped.

    ``scipy`` is imported inside the function: the per-template pools are
    fork-based, and module-scope heavy numerics in the parent is what the
    torch-after-fork rule exists to avoid.
    """
    if x.shape[0] == 0:
        return x
    if window <= 0:
        return x.copy()
    from scipy.ndimage import maximum_filter1d

    return maximum_filter1d(x, size=2 * int(window) + 1, mode="nearest")


def _cluster_boundaries(
    pos: np.ndarray, weight: np.ndarray, n_columns: int, tolerance: int
) -> tuple[np.ndarray, np.ndarray]:
    """Support-ranked greedy clustering of span endpoints within ±tolerance.

    Returns ``(is_seed, mass)``, both ``(F,)``.

    The same rule ``cluster_junctions`` uses for intron donor/acceptor
    positions, and for the same reason: a real boundary is one dominant
    position with a low-support jitter skirt, not a uniform grid. Untolerated
    endpoints are what let a 5' start mode with ±5 nt of soft-clip jitter look
    like 200 separate boundaries of one read each — which is arithmetically
    indistinguishable from a degradation ramp, and is what makes the
    distinction between the two decidable at all.
    """
    is_seed = np.zeros(n_columns, dtype=bool)
    mass = np.zeros(n_columns, dtype=np.float64)
    if pos.shape[0] == 0:
        return is_seed, mass
    raw = np.bincount(
        np.clip(pos, 0, n_columns - 1), weights=weight, minlength=n_columns
    )
    live = np.flatnonzero(raw > 0)
    if live.size == 0:
        return is_seed, mass
    # Highest support first, ties by position so the result is reproducible
    # across worker counts.
    for c in live[np.lexsort((live, -raw[live]))]:
        if raw[c] <= 0:
            continue  # already absorbed
        lo, hi = max(0, c - tolerance), min(n_columns, c + tolerance + 1)
        is_seed[c] = True
        mass[c] = raw[lo:hi].sum()
        raw[lo:hi] = 0.0
    return is_seed, mass


def _context_codes(cres, pwm: np.ndarray) -> np.ndarray:
    """Base codes to measure homopolymer runs against, one per PWM column.

    Not the consensus string, and not ``cres.frame`` verbatim. Two failure
    modes to avoid:

    * ``call_variants`` reads the run at an **anchor** — the last kept column
      before an insertion — which is a different position than the one being
      tested, and on a pooled plan can be arbitrarily far away.
    * ``cres.frame`` writes ``-`` at a gap-winning inserted column, which
      *breaks* the run exactly where it matters: a minority G inserted into a
      5-G run would read as a 1-long run of ``-`` flanked by runs of 2 and 3.

    So a gap-winning column contributes the best base anyone voted there. A
    column still sits inside its homopolymer whether or not the argmax kept
    it, and it is precisely the gap-winning columns that the homopolymer null
    exists to judge.
    """
    f = pwm.shape[0]
    best_base = np.argmax(pwm[:, :4], axis=1).astype(np.int64)
    frame = getattr(cres, "frame", "") or ""
    if len(frame) == f:
        codes = base_codes(frame).astype(np.int64)
    else:  # hand-built results carry no frame string
        w = np.asarray(cres.winner, dtype=np.int64)
        codes = np.where(w < 4, w, _GAP)
    return np.where(codes < 4, codes, best_base)


def column_stats(
    cres,
    *,
    model: ErrorModel | None = None,
    n_assigned: float,
    eps_floor: float = 0.0,
    local_window: int = 200,
    boundary_tolerance: int = 10,
) -> ColumnStats:
    """Summarise every PWM column of ``cres``.

    ``n_assigned`` is the total read weight the E-step placed on this
    template, which is **not** ``depth.max()``: a read assigned to the
    template still fails to cover most columns of a fused 25 kb reference,
    and the difference between those two denominators is the whole coverage
    route.

    ``eps_floor`` is the annealing knob, applied as a floor rather than a
    replacement::

        ε_c = max(ε_context(c), floor)

    A replacement would set round-1 homopolymer ε to the floor against a 22%
    measurement, and round-3 substitution ε *below* its own prior. A floor
    only ever regularizes — early rounds refuse to split on evidence they
    would believe later, which is the point.
    """
    model = model or ErrorModel()
    pwm = np.asarray(cres.pwm, dtype=np.float64)
    f = pwm.shape[0]

    depth = pwm.sum(axis=1)
    base_cov = pwm[:, :4].sum(axis=1)
    major = np.argmax(pwm, axis=1)
    pwm_nm = pwm.copy()
    pwm_nm[np.arange(f), major] = -1.0
    minor = np.argmax(pwm_nm, axis=1)
    n_major = pwm[np.arange(f), major]
    n_minor = np.where(depth > 0, pwm[np.arange(f), minor], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        minor_frac = np.where(depth > 0, n_minor / depth, 0.0)

    total = float(max(n_assigned, 0.0))
    uncov_frac = (
        np.clip(1.0 - depth / total, 0.0, 1.0) if total > 0 else np.zeros(f)
    )
    loc = _local_max(depth, local_window)
    uncov_frac_local = np.where(loc > 0, np.clip(1.0 - depth / np.maximum(loc, 1e-12), 0.0, 1.0), 0.0)

    hp_run = homopolymer_runs(_context_codes(cres, pwm))
    # Either allele being a gap is a length change. On a minority-insertion
    # column the MAJOR allele is the gap, so keying off the minor would score
    # an insertion against the substitution null — the same asymmetry the
    # pre-planned column space exists to remove, one layer up.
    is_gap = (major == _GAP) | (minor == _GAP)
    is_hp = is_gap & (hp_run >= 2)

    eps = np.full(f, model.eps_sub, dtype=np.float64)
    eps[is_gap & ~is_hp] = model.eps_indel
    for i in np.flatnonzero(is_hp):
        eps[i] = model.epsilon_homopolymer(int(hp_run[i]))
    np.maximum(eps, float(eps_floor), out=eps)

    lo, hi = member_spans(cres)
    w = np.array([float(m.weight) for m in cres.members], dtype=np.float64)
    if lo.shape[0] != w.shape[0]:  # hand-built results
        w = np.ones(lo.shape[0], dtype=np.float64)
    # An interior boundary is one endpoint; the two template termini are not
    # boundaries at all (every read that reaches them is anchored there), so
    # they are dropped rather than clustered into the first real mode.
    interior = np.concatenate([lo[lo > 0], hi[hi < f]])
    mass_w = np.concatenate([w[lo > 0], w[hi < f]])
    boundary, boundary_mass = _cluster_boundaries(
        interior, mass_w, f, int(boundary_tolerance)
    )

    kind = getattr(cres, "column_kind", None)
    kind = (
        np.asarray(kind, dtype=np.int8)
        if kind is not None and len(kind) == f
        else np.zeros(f, dtype=np.int8)
    )
    return ColumnStats(
        depth=depth,
        base_cov=base_cov,
        major=major.astype(np.int8),
        minor=minor.astype(np.int8),
        n_major=n_major,
        n_minor=n_minor,
        minor_frac=minor_frac,
        uncov_frac=uncov_frac,
        uncov_frac_local=uncov_frac_local,
        hp_run=hp_run,
        eps=eps,
        is_gap=is_gap,
        is_hp=is_hp,
        column_kind=kind,
        boundary=boundary,
        boundary_mass=boundary_mass,
        n_assigned=total,
        n_columns=f,
    )


def _delimiting_mass(stats: ColumnStats, cols: np.ndarray) -> np.ndarray:
    """Mass of the clustered boundaries that delimit each column's block.

    A block of columns sharing one covered read-set is bounded on each side by
    a boundary (or by a template terminus, which carries no mass), and its
    claim is evidenced by whichever of the two the reads agree on. Run collapse
    guarantees there is no boundary strictly inside a run, so every column of a
    run gets the same pair.
    """
    seeds = np.flatnonzero(stats.boundary)
    out = np.zeros(cols.shape[0], dtype=np.float64)
    if seeds.size == 0 or cols.size == 0:
        return out
    mass = stats.boundary_mass[seeds]
    prev = np.searchsorted(seeds, cols, side="right") - 1
    nxt = np.searchsorted(seeds, cols, side="right")
    lo = np.where(prev >= 0, mass[np.clip(prev, 0, seeds.size - 1)], 0.0)
    hi = np.where(nxt < seeds.size, mass[np.clip(nxt, 0, seeds.size - 1)], 0.0)
    return np.maximum(lo, hi)


def candidate_columns(
    stats: ColumnStats,
    *,
    f_min: float = 0.02,
    a_min: float = 3.0,
    q_candidate: float = 0.01,
    overdispersion: float = 0.01,
    eps_coverage: float = 0.02,
    collapse_runs: bool = True,
    max_columns: int | None = 512,
) -> CandidateSet:
    """Admit columns for covariance testing.

    **Allelic route** — all three of: ``minor_frac ≥ f_min``,
    ``n_minor ≥ a_min``, and BH significance at ``q_candidate`` under
    ``survival(n_minor, depth, ε, ρ)``.

    Both the effect size and the significance are load-bearing, in opposite
    depth regimes. Effect size alone fails shallow: at 20 reads a single
    basecaller error is a 5% minor allele. Significance alone fails deep: at
    1,525 reads a 1% null admits 1,924 columns at ρ=0 — and **0** at ρ=0.01.
    ρ *is* the depth penalty (a beta-binomial caps effective depth at ~1/ρ),
    which is why it defaults on here and off in ``call_variants``.

    **Coverage route** — either ``uncov_frac`` or ``uncov_frac_local`` in
    ``[f_min, 1 − f_min]``.

    Both, because measured on synthetic fixtures they detect different
    things and each misses the other's case:

    *global* (over all assigned reads) answers "which sub-population am I?".
    On a fused template whose two genes are covered by disjoint read sets,
    every column of gene A sits at ``uncov_frac = 0.5`` — squarely in band —
    while the local statistic sees a *step*, not a ramp, and reads 0 inside
    each gene and 1.0 in the gap between them, so it admits **one** column of
    4,258 on that fixture. Fusion is the headline failure this route exists
    for, so the global denominator has to be in.

    *local* (against the max depth over ±``local_window``) answers "is there
    a coverage boundary here?" — a minority alternative start inside one
    population, where the global fraction barely moves. It is also the only
    one that survives when a template's assigned set is dominated by reads
    that cover almost none of it.

    The global route admitting most of a low-coverage template is not a
    problem on its own; it is a problem only if every admitted column is
    tested separately, which is what **run collapse** below prevents.

    **Run collapse** is exact, not an approximation. Adjacent columns have the
    same covered read-set unless some member's span starts or ends between
    them, so a maximal run with no interior boundary partitions the reads
    identically and one representative carrying ``run_len`` says everything
    the run says. This is the primary control on the covariance pass's
    ``Σkᵢ²``: without it, a 500-column start block gives every read outside it
    500 non-major states instead of 1.

    Only *pure*-coverage columns collapse. Two adjacent substitution columns
    may share a covered set and still be different alleles.

    Not gated on here, deliberately, and all four still apply in
    ``call_variants`` for the components path: ``in_core`` / ``_core_region``,
    the terminal-column exclusion, the indel FDR bypass, and the 64-column
    cap.
    """
    depth = stats.depth
    f = stats.n_columns

    # ── allelic ──
    testable = (
        (stats.minor_frac >= f_min) & (stats.n_minor >= a_min) & (depth > 0)
    )
    t_idx = np.flatnonzero(testable)
    p_value = np.ones(f, dtype=np.float64)
    q_cut = -1.0
    allelic = np.zeros(f, dtype=bool)
    if t_idx.size:
        a = np.rint(stats.n_minor[t_idx]).astype(np.int64)
        n = np.rint(depth[t_idx]).astype(np.int64)
        p = survival(a, n, stats.eps[t_idx], overdispersion)
        p_value[t_idx] = p
        sig, q_cut = benjamini_hochberg(p, q_candidate)
        allelic[t_idx[sig]] = True

    # ── coverage ──
    ufl = stats.uncov_frac_local
    ufg = stats.uncov_frac
    in_local = (ufl >= f_min) & (ufl <= 1.0 - f_min)
    in_global = (ufg >= f_min) & (ufg <= 1.0 - f_min)
    coverage = in_local | in_global

    cand = allelic | coverage
    cols = np.flatnonzero(cand)
    if cols.size == 0:
        return CandidateSet(
            columns=cols.astype(np.int64),
            route=np.zeros(0, dtype=np.int8),
            run_len=np.zeros(0, dtype=np.int32),
            effect=np.zeros(0, dtype=np.float64),
            eps=np.zeros(0, dtype=np.float64),
            major=np.zeros(0, dtype=np.int8),
            boundary_mass=np.zeros(0, dtype=np.float64),
            p_value=np.zeros(0, dtype=np.float64),
            q_cut=q_cut,
            n_allelic=0,
            n_coverage=0,
            n_collapsed=0,
            n_capped=0,
            n_tested=int(t_idx.size),
        )

    route = (allelic[cols] * ROUTE_ALLELIC + coverage[cols] * ROUTE_COVERAGE).astype(
        np.int8
    )
    # Minority mass of the split, on one scale for both routes so the
    # max_columns ranking compares like with like. §0's criticism of top-64
    # is about ranking by p-value — which fills the budget with homopolymers,
    # since significance grows with depth and says nothing about how much of
    # the template disagrees — not about capping at all.
    eff_allelic = np.where(allelic[cols], stats.minor_frac[cols], 0.0)
    eff_cov = np.maximum(
        np.where(in_local[cols], np.minimum(ufl[cols], 1.0 - ufl[cols]), 0.0),
        np.where(in_global[cols], np.minimum(ufg[cols], 1.0 - ufg[cols]), 0.0),
    )
    effect = np.maximum(eff_allelic, eff_cov)
    # A coverage column's state is "did this read reach here", whose error is
    # soft-clip / boundary jitter, not a base miscall — so it carries its own
    # ε. Taking the max where both routes fire is the conservative direction
    # (a larger ε is a *smaller* weight in the assignment likelihood).
    eps = stats.eps[cols].copy()
    is_cov = coverage[cols]
    eps[is_cov] = np.maximum(eps[is_cov], float(eps_coverage))
    run_len = np.ones(cols.shape[0], dtype=np.int32)

    n_collapsed = 0
    if collapse_runs and cols.size > 1:
        cum_b = np.cumsum(stats.boundary.astype(np.int64))
        pure_cov = route == ROUTE_COVERAGE
        keep = np.ones(cols.shape[0], dtype=bool)
        prev = -1
        for i in range(cols.shape[0]):
            if not pure_cov[i]:
                continue
            if prev >= 0 and cum_b[cols[i]] == cum_b[cols[prev]]:
                run_len[prev] += 1
                effect[prev] = max(effect[prev], effect[i])
                keep[i] = False
            else:
                prev = i
        n_collapsed = int((~keep).sum())
        cols, route, effect, eps, run_len = (
            cols[keep],
            route[keep],
            effect[keep],
            eps[keep],
            run_len[keep],
        )

    n_capped = 0
    if max_columns is not None and cols.shape[0] > max_columns:
        order = np.lexsort((cols, -run_len, -effect))[:max_columns]
        order.sort()
        n_capped = int(cols.shape[0] - max_columns)
        cols, route, effect, eps, run_len = (
            cols[order],
            route[order],
            effect[order],
            eps[order],
            run_len[order],
        )

    return CandidateSet(
        columns=cols.astype(np.int64),
        route=route,
        run_len=run_len,
        effect=effect,
        eps=eps,
        major=stats.major[cols],
        boundary_mass=_delimiting_mass(stats, cols),
        p_value=p_value[cols],
        q_cut=q_cut,
        n_allelic=int((route & ROUTE_ALLELIC).astype(bool).sum()),
        n_coverage=int((route & ROUTE_COVERAGE).astype(bool).sum()),
        n_collapsed=n_collapsed,
        n_capped=n_capped,
        n_tested=int(t_idx.size),
    )


__all__ = [
    "ROUTE_ALLELIC",
    "ROUTE_COVERAGE",
    "CandidateSet",
    "ColumnStats",
    "candidate_columns",
    "column_stats",
]
