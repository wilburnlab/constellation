"""The E-step's assignment rule: one candidate pool, two ranking regimes.

Written against plain ``(hit, group)`` arrays only — no PAF, no minimap2, no
templates — so the operating point is provable from hand-built inputs rather
than asserted through an end-to-end fixture.

**The pool (every round).** A hit is admitted iff
``n_match / aln_len >= p_floor``, a direct identity measurement from PAF
columns 10-11 with ``--eqx`` on. Equivalently ``AS/span >= 2 - 6*p_tol``,
since ``-x map-ont`` fixes A=2/B=4 and makes AS a linear proxy for identity —
but the PAF-column form is the measurement rather than the proxy, is immune to
the scoring parameters, and avoids AS/span's error-type bias (an inserted base
costs 8 where a substitution or a deleted base costs 6, so an AS-derived floor
rejects insertion-carrying reads it should admit).

This is the first absolute quality gate the pipeline has had. The rule it
replaces banded relative to each read's *own* best hit, so a read's worst
match was admitted whenever it was its only match — and that read then entered
the PWM of whatever it landed on, which is how an errored template's consensus
gets built in the first place.

**Round 1 ranks by abundance and quality.** Every round-1 template is a single
errored read, so a score measuring "how close is this read to that read"
cannot separate signal from the ~1% error both copies carry. Worse, argmax-AS
returns each read its own seed by construction — perfect self-identity scores
~2L against ~1.94L for the gene's true consensus, a ~72-point gap at 1.2 kb
that no near-tie window reaches. Self-capture would be total and permanent.

So round 1 ranks by ORF replication (how many reads carried that template's
ORF) and breaks ties on the seed read's Dorado quality. A read joins the
best-supported, best-sequenced representative within 3% of it; a read with
nothing within 3% has only its own template admitted and stays there as an
honest singleton. Nothing is pruned and no support threshold exists.

**Round 2+ ranks by likelihood.** Templates are consensuses now, so the
comparison has real content — see :mod:`.likelihood` for why it is computed
over the differing positions rather than on AS or identity. Abundance returns
only as an escape: a more abundant candidate wins **only** when it is within
``delta_logl`` of the argmax *and* carries ``support_ratio`` times its
support. A real variant with a couple of clearly discriminating positions
clears the window outright and so escapes capture by a much more abundant
neighbour; a read whose only difference is a homopolymer call does not.
"""

from __future__ import annotations

import numpy as np


#: Reads with no admitted hit get this, rather than being dropped, so the
#: rejection rate is visible per round instead of inferred from a row count.
UNASSIGNED = -1


def _group_of_hit(group_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.diff(group_ptr)
    return np.repeat(np.arange(sizes.size), sizes), sizes


def _first_per_group(
    order: np.ndarray, grp_of_hit: np.ndarray, n_groups: int
) -> np.ndarray:
    """Index of each group's first hit under ``order``."""
    return order[np.searchsorted(grp_of_hit[order], np.arange(n_groups))]


def admit_candidates(
    n_match: np.ndarray,
    aln_len: np.ndarray,
    group_ptr: np.ndarray,
    *,
    p_floor: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    """The candidate pool. Returns ``(admitted mask, n_admitted per group)``.

    ``aln_len == 0`` is not admitted rather than dividing by zero: an
    alignment of no length is no evidence.
    """
    n_match = np.asarray(n_match, dtype=np.int64)
    aln_len = np.asarray(aln_len, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        identity = np.where(aln_len > 0, n_match / np.maximum(aln_len, 1), 0.0)
    admitted = (aln_len > 0) & (identity >= p_floor)
    starts = group_ptr[:-1]
    n_groups = group_ptr.size - 1
    if n_groups <= 0 or admitted.size == 0:
        return admitted, np.zeros(max(n_groups, 0), dtype=np.int64)
    n_admitted = np.add.reduceat(admitted.astype(np.int64), starts)
    # reduceat over an empty trailing group repeats the previous value.
    n_admitted = np.where(np.diff(group_ptr) > 0, n_admitted, 0)
    return admitted, n_admitted


def _length_key(template_len: np.ndarray, read_len: np.ndarray) -> np.ndarray:
    """``|log(template_len / read_len)|`` — smaller is a better length match.

    The tie-break that catches the over-long-reference case: AS is
    read-anchored, so a read contained in a 5.7 kb template scores exactly
    what it would against a template matching it end to end. Template
    coverage is the asymmetry (p10 0.440 against read coverage's 0.998), and
    this is what consults it.
    """
    t = np.maximum(np.asarray(template_len, dtype=np.float64), 1.0)
    r = np.maximum(np.asarray(read_len, dtype=np.float64), 1.0)
    return np.abs(np.log(t / r))


def rank_round1(
    admitted: np.ndarray,
    template_idx: np.ndarray,
    group_ptr: np.ndarray,
    *,
    orf_replication: np.ndarray,
    seed_read_quality: np.ndarray,
) -> np.ndarray:
    """Round-1 winner per group: ORF replication, then seed quality.

    ``orf_replication`` and ``seed_read_quality`` are per-**template** arrays
    indexed by ``template_idx``. Quality is not cosmetic: an erroneous read
    needs an accurately-sequenced template to converge onto, and among
    equally-replicated ORFs the better-sequenced seed is the better frame for
    everything that joins it.

    Returns ``UNASSIGNED`` for a group with no admitted hit.
    """
    n_groups = group_ptr.size - 1
    if n_groups <= 0 or admitted.size == 0:
        return np.full(max(n_groups, 0), UNASSIGNED, dtype=np.int64)
    grp, _ = _group_of_hit(group_ptr)
    order = np.lexsort(
        (
            template_idx,
            -np.asarray(seed_read_quality, dtype=np.float64)[template_idx],
            -np.asarray(orf_replication, dtype=np.int64)[template_idx],
            ~admitted,  # admitted hits sort first
            grp,
        )
    )
    winner = _first_per_group(order, grp, n_groups)
    return np.where(admitted[winner], winner, UNASSIGNED)


def rank_likelihood(
    admitted: np.ndarray,
    template_idx: np.ndarray,
    group_ptr: np.ndarray,
    *,
    logl: np.ndarray,
    support: np.ndarray,
    read_len: np.ndarray,
    template_len: np.ndarray,
    delta_logl: float = 5.0,
    support_ratio: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Round-2+ winner per group. Returns ``(winner_slot, logl_delta)``.

    ``logl`` is per hit; ``support`` and ``template_len`` are per template,
    indexed by ``template_idx``; ``read_len`` is per hit.

    The default winner is argmax-logL. A different candidate takes it only if
    it is **both** within ``delta_logl`` of that argmax **and** carries at
    least ``support_ratio`` times its support. ``delta_logl`` is in nats and
    is directly interpretable — the 5.0 default is about one discriminating
    substitution's worth of evidence — so a real variant distinguished by even
    a couple of clean positions is outside the window and cannot be absorbed
    by a more abundant neighbour however abundant it is.

    ``logl_delta`` is each hit's deficit from its group's best admitted
    likelihood (0 on the argmax), emitted so the rule is auditable per round.
    """
    n_groups = group_ptr.size - 1
    if n_groups <= 0 or admitted.size == 0:
        e = np.empty(0, dtype=np.float64)
        return np.full(max(n_groups, 0), UNASSIGNED, dtype=np.int64), e

    grp, sizes = _group_of_hit(group_ptr)
    logl = np.asarray(logl, dtype=np.float64)
    support_h = np.asarray(support, dtype=np.float64)[template_idx]
    len_key = _length_key(
        np.asarray(template_len, dtype=np.float64)[template_idx], read_len
    )

    masked = np.where(admitted, logl, -np.inf)
    starts = group_ptr[:-1]
    best = np.maximum.reduceat(masked, starts)
    best = np.where(sizes > 0, best, -np.inf)
    best_h = np.repeat(best, sizes)
    logl_delta = np.where(np.isfinite(best_h), best_h - masked, np.inf)

    order = np.lexsort((template_idx, len_key, -masked, ~admitted, grp))
    argmax = _first_per_group(order, grp, n_groups)
    has_any = admitted[argmax]

    # The abundance escape, evaluated against the argmax's own support.
    thresh = np.repeat(support_h[argmax] * float(support_ratio), sizes)
    eligible = (
        admitted
        & np.isfinite(logl_delta)
        & (logl_delta <= float(delta_logl))
        & (support_h >= thresh)
    )
    order2 = np.lexsort((template_idx, len_key, -support_h, ~eligible, grp))
    override = _first_per_group(order2, grp, n_groups)
    take_override = eligible[override]

    winner = np.where(take_override, override, argmax)
    return np.where(has_any, winner, UNASSIGNED), logl_delta


def shortlist_for_likelihood(
    admitted: np.ndarray,
    as_score: np.ndarray,
    group_ptr: np.ndarray,
    *,
    span: np.ndarray,
    z: float = 2.0,
    error_rate: float = 0.01,
    match_mismatch_cost: float = 6.0,
) -> np.ndarray:
    """Which admitted hits are close enough in AS to be worth scoring.

    AS is retained as a **shortlist**, not as the arbiter: computing the
    likelihood needs each candidate's CIGAR, and at ~480M hits a round that
    has to be confined to the hits where the answer is actually in doubt. AS
    and logL agree except where the error model disagrees with uniform
    weighting, and that can only flip the answer when the AS gap is small.

    The window is the handoff's ``w(L, e) = z * (A + B) * sqrt(L * e(1 - e))``
    — mismatches to the *correct* template are ~Binomial(L, e), so AS noise
    grows as sqrt(L), not as L and not as a constant. At z=2 this reproduces
    the hand-tuned ``band_abs = 40`` at the median 1.2 kb read, which is why
    that constant worked at all; it is ~2x too wide at 300 nt and ~1.5x too
    narrow at 2.5 kb.
    """
    n_groups = group_ptr.size - 1
    if n_groups <= 0 or admitted.size == 0:
        return np.zeros_like(admitted, dtype=bool)
    _, sizes = _group_of_hit(group_ptr)
    as_score = np.asarray(as_score, dtype=np.float64)
    masked = np.where(admitted, as_score, -np.inf)
    best = np.maximum.reduceat(masked, group_ptr[:-1])
    best = np.where(sizes > 0, best, -np.inf)
    span = np.maximum(np.asarray(span, dtype=np.float64), 1.0)
    width = (
        float(z)
        * float(match_mismatch_cost)
        * np.sqrt(span * error_rate * (1.0 - error_rate))
    )
    return admitted & (masked >= np.repeat(best, sizes) - width)


def shortlist_by_chain(
    chain_score: np.ndarray,
    group_ptr: np.ndarray,
    *,
    k: int,
    frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two-pass E-step's shortlist: which hits get base-aligned at all.

    Keeps a hit iff its chaining score is ≥ ``frac`` × its read's best **and**
    it is among the read's top ``k`` by chaining score (ties broken by hit
    order, which is deterministic). Returns ``(keep, chain_rank, n_eligible)``:
    the mask, each hit's 0-based rank within its read by chaining score, and
    per read how many hits cleared the ``frac`` cut before ``k`` was applied —
    so ``n_eligible > k`` is exactly "the shortlist cut something".

    Unlike the single-pass pool this is **not** guaranteed to contain every
    template within ``p_floor``: the chaining score is a proxy for identity,
    not the thing itself. ``k`` and ``frac`` are unmeasured operating points,
    which is why the truncation is reported per read.
    """
    n_groups = group_ptr.size - 1
    s1 = np.asarray(chain_score, dtype=np.float64)
    if n_groups <= 0 or s1.size == 0:
        z = np.zeros(s1.size, dtype=bool)
        return z, np.zeros(s1.size, dtype=np.int64), np.zeros(max(n_groups, 0), np.int64)
    grp, sizes = _group_of_hit(group_ptr)
    best = np.maximum.reduceat(s1, group_ptr[:-1])
    eligible = s1 >= float(frac) * np.repeat(best, sizes)
    order = np.lexsort((np.arange(s1.size), -s1, grp))
    rank = np.empty(s1.size, dtype=np.int64)
    rank[order] = np.arange(s1.size) - np.repeat(group_ptr[:-1], sizes)
    keep = eligible & (rank < int(k))
    n_eligible = np.add.reduceat(eligible.astype(np.int64), group_ptr[:-1])
    return keep, rank, n_eligible


__all__ = [
    "UNASSIGNED",
    "admit_candidates",
    "rank_likelihood",
    "rank_round1",
    "shortlist_by_chain",
    "shortlist_for_likelihood",
]
