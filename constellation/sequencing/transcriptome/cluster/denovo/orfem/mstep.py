"""Stage 3 — M-step: covariance-driven template refinement.

For one template and the reads the E-step assigned to it:

1. **Pooled folded consensus** over every assigned read. Round 1 needs no
   alignment work at all — minimap2's own ``cg:Z`` is already a read→template
   alignment in exactly the convention ``project_member`` wants (query = read,
   ``I`` consumes the read, ``D`` the template), so it feeds straight in with
   ``frame_start = t_start`` and ``member_start = q_start``, and the
   soft-clipped read flanks become the end-extension candidates. Only the
   post-splice passes call edlib.
2. **Candidate columns** (:mod:`.columns`) by allele or by coverage.
3. **Covariance** (:mod:`.covariance`) — which of them earn a node, grouped
   into signatures, with every read assigned to one pattern of each.
4. **One node per observed state-tuple**, its consensus built on the pooled
   column plan, its ORF re-predicted under a support gate.

The order matters and is not obvious: the pooled PWM has to come first,
because in round 1 there are no declared columns to group by.

**Emit by state-tuple, not per signature.** A read's label is the tuple of its
per-signature states, and each distinct observed tuple with enough mass is one
node. Three reasons. Every read lands in exactly one node, so
``Σ n_reads == template total`` — the invariant quant depends on. Two
independent signatures correctly give up to **four** templates rather than two
marginals that double-count reads. And the tuple space is bounded by *observed*
combinations, not ``2^k``.

**The support gate.** A template's flanks are one read's sequence and carry no
certificate, so an ORF must not silently extend through them. Certification is
evaluated on *consensus* columns (where gap-winning columns do not exist,
which resolves an ambiguity in the spec): a column is certified when its base
coverage and its agreement both clear thresholds. An ORF reaching past the
seed ORF's boundary is truncated at the last certified column and flagged.
This is only meaningful *because* the consensus kernel can now extend past the
frame at all — before that, an ORF could never reach uncertified ground.

**What is gone, and why.** ``call_variants`` and ``build_haplotypes`` are no
longer called from this path (they are unchanged, and the components path
still uses them). With them go ``_core_region``'s unimodal-plateau assumption,
the indel FDR bypass, the 64-column cap, ``np.unique(A, axis=0)`` treating
uncovered as a *symbol*, modal absorption folding unsupported rows into the
major node distance-blind, and the ``max_r2`` vector that the pairwise test in
:mod:`.covariance` replaces. Each of those was measured producing nodes on the
9.4M-read run: 28.1% of minor nodes differed from their major only at
uncovered positions, and 71.7% of the rest differed at exactly one.

**No ``**kernel_kwargs``.** Every annealing knob is an explicit per-round
argument, because a forwarded ``**kwargs`` is how ``--consensus-max-passes``
survived as a user-facing flag that did nothing for a release.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    ConsensusResult,
    MemberSpec,
    frame_consensus,
    member_allele_events,
)
from constellation.sequencing.transcriptome.cluster.denovo.orf import (
    ORF_CODON_TABLE,
    best_sense_orf,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem import covariance as cv
from constellation.sequencing.transcriptome.cluster.denovo.orfem.columns import (
    candidate_columns,
    column_stats,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import (
    ErrorModel,
    disagreement_stats,
)


@dataclass(slots=True)
class RefinedTemplate:
    """One node: a state-tuple's consensus, its ORF, and its read support."""

    parent_template_id: int
    haplotype_id: int
    consensus: str
    n_reads: int
    node_weight: float
    protein: str | None = None
    orf_start: int = -1
    orf_end: int = -1
    orf_certified_end: int = -1
    orf_is_truncated_by_support: bool = False
    allele_string: str = ""
    declared_variants: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    n_inserted_columns: int = 0
    n_extended_5p: int = 0
    n_extended_3p: int = 0
    n_trimmed_5p: int = 0
    n_trimmed_3p: int = 0
    #: Per-template diagnostics, on the major node only. Carries the
    #: ``disagreement_stats`` the bench driver merges across round *r*'s
    #: templates to refit round *r+1*'s error model, plus the covariance
    #: counters ("the method returned one template" has to be visible rather
    #: than inferred from a quiet result).
    stats: dict = field(default_factory=dict, repr=False)


def certified_columns(
    cres: ConsensusResult, *, min_depth: float, min_agreement: float
) -> np.ndarray:
    """Per **consensus** position: is this column supported by real reads?

    ``base_cov`` counts A/C/G/T votes only. At a ragged end many members vote
    a deletion, so total coverage stays flat there while base coverage ramps
    down — which is exactly the distinction the gate needs.
    """
    f = cres.frame_of_cons
    if f is None or f.size == 0:
        return np.zeros(0, dtype=bool)
    rows = cres.pwm[f]
    base_cov = rows[:, :4].sum(axis=1)
    total = rows.sum(axis=1)
    agreement = rows.max(axis=1) / np.maximum(total, 1.0)
    return (base_cov >= min_depth) & (agreement >= min_agreement)


def gated_orf(
    consensus: str,
    certified: np.ndarray,
    *,
    seed_orf_start: int,
    seed_orf_end: int,
    min_aa_length: int = 30,
):
    """Predict the ORF, refusing sequence that reads do not support.

    ``seed_orf_start`` / ``seed_orf_end`` bound the previous round's ORF in
    *this consensus's* coordinates; that interval is certified by
    construction and only ground **outside** it is judged. Both ends are
    gated: an upstream ATG sitting in single-read flank can extend the
    protein just as silently as a downstream readthrough can, and sharing
    the seed's stop codon makes it look like an ordinary N-terminal
    extension. Returns ``(protein, start, end, certified_end, truncated)``
    or ``None``.
    """
    hit = best_sense_orf(consensus, min_aa_length=min_aa_length)
    if hit is None:
        return None
    prot, st, en = hit
    if certified.size == 0:
        return prot, st, en, en, False
    truncated = False

    # 5': reaching an upstream start means crossing everything between it and
    # the seed's start, so all of that has to be certified.
    if st < seed_orf_start:
        lo, hi = max(0, st), min(seed_orf_start, certified.size)
        if hi > lo and not certified[lo:hi].all():
            resume = lo + int(np.flatnonzero(~certified[lo:hi])[-1]) + 1
            again = best_sense_orf(consensus[resume:], min_aa_length=min_aa_length)
            if again is None:
                return None
            prot, st, en = again[0], again[1] + resume, again[2] + resume
            truncated = True

    if en <= max(seed_orf_end, st):
        return prot, st, en, en, truncated

    # Walk forward from wherever the certificate ends and find the first
    # uncertified column the ORF would have to pass through.
    lo = min(max(seed_orf_end, st), len(consensus))
    limit = lo
    while limit < min(en, certified.size) and certified[limit]:
        limit += 1
    if limit >= en:
        return prot, st, en, en, truncated

    # Report the ORF as ending at the last certified column, on a codon
    # boundary. It no longer ends in a stop — that is the point of the flag.
    trunc_end = st + 3 * ((limit - st) // 3)
    if trunc_end - st < min_aa_length * 3:
        return None
    from constellation.core.sequence.nucleic import translate

    prot_t = translate(
        consensus[st:trunc_end], codon_table=ORF_CODON_TABLE, partial="discard"
    )
    return prot_t, st, trunc_end, limit, True


def specs_from_assignments(
    rows,
    read_seq: dict[str, str],
) -> list[MemberSpec]:
    """Build PWM members straight from the E-step's minimap2 alignments.

    ``rows`` are dicts with ``read_id, cigar, t_start, q_start, weight``. The
    orientation is fixed: minimap2 emits query→target, so the template is the
    reference and ``centroid_is_query`` is False.
    """
    out: list[MemberSpec] = []
    for i, r in enumerate(rows):
        cig = r.get("cigar")
        if not cig:
            continue
        out.append(
            MemberSpec(
                member_seq=read_seq[r["read_id"]],
                weight=float(r.get("weight", 1.0)),
                cigar=cig,
                centroid_is_query=False,
                ref_start=int(r["t_start"]),
                member_start=int(r["q_start"]),
                member_id=i,
            )
        )
    return out




def _child_position(cres: ConsensusResult, column: int, *, forward: bool) -> int:
    """A PWM column's position in one child's own consensus.

    A column dropped by that child (its state votes gap there) has no
    position, so walk to the nearest kept neighbour in the direction the
    caller cares about. Reusing the parent's number instead is how a boundary
    silently slides: deleting 30 upstream bases moves an ORF end from 336 to
    306, and certifying through 336 waves ten unsupported residues past the
    gate.
    """
    c_of = cres.cons_of_frame
    n = c_of.shape[0]
    col = int(np.clip(column, 0, n - 1))
    step = -1 if forward else 1
    while 0 <= col < n and c_of[col] < 0:
        col += step
    if not (0 <= col < n):
        return len(cres.consensus) if forward else 0
    return int(c_of[col])


def _ALLELE_CHAR(v: int) -> str:
    """One character per signature column, for the human-facing node label."""
    if v == cv.MAJOR:
        return "="
    if v == cv.UNOBSERVED:
        return "."
    if v == cv.UNCOVERED:
        return "~"
    return "ACGT-"[v - 1]


def _sig_eps(cand, signatures) -> np.ndarray:
    """Per-column epsilon over the concatenated signature columns."""
    if not signatures:
        return np.zeros(0, dtype=np.float64)
    return np.clip(
        np.concatenate([cand.eps[s] for s in signatures]), 1e-9, 0.5 - 1e-9
    )


def _log_odds(eps: np.ndarray) -> np.ndarray:
    """``log((1 − ε)/ε)`` — the per-column weight of a disagreement."""
    return np.log((1.0 - eps) / eps) if eps.size else np.zeros(0, dtype=np.float64)


def _support_span(cres: ConsensusResult) -> tuple[int, int]:
    """``[lo, hi)`` of the child's consensus that its own reads cover.

    A child built on the POOLED column plan still has a column for every
    position of the parent, and ``_call_winner`` falls back to the template
    base where nobody voted — so a node whose reads all start 200 nt in would
    otherwise emit the parent's first 200 nt as its own reference, sequence no
    read of that node ever supported. Trimming to the covered span is what
    makes consensus **length** an output of the signature, and it is the
    direct answer to references inflating without bound (mean 1,558 → 2,346 nt
    across six rounds while their ORFs shrank).

    Only the flanks are trimmed. An interior hole means the node spans two
    disjoint regions, and splicing those together would fabricate a junction —
    strictly worse than carrying the parent's base through it.
    """
    f = cres.frame_of_cons
    if f is None or f.size == 0:
        return 0, 0
    covered = cres.pwm[f].sum(axis=1) > 0
    if not covered.any():
        return 0, 0
    lo = int(np.argmax(covered))
    hi = covered.shape[0] - int(np.argmax(covered[::-1]))
    return lo, hi


def _tuple_patterns(assignments, tuples: np.ndarray) -> np.ndarray:
    """``(T, ΣS)`` concatenated pattern vector for each distinct state-tuple."""
    if not assignments:
        return np.zeros((tuples.shape[0], 0), dtype=np.int8)
    return np.concatenate(
        [a.patterns[tuples[:, j]] for j, a in enumerate(assignments)], axis=1
    )


def _fold_small_tuples(
    keys: np.ndarray,
    mass: np.ndarray,
    inverse: np.ndarray,
    patterns: np.ndarray,
    weights: np.ndarray,
    column_weight: np.ndarray,
    *,
    min_node_reads: float,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the heaviest tuples; send the rest to the nearest survivor.

    "Nearest" is the same ε-weighted metric §3g assigns reads by, applied one
    level up — which is the point of deleting modal absorption. Folding an
    unsupported combination into the *modal* node regardless of distance is
    what let a read that actively disagreed be counted as agreeing.
    """
    order = np.argsort(-mass, kind="stable")
    survivors = [int(t) for t in order if mass[t] >= min_node_reads][:max_nodes]
    if not survivors:
        survivors = [int(order[0])]
    surv = np.asarray(survivors, dtype=np.int64)
    remap = np.empty(keys.shape[0], dtype=np.int64)
    remap[surv] = np.arange(surv.shape[0])
    lost = np.setdiff1d(np.arange(keys.shape[0]), surv, assume_unique=False)
    if lost.size:
        diff = patterns[lost][:, None, :] != patterns[surv][None, :, :]
        remap[lost] = np.argmin(diff @ column_weight, axis=1)
    labels = remap[inverse]
    new_mass = np.bincount(labels, weights=weights, minlength=surv.shape[0])
    return labels, new_mass, surv


def refine_template(
    frame: str,
    members: list[MemberSpec],
    *,
    template_id: int = 0,
    seed_orf: tuple[int, int] | None = None,
    n_assigned: float | None = None,
    error_model: ErrorModel | None = None,
    overdispersion: float = 0.01,
    eps_floor: float = 0.0,
    f_min: float = 0.02,
    a_min: float = 3.0,
    q_candidate: float = 0.01,
    q_edge: float = 0.01,
    gamma: float = 0.6,
    local_window: int = 200,
    boundary_tolerance: int = 10,
    max_candidate_columns: int = 512,
    max_state_per_read: int = 64,
    max_nodes: int = 8,
    min_node_reads: float = 2.0,
    min_aa_length: int = 30,
    support_min_depth: float = 3.0,
    support_min_agreement: float = 0.6,
    fold_insertions: bool = True,
    min_insertion_support: float = 2.0,
    min_extension_support: float | None = 3.0,
) -> list[RefinedTemplate]:
    """Refine one template into its state-tuple nodes.

    Every annealing knob is an explicit per-round argument so the round loop
    in the bench driver can schedule them; there is no ``**kwargs`` for one to
    hide in.
    """
    if not members:
        return []
    model = error_model or ErrorModel()
    pooled = frame_consensus(
        frame,
        members,
        frame_weight=0.0,
        fold_insertions=fold_insertions,
        min_insertion_support=min_insertion_support,
        min_extension_support=min_extension_support,
    )
    w = np.array([float(m.weight) for m in members], dtype=np.float64)
    total_reads = float(w.sum())

    # The seed ORF interval arrives in TEMPLATE coordinates; resolve it once
    # into the shared PWM column space, where every child can translate it
    # into its own consensus.
    t_start, t_end = seed_orf if seed_orf is not None else (0, 0)
    t_at = pooled.plan.template_at
    seed_col_start = (
        int(t_at[np.clip(t_start, 0, t_at.shape[0] - 1)]) if t_at.size else 0
    )
    seed_col_end = (
        int(t_at[np.clip(t_end - 1, 0, t_at.shape[0] - 1)]) + 1 if t_at.size else 0
    )

    stats = column_stats(
        pooled,
        model=model,
        n_assigned=float(n_assigned if n_assigned is not None else total_reads),
        eps_floor=eps_floor,
        local_window=local_window,
        boundary_tolerance=boundary_tolerance,
    )
    cand = candidate_columns(
        stats,
        f_min=f_min,
        a_min=a_min,
        q_candidate=q_candidate,
        overdispersion=overdispersion,
        max_columns=max_candidate_columns,
    )

    diag: dict = {
        "n_reads": total_reads,
        "n_columns": int(stats.n_columns),
        "n_candidates": int(cand.columns.size),
        "n_allelic": cand.n_allelic,
        "n_coverage": cand.n_coverage,
        "n_collapsed": cand.n_collapsed,
        "n_candidates_capped": cand.n_capped,
        "disagreements": disagreement_stats(pooled),
    }

    signatures: list[np.ndarray] = []
    assignments: list[cv.Assignment] = []
    if cand.columns.size:
        events = member_allele_events(pooled, cand.columns, major=cand.major)
        states = cv.read_states(events, cand, w)
        graph = cv.covariance_graph(
            states,
            cand,
            q_edge=q_edge,
            max_state_per_read=max_state_per_read,
        )
        diag.update(
            n_pairs_seen=graph.n_pairs_seen,
            n_tested_pairs=graph.n_tested,
            n_significant_edges=graph.n_significant,
            n_retained=int(graph.keep.sum()),
            n_reads_capped=graph.n_reads_capped,
            sum_k2=graph.sum_k2,
        )
        signatures = cv.quasi_cliques(graph, gamma=gamma)
        assignments = [
            cv.resolve_signature(
                states, s, eps=cand.eps[s], a_min=a_min, max_patterns=max_nodes
            )
            for s in signatures
        ]
    diag["n_signatures"] = len(signatures)

    declared = (
        np.concatenate([cand.columns[s] for s in signatures])
        if signatures
        else np.empty(0, dtype=np.int64)
    )
    declared.sort()

    def _node(hid: int, idx: np.ndarray, allele: str) -> RefinedTemplate:
        hap = [members[int(i)] for i in idx]
        # Children are built on the POOLED column plan. On their own plans each
        # would have a different column space, and a position taken from the
        # parent would address a different base in every child.
        cres = (
            pooled
            if len(hap) == len(members)
            else frame_consensus(
                frame,
                hap,
                frame_weight=0.0,
                plan=pooled.plan,
                fold_insertions=fold_insertions,
                min_insertion_support=min_insertion_support,
                min_extension_support=min_extension_support,
            )
        )
        certified = certified_columns(
            cres, min_depth=support_min_depth, min_agreement=support_min_agreement
        )
        lo, hi = _support_span(cres)
        consensus = cres.consensus[lo:hi]
        certified = certified[lo:hi]
        # Declared columns are PWM columns; each child reports them in its own
        # consensus coordinates and drops the ones it has no base for (a
        # minority insertion exists for the node that carries it and nowhere
        # else — which is exactly the right answer), and the ones its own
        # reads do not reach.
        child_cols = np.array(
            [
                int(cres.cons_of_frame[c]) - lo
                for c in declared
                if lo <= cres.cons_of_frame[c] < hi
            ],
            dtype=np.int64,
        )
        node = RefinedTemplate(
            parent_template_id=template_id,
            haplotype_id=hid,
            consensus=consensus,
            n_reads=int(round(float(w[idx].sum()))),
            node_weight=float(w[idx].sum()),
            allele_string=allele,
            declared_variants=child_cols,
            n_inserted_columns=cres.n_inserted_columns,
            n_extended_5p=cres.n_extended_5p,
            n_extended_3p=cres.n_extended_3p,
            n_trimmed_5p=lo,
            n_trimmed_3p=len(cres.consensus) - hi,
            stats=diag if hid == 0 else {},
        )
        span = hi - lo
        orf = gated_orf(
            consensus,
            certified,
            seed_orf_start=int(
                np.clip(_child_position(cres, seed_col_start, forward=False) - lo,
                        0, span)
            ),
            seed_orf_end=int(
                np.clip(_child_position(cres, seed_col_end, forward=True) - lo,
                        0, span)
            ),
            min_aa_length=min_aa_length,
        )
        if orf is not None:
            (
                node.protein,
                node.orf_start,
                node.orf_end,
                node.orf_certified_end,
                node.orf_is_truncated_by_support,
            ) = orf
        return node

    if not assignments:
        diag["n_nodes"] = 1
        return [_node(0, np.arange(len(members)), "")]

    # One node per observed state-tuple. Not per signature: two independent
    # signatures give up to four templates, and two marginals would each count
    # every read once — twice in total.
    state = np.stack([a.labels for a in assignments], axis=1)
    keys, inverse = np.unique(state, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    mass = np.bincount(inverse, weights=w, minlength=keys.shape[0])
    labels, new_mass, surv = _fold_small_tuples(
        keys,
        mass,
        inverse,
        _tuple_patterns(assignments, keys),
        w,
        _log_odds(_sig_eps(cand, signatures)),
        min_node_reads=min_node_reads,
        max_nodes=max_nodes,
    )
    diag["n_nodes"] = int(surv.shape[0])
    diag["n_tuples_observed"] = int(keys.shape[0])

    pats = _tuple_patterns(assignments, keys[surv])
    order = np.argsort(-new_mass, kind="stable")
    out: list[RefinedTemplate] = []
    for hid, t in enumerate(order.tolist()):
        allele = "".join(_ALLELE_CHAR(int(v)) for v in pats[t])
        out.append(_node(hid, np.flatnonzero(labels == t), allele))
    return out


__all__ = [
    "RefinedTemplate",
    "certified_columns",
    "gated_orf",
    "refine_template",
    "specs_from_assignments",
]
