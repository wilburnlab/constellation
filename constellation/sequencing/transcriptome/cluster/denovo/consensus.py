"""Frame-anchored consensus over a pre-planned PWM column space (numpy scatter).

Members are projected onto a column space **planned in advance from their own
CIGARs**, and their bases accumulate into an abundance-weighted ``(F, 5)``
position-weight matrix (columns A/C/G/T/gap). The consensus is the
per-position argmax; gap-winning columns drop out, columns with no coverage
fall back to the frame base.

The column space is wider than the template: wherever a supported group of
members carries an insertion, a block of columns is reserved for it before
the template column it precedes, and every member covering that junction
votes there — a base if it inserts, a **gap** if it does not. Terminal blocks
(a member's unaligned 5'/3' flanks) are the same mechanism, so "end
extension" and "insertion folding" are one thing, not two.

**Why plan the columns rather than vote them in.** The obvious alternative —
build the PWM, find junctions where a majority inserts, splice those in and
re-align — produces a good consensus but a bad *haplotype substrate*. It
decides an insertion's fate by majority **before** the variant caller and the
haplotype resolver ever see it, so a 25%-frequency insertion is voted out of
existence and can never define a node, while the identical minority
*deletion* resolves fine (it is just a gap vote in a column that already
exists). That asymmetry turns on nothing but which direction the seed read
happens to differ from its own variants, which for a pipeline whose product
is proteoform nodes is arbitrary in the worst way. Reserving the column
instead leaves the decision where it belongs: the argmax decides what reaches
the consensus, and the resolver decides what defines a haplotype.

It is also strictly less machinery — one pass, no re-alignment, no iteration,
and no insertion-specific thresholds beyond the support floor that sizes each
block.

Two consequences worth knowing:

* An insertion column is **majority-gap by construction** whenever the
  insertion is a minority, so anything that gates on "is this column in the
  consensus" will skip it. ``variants.py`` deliberately does not.
* A minority insertion contributes columns that no one keeps. That costs
  memory, not correctness, and ``min_insertion_support`` bounds it: a
  singleton insertion reserves nothing, so a chimeric read cannot widen a
  template it does not belong to. Its own misfit is the E-step's problem —
  a read carrying sequence no template explains should fail the band and
  seed its own template.

Three coordinate systems, and the maps between them are the public contract:

===========  ========  ==================================================
space        length    indexes
===========  ========  ==================================================
template     ``T``     the frame string handed in. Provenance only.
frame        ``F≥T``   the planned column space. ``pwm`` rows / ``winner``.
consensus    ``C≤F``   the emitted string (gap-winning columns removed).
===========  ========  ==================================================

Planning only ever *adds* columns, so ``F >= T`` always. Use
:func:`frame_of_consensus` / :func:`consensus_of_frame` rather than
open-coding the maps — the two differ on the unresolved-``N`` column
(``_AMBIG``), which is a consensus column but not a base, and getting that
wrong silently misaligns every variant downstream of the first ``N``. To read
a member's allele at a PWM column use :func:`member_alleles`; a
:class:`FrameAlignment`'s coordinates are **template** columns and walking one
against the expanded space is the same class of bug.

Kept torch-free (numpy + edlib only): the per-cluster pool is fork-based and
torch after ``fork()`` deadlocks on OpenMP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo._cigar import (
    base_codes,
    parse_cigar,
)


_IDX_TO_BASE_NP = np.array(
    [ord("A"), ord("C"), ord("G"), ord("T"), ord("-"), ord("N")], dtype=np.uint8
)
_BASE_CHARS = "ACGT"
_GAP = 4
_AMBIG = 5  # renders as 'N' via _IDX_TO_BASE_NP

# column_kind vocabulary
COL_TEMPLATE = 0
COL_INSERTED = 1
COL_EXT_5P = 2
COL_EXT_3P = 3


# ── coordinate maps ───────────────────────────────────────────────────


def frame_of_consensus(winner: np.ndarray) -> np.ndarray:
    """``(C,)`` map consensus position → frame column."""
    return np.flatnonzero(winner != _GAP)


def consensus_of_frame(winner: np.ndarray) -> np.ndarray:
    """``(F,)`` map frame column → consensus position (``-1`` on dropped)."""
    keep = winner != _GAP
    return np.where(keep, np.cumsum(keep) - 1, -1)


# ── types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MemberSpec:
    """One member's alignment onto the frame (from the cache).

    ``centroid_is_query`` records which side of the edlib alignment the frame
    was, and ``ref_start`` is the shorter sequence's start on the longer.
    ``member_start`` overrides the offset derivation for producers that give
    both offsets explicitly (minimap2's ``q_start`` alongside ``t_start``);
    ``None`` keeps the edlib rule.
    """

    member_seq: str
    weight: float
    cigar: str
    centroid_is_query: bool  # True iff the frame was the alignment query
    ref_start: int  # short's start on long (the cached alignment's ref_start)
    member_start: int | None = None
    member_id: int = -1


@dataclass(frozen=True, slots=True)
class FrameAlignment:
    """A member's alignment against the **template**, normalised to
    member-as-query / template-as-ref — so ``I`` is always a member insertion
    and ``D`` always a member deletion, whichever way the aligner ran.

    The coordinates are **template** columns, not expanded-frame columns:
    the PWM's column space contains insertion blocks the template does not,
    so ``template_start`` is not an index into ``ConsensusResult.frame``.
    To read a member's allele at a PWM column use :func:`member_alleles`,
    which resolves the two spaces for you rather than leaving a caller to
    walk a CIGAR in the wrong coordinate system."""

    member_id: int
    weight: float
    cigar: str
    template_start: int
    template_end: int  # exclusive
    member_start: int
    member_end: int  # exclusive
    member_seq: str = field(repr=False, default="")


@dataclass(frozen=True, slots=True)
class ProjectedMember:
    """A member's votes on the frame, plus the events the PWM cannot hold."""

    match_fpos: np.ndarray  # int64 — frame columns receiving a base vote
    match_base: np.ndarray  # int64 — the base codes
    gap_fpos: np.ndarray  # int64 — frame columns receiving a deletion vote
    # (junction the run precedes, inserted base codes) per insertion run
    ins_events: list[tuple[int, np.ndarray]]
    frame_start: int
    frame_end: int  # exclusive
    member_start: int
    member_end: int  # exclusive


@dataclass(slots=True)
class ConsensusResult:
    """Per-cluster consensus + the PWM it was argmax'd from.

    ``pwm`` is the abundance-weighted ``(F, 5)`` matrix in **frame**
    coordinates (used for variant calling). ``winner`` is the per-frame-column
    winning base (0-3 = ACGT, 4 = gap/dropped, 5 = unresolved N) before gap
    columns are removed.

    The first three fields are positional and in their historical order, so
    callers constructing a result by hand keep working; the maps are derived
    in ``__post_init__`` when they are not supplied.
    """

    consensus: str
    pwm: np.ndarray  # (F, 5) float64 abundance-weighted PWM in frame coords
    winner: np.ndarray = field(repr=False)
    frame: str = ""
    frame_of_cons: np.ndarray | None = field(repr=False, default=None)
    cons_of_frame: np.ndarray | None = field(repr=False, default=None)
    template_of_frame: np.ndarray | None = field(repr=False, default=None)
    column_kind: np.ndarray | None = field(repr=False, default=None)
    alignments: list[FrameAlignment] = field(repr=False, default_factory=list)
    n_passes: int = 1
    n_inserted_columns: int = 0
    n_extended_5p: int = 0
    n_extended_3p: int = 0
    n_realign_failures: int = 0
    plan: object | None = field(repr=False, default=None)
    members: object = field(repr=False, default=())
    # Columns the plan *created*; the n_* counters above report how many
    # survived the argmax into the consensus.
    n_columns_planned: int = 0

    def __post_init__(self) -> None:
        if self.frame_of_cons is None:
            self.frame_of_cons = frame_of_consensus(self.winner)
        if self.cons_of_frame is None:
            self.cons_of_frame = consensus_of_frame(self.winner)
        f = self.winner.shape[0]
        if self.template_of_frame is None:
            self.template_of_frame = np.arange(f, dtype=np.int64)
        if self.column_kind is None:
            self.column_kind = np.full(f, COL_TEMPLATE, dtype=np.int8)


# ── projection ────────────────────────────────────────────────────────


def _offsets(spec: MemberSpec) -> tuple[int, int]:
    """``(frame_start, member_start)`` for a spec's cached alignment."""
    if spec.member_start is not None:
        return spec.ref_start, spec.member_start
    if spec.centroid_is_query:
        return 0, spec.ref_start
    return spec.ref_start, 0


def project_member_events(
    cigar_ops: list[tuple[int, str]],
    member_codes: np.ndarray,
    *,
    frame_is_query: bool,
    frame_start: int,
    member_start: int,
) -> ProjectedMember:
    """Project a member onto the frame, keeping insertion events.

    Superset of :func:`project_member`: the ops that consume the member only
    become ``ins_events`` instead of being discarded, and the aligned
    intervals on both sequences are reported so the caller can find each
    member's covering span and its unaligned flanks.
    """
    # Which op consumes the frame only (→ gap) vs the member only (→ ins).
    gap_op = "I" if frame_is_query else "D"
    ins_op = "D" if frame_is_query else "I"

    match_fpos: list[np.ndarray] = []
    match_base: list[np.ndarray] = []
    gap_fpos: list[np.ndarray] = []
    ins_events: list[tuple[int, np.ndarray]] = []
    fpos = frame_start
    mpos = member_start
    for length, op in cigar_ops:
        if op in ("=", "X", "M"):
            match_fpos.append(np.arange(fpos, fpos + length, dtype=np.int64))
            match_base.append(member_codes[mpos : mpos + length])
            fpos += length
            mpos += length
        elif op == gap_op:
            gap_fpos.append(np.arange(fpos, fpos + length, dtype=np.int64))
            fpos += length
        elif op == ins_op:
            # Keyed by the frame column the run precedes: after consuming
            # `fpos` frame columns the inserted bases sit between column
            # fpos-1 and column fpos, i.e. at junction `fpos`.
            ins_events.append((fpos, member_codes[mpos : mpos + length]))
            mpos += length
    mf = np.concatenate(match_fpos) if match_fpos else np.empty(0, dtype=np.int64)
    mb = (
        np.concatenate(match_base).astype(np.int64)
        if match_base
        else np.empty(0, dtype=np.int64)
    )
    gf = np.concatenate(gap_fpos) if gap_fpos else np.empty(0, dtype=np.int64)
    return ProjectedMember(
        match_fpos=mf,
        match_base=mb,
        gap_fpos=gf,
        ins_events=ins_events,
        frame_start=frame_start,
        frame_end=fpos,
        member_start=member_start,
        member_end=mpos,
    )


def project_member(
    cigar_ops: list[tuple[int, str]],
    member_codes: np.ndarray,
    *,
    frame_is_query: bool,
    frame_start: int,
    member_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a member onto the frame.

    Returns ``(match_fpos, match_base, gap_fpos)``: frame positions receiving
    a member base vote + the base codes, and frame positions receiving a gap
    (deletion) vote. Insertions relative to the frame are dropped — callers
    that need them (the consensus kernel) use
    :func:`project_member_events`.
    """
    p = project_member_events(
        cigar_ops,
        member_codes,
        frame_is_query=frame_is_query,
        frame_start=frame_start,
        member_start=member_start,
    )
    return p.match_fpos, p.match_base, p.gap_fpos


# ── re-alignment ──────────────────────────────────────────────────────

_CIGAR_TRANSPOSE = str.maketrans("ID", "DI")


def default_realign(
    frame_seq: str, member_seq: str, *, max_frac: float = 0.25
) -> tuple[str, int, int] | None:
    """Re-align one member to a frame; ``(cigar, frame_start, member_start)``.

    Normalised to member-as-query, so the returned CIGAR's ``I`` is always a
    member insertion. Shorter-as-query matches ``verify._order_pair`` so the
    verify and consensus stages agree on orientation. Returns ``None`` when
    the alignment exceeds the budget.
    """
    import edlib

    if not frame_seq or not member_seq:
        return None
    budget = max(1, int(max_frac * min(len(frame_seq), len(member_seq))))
    if len(member_seq) <= len(frame_seq):
        res = edlib.align(member_seq, frame_seq, mode="HW", task="path", k=budget)
        if res["editDistance"] < 0 or not res["locations"]:
            return None
        return (res["cigar"] or ""), int(res["locations"][0][0]), 0
    res = edlib.align(frame_seq, member_seq, mode="HW", task="path", k=budget)
    if res["editDistance"] < 0 or not res["locations"]:
        return None
    # Frame was the query — transpose I↔D into the member-as-query frame.
    cig = (res["cigar"] or "").translate(_CIGAR_TRANSPOSE)
    return cig, 0, int(res["locations"][0][0])


# ── the kernel ────────────────────────────────────────────────────────


def _weighted_mode(codes: np.ndarray, weights: np.ndarray, n_bins: int) -> int:
    """Weighted argmax over small non-negative integer codes; -1 if empty."""
    if codes.size == 0:
        return -1
    acc = np.bincount(codes, weights=weights, minlength=n_bins)
    if acc.max() <= 0.0:
        return -1
    return int(np.argmax(acc))


def _build_pwm(
    frame_codes: np.ndarray,
    frame_weight: float,
    projections: Sequence[tuple[ProjectedMember, float]],
) -> np.ndarray:
    """Scatter-add every vote into the ``(F, 5)`` PWM."""
    f = frame_codes.shape[0]
    flat_idx: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    if frame_weight:
        fpos0 = np.arange(f, dtype=np.int64)
        valid0 = frame_codes < 4
        flat_idx.append(fpos0[valid0] * 5 + frame_codes[valid0])
        weights.append(np.full(int(valid0.sum()), frame_weight, dtype=np.float64))

    for proj, w in projections:
        mf, mb, gf = proj.match_fpos, proj.match_base, proj.gap_fpos
        # Keep votes inside the frame with an unambiguous base.
        good = (mf >= 0) & (mf < f) & (mb < 4)
        flat_idx.append(mf[good] * 5 + mb[good])
        weights.append(np.full(int(good.sum()), w, dtype=np.float64))
        gf_in = gf[(gf >= 0) & (gf < f)]
        if gf_in.shape[0]:
            flat_idx.append(gf_in * 5 + _GAP)
            weights.append(np.full(gf_in.shape[0], w, dtype=np.float64))

    # numpy bincount, not torch — this runs inside fork()ed workers, and the
    # parent has already spawned torch's thread pool during minimizer
    # extraction, so a torch op here deadlocks on OpenMP.
    if flat_idx:
        flat = np.bincount(
            np.concatenate(flat_idx),
            weights=np.concatenate(weights),
            minlength=f * 5,
        )
    else:
        flat = np.zeros(f * 5, dtype=np.float64)
    return flat.reshape(f, 5)


def _call_winner(pwm: np.ndarray, frame_codes: np.ndarray) -> np.ndarray:
    """Per-column winning code: 0-3 base, 4 gap (dropped), 5 unresolved N."""
    row_max = pwm.max(axis=1)
    winner = np.where(row_max <= 0.0, frame_codes, pwm.argmax(axis=1))
    # An N the frame carries and no member covers is UNRESOLVED, not a
    # deletion. base_codes encodes both as 4, so such a column used to be
    # dropped: two 300-base sequences sharing an N produced a 299-base
    # consensus though neither had a deletion, shifting every downstream
    # coordinate and potentially the ORF reading frame. Emit N and keep the
    # column; only a genuine gap-winning column is removed.
    return np.where((row_max <= 0.0) & (frame_codes >= _GAP), _AMBIG, winner)

@dataclass(frozen=True, slots=True)
class ColumnPlan:
    """The expanded column space the PWM is built over.

    ``block_width[j]`` insertion columns sit immediately before template
    column ``j`` (and ``block_width[T]`` after the last one), so the frame
    can hold every insertion any supported group of members carries. Nothing
    votes them in or out at planning time — they are ordinary PWM columns,
    and the per-column argmax decides whether each survives into the
    consensus.
    """

    n_columns: int
    n_template: int
    template_at: np.ndarray  # (T,) expanded index of each template column
    block_start: np.ndarray  # (T+1,) expanded index each junction's block begins
    block_width: np.ndarray  # (T+1,) columns in each junction's block


def _member_events(
    proj: ProjectedMember, member_codes: np.ndarray, n_template: int
) -> list[tuple[int, np.ndarray]]:
    """A member's insertion events, with its unaligned flanks folded in.

    A 5' flank and an insertion at junction 0 are the same thing physically —
    bases the member carries before the template starts — and differ only in
    whether the aligner encoded them as a soft clip or an ``I`` op. Treating
    them as one event class is what collapses "end extension" and "insertion
    folding" into a single mechanism.
    """
    events = [(j, c) for j, c in proj.ins_events if c.size]
    if proj.frame_start == 0 and proj.member_start > 0:
        events.append((0, member_codes[: proj.member_start]))
    if proj.frame_end == n_template and proj.member_end < member_codes.shape[0]:
        events.append((n_template, member_codes[proj.member_end :]))
    return events


def plan_columns(
    projections: Sequence[tuple[ProjectedMember, float]],
    member_codes: Sequence[np.ndarray],
    n_template: int,
    *,
    min_support: float = 2.0,
    max_block: int | None = None,
) -> ColumnPlan:
    """Size the insertion block at every junction from the members' CIGARs.

    A junction's block is as wide as the longest insertion carried by members
    of at least ``min_support`` total weight — "replication is a certificate"
    applied to length. One read's 300-nt insertion therefore creates no
    columns at all, which is the only bound this needs: a read carrying an
    insertion no template explains fails the E-step's band and spawns its own
    template rather than quietly widening someone else's. ``max_block`` is a
    belt-and-braces cap and is normally ``None``.
    """
    width = np.zeros(n_template + 1, dtype=np.int64)
    if projections:
        by_junction: dict[int, list[tuple[int, float]]] = {}
        for (proj, w), codes in zip(projections, member_codes):
            for j, c in _member_events(proj, codes, n_template):
                if 0 <= j <= n_template:
                    by_junction.setdefault(j, []).append((int(c.size), float(w)))
        for j, evs in by_junction.items():
            lengths = np.array([ln for ln, _ in evs], dtype=np.int64)
            weights = np.array([w for _, w in evs], dtype=np.float64)
            order = np.argsort(-lengths)
            # support(L) = weight of every event at least L long.
            cum = np.cumsum(weights[order])
            ok = np.flatnonzero(cum >= min_support)
            if ok.size:
                width[j] = int(lengths[order][ok[0]])
        if max_block is not None:
            np.minimum(width, max_block, out=width)

    template_at = np.zeros(n_template, dtype=np.int64)
    block_start = np.zeros(n_template + 1, dtype=np.int64)
    pos = 0
    for t in range(n_template):
        block_start[t] = pos
        pos += int(width[t])
        template_at[t] = pos
        pos += 1
    block_start[n_template] = pos
    pos += int(width[n_template])
    return ColumnPlan(
        n_columns=pos,
        n_template=n_template,
        template_at=template_at,
        block_start=block_start,
        block_width=width,
    )


def _scatter_expanded(
    plan: ColumnPlan,
    frame_codes: np.ndarray,
    frame_weight: float,
    projections: Sequence[tuple[ProjectedMember, float]],
    member_codes: Sequence[np.ndarray],
) -> np.ndarray:
    """Accumulate every vote into the ``(n_columns, 5)`` PWM."""
    n = plan.n_columns
    t_at, b_start, b_width = plan.template_at, plan.block_start, plan.block_width
    active = np.flatnonzero(b_width > 0)
    flat_idx: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    def _add(idx: np.ndarray, codes: np.ndarray, w: float) -> None:
        good = (idx >= 0) & (idx < n) & (codes < 4)
        if good.any():
            flat_idx.append(idx[good] * 5 + codes[good])
            weights.append(np.full(int(good.sum()), w, dtype=np.float64))

    def _add_gaps(idx: np.ndarray, w: float) -> None:
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.shape[0]:
            flat_idx.append(idx * 5 + _GAP)
            weights.append(np.full(idx.shape[0], w, dtype=np.float64))

    if frame_weight:
        _add(t_at, frame_codes, frame_weight)

    for (proj, w), codes in zip(projections, member_codes):
        mf, mb, gf = proj.match_fpos, proj.match_base, proj.gap_fpos
        keep = (mf >= 0) & (mf < plan.n_template)
        _add(t_at[mf[keep]], mb[keep], w)
        gk = gf[(gf >= 0) & (gf < plan.n_template)]
        _add_gaps(t_at[gk], w)

        # Insertion blocks. Everything the member covers but does not fill
        # votes gap, so a block column's depth is the covering depth and its
        # minor-allele fraction is the fraction of covering reads that insert.
        filled = {j: c for j, c in _member_events(proj, codes, plan.n_template)}
        for j in active:
            j = int(j)
            if not (proj.frame_start <= j <= proj.frame_end):
                continue  # member never reaches this junction — no vote
            wdt = int(b_width[j])
            c = filled.get(j)
            k = 0 if c is None else min(int(c.size), wdt)
            if k:
                # Interior and 3'-terminal blocks abut the column on their
                # left, so they left-justify; junction 0 has no left neighbour
                # and abuts template column 0, so it right-justifies.
                if j == 0:
                    at = b_start[j] + wdt - k
                    used = c[-k:]
                else:
                    at = b_start[j]
                    used = c[:k]
                _add(np.arange(at, at + k, dtype=np.int64), used, w)
            if k < wdt:
                empty = (
                    np.arange(b_start[j], b_start[j] + wdt - k, dtype=np.int64)
                    if j == 0
                    else np.arange(b_start[j] + k, b_start[j] + wdt, dtype=np.int64)
                )
                _add_gaps(empty, w)

    # numpy bincount, not torch — this runs inside fork()ed workers, and the
    # parent has already spawned torch's thread pool during minimizer
    # extraction, so a torch op here deadlocks on OpenMP.
    if flat_idx:
        flat = np.bincount(
            np.concatenate(flat_idx),
            weights=np.concatenate(weights),
            minlength=n * 5,
        )
    else:
        flat = np.zeros(n * 5, dtype=np.float64)
    return flat.reshape(n, 5)


def frame_consensus(
    frame_seq: str,
    members: Sequence[MemberSpec],
    *,
    frame_weight: float = 0.0,
    fold_insertions: bool = True,
    min_insertion_support: float = 2.0,
    max_insertion_block: int | None = None,
    template_of_frame: np.ndarray | None = None,
    **_legacy,
) -> ConsensusResult:
    """Build the abundance-weighted consensus for one cluster.

    The PWM is built over a column space planned in advance from the members'
    own CIGARs (:func:`plan_columns`), so every insertion a supported group
    carries **has a column** whether or not it is the majority there. That is
    the difference between a consensus builder and a haplotype substrate: a
    25%-frequency insertion has to survive as a column for the variant caller
    and the haplotype resolver to see it at all, and voting it away before
    they run makes a minority insertion unresolvable while the same minority
    *deletion* resolves fine — an asymmetry that would depend on nothing more
    than which direction the seed read happened to differ.

    One pass, no re-alignment: the plan accommodates every member up front,
    so there is no frame to grow into and iterate over.

    ``frame_weight`` is the frame's own self-vote and defaults to **0** — in
    the EM M-step the frame is a scaffold, not an observation: in round 1 the
    seed read is also an assigned member (a self-vote double-counts it), and
    in later rounds the frame is a consensus with no reads of its own. It also
    keeps the PWM column sums equal to raw read multiplicity, which
    ``variants.py`` relies on as its binomial ``n``.
    """
    n_template = len(frame_seq)
    frame_codes = base_codes(frame_seq).astype(np.int64)
    codes_of = [base_codes(m.member_seq) for m in members]

    projections: list[tuple[ProjectedMember, float]] = []
    for spec, codes in zip(members, codes_of):
        fstart, mstart = _offsets(spec)
        projections.append(
            (
                project_member_events(
                    parse_cigar(spec.cigar),
                    codes,
                    frame_is_query=spec.centroid_is_query,
                    frame_start=fstart,
                    member_start=mstart,
                ),
                float(spec.weight),
            )
        )

    plan = (
        plan_columns(
            projections,
            codes_of,
            n_template,
            min_support=min_insertion_support,
            max_block=max_insertion_block,
        )
        if fold_insertions
        else ColumnPlan(
            n_columns=n_template,
            n_template=n_template,
            template_at=np.arange(n_template, dtype=np.int64),
            block_start=np.zeros(n_template + 1, dtype=np.int64),
            block_width=np.zeros(n_template + 1, dtype=np.int64),
        )
    )

    pwm = _scatter_expanded(plan, frame_codes, frame_weight, projections, codes_of)

    # Frame bases occupy their own columns; inserted columns have no frame
    # base, so an uncovered one is a gap rather than a fallback.
    expanded_codes = np.full(plan.n_columns, _GAP, dtype=np.int64)
    expanded_codes[plan.template_at] = frame_codes
    winner = _call_winner(pwm, expanded_codes)

    tmpl_src = (
        np.arange(n_template, dtype=np.int64)
        if template_of_frame is None
        else np.asarray(template_of_frame, dtype=np.int64)
    )
    tmpl = np.full(plan.n_columns, -1, dtype=np.int64)
    tmpl[plan.template_at] = tmpl_src
    kind = np.full(plan.n_columns, COL_INSERTED, dtype=np.int8)
    kind[plan.template_at] = COL_TEMPLATE
    if plan.block_width[0]:
        kind[: plan.block_width[0]] = COL_EXT_5P
    if plan.block_width[n_template]:
        kind[plan.block_start[n_template] :] = COL_EXT_3P

    keep = winner != _GAP
    consensus = _IDX_TO_BASE_NP[winner[keep]].tobytes().decode("ascii")
    frame = _IDX_TO_BASE_NP[np.where(winner < 4, winner, expanded_codes)].tobytes()

    alignments = [
        FrameAlignment(
            member_id=members[i].member_id,
            weight=float(members[i].weight),
            cigar=(
                members[i].cigar.translate(_CIGAR_TRANSPOSE)
                if members[i].centroid_is_query
                else members[i].cigar
            ),
            template_start=proj.frame_start,
            template_end=proj.frame_end,
            member_start=proj.member_start,
            member_end=proj.member_end,
            member_seq=members[i].member_seq,
        )
        for i, (proj, _w) in enumerate(projections)
    ]
    # The counters report what reached the CONSENSUS, not what the plan
    # created. A block column only exists so the variant caller and the
    # haplotype resolver can see the allele; whether its base is part of the
    # consensus is the ordinary argmax's decision, and that is the number a
    # reader means by "bases the consensus gained".
    kept_kind = kind[keep]
    return ConsensusResult(
        consensus=consensus,
        pwm=pwm,
        winner=winner,
        frame=frame.decode("ascii"),
        frame_of_cons=frame_of_consensus(winner),
        cons_of_frame=consensus_of_frame(winner),
        template_of_frame=tmpl,
        column_kind=kind,
        alignments=alignments,
        plan=plan,
        members=tuple(members),
        n_passes=1,
        n_columns_planned=int(plan.n_columns),
        n_inserted_columns=int((kept_kind == COL_INSERTED).sum()),
        n_extended_5p=int((kept_kind == COL_EXT_5P).sum()),
        n_extended_3p=int((kept_kind == COL_EXT_3P).sum()),
    )


def _classify_columns(plan: ColumnPlan, cols: np.ndarray):
    """Split requested PWM columns into template columns and block columns.

    Returns ``(template_of, junction_of, offset_of)``, each ``(V,)`` with -1
    where the column is not of that kind.
    """
    inv_t = np.full(plan.n_columns, -1, dtype=np.int64)
    inv_t[plan.template_at] = np.arange(plan.n_template, dtype=np.int64)
    t_of = np.where((cols >= 0) & (cols < plan.n_columns), inv_t[cols], -1)
    j_of = np.full(cols.shape[0], -1, dtype=np.int64)
    o_of = np.full(cols.shape[0], -1, dtype=np.int64)
    for j in np.flatnonzero(plan.block_width > 0):
        lo = int(plan.block_start[j])
        w = int(plan.block_width[j])
        m = (t_of < 0) & (cols >= lo) & (cols < lo + w)
        j_of[m] = j
        o_of[m] = cols[m] - lo
    return t_of, j_of, o_of


def _lookup_sorted(keys: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Boolean + position of each ``want`` inside ascending ``keys``."""
    if keys.shape[0] == 0:
        return np.zeros(want.shape[0], dtype=bool), np.zeros(want.shape[0], np.int64)
    pos = np.clip(np.searchsorted(keys, want), 0, keys.shape[0] - 1)
    return keys[pos] == want, pos


def member_alleles(cres: ConsensusResult, columns) -> np.ndarray:
    """``(M, V) int8`` of every member's allele at the given **PWM columns**.

    ``0-3`` = ACGT, ``4`` = gap (the member is covered here and carries
    nothing), ``-1`` = uncovered. This is the only correct way to read an
    allele off the expanded column space: a caller walking a member's CIGAR
    would be in template coordinates while the variant columns are in PWM
    coordinates, and the two differ by every insertion block in between.

    A block column reads ``4`` for a member that spans the junction without
    inserting — which is what makes an insertion an ordinary two-allele
    column rather than a special case.
    """
    plan = cres.plan
    members = list(cres.members)
    cols = np.asarray(columns, dtype=np.int64)
    out = np.full((len(members), cols.shape[0]), -1, dtype=np.int8)
    if plan is None or cols.shape[0] == 0 or not members:
        return out

    t_of, j_of, o_of = _classify_columns(plan, cols)
    t_idx = np.flatnonzero(t_of >= 0)
    t_want = t_of[t_idx]
    b_idx = np.flatnonzero(j_of >= 0)

    for i, spec in enumerate(members):
        codes = base_codes(spec.member_seq)
        fstart, mstart = _offsets(spec)
        proj = project_member_events(
            parse_cigar(spec.cigar),
            codes,
            frame_is_query=spec.centroid_is_query,
            frame_start=fstart,
            member_start=mstart,
        )
        row = out[i]

        if t_idx.shape[0]:
            hit, pos = _lookup_sorted(proj.match_fpos, t_want)
            if hit.any():
                vals = proj.match_base[pos[hit]].astype(np.int8)
                # An ambiguous member base is no evidence, not a gap.
                vals = np.where(vals >= 4, np.int8(-1), vals)
                row[t_idx[hit]] = vals
            hitg, _posg = _lookup_sorted(proj.gap_fpos, t_want)
            if hitg.any():
                row[t_idx[hitg]] = 4

        if b_idx.shape[0]:
            events = dict(_member_events(proj, codes, plan.n_template))
            for k in b_idx:
                j = int(j_of[k])
                if not (proj.frame_start <= j <= proj.frame_end):
                    continue  # never reaches this junction — no evidence
                w = int(plan.block_width[j])
                o = int(o_of[k])
                c = events.get(j)
                if c is None or c.size == 0:
                    row[k] = 4
                    continue
                kk = min(int(c.size), w)
                # Junction 0's block abuts template column 0 on its right, so
                # it right-justifies; every other block abuts the column on
                # its left and left-justifies.
                if j == 0:
                    filled = o >= w - kk
                    src = int(c.size) - (w - o) if filled else -1
                else:
                    filled = o < kk
                    src = o if filled else -1
                if not filled:
                    row[k] = 4
                else:
                    base = int(c[src])
                    row[k] = -1 if base >= 4 else base
    return out


def centroid_consensus(
    centroid_seq: str,
    centroid_weight: float,
    members: list[MemberSpec],
) -> ConsensusResult:
    """Single-pass centroid-anchored consensus (pre-insertion-folding v1).

    Back-compat shim: no folding, no extension, one pass, and the centroid
    self-votes with ``centroid_weight``. Bit-identical to the kernel that
    shipped before insertion folding landed. New callers should use
    :func:`frame_consensus`.
    """
    return frame_consensus(
        centroid_seq,
        members,
        frame_weight=centroid_weight,
        fold_insertions=False,
    )


__all__ = [
    "MemberSpec",
    "FrameAlignment",
    "ProjectedMember",
    "ConsensusResult",
    "frame_consensus",
    "centroid_consensus",
    "project_member",
    "project_member_events",
    "plan_columns",
    "member_alleles",
    "ColumnPlan",
    "frame_of_consensus",
    "consensus_of_frame",
    "COL_TEMPLATE",
    "COL_INSERTED",
    "COL_EXT_5P",
    "COL_EXT_3P",
]
