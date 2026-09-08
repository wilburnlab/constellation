"""Frame-anchored consensus via an iterated CIGAR-walk PWM (numpy scatter).

Each member is projected onto a **frame** coordinate system and its bases
accumulate into an abundance-weighted ``(F, 5)`` position-weight matrix
(columns A/C/G/T/gap). The consensus is the per-position argmax; gap-winning
positions drop out, positions with no coverage fall back to the frame base.

The frame starts as the caller's template (round 1: a seed read's window;
later rounds: the previous consensus) and **grows**:

* **insertion folding** — where a weighted majority of the members *covering*
  a junction carry an insertion there, the majority inserted bases are spliced
  into the frame and every affected member is re-aligned. Without this the
  consensus can never be longer than its frame, so a deletion error *in the
  frame* is unrepairable no matter how many members carry the base: 120 reads
  carrying a base cannot put it back into a centroid that lacks it. That was
  worth ~25% frameshifted consensuses at single-read-frame depth.
* **end extension** — member bases beyond the frame's 5'/3' ends (soft clips
  in minimap2 terms, the free ref-end flanks in edlib HW terms) extend the
  consensus under the same majority rule plus an absolute depth floor. The
  per-column depth this computes is the substrate for the ORF support gate.

Together these make the result independent of which read became the frame,
which is what lets the frame be chosen for coverage rather than for length.

Three coordinate systems, and the maps between them are the public contract:

===========  ========  ==================================================
space        length    indexes
===========  ========  ==================================================
template     ``T``     the frame string handed in. Provenance only.
frame        ``F≥T``   the final spliced frame. ``pwm`` rows / ``winner``.
consensus    ``C≤F``   the emitted string (gap-winning columns removed).
===========  ========  ==================================================

Splicing only ever *adds* columns, so ``F >= T`` always. Use
:func:`frame_of_consensus` / :func:`consensus_of_frame` rather than
open-coding the maps — the two differ on the unresolved-``N`` column
(``_AMBIG``), which is a consensus column but not a base, and getting that
wrong silently misaligns every variant downstream of the first ``N``.

Kept torch-free (numpy + edlib only): the per-cluster pool is fork-based and
torch after ``fork()`` deadlocks on OpenMP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

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
    """A member's final alignment against the final frame, always normalised
    to member-as-query / frame-as-ref — so ``I`` is always a member insertion
    and ``D`` always a member deletion, whichever way the edlib call ran."""

    member_id: int
    weight: float
    cigar: str
    frame_start: int
    frame_end: int  # exclusive
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


def _discover_insertions(
    projections: Sequence[tuple[ProjectedMember, float]],
    n_frame: int,
    *,
    min_fraction: float,
    min_weight: float,
) -> dict[int, str]:
    """Junctions where a weighted majority of *covering* members insert.

    The denominator is the weight of members spanning the junction, not of
    all members: a member whose alignment does not reach a column is not
    evidence against an insertion there.
    """
    if not projections:
        return {}
    # Coverage per junction 0..n_frame via a weighted difference array.
    delta = np.zeros(n_frame + 2, dtype=np.float64)
    ins_w = np.zeros(n_frame + 1, dtype=np.float64)
    by_junction: dict[int, list[tuple[np.ndarray, float]]] = {}
    for proj, w in projections:
        lo = max(0, min(proj.frame_start, n_frame))
        hi = max(lo, min(proj.frame_end, n_frame))
        delta[lo] += w
        delta[hi + 1] -= w
        for junction, codes in proj.ins_events:
            if 0 <= junction <= n_frame and codes.size:
                ins_w[junction] += w
                by_junction.setdefault(junction, []).append((codes, w))
    cov_w = np.cumsum(delta)[: n_frame + 1]

    out: dict[int, str] = {}
    for junction in sorted(by_junction):
        if ins_w[junction] < min_weight:
            continue
        if ins_w[junction] <= min_fraction * cov_w[junction]:
            continue
        events = by_junction[junction]
        lengths = np.array([c.size for c, _ in events], dtype=np.int64)
        ev_w = np.array([w for _, w in events], dtype=np.float64)
        # Modal run length (ties resolve to the shorter, since bincount is
        # ordered ascending) — this is what makes a homopolymer settle on its
        # modal length rather than its longest observation.
        run_len = _weighted_mode(lengths, ev_w, int(lengths.max()) + 1)
        if run_len <= 0:
            continue
        chars: list[str] = []
        for t in range(run_len):
            # Every event long enough to speak at `t` votes, not just the
            # modal-length ones.
            codes = np.array(
                [int(c[t]) for c, _ in events if c.size > t], dtype=np.int64
            )
            wts = np.array([w for c, w in events if c.size > t], dtype=np.float64)
            valid = codes < 4
            best = _weighted_mode(codes[valid], wts[valid], 4)
            chars.append("N" if best < 0 else _BASE_CHARS[best])
        out[junction] = "".join(chars)
    return out


def _discover_extension(
    projections: Sequence[tuple[ProjectedMember, float]],
    member_codes: Sequence[np.ndarray],
    n_frame: int,
    *,
    five_prime: bool,
    min_fraction: float,
    min_weight: float,
    max_len: int,
) -> str:
    """Bases past one end of the frame that a weighted majority supports.

    Only members anchored at that end vote (``frame_start == 0`` for 5',
    ``frame_end == n_frame`` for 3'). A read clipped a couple of bases into
    the frame therefore forfeits its extension vote — deliberate, since
    accepting extensions from mis-anchored alignments manufactures chimeric
    ends. Extends outward one base at a time and stops where support stops,
    so the result is graded by depth rather than by any single member's
    reach.
    """
    # (codes, boundary, n_available, weight) — `boundary` is the member index
    # the extension grows away from, so offset k reads codes[boundary - k] at
    # the 5' end and codes[boundary + k - 1] at the 3' end.
    anchored: list[tuple[np.ndarray, int, int, float]] = []
    total_w = 0.0
    for (proj, w), codes in zip(projections, member_codes):
        if five_prime:
            if proj.frame_start != 0:
                continue
            boundary, avail = proj.member_start, proj.member_start
        else:
            if proj.frame_end != n_frame:
                continue
            boundary, avail = proj.member_end, codes.shape[0] - proj.member_end
        total_w += w
        if avail > 0:
            anchored.append((codes, boundary, avail, w))
    if not anchored or total_w <= 0.0:
        return ""

    chars: list[str] = []
    for k in range(1, max_len + 1):
        reach = [(c, b, w) for c, b, avail, w in anchored if avail >= k]
        depth = sum(w for _, _, w in reach)
        if depth < min_weight or depth <= min_fraction * total_w:
            break
        codes = np.array(
            [int(c[b - k]) if five_prime else int(c[b + k - 1]) for c, b, _ in reach],
            dtype=np.int64,
        )
        wts = np.array([w for _, _, w in reach], dtype=np.float64)
        valid = codes < 4
        best = _weighted_mode(codes[valid], wts[valid], 4)
        if best < 0:
            break
        chars.append(_BASE_CHARS[best])
    if five_prime:
        chars.reverse()  # built outward from the frame; the string reads 5'→3'
    return "".join(chars)


def _splice(
    frame_seq: str,
    winner: np.ndarray,
    template_of_frame: np.ndarray,
    column_kind: np.ndarray,
    inserts: dict[int, str],
    ext5: str,
    ext3: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Build the next frame + its provenance maps.

    A column whose winner is a real base adopts it (so the frame converges
    toward the consensus, which is what keeps the next pass's re-alignment
    cheap); gap-winning and unresolved-N columns keep a placeholder so they
    stay addressable PWM rows.
    """
    n = len(frame_seq)
    parts: list[str] = []
    tmpl: list[np.ndarray] = []
    kind: list[np.ndarray] = []

    def _add(text: str, k: int) -> None:
        if not text:
            return
        parts.append(text)
        tmpl.append(np.full(len(text), -1, dtype=np.int64))
        kind.append(np.full(len(text), k, dtype=np.int8))

    _add(ext5, COL_EXT_5P)
    _add(inserts.get(0, ""), COL_INSERTED)
    body = bytearray(n)
    for f in range(n):
        w = int(winner[f])
        body[f] = _IDX_TO_BASE_NP[w] if w < 4 else ord(frame_seq[f])
    # Emit the body in runs between insertion points to keep this O(F).
    cuts = sorted(j for j in inserts if 0 < j <= n)
    prev = 0
    for j in cuts:
        if j > prev:
            parts.append(body[prev:j].decode("ascii"))
            tmpl.append(template_of_frame[prev:j])
            kind.append(column_kind[prev:j])
        _add(inserts[j], COL_INSERTED)
        prev = j
    if prev < n:
        parts.append(body[prev:n].decode("ascii"))
        tmpl.append(template_of_frame[prev:n])
        kind.append(column_kind[prev:n])
    _add(ext3, COL_EXT_3P)

    new_frame = "".join(parts)
    new_tmpl = np.concatenate(tmpl) if tmpl else np.empty(0, dtype=np.int64)
    new_kind = np.concatenate(kind) if kind else np.empty(0, dtype=np.int8)
    return new_frame, new_tmpl, new_kind, np.array(cuts, dtype=np.int64)


def frame_consensus(
    frame_seq: str,
    members: Sequence[MemberSpec],
    *,
    frame_weight: float = 0.0,
    fold_insertions: bool = True,
    extend_ends: bool = True,
    ins_min_fraction: float = 0.5,
    ins_min_weight: float = 3.0,
    ext_min_fraction: float = 0.5,
    ext_min_weight: float = 3.0,
    max_extension_per_pass: int = 300,
    max_passes: int = 3,
    realign_max_frac: float = 0.25,
    template_of_frame: np.ndarray | None = None,
    realign: Callable[[str, str], tuple[str, int, int] | None] | None = None,
) -> ConsensusResult:
    """Build the abundance-weighted consensus for one cluster.

    ``frame_weight`` is the frame's own self-vote. It defaults to **0** — in
    the EM M-step the frame is a scaffold, not an observation: in round 1 the
    seed read is also an assigned member (a self-vote double-counts it, and at
    5-read depth a doubled seed is a 2:4 → 3:3 swing that flips a column), and
    in later rounds the frame is a consensus with no reads of its own. It also
    keeps the PWM column sums equal to raw read multiplicity, which
    ``variants.py`` relies on as its binomial ``n``.

    ``max_passes`` bounds the PWM builds, so at most ``max_passes - 1``
    splices happen. 2-3 passes settle everything measured.
    """
    n_members = len(members)
    codes_of = [base_codes(m.member_seq) for m in members]
    weights = [float(m.weight) for m in members]
    # Every member starts from its cached alignment; re-alignment replaces
    # these in place as the frame changes.
    live: list[tuple[str, int, int, bool] | None] = []
    for spec in members:
        fstart, mstart = _offsets(spec)
        live.append((spec.cigar, fstart, mstart, spec.centroid_is_query))

    frame = frame_seq
    tmpl = (
        np.arange(len(frame), dtype=np.int64)
        if template_of_frame is None
        else np.asarray(template_of_frame, dtype=np.int64)
    )
    kind = np.full(len(frame), COL_TEMPLATE, dtype=np.int8)
    n_inserted = n_ext5 = n_ext3 = 0
    n_fail = 0
    passes = 0

    while True:
        passes += 1
        frame_codes = base_codes(frame).astype(np.int64)
        n_frame = frame_codes.shape[0]

        projections: list[tuple[ProjectedMember, float]] = []
        proj_codes: list[np.ndarray] = []
        proj_index: list[int] = []
        for i in range(n_members):
            cur = live[i]
            if cur is None:
                continue
            cigar, fstart, mstart, frame_is_query = cur
            proj = project_member_events(
                parse_cigar(cigar),
                codes_of[i],
                frame_is_query=frame_is_query,
                frame_start=fstart,
                member_start=mstart,
            )
            projections.append((proj, weights[i]))
            proj_codes.append(codes_of[i])
            proj_index.append(i)

        pwm = _build_pwm(frame_codes, frame_weight, projections)
        winner = _call_winner(pwm, frame_codes)

        if passes >= max_passes or not (fold_insertions or extend_ends):
            break

        inserts = (
            _discover_insertions(
                projections,
                n_frame,
                min_fraction=ins_min_fraction,
                min_weight=ins_min_weight,
            )
            if fold_insertions
            else {}
        )
        ext5 = ext3 = ""
        if extend_ends:
            ext5 = _discover_extension(
                projections,
                proj_codes,
                n_frame,
                five_prime=True,
                min_fraction=ext_min_fraction,
                min_weight=ext_min_weight,
                max_len=max_extension_per_pass,
            )
            ext3 = _discover_extension(
                projections,
                proj_codes,
                n_frame,
                five_prime=False,
                min_fraction=ext_min_fraction,
                min_weight=ext_min_weight,
                max_len=max_extension_per_pass,
            )
        if not inserts and not ext5 and not ext3:
            break  # fixpoint

        frame, tmpl, kind, cuts = _splice(
            frame, winner, tmpl, kind, inserts, ext5, ext3
        )
        n_inserted += sum(len(s) for s in inserts.values())
        n_ext5 += len(ext5)
        n_ext3 += len(ext3)

        # Members whose aligned span is untouched by any splice just shift;
        # everyone else is re-aligned onto the new frame.
        shift = np.zeros(n_frame + 1, dtype=np.int64)
        pos = len(ext5)
        for f in range(n_frame):
            pos += len(inserts.get(f, ""))
            shift[f] = pos
            pos += 1
        shift[n_frame] = pos + len(inserts.get(n_frame, ""))
        aligner = realign or (
            lambda fr, me: default_realign(fr, me, max_frac=realign_max_frac)
        )
        for slot, (proj, _w) in enumerate(projections):
            i = proj_index[slot]
            touched = bool(
                cuts.size
                and np.any((cuts >= proj.frame_start) & (cuts <= proj.frame_end))
            )
            if inserts.get(0) and proj.frame_start == 0:
                touched = True
            if ext5 and proj.frame_start == 0:
                touched = True
            if ext3 and proj.frame_end == n_frame:
                touched = True
            if not touched:
                # Nothing was spliced inside this member's span, so its
                # alignment is unchanged and only its origin moves.
                new_start = int(shift[min(proj.frame_start, n_frame)])
                cigar, _fs, mstart, fiq = live[i]
                live[i] = (cigar, new_start, mstart, fiq)
                continue
            res = aligner(frame, members[i].member_seq)
            if res is None:
                live[i] = None
                n_fail += 1
                continue
            cig, fstart, mstart = res
            live[i] = (cig, fstart, mstart, False)

    keep = winner != _GAP
    consensus = _IDX_TO_BASE_NP[winner[keep]].tobytes().decode("ascii")
    alignments = []
    for slot, (proj, _w) in enumerate(projections):
        i = proj_index[slot]
        cigar, _fs, _ms, frame_is_query = live[i]  # type: ignore[misc]
        # FrameAlignment promises member-as-query, but a member that never
        # needed re-aligning still carries its cached CIGAR — which may be
        # frame-as-query, with I and D meaning the opposite of what a consumer
        # assumes. Normalise here rather than making every consumer ask.
        alignments.append(
            FrameAlignment(
                member_id=members[i].member_id,
                weight=weights[i],
                cigar=cigar.translate(_CIGAR_TRANSPOSE) if frame_is_query else cigar,
                frame_start=proj.frame_start,
                frame_end=proj.frame_end,
                member_start=proj.member_start,
                member_end=proj.member_end,
                member_seq=members[i].member_seq,
            )
        )
    return ConsensusResult(
        consensus=consensus,
        pwm=pwm,
        winner=winner,
        frame=frame,
        frame_of_cons=frame_of_consensus(winner),
        cons_of_frame=consensus_of_frame(winner),
        template_of_frame=tmpl,
        column_kind=kind,
        alignments=alignments,
        n_passes=passes,
        n_inserted_columns=n_inserted,
        n_extended_5p=n_ext5,
        n_extended_3p=n_ext3,
        n_realign_failures=n_fail,
    )


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
        extend_ends=False,
        max_passes=1,
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
    "default_realign",
    "frame_of_consensus",
    "consensus_of_frame",
    "COL_TEMPLATE",
    "COL_INSERTED",
    "COL_EXT_5P",
    "COL_EXT_3P",
]
