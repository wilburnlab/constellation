"""Finalist alignment for the two-pass E-step (``--estep-aligner edlib``).

The single-pass E-step runs minimap2 with ``-c``, i.e. base-level alignment of
every one of up to ``-N`` candidates per read. Measured on 400k reads against
the 994k round-1 templates of the full em-kmer run, that is **67% of the
E-step's wall time and 79% of its CPU**, while the M-step consumes exactly one
CIGAR per read. The two-pass path shortlists on minimap2's chaining score
(no ``-c``) and base-aligns only the shortlist here.

**Why the chained columns are not used for anything but shortlisting.**
Without ``-c`` PAF columns 10/11 are chain approximations, which
underestimate identity badly (``refine.py`` records the same fact for its own
pass), and there is no ``AS:i`` or ``cg:Z``. So the admission gate
(``n_match / aln_len >= p_floor``), the likelihood ranker (which walks the
CIGAR) and the M-step (which consumes it) all take their inputs from the
alignment produced here — so that ``p_floor`` means exactly what it means
under minimap2 ``-c``.

**Orientation.** edlib's ``HW`` mode is an *infix* alignment: the query must
align end to end inside the target. Read-into-template is the right
orientation when the read is contained in the template, which is the common
case, and wrong when the read overhangs a template end — which is the 5'/3'
extension regime this pipeline lives in (median template extent is only
0.31-0.44 of its transcript). Aligning the whole read and scoring
``1 - ED/len(read)`` charges every overhanging base as an edit, which is the
most likely explanation for the identity tail (p10 0.627) the benchmark
harness measured. So the read is first clipped to the part the chain says
lies on the template (± ``pad``, for the chain's imprecise endpoints),
aligned infix for the trustworthy interior, and its ends are then re-derived
by anchored extension and an affine score trim — see :func:`align_finalist`.
What remains is a clip-free ``=XID``
CIGAR with ``(q_start, q_end, t_start, t_end)`` in exactly minimap2's PAF
convention, so the M-step's terminal-extension logic (keyed on
``t_start == 0`` / ``t_end == len(template)``) sees the shape it sees today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from constellation.sequencing.align.pairwise import pairwise_align

_OP_RE = re.compile(r"(\d+)([=XID])")

#: minimap2 ``-x map-ont`` scoring, so a recomputed AS is on the same scale
#: as the one ``-c`` reports: match A, mismatch B, two-piece affine gaps.
_A, _B, _O, _E, _O2, _E2 = 2, 4, 4, 2, 24, 1

DEFAULT_PAD = 20
DEFAULT_K_ANCHOR = 5


@dataclass(frozen=True, slots=True)
class Finalist:
    """One read-to-template alignment, in minimap2's PAF convention.

    ``cigar`` is clip-free extended CIGAR (``=XID``) with the read as query
    and the template as reference; ``q_*`` / ``t_*`` are 0-based half-open.
    ``score`` is the map-ont-equivalent AS recomputed from the CIGAR — not
    minimap2's DP optimum, but on its scale.
    """

    cigar: str
    q_start: int
    q_end: int
    t_start: int
    t_end: int
    n_match: int
    aln_len: int
    score: int

    @property
    def identity(self) -> float:
        return self.n_match / self.aln_len if self.aln_len else 0.0


def _gap_cost(k: int) -> int:
    return min(_O + k * _E, _O2 + k * _E2)


def cigar_stats(ops: list[tuple[int, str]]) -> tuple[int, int, int]:
    """``(n_match, aln_len, map-ont score)`` over parsed ops."""
    n_match = aln_len = score = 0
    for n, op in ops:
        aln_len += n
        if op == "=":
            n_match += n
            score += _A * n
        elif op == "X":
            score -= _B * n
        else:
            score -= _gap_cost(n)
    return n_match, aln_len, score


def _consumes(op: str) -> tuple[int, int]:
    """(query, reference) bases consumed per unit of ``op``."""
    if op in "=X":
        return 1, 1
    if op == "I":
        return 1, 0
    return 0, 1  # D


def anchor_trim(
    ops: list[tuple[int, str]], *, k_anchor: int
) -> tuple[list[tuple[int, str]], int, int, int, int]:
    """Strip everything outside the first and last ``=`` run of ``k_anchor``.

    Returns ``(ops, q_front, t_front, q_back, t_back)`` — the kept ops and the
    query/reference bases stripped from each end. With no qualifying run the
    alignment is returned untouched: it has no trustworthy end to trim to,
    and it will fail the identity floor on its own.
    """
    first = next(
        (i for i, (n, op) in enumerate(ops) if op == "=" and n >= k_anchor), None
    )
    if first is None:
        return ops, 0, 0, 0, 0
    last = next(
        i
        for i in range(len(ops) - 1, -1, -1)
        if ops[i][1] == "=" and ops[i][0] >= k_anchor
    )
    q_front = t_front = q_back = t_back = 0
    for n, op in ops[:first]:
        dq, dt = _consumes(op)
        q_front += n * dq
        t_front += n * dt
    for n, op in ops[last + 1 :]:
        dq, dt = _consumes(op)
        q_back += n * dq
        t_back += n * dt
    return ops[first : last + 1], q_front, t_front, q_back, t_back


def _transpose(ops: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Swap the roles of query and reference (``I`` <-> ``D``)."""
    swap = {"I": "D", "D": "I"}
    return [(n, swap.get(op, op)) for n, op in ops]


def _parse(cigar: str | None) -> list[tuple[int, str]]:
    return [(int(n), op) for n, op in _OP_RE.findall(cigar or "")]


def _extend(q_seq: str, t_seq: str) -> tuple[list[tuple[int, str]], int, int]:
    """Anchored extension: both sequences start at the anchor.

    The **shorter** side is aligned end to end (edlib ``SHW``, prefix mode)
    and the longer one's remainder is left unaligned — overlap semantics,
    the same shorter-into-longer rule ``verify`` uses. Returns
    ``(ops, q_used, t_used)`` with the read as query.
    """
    if not q_seq or not t_seq:
        return [], 0, 0
    if len(q_seq) <= len(t_seq):
        res = pairwise_align(q_seq, t_seq, mode="prefix", return_cigar=True)
        return _parse(res.cigar), len(q_seq), res.ref_end
    res = pairwise_align(t_seq, q_seq, mode="prefix", return_cigar=True)
    return _transpose(_parse(res.cigar)), res.ref_end, len(t_seq)


def score_trim(
    ops: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], int, int, int, int]:
    """Drop the end segments whose map-ont score contribution is negative.

    Keeps the op range maximising the alignment score over prefix/suffix
    cuts at op boundaries — the effect minimap2's extension has, so a junk
    end (adapter remnant, chimera junction) is clipped rather than charged
    as edits. Returns ``(ops, q_front, t_front, q_back, t_back)``.
    """
    if not ops:
        return ops, 0, 0, 0, 0

    def contrib(n: int, op: str) -> int:
        if op == "=":
            return _A * n
        if op == "X":
            return -_B * n
        return -_gap_cost(n)

    vals = [contrib(n, op) for n, op in ops]
    # Front: cut before the index where the running prefix sum is lowest.
    run, best, cut_f = 0, 0, 0
    for i, v in enumerate(vals):
        run += v
        if run < best:
            best, cut_f = run, i + 1
    run, best, cut_b = 0, 0, len(vals)
    for i in range(len(vals) - 1, cut_f - 1, -1):
        run += vals[i]
        if run < best:
            best, cut_b = run, i
    q_front = t_front = q_back = t_back = 0
    for n, op in ops[:cut_f]:
        dq, dt = _consumes(op)
        q_front += n * dq
        t_front += n * dt
    for n, op in ops[cut_b:]:
        dq, dt = _consumes(op)
        q_back += n * dq
        t_back += n * dt
    return ops[cut_f:cut_b], q_front, t_front, q_back, t_back


def _merge(ops: list[tuple[int, str]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for n, op in ops:
        if n <= 0:
            continue
        if out and out[-1][1] == op:
            out[-1] = (out[-1][0] + n, op)
        else:
            out.append((n, op))
    return out


def align_finalist(
    read: str,
    template: str,
    *,
    q_start: int,
    q_end: int,
    t_start: int,
    t_end: int,
    pad: int = DEFAULT_PAD,
    k_anchor: int = DEFAULT_K_ANCHOR,
) -> Finalist | None:
    """Align ``read`` to ``template`` guided by a chained hit's coordinates.

    ``q_start`` .. ``t_end`` are the (approximate) chained coordinates from a
    no-``-c`` PAF line. Three steps:

    1. **Core** — the read span the chain's diagonal projects onto the
       template, ± ``pad``, aligned infix (``HW``) and cut back to its first
       and last run of ``k_anchor`` exact matches. Unit-cost edit distance
       has no preference between a clean ``20I`` and ``1=3I1=3I…`` for an
       overhanging pad, so the ends of an HW alignment are not trustworthy;
       its interior is.
    2. **Extend** each end from its anchor (:func:`_extend`), shorter side
       end to end — so a template start lying inside the read is reached
       exactly (``t_start == 0``), which is what the M-step's 5' extension
       vote keys on.
    3. **Score-trim** both ends under map-ont's affine scoring
       (:func:`score_trim`).

    Returns ``None`` when nothing of the read can lie on the template.
    """
    q_len, t_len = len(read), len(template)
    if q_len == 0 or t_len == 0:
        return None
    lo = max(0, (q_start - t_start) - pad)
    hi = min(q_len, (q_end + (t_len - t_end)) + pad)
    if hi <= lo:
        return None
    res = pairwise_align(read[lo:hi], template, mode="infix", return_cigar=True)
    core, q_front, t_front, q_back, t_back = anchor_trim(
        _parse(res.cigar), k_anchor=k_anchor
    )
    if not core:
        return None
    qa, ta = lo + q_front, res.ref_start + t_front
    qb, tb = hi - q_back, res.ref_end - t_back

    left, lq, lt = _extend(read[:qa][::-1], template[:ta][::-1])
    right, rq, rt = _extend(read[qb:], template[tb:])
    ops = _merge([*reversed(left), *core, *right])
    ops, fq, ft, bq, bt = score_trim(ops)
    if not ops:
        return None
    n_match, aln_len, score = cigar_stats(ops)
    return Finalist(
        cigar="".join(f"{n}{op}" for n, op in ops),
        q_start=qa - lq + fq,
        q_end=qb + rq - bq,
        t_start=ta - lt + ft,
        t_end=tb + rt - bt,
        n_match=n_match,
        aln_len=aln_len,
        score=score,
    )


__all__ = [
    "DEFAULT_K_ANCHOR",
    "DEFAULT_PAD",
    "Finalist",
    "align_finalist",
    "anchor_trim",
    "cigar_stats",
    "score_trim",
]
