"""The two-pass E-step's finalist aligner, against known geometry.

Every case builds a read from a template with a known placement and asserts
the alignment lands where minimap2 ``-c`` would put it: a clip-free ``=XID``
CIGAR plus ``(q_start, q_end, t_start, t_end)`` in PAF convention, with
``n_match / aln_len`` equal to the identity of that placement. The chained
coordinates handed in are deliberately a few bases *inside* the truth, which
is what minimap2 without ``-c`` reports (anchors stop short of the ends).

The overhang cases are the ones that matter. An infix alignment of the whole
read charges every overhanging base as an edit — the likely source of the
benchmark harness's p10 identity of 0.627 — and a unit-cost aligner has no
preference between a clean ``20I`` and ``1=3I1=3I…`` for an overhang, which
without the anchored extension put ``t_start`` 8 bases into a template whose
start the read actually spans.
"""

from __future__ import annotations

import random

import pytest

from constellation.sequencing.transcriptome.cluster.denovo._cigar import parse_cigar
from constellation.sequencing.transcriptome.cluster.denovo.em.realign import (
    align_finalist,
    anchor_trim,
    score_trim,
)

_RNG = random.Random(7)


def _rand(n: int, rng: random.Random = _RNG) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


TEMPLATE = _rand(1500)


def _walk(read: str, template: str, f) -> tuple[int, int]:
    """Walk ``f.cigar`` over the sequences; return (n_match, aln_len)."""
    ops = parse_cigar(f.cigar)  # strict: raises on S/H or unknown ops
    q, t = f.q_start, f.t_start
    n_match = aln_len = 0
    for n, op in ops:
        aln_len += n
        if op == "=":
            assert read[q : q + n] == template[t : t + n]
            n_match += n
            q, t = q + n, t + n
        elif op == "X":
            assert all(a != b for a, b in zip(read[q : q + n], template[t : t + n]))
            q, t = q + n, t + n
        elif op == "I":
            q += n
        else:
            t += n
    assert (q, t) == (f.q_end, f.t_end)
    return n_match, aln_len


def _substitute(seq: str, positions) -> str:
    s = list(seq)
    for p in positions:
        s[p] = {"A": "C", "C": "G", "G": "T", "T": "A"}[s[p]]
    return "".join(s)


def _align(read, *, q, t, inset=6):
    """Chained coordinates `inset` bases inside the true (q, t) spans."""
    (qs, qe), (ts, te) = q, t
    return align_finalist(
        read,
        TEMPLATE,
        q_start=qs + inset,
        q_end=qe - inset,
        t_start=ts + inset,
        t_end=te - inset,
    )


def test_read_contained_in_template():
    read = _substitute(TEMPLATE[300:900], [100, 250, 400])
    f = _align(read, q=(0, 600), t=(300, 900))
    assert (f.q_start, f.q_end, f.t_start, f.t_end) == (0, 600, 300, 900)
    assert _walk(read, TEMPLATE, f) == (f.n_match, f.aln_len) == (597, 600)


def test_read_overhangs_template_start():
    """The 5' extension regime: the template's first base must be reached,
    because ``t_start == 0`` is what enfranchises the read's 5' flank."""
    read = _rand(80) + TEMPLATE[:700]
    f = _align(read, q=(80, 780), t=(0, 700))
    assert (f.t_start, f.q_start) == (0, 80)
    assert (f.t_end, f.q_end) == (700, 780)
    assert f.identity == 1.0
    _walk(read, TEMPLATE, f)


def test_read_overhangs_template_end():
    read = TEMPLATE[900:] + _rand(60)
    f = _align(read, q=(0, 600), t=(900, 1500))
    assert (f.t_start, f.t_end) == (900, 1500)
    assert (f.q_start, f.q_end) == (0, 600)
    assert f.identity == 1.0
    _walk(read, TEMPLATE, f)


def test_template_contained_in_read():
    read = _rand(50) + TEMPLATE + _rand(40)
    f = _align(read, q=(50, 1550), t=(0, 1500))
    assert (f.q_start, f.q_end, f.t_start, f.t_end) == (50, 1550, 0, 1500)
    assert f.identity == 1.0


def test_interior_indels_are_scored_as_the_truth():
    body = TEMPLATE[100:600]
    read = body[:200] + body[203:350] + "GT" + body[350:]  # 3-base del, 2-base ins
    f = _align(read, q=(0, len(read)), t=(100, 600))
    assert (f.t_start, f.t_end) == (100, 600)
    n_match, aln_len = _walk(read, TEMPLATE, f)
    assert (n_match, aln_len) == (f.n_match, f.aln_len)
    assert n_match == 497 and aln_len == 502


def test_junk_read_end_is_clipped_not_charged():
    """A chimeric / adapter tail where the template continues is clipped the
    way minimap2's extension clips it, not forced into the alignment."""
    read = TEMPLATE[0:500] + _rand(30, random.Random(3))
    f = _align(read, q=(0, 500), t=(0, 500))
    assert f.q_end <= 503 and f.t_end <= 503
    assert f.identity > 0.99


@pytest.mark.parametrize("pad", [0, 5, 20, 60])
def test_answer_does_not_depend_on_pad(pad):
    read = _rand(80, random.Random(11)) + TEMPLATE[:700]
    f = align_finalist(
        read, TEMPLATE, q_start=86, q_end=774, t_start=6, t_end=694, pad=pad
    )
    assert (f.q_start, f.q_end, f.t_start, f.t_end) == (80, 780, 0, 700)


def test_anchor_trim_and_score_trim():
    ops = [(1, "="), (3, "I"), (1, "="), (2, "I"), (16, "="), (4, "X"), (1, "=")]
    kept, qf, tf, qb, tb = anchor_trim(ops, k_anchor=5)
    assert kept == [(16, "=")]
    assert (qf, tf, qb, tb) == (7, 2, 5, 5)
    kept, qf, tf, qb, tb = score_trim(ops)
    assert kept[0] == (16, "=") and kept[-1] == (16, "=")
    # A clean alignment loses nothing.
    assert score_trim([(40, "="), (1, "X"), (40, "=")])[0] == [
        (40, "="),
        (1, "X"),
        (40, "="),
    ]


def _align_finalist_reference(read, template, *, q_start, q_end, t_start, t_end,
                              pad=20, k_anchor=5):
    """`align_finalist` as it was before 2026-11-20: the list functions,
    composed in full (merge everywhere, score trim, then statistics)."""
    from constellation.sequencing.align.pairwise import pairwise_align
    from constellation.sequencing.transcriptome.cluster.denovo.em import realign as R

    q_len, t_len = len(read), len(template)
    if q_len == 0 or t_len == 0:
        return None
    lo = max(0, (q_start - t_start) - pad)
    hi = min(q_len, (q_end + (t_len - t_end)) + pad)
    if hi <= lo:
        return None
    res = pairwise_align(read[lo:hi], template, mode="infix", return_cigar=True)
    core, q_front, t_front, q_back, t_back = R.anchor_trim(R._parse(res.cigar), k_anchor=k_anchor)
    if not core:
        return None
    qa, ta = lo + q_front, res.ref_start + t_front
    qb, tb = hi - q_back, res.ref_end - t_back
    left, lq, lt = R._extend(read[:qa][::-1], template[:ta][::-1])
    right, rq, rt = R._extend(read[qb:], template[tb:])
    ops = R._merge([*reversed(left), *core, *right])
    ops, fq, ft, bq, bt = R.score_trim(ops)
    if not ops:
        return None
    n_match, aln_len, score = R.cigar_stats(ops)
    return R.Finalist(
        cigar="".join(f"{n}{op}" for n, op in ops),
        q_start=qa - lq + fq, q_end=qb + rq - bq,
        t_start=ta - lt + ft, t_end=tb + rt - bt,
        n_match=n_match, aln_len=aln_len, score=score,
    )


def test_align_finalist_is_its_list_form() -> None:
    """The fewer-pass `align_finalist` gives exactly the composed list
    functions' result: contained, overhanging and junk-ended reads, gappy
    chains, unrelated pairs."""
    rng = random.Random(20261120)

    def mutate(s: str, rate: float) -> str:
        out = []
        for b in s:
            r = rng.random()
            if r < rate / 3:
                continue
            if r < 2 * rate / 3:
                out.append(b + rng.choice("ACGT"))
            elif r < rate:
                out.append(rng.choice([c for c in "ACGT" if c != b]))
            else:
                out.append(b)
        return "".join(out)

    n_aligned = 0
    for k in range(1500):
        t = _rand(rng.randint(60, 2500), rng)
        lo = rng.randint(0, max(0, len(t) // 3))
        hi = len(t) - rng.randint(0, max(0, len(t) // 3))
        body = mutate(t[lo:hi], rng.choice([0.0, 0.01, 0.03, 0.08]))
        junk5 = _rand(rng.choice([0, 0, 5, 40]), rng)
        junk3 = _rand(rng.choice([0, 0, 5, 40]), rng)
        read = junk5 + body if k % 7 else _rand(rng.randint(50, 800), rng)
        read = read + junk3
        kw = dict(q_start=len(junk5), q_end=len(read) - len(junk3),
                  t_start=lo, t_end=hi, pad=rng.choice([0, 20, 60]),
                  k_anchor=rng.choice([3, 5, 11]))
        want = _align_finalist_reference(read, t, **kw)
        assert align_finalist(read, t, **kw) == want, (k, kw)
        n_aligned += want is not None
    assert n_aligned > 1000
