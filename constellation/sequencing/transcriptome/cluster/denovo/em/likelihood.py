"""Read-to-template likelihood over the positions that actually differ.

The E-step has to choose between candidate templates that are all ~97-100%
identical to the read. Both ``AS`` and percent-identity put the ~97% of
*identical* sites into the denominator, so the handful of positions that
genuinely discriminate are diluted by construction. This module scores only
those positions, under the context-aware error model that already ships in
:mod:`..variants`.

That distinction is the whole point, and it is not cosmetic. Under a *uniform*
error rate the log-likelihood ratio is monotone in the mismatch count and
reduces to roughly what ``AS`` already computes. Under the real model it does
not, because nanopore error is not uniform by type:

======================================  =========  ==============
differing position                      epsilon    evidence
======================================  =========  ==============
substitution                            ~0.003     ~5.8 nats
non-homopolymer indel                   ~0.004     ~5.5 nats
homopolymer indel, run length 5         ~0.22      ~1.5 nats
======================================  =========  ==============

``AS`` charges all three about 6 points. So a read differing from template A
by one homopolymer-length call and from template B by one substitution is
*much* better explained by A — and AS calls it a tie. This is the same
reasoning :mod:`..em.covariance` already applies *within* a template (its
epsilon-weighted metric scores a substitution column at 5.8 nats and a 22%
homopolymer column at 1.27); here it decides *between* templates.

.. warning::

   **The default error model throttles this by ~3.5x, and that is a known
   defect rather than a modelling choice.** The 0.22 in the table above is the
   rate measured at the H3f3b 5-G run. :class:`~..variants.ErrorModel`'s
   default curve returns **0.0103** there — 21x lower — which makes a
   homopolymer indel worth 4.58 nats instead of 1.51, and shrinks its margin
   against a substitution from 4.30 nats to **1.23**. The ordering this module
   exists to produce still holds at the default, but the separation is a third
   of what the data supports.

   The fix is to refit the homopolymer curve, which is ledger #40 and needs
   more than the single run-length the bench pinned — so it is deliberately
   *not* done by inventing a slope here. Pass a refitted model (that is what
   ``--error-model empirical`` produces) and this ranker sharpens for free;
   the shared default is left alone because ``call_variants`` on the
   components path uses the same object, and silently moving it would change
   which variants that path calls real.

Three terms, not one:

``n_match * log(1 - eps_match)``
    The matched bases. Small per base, but it is what stops a short perfect
    alignment beating a long near-perfect one — the failure mode that makes
    raw identity unusable as a ranking statistic.
``sum over differing positions of log(eps_context)``
    The discriminating evidence.
``n_unaligned * log(eps_unaligned)``
    Read bases the template does not explain at all. Without this a template
    covering half the read at 100% identity outranks one covering all of it
    at 99.9%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from constellation.sequencing.transcriptome.cluster.denovo.variants import ErrorModel


# Per-base likelihood of a read base the template does not explain. A soft
# clip is not evidence *against* the template so much as evidence the template
# is incomplete; 0.25 is the uninformative choice (one base in four) and makes
# a clipped base cost ~1.39 nats — a quarter of a substitution.
EPS_UNALIGNED = 0.25

_OP_EQ, _OP_X, _OP_I, _OP_D = 0, 1, 2, 3
_OP_CODE = np.full(256, -1, dtype=np.int8)
_OP_CODE[ord("=")] = _OP_EQ
_OP_CODE[ord("X")] = _OP_X
_OP_CODE[ord("I")] = _OP_I
_OP_CODE[ord("D")] = _OP_D
_OP_CODE[ord("M")] = _OP_EQ  # without --eqx there is no X; treat M as match
_OP_CODE[ord("S")] = -2  # clipping: no template or evidence consumption
_OP_CODE[ord("H")] = -2
_OP_CODE[ord("N")] = -3  # reference skip: consumes template, not evidence


@dataclass(frozen=True, slots=True)
class CigarOps:
    """Flattened ``(candidate, op, length)`` triples over many CIGARs."""

    candidate: np.ndarray  # int64 — which input CIGAR the op belongs to
    op: np.ndarray  # int8 — _OP_* code
    length: np.ndarray  # int64
    n_candidates: int


def parse_cigars(cigars: pa.Array | list[str | None]) -> CigarOps:
    """Vectorized ``<len><op>`` parse over a whole array of CIGAR strings.

    One pass of buffer surgery rather than a Python ``re.findall`` per string:
    at ~20-30 ops per CIGAR and millions of candidates a round, the per-string
    form is the same anti-pattern the PAF row loop was.
    """
    if not isinstance(cigars, pa.Array):
        cigars = pa.array(cigars, type=pa.large_string())
    cigars = cigars.cast(pa.large_string())
    n = len(cigars)
    if n == 0:
        return CigarOps(
            np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.int64), 0
        )

    offsets = np.asarray(cigars.buffers()[1]).view(np.int64)[: n + 1]
    total = int(offsets[-1])
    if total == 0:
        return CigarOps(
            np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.int64), n
        )
    data = np.asarray(cigars.buffers()[2]).view(np.uint8)[:total]

    is_digit = (data >= ord("0")) & (data <= ord("9"))
    op_pos = np.flatnonzero(~is_digit)
    if op_pos.size == 0:
        return CigarOps(
            np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.int64), n
        )

    # Each op's digits run from just after the previous op (or the start of
    # this CIGAR) up to the op letter itself.
    candidate = np.searchsorted(offsets, op_pos, side="right") - 1
    prev_op = np.empty(op_pos.size, dtype=np.int64)
    prev_op[0] = -1
    prev_op[1:] = op_pos[:-1]
    # Reset at each CIGAR boundary so the first op of a string does not take
    # its digit span from the tail of the previous one.
    starts = np.maximum(prev_op + 1, offsets[candidate])
    lengths = _parse_digits(data, starts, op_pos)
    return CigarOps(
        candidate=candidate.astype(np.int64),
        op=_OP_CODE[data[op_pos]],
        length=lengths,
        n_candidates=n,
    )


def _parse_digits(data: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Right-aligned digit-matrix parse, as in :mod:`.paf_scan`."""
    widths = ends - starts
    if np.any(widths <= 0):
        raise ValueError("CIGAR operation without a length")
    max_width = int(widths.max())
    cols = np.arange(max_width, dtype=np.int64)
    idx = (ends[:, None] - max_width) + cols
    np.clip(idx, 0, data.size - 1, out=idx)
    digits = data[idx].astype(np.int64) - ord("0")
    digits *= cols >= (max_width - widths)[:, None]
    pow10 = (10 ** np.arange(max_width, dtype=np.int64))[::-1]
    return digits @ pow10


def _epsilon_table(model: ErrorModel, max_run: int = 64) -> np.ndarray:
    """``epsilon_homopolymer`` precomputed by run length, index 0 unused."""
    return np.array(
        [model.epsilon_homopolymer(max(int(k), 1)) for k in range(max_run + 1)],
        dtype=np.float64,
    )


def read_template_loglik(
    cigars: pa.Array | list[str | None],
    *,
    t_start: np.ndarray,
    q_start: np.ndarray,
    q_end: np.ndarray,
    q_len: np.ndarray,
    template_row: np.ndarray,
    store,
    model: ErrorModel | None = None,
    eps_unaligned: float = EPS_UNALIGNED,
) -> np.ndarray:
    """log P(read | template) for each candidate, in nats.

    ``store`` is a :class:`~.templates.TemplateStore` — used only for
    ``hp_run_at``, to give each indel its homopolymer context. A substitution
    needs no context, which is why the template sequence is not otherwise
    consulted.

    Candidates with no CIGAR score ``-inf``: without an alignment there is no
    evidence, and returning 0.0 would make them win.
    """
    model = model or ErrorModel()
    ops = parse_cigars(cigars)
    n = ops.n_candidates
    out = np.full(n, -np.inf, dtype=np.float64)
    if n == 0:
        return out

    q_start = np.asarray(q_start, dtype=np.int64)
    q_end = np.asarray(q_end, dtype=np.int64)
    q_len = np.asarray(q_len, dtype=np.int64)
    t_start = np.asarray(t_start, dtype=np.int64)
    template_row = np.asarray(template_row, dtype=np.int64)

    if ops.candidate.size == 0:
        return out
    has_ops = np.zeros(n, dtype=bool)
    has_ops[np.unique(ops.candidate)] = True

    op, length, cand = ops.op, ops.length, ops.candidate

    # --- matched bases -------------------------------------------------
    n_match = np.bincount(
        cand, weights=np.where(op == _OP_EQ, length, 0.0), minlength=n
    )
    log_match = float(np.log1p(-model.eps_sub - model.eps_indel))
    out_match = n_match * log_match

    # --- substitutions -------------------------------------------------
    n_sub = np.bincount(cand, weights=np.where(op == _OP_X, length, 0.0), minlength=n)
    out_sub = n_sub * float(np.log(model.eps_sub))

    # --- indels, with homopolymer context ------------------------------
    # Template position of each op: the running sum of template-consuming
    # ops (=, X, D, N) before it, within its own candidate.
    consumes_t = np.isin(op, (_OP_EQ, _OP_X, _OP_D)) | (op == -3)
    adv = np.where(consumes_t, length, 0)
    cum = np.cumsum(adv)
    first = np.searchsorted(cand, np.arange(n), side="left")
    base = np.where(has_ops, np.take(cum - adv, np.clip(first, 0, cum.size - 1)), 0)
    t_pos = t_start[cand] + (cum - adv) - base[cand]

    is_indel = (op == _OP_I) | (op == _OP_D)
    out_indel = np.zeros(n, dtype=np.float64)
    if np.any(is_indel):
        sel = np.flatnonzero(is_indel)
        runs = store.hp_run_at(template_row[cand[sel]], t_pos[sel])
        eps_tab = _epsilon_table(model)
        eps = np.where(
            runs >= 3,
            eps_tab[np.clip(runs, 0, eps_tab.size - 1)],
            model.eps_indel,
        )
        # One event per indel RUN, not per base: a 2-base homopolymer slip is
        # one basecalling mistake, not two independent ones. Charging per base
        # would make a single 3-base run outweigh three separate substitutions.
        out_indel = np.bincount(cand[sel], weights=np.log(eps), minlength=n)

    # --- read bases the template does not explain ----------------------
    n_unaligned = np.maximum(q_len - (q_end - q_start), 0)
    out_clip = n_unaligned * float(np.log(eps_unaligned))

    total = out_match + out_sub + out_indel + out_clip
    out[has_ops] = total[has_ops]
    return out


__all__ = [
    "EPS_UNALIGNED",
    "CigarOps",
    "parse_cigars",
    "read_template_loglik",
]
