"""Low-level CIGAR helpers shared by verify + consensus (torch-free).

edlib (``task="path"``) emits the *extended* CIGAR alphabet — ``=`` (match),
``X`` (mismatch), ``I`` (insertion in query vs ref), ``D`` (deletion in
query vs ref) — so match / mismatch / indel counts read straight off the
ops without comparing bases. ``M`` (ambiguous match/mismatch, from a
standard-CIGAR edlib build) is tolerated and folded into matches.

These helpers operate on the **edlib convention**: the alignment was run
``query`` → ``ref``, ``I`` consumes query only, ``D`` consumes ref only.
minimap2's PAF ``cg:Z:`` string uses the same convention with
``query = read`` and ``ref = target``, so it parses here directly (run
minimap2 with ``--eqx`` or every op is ``M`` and the mismatch count is
silently zero).

``parse_cigar`` is **strict**: a CIGAR carrying ops it does not model
raises rather than being silently truncated. The permissive version
returned ``[(90, "M")]`` for ``"10S90M"``, which would have shifted every
projected member by the clip length with no error anywhere — harmless
while edlib was the only producer, a live coordinate bug the moment a
SAM-sourced CIGAR reaches this module. Use :func:`parse_cigar_clipped`
when terminal soft/hard clips are expected.
"""

from __future__ import annotations

import re

import numpy as np


_CIGAR_RE = re.compile(r"(\d+)([=XIDM])")
# Superset used only to produce a useful error message on rejection.
_CIGAR_ANY_RE = re.compile(r"(\d+)([A-Z=])")
_CLIP_OPS = frozenset({"S", "H"})

# A=0 C=1 G=2 T=3, else 4 (ambiguous). Mirrors align/consensus.py.
_BASE_LUT = np.full(256, 4, dtype=np.int8)
for _ch, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_LUT[ord(_ch)] = _code
    _BASE_LUT[ord(_ch.lower())] = _code


def _consumed(ops: list[tuple[str, str]]) -> int:
    """Characters the match list accounts for — one per op letter plus the
    digits of each length. Comparing this to ``len(cigar)`` catches ops the
    regex skipped without paying for a second full pass."""
    return sum(len(n) for n, _ in ops) + len(ops)


def parse_cigar(cigar: str) -> list[tuple[int, str]]:
    """``'9=1I3='`` → ``[(9, '='), (1, 'I'), (3, '=')]``.

    Raises ``ValueError`` if the string contains anything outside
    ``=XIDM`` — including the soft/hard clips a SAM-sourced CIGAR carries.
    """
    raw = _CIGAR_RE.findall(cigar)
    if _consumed(raw) != len(cigar):
        bad = [op for _, op in _CIGAR_ANY_RE.findall(cigar) if op not in "=XIDM"]
        hint = (
            " (terminal clips: use parse_cigar_clipped)"
            if bad and set(bad) <= _CLIP_OPS
            else ""
        )
        raise ValueError(
            f"unparseable CIGAR {cigar!r}: unsupported op(s) "
            f"{sorted(set(bad)) or '?'}{hint}"
        )
    return [(int(n), op) for n, op in raw]


def parse_cigar_clipped(cigar: str) -> tuple[int, list[tuple[int, str]], int]:
    """Split a clipped CIGAR into ``(lead_clip, ops, trail_clip)``.

    Terminal ``S`` / ``H`` runs are stripped into the clip counts (their
    lengths summed, so ``'5H10S…'`` yields 15) and the remainder is parsed
    by :func:`parse_cigar`. Raises ``ValueError`` on a clip that is not at
    an end, since that has no coordinate interpretation.
    """
    ops = _CIGAR_ANY_RE.findall(cigar)
    if _consumed(ops) != len(cigar):
        raise ValueError(f"unparseable CIGAR {cigar!r}")
    lo, hi = 0, len(ops)
    lead = trail = 0
    while lo < hi and ops[lo][1] in _CLIP_OPS:
        lead += int(ops[lo][0])
        lo += 1
    while hi > lo and ops[hi - 1][1] in _CLIP_OPS:
        trail += int(ops[hi - 1][0])
        hi -= 1
    interior = ops[lo:hi]
    if any(op in _CLIP_OPS for _, op in interior):
        raise ValueError(f"interior soft/hard clip in CIGAR {cigar!r}")
    core = "".join(f"{n}{op}" for n, op in interior)
    return lead, parse_cigar(core), trail


def cigar_stats(cigar: str) -> tuple[int, int, int, int]:
    """Return ``(n_match, n_mismatch, n_insert, n_delete)`` from an
    extended CIGAR. ``n_insert`` / ``n_delete`` are query / ref relative
    per the edlib convention."""
    nm = nx = ni = nd = 0
    for n, op in parse_cigar(cigar):
        if op in ("=", "M"):
            nm += n
        elif op == "X":
            nx += n
        elif op == "I":
            ni += n
        elif op == "D":
            nd += n
    return nm, nx, ni, nd


def indel_runs(cigar: str) -> tuple[int, int]:
    """Return ``(n_insert_runs, n_delete_runs)`` — consecutive-op *runs*,
    not bases.

    The fold stage's frameshift-pair rule is "exactly one indel run in the
    shared span", which is a statement about events: a single 2-nt deletion
    is one run, two separate 1-nt deletions are two. Adjacent same-op ops
    merge, so ``'5=1I1I3='`` counts one insertion run.
    """
    ins = dels = 0
    prev = ""
    for _, op in parse_cigar(cigar):
        if op != prev:
            if op == "I":
                ins += 1
            elif op == "D":
                dels += 1
        prev = op
    return ins, dels


def base_codes(seq: str) -> np.ndarray:
    """ASCII sequence → int8 base-index array (A=0 C=1 G=2 T=3, else 4)."""
    return _BASE_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


__all__ = [
    "parse_cigar",
    "parse_cigar_clipped",
    "cigar_stats",
    "indel_runs",
    "base_codes",
]
