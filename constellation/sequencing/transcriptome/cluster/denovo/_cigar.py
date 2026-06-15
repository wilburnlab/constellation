"""Low-level CIGAR helpers shared by verify + consensus (torch-free).

edlib (``task="path"``) emits the *extended* CIGAR alphabet — ``=`` (match),
``X`` (mismatch), ``I`` (insertion in query vs ref), ``D`` (deletion in
query vs ref) — so match / mismatch / indel counts read straight off the
ops without comparing bases. ``M`` (ambiguous match/mismatch, from a
standard-CIGAR edlib build) is tolerated and folded into matches.

These helpers operate on the **edlib convention**: the alignment was run
``query`` → ``ref``, ``I`` consumes query only, ``D`` consumes ref only.
"""

from __future__ import annotations

import re

import numpy as np


_CIGAR_RE = re.compile(r"(\d+)([=XIDM])")

# A=0 C=1 G=2 T=3, else 4 (ambiguous). Mirrors align/consensus.py.
_BASE_LUT = np.full(256, 4, dtype=np.int8)
for _ch, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_LUT[ord(_ch)] = _code
    _BASE_LUT[ord(_ch.lower())] = _code


def parse_cigar(cigar: str) -> list[tuple[int, str]]:
    """``'9=1I3='`` → ``[(9, '='), (1, 'I'), (3, 'D')]``."""
    return [(int(n), op) for n, op in _CIGAR_RE.findall(cigar)]


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


def base_codes(seq: str) -> np.ndarray:
    """ASCII sequence → int8 base-index array (A=0 C=1 G=2 T=3, else 4)."""
    return _BASE_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


__all__ = ["parse_cigar", "cigar_stats", "base_codes"]
