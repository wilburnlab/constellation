"""Longest sense-strand ORF prediction (leaf — imports only ``core``).

Shared by the consensus stage (ORF on each cluster consensus) and the
ORF-anchored EM's seeding stage (ORF on each read window). Lives in its own
leaf module so ``orfem/`` can use it without importing ``pipeline``.

Semantics, matching the shipped de novo pipeline:

* **ATG-only starts.** ``core.sequence.nucleic.STANDARD`` allows the NCBI
  alternative starts CTG/TTG, which over-call ORFs badly on noisy reads.
* **Forward strand only.** Demux windows are adapter-oriented, so "+" is
  sense; scanning both strands would double the spurious-ORF rate for
  nothing.
* **Longest internal-stop-free ORF**, deduplicated per ``(frame, stop)`` so
  alternative starts sharing a stop collapse to their longest form.
"""

from __future__ import annotations

import functools

from constellation.core.sequence.nucleic import STANDARD, CodonTable, translate


ORF_CODON_TABLE: CodonTable = CodonTable(
    transl_table=STANDARD.transl_table,
    name=f"{STANDARD.name} (ATG-only starts)",
    forward=STANDARD.forward,
    starts=frozenset({"ATG"}),
    stops=STANDARD.stops,
)


@functools.lru_cache(maxsize=8)
def orf_regex(min_aa_length: int):
    import regex

    # ATG start … ≥ min_aa_length codons … stop. Compiled once per length
    # (vs once per call inside find_orfs — at 10M+ clusters that recompile
    # dominated the per-cluster cost).
    return regex.compile(
        f"(?:ATG)(?:[ACGT]{{3}}){{{min_aa_length - 1},}}?(?:TAA|TAG|TGA)"
    )


def is_low_complexity(seq: str) -> bool:
    """Cheap guard: a sequence dominated by one base or a long homopolymer
    is almost always a connected-components chaining artifact. Such sequences
    make the overlapped-ORF regex pathological, so skip ORF search."""
    n = len(seq)
    if n < 30:
        return False
    counts = (seq.count("A"), seq.count("C"), seq.count("G"), seq.count("T"))
    return max(counts) > 0.8 * n


def best_sense_orf(seq: str, *, min_aa_length: int = 30):
    """Longest internal-stop-free ATG ORF as ``(protein, start, end)``, or
    ``None``.

    ``start``/``end`` are 0-based half-open offsets into ``seq``; the interval
    spans the start codon through the stop codon inclusive, so its length is
    always a multiple of 3 (which the fold stage's length rules rely on).
    """
    if len(seq) < min_aa_length * 3 or is_low_complexity(seq):
        return None
    s = seq.upper().replace("U", "T")
    pat = orf_regex(min_aa_length)
    # Best (longest) internal-stop-free protein per (frame, stop position).
    best: dict[tuple[int, int], tuple] = {}
    for m in pat.finditer(s, overlapped=True):
        nt = s[m.start() : m.end()]
        prot = translate(nt[:-3], codon_table=ORF_CODON_TABLE, partial="discard")
        if "*" in prot or len(prot) < min_aa_length:
            continue
        key = (m.start() % 3, m.end())
        prior = best.get(key)
        if prior is None or len(prot) > len(prior[0]):
            best[key] = (prot, m.start(), m.end())
    if not best:
        return None
    prot, st, en = max(best.values(), key=lambda x: len(x[0]))
    return prot, int(st), int(en)


__all__ = [
    "ORF_CODON_TABLE",
    "best_sense_orf",
    "is_low_complexity",
    "orf_regex",
]
