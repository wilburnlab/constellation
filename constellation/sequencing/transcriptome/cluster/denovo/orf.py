"""Longest sense-strand ORF prediction (leaf — imports only ``core``).

Shared by the consensus stage (ORF on each cluster consensus) and the
ORF-anchored EM's seeding stage (ORF on each read window). Lives in its own
leaf module so ``em/`` can use it without importing ``pipeline``.

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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from constellation.core.sequence.nucleic import STANDARD, CodonTable, longest_orf


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

    # The regular expression ORF calling used before `longest_orf`, kept
    # for comparison. ATG start … ≥ min_aa_length codons … stop. Compiled once per length
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
    always a multiple of 3 (which the fold stage's length rules rely on). Ties
    go to the earliest start.

    One pass over the codons (``core.sequence.nucleic.longest_orf``); until
    2026-11-20 every ATG's match of :func:`orf_regex` was translated in full
    and the stop-free ones kept, which gives the same answer about ten times
    slower (``tests/test_orf_one_pass.py`` keeps that form as the reference).
    """
    if len(seq) < min_aa_length * 3 or is_low_complexity(seq):
        return None
    orf = longest_orf(seq, codon_table=ORF_CODON_TABLE, min_aa_length=min_aa_length)
    if orf is None:
        return None
    return orf.protein, int(orf.start), int(orf.end)


# ── ORF prediction over many sequences (fork pool) ────────────────────

# Set in the parent before the pool forks; workers read it copy-on-write.
_POOL_SEQS: list[str] | None = None


def _orf_chunk(lo: int, hi: int, min_aa_length: int) -> list[tuple]:
    seqs = _POOL_SEQS
    assert seqs is not None
    out: list[tuple] = []
    for i in range(lo, hi):
        hit = best_sense_orf(seqs[i], min_aa_length=min_aa_length)
        if hit is not None:
            _prot, st, en = hit
            out.append((i, st, en))
    return out


def predict_orfs_parallel(
    seqs: list[str], *, min_aa_length: int, threads: int = 1, chunk: int = 20_000
) -> list[tuple]:
    """``[(index, orf_start, orf_end), …]`` for the sequences that have one.

    Sequences without a qualifying ORF are simply absent — the caller decides
    what that means (the ORF seeder drops the read; the kmer seeder keeps the
    template with a null ORF interval, because that cluster earned its
    template on read support, not on carrying a protein).

    Shared by both seeders, which call it at very different cardinalities:
    once per unique cDNA (~9.15M at PromethION scale) for ORF seeding, once
    per *elected template* (~0.9M) for kmer seeding.
    """
    global _POOL_SEQS
    n = len(seqs)
    if n == 0:
        return []
    _POOL_SEQS = seqs
    try:
        if threads <= 1 or n < chunk:
            return _orf_chunk(0, n, min_aa_length)
        bounds = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
        ctx = mp.get_context("fork")
        out: list[tuple] = []
        with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as ex:
            futs = [ex.submit(_orf_chunk, lo, hi, min_aa_length) for lo, hi in bounds]
            for fut in futs:
                out.extend(fut.result())
        return out
    finally:
        _POOL_SEQS = None


__all__ = [
    "ORF_CODON_TABLE",
    "best_sense_orf",
    "is_low_complexity",
    "orf_regex",
    "predict_orfs_parallel",
]
