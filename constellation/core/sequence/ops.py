"""Alphabet-agnostic sequence operations.

Functions here operate on `str` sequences against an `Alphabet` of any
kind. Alphabet-specific math (translation, ORFs, cleavage) lives in
`nucleic.py` and `protein.py`. ProForma 2.0 modseq parsing lives in
``constellation.core.sequence.proforma`` (peptide-only by spec; nucleic
modifications get a sibling module when nanopore detection lands).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from constellation.core.sequence.alphabets import (
    AA,
    AA_IUPAC,
    DNA,
    DNA_IUPAC,
    RNA,
    RNA_IUPAC,
    Alphabet,
    degenerate_ok,
)

# ──────────────────────────────────────────────────────────────────────
# Identification + validation
# ──────────────────────────────────────────────────────────────────────


# Order matters: canonical comes first so a sequence that fits both
# canonical *and* IUPAC resolves to canonical. AA last because some pure
# DNA sequences (e.g. "GATTACA") fit the AA alphabet too.
_DEFAULT_CANDIDATES: tuple[Alphabet, ...] = (
    DNA,
    RNA,
    DNA_IUPAC,
    RNA_IUPAC,
    AA,
    AA_IUPAC,
)


def identify_alphabet(
    sequence: str,
    candidates: Iterable[Alphabet] = _DEFAULT_CANDIDATES,
) -> Alphabet:
    """Return the first alphabet from `candidates` that contains every
    character of `sequence`. Raises ValueError if none fit.

    With the default candidate order, canonical alphabets are preferred
    over IUPAC, and DNA before RNA — a sequence with `T` but no `U`
    resolves to DNA. AA candidates come last; ambiguous sequences that
    fit both nucleic and amino-acid alphabets (e.g. "GATTACA") will
    be classified as DNA, which is almost always what the caller wants.
    Override with an explicit `candidates` argument to change priority.
    """
    if not sequence:
        raise ValueError("cannot identify alphabet for empty sequence")
    upper = sequence.upper()
    for alpha in candidates:
        if alpha.validate(upper):
            return alpha
    raise ValueError(
        f"sequence does not match any candidate alphabet "
        f"({[a.name for a in candidates]})"
    )


def validate(sequence: str, alphabet: Alphabet) -> None:
    """Raise ValueError listing offending positions if any character of
    `sequence` is not in `alphabet`. No-op on empty sequences."""
    bad = [(i, c) for i, c in enumerate(sequence) if not alphabet.contains(c)]
    if bad:
        sample = ", ".join(f"{c!r}@{i}" for i, c in bad[:5])
        more = "" if len(bad) <= 5 else f" (and {len(bad) - 5} more)"
        raise ValueError(
            f"sequence has {len(bad)} character(s) not in alphabet "
            f"{alphabet.name!r}: {sample}{more}"
        )


def normalize(sequence: str, alphabet: Alphabet) -> str:
    """Strip whitespace, uppercase, and (for RNA alphabets) substitute
    `T` → `U`. Used as a common entry-point cleanup before validation."""
    out = "".join(sequence.split()).upper()
    if alphabet.kind == "rna":
        out = out.replace("T", "U")
    elif alphabet.kind == "dna":
        out = out.replace("U", "T")
    return out


# ──────────────────────────────────────────────────────────────────────
# Subsequence iteration
# ──────────────────────────────────────────────────────────────────────


@degenerate_ok
def kmerize(sequence: str, k: int, *, step: int = 1) -> list[str]:
    """All length-`k` substrings of `sequence`, advancing by `step`.

    Tolerates degenerate characters — the caller decides whether to
    filter. Returns a list (not generator) since k-mer counts and
    embeddings nearly always need random access.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}")
    n = len(sequence)
    if n < k:
        return []
    return [sequence[i : i + k] for i in range(0, n - k + 1, step)]


@degenerate_ok
def sliding_window(sequence: str, size: int, *, step: int = 1) -> Iterator[str]:
    """Generator form of `kmerize` for memory-bounded iteration."""
    if size <= 0:
        raise ValueError(f"size must be positive; got {size}")
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}")
    n = len(sequence)
    for i in range(0, max(0, n - size + 1), step):
        yield sequence[i : i + size]


@degenerate_ok
def hamming_distance(a: str, b: str) -> int:
    """Number of positions at which `a` and `b` differ. Raises if lengths
    differ. Degenerate matches (e.g. `N` vs `A`) count as mismatches —
    callers needing IUPAC-aware matching should expand both sides via
    `alphabets.expand_token` first."""
    if len(a) != len(b):
        raise ValueError(f"hamming_distance requires equal lengths; got {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)


__all__ = [
    "identify_alphabet",
    "validate",
    "normalize",
    "kmerize",
    "sliding_window",
    "hamming_distance",
]
