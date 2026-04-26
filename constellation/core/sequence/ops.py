"""Alphabet-agnostic sequence operations.

Functions here operate on `str` sequences against an `Alphabet` of any
kind. Alphabet-specific math (translation, ORFs, cleavage) lives in
`nucleic.py` and `protein.py`. Modified-sequence parsing lives here
because the bracket syntax (``S[UNIMOD:21]``) is the same shape across
alphabets — protein, nucleic, or hypothetical extension.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping

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


# ──────────────────────────────────────────────────────────────────────
# Modified-sequence bracket notation (alphabet-agnostic)
# ──────────────────────────────────────────────────────────────────────

# Mass-notation form: "[+15.994]" or "[-17.027]". The leading sign is
# required to disambiguate it from string keys like "UNIMOD:35" or "Ox".
_MASS_NOTATION_RE = re.compile(r"^[+-][0-9]")


@degenerate_ok
def parse_modified_sequence(modseq: str) -> tuple[str, dict[int, str | float]]:
    """Decompose a bracket-annotated sequence into `(sequence, mods)`.

    Recognized bracket payloads:
        ``UNIMOD:N``  → kept as ``"UNIMOD:N"`` (str)
        ``+N.NNN`` / ``-N.NNN``  → kept as float (signed delta in Da)
        anything else (e.g. ``Ox``, ``Phospho``)  → kept as the literal str;
            the caller can resolve via UNIMOD aliases if needed.

    The mod dict is keyed by **0-indexed position in the stripped
    sequence** — the position of the residue the modification attaches
    to. N-terminal mods (`[Acetyl]MASTERPROTEIN`) attach to position 0;
    C-terminal mods attach to the last position.

    Examples:
        parse_modified_sequence("PEPC[UNIMOD:4]TIDE")
            → ("PEPCTIDE", {3: "UNIMOD:4"})

        parse_modified_sequence("[+42.011]MASTERPROTEIN")
            → ("MASTERPROTEIN", {0: 42.011})

        parse_modified_sequence("PEPTIDE")  # no mods
            → ("PEPTIDE", {})
    """
    sequence_chars: list[str] = []
    modifications: dict[int, str | float] = {}
    i = 0
    n = len(modseq)
    while i < n:
        c = modseq[i]
        if c == "[":
            end = modseq.find("]", i + 1)
            if end == -1:
                raise ValueError(
                    f"unterminated '[' at position {i} in {modseq!r}"
                )
            payload = modseq[i + 1 : end]
            # Position is the index of the residue this mod attaches to.
            # If the bracket is at the very start, attach to position 0
            # (the N-terminal residue, once it appears). If the bracket
            # follows a residue, attach to that residue's stripped index.
            attach = max(0, len(sequence_chars) - 1) if sequence_chars else 0
            value: str | float
            if _MASS_NOTATION_RE.match(payload):
                value = float(payload)
            else:
                value = payload
            # If multiple mods land on the same index, merge into a list.
            existing = modifications.get(attach)
            if existing is None:
                modifications[attach] = value
            elif isinstance(existing, list):
                existing.append(value)  # type: ignore[arg-type]
            else:
                modifications[attach] = [existing, value]  # type: ignore[assignment]
            i = end + 1
        else:
            sequence_chars.append(c)
            i += 1
    return "".join(sequence_chars), modifications


@degenerate_ok
def format_modified_sequence(
    sequence: str,
    modifications: Mapping[int, str | float],
) -> str:
    """Inverse of `parse_modified_sequence`. Mods on a given index are
    inserted as ``[payload]`` *after* that residue. Position 0 is the
    one exception — if a mod attaches to position 0 *and* the sequence
    is non-empty, it still emits after the first residue (matching the
    parse rule above)."""
    if not modifications:
        return sequence
    parts: list[str] = []
    for i, c in enumerate(sequence):
        parts.append(c)
        if i in modifications:
            payload = modifications[i]
            if isinstance(payload, list):
                for item in payload:
                    parts.append(f"[{_format_mod_payload(item)}]")
            else:
                parts.append(f"[{_format_mod_payload(payload)}]")
    # Mods at indices past the sequence are tolerated by the parser
    # (they attach to the last residue) but format-time we just append.
    for i, payload in modifications.items():
        if i >= len(sequence):
            if isinstance(payload, list):
                for item in payload:
                    parts.append(f"[{_format_mod_payload(item)}]")
            else:
                parts.append(f"[{_format_mod_payload(payload)}]")
    return "".join(parts)


def _format_mod_payload(value: str | float) -> str:
    if isinstance(value, str):
        return value
    return f"{value:+f}"


__all__ = [
    "identify_alphabet",
    "validate",
    "normalize",
    "kmerize",
    "sliding_window",
    "hamming_distance",
    "parse_modified_sequence",
    "format_modified_sequence",
]
