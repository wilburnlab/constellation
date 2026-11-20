"""Nucleic-acid-specific sequence operations.

    reverse_complement / complement
    CodonTable + CODON_TABLES + STANDARD
    translate              degenerate-codon-aware, pluggable codon table
    find_orfs / best_orf   regex-based ORF search, both strands
    Orf                    dataclass record for ORF results
    gc_content             whole-sequence float or sliding-window tensor

Translation policy for degenerate codons: each codon is expanded into
all canonical forms it represents, every form is looked up in the codon
table, and the resulting amino-acid set is collapsed:

    one residue            → that residue (wobble synonymy: e.g. CTN → L)
    {N, D}                 → B
    {Q, E}                 → Z
    {I, L}                 → J
    anything else          → X

Stop codons in the expanded set collapse to "*" if all forms are stops,
or to "X" if mixed with non-stops (genuinely ambiguous between coding
and termination).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from importlib import resources
from typing import Literal

import numpy as np
import regex
import torch

from constellation.core.sequence.alphabets import (
    COMPLEMENT_DNA,
    COMPLEMENT_DNA_IUPAC,
    COMPLEMENT_RNA,
    COMPLEMENT_RNA_IUPAC,
    DEGENERATE_DNA,
    DNA_IUPAC,
    Alphabet,
    degenerate_ok,
)

# ──────────────────────────────────────────────────────────────────────
# Reverse complement / complement
# ──────────────────────────────────────────────────────────────────────


def _complement_table(alphabet: Alphabet) -> dict[str, str]:
    if alphabet.kind == "dna":
        return COMPLEMENT_DNA_IUPAC if alphabet.degenerate else COMPLEMENT_DNA
    if alphabet.kind == "rna":
        return COMPLEMENT_RNA_IUPAC if alphabet.degenerate else COMPLEMENT_RNA
    raise ValueError(
        f"complement requires a nucleic alphabet; got kind={alphabet.kind!r}"
    )


@degenerate_ok
def complement(seq: str, alphabet: Alphabet = DNA_IUPAC) -> str:
    """Per-base complement, no reversal. Preserves case-handling by
    applying `str.upper()` to the input — caller is responsible for any
    case-preserving needs (rare for nucleic ops)."""
    table = _complement_table(alphabet)
    upper = seq.upper()
    out: list[str] = []
    for c in upper:
        try:
            out.append(table[c])
        except KeyError:
            raise KeyError(
                f"character {c!r} has no complement under alphabet {alphabet.name!r}"
            ) from None
    return "".join(out)


@degenerate_ok
def reverse_complement(seq: str, alphabet: Alphabet = DNA_IUPAC) -> str:
    """Reverse complement under DNA or RNA alphabet (degenerate by default).

    Uses `str.maketrans` for a single-pass translation — fast for long
    sequences. Unknown characters raise.
    """
    table = _complement_table(alphabet)
    # str.maketrans takes single-char keys/values.
    trans = str.maketrans(
        "".join(table.keys()),
        "".join(table.values()),
    )
    upper = seq.upper()
    # Validate before translate — translate would silently leave
    # unknowns in place, which is worse than raising.
    bad = [c for c in upper if c not in table]
    if bad:
        raise KeyError(
            f"character(s) with no complement under {alphabet.name!r}: {sorted(set(bad))}"
        )
    return upper.translate(trans)[::-1]


# ──────────────────────────────────────────────────────────────────────
# Codon tables
# ──────────────────────────────────────────────────────────────────────


_AMBIGUITY_PAIRS: dict[frozenset[str], str] = {
    frozenset({"N", "D"}): "B",
    frozenset({"Q", "E"}): "Z",
    frozenset({"I", "L"}): "J",
}


@dataclass(frozen=True, slots=True)
class CodonTable:
    """A genetic code keyed by NCBI ``transl_table`` number.

    `forward` maps every canonical (DNA-style) codon to its 1-letter
    amino acid, with stops as ``"*"``. `starts` and `stops` are the
    explicit codon sets — alternative starts (e.g. ``GTG``) are listed
    in `starts` but still translate as their normal AA in `forward`;
    `treat_first_codon_as_start=True` in `translate()` substitutes ``M``.
    """

    transl_table: int
    name: str
    forward: dict[str, str]
    starts: frozenset[str]
    stops: frozenset[str]

    def translate_codon(self, codon: str) -> str:
        """Look up a canonical codon. Raises on unknown / degenerate."""
        codon = codon.upper().replace("U", "T")
        try:
            return self.forward[codon]
        except KeyError:
            raise KeyError(
                f"codon {codon!r} not in transl_table {self.transl_table}"
            ) from None

    def expand_degenerate(self, codon: str) -> str:
        """Resolve a possibly-degenerate codon to a single residue token.

        Algorithm: expand each base to its canonical alternatives,
        enumerate the cartesian product, look up each canonical codon,
        and collapse the resulting residue set per the module docstring's
        rules. Pure-canonical codons short-circuit to a direct lookup.
        """
        codon = codon.upper().replace("U", "T")
        if len(codon) != 3:
            raise ValueError(f"codon must be 3 nt; got {codon!r}")

        # Fast path: every base is canonical.
        if all(b in "ACGT" for b in codon):
            return self.forward.get(codon, "X")

        # Expand each base.
        per_base: list[tuple[str, ...]] = []
        for b in codon:
            if b in "ACGT":
                per_base.append((b,))
            elif b in DEGENERATE_DNA:
                per_base.append(DEGENERATE_DNA[b])
            else:
                # Unknown character → cannot resolve.
                return "X"

        residues: set[str] = set()
        for b1 in per_base[0]:
            for b2 in per_base[1]:
                for b3 in per_base[2]:
                    aa = self.forward.get(b1 + b2 + b3, "X")
                    residues.add(aa)

        # Collapse the set.
        if len(residues) == 1:
            return next(iter(residues))
        # Mixed coding/stop → genuinely ambiguous → X.
        if "*" in residues and len(residues) > 1:
            return "X"
        # Defined ambiguity codes for two-residue collapses.
        if len(residues) == 2:
            key = frozenset(residues)
            if key in _AMBIGUITY_PAIRS:
                return _AMBIGUITY_PAIRS[key]
        return "X"


def _load_codon_tables() -> dict[int, CodonTable]:
    with resources.files("constellation.data").joinpath("codon_tables.json").open() as f:
        doc = json.load(f)
    out: dict[int, CodonTable] = {}
    for rec in doc["tables"]:
        out[rec["transl_table"]] = CodonTable(
            transl_table=rec["transl_table"],
            name=rec["name"],
            forward=dict(rec["forward"]),
            starts=frozenset(rec["starts"]),
            stops=frozenset(rec["stops"]),
        )
    return out


CODON_TABLES: dict[int, CodonTable] = _load_codon_tables()
STANDARD: CodonTable = CODON_TABLES[1]


# ──────────────────────────────────────────────────────────────────────
# Translation
# ──────────────────────────────────────────────────────────────────────


#: ``str.translate`` table deleting canonical bases: what is left of a
#: sequence is its non-canonical characters.
_DELETE_ACGT = {ord(b): None for b in "ACGT"}
#: Code point -> base code (A 0, C 1, G 2, T 3; anything else 4), for code
#: points below 128.
_BASE_CODE = np.full(128, 4, dtype=np.int16)
for _i, _b in enumerate("ACGT"):
    _BASE_CODE[ord(_b)] = _i
del _i, _b
_CANONICAL_CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
#: Per codon table: its 64 residues as bytes, indexed 16a + 4b + c. Keyed by
#: ``id`` and holding the table, so an id is never reused while cached.
_RESIDUE_BYTES: dict[int, tuple["CodonTable", np.ndarray]] = {}


#: Byte -> base code, for ASCII text.
_BASE_CODE_BYTE = np.full(256, 4, dtype=np.int16)
_BASE_CODE_BYTE[:128] = _BASE_CODE


def _base_codes(s: str) -> np.ndarray:
    """Each character of ``s`` as a base code (A 0, C 1, G 2, T 3, other 4)."""
    if s.isascii():
        return _BASE_CODE_BYTE[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
    cp = np.frombuffer(s.encode("utf-32-le"), dtype="<u4")
    return np.where(cp < 128, _BASE_CODE[np.minimum(cp, 127)], 4).astype(np.int16)


def _residue_bytes(codon_table: "CodonTable") -> np.ndarray:
    got = _RESIDUE_BYTES.get(id(codon_table))
    if got is None or got[0] is not codon_table:
        lut = np.frombuffer(
            "".join(codon_table.forward.get(c, "X") for c in _CANONICAL_CODONS).encode("ascii"),
            dtype=np.uint8,
        ).copy()
        got = _RESIDUE_BYTES[id(codon_table)] = (codon_table, lut)
    return got[1]


def _translate_canonical(s: str, codon_table: "CodonTable") -> str:
    """``s`` (canonical bases only, a whole number of codons) translated."""
    codes = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    b = _BASE_CODE[codes].reshape(-1, 3)
    idx = b[:, 0] * 16 + b[:, 1] * 4 + b[:, 2]
    return _residue_bytes(codon_table)[idx].tobytes().decode("ascii")


@degenerate_ok
def translate(
    seq: str,
    *,
    codon_table: CodonTable = STANDARD,
    frame: int = 0,
    to_stop: bool = False,
    treat_first_codon_as_start: bool = False,
    partial: Literal["discard", "pad", "raise"] = "discard",
) -> str:
    """Translate a nucleic-acid sequence into a protein string.

    `frame` is 0/1/2; the leading ``frame`` nucleotides are skipped. If
    the remaining length is not divisible by 3:

        ``"discard"`` (default)  drop the trailing 1–2 nt silently
        ``"pad"``                pad with N to a full codon and translate
        ``"raise"``              raise ValueError

    Stops translate as ``"*"``. With ``to_stop=True`` the output is
    truncated at the first stop and the ``"*"`` is omitted.

    Degenerate codons are resolved via `codon_table.expand_degenerate`.
    With `treat_first_codon_as_start=True`, the first codon — if it is
    in `codon_table.starts` — is emitted as ``M``; subsequent codons
    use the normal forward map.
    """
    if frame not in (0, 1, 2):
        raise ValueError(f"frame must be 0, 1, or 2; got {frame}")

    s = seq.upper().replace("U", "T")
    if frame:
        s = s[frame:]
    rem = len(s) % 3
    if rem:
        if partial == "raise":
            raise ValueError(
                f"sequence length not divisible by 3 in frame={frame}; "
                f"got {len(seq)} nt → {len(s)} after trim"
            )
        if partial == "pad":
            s = s + "N" * (3 - rem)
        else:  # discard
            s = s[: len(s) - rem]

    if s and not s.translate(_DELETE_ACGT):
        # Every base canonical: one table lookup per codon, in arrays.
        # Exactly what the loop below gives (`expand_degenerate`'s fast
        # path is `forward.get(codon, "X")`).
        prot = _translate_canonical(s, codon_table)
        if treat_first_codon_as_start and s[:3] in codon_table.starts:
            prot = "M" + prot[1:]
        if to_stop:
            stop = prot.find("*")
            if stop >= 0:
                prot = prot[:stop]
        return prot

    out: list[str] = []
    n = len(s)
    for i in range(0, n, 3):
        codon = s[i : i + 3]
        if i == 0 and treat_first_codon_as_start and codon in codon_table.starts:
            aa = "M"
        else:
            aa = codon_table.expand_degenerate(codon)
        if aa == "*":
            if to_stop:
                return "".join(out)
            out.append(aa)
        else:
            out.append(aa)
    return "".join(out)


# ──────────────────────────────────────────────────────────────────────
# ORF finding
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Orf:
    """A single ORF result.

    `start` and `end` are 0-indexed half-open coordinates on the strand
    indicated by `strand`. For ``"-"`` ORFs the coordinates refer to the
    reverse-complemented sequence; convert with ``len(seq) - end`` if
    you need the parent-strand coordinate.
    """

    nucleotide: str
    protein: str
    start: int
    end: int
    strand: Literal["+", "-"]
    length: int  # protein length (excl. stop)
    transl_table: int


# Build an ORF regex on the fly per codon table — start codons and
# stop codons differ between tables. Using `regex.finditer(overlapped=True)`
# so overlapping ORFs from alternative starts are all captured.
def _build_orf_regex(codon_table: CodonTable, min_aa_length: int) -> regex.Pattern:
    starts_alt = "|".join(sorted(codon_table.starts))
    stops_alt = "|".join(sorted(codon_table.stops))
    pat = (
        f"(?:{starts_alt})"
        f"(?:[ACGT]{{3}}){{{min_aa_length - 1},}}?"
        f"(?:{stops_alt})"
    )
    return regex.compile(pat)


@degenerate_ok
def find_orfs(
    seq: str,
    *,
    codon_table: CodonTable = STANDARD,
    min_aa_length: int = 30,
    both_strands: bool = True,
    longest_per_stop: bool = True,
) -> list[Orf]:
    """Locate all ORFs of length ≥ `min_aa_length` (in amino acids).

    An ORF is a stretch starting at a codon in `codon_table.starts` and
    ending at the first in-frame codon in `codon_table.stops`. Search
    is performed on the canonical strand and (if `both_strands=True`)
    on the reverse complement.

    `longest_per_stop=True` keeps only the longest ORF terminating at
    any given stop position on each strand — collapsing alternative
    in-frame starts that share a single stop. With
    `longest_per_stop=False`, every alternative-start ORF is returned.

    Note: degenerate bases (N/R/Y/...) in the input are *not* expanded
    into hypothetical ORF starts; the regex matches only canonical
    codons. Sequences with extensive degeneracy may need expansion or
    base-calling consensus before ORF search.
    """
    s = seq.upper().replace("U", "T")
    pat = _build_orf_regex(codon_table, min_aa_length)
    results: list[Orf] = []

    for strand_seq, strand in (
        (s, "+"),
        *(((reverse_complement(s, alphabet=DNA_IUPAC), "-"),) if both_strands else ()),
    ):
        # Per-stop deduplication: key by (frame, end).
        per_key: dict[tuple[int, int], Orf] = {}
        for m in pat.finditer(strand_seq, overlapped=True):
            start, end = m.start(), m.end()
            nt = strand_seq[start:end]
            # Translate without the trailing stop for the .protein field.
            prot = translate(
                nt[:-3], codon_table=codon_table, partial="discard"
            )
            length = len(prot)
            if length < min_aa_length:
                continue
            orf = Orf(
                nucleotide=nt,
                protein=prot,
                start=start,
                end=end,
                strand=strand,  # type: ignore[arg-type]
                length=length,
                transl_table=codon_table.transl_table,
            )
            key = (start % 3, end)
            if longest_per_stop:
                prior = per_key.get(key)
                if prior is None or length > prior.length:
                    per_key[key] = orf
            else:
                results.append(orf)
        if longest_per_stop:
            results.extend(per_key.values())

    # Stable order: by strand, then start.
    results.sort(key=lambda o: (o.strand, o.start, -o.length))
    return results


def best_orf(
    seq: str,
    *,
    codon_table: CodonTable = STANDARD,
    min_aa_length: int = 30,
    both_strands: bool = True,
) -> Orf | None:
    """Longest ORF over both strands, or None if no ORF passes
    `min_aa_length`. Thin wrapper over `find_orfs`."""
    orfs = find_orfs(
        seq,
        codon_table=codon_table,
        min_aa_length=min_aa_length,
        both_strands=both_strands,
        longest_per_stop=True,
    )
    if not orfs:
        return None
    return max(orfs, key=lambda o: o.length)


#: A codon's class in :func:`longest_orf`: inside an ORF, a start, a stop,
#: or a codon holding a non-canonical base (code 64).
_CODON_INNER, _CODON_START, _CODON_STOP, _CODON_BAD = 0, 1, 2, 3


@functools.lru_cache(maxsize=32)
def _codon_classes(starts: frozenset[str], stops: frozenset[str]) -> np.ndarray:
    """Codon code (16a + 4b + c; 64 for a non-canonical codon) -> class."""
    lut = np.full(65, _CODON_INNER, dtype=np.uint8)
    lut[64] = _CODON_BAD
    for codons, cls in ((starts, _CODON_START), (stops, _CODON_STOP)):
        for c in codons:
            c = c.upper().replace("U", "T")
            if len(c) == 3 and all(b in "ACGT" for b in c):
                lut["ACGT".index(c[0]) * 16 + "ACGT".index(c[1]) * 4 + "ACGT".index(c[2])] = cls
    return lut


def longest_orf(
    seq: str,
    *,
    codon_table: CodonTable = STANDARD,
    min_aa_length: int = 30,
) -> Orf | None:
    """The longest forward-strand ORF of at least ``min_aa_length``
    residues, or None: a start codon of ``codon_table``, then codons of
    canonical bases up to the first in-frame stop, the stop included.

    Ties go to the earliest start. Exactly what enumerating
    :func:`find_orfs`' matches, dropping those with an internal stop and
    keeping the longest per stop would give, in one pass over the codons:
    a start is an ORF's iff the first in-frame codon after it that is a
    stop or holds a non-canonical base is a stop at least
    ``min_aa_length`` codons on, and of the starts sharing a stop the
    earliest is the longest. ``seq`` is read upper-cased with U as T.
    """
    s = seq.upper().replace("U", "T")
    n = len(s)
    if n < 3 * (min_aa_length + 1):
        return None
    codes = _base_codes(s)
    # The codon starting at every position, classified once; a frame is
    # every third of them.
    bad = codes >= 4
    cod = codes[:-2] * 16 + codes[1:-1] * 4 + codes[2:]
    cod[bad[:-2] | bad[1:-1] | bad[2:]] = 64
    cls = _codon_classes(codon_table.starts, codon_table.stops)[cod]
    best: tuple[int, int, int] | None = None  # (length, start, end)
    for frame in range(3):
        k = cls[frame::3]
        if k.size <= min_aa_length:
            continue
        barrier = np.flatnonzero(k >= _CODON_STOP)
        starts = np.flatnonzero(k == _CODON_START)
        if starts.size == 0 or barrier.size == 0:
            continue
        at = np.searchsorted(barrier, starts, side="right")
        has = at < barrier.size
        i = starts[has]
        j = barrier[at[has]]
        ok = (k[j] == _CODON_STOP) & (j - i >= min_aa_length)
        i, j = i[ok], j[ok]
        if i.size == 0:
            continue
        # Starts are ascending, so the first start of each stop is its
        # earliest; the longest of those is the frame's best, earliest first.
        first = np.ones(i.size, dtype=bool)
        first[1:] = j[1:] != j[:-1]
        i, j = i[first], j[first]
        lengths = j - i
        w = int(np.argmax(lengths))  # the first maximum: the earliest start
        cand = (int(lengths[w]), frame + 3 * int(i[w]), frame + 3 * int(j[w]) + 3)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
            best = cand
    if best is None:
        return None
    length, start, end = best
    nt = s[start:end]
    return Orf(
        nucleotide=nt,
        protein=translate(nt[:-3], codon_table=codon_table, partial="discard"),
        start=start,
        end=end,
        strand="+",
        length=length,
        transl_table=codon_table.transl_table,
    )


# ──────────────────────────────────────────────────────────────────────
# GC content
# ──────────────────────────────────────────────────────────────────────


# Fractional GC contribution per IUPAC base. Canonical: G/C → 1.0,
# A/T/U → 0.0. Degenerate codes contribute the fraction of their
# expansions that are G or C (e.g. S → 1.0, B = {C,G,T} → 2/3, N → 1/2).
_GC_CONTRIBUTION: dict[str, float] = {
    "G": 1.0,
    "C": 1.0,
    "A": 0.0,
    "T": 0.0,
    "U": 0.0,
    "S": 1.0,  # C|G
    "W": 0.0,  # A|T
    "R": 0.5,  # A|G
    "Y": 0.5,  # C|T
    "K": 0.5,  # G|T
    "M": 0.5,  # A|C
    "B": 2 / 3,  # C|G|T
    "V": 2 / 3,  # A|C|G
    "D": 1 / 3,  # A|G|T
    "H": 1 / 3,  # A|C|T
    "N": 0.5,  # A|C|G|T
}


@degenerate_ok
def gc_content(
    seq: str,
    *,
    window: int | None = None,
    step: int = 1,
) -> float | torch.Tensor:
    """GC fraction in [0, 1]. Whole-sequence float when `window=None`,
    else a 1-D float32 tensor of windowed fractions advancing by `step`.

    Degenerate bases contribute fractionally (e.g. ``N`` → 0.5,
    ``S`` → 1.0); ambiguous-but-non-GC codes (``W``) → 0. Unknown
    characters raise KeyError.
    """
    upper = seq.upper()
    contrib = [_GC_CONTRIBUTION.get(c) for c in upper]
    if any(c is None for c in contrib):
        bad = sorted({c for c in upper if c not in _GC_CONTRIBUTION})
        raise KeyError(f"unknown base(s) for GC content: {bad}")
    arr = torch.tensor(contrib, dtype=torch.float32)
    if window is None:
        if arr.numel() == 0:
            return 0.0
        return float(arr.mean().item())
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}")
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}")
    n = arr.numel()
    if n < window:
        return torch.empty(0, dtype=torch.float32)
    # Cumulative-sum trick for O(N) windowing.
    cs = torch.cat([torch.zeros(1, dtype=torch.float32), arr.cumsum(dim=0).to(torch.float32)])
    starts = torch.arange(0, n - window + 1, step)
    return (cs[starts + window] - cs[starts]) / float(window)


__all__ = [
    "complement",
    "reverse_complement",
    "CodonTable",
    "CODON_TABLES",
    "STANDARD",
    "translate",
    "Orf",
    "find_orfs",
    "best_orf",
    "longest_orf",
    "gc_content",
]
