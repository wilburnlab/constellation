"""Biological alphabets — canonical, IUPAC-degenerate, and the rules
that gate composition-aware operations.

An `Alphabet` is a frozen dataclass naming a token set, an optional gap
and stop, and (for canonical alphabets) a `dict[str, Composition]`
mapping each token to its residue-form chemical composition. Degenerate
alphabets carry no compositions — by design, since degenerate codes
cannot be assigned a single composition.

Public alphabets:

    DNA           ACGT
    RNA           ACGU
    AA            ACDEFGHIKLMNPQRSTVWY  (with residue compositions)
    DNA_IUPAC     adds R Y S W K M B D H V N
    RNA_IUPAC     same set; U replaces T
    AA_IUPAC      adds B Z J X U O

Degeneracy expansion:

    DEGENERATE_DNA / DEGENERATE_RNA   N → ACGT / ACGU; R → AG; B → CGT; ...
    DEGENERATE_AA                     B → ND; Z → QE; J → IL; X → all 20

Complement maps:

    COMPLEMENT_DNA / COMPLEMENT_RNA   canonical-only
    COMPLEMENT_DNA_IUPAC / RNA_IUPAC  with full IUPAC pairing

Decorators:

    requires_canonical    — wrapper validates `alphabet=` kwarg if present;
                            functions also use it for documentation.
    degenerate_ok         — informational marker.

Residue compositions are in residue (in-chain) form — no terminal H
or OH. `core.sequence.protein.peptide_composition` adds those when
assembling a peptide.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Callable, Literal, TypeVar

from constellation.core.chem.composition import Composition

# ──────────────────────────────────────────────────────────────────────
# Alphabet dataclass
# ──────────────────────────────────────────────────────────────────────


AlphabetKind = Literal["dna", "rna", "aa", "codon"]


@dataclass(frozen=True, slots=True)
class Alphabet:
    """Named token set with optional residue compositions.

    Canonical alphabets (`degenerate=False`) carry a `compositions` dict
    so callers can build chemical compositions by summing residue
    contributions. Degenerate alphabets set `compositions=None` and
    `degenerate=True`; functions that need compositions reject them
    via `requires_canonical`.
    """

    name: str
    tokens: tuple[str, ...]
    kind: AlphabetKind
    degenerate: bool
    gap: str | None = "-"
    stop: str | None = None
    compositions: dict[str, Composition] | None = field(default=None)

    def __post_init__(self) -> None:
        # Compositions must be present iff alphabet is canonical.
        if self.degenerate and self.compositions is not None:
            raise ValueError(
                f"degenerate alphabet {self.name!r} cannot carry compositions"
            )

    def contains(self, token: str) -> bool:
        if token in self.tokens:
            return True
        if self.gap is not None and token == self.gap:
            return True
        if self.stop is not None and token == self.stop:
            return True
        return False

    def validate(self, sequence: str) -> bool:
        """True iff every character of `sequence` is in `tokens`, gap, or stop."""
        return all(self.contains(c) for c in sequence)

    def __repr__(self) -> str:  # short, since tokens can be long
        return f"Alphabet(name={self.name!r}, kind={self.kind!r}, degenerate={self.degenerate})"


# ──────────────────────────────────────────────────────────────────────
# Residue compositions (in-chain form) — drives canonical AA alphabet
# ──────────────────────────────────────────────────────────────────────


def _comp(c: int, h: int, n: int, o: int, s: int = 0, p: int = 0) -> Composition:
    return Composition.from_dict({"C": c, "H": h, "N": n, "O": o, "S": s, "P": p})


# Source: residue (in-chain) compositions — sum 20 of these + H2O for a
# free peptide. Matches Cartographer's `AMINO_ACID_COMPOSITIONS`.
_AA_RESIDUE_COMPS: dict[str, Composition] = {
    "A": _comp(3, 5, 1, 1),
    "C": _comp(3, 5, 1, 1, s=1),
    "D": _comp(4, 5, 1, 3),
    "E": _comp(5, 7, 1, 3),
    "F": _comp(9, 9, 1, 1),
    "G": _comp(2, 3, 1, 1),
    "H": _comp(6, 7, 3, 1),
    "I": _comp(6, 11, 1, 1),
    "K": _comp(6, 12, 2, 1),
    "L": _comp(6, 11, 1, 1),
    "M": _comp(5, 9, 1, 1, s=1),
    "N": _comp(4, 6, 2, 2),
    "P": _comp(5, 7, 1, 1),
    "Q": _comp(5, 8, 2, 2),
    "R": _comp(6, 12, 4, 1),
    "S": _comp(3, 5, 1, 2),
    "T": _comp(4, 7, 1, 2),
    "V": _comp(5, 9, 1, 1),
    "W": _comp(11, 10, 2, 1),
    "Y": _comp(9, 9, 1, 2),
}


# ──────────────────────────────────────────────────────────────────────
# Canonical alphabets
# ──────────────────────────────────────────────────────────────────────


DNA: Alphabet = Alphabet(
    name="DNA",
    tokens=("A", "C", "G", "T"),
    kind="dna",
    degenerate=False,
    stop=None,
)

RNA: Alphabet = Alphabet(
    name="RNA",
    tokens=("A", "C", "G", "U"),
    kind="rna",
    degenerate=False,
    stop=None,
)

AA: Alphabet = Alphabet(
    name="AA",
    tokens=tuple("ACDEFGHIKLMNPQRSTVWY"),
    kind="aa",
    degenerate=False,
    stop="*",
    compositions=dict(_AA_RESIDUE_COMPS),
)


# ──────────────────────────────────────────────────────────────────────
# IUPAC degenerate alphabets
# ──────────────────────────────────────────────────────────────────────


_IUPAC_NUC_DEGEN = ("R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N")

DNA_IUPAC: Alphabet = Alphabet(
    name="DNA_IUPAC",
    tokens=DNA.tokens + _IUPAC_NUC_DEGEN,
    kind="dna",
    degenerate=True,
)

RNA_IUPAC: Alphabet = Alphabet(
    name="RNA_IUPAC",
    tokens=RNA.tokens + _IUPAC_NUC_DEGEN,
    kind="rna",
    degenerate=True,
)

# Extended AA tokens. B/Z/J/X are IUPAC ambiguity codes; U (Sec) and
# O (Pyl) are real but rare amino acids encoded by recoded stop codons.
# Lumped here because most consumers want a single "tolerant" alphabet.
AA_IUPAC: Alphabet = Alphabet(
    name="AA_IUPAC",
    tokens=AA.tokens + ("B", "Z", "J", "X", "U", "O"),
    kind="aa",
    degenerate=True,
    stop="*",
)


ALPHABETS: dict[str, Alphabet] = {
    a.name: a for a in (DNA, RNA, AA, DNA_IUPAC, RNA_IUPAC, AA_IUPAC)
}


# ──────────────────────────────────────────────────────────────────────
# Degeneracy expansion tables
# ──────────────────────────────────────────────────────────────────────

# DNA. Each degenerate code maps to the canonical bases it represents.
DEGENERATE_DNA: dict[str, tuple[str, ...]] = {
    "R": ("A", "G"),
    "Y": ("C", "T"),
    "S": ("C", "G"),
    "W": ("A", "T"),
    "K": ("G", "T"),
    "M": ("A", "C"),
    "B": ("C", "G", "T"),
    "D": ("A", "G", "T"),
    "H": ("A", "C", "T"),
    "V": ("A", "C", "G"),
    "N": ("A", "C", "G", "T"),
}

# RNA — same codes, U substitutes T.
DEGENERATE_RNA: dict[str, tuple[str, ...]] = {
    code: tuple("U" if b == "T" else b for b in bases)
    for code, bases in DEGENERATE_DNA.items()
}

# Amino acids. B/Z/J carry well-defined ambiguities; X is "any of 20".
DEGENERATE_AA: dict[str, tuple[str, ...]] = {
    "B": ("N", "D"),
    "Z": ("Q", "E"),
    "J": ("I", "L"),
    "X": tuple(AA.tokens),
}


# ──────────────────────────────────────────────────────────────────────
# Complement maps
# ──────────────────────────────────────────────────────────────────────


COMPLEMENT_DNA: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

COMPLEMENT_RNA: dict[str, str] = {"A": "U", "U": "A", "C": "G", "G": "C"}

# IUPAC base-pairing. Self-complementary codes (S, W, N) map to themselves.
# B (not-A) ↔ V (not-T); D (not-C) ↔ H (not-G).
_COMPLEMENT_IUPAC_DEGEN: dict[str, str] = {
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "K": "M",
    "M": "K",
    "B": "V",
    "V": "B",
    "D": "H",
    "H": "D",
    "N": "N",
    "-": "-",
    ".": ".",
}

COMPLEMENT_DNA_IUPAC: dict[str, str] = {**COMPLEMENT_DNA, **_COMPLEMENT_IUPAC_DEGEN}

COMPLEMENT_RNA_IUPAC: dict[str, str] = {**COMPLEMENT_RNA, **_COMPLEMENT_IUPAC_DEGEN}


# ──────────────────────────────────────────────────────────────────────
# Decorators
# ──────────────────────────────────────────────────────────────────────


F = TypeVar("F", bound=Callable)


def requires_canonical(fn: F) -> F:
    """Mark a function as rejecting degenerate-alphabet input.

    If the function has an `alphabet` parameter, the wrapper verifies
    that any `Alphabet` passed there is canonical — otherwise raises
    ValueError. Functions that operate on raw strings without an
    alphabet kwarg should validate their inputs themselves; the
    decorator still attaches a `_requires_canonical = True` attribute
    so introspection code can identify them.
    """
    sig = inspect.signature(fn)
    has_alphabet = "alphabet" in sig.parameters

    if not has_alphabet:
        fn._requires_canonical = True  # type: ignore[attr-defined]
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        alpha = bound.arguments.get("alphabet")
        if isinstance(alpha, Alphabet) and alpha.degenerate:
            raise ValueError(
                f"{fn.__name__} requires a canonical alphabet; got {alpha.name!r}"
            )
        return fn(*args, **kwargs)

    wrapper._requires_canonical = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def degenerate_ok(fn: F) -> F:
    """Mark a function as tolerating degenerate alphabets.

    Informational only — the function body is responsible for handling
    degenerate tokens correctly.
    """
    fn._degenerate_ok = True  # type: ignore[attr-defined]
    return fn


# ──────────────────────────────────────────────────────────────────────
# Convenience accessors
# ──────────────────────────────────────────────────────────────────────


def canonical_for(alphabet: Alphabet) -> Alphabet:
    """Return the canonical alphabet matching `alphabet.kind`."""
    if not alphabet.degenerate:
        return alphabet
    if alphabet.kind == "dna":
        return DNA
    if alphabet.kind == "rna":
        return RNA
    if alphabet.kind == "aa":
        return AA
    raise ValueError(f"no canonical alphabet for kind {alphabet.kind!r}")


def expansion_table(alphabet: Alphabet) -> dict[str, tuple[str, ...]]:
    """Return the degenerate→canonical expansion table for `alphabet.kind`."""
    if alphabet.kind == "dna":
        return DEGENERATE_DNA
    if alphabet.kind == "rna":
        return DEGENERATE_RNA
    if alphabet.kind == "aa":
        return DEGENERATE_AA
    raise ValueError(f"no expansion table for kind {alphabet.kind!r}")


def expand_token(token: str, alphabet: Alphabet) -> tuple[str, ...]:
    """Canonical tokens that `token` represents under `alphabet.kind`.

    A canonical token expands to itself; a degenerate code expands to
    its IUPAC set. Unknown tokens raise KeyError.
    """
    canonical = canonical_for(alphabet)
    if token in canonical.tokens:
        return (token,)
    table = expansion_table(alphabet)
    if token in table:
        return table[token]
    raise KeyError(f"unknown token {token!r} for alphabet kind {alphabet.kind!r}")


__all__ = [
    "Alphabet",
    "AlphabetKind",
    "DNA",
    "RNA",
    "AA",
    "DNA_IUPAC",
    "RNA_IUPAC",
    "AA_IUPAC",
    "ALPHABETS",
    "DEGENERATE_DNA",
    "DEGENERATE_RNA",
    "DEGENERATE_AA",
    "COMPLEMENT_DNA",
    "COMPLEMENT_RNA",
    "COMPLEMENT_DNA_IUPAC",
    "COMPLEMENT_RNA_IUPAC",
    "requires_canonical",
    "degenerate_ok",
    "canonical_for",
    "expansion_table",
    "expand_token",
]
