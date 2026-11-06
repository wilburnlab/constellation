"""Token vocabularies: what a model reads a peptidoform as.

Three layers, each usable alone:

* :class:`Sites` — a peptidoform reduced to what a sequence model can be
  told: its residues and the UNIMOD accessions at its N-terminus, each
  residue and its C-terminus. ``Sites.from_peptidoform`` refuses what has
  no such reading (isotope labels, ranges, labile or unlocalised
  modifications, a modification named by mass or name rather than by
  accession), so a model never receives a molecule it was not told about.
* a *spelling*: one character per position, N-terminus first, end mark
  last (``-PEPTIDEK_``). How a model spells a peptide is its own
  convention; :class:`SiteTokens` is the plain accession-keyed one
  (a character for each (residue, accession) and each N-terminal
  accession), and model families with stranger conventions write their
  own (``tokenize.jchronologer``).
* :class:`Vocabulary` — the characters a model reads, the integer id it
  reads each as, and the fixed width it is padded to. ``encode`` turns
  many spellings into one int64 array in a few numpy operations.

A refusal is a :class:`TokenizationRefused` carrying one of
:data:`REASONS` (stable, so drops can be counted) and a detail naming
the position and modification.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from constellation.core.sequence.proforma import ModRef, MultiPeptidoform, Peptidoform, parse_proforma

__all__ = [
    "LENGTH_OUT_OF_RANGE",
    "REASONS",
    "UNKNOWN_RESIDUE",
    "UNSUPPORTED_FORM",
    "UNSUPPORTED_MODIFICATION",
    "SiteTokens",
    "Sites",
    "TokenizationRefused",
    "TokenizedBatch",
    "Vocabulary",
    "accession_of",
]

#: Longer or shorter than the model takes.
LENGTH_OUT_OF_RANGE = "length_out_of_range"
#: A residue letter the model has no token for (U, O, X, B, Z, J, ...).
UNKNOWN_RESIDUE = "unknown_residue"
#: A modification, at that site, the model has no token for.
UNSUPPORTED_MODIFICATION = "unsupported_modification"
#: Something a sequence of sites cannot say: isotope labels, fixed-mod
#: declarations, labile, unlocalised or ranged modifications, several chains.
UNSUPPORTED_FORM = "unsupported_form"
REASONS = (LENGTH_OUT_OF_RANGE, UNKNOWN_RESIDUE, UNSUPPORTED_MODIFICATION, UNSUPPORTED_FORM)


class TokenizationRefused(ValueError):
    """A peptidoform a model cannot be given, and why."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"unknown refusal reason {reason!r}")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def accession_of(mod: ModRef | None) -> str:
    """``"UNIMOD:35"`` for a modification named by UNIMOD accession.

    Anything else (a mass, a formula, a name, another vocabulary) is
    refused: a model's tokens stand for particular modifications, and a
    mass or a name is not one (two UNIMOD entries can share a mass).
    Library peptides are canonical, so their modifications always carry
    accessions.
    """
    if mod is None or mod.cv != "UNIMOD" or not mod.accession:
        what = (
            "nothing" if mod is None
            else f"{mod.mass_delta:+g}" if mod.mass_delta is not None
            else mod.name or mod.formula or mod.glycan or repr(mod)
        )
        raise TokenizationRefused(
            UNSUPPORTED_MODIFICATION, f"a modification without a UNIMOD accession ({what})"
        )
    return f"UNIMOD:{mod.accession}"


# The canonical form a library writes (``format_proforma`` of a canonical
# peptide) when every modification is a UNIMOD accession: optional
# N-terminal accessions and a hyphen, residues each followed by any number
# of accessions, optional hyphen and C-terminal accessions. Anything else
# (labels, names, masses, groups, charges, ...) leaves a character in the
# residues that is not an upper-case letter, and goes through
# ``parse_proforma``.
_BRACKET = re.compile(r"\[(UNIMOD:\d+)\]")
_TERMINAL = re.compile(r"(?:\[UNIMOD:\d+\])+")


def _terminal(block: str) -> tuple[str, ...] | None:
    if _TERMINAL.fullmatch(block) is None:
        return None
    return tuple(sorted(_BRACKET.findall(block)))


@dataclass(frozen=True, slots=True)
class Sites:
    """A peptidoform as a model reads it: residues, and the UNIMOD
    accessions at the N-terminus, at each residue and at the C-terminus.

    Each position's accessions are sorted, so two spellings of one
    molecule are one ``Sites``.
    """

    residues: str
    n_term: tuple[str, ...] = ()
    residue_mods: tuple[tuple[str, ...], ...] = ()
    c_term: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.residue_mods:
            object.__setattr__(self, "residue_mods", ((),) * len(self.residues))
        elif len(self.residue_mods) != len(self.residues):
            raise ValueError(
                f"{len(self.residue_mods)} residue modification entries for {len(self.residues)} residues"
            )

    @classmethod
    def from_peptidoform(cls, peptide: Peptidoform | MultiPeptidoform | str) -> Sites:
        """Reduce a peptidoform (or a ProForma string) to its sites.

        Raises :class:`TokenizationRefused` (``unsupported_form`` or
        ``unsupported_modification``) for what sites cannot say. A charge
        on the peptidoform is ignored: the precursor charge is a separate
        input to the models that take one.
        """
        if isinstance(peptide, str):
            fast = cls._from_canonical(peptide)
            if fast is not None:
                return fast
            peptide = parse_proforma(peptide)
        if isinstance(peptide, MultiPeptidoform):
            raise TokenizationRefused(UNSUPPORTED_FORM, "more than one chain")
        if peptide.global_isotopes:
            raise TokenizationRefused(
                UNSUPPORTED_FORM,
                f"isotope label <{','.join(peptide.global_isotopes)}> (predict the unlabelled peptide)",
            )
        for what, present in (
            ("a fixed-modification declaration", peptide.fixed_mods),
            ("a labile modification", peptide.labile_mods),
            ("a modification of unknown position", peptide.unknown_pos_mods),
            ("a modification on a range of residues", peptide.ranges),
            ("an adduct", peptide.adducts),
        ):
            if present:
                raise TokenizationRefused(UNSUPPORTED_FORM, what)
        tagged_all = (*peptide.n_term_mods, *peptide.c_term_mods,
                      *(t for ts in peptide.residue_mods.values() for t in ts))
        if any(t.group_id is not None for t in tagged_all):
            raise TokenizationRefused(
                UNSUPPORTED_FORM, "a cross-link, branch or localisation group"
            )
        n = len(peptide.sequence)
        mods: list[tuple[str, ...]] = [()] * n
        for index, tagged in peptide.residue_mods.items():
            mods[index] = tuple(sorted(accession_of(t.mod) for t in tagged))
        return cls(
            residues=peptide.sequence,
            n_term=tuple(sorted(accession_of(t.mod) for t in peptide.n_term_mods)),
            residue_mods=tuple(mods),
            c_term=tuple(sorted(accession_of(t.mod) for t in peptide.c_term_mods)),
        )

    @classmethod
    def _from_canonical(cls, text: str) -> Sites | None:
        """The sites of a canonical ProForma string whose modifications
        are all UNIMOD accessions, read without building a Peptidoform
        (several times faster); None for any other string."""
        n_term: tuple[str, ...] | None = ()
        c_term: tuple[str, ...] | None = ()
        body = text
        if body.startswith("["):
            end = body.find("]-")
            if end < 0:
                return None
            n_term = _terminal(body[: end + 1])
            body = body[end + 2 :]
        cut = body.rfind("-[")
        if cut >= 0:
            c_term = _terminal(body[cut + 1 :])
            body = body[:cut]
        if n_term is None or c_term is None:
            return None
        # Runs of residues alternate with accessions; each accession is on
        # the last residue before it.
        parts = _BRACKET.split(body)
        residues = "".join(parts[0::2])
        if not (residues.isascii() and residues.isalpha() and residues.isupper()):
            return None
        mods: list[tuple[str, ...]] = [()] * len(residues)
        at = -1
        stacked = False
        for run, accession in zip(parts[0::2], parts[1::2]):
            at += len(run)
            if at < 0:
                return None
            stacked |= bool(mods[at])
            mods[at] = (*mods[at], accession)
        if stacked:
            mods = [m if len(m) < 2 else tuple(sorted(m)) for m in mods]
        return cls(residues=residues, n_term=n_term, residue_mods=tuple(mods), c_term=c_term)

    def describe(self, index: int) -> str:
        """``K5[UNIMOD:1]``: residue ``index`` and its modifications."""
        mods = "".join(f"[{a}]" for a in self.residue_mods[index])
        return f"{self.residues[index]}{index + 1}{mods}"


@dataclass(frozen=True)
class SiteTokens:
    """The plain spelling: one character for each residue, for each
    (residue, UNIMOD accession) pair, and for each N-terminal accession.

    A position may hold one modification, the N-terminus one, the
    C-terminus none: what a map keyed by single accessions can say.
    """

    residue_tokens: Mapping[str, str]
    modified_residue_tokens: Mapping[tuple[str, str], str]
    n_term_tokens: Mapping[str, str]
    free_n_term: str = "-"
    end: str = "_"
    min_length: int = 1
    max_length: int | None = None

    def spell(self, sites: Sites) -> str:
        n = len(sites.residues)
        if n < self.min_length or (self.max_length is not None and n > self.max_length):
            raise TokenizationRefused(
                LENGTH_OUT_OF_RANGE, f"{n} residues (takes {self.min_length}-{self.max_length})"
            )
        if sites.c_term:
            raise TokenizationRefused(
                UNSUPPORTED_MODIFICATION, f"C-terminal modification {', '.join(sites.c_term)}"
            )
        if len(sites.n_term) > 1:
            raise TokenizationRefused(
                UNSUPPORTED_MODIFICATION, f"more than one N-terminal modification ({', '.join(sites.n_term)})"
            )
        if sites.n_term:
            head = self.n_term_tokens.get(sites.n_term[0])
            if head is None:
                raise TokenizationRefused(
                    UNSUPPORTED_MODIFICATION, f"N-terminal {sites.n_term[0]}"
                )
        else:
            head = self.free_n_term
        out = [head]
        for i, (residue, mods) in enumerate(zip(sites.residues, sites.residue_mods)):
            if not mods:
                token = self.residue_tokens.get(residue)
                if token is None:
                    raise TokenizationRefused(UNKNOWN_RESIDUE, f"{residue}{i + 1}")
            elif len(mods) == 1:
                token = self.modified_residue_tokens.get((residue, mods[0]))
                if token is None:
                    raise TokenizationRefused(UNSUPPORTED_MODIFICATION, sites.describe(i))
            else:
                raise TokenizationRefused(
                    UNSUPPORTED_MODIFICATION, f"more than one modification on {sites.describe(i)}"
                )
            out.append(token)
        out.append(self.end)
        return "".join(out)


@dataclass(frozen=True)
class TokenizedBatch:
    """Tokens for a batch of peptides, with why any were refused.

    ``tokens[i]`` is all padding where ``reasons[i]`` is set.
    """

    tokens: np.ndarray
    reasons: tuple[str | None, ...]
    details: tuple[str | None, ...]

    @property
    def accepted(self) -> np.ndarray:
        return np.fromiter((r is None for r in self.reasons), bool, len(self.reasons))

    def __len__(self) -> int:
        return len(self.reasons)


@dataclass(frozen=True)
class Vocabulary:
    """The characters a model reads, the id it reads each as, and the
    width its token rows are padded to (``max_length`` residues plus an
    N-terminal and an end token)."""

    ids: Mapping[str, int]
    max_length: int
    padding_index: int = 0
    _lut: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        lut = np.full(128, -1, dtype=np.int64)
        for char, value in self.ids.items():
            if len(char) != 1 or ord(char) >= 128:
                raise ValueError(f"vocabulary token {char!r} is not one ASCII character")
            if value == self.padding_index:
                raise ValueError(f"token {char!r} has the padding id {value}")
            lut[ord(char)] = value
        object.__setattr__(self, "_lut", lut)

    @property
    def width(self) -> int:
        return self.max_length + 2

    def encode(self, spellings: Sequence[str | None]) -> np.ndarray:
        """int64 ``[len(spellings), width]``: each spelling's ids, then
        padding; a ``None`` row is all padding."""
        n = len(spellings)
        out = np.full((n, self.width), self.padding_index, dtype=np.int64)
        lengths = np.fromiter((0 if s is None else len(s) for s in spellings), np.int64, n)
        if n == 0 or not lengths.any():
            return out
        if lengths.max() > self.width:
            row = int(lengths.argmax())
            raise ValueError(f"spelling {spellings[row]!r} is longer than the width {self.width}")
        joined = "".join(s for s in spellings if s).encode("ascii")
        ids = self._lut[np.frombuffer(joined, dtype=np.uint8)]
        if (ids < 0).any():
            bad = sorted({chr(c) for c in np.frombuffer(joined, np.uint8)[ids < 0]})
            raise ValueError(f"characters not in the vocabulary: {bad}")
        rows = np.repeat(np.arange(n), lengths)
        starts = np.cumsum(lengths) - lengths
        cols = np.arange(len(ids)) - np.repeat(starts, lengths)
        out[rows, cols] = ids
        return out

    def decode(self, tokens: np.ndarray) -> list[str]:
        """The spellings of rows of ids (padding dropped)."""
        chars = {v: k for k, v in self.ids.items()}
        return ["".join(chars[int(t)] for t in row if int(t) != self.padding_index) for row in np.atleast_2d(tokens)]
