"""Protein-specific sequence operations.

    Protease, ProteaseRegistry, PROTEASES   protease registry (regex rules)
    Peptide                                 (sequence, start, end, n_missed)
    cleave / cleave_sites                   regex-based proteolysis, N→C ordered
    peptide_composition                     residue sum + H2O (+ optional mods)
    peptide_mass                            convenience over composition.mass
    protein_composition                     alias of peptide_composition w/o mods

The cleavage rule registry is loaded from `constellation/data/proteases.json`
at import time. Patterns are zero-width regex assertions identical in
shape to pyteomics' `expasy_rules`, but pyteomics is not imported.

`peptide_composition` and `peptide_mass` accept a `Peptidoform`
(``core.sequence.proforma``); bare-sequence callers wrap one explicitly
(``Peptidoform(sequence="PEPTIDE")``). Both return a `core.chem.Composition`
or a float, so the caller gets monoisotopic mass, average mass, isotope
envelopes, and Hill-formula formatting for free.
"""

from __future__ import annotations

import bisect
import itertools
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import regex
import torch

from constellation.core.chem.composition import Composition
from constellation.core.chem.elements import ELEMENTS
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import AA, requires_canonical
from constellation.core.sequence.ops import validate
from constellation.core.sequence.proforma import (
    ModRef,
    MultiPeptidoform,
    Peptidoform,
    TaggedMod,
    parse_isotope_label,
)

# ──────────────────────────────────────────────────────────────────────
# Protease dataclass + registry
# ──────────────────────────────────────────────────────────────────────


SpecificityKind = Literal["high", "low"]


@dataclass(frozen=True, slots=True)
class Protease:
    """A cleavage specification: name + zero-width regex pattern.

    `regex_pattern` is meant for `regex.finditer` against an upper-case
    protein sequence. Zero-width assertions identify the cleavage *site*;
    `cleave()` walks these to enumerate peptides.
    """

    id: str
    name: str
    regex_pattern: str
    specificity: SpecificityKind = "high"
    description: str = ""


@dataclass(slots=True)
class ProteaseRegistry:
    """Indexable registry of `Protease` objects.

    Mirrors `core.chem.modifications.ModVocab` in shape: load/save JSON,
    register custom proteases, look up by id, raise on collisions.
    """

    _by_id: dict[str, Protease] = field(default_factory=dict)
    _patterns: dict[str, regex.Pattern] = field(default_factory=dict)

    def add(self, p: Protease) -> None:
        if p.id in self._by_id:
            raise KeyError(f"protease id already registered: {p.id}")
        self._by_id[p.id] = p
        # Pre-compile the pattern; `regex.compile` raises on syntax errors.
        self._patterns[p.id] = regex.compile(p.regex_pattern)

    def get(self, key: str) -> Protease:
        try:
            return self._by_id[key]
        except KeyError:
            raise KeyError(f"unknown protease: {key!r}") from None

    def __getitem__(self, key: str) -> Protease:
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._by_id

    def __iter__(self) -> Iterator[Protease]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id.keys())

    def pattern(self, key: str) -> regex.Pattern:
        """Return the pre-compiled regex for `key`."""
        if key not in self._by_id:
            raise KeyError(f"unknown protease: {key!r}")
        return self._patterns[key]

    def register_custom(
        self,
        id: str,
        regex_pattern: str,
        *,
        name: str | None = None,
        specificity: SpecificityKind = "high",
        description: str = "",
    ) -> Protease:
        p = Protease(
            id=id,
            name=name or id,
            regex_pattern=regex_pattern,
            specificity=specificity,
            description=description,
        )
        self.add(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ProteaseRegistry":
        with Path(path).open() as f:
            doc = json.load(f)
        return cls._from_doc(doc)

    @classmethod
    def _from_doc(cls, doc: Mapping) -> "ProteaseRegistry":
        out = cls()
        for rec in doc["proteases"]:
            out.add(
                Protease(
                    id=rec["id"],
                    name=rec.get("name", rec["id"]),
                    regex_pattern=rec["regex_pattern"],
                    specificity=rec.get("specificity", "high"),
                    description=rec.get("description", ""),
                )
            )
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "source": {"generator": "ProteaseRegistry.save"},
                    "proteases": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "regex_pattern": p.regex_pattern,
                            "specificity": p.specificity,
                            "description": p.description,
                        }
                        for p in self
                    ],
                },
                indent=2,
            )
            + "\n"
        )


def _load_proteases() -> ProteaseRegistry:
    with resources.files("constellation.data").joinpath("proteases.json").open() as f:
        doc = json.load(f)
    return ProteaseRegistry._from_doc(doc)


PROTEASES: ProteaseRegistry = _load_proteases()


# ──────────────────────────────────────────────────────────────────────
# Peptide record
# ──────────────────────────────────────────────────────────────────────


#: How a span's end was made: by the protease's rule, at one of the
#: protein's own termini (its first or last residue, after an initiator Met,
#: or a declared mature-chain end), or by neither (semi- / unspecific).
TerminusKind = Literal["enzyme", "protein", "nonspecific"]
TERMINUS_KINDS: tuple[str, ...] = ("enzyme", "protein", "nonspecific")

#: Digestion specificity: both ends by the rule, at least one, or neither.
CleavageSpecificity = Literal["specific", "semi", "unspecific"]


@dataclass(frozen=True, slots=True)
class Peptide:
    """A peptide span within a parent protein.

    ``n_missed`` counts the protease's cut sites strictly inside the span.
    ``n_term`` / ``c_term`` say how each end was made (:data:`TerminusKind`);
    ``preceding`` / ``following`` are the residues either side of the span,
    ``"-"`` at a protein end (the flanking notation search engines use).
    """

    sequence: str
    start: int  # 0-indexed inclusive
    end: int  # 0-indexed exclusive
    n_missed: int
    n_term: TerminusKind = "enzyme"
    c_term: TerminusKind = "enzyme"
    preceding: str = ""
    following: str = ""

    @property
    def is_protein_n_terminal(self) -> bool:
        return self.n_term == "protein"

    @property
    def is_protein_c_terminal(self) -> bool:
        return self.c_term == "protein"


# ──────────────────────────────────────────────────────────────────────
# Cleavage
# ──────────────────────────────────────────────────────────────────────


def _resolve_protease(key: str | Protease) -> Protease:
    if isinstance(key, Protease):
        return key
    return PROTEASES[key]


@requires_canonical
def cleave_sites(
    seq: str,
    protease: str | Protease = "Trypsin",
    *,
    validate_alphabet: bool = True,
) -> list[int]:
    """Cut-site indices (ascending). Each index `i` means a cut between
    `seq[i-1]` and `seq[i]` — i.e. the peptide starting at `i` is the
    one immediately after a cut.

    Always includes 0 (sequence start) and `len(seq)` (sequence end);
    these aren't actual cuts but anchor the peptide enumeration.

    ``validate_alphabet=False`` skips the canonical-AA check so sequences
    with non-canonical residues digest normally — the protease regex only
    matches K/R, so cut sites are unaffected by the odd residue.
    """
    if validate_alphabet:
        validate(seq, AA)
    p = _resolve_protease(protease)
    pat = (
        PROTEASES.pattern(p.id)
        if p.id in PROTEASES
        else regex.compile(p.regex_pattern)
    )
    cuts = [m.start() for m in pat.finditer(seq, overlapped=False)]
    # Drop 0 and len(seq) if they show up as match positions — we add
    # them as sentinels separately.
    interior = [c for c in cuts if 0 < c < len(seq)]
    return [0, *interior, len(seq)]


@requires_canonical
def cleave(
    seq: str,
    protease: str | Protease = "Trypsin",
    *,
    missed_cleavages: int = 0,
    min_length: int = 6,
    max_length: int | None = 50,
    semi_specific: bool = False,
    specificity: CleavageSpecificity | None = None,
    excise_initiator_met: bool = False,
    extra_n_termini: Sequence[int] = (),
    extra_c_termini: Sequence[int] = (),
    validate_alphabet: bool = True,
    return_spans: bool = False,
) -> list[str] | list[Peptide]:
    """Enzymatically cleave `seq` into peptides.

    A span's two ends are each one of: a cut site of the protease, one of
    the protein's termini, or (semi- / unspecific digestion) neither.
    ``missed_cleavages`` bounds the cut sites strictly inside a span.

    ``specificity`` (default ``"specific"``):

    - ``"specific"``: both ends are cut sites or protein termini — the
      spans across 0..``missed_cleavages`` consecutive cuts;
    - ``"semi"``: at least one end is, the other anywhere within the
      length bounds (both the N-terminus and the C-terminus may float);
    - ``"unspecific"``: every span within the length bounds, whatever the
      protease (which then only names how ends were made and counts the
      cut sites inside; ``missed_cleavages`` does not apply).
      ``max_length`` is required.

    ``semi_specific=True`` is the old spelling of ``specificity="semi"``.
    Until 2026-11-05 it floated only the N-terminus (suffixes of each
    specific span); it now floats either end.

    Protein termini: position 0 and ``len(seq)``, plus ``extra_n_termini``
    / ``extra_c_termini`` (0-based start / exclusive end positions; a
    mature chain after a signal peptide, say). With
    ``excise_initiator_met=True`` and a sequence starting with M, position
    1 is a protein N-terminus too: every N-terminal span also has its
    Met-clipped form, each under the length filter on its own. The
    clipping is **unconditional** (not gated on the second residue): the
    behaviour verified against EncyclopeDIA 6.5.15's predicted library,
    where ~87% of protein N-termini carry both forms. A span starting at a
    protein N-terminus has ``n_term="protein"`` (which is what licenses a
    Protein N-term modification on it), and likewise at the C-terminus.

    Output ordering: sorted by ``(start, end)``. With ``return_spans=True``
    the records are :class:`Peptide` spans with protein coordinates, one
    per ``(start, end)``; otherwise the sequences, de-duplicated keeping
    the first occurrence (a sequence occurring twice is returned once).

    With `validate_alphabet=False`, the canonical-AA check is skipped so
    sequences carrying non-canonical residues (selenocysteine ``U``,
    pyrrolysine ``O``, ambiguity codes ``B``/``Z``/``X``/``J``) digest
    normally instead of raising. The cut sites only depend on the
    protease's residues, so peptides not containing the odd residue come
    out exactly as they would for a clean sequence — important when
    digesting whole proteomes, where one selenoprotein must not cost you
    all of its (otherwise canonical) tryptic peptides.
    """
    if validate_alphabet:
        validate(seq, AA)
    if missed_cleavages < 0:
        raise ValueError(f"missed_cleavages must be ≥ 0; got {missed_cleavages}")
    if specificity is None:
        specificity = "semi" if semi_specific else "specific"
    elif semi_specific and specificity != "semi":
        raise ValueError(f"semi_specific=True contradicts specificity={specificity!r}")
    if specificity not in ("specific", "semi", "unspecific"):
        raise ValueError(f"unknown specificity {specificity!r}")
    if specificity == "unspecific" and max_length is None:
        raise ValueError("unspecific digestion needs a max_length")
    n = len(seq)
    for pos in (*extra_n_termini, *extra_c_termini):
        if not 0 <= pos <= n:
            raise ValueError(f"protein terminus {pos} outside a length-{n} sequence")

    sites = cleave_sites(seq, protease, validate_alphabet=validate_alphabet)
    cuts = sites[1:-1]  # the protease's own interior cuts
    cut_set = set(cuts)
    n_termini = {0, *extra_n_termini}
    if excise_initiator_met and seq.startswith("M") and n > 1:
        n_termini.add(1)
    c_termini = {n, *extra_c_termini}
    starts = sorted(n_termini | cut_set)  # where a specific span may start
    ends = sorted(c_termini | cut_set)  # and end
    cut_arr = cuts  # ascending

    def inside(start: int, end: int) -> int:
        """Cut sites strictly inside [start, end)."""
        return bisect.bisect_left(cut_arr, end) - bisect.bisect_right(cut_arr, start)

    def kind(pos: int, termini: set[int]) -> str:
        if pos in termini:
            return "protein"
        if pos in cut_set:
            return "enzyme"
        return "nonspecific"

    max_len = n if max_length is None else max_length
    keep: set[tuple[int, int]] = set()
    if specificity == "unspecific":
        for start in range(n):
            for end in range(start + max(min_length, 1), min(n, start + max_len) + 1):
                keep.add((start, end))
    else:
        for start in starts:
            # ends after `start` with at most `missed_cleavages` cuts inside
            for end in ends[bisect.bisect_right(ends, start):]:
                if inside(start, end) > missed_cleavages:
                    break
                if _length_ok_span(end - start, min_length, max_length):
                    keep.add((start, end))
        if specificity == "semi":
            for start in starts:  # specific start, floating end
                for end in range(start + max(min_length, 1), min(n, start + max_len) + 1):
                    if inside(start, end) > missed_cleavages:
                        break
                    keep.add((start, end))
            for end in ends:  # floating start, specific end
                for start in range(end - max(min_length, 1), max(0, end - max_len) - 1, -1):
                    if inside(start, end) > missed_cleavages:
                        break
                    keep.add((start, end))

    spans = [
        Peptide(
            seq[start:end],
            start,
            end,
            n_missed=inside(start, end),
            n_term=kind(start, n_termini),  # type: ignore[arg-type]
            c_term=kind(end, c_termini),  # type: ignore[arg-type]
            preceding=seq[start - 1] if start > 0 else "-",
            following=seq[end] if end < n else "-",
        )
        for start, end in sorted(keep)
    ]
    if return_spans:
        return spans

    # Deduplicate sequences, keep first occurrence under (start, end) order.
    seen: set[str] = set()
    out: list[str] = []
    for sp in spans:
        if sp.sequence in seen:
            continue
        seen.add(sp.sequence)
        out.append(sp.sequence)
    return out


def _length_ok(peptide: str, min_length: int, max_length: int | None) -> bool:
    return _length_ok_span(len(peptide), min_length, max_length)


def _length_ok_span(n: int, min_length: int, max_length: int | None) -> bool:
    if n < min_length:
        return False
    if max_length is not None and n > max_length:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Modification enumeration
# ──────────────────────────────────────────────────────────────────────

N_TERM = "N-term"
C_TERM = "C-term"
#: Site sentinels for a modification UNIMOD places on the protein's own
#: terminus: offered only on a span at that terminus (``SiteContext``).
PROTEIN_N_TERM = "Protein N-term"
PROTEIN_C_TERM = "Protein C-term"
_TERMINAL_SITES = {N_TERM: N_TERM, C_TERM: C_TERM, PROTEIN_N_TERM: N_TERM, PROTEIN_C_TERM: C_TERM}

#: Where a modification of an enumerated form came from.
ModOrigin = Literal["fixed", "variable"]


@dataclass(frozen=True, slots=True)
class SiteContext:
    """Where in its protein the peptide being modified sits: whether its
    N-terminus is the protein's own (position 0, after an initiator Met, or
    a declared mature start) and likewise its C-terminus. This is what lets
    a Protein N-term modification be placed only where it can occur."""

    protein_n_term: bool = False
    protein_c_term: bool = False


@dataclass(frozen=True, slots=True)
class PlacedMod:
    """One modification of an enumerated form, and why it is there.

    ``slot`` is a 0-based residue index, or ``"N-term"`` / ``"C-term"``;
    ``site`` is the key of the ``fixed`` / ``variable`` mapping that placed
    it (a residue letter, ``N-term``, ``Protein N-term``, ...)."""

    slot: int | str
    key: str
    origin: ModOrigin
    site: str


def _normalize_mod_spec(
    spec: Mapping[str, str | Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """Site → tuple-of-mod-keys, accepting a bare string per site."""
    out: dict[str, tuple[str, ...]] = {}
    for site, keys in (spec or {}).items():
        out[site] = (keys,) if isinstance(keys, str) else tuple(keys)
    return out


def _slots_for_site(site: str, seq: str, context: SiteContext | None = None) -> list[int | str]:
    """Positions in `seq` a site specifier applies to.

    Residue sites yield 0-indexed offsets; ``N-term`` / ``C-term`` yield
    themselves as sentinel slots, and ``Protein N-term`` / ``Protein
    C-term`` the same sentinels when ``context`` puts the peptide at that
    terminus of its protein (none otherwise).
    """
    if site in (N_TERM, C_TERM):
        return [site]
    if site == PROTEIN_N_TERM:
        return [N_TERM] if context is not None and context.protein_n_term else []
    if site == PROTEIN_C_TERM:
        return [C_TERM] if context is not None and context.protein_c_term else []
    return [i for i, aa in enumerate(seq) if aa == site]


def _licensed(
    key: str, site: str, slot: int | str, vocab: ModVocab, peptide: str, context: SiteContext | None
) -> bool:
    """Whether UNIMOD (shown or hidden placements) lists ``key`` at ``slot``.
    Without a context the protein terminus is unknown and not held against
    a placement (what ``enumerate_modforms`` always did)."""
    mod = vocab[key]
    if slot == N_TERM:
        protein = context is None or context.protein_n_term
        return bool(peptide) and mod.licenses(N_TERM, terminal_residue=peptide[0], protein_terminus=protein)
    if slot == C_TERM:
        protein = context is None or context.protein_c_term
        return bool(peptide) and mod.licenses(C_TERM, terminal_residue=peptide[-1], protein_terminus=protein)
    return mod.licenses(site)


def _check_specificity(
    key: str, site: str, vocab: ModVocab, peptide: str, context: SiteContext | None = None
) -> None:
    mod = vocab[key]
    slot = _TERMINAL_SITES.get(site, site)
    if slot in (N_TERM, C_TERM):
        if site in (PROTEIN_N_TERM, PROTEIN_C_TERM) and not _slots_for_site(site, peptide, context):
            return  # not at the protein's terminus: nothing to place, nothing to check
        ok = _licensed(key, site, slot, vocab, peptide, context)
        end = peptide[:1] if slot == N_TERM else peptide[-1:]
        word = "starting" if slot == N_TERM else "ending"
        detail = f" for a peptide {word} with {end!r}" if peptide else ""
    else:
        ok = mod.licenses(site)
        detail = ""
    if not ok:
        raise ValueError(
            f"{key} ({mod.name}) has no UNIMOD specificity for site {site!r}"
            f"{detail}; pass validate_specificity=False to apply it anyway"
        )


def _tagged(key: str) -> TaggedMod:
    cv, _, acc = key.partition(":")
    return TaggedMod(mod=ModRef(cv=cv, accession=acc))


def _count_assignments(
    candidates: dict[int | str, list[tuple[str, str]]],
    ordered: list[int | str],
    k_max: int,
    limits: Mapping[str, int],
) -> int:
    """How many variable assignments there are: up to ``k_max`` slots
    occupied, each by one of its candidates, no limited key more than its
    limit. A DP over slots; states are (occupied, counts of limited keys)."""
    limited = sorted(limits)
    states: dict[tuple[int, tuple[int, ...]], int] = {(0, (0,) * len(limited)): 1}
    for slot in ordered:
        nxt: dict[tuple[int, tuple[int, ...]], int] = dict(states)
        for (k, counts), ways in states.items():
            if k == k_max:
                continue
            for key, _site in candidates[slot]:
                c = list(counts)
                if key in limits:
                    j = limited.index(key)
                    if c[j] == limits[key]:
                        continue
                    c[j] += 1
                state = (k + 1, tuple(c))
                nxt[state] = nxt.get(state, 0) + ways
        states = nxt
    return sum(states.values())


def _plan_modforms(
    peptide: str,
    *,
    fixed: Mapping[str, str | Sequence[str]] | None,
    variable: Mapping[str, str | Sequence[str]] | None,
    max_variable: int,
    vocab: ModVocab,
    validate_specificity: bool,
    validate_alphabet: bool,
    context: SiteContext | None,
    max_per_mod: Mapping[str, int] | None,
    skip_unlicensed: bool,
):
    """Validate a modification spec against one peptide and lay out its
    slots: the fixed assignment, each variable slot's candidates, the slot
    order, how many may be occupied, and the per-key limits. Shared by
    :func:`enumerate_modforms` and :func:`count_modforms`."""
    if validate_alphabet:
        validate(peptide, AA)
    if max_variable < 0:
        raise ValueError(f"max_variable must be >= 0, got {max_variable}")

    fixed_spec = _normalize_mod_spec(fixed)
    var_spec = _normalize_mod_spec(variable)
    limits = dict(max_per_mod or {})
    for key, limit in limits.items():
        if limit < 0:
            raise ValueError(f"max_per_mod[{key!r}] must be >= 0, got {limit}")

    for site, keys in (*fixed_spec.items(), *var_spec.items()):
        if site in (PROTEIN_N_TERM, PROTEIN_C_TERM) and context is None:
            raise ValueError(f"site {site!r} needs a context saying where the peptide sits in its protein")
        for key in keys:
            if key not in vocab:
                raise ValueError(f"unknown modification key {key!r} for site {site!r}")
            if validate_specificity and not skip_unlicensed:
                _check_specificity(key, site, vocab, peptide, context)

    for site, keys in fixed_spec.items():
        if len(keys) != 1:
            raise ValueError(
                f"fixed site {site!r} takes exactly one modification, got {list(keys)}"
            )

    def licensed(key: str, site: str, slot: int | str) -> bool:
        return not (validate_specificity and skip_unlicensed) or _licensed(
            key, site, slot, vocab, peptide, context
        )

    # Fixed application — every matching slot, always.
    fixed_slots: dict[int | str, tuple[str, str]] = {}
    for site, keys in fixed_spec.items():
        for slot in _slots_for_site(site, peptide, context):
            if licensed(keys[0], site, slot):
                fixed_slots[slot] = (keys[0], site)

    # Variable candidates — slot → the (mod key, site) pairs that may occupy it.
    candidates: dict[int | str, list[tuple[str, str]]] = {}
    for site, keys in var_spec.items():
        for slot in _slots_for_site(site, peptide, context):
            if slot in fixed_slots:
                continue
            have = candidates.setdefault(slot, [])
            for key in keys:
                if licensed(key, site, slot) and all(key != k for k, _ in have):
                    have.append((key, site))
    candidates = {slot: c for slot, c in candidates.items() if c}

    ordered = sorted(
        candidates, key=lambda s: (isinstance(s, str), s if isinstance(s, int) else str(s))
    )
    k_max = min(max_variable, len(ordered))

    return fixed_slots, candidates, ordered, k_max, limits


def _count_planned(
    candidates: dict[int | str, list[tuple[str, str]]],
    ordered: list[int | str],
    k_max: int,
    limits: Mapping[str, int],
) -> int:
    if limits:
        return _count_assignments(candidates, ordered, k_max, limits)
    return sum(
        math.prod(len(candidates[s]) for s in combo)
        for k in range(k_max + 1)
        for combo in itertools.combinations(ordered, k)
    )


@requires_canonical
def count_modforms(
    peptide: str,
    *,
    fixed: Mapping[str, str | Sequence[str]] | None = None,
    variable: Mapping[str, str | Sequence[str]] | None = None,
    max_variable: int = 3,
    vocab: ModVocab = UNIMOD,
    validate_specificity: bool = True,
    validate_alphabet: bool = True,
    context: SiteContext | None = None,
    max_per_mod: Mapping[str, int] | None = None,
    skip_unlicensed: bool = False,
) -> int:
    """How many forms :func:`enumerate_modforms` would return with these
    arguments, without building them (so a budget can be checked first)."""
    _, candidates, ordered, k_max, limits = _plan_modforms(
        peptide,
        fixed=fixed,
        variable=variable,
        max_variable=max_variable,
        vocab=vocab,
        validate_specificity=validate_specificity,
        validate_alphabet=validate_alphabet,
        context=context,
        max_per_mod=max_per_mod,
        skip_unlicensed=skip_unlicensed,
    )
    return _count_planned(candidates, ordered, k_max, limits)


@requires_canonical
def enumerate_modforms(
    peptide: str,
    *,
    fixed: Mapping[str, str | Sequence[str]] | None = None,
    variable: Mapping[str, str | Sequence[str]] | None = None,
    max_variable: int = 3,
    max_forms: int | None = None,
    vocab: ModVocab = UNIMOD,
    validate_specificity: bool = True,
    validate_alphabet: bool = True,
    context: SiteContext | None = None,
    max_per_mod: Mapping[str, int] | None = None,
    skip_unlicensed: bool = False,
    return_origin: bool = False,
) -> list[Peptidoform] | list[tuple[Peptidoform, tuple[PlacedMod, ...]]]:
    """Enumerate the modified forms of one peptide.

    ``fixed`` mods are applied to **every** matching site and are not
    enumerated. ``variable`` mods are enumerated over all combinations of
    0..``max_variable`` simultaneously-modified sites. Both map a *site*
    — a residue letter, the ``N-term`` / ``C-term`` sentinels (the
    peptide's ends), or ``Protein N-term`` / ``Protein C-term`` (its ends
    only where they are its protein's, per ``context``) — to one mod key
    (``"UNIMOD:35"``) or several (``["UNIMOD:21", "UNIMOD:35"]``), the
    latter meaning "any one of these may occupy this site".

    A site already carrying a fixed mod is not offered to the variable
    enumeration, so ``fixed={"C": "UNIMOD:4"}`` and
    ``variable={"C": ...}`` do not stack on the same cysteine.
    ``max_per_mod`` limits how many sites one variable key occupies in a
    form (``{"UNIMOD:21": 2}``), within the overall ``max_variable``.

    Returns forms ordered by increasing variable-mod count, the
    fixed-only form first. Mod keys are validated against ``vocab``, and
    against each mod's UNIMOD placements (shown and hidden) unless
    ``validate_specificity=False`` — the escape hatch for deliberately
    non-canonical placements. With ``skip_unlicensed=True`` a placement
    UNIMOD does not list is left out instead of raising, which is what
    applying one rule set across many peptides wants (pyro-Glu at the
    N-terminus where it starts with Q, and nowhere else). ``context``
    says where the peptide sits in its protein; without one, protein
    termini are neither offered (``Protein N-term`` sites raise) nor
    held against a placement.

    With ``return_origin=True`` each form comes with the
    :class:`PlacedMod` of every modification it carries, fixed and
    variable, in slot order.

    Raises ``ValueError`` if the enumeration would exceed ``max_forms``
    (when set) rather than silently truncating, since a truncated set
    reads downstream as a complete one.
    """
    fixed_slots, candidates, ordered, k_max, limits = _plan_modforms(
        peptide,
        fixed=fixed,
        variable=variable,
        max_variable=max_variable,
        vocab=vocab,
        validate_specificity=validate_specificity,
        validate_alphabet=validate_alphabet,
        context=context,
        max_per_mod=max_per_mod,
        skip_unlicensed=skip_unlicensed,
    )
    if max_forms is not None:
        total = _count_planned(candidates, ordered, k_max, limits)
        if total > max_forms:
            raise ValueError(
                f"{peptide} would yield {total} modforms, exceeding max_forms="
                f"{max_forms}; lower max_variable or narrow the variable mods"
            )

    tags: dict[str, TaggedMod] = {}

    def tag(key: str) -> TaggedMod:
        t = tags.get(key)
        if t is None:
            t = tags[key] = _tagged(key)
        return t

    forms: list = []
    for k in range(k_max + 1):
        for combo in itertools.combinations(ordered, k):
            for choice in itertools.product(*(candidates[s] for s in combo)):
                if limits:
                    keys_chosen = [key for key, _ in choice]
                    if any(keys_chosen.count(key) > limit for key, limit in limits.items()):
                        continue
                assigned: dict[int | str, tuple[str, str, ModOrigin]] = {
                    slot: (key, site, "fixed") for slot, (key, site) in fixed_slots.items()
                }
                for slot, (key, site) in zip(combo, choice, strict=True):
                    assigned[slot] = (key, site, "variable")
                residue_mods = {
                    slot: (tag(key),) for slot, (key, _, _) in assigned.items() if isinstance(slot, int)
                }
                form = Peptidoform(
                    sequence=peptide,
                    residue_mods=residue_mods,
                    n_term_mods=(tag(assigned[N_TERM][0]),) if N_TERM in assigned else (),
                    c_term_mods=(tag(assigned[C_TERM][0]),) if C_TERM in assigned else (),
                )
                if return_origin:
                    placed = tuple(
                        PlacedMod(slot, key, origin, site)
                        for slot, (key, site, origin) in sorted(
                            assigned.items(),
                            key=lambda kv: (-1, "") if kv[0] == N_TERM
                            else (len(peptide), "") if kv[0] == C_TERM
                            else (kv[0], ""),
                        )
                    )
                    forms.append((form, placed))
                else:
                    forms.append(form)
    return forms


# ──────────────────────────────────────────────────────────────────────
# Peptide composition + mass
# ──────────────────────────────────────────────────────────────────────


_H2O: Composition = Composition.from_dict({"H": 2, "O": 1})

# CV-shorthand aliases per ProForma §4.2 — `U:35` is shorthand for
# `UNIMOD:35`. Used to resolve a ModRef into a vocab-lookup key.
_CV_ALIASES = {"U": "UNIMOD", "M": "MOD", "R": "RESID", "X": "XLMOD", "G": "GNO"}


def _modref_lookup_key(modref: ModRef) -> str:
    """Build a vocab-lookup key from a CV/name ModRef."""
    if modref.accession is not None:
        cv = modref.cv or "UNIMOD"  # bare accession assumed UNIMOD per spec
        cv = _CV_ALIASES.get(cv, cv)
        return f"{cv}:{modref.accession}"
    if modref.name is not None:
        return modref.name
    raise ValueError(f"ModRef has neither accession nor name: {modref!r}")


def _is_bare_mass_delta(modref: ModRef) -> bool:
    """True for ModRefs that carry only a mass delta (no CV / name /
    formula / glycan / INFO payload) — these can't be resolved to a
    composition and only contribute mass."""
    return (
        modref.mass_delta is not None
        and modref.accession is None
        and modref.name is None
        and modref.formula is None
        and modref.glycan is None
        and modref.cv != "INFO"
    )


def _modref_contribution(
    modref: ModRef,
    vocab: ModVocab,
    *,
    monoisotopic: bool,
) -> tuple[Composition, float]:
    """Resolve a ModRef into (composition_delta, extra_mass_from_override).

    - CV/name refs → vocab lookup; composition is the light-atom skeleton
      and any ``mass_override`` correction lands in ``extra_mass``.
    - Formula refs parse into Composition directly.
    - Bare mass-delta refs contribute mass only (zero composition delta).
    - INFO refs contribute nothing.
    - Glycan refs raise NotImplementedError.
    """
    if modref.cv == "INFO":
        return Composition.zeros(), 0.0
    if modref.formula is not None:
        return Composition.from_formula(modref.formula), 0.0
    if modref.accession is not None or modref.name is not None:
        mod = vocab.get(_modref_lookup_key(modref))
        extra = 0.0
        if mod.mass_override is not None and monoisotopic:
            extra = mod.mass_override - mod.delta_composition.mass
        return mod.delta_composition, extra
    if modref.glycan is not None:
        raise NotImplementedError(
            f"glycan-payload mods ({modref.glycan!r}) are not yet resolved "
            "to compositions; glycan dictionary support lands when a "
            "concrete consumer arrives"
        )
    if modref.mass_delta is not None:
        return Composition.zeros(), modref.mass_delta
    raise ValueError(f"empty ModRef: {modref!r}")


def _walk_modrefs(p: Peptidoform) -> Iterator[ModRef]:
    """Yield each ModRef contributing to composition/mass exactly once.

    Deduplicates by ``group_id`` — XL / ambiguity / branch groups share
    one chemical entity across all anchor positions, so the contribution
    counts once. Bare label references (``mod=None``) are skipped.

    Excludes ``fixed_mods`` because their location-rule expansion depends
    on per-residue overrides (handled separately at the call site).
    """
    seen: set[str] = set()

    def _maybe_yield(tm: TaggedMod) -> Iterator[ModRef]:
        if tm.mod is None:
            return
        if tm.group_id is not None:
            if tm.group_id in seen:
                return
            seen.add(tm.group_id)
        yield tm.mod

    for tms in p.residue_mods.values():
        for tm in tms:
            yield from _maybe_yield(tm)
    for tm in p.n_term_mods:
        yield from _maybe_yield(tm)
    for tm in p.c_term_mods:
        yield from _maybe_yield(tm)
    for mref in p.labile_mods:
        yield mref
    for mref, count in p.unknown_pos_mods:
        for _ in range(count):
            yield mref
    for rng in p.ranges:
        yield from rng.mods


def _count_fixed_mod_locations(seq: str, loc: str, p: Peptidoform) -> int:
    """Number of residues matching a fixed-mod location rule, skipping
    positions already carrying an explicit mod (manual override wins per
    ProForma §4.6.2).

    Recognized location rules:

        ``"*"``     → every residue (subject to override-skip)
        single AA   → every residue of that letter
        ``"N-term"``/``"C-term"`` → terminal slot if not occupied
    """
    if loc == "*":
        return sum(1 for i in range(len(seq)) if i not in p.residue_mods)
    if loc == "N-term":
        return 0 if p.n_term_mods else 1
    if loc == "C-term":
        return 0 if p.c_term_mods else 1
    if len(loc) == 1:
        return sum(
            1 for i, r in enumerate(seq) if r == loc and i not in p.residue_mods
        )
    raise ValueError(f"unsupported fixed-mod location: {loc!r}")


def _residue_index_check(p: Peptidoform) -> None:
    n = len(p.sequence)
    for pos in p.residue_mods:
        if not 0 <= pos < n:
            raise IndexError(
                f"modification index {pos} out of range for length-{n} peptide"
            )


# ── batched composition and mass ──────────────────────────────────────
#
# One implementation, batched: the single-peptidoform functions below are
# its one-item case. A residue's composition is counted, not added one
# residue at a time (the old loop did one torch add per residue, ~100 us a
# peptide), and each distinct ModRef is resolved once per call. The mass is
# ``batched_mass`` of the composition plus the Python-float corrections in
# the order the scalar code made them, so the result is the same bits.

_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_RESIDUE_INDEX = np.full(256, -1, dtype=np.int64)
for _i, _aa in enumerate(_AA_ORDER):
    _RESIDUE_INDEX[ord(_aa)] = _i
del _i, _aa
_RESIDUE_COMPOSITIONS: np.ndarray | None = None
_H2O_COUNTS: np.ndarray | None = None


def _residue_matrix() -> tuple[np.ndarray, np.ndarray]:
    """(20, E) int64 compositions of the canonical residues and H2O's (E,)
    counts, built once."""
    global _RESIDUE_COMPOSITIONS, _H2O_COUNTS
    if _RESIDUE_COMPOSITIONS is None:
        assert AA.compositions is not None  # canonical alphabet always carries them
        _RESIDUE_COMPOSITIONS = np.stack(
            [AA.compositions[aa].counts.numpy().astype(np.int64) for aa in _AA_ORDER]
        )
        _H2O_COUNTS = _H2O.counts.numpy().astype(np.int64)
    assert _H2O_COUNTS is not None
    return _RESIDUE_COMPOSITIONS, _H2O_COUNTS


def _check_peptidoform(peptidoform: Peptidoform, what: str) -> None:
    if isinstance(peptidoform, MultiPeptidoform):
        plural = "compositions" if what == "composition" else "masses"
        raise NotImplementedError(
            f"joint multi-chain {what} is not yet supported; iterate "
            f"peptidoform.chains for per-chain {plural}"
        )


def _batched(
    peptidoforms: Sequence[Peptidoform],
    vocab: ModVocab,
    *,
    monoisotopic: bool,
    for_mass: bool,
) -> tuple[np.ndarray, list[float]]:
    """(N, E) int64 compositions and, per peptidoform, the mass the
    composition does not carry (``mass_override`` corrections and bare mass
    deltas), accumulated in the scalar code's order."""
    what = "mass" if for_mass else "composition"
    n = len(peptidoforms)
    seqs: list[str] = []
    for pf in peptidoforms:
        _check_peptidoform(pf, what)
        seqs.append(pf.sequence)
    residue_comp, h2o = _residue_matrix()
    joined = "".join(seqs)
    residues = _RESIDUE_INDEX[np.frombuffer(joined.encode("latin-1", "replace"), dtype=np.uint8)]
    if (residues < 0).any():
        lengths = np.fromiter((len(q) for q in seqs), dtype=np.int64, count=n)
        first = int(np.flatnonzero(residues < 0)[0])
        validate(seqs[int(np.searchsorted(np.cumsum(lengths), first, side="right"))], AA)  # raises
    if n == 1:
        counts = np.bincount(residues, minlength=20).reshape(1, 20)
    else:
        lengths = np.fromiter((len(q) for q in seqs), dtype=np.int64, count=n)
        form = np.repeat(np.arange(n, dtype=np.int64), lengths)
        counts = np.bincount(form * 20 + residues, minlength=n * 20).reshape(n, 20)
    comp = counts @ residue_comp
    comp += h2o

    resolved: dict[ModRef, tuple[np.ndarray, float]] = {}
    extra = [0.0] * n

    def resolve(modref: ModRef) -> tuple[np.ndarray, float]:
        hit = resolved.get(modref)
        if hit is None:
            d_comp, d_mass = _modref_contribution(modref, vocab, monoisotopic=monoisotopic)
            hit = resolved[modref] = (d_comp.counts.numpy(), d_mass)  # int32 view
        return hit

    for i, pf in enumerate(peptidoforms):
        _residue_index_check(pf)
        if not (pf.residue_mods or pf.n_term_mods or pf.c_term_mods or pf.labile_mods
                or pf.unknown_pos_mods or pf.ranges or pf.fixed_mods):
            continue
        row = comp[i]
        acc = 0.0
        for modref in _walk_modrefs(pf):
            if not for_mass and _is_bare_mass_delta(modref):
                raise ValueError(
                    f"cannot derive composition from bare mass-delta ModRef "
                    f"({modref.mass_delta:+f}) — peptide_mass accepts mass-only "
                    "mods, but peptide_composition needs a UNIMOD/name/formula "
                    "payload to determine element counts"
                )
            d_comp, d_mass = resolve(modref)
            row += d_comp
            acc += d_mass
        for modref, locations in pf.fixed_mods:
            d_comp, d_mass = resolve(modref)
            for loc in locations:
                count = _count_fixed_mod_locations(pf.sequence, loc, pf)
                if count:
                    row += d_comp * count
                    acc += d_mass * count
        extra[i] = acc
    return comp, extra


@requires_canonical
def peptide_compositions(
    peptidoforms: Sequence[Peptidoform],
    *,
    vocab: ModVocab = UNIMOD,
) -> torch.Tensor:
    """The compositions of many peptidoforms at once: an ``(N, E)`` int32
    tensor, row ``i`` being ``peptide_composition(peptidoforms[i]).counts``
    (same rules, same errors)."""
    comp, _ = _batched(peptidoforms, vocab, monoisotopic=True, for_mass=False)
    return torch.from_numpy(comp.astype(np.int32))


def _masses(peptidoforms: Sequence[Peptidoform], vocab: ModVocab, monoisotopic: bool) -> np.ndarray:
    comp, extra = _batched(peptidoforms, vocab, monoisotopic=monoisotopic, for_mass=True)
    if monoisotopic:
        # The reduction Composition.mass and batched_mass make (counts as
        # float64 times the element masses, summed by torch), so the same bits.
        base = (torch.from_numpy(comp.astype(np.float64)) * ELEMENTS.mass_tensor).sum(dim=-1).numpy()
    else:
        # Composition.average_mass sums element by element in Python; kept
        # as it is so an average mass is the same bits too (a rare path).
        rows = torch.from_numpy(comp.astype(np.int32))
        base = np.array([Composition(row).average_mass for row in rows], dtype=np.float64)
    for i, pf in enumerate(peptidoforms):
        if not pf.global_isotopes:
            continue
        if not monoisotopic:
            raise ValueError(
                "global isotope labels are only meaningful in monoisotopic "
                "mode — pass monoisotopic=True or strip global_isotopes"
            )
        acc = extra[i]
        for label in pf.global_isotopes:
            mass_n, sym = parse_isotope_label(label)
            atom_count = int(comp[i, ELEMENTS.index(sym)])
            if atom_count == 0:
                continue
            elem = ELEMENTS[sym]
            acc += (elem.isotope_mass(mass_n) - elem.monoisotopic_mass) * atom_count
        extra[i] = acc
    return base + np.asarray(extra, dtype=np.float64)


@requires_canonical
def peptide_masses(
    peptidoforms: Sequence[Peptidoform],
    *,
    monoisotopic: bool = True,
    vocab: ModVocab = UNIMOD,
) -> torch.Tensor:
    """The masses of many peptidoforms at once: an ``(N,)`` float64 tensor,
    element ``i`` being ``peptide_mass(peptidoforms[i])`` to the bit (same
    rules, same errors)."""
    return torch.from_numpy(_masses(peptidoforms, vocab, monoisotopic))


@requires_canonical
def peptide_composition(
    peptidoform: Peptidoform,
    *,
    vocab: ModVocab = UNIMOD,
) -> Composition:
    """Sum residue compositions + terminal H₂O + every modification's
    composition delta into a single ``Composition``.

    Modification handling:

    - Residue mods, terminal mods, labile mods, unknown-position mods
      (multiplied by their count), range mods, and fixed mods all
      contribute their ``delta_composition``.
    - Cross-link / ambiguity / branch groups (TaggedMod ``group_id``)
      contribute exactly once across all anchor positions — the chemistry
      is shared between participants.
    - Heavy-isotope modifications (``mass_override``) contribute the
      *light-atom skeleton*; ``peptide_mass`` resolves the canonical mass.
    - Global isotope labels (``<13C>``, ``<15N>``, ...) contribute zero
      composition; they shift mass only and are handled in ``peptide_mass``.

    Raises:

    - ``KeyError`` for residues outside canonical AA or unknown vocab keys.
    - ``ValueError`` for ModRefs that carry only a bare mass delta (no
      composition info available — call ``peptide_mass`` instead).
    - ``IndexError`` for residue-mod positions outside the sequence.
    - ``NotImplementedError`` for ``MultiPeptidoform`` inputs (iterate
      ``.chains`` for per-chain compositions) and for glycan-payload mods.

    The one-item case of :func:`peptide_compositions`.
    """
    comp, _ = _batched([peptidoform], vocab, monoisotopic=True, for_mass=False)
    return Composition(torch.from_numpy(comp[0].astype(np.int32)))


@requires_canonical
def peptide_mass(
    peptidoform: Peptidoform,
    *,
    monoisotopic: bool = True,
    vocab: ModVocab = UNIMOD,
) -> float:
    """Monoisotopic (default) or standard-weight mass of a peptidoform.

    Aggregates: residue + terminal-water masses + every modification's
    canonical mass contribution + global-isotope label corrections.

    Heavy-isotope modifications (``mass_override``) and global isotope
    labels (``<13C>``) only adjust the result in ``monoisotopic=True``
    mode — average-mass calculations operate on standard atomic weights,
    where these distinctions are not meaningful.

    The one-item case of :func:`peptide_masses`; for many peptidoforms call
    that, about ten times faster per peptidoform.
    """
    return float(_masses([peptidoform], vocab, monoisotopic)[0])


@requires_canonical
def protein_composition(seq: str) -> Composition:
    """Unmodified peptide composition — equivalent to
    ``peptide_composition(Peptidoform(sequence=seq))`` but documents
    intent for whole-protein callers."""
    return peptide_composition(Peptidoform(sequence=seq))


__all__ = [
    "Protease",
    "ProteaseRegistry",
    "PROTEASES",
    "Peptide",
    "cleave",
    "cleave_sites",
    "enumerate_modforms",
    "count_modforms",
    "N_TERM",
    "C_TERM",
    "PROTEIN_N_TERM",
    "PROTEIN_C_TERM",
    "PlacedMod",
    "SiteContext",
    "TERMINUS_KINDS",
    "peptide_composition",
    "peptide_compositions",
    "peptide_mass",
    "peptide_masses",
    "protein_composition",
]
