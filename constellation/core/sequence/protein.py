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

`peptide_composition` returns a `core.chem.Composition`, so the caller
gets monoisotopic mass, average mass, isotope envelopes, and Hill-formula
formatting for free.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

import regex

from constellation.core.chem.composition import Composition
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import AA, requires_canonical
from constellation.core.sequence.ops import validate

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


@dataclass(frozen=True, slots=True)
class Peptide:
    """A peptide span within a parent protein."""

    sequence: str
    start: int  # 0-indexed inclusive
    end: int  # 0-indexed exclusive
    n_missed: int


# ──────────────────────────────────────────────────────────────────────
# Cleavage
# ──────────────────────────────────────────────────────────────────────


def _resolve_protease(key: str | Protease) -> Protease:
    if isinstance(key, Protease):
        return key
    return PROTEASES[key]


@requires_canonical
def cleave_sites(seq: str, protease: str | Protease = "Trypsin") -> list[int]:
    """Cut-site indices (ascending). Each index `i` means a cut between
    `seq[i-1]` and `seq[i]` — i.e. the peptide starting at `i` is the
    one immediately after a cut.

    Always includes 0 (sequence start) and `len(seq)` (sequence end);
    these aren't actual cuts but anchor the peptide enumeration.
    """
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
    return_spans: bool = False,
) -> list[str] | list[Peptide]:
    """Enzymatically cleave `seq` into peptides.

    Iterates the protease's cut sites, then enumerates spans across
    0..missed_cleavages consecutive cuts. With `semi_specific=True`,
    additionally emits all suffixes of each fully-specific peptide
    (the C-terminal cut is preserved; the N-terminus floats) — the
    standard "semi-tryptic" search-engine option.

    Output ordering: peptides are sorted by `(start ASC, end ASC)`,
    so along the protein you read the 0-missed peptide first, then
    the 1-missed, then 2-missed at each starting cut. Pyteomics
    returns a `set` (random); we explicitly sort.

    Duplicate sequences (same residue content, different positions)
    are de-duplicated by keeping the first occurrence under the
    `(start, end)` ordering.

    With `return_spans=True`, returns `list[Peptide]` with the
    parent-protein coordinates intact (no de-duplication on sequence).
    """
    validate(seq, AA)
    if missed_cleavages < 0:
        raise ValueError(f"missed_cleavages must be ≥ 0; got {missed_cleavages}")

    sites = cleave_sites(seq, protease)
    # `sites` includes 0 and len(seq); spans are between consecutive
    # cuts plus extensions across `missed_cleavages` interior cuts.
    spans: list[Peptide] = []
    n_sites = len(sites)
    for i in range(n_sites - 1):
        for k in range(missed_cleavages + 1):
            j = i + 1 + k
            if j >= n_sites:
                break
            start = sites[i]
            end = sites[j]
            peptide = seq[start:end]
            if not _length_ok(peptide, min_length, max_length):
                continue
            spans.append(Peptide(peptide, start, end, n_missed=k))

    if semi_specific:
        # Add all suffixes of each span — N-terminus floats, C-terminus
        # is a real cut. Implemented as: for each span (s, e), add
        # (s+1..e), (s+2..e), ... down to min_length.
        extras: list[Peptide] = []
        for sp in spans:
            for new_start in range(sp.start + 1, sp.end):
                sub = seq[new_start : sp.end]
                if not _length_ok(sub, min_length, max_length):
                    continue
                extras.append(Peptide(sub, new_start, sp.end, n_missed=sp.n_missed))
        spans.extend(extras)

    spans.sort(key=lambda p: (p.start, p.end))

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
    n = len(peptide)
    if n < min_length:
        return False
    if max_length is not None and n > max_length:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Peptide composition + mass
# ──────────────────────────────────────────────────────────────────────


_H2O: Composition = Composition.from_dict({"H": 2, "O": 1})


@requires_canonical
def peptide_composition(
    seq: str,
    *,
    modifications: Mapping[int, str] | None = None,
    vocab: ModVocab = UNIMOD,
) -> Composition:
    """Sum residue compositions + terminal H₂O (+ modification deltas).

    `modifications` maps residue index → modification key (UNIMOD id or
    alias). Each key is looked up in `vocab` (default: full UNIMOD)
    and its `delta_composition` is added to the running sum. Heavy-isotope
    mods (those with `mass_override` set) contribute the **light-atom
    skeleton** here — `peptide_mass` resolves the canonical mass.

    Raises KeyError for residues outside the canonical AA alphabet or
    modification ids not in `vocab`.
    """
    validate(seq, AA)
    assert AA.compositions is not None  # canonical alphabet always carries them
    comp = Composition.zeros()
    for residue in seq:
        comp = comp + AA.compositions[residue]
    comp = comp + _H2O
    if modifications:
        for idx, mod_key in modifications.items():
            if not 0 <= idx < len(seq):
                raise IndexError(
                    f"modification index {idx} out of range for length-{len(seq)} peptide"
                )
            mod = vocab.get(mod_key)
            comp = comp + mod.delta_composition
    return comp


@requires_canonical
def peptide_mass(
    seq: str,
    *,
    modifications: Mapping[int, str] | None = None,
    monoisotopic: bool = True,
    vocab: ModVocab = UNIMOD,
) -> float:
    """Monoisotopic (default) or standard-weight mass of a peptide.

    For heavy-isotope modifications the light-skeleton `delta_composition`
    underestimates the canonical mass; this function adds the difference
    (`mod.mass_override - mod.delta_composition.mass`) on top so the
    result matches the conventional UNIMOD value.
    """
    comp = peptide_composition(seq, modifications=modifications, vocab=vocab)
    base = comp.mass if monoisotopic else comp.average_mass
    if not modifications:
        return base
    # Apply heavy-isotope corrections for any mod with a mass_override.
    delta = 0.0
    for mod_key in modifications.values():
        mod = vocab.get(mod_key)
        if mod.mass_override is not None and monoisotopic:
            delta += mod.mass_override - mod.delta_composition.mass
    return base + delta


@requires_canonical
def protein_composition(seq: str) -> Composition:
    """Unmodified peptide composition — equivalent to
    `peptide_composition(seq)` but documents intent for whole-protein
    callers."""
    return peptide_composition(seq)


__all__ = [
    "Protease",
    "ProteaseRegistry",
    "PROTEASES",
    "Peptide",
    "cleave",
    "cleave_sites",
    "peptide_composition",
    "peptide_mass",
    "protein_composition",
]
