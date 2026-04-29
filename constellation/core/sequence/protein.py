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

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

import regex

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
    """
    if isinstance(peptidoform, MultiPeptidoform):
        raise NotImplementedError(
            "joint multi-chain composition is not yet supported; iterate "
            "peptidoform.chains for per-chain compositions"
        )

    seq = peptidoform.sequence
    validate(seq, AA)
    assert AA.compositions is not None  # canonical alphabet always carries them
    _residue_index_check(peptidoform)

    comp = Composition.zeros()
    for residue in seq:
        comp = comp + AA.compositions[residue]
    comp = comp + _H2O

    for modref in _walk_modrefs(peptidoform):
        if _is_bare_mass_delta(modref):
            raise ValueError(
                f"cannot derive composition from bare mass-delta ModRef "
                f"({modref.mass_delta:+f}) — peptide_mass accepts mass-only "
                "mods, but peptide_composition needs a UNIMOD/name/formula "
                "payload to determine element counts"
            )
        d_comp, _ = _modref_contribution(modref, vocab, monoisotopic=True)
        comp = comp + d_comp

    for modref, locations in peptidoform.fixed_mods:
        d_comp, _ = _modref_contribution(modref, vocab, monoisotopic=True)
        for loc in locations:
            count = _count_fixed_mod_locations(seq, loc, peptidoform)
            if count:
                comp = comp + d_comp * count

    return comp


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
    """
    if isinstance(peptidoform, MultiPeptidoform):
        raise NotImplementedError(
            "joint multi-chain mass is not yet supported; iterate "
            "peptidoform.chains for per-chain masses"
        )

    seq = peptidoform.sequence
    validate(seq, AA)
    assert AA.compositions is not None
    _residue_index_check(peptidoform)

    comp = Composition.zeros()
    for residue in seq:
        comp = comp + AA.compositions[residue]
    comp = comp + _H2O

    extra_mass = 0.0
    for modref in _walk_modrefs(peptidoform):
        d_comp, d_mass = _modref_contribution(modref, vocab, monoisotopic=monoisotopic)
        comp = comp + d_comp
        extra_mass += d_mass

    for modref, locations in peptidoform.fixed_mods:
        d_comp, d_mass = _modref_contribution(modref, vocab, monoisotopic=monoisotopic)
        for loc in locations:
            count = _count_fixed_mod_locations(seq, loc, peptidoform)
            if count:
                comp = comp + d_comp * count
                extra_mass += d_mass * count

    base = comp.mass if monoisotopic else comp.average_mass

    if peptidoform.global_isotopes:
        if not monoisotopic:
            raise ValueError(
                "global isotope labels are only meaningful in monoisotopic "
                "mode — pass monoisotopic=True or strip global_isotopes"
            )
        atoms = comp.atoms
        for label in peptidoform.global_isotopes:
            mass_n, sym = parse_isotope_label(label)
            atom_count = atoms.get(sym, 0)
            if atom_count == 0:
                continue
            elem = ELEMENTS[sym]
            extra_mass += (elem.isotope_mass(mass_n) - elem.monoisotopic_mass) * atom_count

    return base + extra_mass


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
    "peptide_composition",
    "peptide_mass",
    "protein_composition",
]
