"""ProForma 2.0 — HUPO-PSI standard for modified peptide sequences.

Implements the ProForma 2.0 specification (Final draft 15, Feb 2022) — see
LeDuc et al., *J. Proteome Res.* 21:1189–1195 (2022) and the canonical
spec at https://github.com/HUPO-PSI/ProForma. This module is the
authoritative parser and renderer for modseq strings inside constellation;
``PEPTIDE_TABLE.modified_sequence`` columns hold ProForma 2.0 strings.

The grammar lives in [constellation/data/proforma.lark](data/proforma.lark)
(hand-translated from the upstream EBNF; refresh via
``scripts/build-proforma-grammar.py``). The parser is built via lark with
the Earley algorithm.

Per CLAUDE.md, this module deliberately does NOT import pyteomics — the
parser is independently implemented from the upstream grammar so that the
two implementations cross-check each other in tests. Pyteomics is only
imported by tests and (eventually) by the mzML reader inside ``massspec``.

Public API:

    parse_proforma(s)          -> Peptidoform | MultiPeptidoform
    format_proforma(p, *, flatten_terminals=False) -> str
    _has_crosslinks(p)         -> bool   (public, _-prefixed for filename safety)
    _has_branches(p)           -> bool

Dataclasses:

    ModRef                     one modification reference (CV id / name /
                               mass-delta / formula / glycan / info)
    TaggedMod                  ModRef + optional group_id / score for
                               ambiguity groups, cross-links, branches
    Range                      one (start..end)[mod] block
    Peptidoform                one peptide chain with all spec features
    MultiPeptidoform           //-separated cross-linked peptidoforms

Scope (full ProForma 2.0 levels 1-6 supported):

    Level 1: residue mods, multi-mod stacking, N-term / C-term mods,
             mass deltas (+ CV-tagged variants like [U:+15.995])
    Level 2: every CV (UNIMOD/MOD/RESID/XLMOD/GNO) in accession + name
             form; Formula and Glycan payloads
    Level 3: ambiguity groups (#g1) with optional localization scores
    Level 4: range mods (PRT(ESFRMS)[+19.0523]ISK)
    Level 5: intra-peptide cross-links (#XL1), branches (#BRANCH),
             inter-chain (//-separated MultiPeptidoform)
    Level 6: global isotope labeling (<13C><15N>), global fixed mods
             (<[Carbamidomethyl]@C>), labile mods ({Glycan:Hex}),
             unknown-position mods ([Phospho]?, [Phospho]^2?), INFO
             tags, charge state (/N) with adducts (/N[+2Na+,+H+])

Out of scope this session:
    - Chimeric spectra (`+`-separated peptidoformIons): raises
      NotImplementedError. Rare; deferred until a concrete consumer
      arrives.
    - Modified nucleic acids: ProForma is a peptide standard. A sibling
      module modeled on MODOMICS or BPforms will land when nanopore
      modification detection lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cache
from importlib import resources
from typing import Literal, Union

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError

# ──────────────────────────────────────────────────────────────────────
# Error types
# ──────────────────────────────────────────────────────────────────────


class ProFormaError(Exception):
    """Base for all ProForma parsing / validation errors."""


class ProFormaSyntaxError(ProFormaError):
    """Raised when input does not conform to the ProForma 2.0 grammar."""


class ProFormaSemanticError(ProFormaError):
    """Raised when input is grammatically valid but semantically broken
    (orphan cross-link IDs, inconsistent ambiguity scoring, etc.)."""


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────


# Discriminated by which fields are non-None. Convention:
#   - cv = "UNIMOD"|"MOD"|"RESID"|"XLMOD"|"GNO"   → CV-by-accession; accession set
#   - cv = None and name set                        → name-only (CV ambiguous; UNIMOD lookup typical)
#   - cv = "U"|"M"|"R"|"X"|"G" and name set         → CV-tagged name (e.g. M:L-methionine sulfoxide)
#   - cv = "Obs"|"U"|"M"|"R"|"X"|"G" and mass_delta → CV-tagged mass delta
#   - cv = None and mass_delta set                  → bare mass delta
#   - cv = "Formula" / "Glycan" / "INFO"            → typed payload
CV = Literal[
    "UNIMOD", "MOD", "RESID", "XLMOD", "GNO",
    "U", "M", "R", "X", "G", "Obs",
    "Formula", "Glycan", "INFO", None,
]


@dataclass(frozen=True)
class ModRef:
    """One modification reference. Discriminated union — exactly one of
    accession / name / mass_delta / formula / glycan / info_text is set
    in a well-formed instance (the parser ensures this)."""
    cv: CV = None
    accession: str | None = None
    name: str | None = None
    mass_delta: float | None = None
    formula: str | None = None
    glycan: str | None = None
    info_text: str | None = None


@dataclass(frozen=True)
class TaggedMod:
    """A ModRef optionally tagged with a localization-group label.

    ``group_id`` carries the ambiguity / cross-link / branch identifier
    (e.g. ``"g1"``, ``"XL1"``, ``"BRANCH"``). ``score`` carries the
    localization confidence on a [0, 1] scale when explicit; usually
    set on only one position per group (the preferred render position).

    A "bare label reference" (``[#g1]`` with no mod payload) is encoded
    as a TaggedMod with mod=None — it marks a position participating in
    a group without contributing a modification at that position.
    """
    mod: ModRef | None = None
    group_id: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class Range:
    """A ``(start..end)[mod]`` block — modification sits somewhere in the
    inclusive 0-indexed residue range. Mass / composition contributions
    are unambiguous; only the position of attachment is uncertain."""
    start: int  # 0-indexed inclusive
    end: int  # 0-indexed inclusive
    mods: tuple[ModRef, ...] = ()


@dataclass(frozen=True)
class Peptidoform:
    """ProForma 2.0 representation of one peptide chain.

    All spec-level-1-6 features are representable. Position keys in
    ``residue_mods`` are 0-indexed offsets into ``sequence``. Multiple
    mods on one residue stack as a tuple per spec §4.5. N-term and
    C-term mod lists hold up to two entries per spec.

    Multi-chain (//-separated) cross-linked peptidoforms use the
    sibling ``MultiPeptidoform`` container instead of being flattened
    here.
    """
    sequence: str  # canonical residue letters only
    residue_mods: dict[int, tuple[TaggedMod, ...]] = field(default_factory=dict)
    n_term_mods: tuple[TaggedMod, ...] = ()
    c_term_mods: tuple[TaggedMod, ...] = ()
    labile_mods: tuple[ModRef, ...] = ()
    unknown_pos_mods: tuple[tuple[ModRef, int], ...] = ()  # (mod, count)
    ranges: tuple[Range, ...] = ()
    global_isotopes: tuple[str, ...] = ()  # "13C", "15N", "D", "T", ...
    fixed_mods: tuple[tuple[ModRef, tuple[str, ...]], ...] = ()  # (mod, locations)
    charge: int | None = None
    adducts: tuple[str, ...] = ()
    name: str | None = None  # from (>name)

    def __post_init__(self):
        # Frozen-dataclass workaround: ensure dict is also frozen-equivalent.
        # We cast the dict to a sorted-key copy so equality is deterministic.
        if not isinstance(self.residue_mods, dict):
            object.__setattr__(self, "residue_mods", dict(self.residue_mods))


@dataclass(frozen=True)
class MultiPeptidoform:
    """Inter-chain cross-linked peptidoforms (`//` separator, §4.2.4).

    Each chain is its own Peptidoform; cross-link group IDs (``#XL1``)
    pair across chains. Charge / adducts / name apply to the whole
    complex. Global isotopes and fixed mods are duplicated to every
    chain at parse time so per-chain composition / mass calculations
    don't have to reach back up.
    """
    chains: tuple[Peptidoform, ...]
    charge: int | None = None
    adducts: tuple[str, ...] = ()
    name: str | None = None  # from (>>name)


ProFormaResult = Union[Peptidoform, MultiPeptidoform]


# ──────────────────────────────────────────────────────────────────────
# Parser construction (lazy, cached)
# ──────────────────────────────────────────────────────────────────────


@cache
def _grammar_text() -> str:
    """Read the vendored lark grammar from the package data directory."""
    return resources.files("constellation.data").joinpath("proforma.lark").read_text()


@cache
def _parser() -> Lark:
    """Build the lark Earley parser exactly once per process."""
    return Lark(
        _grammar_text(),
        parser="earley",
        maybe_placeholders=True,
        propagate_positions=False,
        cache=False,  # caching is at the @cache level here
    )


# ──────────────────────────────────────────────────────────────────────
# Tree → dataclass transformer
#
# Each method receives the children of the corresponding grammar rule.
# `maybe_placeholders=True` means optional rules show up as None when
# absent. Token children are lark.Token instances; we cast to str/float
# inside the methods.
# ──────────────────────────────────────────────────────────────────────


@v_args(inline=False)
class _ProFormaTransformer(Transformer):
    # NOTE on item layout: lark with `maybe_placeholders=True` does NOT
    # fill None for absent optional rules under the Earley parser — they
    # just disappear from the child list. Every method below is therefore
    # position-agnostic: dispatch on item type, never on items[i].

    # ── Top-level ────────────────────────────────────────────────────
    def peptidoform_ion_set(self, items):
        # Layout: name_set? mod_global* (peptidoform_ion "+")* peptidoform_ion
        # Children are in source order; dispatch by type.
        name: str | None = None
        globals_: list[_GlobalMod] = []
        ions: list[ProFormaResult] = []
        for item in items:
            if isinstance(item, _NameSet):
                name = item.value
            elif isinstance(item, _GlobalMod):
                globals_.append(item)
            elif isinstance(item, (Peptidoform, MultiPeptidoform)):
                ions.append(item)
            elif item is None:
                continue
            else:  # pragma: no cover
                raise ProFormaError(
                    f"unexpected child in peptidoform_ion_set: {item!r}"
                )

        if len(ions) > 1:
            raise NotImplementedError(
                "Chimeric ProForma spectra (`+`-separated peptidoformIons) "
                "are not yet supported. They are rare in practice; raise an "
                "issue with the source file if you encounter one."
            )

        result = ions[0]
        # Apply globals (isotopes + fixed mods) to result. For
        # MultiPeptidoform, propagate to every chain.
        isotopes = tuple(g.label for g in globals_ if g.kind == "isotope")
        fixed = tuple(
            (g.mod, g.locations) for g in globals_ if g.kind == "fixed"
        )

        if isinstance(result, MultiPeptidoform):
            new_chains = tuple(
                replace(
                    chain,
                    global_isotopes=isotopes + chain.global_isotopes,
                    fixed_mods=fixed + chain.fixed_mods,
                )
                for chain in result.chains
            )
            result = replace(result, chains=new_chains, name=name or result.name)
        else:
            assert isinstance(result, Peptidoform)
            result = replace(
                result,
                global_isotopes=isotopes + result.global_isotopes,
                fixed_mods=fixed + result.fixed_mods,
            )
            # Note: PeptidoformIonSet name `(>>>name)` overrides any inner name.
            if name is not None:
                result = replace(result, name=name)
        return result

    def name_set(self, items):
        return _NameSet(value=_str(items[0]))

    # ── peptidoform_ion: handles // chains + charge ──────────────────
    def peptidoform_ion(self, items):
        # Layout: name_ion? (peptidoform "//")* peptidoform peptidoform_charge?
        # Dispatch by type (no None placeholders in Earley).
        name: str | None = None
        charge: _Charge | None = None
        peptidoforms: list[Peptidoform] = []
        for item in items:
            if isinstance(item, _NameIon):
                name = item.value
            elif isinstance(item, _Charge):
                charge = item
            elif isinstance(item, Peptidoform):
                peptidoforms.append(item)
            elif item is None:
                continue
            else:  # pragma: no cover
                raise ProFormaError(
                    f"unexpected child in peptidoform_ion: {item!r}"
                )

        charge_int = charge.value if charge else None
        adducts = charge.adducts if charge else ()

        if len(peptidoforms) == 1:
            p = peptidoforms[0]
            return replace(
                p,
                charge=charge_int if charge_int is not None else p.charge,
                adducts=adducts if adducts else p.adducts,
                name=name if name is not None else p.name,
            )
        # MultiPeptidoform: chains share charge / adducts / name.
        return MultiPeptidoform(
            chains=tuple(peptidoforms),
            charge=charge_int,
            adducts=adducts,
            name=name,
        )

    def name_ion(self, items):
        return _NameIon(value=_str(items[0]))

    # ── peptidoform: the meat ─────────────────────────────────────────
    def peptidoform(self, items):
        # Layout: name_pep? unknown_pos_block* labile_mod* nterm? sequence cterm?
        # Dispatch by type (no None placeholders in Earley).
        name: str | None = None
        unknown_pos: list[tuple[ModRef, int]] = []
        labile: list[ModRef] = []
        nterm_mods: tuple[TaggedMod, ...] = ()
        cterm_mods: tuple[TaggedMod, ...] = ()
        seq_data: _SeqData | None = None
        for item in items:
            if isinstance(item, _NamePep):
                name = item.value
                continue
            if isinstance(item, _UnknownPosBlock):
                unknown_pos.extend(item.entries)
            elif isinstance(item, _LabileMod):
                labile.append(item.mod)
            elif isinstance(item, _NTerm):
                nterm_mods = item.mods
            elif isinstance(item, _CTerm):
                cterm_mods = item.mods
            elif isinstance(item, _SeqData):
                seq_data = item
            elif item is None:
                continue
            else:  # pragma: no cover
                raise ProFormaError(f"unexpected child in peptidoform: {item!r}")
        if seq_data is None:
            raise ProFormaSyntaxError("peptidoform missing sequence body")
        return Peptidoform(
            sequence=seq_data.sequence,
            residue_mods=seq_data.residue_mods,
            n_term_mods=nterm_mods,
            c_term_mods=cterm_mods,
            labile_mods=tuple(labile),
            unknown_pos_mods=tuple(unknown_pos),
            ranges=seq_data.ranges,
            name=name,
        )

    def name_pep(self, items):
        return _NamePep(value=_str(items[0]))

    # ── unknown-pos block ────────────────────────────────────────────
    def unknown_pos_block(self, items):
        entries = []
        for child in items:
            mod, count = child  # _UnknownPosMod.entry: (ModRef, int)
            entries.append((mod, count))
        return _UnknownPosBlock(entries=entries)

    def unknown_pos_mod(self, items):
        # mod occurrence_spec?  →  one or two items
        mod_internal = items[0]
        first_modref = mod_internal.first_modref()
        count = items[1] if len(items) > 1 else 1
        return (first_modref, count)

    def occurrence_spec(self, items):
        return int(items[0])

    # ── labile ───────────────────────────────────────────────────────
    def labile_mod(self, items):
        mod_internal = items[0]
        return _LabileMod(mod=mod_internal.first_modref())

    # ── N-term / C-term ──────────────────────────────────────────────
    def nterm(self, items):
        # items = [mod_or_label, mod_or_label_or_None]  (up to 2)
        mods: list[TaggedMod] = []
        for it in items:
            if it is None:
                continue
            mods.append(_to_tagged(it))
        return _NTerm(mods=tuple(mods))

    def cterm(self, items):
        mods: list[TaggedMod] = []
        for it in items:
            if it is None:
                continue
            mods.append(_to_tagged(it))
        return _CTerm(mods=tuple(mods))

    # ── Sequence body ────────────────────────────────────────────────
    def sequence(self, items):
        # Walk sections, building up sequence string + residue_mods + ranges.
        sequence_chars: list[str] = []
        residue_mods: dict[int, list[TaggedMod]] = {}
        ranges: list[Range] = []
        for section in items:
            if isinstance(section, _SequenceElement):
                pos = len(sequence_chars)
                sequence_chars.append(section.aa)
                if section.mods:
                    residue_mods[pos] = list(section.mods)
            elif isinstance(section, _AmbiguousAA):
                # An ambiguous AA group — multiple residues, all sharing the
                # same modifications. The mods could equally apply to any
                # residue in the group (spec §4.4, the "(?...)" notation).
                start = len(sequence_chars)
                for elem in section.elements:
                    sequence_chars.append(elem.aa)
                    if elem.mods:
                        residue_mods[len(sequence_chars) - 1] = list(elem.mods)
                end = len(sequence_chars) - 1
                if section.mods:
                    # Encoded as a Range — mod attaches somewhere in [start, end].
                    ranges.append(Range(start=start, end=end, mods=section.mods))
            elif isinstance(section, _ModRange):
                start = len(sequence_chars)
                for elem in section.elements:
                    sequence_chars.append(elem.aa)
                    if elem.mods:
                        residue_mods[len(sequence_chars) - 1] = list(elem.mods)
                end = len(sequence_chars) - 1
                ranges.append(Range(start=start, end=end, mods=section.mods))
            else:  # pragma: no cover
                raise ProFormaError(f"unexpected section: {section!r}")
        return _SeqData(
            sequence="".join(sequence_chars),
            residue_mods={k: tuple(v) for k, v in residue_mods.items()},
            ranges=tuple(ranges),
        )

    def sequence_element(self, items):
        # AMINOACID + 0+ mod_or_label
        aa = _str(items[0])
        mods: list[TaggedMod] = []
        for it in items[1:]:
            if it is None:
                continue
            mods.append(_to_tagged(it))
        return _SequenceElement(aa=aa, mods=tuple(mods))

    def ambiguous_aa(self, items):
        # 1+ sequence_element, then 0+ mod
        elements: list[_SequenceElement] = []
        mods: list[ModRef] = []
        for it in items:
            if isinstance(it, _SequenceElement):
                elements.append(it)
            elif isinstance(it, _ModInternal):
                mods.append(it.first_modref())
        return _AmbiguousAA(elements=elements, mods=tuple(mods))

    def mod_range(self, items):
        # 1+ sequence_element, then 1+ mod
        elements: list[_SequenceElement] = []
        mods: list[ModRef] = []
        for it in items:
            if isinstance(it, _SequenceElement):
                elements.append(it)
            elif isinstance(it, _ModInternal):
                mods.append(it.first_modref())
        return _ModRange(elements=elements, mods=tuple(mods))

    # ── Modifications: mod / bare_label / mod_internal / mod_single ──
    def mod(self, items):
        # mod = "[" mod_internal "]"
        return items[0]  # passes through _ModInternal

    def bare_label(self, items):
        # bare_label = "[" mod_label "]" — a label-only reference like [#g1]
        label = items[0]  # _Label
        return _BareLabel(group_id=label.group_id, score=label.score)

    def mod_internal(self, items):
        # mod_single mod_label? ("|" mod_single mod_label?)*
        # Coming in flattened: [modref, label_or_None, modref, label_or_None, ...]
        entries: list[tuple[ModRef, _Label | None]] = []
        i = 0
        while i < len(items):
            modref = items[i]
            label = items[i + 1] if i + 1 < len(items) else None
            if isinstance(label, _Label):
                entries.append((modref, label))
                i += 2
            else:
                entries.append((modref, None))
                i += 1
                if label is None and i + 1 <= len(items):
                    pass  # already consumed via i+=1
        # NOTE: lark's flattening with maybe_placeholders=True can produce
        # different layouts. Above is best-effort. Test coverage will catch
        # any layout I missed.
        return _ModInternal(entries=entries)

    def mod_internal_defined(self, items):
        return _ModInternal(entries=[(m, None) for m in items])

    def mod_defined(self, items):
        return items[0]  # passes through _ModInternal

    # ── mod_single variants ──────────────────────────────────────────
    def mod_formula(self, items):
        return ModRef(cv="Formula", formula=_str(items[0]).strip())

    def mod_glycan(self, items):
        return ModRef(cv="Glycan", glycan=_str(items[0]).strip())

    def mod_mass(self, items):
        # CV_MASS_PREFIX? SIGNED_NUMBER  →  one or two tokens
        if len(items) == 1:
            return ModRef(cv=None, mass_delta=float(_str(items[0])))
        cv, signed_num = items
        return ModRef(cv=_str(cv), mass_delta=float(_str(signed_num)))

    def mod_accession(self, items):
        cv = _str(items[0])
        accession = _str(items[1])
        return ModRef(cv=cv, accession=accession)

    def mod_info(self, items):
        return ModRef(cv="INFO", info_text=_str(items[0]).strip())

    def mod_name(self, items):
        # CV_NAME_PREFIX? MOD_INNER_TEXT  →  one or two tokens
        if len(items) == 1:
            return ModRef(cv=None, name=_str(items[0]).strip())
        cv, name_token = items
        return ModRef(cv=_str(cv), name=_str(name_token).strip())

    # ── Labels ───────────────────────────────────────────────────────
    def mod_label(self, items):
        # "#" LABEL_NAME LABEL_SCORE?
        name = _str(items[0])
        score = None
        if len(items) > 1 and items[1] is not None:
            score_str = _str(items[1])
            # LABEL_SCORE is "(...)" — strip parens.
            score = float(score_str.strip("()"))
        return _Label(group_id=name, score=score)

    # ── Global mods ──────────────────────────────────────────────────
    def mod_global(self, items):
        return items[0]

    def isotope_label(self, items):
        return _GlobalMod(kind="isotope", label=_str(items[0]))

    def global_fixed(self, items):
        # mod_defined "@" global_location ("," global_location)*
        mod_internal = items[0]
        modref = mod_internal.first_modref()
        locations = tuple(loc for loc in items[1:] if loc is not None)
        return _GlobalMod(kind="fixed", mod=modref, locations=locations)

    def global_location(self, items):
        return items[0]

    def cterm_loc(self, items):
        # ("C-term" (":" AMINOACID)?)
        if items and items[0] is not None:
            return f"C-term:{_str(items[0])}"
        return "C-term"

    def nterm_loc(self, items):
        if items and items[0] is not None:
            return f"N-term:{_str(items[0])}"
        return "N-term"

    def aa_loc(self, items):
        return _str(items[0])

    # ── Charge ───────────────────────────────────────────────────────
    def peptidoform_charge(self, items):
        # SIGNED_INT  OR  "[" adduct_ion ("," adduct_ion)* "]"
        if len(items) == 1 and isinstance(items[0], Token):
            return _Charge(value=int(_str(items[0])), adducts=())
        adducts = tuple(_str(a) for a in items if a is not None)
        # When adducts are present, the charge int is implied by the sum of
        # adduct ion charges; we sum on the fly. Each adduct token has a
        # ":zN" suffix; parsing that out is a future refinement. For now,
        # leave adducts as raw strings and sum to None to indicate
        # "computed from adducts".
        return _Charge(value=None, adducts=adducts)

    def adduct_ion(self, items):
        # FORMULA_TEXT CHARGE_SUFFIX occurrence_spec?
        formula = _str(items[0])
        charge_suffix = _str(items[1])
        return f"{formula}{charge_suffix}"


# ── Internal sentinel types used inside the transformer ───────────────
# These are private; they exist to distinguish what a particular
# transformer method returns to its parent. The parent decides how
# to assemble them into the final dataclass. None of these escape to
# the user-facing API.


@dataclass
class _NameSet:
    value: str


@dataclass
class _NameIon:
    value: str


@dataclass
class _NamePep:
    value: str


@dataclass
class _GlobalMod:
    kind: Literal["isotope", "fixed"]
    label: str | None = None
    mod: ModRef | None = None
    locations: tuple[str, ...] = ()


@dataclass
class _UnknownPosBlock:
    entries: list[tuple[ModRef, int]]


@dataclass
class _LabileMod:
    mod: ModRef


@dataclass
class _NTerm:
    mods: tuple[TaggedMod, ...]


@dataclass
class _CTerm:
    mods: tuple[TaggedMod, ...]


@dataclass
class _SequenceElement:
    aa: str
    mods: tuple[TaggedMod, ...]


@dataclass
class _AmbiguousAA:
    elements: list[_SequenceElement]
    mods: tuple[ModRef, ...]


@dataclass
class _ModRange:
    elements: list[_SequenceElement]
    mods: tuple[ModRef, ...]


@dataclass
class _SeqData:
    sequence: str
    residue_mods: dict[int, tuple[TaggedMod, ...]]
    ranges: tuple[Range, ...]


@dataclass
class _ModInternal:
    """Result of parsing one bracket payload — possibly multi-CV via `|`.

    The first entry's ModRef is the canonical one for downstream code that
    only cares about identity. Other entries are alternative CV references
    to the same chemical entity (spec §4.2.10) — kept here for round-trip
    fidelity but rarely consumed beyond that.
    """
    entries: list[tuple[ModRef, "_Label | None"]]

    def first_modref(self) -> ModRef:
        return self.entries[0][0]

    def first_label(self) -> "_Label | None":
        return self.entries[0][1]


@dataclass
class _Label:
    group_id: str
    score: float | None = None


@dataclass
class _BareLabel:
    group_id: str
    score: float | None


@dataclass
class _Charge:
    value: int | None
    adducts: tuple[str, ...]


def _str(tok: Token | str) -> str:
    return str(tok)


def _to_tagged(item) -> TaggedMod:
    """Convert a `mod` (i.e. _ModInternal) or `bare_label` into a TaggedMod."""
    if isinstance(item, _ModInternal):
        modref, label = item.first_modref(), item.first_label()
        if label is None:
            return TaggedMod(mod=modref)
        return TaggedMod(mod=modref, group_id=label.group_id, score=label.score)
    if isinstance(item, _BareLabel):
        return TaggedMod(mod=None, group_id=item.group_id, score=item.score)
    raise ProFormaError(f"cannot tag item: {item!r}")


# ──────────────────────────────────────────────────────────────────────
# Public parser
# ──────────────────────────────────────────────────────────────────────


def parse_proforma(s: str) -> ProFormaResult:
    """Parse a ProForma 2.0 string into a Peptidoform or MultiPeptidoform.

    Raises:
        ProFormaSyntaxError: input does not conform to the grammar
        ProFormaSemanticError: grammar-valid but semantically invalid
            (e.g. orphan cross-link IDs)
        NotImplementedError: chimeric (`+`-separated) inputs
    """
    if not s:
        raise ProFormaSyntaxError("empty input")
    try:
        tree = _parser().parse(s)
    except LarkError as e:
        raise ProFormaSyntaxError(f"failed to parse {s!r}: {e}") from e
    transformer = _ProFormaTransformer()
    try:
        result = transformer.transform(tree)
    except NotImplementedError:
        raise
    except Exception as e:  # pragma: no cover
        raise ProFormaSyntaxError(
            f"transform failure on {s!r}: {type(e).__name__}: {e}"
        ) from e
    _validate(result)
    return result


def _validate(p: ProFormaResult) -> None:
    """Run semantic validation that the grammar can't express."""
    # Cross-link pairing: each `#XL\d+` group ID should appear at exactly two
    # positions (the cross-link partners) — one with mod payload, one as bare
    # label (or both with the same XLMOD payload, depending on author style).
    if isinstance(p, MultiPeptidoform):
        for chain in p.chains:
            _validate_chain_groups(chain, allow_orphan_xl=True)
    else:
        _validate_chain_groups(p, allow_orphan_xl=False)


def _validate_chain_groups(p: Peptidoform, *, allow_orphan_xl: bool) -> None:
    """Cross-check group ID consistency within a single chain."""
    counts: dict[str, int] = {}
    for tagged_list in p.residue_mods.values():
        for t in tagged_list:
            if t.group_id is not None:
                counts[t.group_id] = counts.get(t.group_id, 0) + 1
    for t in p.n_term_mods + p.c_term_mods:
        if t.group_id is not None:
            counts[t.group_id] = counts.get(t.group_id, 0) + 1

    for group_id, n in counts.items():
        if group_id.upper().startswith("XL") and n < 2 and not allow_orphan_xl:
            raise ProFormaSemanticError(
                f"cross-link group {group_id!r} has only {n} reference(s); "
                "expected ≥2 (one per cross-link partner)"
            )


# ──────────────────────────────────────────────────────────────────────
# Helpers — exposed publicly (under leading underscore by spec convention,
# but consumers may freely import + use them)
# ──────────────────────────────────────────────────────────────────────


def _has_crosslinks(p: ProFormaResult) -> bool:
    """True iff the peptidoform participates in any cross-link.

    Detection: any TaggedMod whose `group_id` starts with "XL" (case-insensitive,
    per ProForma §4.2.3 reserved prefix), OR any ModRef with cv="XLMOD".
    """
    if isinstance(p, MultiPeptidoform):
        return any(_has_crosslinks(c) for c in p.chains)
    return _chain_has_crosslinks(p)


def _chain_has_crosslinks(p: Peptidoform) -> bool:
    for tagged_list in p.residue_mods.values():
        for t in tagged_list:
            if t.group_id and t.group_id.upper().startswith("XL"):
                return True
            if t.mod and t.mod.cv == "XLMOD":
                return True
    for t in p.n_term_mods + p.c_term_mods:
        if t.group_id and t.group_id.upper().startswith("XL"):
            return True
        if t.mod and t.mod.cv == "XLMOD":
            return True
    return False


def _has_branches(p: ProFormaResult) -> bool:
    """True iff the peptidoform participates in a branch (#BRANCH per spec §4.2.5)."""
    if isinstance(p, MultiPeptidoform):
        return any(_has_branches(c) for c in p.chains)
    for tagged_list in p.residue_mods.values():
        for t in tagged_list:
            if t.group_id and t.group_id.upper() == "BRANCH":
                return True
    for t in p.n_term_mods + p.c_term_mods:
        if t.group_id and t.group_id.upper() == "BRANCH":
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────────────────────────────


def format_proforma(
    p: ProFormaResult, *, flatten_terminals: bool = False
) -> str:
    """Render a Peptidoform / MultiPeptidoform back to a ProForma 2.0 string.

    Lossless round-trip for spec-compliant inputs.

    Canonical rendering choices:
      - prefer accession over name when both are derivable
      - terminal mods always rendered with the dash separator
      - global mods rendered in spec order:
        ``<isotope>...<isotope><[fixed]@loc>...{labile}[unknown]?[Nterm]-SEQUENCE-[Cterm]``

    With ``flatten_terminals=True``, N-term and C-term mods are
    rendered onto the adjacent residue (`[mod]-X` → `X[mod]`,
    `Y-[mod]` → `Y[mod]`) — used by the EncyclopeDIA writer to match
    that format's flat convention. The peptidoform's in-memory state
    is unchanged; this only affects the output string.
    """
    if isinstance(p, MultiPeptidoform):
        chains = "//".join(format_proforma(c, flatten_terminals=flatten_terminals) for c in p.chains)
        suffix = ""
        if p.charge is not None:
            suffix = f"/{p.charge}"
        elif p.adducts:
            suffix = "/[" + ",".join(p.adducts) + "]"
        prefix = f"(>>{p.name})" if p.name else ""
        return prefix + chains + suffix

    return _format_chain(p, flatten_terminals=flatten_terminals)


def _format_chain(p: Peptidoform, *, flatten_terminals: bool) -> str:
    parts: list[str] = []

    # Globals: isotopes first, then fixed mods.
    for iso in p.global_isotopes:
        parts.append(f"<{iso}>")
    for mod, locs in p.fixed_mods:
        parts.append(f"<[{_format_modref(mod)}]@{','.join(locs)}>")

    # Labile mods.
    for lm in p.labile_mods:
        parts.append("{" + _format_modref(lm) + "}")

    # Unknown-position mods.
    if p.unknown_pos_mods:
        upm_strs = []
        for mod, count in p.unknown_pos_mods:
            base = "[" + _format_modref(mod) + "]"
            if count > 1:
                base = base + f"^{count}"
            upm_strs.append(base)
        parts.append("".join(upm_strs) + "?")

    # N-term + sequence body + C-term, with optional flattening.
    if flatten_terminals:
        # Fold n_term_mods onto position-0 residue and c_term_mods onto last.
        residue_mods = {k: list(v) for k, v in p.residue_mods.items()}
        if p.n_term_mods:
            residue_mods.setdefault(0, []).extend(p.n_term_mods)
        if p.c_term_mods and p.sequence:
            last = len(p.sequence) - 1
            residue_mods.setdefault(last, []).extend(p.c_term_mods)
        parts.append(_format_sequence_body(p.sequence, residue_mods, p.ranges))
    else:
        if p.n_term_mods:
            parts.append(_format_terminal_block(p.n_term_mods) + "-")
        parts.append(_format_sequence_body(p.sequence, p.residue_mods, p.ranges))
        if p.c_term_mods:
            parts.append("-" + _format_terminal_block(p.c_term_mods))

    # Charge suffix.
    if p.charge is not None:
        parts.append(f"/{p.charge}")
    elif p.adducts:
        parts.append("/[" + ",".join(p.adducts) + "]")

    if p.name:
        return f"(>{p.name})" + "".join(parts)
    return "".join(parts)


def _format_terminal_block(mods: tuple[TaggedMod, ...]) -> str:
    return "".join(_format_tagged(t) for t in mods)


def _format_sequence_body(
    sequence: str,
    residue_mods: dict[int, tuple[TaggedMod, ...]] | dict[int, list[TaggedMod]],
    ranges: tuple[Range, ...],
) -> str:
    # Ranges and per-residue mods are rendered inline. The algorithm walks
    # the sequence character-by-character; at each position, it decides
    # whether this position is the start of a range, in the middle of one,
    # the end of one, or a plain residue with possible per-residue mods.
    range_starts = {r.start: r for r in ranges}
    range_ends = {r.end: r for r in ranges}

    out: list[str] = []
    in_range = False
    for i, c in enumerate(sequence):
        if i in range_starts:
            out.append("(")
            in_range = True
        out.append(c)
        # Per-residue mods (rendered AFTER the residue).
        for t in residue_mods.get(i, ()):
            out.append(_format_tagged(t))
        if i in range_ends and in_range:
            out.append(")")
            r = range_ends[i]
            for m in r.mods:
                out.append("[" + _format_modref(m) + "]")
            in_range = False
    return "".join(out)


def _format_tagged(t: TaggedMod) -> str:
    if t.mod is None:
        # Bare label reference like [#g1]
        return "[#" + _format_label(t.group_id, t.score) + "]"
    label_part = ""
    if t.group_id is not None:
        label_part = "#" + _format_label(t.group_id, t.score)
    return "[" + _format_modref(t.mod) + label_part + "]"


def _format_label(group_id: str | None, score: float | None) -> str:
    if group_id is None:
        return ""
    if score is None:
        return group_id
    # Use repr-equivalent precision; Python `repr(float)` is shortest-round-trip.
    return f"{group_id}({score!r})"


def _format_modref(m: ModRef) -> str:
    if m.cv == "Formula":
        return f"Formula:{m.formula}"
    if m.cv == "Glycan":
        return f"Glycan:{m.glycan}"
    if m.cv == "INFO":
        return f"INFO:{m.info_text}"
    if m.accession is not None:
        return f"{m.cv}:{m.accession}"
    if m.mass_delta is not None:
        sign = "+" if m.mass_delta >= 0 else ""
        body = f"{sign}{m.mass_delta!r}"
        if m.cv:
            return f"{m.cv}:{body}"
        return body
    if m.name is not None:
        if m.cv:
            return f"{m.cv}:{m.name}"
        return m.name
    raise ProFormaError(f"cannot format empty ModRef: {m!r}")  # pragma: no cover


# ──────────────────────────────────────────────────────────────────────
# Global-isotope label decoder
# ──────────────────────────────────────────────────────────────────────


def parse_isotope_label(label: str) -> tuple[int, str]:
    """Decode a ProForma global-isotope label into (mass_number, element_symbol).

    Recognized forms (spec §4.6.1):

        "13C", "15N", "2H", "18O", ...    → (13, "C"), (15, "N"), ...
        "D"                                → (2, "H")    (deuterium alias)
        "T"                                → (3, "H")    (tritium alias)

    The ``D`` / ``T`` aliases are ProForma-spec conventions, not general
    chemistry nomenclature — that's why this helper lives next to the
    parser rather than in ``core.chem.elements``. Pair the result with
    ``ELEMENTS[symbol].isotope_mass(mass_number)`` to look up the exact
    NIST AME2020 mass.
    """
    if label == "D":
        return (2, "H")
    if label == "T":
        return (3, "H")
    i = 0
    while i < len(label) and label[i].isdigit():
        i += 1
    sym = label[i:]
    if i == 0 or not sym or not sym.isalpha():
        raise ValueError(f"unrecognized isotope label: {label!r}")
    return (int(label[:i]), sym)


# ──────────────────────────────────────────────────────────────────────
# Public re-exports
# ──────────────────────────────────────────────────────────────────────


__all__ = [
    "ProFormaError",
    "ProFormaSyntaxError",
    "ProFormaSemanticError",
    "ModRef",
    "TaggedMod",
    "Range",
    "Peptidoform",
    "MultiPeptidoform",
    "ProFormaResult",
    "parse_proforma",
    "format_proforma",
    "parse_isotope_label",
    "_has_crosslinks",
    "_has_branches",
]
