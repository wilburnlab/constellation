"""A search space: FASTA files → every precursor to look for, with how
each one was made.

The generative chain is protein → span → peptidoform → precursor (→
fragment, which the sweep compiles from the precursor). Each stage is kept
as its own table of a :class:`~constellation.massspec.library.Library`,
keyed to the stage before, so nothing is collapsed on the way:

- proteins (``PROTEIN_TABLE``), target and decoy;
- spans (``PEPTIDE_SPAN_TABLE``): where in which protein a sequence was
  cut, how each end was made, missed cleavages, flanks;
- span → peptidoform edges (``SPAN_PEPTIDE_EDGE``): a peptidoform is made
  only from the spans whose context licenses its modifications (a Protein
  N-term acetyl only from a span at the protein's N-terminus);
- peptidoforms (``PEPTIDE_TABLE``) with each modification and why it is
  there (``PEPTIDE_MODIFICATION_TABLE``: fixed, variable or label, and the
  rule);
- precursors (``PRECURSOR_TABLE``): each peptidoform at each charge whose
  m/z is in range.

A molecule appears once, under its content-derived id
(``massspec.library.ids``), with an edge to every path that makes it: the
only de-duplication is by identity. Fragments are not stored; the sweep
makes them from the precursors (``massspec.sweep.compile_queries``), so
one search space serves any fragmentation settings.

Modifications are given as rules (:class:`ModRule`), parsed from
``MOD[@SITE,...][:max=N]``. A rule's sites are residues, ``N-term`` /
``C-term`` (any peptide's ends) or ``Protein N-term`` / ``Protein C-term``
(a peptide's ends where they are its protein's); with no sites, the ones
UNIMOD shows for the modification (or, for one UNIMOD shows nowhere, such
as SILAC K, the ones it lists hidden); ``@*`` is every placement UNIMOD
lists. A rule places a modification only where UNIMOD lists it, so one rule
set applies across every peptide (pyro-Glu at an N-terminal Q and nowhere
else). Isotope labels (``<13C><15N>``) are a modification axis too: given
per FASTA or per protein, they make the labelled peptidoforms of that
protein's spans, beside or instead of the light ones.

Budgets: the number of peptidoforms is counted before any is built, and
the number of precursors before any table is, and either over its budget
raises with the counts rather than truncating.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.chem.elements import PROTON_MASS
from constellation.core.chem.modifications import UNIMOD, Modification, ModVocab
from constellation.core.io.fasta import iter_fasta, parse_header
from constellation.core.sequence.alphabets import AA
from constellation.core.sequence.proforma import Peptidoform, format_proforma, normalize_isotopes
from constellation.core.sequence.protein import (
    C_TERM,
    N_TERM,
    PROTEIN_C_TERM,
    PROTEIN_N_TERM,
    CleavageSpecificity,
    SiteContext,
    cleave,
    count_modforms,
    enumerate_modforms,
    peptide_masses,
)
from constellation.core.stages import file_sha256
from constellation.massspec.library.decoys import DecoySet, il_key, peptide_decoys, reverse_protein
from constellation.massspec.library.library import Library
from constellation.massspec.library.schemas import LIBRARY_FRAGMENT_TABLE

__all__ = [
    "DEFAULT_MAX_MZ",
    "DEFAULT_MIN_MZ",
    "Digestion",
    "ModRule",
    "SEARCH_SPACE_METADATA_KEY",
    "SearchSpaceBudgetError",
    "SearchSpaceSettings",
    "build_search_space",
    "count_search_space",
    "normalize_isotopes",
    "parse_label",
    "parse_protein_terminus",
    "rules_from_mapping",
]

#: The library metadata key under which a search space records how it was made.
SEARCH_SPACE_METADATA_KEY = "x.constellation.search_space"

#: Bumped when what a search space holds changes for the same FASTA files
#: and settings, so that one built by older code is not taken for finished
#: (``--resume``) by newer code (the ``PREDICTION_LAYOUT`` convention). With
#: the modification vocabulary's digest and the id scheme, it is what the
#: resume key holds besides the inputs and settings (:func:`search_space_key`).
#: Until 2026-11-10 the key held ``__version__``, which an editable install
#: never rewrites (it read ``0.0.1a2.dev123`` from 2026-05-28, before this
#: module existed), so a search space made before any fix passed as current,
#: and so did a prediction keyed on it.
#:
#: 1 (2026-11-10): first keyed so. Peptide-level decoys on decoy proteins; a
#: variable modification replacing a fixed one on its site; a ``:max=``
#: limit per rule; reverse-c-fixed keeping a cyclized first residue in place;
#: protein-level decoys counted as decoys.
SEARCH_SPACE_LAYOUT = 1


def search_space_key(fasta: Sequence[Path | str], settings: SearchSpaceSettings) -> str:
    """The stage key of a search space built from ``fasta`` with
    ``settings``: the inputs' SHA-256, the settings, :data:`SEARCH_SPACE_LAYOUT`,
    the SHA-256 of the UNIMOD vocabulary the modifications were read from,
    and the content-id scheme."""
    from constellation.core.stages import stage_key
    from constellation.massspec.library.ids import ID_SCHEME

    return stage_key(
        settings=settings.as_dict(),
        inputs=[file_sha256(Path(f)) for f in fasta],
        layout=SEARCH_SPACE_LAYOUT,
        unimod=_vocabulary_digest(),
        id_scheme=ID_SCHEME,
    )


def _vocabulary_digest() -> str:
    """The SHA-256 of the packaged UNIMOD vocabulary (``unimod.json``):
    the file carries no release of its own, so its bytes are its identity."""
    import hashlib
    from importlib import resources

    return hashlib.sha256(resources.files("constellation.data").joinpath("unimod.json").read_bytes()).hexdigest()

#: Wide enough for the MS1 scan range of most methods; narrow it to the
#: isolation windows for a DIA run if the extra precursors are not wanted.
DEFAULT_MIN_MZ = 300.0
DEFAULT_MAX_MZ = 2000.0

_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
_SITE_SPELLINGS = {
    "n-term": N_TERM,
    "nterm": N_TERM,
    "c-term": C_TERM,
    "cterm": C_TERM,
    "protein n-term": PROTEIN_N_TERM,
    "protein-n-term": PROTEIN_N_TERM,
    "protein_n_term": PROTEIN_N_TERM,
    "protein c-term": PROTEIN_C_TERM,
    "protein-c-term": PROTEIN_C_TERM,
    "protein_c_term": PROTEIN_C_TERM,
}
_POSITION_SITE = {
    "Any N-term": N_TERM,
    "Protein N-term": PROTEIN_N_TERM,
    "Any C-term": C_TERM,
    "Protein C-term": PROTEIN_C_TERM,
}


class SearchSpaceBudgetError(ValueError):
    """A search space would be larger than its budget."""


# ──────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Digestion:
    """How proteins are cut (``core.sequence.protein.cleave``)."""

    protease: str = "Trypsin"
    missed_cleavages: int = 2
    specificity: CleavageSpecificity = "specific"
    min_length: int = 6
    max_length: int | None = 30
    excise_initiator_met: bool = True


@dataclass(frozen=True, slots=True)
class ModRule:
    """A modification and where it may go.

    ``modification`` is the canonical id (``"UNIMOD:35"``); ``sites`` are
    residue letters and the terminal sentinels; ``kind`` is ``"fixed"``
    (every site that licenses it) or ``"variable"`` (enumerated);
    ``max_per_peptide`` limits a variable modification's sites in one
    peptidoform; ``name`` is the rule as given, recorded with every
    modification it places.
    """

    modification: str
    sites: tuple[str, ...]
    kind: Literal["fixed", "variable"] = "variable"
    max_per_peptide: int | None = None
    name: str = ""

    @classmethod
    def parse(cls, text: str, kind: Literal["fixed", "variable"], *, vocab: ModVocab = UNIMOD) -> ModRule:
        """``MOD[@SITE,...][:max=N]``: MOD a UNIMOD accession or name; SITE a
        residue, ``N-term``, ``C-term``, ``Protein N-term``, ``Protein
        C-term``, or ``*`` for every placement UNIMOD lists; no ``@`` for the
        placements UNIMOD shows (or those it lists hidden, for a modification
        it shows nowhere)."""
        spec = text.strip()
        limit = None
        m = re.search(r":max=(\d+)$", spec)
        if m:
            limit = int(m.group(1))
            spec = spec[: m.start()]
        name, at, site_text = spec.partition("@")
        name = name.strip()
        if name not in vocab:
            raise ValueError(f"unknown modification {name!r} in rule {text!r}")
        mod = vocab[name]
        if not at:
            specs = mod.specificities or mod.hidden_specificities
            sites = _sites_of(specs)
        elif site_text.strip() == "*":
            sites = _sites_of(mod.all_specificities)
        else:
            sites = tuple(_site(t, text) for t in site_text.split(",") if t.strip())
            for site in sites:
                if not _listed_anywhere(mod, site):
                    raise ValueError(
                        f"UNIMOD lists no placement of {mod.id} ({mod.name}) at {site!r} (rule {text!r})"
                    )
        if not sites:
            raise ValueError(f"rule {text!r} names no site")
        _refuse_links(mod, text)
        if kind == "fixed" and limit is not None:
            raise ValueError(f"a fixed rule takes no max= ({text!r})")
        return cls(mod.id, tuple(dict.fromkeys(sites)), kind, limit, text.strip())


#: Modifications that are one end of a link between two residues: placed on
#: one residue of a linear peptide, they describe no molecule (a lone half
#: disulfide; an intact crosslinker counted once per lysine it could reach).
#: Linked species are not generated yet, so a rule naming one is refused.
#: Quenched and hydrolysed monolinks, which are whole molecules on one
#: residue, are not links.
_HALF_LINKS = frozenset({"UNIMOD:374"})  # Dehydro: half of a disulfide bridge
_INTACT_CROSSLINKER = re.compile(r"^Intact (?!or monolink).*cross-?link", re.IGNORECASE)


def _refuse_links(mod: Modification, rule: str) -> None:
    if mod.id in _HALF_LINKS or _INTACT_CROSSLINKER.match(mod.name):
        raise ValueError(
            f"{mod.id} ({mod.name}) links two residues (rule {rule!r}); a search space "
            "holds linear peptidoforms, and linked species (disulfides, cross-links) "
            "are not generated yet"
        )


def _site(token: str, rule: str) -> str:
    t = token.strip()
    if len(t) == 1 and t in _CANONICAL:
        return t
    site = _SITE_SPELLINGS.get(t.lower())
    if site is None:
        raise ValueError(f"unknown site {t!r} in rule {rule!r}")
    return site


def _sites_of(specs) -> tuple[str, ...]:
    out: list[str] = []
    for s in specs:
        if s.position == "Anywhere":
            out.append(s.site)
        elif s.position in _POSITION_SITE:
            out.append(_POSITION_SITE[s.position])
    return tuple(dict.fromkeys(out))


_TERMINAL_POSITIONS = {
    N_TERM: ("Any N-term", "Protein N-term"),
    PROTEIN_N_TERM: ("Any N-term", "Protein N-term"),
    C_TERM: ("Any C-term", "Protein C-term"),
    PROTEIN_C_TERM: ("Any C-term", "Protein C-term"),
}


def _listed_anywhere(mod, site: str) -> bool:
    """Whether UNIMOD lists the modification at ``site`` at all (shown or
    hidden): a rule naming a site it never does is a mistake, not a rule
    that places nothing. Narrowing an "Any N-term" placement to protein
    N-termini is allowed."""
    positions = _TERMINAL_POSITIONS.get(site)
    if positions is not None:
        return any(s.position in positions for s in mod.all_specificities)
    return mod.licenses(site)


def parse_label(text: str) -> tuple[str, tuple[str, ...]]:
    """``TARGET=ISOTOPES``: TARGET a FASTA (as given, or its file name) or
    an accession; ISOTOPES comma-separated (``13C,15N``) or ``light`` for
    the unlabelled forms. Several labels for one target make several forms
    of each of its peptides."""
    target, eq, value = text.partition("=")
    if not eq or not target.strip() or not value.strip():
        raise ValueError(f"a label is TARGET=ISOTOPES, got {text!r}")
    value = value.strip()
    if value.lower() == "light":
        return target.strip(), ()
    raw = [v.strip().lstrip("<").rstrip(">") for v in value.split(",") if v.strip()]
    return target.strip(), normalize_isotopes(raw, where=text)


def parse_protein_terminus(text: str) -> tuple[str, int | None, int | None]:
    """``ACCESSION=START[-END]`` or ``ACCESSION=-END``: a mature chain's
    first and last residue, 1-based inclusive as UniProt numbers them (BSA's
    chain is ``P02769=25-607``)."""
    acc, eq, value = text.partition("=")
    m = re.fullmatch(r"\s*(\d+)?\s*(?:-\s*(\d+))?\s*", value)
    if not eq or not acc.strip() or m is None or not (m.group(1) or m.group(2)):
        raise ValueError(f"a protein terminus is ACCESSION=START[-END], got {text!r}")
    start = int(m.group(1)) if m.group(1) else None
    end = int(m.group(2)) if m.group(2) else None
    if start is not None and start < 1 or end is not None and end < 1 or (start and end and end < start):
        raise ValueError(f"bad terminus positions in {text!r}")
    return acc.strip(), start, end


def _fixed_overlap(a: ModRule, site_a: str, b: ModRule, site_b: str) -> str | None:
    """Where two fixed placements land on the same slot of some peptide,
    as words, or None. Residue sites meet on the residue; terminal sites
    (``N-term`` and ``Protein N-term`` both being the N-terminal slot of a
    protein-terminal peptide) meet where both are licensed for some first
    (last) residue, since a rule is placed only where UNIMOD licenses it."""
    end_a, end_b = _TERMINUS_OF.get(site_a), _TERMINUS_OF.get(site_b)
    if end_a is None or end_b is None:
        return f"residue {site_a}" if site_a == site_b else None
    if end_a != end_b:
        return None
    if a.modification not in UNIMOD or b.modification not in UNIMOD:
        return f"the {end_a}"
    mod_a, mod_b = UNIMOD[a.modification], UNIMOD[b.modification]
    for residue in AA.tokens:
        if mod_a.licenses(end_a, terminal_residue=residue, protein_terminus=True) and mod_b.licenses(
            end_b, terminal_residue=residue, protein_terminus=True
        ):
            return f"the {end_a} of a protein-terminal peptide with {residue} there"
    return None


_TERMINUS_OF = {N_TERM: N_TERM, PROTEIN_N_TERM: N_TERM, C_TERM: C_TERM, PROTEIN_C_TERM: C_TERM}


def _check_rule_limits(rules: Sequence[ModRule]) -> None:
    """Refuse two variable rules of one modification on a shared site with
    different limits: a placement there would count toward both, and which
    was meant cannot be told."""
    variable = [r for r in rules if r.kind == "variable"]
    for i, a in enumerate(variable):
        for b in variable[i + 1:]:
            shared = set(a.sites) & set(b.sites)
            if a.modification == b.modification and shared and a.max_per_peptide != b.max_per_peptide:
                raise ValueError(
                    f"variable rules {a.name or a.modification!r} and {b.name or b.modification!r} place "
                    f"{a.modification} at {sorted(shared)} with different limits; give one rule there"
                )


def _check_fixed_rules(rules: Sequence[ModRule]) -> None:
    """Two fixed rules may not put different modifications on one slot:
    which would win is a contradiction, not a precedence. An isotopic
    substitution (SILAC) and a chemical modification share a residue."""
    fixed = [r for r in rules if r.kind == "fixed"]

    def layer(rule: ModRule) -> bool:
        return rule.modification in UNIMOD and UNIMOD[rule.modification].is_isotopic_substitution

    for i, a in enumerate(fixed):
        for b in fixed[i + 1 :]:
            if a.modification == b.modification or layer(a) != layer(b):
                continue
            for site_a in a.sites:
                for site_b in b.sites:
                    where = _fixed_overlap(a, site_a, b, site_b)
                    if where is not None:
                        raise ValueError(
                            f"two fixed rules, {a.name or a.modification!r} and {b.name or b.modification!r}, "
                            f"both place a modification on {where}; make one of them variable"
                        )


@dataclass(frozen=True, slots=True)
class SearchSpaceSettings:
    """Everything a search space depends on besides its FASTA files."""

    digestion: Digestion = field(default_factory=Digestion)
    rules: tuple[ModRule, ...] = ()
    #: Variable modifications per peptidoform, all rules together.
    max_variable: int = 2
    #: (target, isotopes) pairs, from :func:`parse_label`.
    labels: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: (accession, start, end) mature-chain termini, from :func:`parse_protein_terminus`.
    protein_termini: tuple[tuple[str, int | None, int | None], ...] = ()
    charges: tuple[int, ...] = (2, 3, 4)
    min_mz: float = DEFAULT_MIN_MZ
    max_mz: float = DEFAULT_MAX_MZ
    decoys: tuple[DecoySet, ...] = (DecoySet(),)
    #: Budgets (None: none). Peptidoforms are counted before any is built
    #: (an upper bound: a form two contexts make is counted twice).
    max_peptidoforms: int | None = 5_000_000
    max_precursors: int | None = 5_000_000

    def __post_init__(self) -> None:
        if not self.charges or any(int(z) < 1 for z in self.charges):
            raise ValueError(f"charges must be positive, got {self.charges}")
        if self.min_mz > self.max_mz:
            raise ValueError(f"min_mz {self.min_mz} > max_mz {self.max_mz}")
        labels = tuple(
            (target, normalize_isotopes(isotopes, where=f"{target}={','.join(isotopes)}"))
            for target, isotopes in self.labels
        )
        object.__setattr__(self, "labels", tuple(dict.fromkeys(labels)))
        for rule in self.rules:
            if rule.modification in UNIMOD:
                _refuse_links(UNIMOD[rule.modification], rule.name or rule.modification)
        _check_fixed_rules(self.rules)
        _check_rule_limits(self.rules)
        seen: set[tuple[int, str]] = set()
        for d in self.decoys:
            if (d.index, d.method) in seen:
                raise ValueError(f"decoy set {d.index} given twice for {d.method}")
            seen.add((d.index, d.method))
        # A method whose decoys depend on neither the set nor the seed makes
        # the same decoys every time, so a second set of it is a copy of the
        # first (reverse-protein:sets=2 made DECOY1_ proteins identical to
        # DECOY_ ones and set 1 no peptide; a second reverse-interior set
        # would be the first's fallback shuffles under its name).
        from constellation.massspec.library.decoys import DECOY_METHODS

        methods = [d.method for d in self.decoys]
        for method in dict.fromkeys(methods):
            if methods.count(method) > 1 and not DECOY_METHODS[method].seeded:
                raise ValueError(
                    f"{method} makes the same decoys for every set and seed, so a second set of it would copy "
                    f"the first: give it once, and shuffle-interior (seeded) for more sets"
                )

    def as_dict(self) -> dict[str, Any]:
        """The settings as plain data, for a manifest."""
        out = asdict(self)
        out["rules"] = [asdict(r) for r in self.rules]
        out["decoys"] = [asdict(d) for d in self.decoys]
        out["labels"] = [[t, list(i)] for t, i in self.labels]
        out["protein_termini"] = [list(t) for t in self.protein_termini]
        out["charges"] = list(self.charges)
        return out


# ──────────────────────────────────────────────────────────────────────
# Building
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _Protein:
    accession: str
    sequence: str
    description: str
    sources: list[str]
    is_decoy: bool = False
    decoy_set: int | None = None
    states: tuple[tuple[str, ...], ...] = ((),)


def _read_proteins(fastas: Sequence[str | Path]) -> tuple[list[_Protein], list[dict[str, Any]]]:
    by_acc: dict[str, _Protein] = {}
    inputs: list[dict[str, Any]] = []
    for fasta in fastas:
        path = Path(fasta)
        n = 0
        for record in iter_fasta(path):
            acc = record.id
            if not acc:
                raise ValueError(f"{path}: a record has no id")
            seq = record.sequence.upper().rstrip("*")
            have = by_acc.get(acc)
            if have is None:
                by_acc[acc] = _Protein(acc, seq, record.description, [str(fasta), path.name])
            elif have.sequence != seq:
                raise ValueError(f"accession {acc!r} has two different sequences ({have.sources[0]} and {path})")
            else:
                have.sources += [str(fasta), path.name]
            n += 1
        inputs.append({"path": str(path), "sha256": file_sha256(path), "n_records": n})
    return list(by_acc.values()), inputs


#: Accession prefixes other tools give decoys: a FASTA entry named so is
#: almost surely one, and would be searched as a target.
_DECOY_LIKE = re.compile(r"^(decoy\d*_|rev_|reverse_|xxx_|shuffle[d]?_)", re.IGNORECASE)


def _check_decoy_accessions(targets: Sequence[_Protein], decoys: Sequence[_Protein], counts: dict) -> None:
    """Refuse a FASTA entry that has the accession of a decoy made here (two
    proteins, one id), and warn of entries named as other tools name decoys
    (``DECOY_P1``, ``rev_P2``), which are read as targets: a decoy-bearing
    FASTA searched again would double its decoys and count them as targets
    (until 2026-11-10, with no word)."""
    import warnings

    made = {p.accession for p in decoys}
    clash = sorted(p.accession for p in targets if p.accession in made)
    if clash:
        raise ValueError(
            f"FASTA entries have the accessions of the decoys made for this search space ({', '.join(clash[:3])}"
            f"{', ...' if len(clash) > 3 else ''}): remove the decoys from the FASTA, which are made here"
        )
    named = [p.accession for p in targets if _DECOY_LIKE.match(p.accession)]
    counts["decoy_named_targets"] = len(named)
    if named:
        warnings.warn(
            f"{len(named)} FASTA entries are named as decoys are ({', '.join(named[:3])}"
            f"{', ...' if len(named) > 3 else ''}) and are read as targets; give the FASTA without its decoys",
            stacklevel=3,
        )


def _states(protein: _Protein, labels: Sequence[tuple[str, tuple[str, ...]]]) -> tuple[tuple[str, ...], ...]:
    names = {protein.accession, parse_header(protein.accession).accession, *protein.sources}
    found = [iso for target, iso in labels if target in names]
    return tuple(dict.fromkeys(found)) if found else ((),)


def _termini(protein: _Protein, termini) -> tuple[list[int], list[int]]:
    names = {protein.accession, parse_header(protein.accession).accession}
    n_extra: list[int] = []
    c_extra: list[int] = []
    for acc, start, end in termini:
        if acc not in names:
            continue
        if start is not None:
            if start > len(protein.sequence):
                raise ValueError(f"{acc}: terminus {start} past the protein's {len(protein.sequence)} residues")
            n_extra.append(start - 1)
        if end is not None:
            if end > len(protein.sequence):
                raise ValueError(f"{acc}: terminus {end} past the protein's {len(protein.sequence)} residues")
            c_extra.append(end)
    return n_extra, c_extra


def _enumeration_args(rules: Sequence[ModRule]) -> tuple[dict, dict, dict, dict]:
    """``enumerate_modforms`` arguments from the rules, and a map from what
    a placement records back to the rule that made it."""
    fixed: dict[str, list[str]] = {}
    variable: dict[str, list[str]] = {}
    limits: list[tuple[str, tuple[str, ...], int]] = []
    rule_of: dict[tuple[str, str, str], str] = {}
    for rule in rules:
        for site in rule.sites:
            rule_of.setdefault((rule.kind, site, rule.modification), rule.name)
            if rule.kind == "fixed":
                keys = fixed.setdefault(site, [])
                if rule.modification not in keys:
                    keys.append(rule.modification)
            else:
                keys = variable.setdefault(site, [])
                if rule.modification not in keys:
                    keys.append(rule.modification)
        # A rule's limit is on its own placements (its modification at its
        # sites): until 2026-11-10 limits were pooled by modification, so
        # Oxidation@M with Oxidation@W:max=1 allowed one oxidation in all.
        if rule.kind == "variable" and rule.max_per_peptide is not None:
            limits.append((rule.modification, tuple(rule.sites), int(rule.max_per_peptide)))
    return fixed, variable, limits, rule_of


def build_search_space(
    fastas: Sequence[str | Path],
    settings: SearchSpaceSettings | None = None,
    *,
    vocab: ModVocab = UNIMOD,
    count_only: bool = False,
) -> Library | dict[str, Any]:
    """The search space of ``fastas`` under ``settings``: a
    :class:`~constellation.massspec.library.Library` holding every stage
    (module docstring), or with ``count_only`` only the counts per stage."""
    from constellation.massspec.library.ids import (
        ID_SCHEME,
        ID_SCHEME_METADATA_KEY,
        canonical_peptide_key,
        canonical_protein_key,
        content_ids,
    )

    s = settings or SearchSpaceSettings()
    d = s.digestion
    counts: dict[str, Any] = {}

    # ── proteins, target then decoy ───────────────────────────────────
    proteins, inputs = _read_proteins(fastas)
    for p in proteins:
        p.states = _states(p, s.labels)
    unmatched = [t for t, _ in s.labels if not any(t in {q.accession, parse_header(q.accession).accession, *q.sources} for q in proteins)]
    if unmatched:
        raise ValueError(f"labels name no FASTA or protein given: {unmatched}")
    unmatched = [
        acc for acc, _, _ in s.protein_termini
        if not any(acc in {q.accession, parse_header(q.accession).accession} for q in proteins)
    ]
    if unmatched:
        raise ValueError(f"protein termini name no protein given: {sorted(set(unmatched))}")
    substitutions = [r.name or r.modification for r in s.rules if r.modification in vocab and vocab[r.modification].is_isotopic_substitution]
    if substitutions and any(any(state) for p in proteins for state in p.states):
        raise ValueError(
            f"isotopic substitutions ({', '.join(substitutions)}) and whole-protein labels would "
            "label one residue's atoms twice; give the labelled proteins their own search space"
        )
    targets = list(proteins)
    for ds in s.decoys:
        if ds.level != "protein":
            continue
        prefix = "DECOY_" if ds.index == 0 else f"DECOY{ds.index}_"
        for p in targets:
            proteins.append(
                _Protein(f"{prefix}{p.accession}", reverse_protein(p.sequence), "", [], True, ds.index, p.states)
            )
    counts["proteins"] = len(targets)
    counts["decoy_proteins"] = len(proteins) - len(targets)

    # ── spans ─────────────────────────────────────────────────────────
    span_rows: list[tuple[int, Any]] = []  # (protein index, Peptide)
    n_noncanonical = 0
    for pi, p in enumerate(proteins):
        n_extra, c_extra = _termini(p, s.protein_termini) if not p.is_decoy else ([], [])
        for sp in cleave(
            p.sequence,
            d.protease,
            missed_cleavages=d.missed_cleavages,
            min_length=d.min_length,
            max_length=d.max_length,
            specificity=d.specificity,
            excise_initiator_met=d.excise_initiator_met,
            extra_n_termini=n_extra,
            extra_c_termini=c_extra,
            validate_alphabet=False,
            return_spans=True,
        ):
            if not set(sp.sequence) <= _CANONICAL:
                n_noncanonical += 1
                continue
            span_rows.append((pi, sp))
    counts["spans"] = len(span_rows)
    counts["spans_noncanonical_skipped"] = n_noncanonical

    # ── groups: one enumeration per (sequence, protein-terminal context, label) ──
    fixed, variable, limits, rule_of = _enumeration_args(s.rules)
    groups: dict[tuple[str, bool, bool, tuple[str, ...]], list[int]] = {}
    for si, (pi, sp) in enumerate(span_rows):
        for state in proteins[pi].states:
            groups.setdefault((sp.sequence, sp.n_term == "protein", sp.c_term == "protein", state), []).append(si)
    counts["enumeration_groups"] = len(groups)
    kw = dict(
        fixed=fixed or None,
        variable=variable or None,
        max_variable=s.max_variable,
        vocab=vocab,
        skip_unlicensed=True,
        limit_groups=limits or None,
    )
    planned = 0
    # one count per (sequence, context): the label does not change the forms
    counted: dict[tuple[str, bool, bool], int] = {}
    for seq, pn, pc_, _state in groups:
        c = counted.get((seq, pn, pc_))
        if c is None:
            c = counted[(seq, pn, pc_)] = count_modforms(seq, context=SiteContext(pn, pc_), **kw)
        planned += c
    counts["peptidoforms_planned"] = planned
    if s.max_peptidoforms is not None and planned > s.max_peptidoforms:
        raise SearchSpaceBudgetError(
            f"the search space would hold up to {planned:,} peptidoforms, over the budget of "
            f"{s.max_peptidoforms:,} (spans {len(span_rows):,}, enumeration groups {len(groups):,}); "
            "narrow the variable modifications, lower max_variable or the missed cleavages, "
            "or raise max_peptidoforms"
        )

    # ── peptidoforms, their modifications and their spans ─────────────
    forms: list[Peptidoform] = []
    keys: list[str] = []
    index_of: dict[str, int] = {}
    # One row per (peptidoform, site, modification, origin, rule): a form two
    # paths make (fixed Protein N-term acetyl at one protein's start, variable
    # N-term acetyl inside another) keeps both origins.
    mod_rows: dict[tuple[int, int | None, str | None, str, str, str | None], None] = {}
    edges: list[tuple[int, int]] = []  # (span index, peptide index)
    enumerated: dict[tuple[str, bool, bool], list] = {}
    for (seq, pn, pc_, state), span_ids in groups.items():
        base = enumerated.get((seq, pn, pc_))
        if base is None:
            base = enumerated[(seq, pn, pc_)] = enumerate_modforms(
                seq, context=SiteContext(pn, pc_), return_origin=True, **kw
            )
        for form, placed in base:
            if state:
                form = replace(form, global_isotopes=state)
            key = canonical_peptide_key(format_proforma(form), vocab=vocab)
            k = index_of.get(key)
            if k is None:
                k = index_of[key] = len(forms)
                forms.append(form)
                keys.append(key)
            for pm in placed:
                position = pm.slot if isinstance(pm.slot, int) else None
                terminus = None if position is not None else ("N" if pm.slot == N_TERM else "C")
                mod_rows[(k, position, terminus, pm.key, pm.origin, rule_of.get((pm.origin, pm.site, pm.key)))] = None
            for iso in state:
                mod_rows[(k, None, None, iso, "label", ",".join(state))] = None
            for si in span_ids:
                edges.append((si, k))

    # ── precursors ───────────────────────────────────────────────────
    masses = peptide_masses(forms, vocab=vocab).numpy() if forms else np.empty(0)
    charges = np.asarray(sorted(set(int(z) for z in s.charges)), dtype=np.int64)
    mz = (masses[:, None] + charges[None, :] * PROTON_MASS) / charges[None, :]
    in_range = (mz >= s.min_mz) & (mz <= s.max_mz)
    counts["peptidoforms_enumerated"] = len(forms)
    counts["peptidoforms_out_of_mz_range"] = int((~in_range.any(axis=1)).sum())

    # target vs decoy: a peptidoform is a decoy when every span making it is
    # from a decoy protein (a sequence a target protein also makes is a target)
    from_target = np.zeros(len(forms), dtype=bool)
    decoy_set_of = np.full(len(forms), np.iinfo(np.int16).max, dtype=np.int64)
    for si, k in edges:
        p = proteins[span_rows[si][0]]
        if p.is_decoy:
            decoy_set_of[k] = min(decoy_set_of[k], p.decoy_set)
        else:
            from_target[k] = True
    # A reversed protein's peptide that is a target's with I and L made equal
    # has the target's masses and fragments: it is dropped, as a
    # peptide-level decoy colliding with a target is.
    target_il = {il_key(sp.sequence) for pi, sp in span_rows if not proteins[pi].is_decoy}
    collides = np.array(
        [not from_target[k] and il_key(forms[k].sequence) in target_il for k in range(len(forms))], dtype=bool
    )
    keep_peptide = in_range.any(axis=1) & ~collides

    # ── peptide-level decoys ─────────────────────────────────────────
    target_peptides = [k for k in range(len(forms)) if keep_peptide[k] and from_target[k]]
    decoy_peptides: list[tuple[Peptidoform, str, int, int, list[int]]] = []  # (form, key, target, set, order)
    peptide_sets = [ds for ds in s.decoys if ds.level == "peptide"]
    decoy_counts: list[dict[str, Any]] = []
    if peptide_sets:
        taken = {il_key(sp.sequence) for _pi, sp in span_rows}
        made_keys: set[str] = set(keys)
        for ds in peptide_sets:
            out = peptide_decoys(
                [forms[k] for k in target_peptides],
                ds,
                taken=taken,
                key=lambda f: canonical_peptide_key(format_proforma(f), vocab=vocab),
            )
            n_made = 0
            for ti, decoy, order in zip(out.target, out.decoy, out.order, strict=True):
                key = canonical_peptide_key(format_proforma(decoy), vocab=vocab)
                if key in made_keys:  # another set made it already
                    continue
                made_keys.add(key)
                decoy_peptides.append((decoy, key, target_peptides[ti], ds.index, order))
                n_made += 1
            decoy_counts.append(
                {"set": ds.index, "method": ds.method, "seed": ds.seed, "made": n_made,
                 "fell_back": out.n_fallback, "dropped": out.n_dropped + (len(out.decoy) - n_made)}
            )
    for ds in s.decoys:
        if ds.level == "protein":
            in_set = (~from_target) & (decoy_set_of == ds.index) & in_range.any(axis=1)
            decoy_counts.append(
                {"set": ds.index, "method": ds.method, "seed": ds.seed,
                 "made": int((in_set & keep_peptide).sum()), "dropped": int((in_set & collides).sum())}
            )
    # Peptide-level decoys on decoy proteins. Each decoy takes the spans of
    # the target form it was made from (on target proteins), with its own
    # sequence, on that protein's decoy of its set (DECOY_<accession>,
    # DECOY<n>_<accession>, as protein-level sets are named), so a
    # protein-level FDR has a decoy population of proteins with the targets'
    # structure: a peptide shared by two proteins gives a decoy shared by
    # their decoys. Built from ``decoy_of`` at construction, the use the PI
    # allows ("building or subsetting the population"); nothing downstream
    # pairs on it. Until 2026-11-10 no peptide-level decoy had a protein, and
    # one whose target was filtered away was lost to subset_proteins.
    target_spans: dict[int, list[int]] = {}
    for si, k in edges:
        if keep_peptide[k] and not proteins[span_rows[si][0]].is_decoy:
            target_spans.setdefault(k, []).append(si)
    decoy_protein_of: dict[tuple[int, int], int] = {}
    decoy_spans: dict[tuple[int, int, int], tuple[Any, str]] = {}  # (decoy protein, start, end) → (span, sequence)
    decoy_edges: dict[tuple[int, int, int, int], None] = {}  # (decoy protein, start, end, decoy j)
    for j, (form, _key, t, ds_index, _order) in enumerate(decoy_peptides):
        for si in dict.fromkeys(target_spans.get(t, [])):
            pi, sp = span_rows[si]
            dp = decoy_protein_of.get((pi, ds_index))
            if dp is None:
                dp = decoy_protein_of[(pi, ds_index)] = len(proteins)
                prefix = "DECOY_" if ds_index == 0 else f"DECOY{ds_index}_"
                proteins.append(_Protein(f"{prefix}{proteins[pi].accession}", None, "", [], True, ds_index,
                                         proteins[pi].states))  # fmt: skip
            decoy_spans.setdefault((dp, sp.start, sp.end), (sp, form.sequence))
            decoy_edges[(dp, sp.start, sp.end, j)] = None
    counts["decoy_proteins"] = len(proteins) - len(targets)
    _check_decoy_accessions(targets, proteins[len(targets):], counts)
    # Targets and decoys apart: the forms a reversed protein makes are decoys
    # too (until 2026-11-10 they were counted with the targets, and a
    # reverse-protein space showed no decoys at all).
    target_kept = keep_peptide & from_target
    protein_decoys = keep_peptide & ~from_target
    n_target_precursors = int(in_range[target_kept].sum())
    n_decoy_precursors = int(in_range[protein_decoys].sum()) + sum(
        int(in_range[t].sum()) for _f, _k, t, _s, _o in decoy_peptides
    )
    counts["peptidoforms"] = int(target_kept.sum())
    counts["precursors"] = n_target_precursors
    counts["decoy_peptidoforms"] = int(protein_decoys.sum()) + len(decoy_peptides)
    counts["decoy_precursors"] = n_decoy_precursors
    counts["decoy_sets"] = decoy_counts
    total_precursors = n_target_precursors + n_decoy_precursors
    if s.max_precursors is not None and total_precursors > s.max_precursors:
        raise SearchSpaceBudgetError(
            f"the search space would hold {total_precursors:,} precursors ({n_target_precursors:,} from "
            f"{counts['peptidoforms']:,} target peptidoforms, {n_decoy_precursors:,} decoys), over the budget "
            f"of {s.max_precursors:,}; narrow it or raise max_precursors"
        )
    if count_only:
        return counts

    # ── tables ───────────────────────────────────────────────────────
    protein_ids = content_ids([canonical_protein_key(p.accession) for p in proteins]).to_numpy()
    kept = np.flatnonzero(keep_peptide)
    new_index = np.full(len(forms), -1, dtype=np.int64)
    new_index[kept] = np.arange(kept.size)
    pep_keys = [keys[k] for k in kept] + [key for _f, key, _t, _s, _o in decoy_peptides]
    pep_forms = [forms[k] for k in kept] + [f for f, _k, _t, _s, _o in decoy_peptides]
    peptide_ids = content_ids(pep_keys).to_numpy()
    n_kept = kept.size
    target_id_of = {int(k): int(peptide_ids[new_index[k]]) for k in kept}
    is_decoy = np.concatenate([~from_target[kept], np.ones(len(decoy_peptides), dtype=bool)])
    decoy_of = [None] * n_kept + [target_id_of[t] for _f, _k, t, _s, _o in decoy_peptides]
    decoy_set = [
        (None if from_target[k] else int(decoy_set_of[k])) for k in kept
    ] + [ds for _f, _k, _t, ds, _o in decoy_peptides]
    peptides = pa.table(
        {
            "peptide_id": pa.array(peptide_ids, pa.int64()),
            "sequence": pa.array([f.sequence for f in pep_forms], pa.string()),
            "modified_sequence": pa.array(pep_keys, pa.string()),
            "is_decoy": pa.array(is_decoy, pa.bool_()),
            "decoy_of": pa.array(decoy_of, pa.int64()),
            "decoy_set": pa.array(decoy_set, pa.int16()),
        }
    )

    # precursors: kept targets at their charges, then decoys at their targets'
    rows_k = np.concatenate([np.repeat(np.arange(n_kept), in_range[kept].sum(axis=1))]) if n_kept else np.empty(0, np.int64)
    z_k = np.broadcast_to(charges, in_range[kept].shape)[in_range[kept]] if n_kept else np.empty(0, np.int64)
    mz_k = mz[kept][in_range[kept]] if n_kept else np.empty(0)
    d_rows, d_z, d_mz = [], [], []
    for j, (_f, _k, t, _s, _o) in enumerate(decoy_peptides):
        mask = in_range[t]
        d_rows.append(np.full(int(mask.sum()), n_kept + j, dtype=np.int64))
        d_z.append(charges[mask])
        d_mz.append(mz[t][mask])
    prec_rows = np.concatenate([rows_k, *d_rows]).astype(np.int64)
    prec_z = np.concatenate([z_k, *d_z]).astype(np.int64)
    prec_mz = np.concatenate([mz_k, *d_mz]).astype(np.float64)
    precursor_ids = content_ids([f"{pep_keys[r]}/{z}" for r, z in zip(prec_rows.tolist(), prec_z.tolist(), strict=True)])
    precursors = pa.table(
        {
            "precursor_id": precursor_ids,
            "peptide_id": pa.array(peptide_ids[prec_rows], pa.int64()),
            "charge": pa.array(prec_z, pa.int32()),
            "precursor_mz": pa.array(prec_mz, pa.float64()),
            "rt_predicted": pa.array(np.full(prec_rows.size, -1.0)),
            "ccs_predicted": pa.array(np.full(prec_rows.size, -1.0)),
        }
    )

    # spans that make a kept peptidoform, and their edges
    kept_edges = [(si, int(new_index[k])) for si, k in edges if keep_peptide[k]]
    used_spans = sorted({si for si, _ in kept_edges})
    # Target (and protein-level decoy) spans, then the peptide-level decoys'
    # spans on their decoy proteins: (protein index, span, its sequence).
    span_list = [(span_rows[si][0], span_rows[si][1], span_rows[si][1].sequence) for si in used_spans]
    span_list += [(dp, sp, sequence) for (dp, _s, _e), (sp, sequence) in decoy_spans.items()]
    peptide_spans = pa.table(
        {
            "protein_id": pa.array(protein_ids[[p for p, _, _ in span_list]] if span_list else [], pa.int64()),
            "start": pa.array([sp.start for _, sp, _ in span_list], pa.int32()),
            "end": pa.array([sp.end for _, sp, _ in span_list], pa.int32()),
            "sequence": pa.array([seq for _, _, seq in span_list], pa.string()),
            "missed_cleavages": pa.array([sp.n_missed for _, sp, _ in span_list], pa.int16()),
            "n_term": pa.array([sp.n_term for _, sp, _ in span_list], pa.string()),
            "c_term": pa.array([sp.c_term for _, sp, _ in span_list], pa.string()),
            "preceding": pa.array([sp.preceding for _, sp, _ in span_list], pa.string()),
            "following": pa.array([sp.following for _, sp, _ in span_list], pa.string()),
        }
    )
    # In key order, as Library.subset_precursors leaves them.
    peptide_spans = peptide_spans.take(
        pc.sort_indices(peptide_spans, sort_keys=[(k, "ascending") for k in ("protein_id", "start", "end")])
    )
    e_prot = [span_rows[si][0] for si, _ in kept_edges] + [dp for dp, _s, _e, _j in decoy_edges]
    e_start = [span_rows[si][1].start for si, _ in kept_edges] + [s_ for _dp, s_, _e, _j in decoy_edges]
    e_end = [span_rows[si][1].end for si, _ in kept_edges] + [e_ for _dp, _s, e_, _j in decoy_edges]
    e_pep = np.asarray([k for _, k in kept_edges] + [n_kept + j for _dp, _s, _e, j in decoy_edges], dtype=np.int64)
    span_peptide = pa.table(
        {
            "protein_id": pa.array(protein_ids[e_prot] if e_prot else [], pa.int64()),
            "start": pa.array(e_start, pa.int32()),
            "end": pa.array(e_end, pa.int32()),
            "peptide_id": pa.array(peptide_ids[e_pep] if e_prot else [], pa.int64()),
        }
    )
    pairs = list(dict.fromkeys(zip(span_peptide.column("protein_id").to_pylist(),
                                   span_peptide.column("peptide_id").to_pylist(), strict=True)))
    protein_peptide = pa.table(
        {
            "protein_id": pa.array([a for a, _ in pairs], pa.int64()),
            "peptide_id": pa.array([b for _, b in pairs], pa.int64()),
        }
    )

    # modifications: the kept targets', then each decoy's, its positions permuted
    by_peptide: dict[int, list[tuple]] = {}
    for row in mod_rows:
        by_peptide.setdefault(row[0], []).append(row)
    m_pep, m_pos, m_term, m_mod, m_origin, m_rule = [], [], [], [], [], []

    def emit(pid: int, rows: list[tuple], where: dict[int, int] | None) -> None:
        for _k, position, terminus, mod, origin, rule in rows:
            m_pep.append(pid)
            m_pos.append(where[position] if where is not None and position is not None else position)
            m_term.append(terminus)
            m_mod.append(mod)
            m_origin.append(origin)
            m_rule.append(rule)

    for k in kept:
        emit(int(peptide_ids[new_index[k]]), by_peptide.get(int(k), []), None)
    for j, (_f, _key, t, _s, order) in enumerate(decoy_peptides):
        where = {old: new for new, old in enumerate(order)}
        emit(int(peptide_ids[n_kept + j]), by_peptide.get(t, []), where)
    peptide_modifications = pa.table(
        {
            "peptide_id": pa.array(m_pep, pa.int64()),
            "position": pa.array(m_pos, pa.int32()),
            "terminus": pa.array(m_term, pa.string()),
            "modification": pa.array(m_mod, pa.string()),
            "origin": pa.array(m_origin, pa.string()),
            "rule": pa.array(m_rule, pa.string()),
        }
    )

    protein_table = pa.table(
        {
            "protein_id": pa.array(protein_ids, pa.int64()),
            "accession": pa.array([p.accession for p in proteins], pa.string()),
            "sequence": pa.array([p.sequence for p in proteins], pa.string()),
            "description": pa.array([p.description or None for p in proteins], pa.string()),
            "is_decoy": pa.array([p.is_decoy for p in proteins], pa.bool_()),
        }
    )
    counts["spans_used"] = len(used_spans)
    counts["decoy_spans"] = len(decoy_spans)
    metadata = {
        ID_SCHEME_METADATA_KEY: ID_SCHEME,
        SEARCH_SPACE_METADATA_KEY: {
            "settings": s.as_dict(),
            "inputs": inputs,
            "counts": counts,
            "vocab_size": len(vocab),
        },
    }
    return Library(
        proteins=protein_table,
        peptides=peptides,
        precursors=precursors,
        fragments=LIBRARY_FRAGMENT_TABLE.empty_table(),
        protein_peptide=protein_peptide,
        metadata_extras=metadata,
        peptide_spans=peptide_spans,
        span_peptide=span_peptide,
        peptide_modifications=peptide_modifications,
    )


def count_search_space(
    fastas: Sequence[str | Path], settings: SearchSpaceSettings | None = None, *, vocab: ModVocab = UNIMOD
) -> dict[str, Any]:
    """The counts per stage :func:`build_search_space` would give, with no
    tables built. Budgets apply."""
    out = build_search_space(fastas, settings, vocab=vocab, count_only=True)
    assert isinstance(out, dict)
    return out


def rules_from_mapping(
    fixed: Mapping[str, str] | None,
    variable: Mapping[str, str | Sequence[str]] | None,
    *,
    vocab: ModVocab = UNIMOD,
) -> tuple[ModRule, ...]:
    """Rules from ``enumerate_modforms``-style ``{site: key}`` mappings."""
    rules: list[ModRule] = []
    for site, key in (fixed or {}).items():
        mod = vocab[key]
        rules.append(ModRule(mod.id, (site,), "fixed", None, f"{key}@{site}"))
    for site, keys in (variable or {}).items():
        for key in (keys,) if isinstance(keys, str) else keys:
            mod = vocab[key]
            rules.append(ModRule(mod.id, (site,), "variable", None, f"{key}@{site}"))
    return tuple(rules)
