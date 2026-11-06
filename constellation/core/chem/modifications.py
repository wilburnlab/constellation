"""UNIMOD modifications and the per-tool subset abstraction.

`Modification` is a named composition delta — a chemical entity. `ModVocab`
is a registry of them, with first-class subsetting so downstream tools
(spectral predictors, search engines, pipelines) can declare which mods
they accept *for their context*. The chemistry layer ships the
universe (`UNIMOD`); each tool ships its own enabled subset.

This is a deliberate departure from cartographer/Chronologer, where
`user_modifications.txt` served as both vocabulary loader AND implicit
enablement list. The two concerns are separated here:

    define a chemical entity   →  Modification + register_custom()
    declare what's enabled     →  ModVocab.subset()

The legacy tab-separated text format from cartographer is NOT carried
forward. If batch-loading custom mods from disk ever becomes a real
need, design a structured JSON/YAML schema then.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from constellation.core.chem.composition import Composition

# ──────────────────────────────────────────────────────────────────────
# Modification dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Specificity:
    """A (position, site) pair from UNIMOD's <umod:specificity> block.

    `position` is one of: "Anywhere", "Any N-term", "Any C-term",
    "Protein N-term", "Protein C-term".

    `site` is a single residue letter ("K", "S", ...) or one of the
    keywords "N-term" / "C-term" — the latter pair when the modification
    targets the terminus itself rather than a residue side chain.

    Together these distinguish chemically identical mass deltas at
    different molecular positions. The K[Acetyl]VERPD vs [Acetyl]-KVERPD
    case is exactly this distinction — both are 42.010565 Da, but one
    is a side-chain modification at K (specificity site="K",
    position="Anywhere") and the other is an N-terminal α-amine
    modification (site="N-term", position="Any N-term"). The reader
    boundary disambiguates by looking up which specificities the
    modification carries.
    """

    position: str
    site: str
    #: UNIMOD's ``hidden="1"``: a Mascot display flag, not a judgement on
    #: the chemistry (SILAC K/R and GlyGly-K are hidden).
    hidden: bool = False
    #: UNIMOD's classification of this placement ("Post-translational",
    #: "Artefact", "Chemical derivative", "Isotopic label", ...).
    classification: str = ""
    #: The neutral losses UNIMOD lists for this placement.
    neutral_losses: tuple["NeutralLoss", ...] = ()

    @property
    def is_protein_terminal(self) -> bool:
        """True for "Protein N-term" / "Protein C-term": the placement needs
        the protein's own terminus, not any peptide's."""
        return self.position in _PROTEIN_TERM_POSITIONS


@dataclass(frozen=True, slots=True)
class NeutralLoss:
    """One <umod:NeutralLoss> of a specificity. ``mono_mass == 0`` is
    UNIMOD's "no loss" entry, kept as UNIMOD lists it. ``flag`` is UNIMOD's
    attribute of that name (a loss to be expected alongside the intact
    fragment)."""

    mono_mass: float
    avge_mass: float = 0.0
    flag: bool = False
    composition: Composition | None = None


_N_TERM_POSITIONS = frozenset({"Any N-term", "Protein N-term"})
_C_TERM_POSITIONS = frozenset({"Any C-term", "Protein C-term"})
_PROTEIN_TERM_POSITIONS = frozenset({"Protein N-term", "Protein C-term"})

#: The site sentinels a placement can name besides a residue letter.
TERMINI = ("N-term", "C-term")


@dataclass(frozen=True, slots=True)
class Modification:
    """A named composition delta — e.g. UNIMOD:35 (Oxidation).

    `delta_composition` is the *light-atom skeleton*. For heavy-isotope
    mods (UNIMOD:737 TMT6plex/10plex, UNIMOD:259 SILAC ¹³C₆¹⁵N₂ lysine,
    UNIMOD:188 SILAC ¹³C₆), the skeleton's mass
    differs from the canonical mono mass by the heavy-isotope offset;
    `mass_override` then holds the canonical mass and
    `is_canonical_decomposition` returns False.

    `specificities` is the set of (position, site) pairs UNIMOD declares
    valid for this mod. Used by terminal-mod-aware readers (EncyclopeDIA,
    mzSpecLib) to decide whether `[+42]X` should serialize as `[Mod]-X`
    (N-terminal) or `X[Mod]` (residue side-chain). Empty tuple for
    custom mods that don't declare specificities.
    """

    id: str  # canonical "UNIMOD:35"
    name: str  # UNIMOD's full name (e.g. "Oxidation or Hydroxylation")
    delta_composition: Composition
    aliases: tuple[str, ...] = ()
    mass_override: float | None = None
    specificities: tuple[Specificity, ...] = ()
    description: str = ""
    #: The placements UNIMOD lists as hidden. Kept apart from
    #: ``specificities`` so that what the readers disambiguate with is
    #: unchanged; :attr:`all_specificities` has both.
    hidden_specificities: tuple[Specificity, ...] = ()

    @property
    def all_specificities(self) -> tuple[Specificity, ...]:
        """Every placement UNIMOD lists, shown and hidden."""
        return self.specificities + self.hidden_specificities

    def licenses(
        self,
        site: str,
        *,
        terminal_residue: str | None = None,
        protein_terminus: bool = False,
        include_hidden: bool = True,
    ) -> bool:
        """Whether UNIMOD lists a placement of this modification at ``site``.

        ``site`` is a residue letter (a side chain) or ``"N-term"`` /
        ``"C-term"`` (the terminus itself). For a terminus,
        ``terminal_residue`` is the residue there, which a terminal
        specificity may name ("Any N-term" on Q for pyro-Glu), and
        ``protein_terminus`` says whether it is the protein's own terminus:
        a "Protein N-term" / "Protein C-term" placement is licensed only
        there, an "Any" one anywhere. A side-chain site is licensed by an
        "Anywhere" placement on that residue; a terminal placement that
        names a residue is a terminal modification, asked about with
        ``site="N-term"`` / ``"C-term"`` and ``terminal_residue``.
        """
        specs = self.all_specificities if include_hidden else self.specificities
        if site in TERMINI:
            positions = _N_TERM_POSITIONS if site == "N-term" else _C_TERM_POSITIONS
            for spec in specs:
                if spec.position not in positions:
                    continue
                if spec.is_protein_terminal and not protein_terminus:
                    continue
                if spec.site == site or (terminal_residue is not None and spec.site == terminal_residue):
                    return True
            return False
        for spec in specs:
            if spec.site != site:
                continue
            if spec.position == "Anywhere":
                return True
        return False

    @property
    def delta_mass(self) -> float:
        """Canonical monoisotopic delta mass.

        Returns `mass_override` if set, else `delta_composition.mass`.
        """
        if self.mass_override is not None:
            return self.mass_override
        return self.delta_composition.mass

    @property
    def is_canonical_decomposition(self) -> bool:
        """True iff the delta's mass equals the light-skeleton's mass —
        i.e. the modification carries no heavy isotopes."""
        return self.mass_override is None

    @property
    def is_isotopic_substitution(self) -> bool:
        """True iff the modification adds and removes no atoms and only
        makes some of the residue's own atoms heavier (SILAC lysine and
        arginine, UNIMOD's ``Label:`` entries): a ``mass_override`` over an
        empty skeleton. Such a label leaves the residue's reactive groups
        free, so it can sit on a residue beside a chemical modification
        (heavy lysine, acetylated)."""
        return self.mass_override is not None and not self.delta_composition.atoms

    @property
    def has_n_term_specificity(self) -> bool:
        """True iff this mod has at least one N-terminal specificity
        (Any N-term or Protein N-term)."""
        return any(s.position in _N_TERM_POSITIONS for s in self.specificities)

    @property
    def has_c_term_specificity(self) -> bool:
        """True iff this mod has at least one C-terminal specificity."""
        return any(s.position in _C_TERM_POSITIONS for s in self.specificities)

    @property
    def has_only_terminal_specificity(self) -> bool:
        """True iff every declared specificity is terminal (N-term or
        C-term) — i.e. the mod cannot legitimately attach to a side chain.
        Returns False when `specificities` is empty (no claim either way)."""
        if not self.specificities:
            return False
        return all(
            s.position in _N_TERM_POSITIONS or s.position in _C_TERM_POSITIONS
            for s in self.specificities
        )

    def has_residue_specificity(self, residue: str) -> bool:
        """True iff this mod can attach to side-chain `residue` ("Anywhere"
        position with matching site)."""
        return any(
            s.position == "Anywhere" and s.site == residue
            for s in self.specificities
        )

    def has_terminal_specificity_for(self, terminus: str, residue: str) -> bool:
        """True iff this mod may sit on `terminus` when the terminal
        residue is `residue`.

        The site of a terminal specificity is load-bearing, so this is
        strictly narrower than :attr:`has_n_term_specificity`. UNIMOD:28
        (Gln->pyro-Glu) is "Any N-term" with site "Q" — licensed only
        when the peptide actually starts with Q. Asking merely whether
        the mod has *any* N-terminal specificity would license pyro-Glu
        on P.

        Site "N-term" / "C-term" means the mod targets the terminus
        itself (an alpha-amine acetyl, say), so it is residue-agnostic.

        `terminus` is "N-term" or "C-term".
        """
        if terminus == "N-term":
            positions, own_site = _N_TERM_POSITIONS, "N-term"
        elif terminus == "C-term":
            positions, own_site = _C_TERM_POSITIONS, "C-term"
        else:
            raise ValueError(
                f"terminus must be 'N-term' or 'C-term'; got {terminus!r}"
            )
        for spec in self.specificities:
            if spec.position not in positions:
                continue
            if spec.site == own_site or spec.site == residue:
                return True
        return False


# ──────────────────────────────────────────────────────────────────────
# ModVocab — registry with first-class subsetting
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ModVocab:
    """Registry of `Modification` objects, indexable by id or alias.

    ModVocab is mutable (you can `add` / `register_custom`), but the
    `subset` operation returns a fresh ModVocab — subsetting never
    mutates the source.
    """

    _by_id: dict[str, Modification] = field(default_factory=dict)
    _by_alias: dict[str, str] = field(default_factory=dict)  # alias → id

    # ── core registry ────────────────────────────────────────────
    def add(self, mod: Modification) -> None:
        if mod.id in self._by_id:
            raise KeyError(f"modification id already registered: {mod.id}")
        self._by_id[mod.id] = mod
        for a in mod.aliases:
            # Aliases may collide across mods (e.g. multiple TMT plexes
            # share a short label). Last-write-wins is the simplest
            # contract; tools that care should disambiguate by id.
            self._by_alias[a] = mod.id

    def get(self, key: str) -> Modification:
        if key in self._by_id:
            return self._by_id[key]
        if key in self._by_alias:
            return self._by_id[self._by_alias[key]]
        raise KeyError(f"unknown modification: {key!r}")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self._by_id or key in self._by_alias)

    def __getitem__(self, key: str) -> Modification:
        return self.get(key)

    def __iter__(self) -> Iterator[Modification]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id.keys())

    # ── subsetting (first-class) ─────────────────────────────────
    def subset(self, ids: Iterable[str]) -> "ModVocab":
        """New ModVocab containing only the listed mods (and their aliases).

        Accepts either canonical ids ("UNIMOD:35") or aliases ("Ox") —
        aliases are resolved through this vocab.
        """
        out = ModVocab()
        seen: set[str] = set()
        for key in ids:
            mod = self.get(key)
            if mod.id in seen:
                continue
            seen.add(mod.id)
            out.add(mod)
        return out

    def supports(self, key: str) -> bool:
        """Predicate for tool-side input validation."""
        return key in self

    def union(self, other: "ModVocab") -> "ModVocab":
        """New ModVocab with everything from both. On id collision,
        `self` wins (the receiver of the call)."""
        out = ModVocab()
        for mod in self:
            out.add(mod)
        for mod in other:
            if mod.id not in out._by_id:
                out.add(mod)
        return out

    # ── lookup helpers (used by IO translation layers) ───────────
    def find_by_mass(
        self, mass: float, tolerance_da: float = 1e-3
    ) -> tuple[Modification, ...]:
        """All modifications whose `delta_mass` matches `mass` within
        `tolerance_da`. Used by EncyclopeDIA / PeptideModSeq translation
        layers (`[+15.994]` → UNIMOD:35) — those layers live in
        `constellation.thirdparty.*`, not here."""
        return tuple(m for m in self if abs(m.delta_mass - mass) <= tolerance_da)

    # ── escape hatch for genuinely novel (non-UNIMOD) mods ───────
    def register_custom(
        self,
        id: str,
        name: str,
        delta_composition: Composition,
        mass_override: float | None = None,
        aliases: tuple[str, ...] = (),
        description: str = "",
    ) -> Modification:
        """Add a non-UNIMOD modification. Convention: `id` should start
        with `"CUSTOM:"` to make non-UNIMOD provenance obvious in logs.
        Typical use is in notebooks or per-project init scripts —
        constellation never auto-loads custom mods at import time."""
        mod = Modification(
            id=id,
            name=name,
            delta_composition=delta_composition,
            aliases=aliases,
            mass_override=mass_override,
            description=description,
        )
        self.add(mod)
        return mod

    # ── serialization ────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path) -> "ModVocab":
        """Load a ModVocab from a JSON file produced by `save()` or by
        `scripts/build-unimod-json.py`."""
        path = Path(path)
        with path.open() as f:
            doc = json.load(f)
        return cls._from_doc(doc)

    @classmethod
    def _from_doc(cls, doc: Mapping) -> "ModVocab":
        out = cls()
        for rec in doc["modifications"]:
            comp = Composition.from_dict(rec["delta_composition"])
            specs = [_specificity_from(s) for s in rec.get("specificities", ())]
            mod = Modification(
                id=rec["id"],
                name=rec.get("name", rec["id"]),
                delta_composition=comp,
                aliases=tuple(rec.get("aliases", ())),
                mass_override=rec.get("mass_override"),
                specificities=tuple(s for s in specs if not s.hidden),
                description=rec.get("description", ""),
                hidden_specificities=tuple(s for s in specs if s.hidden),
            )
            out.add(mod)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        doc = {
            "schema_version": "2",
            "source": {"generator": "ModVocab.save"},
            "modifications": [
                {
                    "id": m.id,
                    "name": m.name,
                    "delta_composition": m.delta_composition.atoms,
                    "mass_override": m.mass_override,
                    "aliases": list(m.aliases),
                    "specificities": [_specificity_doc(s) for s in m.all_specificities],
                    "description": m.description,
                }
                for m in self
            ],
        }
        path.write_text(json.dumps(doc, indent=1) + "\n")


def _specificity_from(rec: Mapping) -> Specificity:
    """A specificity record of a vocab JSON (schema 1 had only position and
    site, and listed shown placements only)."""
    return Specificity(
        position=rec["position"],
        site=rec["site"],
        hidden=bool(rec.get("hidden", False)),
        classification=rec.get("classification", ""),
        neutral_losses=tuple(
            NeutralLoss(
                mono_mass=float(loss["mono_mass"]),
                avge_mass=float(loss.get("avge_mass", 0.0)),
                flag=bool(loss.get("flag", False)),
                composition=Composition.from_dict(loss["composition"])
                if loss.get("composition")
                else None,
            )
            for loss in rec.get("neutral_losses", ())
        ),
    )


def _specificity_doc(s: Specificity) -> dict:
    return {
        "position": s.position,
        "site": s.site,
        "hidden": s.hidden,
        "classification": s.classification,
        "neutral_losses": [
            {
                "mono_mass": loss.mono_mass,
                "avge_mass": loss.avge_mass,
                "flag": loss.flag,
                "composition": loss.composition.atoms if loss.composition is not None else {},
            }
            for loss in s.neutral_losses
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Built-in UNIMOD vocabulary (loaded at import from packaged JSON)
# ──────────────────────────────────────────────────────────────────────


def _load_unimod() -> ModVocab:
    with resources.files("constellation.data").joinpath("unimod.json").open() as f:
        doc = json.load(f)
    return ModVocab._from_doc(doc)


UNIMOD: ModVocab = _load_unimod()


__all__ = ["TERMINI", "Modification", "ModVocab", "NeutralLoss", "Specificity", "UNIMOD"]
