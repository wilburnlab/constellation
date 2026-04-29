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


_N_TERM_POSITIONS = frozenset({"Any N-term", "Protein N-term"})
_C_TERM_POSITIONS = frozenset({"Any C-term", "Protein C-term"})


@dataclass(frozen=True, slots=True)
class Modification:
    """A named composition delta — e.g. UNIMOD:35 (Oxidation).

    `delta_composition` is the *light-atom skeleton*. For heavy-isotope
    mods (UNIMOD:737 TMT10, UNIMOD:259 SILAC ¹³C₆), the skeleton's mass
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
    name: str  # full UNIMOD name (e.g. "Oxidation")
    delta_composition: Composition
    aliases: tuple[str, ...] = ()
    mass_override: float | None = None
    specificities: tuple[Specificity, ...] = ()
    description: str = ""

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
            specs = tuple(
                Specificity(position=s["position"], site=s["site"])
                for s in rec.get("specificities", ())
            )
            mod = Modification(
                id=rec["id"],
                name=rec.get("name", rec["id"]),
                delta_composition=comp,
                aliases=tuple(rec.get("aliases", ())),
                mass_override=rec.get("mass_override"),
                specificities=specs,
                description=rec.get("description", ""),
            )
            out.add(mod)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        doc = {
            "schema_version": "1",
            "source": {"generator": "ModVocab.save"},
            "modifications": [
                {
                    "id": m.id,
                    "name": m.name,
                    "delta_composition": m.delta_composition.atoms,
                    "mass_override": m.mass_override,
                    "aliases": list(m.aliases),
                    "specificities": [
                        {"position": s.position, "site": s.site}
                        for s in m.specificities
                    ],
                    "description": m.description,
                }
                for m in self
            ],
        }
        path.write_text(json.dumps(doc, indent=1) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Built-in UNIMOD vocabulary (loaded at import from packaged JSON)
# ──────────────────────────────────────────────────────────────────────


def _load_unimod() -> ModVocab:
    with resources.files("constellation.data").joinpath("unimod.json").open() as f:
        doc = json.load(f)
    return ModVocab._from_doc(doc)


UNIMOD: ModVocab = _load_unimod()


__all__ = ["Modification", "ModVocab", "Specificity", "UNIMOD"]
