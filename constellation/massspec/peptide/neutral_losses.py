"""Neutral losses — characteristic small-molecule departures from fragment ions.

A `NeutralLoss` is a chemical delta (`Composition`) plus a biochemical-rule
predicate: which residues license the loss, which modifications license it,
and whether it's restricted to certain backbone ion types. Fragment ions
emit one of several losses during MS/MS, and library prediction + scoring
both need to know which losses are *biochemically possible* for each ion.

This module is the parallel of `core.sequence.protein.PROTEASES` —
`LOSS_REGISTRY` is loaded from `constellation/data/neutral_losses.json`
at import time, mirrors `ProteaseRegistry` / `ModVocab` in shape, and
exposes `register_custom` for novel losses (K-Ub LRGG remnant, glycopeptide
sugar-ladder losses, ...).

The applicability predicate `loss_applies` is intentionally separate from
the `NeutralLoss` data class so the dataclass stays JSON-round-trippable
and free of `IonType` imports. `IonType` is named here only as a string
("B", "Y", "C", ...) — `core.massspec.peptide.ions` resolves the name to
its enum value when checking validity during ladder generation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from constellation.core.chem.composition import Composition

# ──────────────────────────────────────────────────────────────────────
# NeutralLoss dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NeutralLoss:
    """A neutral-loss species — chemical delta plus the biochemistry that
    licenses it.

    Trigger semantics:
        - `triggering_residues` non-empty: any of the listed bare residue
          tokens must appear in the fragment.
        - `triggering_mods` non-empty: a modification with one of the listed
          UNIMOD ids must be on a residue in the fragment.
        - Both non-empty: the modification must sit on one of the listed
          residues (e.g. H3PO4 requires Phospho on S or T, not Y).
        - Both empty: loss is unconditional.

    `applies_to_ion_types` restricts the loss to specified backbone ion
    types (canonical names: "A", "B", "C", "X", "Y", "Z"). Empty set
    means "applies to any ion type".
    """

    id: str
    name: str
    delta_composition: Composition
    triggering_residues: frozenset[str] = field(default_factory=frozenset)
    triggering_mods: frozenset[str] = field(default_factory=frozenset)
    applies_to_ion_types: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    @property
    def delta_mass(self) -> float:
        """Monoisotopic mass of the loss (always positive — the *amount*
        subtracted from the parent ion's mass)."""
        return self.delta_composition.mass


# ──────────────────────────────────────────────────────────────────────
# LossRegistry — singleton + first-class subset
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class LossRegistry:
    """Indexable registry of `NeutralLoss` objects.

    Mirrors `ProteaseRegistry` and `ModVocab` in shape: load/save JSON,
    `register_custom` for typed escape hatch, lookup by id, raise on
    collisions. Subsetting via `subset(ids)` returns a fresh registry.
    """

    _by_id: dict[str, NeutralLoss] = field(default_factory=dict)

    def add(self, loss: NeutralLoss) -> None:
        if loss.id in self._by_id:
            raise KeyError(f"neutral-loss id already registered: {loss.id}")
        self._by_id[loss.id] = loss

    def get(self, key: str) -> NeutralLoss:
        try:
            return self._by_id[key]
        except KeyError:
            raise KeyError(f"unknown neutral loss: {key!r}") from None

    def __getitem__(self, key: str) -> NeutralLoss:
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._by_id

    def __iter__(self) -> Iterator[NeutralLoss]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id.keys())

    def subset(self, ids: Iterable[str]) -> "LossRegistry":
        out = LossRegistry()
        seen: set[str] = set()
        for key in ids:
            loss = self.get(key)
            if loss.id in seen:
                continue
            seen.add(loss.id)
            out.add(loss)
        return out

    def register_custom(
        self,
        id: str,
        delta_composition: Composition,
        *,
        name: str | None = None,
        triggering_residues: Iterable[str] = (),
        triggering_mods: Iterable[str] = (),
        applies_to_ion_types: Iterable[str] = (),
        description: str = "",
    ) -> NeutralLoss:
        loss = NeutralLoss(
            id=id,
            name=name or id,
            delta_composition=delta_composition,
            triggering_residues=frozenset(triggering_residues),
            triggering_mods=frozenset(triggering_mods),
            applies_to_ion_types=frozenset(applies_to_ion_types),
            description=description,
        )
        self.add(loss)
        return loss

    @classmethod
    def load(cls, path: str | Path) -> "LossRegistry":
        with Path(path).open() as f:
            doc = json.load(f)
        return cls._from_doc(doc)

    @classmethod
    def _from_doc(cls, doc: Mapping) -> "LossRegistry":
        out = cls()
        for rec in doc["neutral_losses"]:
            out.add(
                NeutralLoss(
                    id=rec["id"],
                    name=rec.get("name", rec["id"]),
                    delta_composition=Composition.from_dict(rec["delta_composition"]),
                    triggering_residues=frozenset(rec.get("triggering_residues", ())),
                    triggering_mods=frozenset(rec.get("triggering_mods", ())),
                    applies_to_ion_types=frozenset(rec.get("applies_to_ion_types", ())),
                    description=rec.get("description", ""),
                )
            )
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "source": {"generator": "LossRegistry.save"},
                    "neutral_losses": [
                        {
                            "id": loss.id,
                            "name": loss.name,
                            "delta_composition": loss.delta_composition.atoms,
                            "triggering_residues": sorted(loss.triggering_residues),
                            "triggering_mods": sorted(loss.triggering_mods),
                            "applies_to_ion_types": sorted(loss.applies_to_ion_types),
                            "description": loss.description,
                        }
                        for loss in self
                    ],
                },
                indent=2,
            )
            + "\n"
        )


def _load_neutral_losses() -> LossRegistry:
    with resources.files("constellation.data").joinpath("neutral_losses.json").open() as f:
        doc = json.load(f)
    return LossRegistry._from_doc(doc)


LOSS_REGISTRY: LossRegistry = _load_neutral_losses()


# ──────────────────────────────────────────────────────────────────────
# Applicability predicate
# ──────────────────────────────────────────────────────────────────────


def loss_applies(
    loss: NeutralLoss,
    *,
    ion_type_name: str,
    fragment_residues: Sequence[str],
    fragment_mods: Mapping[int, object],
) -> bool:
    """True iff `loss` is biochemically valid for the given fragment.

    `ion_type_name` is the canonical name of the IonType ("B", "Y", ...).
    `fragment_residues` is the bare-residue token sequence inside the
    fragment (e.g. ["P", "E", "P"] for b3 of PEPTIDE).
    `fragment_mods` is ``{fragment_relative_index: mod_id_or_list}`` where
    each value is a canonical modification id (e.g. ``"UNIMOD:21"``) — the
    ladder driver in ``massspec.peptide.ions`` builds this from
    ``Peptidoform`` ``TaggedMod``s by resolving each to its UNIMOD id.
    Mods that don't resolve (bare mass deltas, glycan payloads) are simply
    omitted; they can't trigger UNIMOD-keyed losses by definition.
    """
    if loss.applies_to_ion_types and ion_type_name not in loss.applies_to_ion_types:
        return False

    has_required_residue = (
        not loss.triggering_residues
        or any(r in loss.triggering_residues for r in fragment_residues)
    )
    if not loss.triggering_mods:
        return has_required_residue

    # Mod required. If residues are also restricted, the mod must sit
    # *on* one of those residues; otherwise any presence of the mod in
    # the fragment is enough.
    for pos, raw in fragment_mods.items():
        candidates = raw if isinstance(raw, list) else [raw]
        for m in candidates:
            if not isinstance(m, str) or m not in loss.triggering_mods:
                continue
            if not loss.triggering_residues:
                return True
            if 0 <= pos < len(fragment_residues) and fragment_residues[pos] in loss.triggering_residues:
                return True
    return False


__all__ = [
    "NeutralLoss",
    "LossRegistry",
    "LOSS_REGISTRY",
    "loss_applies",
]
