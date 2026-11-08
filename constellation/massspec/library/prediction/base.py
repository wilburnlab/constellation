"""The contract a predictor meets.

A *predictor* produces one *property* of precursors: ``fragments``
(fragment intensities), ``retention_time``, ``ccs`` or ``charge_state``
(the probability the peptide is seen at the precursor's charge). It is
asked about :class:`Requests`: unique (peptide, charge, collision energy)
triples, the peptide given as :class:`~constellation.massspec.tokenize.Sites`
with isotope labels already removed (``prediction.requests``). It answers
in two steps:

* ``check(requests)`` — which requests it cannot take and why, without
  running a model (a peptide its tokenizer refuses, a charge or energy
  outside its range). The orchestration (``prediction.run``) drops a
  precursor any requested predictor refuses, on its own, before anything
  is predicted;
* ``predict(requests)`` — the property for requests it accepted, as plain
  numpy arrays: :class:`FragmentArrays` or :class:`ValueArray`.

Outputs name fragments by ion type, ion *number* (the n of b_n / y_n) and
fragment charge; m/z, bond positions and precursor ids are the
orchestration's (``prediction.assemble``), from the source library's own
peptides. ``identity()`` says exactly which model ran (backend, model,
files and their SHA-256), for metadata and stage keys.

Koina, JChronologer and an in-house model meet the same contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from constellation.massspec.peptide.ions import IonType
from constellation.massspec.tokenize.vocabulary import Sites

__all__ = [
    "PROPERTIES",
    "Capabilities",
    "FragmentArrays",
    "Predictor",
    "Property",
    "Refusals",
    "Requests",
    "ValueArray",
    "merge_refusals",
    "sites_key",
]

Property = Literal["fragments", "retention_time", "ccs", "charge_state"]
PROPERTIES: tuple[Property, ...] = ("fragments", "retention_time", "ccs", "charge_state")

#: Refusal reasons a predictor gives beside the tokenizer's (``tokenize.REASONS``).
CHARGE_OUT_OF_RANGE = "charge_out_of_range"
COLLISION_ENERGY_OUT_OF_RANGE = "collision_energy_out_of_range"


@dataclass(frozen=True)
class Capabilities:
    """What the orchestration needs to know about a predictor."""

    property: Property
    #: The precursor charge is an input (fragments, CCS).
    needs_charge: bool = False
    #: The collision energy is an input (fragments of an energy-aware model).
    needs_collision_energy: bool = False
    #: Fragments: the ion series and highest fragment charge it predicts,
    #: and the neutral losses (none = intact ions only).
    ion_types: tuple[IonType, ...] = ()
    max_fragment_charge: int = 0
    losses: tuple[str, ...] = ()
    #: Fragments: the fragmentation it was trained on ("HCD").
    fragmentation: str | None = None
    #: Fragments: the highest ion number (the n of b_n) it predicts, None for
    #: every ion the peptide has (up to its length − 1).
    max_ion_number: int | None = None
    #: Fragments: the intensity below which it reports no ion, and whether
    #: an ion at exactly that value is reported (``"at_least"``) or not
    #: (``"above"``). An ion it covers and did not report was predicted
    #: below this floor, which is not the same as not predicted.
    min_intensity: float | None = None
    intensity_floor: Literal["at_least", "above"] | None = None
    #: Retention time: the scale its seconds are on. A model's scale, not a
    #: run's; mapping it onto a run is ``massspec.rt``'s.
    rt_scale: str | None = None

    def fragment_coverage(self) -> dict[str, Any]:
        """Which fragment ions this predictor speaks for, as plain data:
        an ion inside it with no predicted row was predicted below the
        floor (an abundance of 0), an ion outside it was never predicted
        (unknown). Fragment charges run to ``min(max_fragment_charge, z)``
        for a precursor of charge z."""
        return {
            "ion_types": [IonType(t).name.lower() for t in self.ion_types],
            "max_fragment_charge": self.max_fragment_charge,
            "fragment_charge_at_most_precursor": True,
            "max_ion_number": self.max_ion_number,
            "losses": list(self.losses),
            "min_intensity": self.min_intensity,
            "intensity_floor": self.intensity_floor,
        }


@dataclass(frozen=True)
class Requests:
    """Unique (peptide, charge, collision energy) requests.

    ``peptides`` holds each distinct peptide once (unlabelled sites);
    ``peptide[i]`` is request ``i``'s index into it. ``collision_energy``
    is NaN where no energy was set.
    """

    peptides: tuple[Sites, ...]
    peptide: np.ndarray
    charge: np.ndarray
    collision_energy: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.peptide)
        if len(self.charge) != n or len(self.collision_energy) != n:
            raise ValueError("peptide, charge and collision_energy must have one entry per request")

    def __len__(self) -> int:
        return len(self.peptide)

    def take(self, rows: np.ndarray) -> Requests:
        """Requests ``rows`` (the peptide tuple is shared)."""
        return Requests(self.peptides, self.peptide[rows], self.charge[rows], self.collision_energy[rows])

    def unique_peptides(self) -> tuple[np.ndarray, np.ndarray]:
        """(the distinct peptide indices these requests use, sorted; each
        request's position among them)."""
        return np.unique(self.peptide, return_inverse=True)


@dataclass(frozen=True)
class Refusals:
    """Per request: ``None`` if taken, else a reason and a detail."""

    reasons: tuple[str | None, ...]
    details: tuple[str | None, ...]

    @classmethod
    def none(cls, n: int) -> Refusals:
        return cls((None,) * n, (None,) * n)

    @property
    def refused(self) -> np.ndarray:
        return np.fromiter((r is not None for r in self.reasons), bool, len(self.reasons))


@dataclass(frozen=True)
class FragmentArrays:
    """Predicted fragments of requests, one entry per fragment.

    ``request`` indexes the requests that were predicted. ``loss`` is 0 for
    an intact ion and ``k`` for ``loss_ids[k - 1]``. ``mz_reported`` is the
    m/z a backend returned, if it returns one (checked against the
    peptide that was sent, never used as the library's m/z).
    """

    request: np.ndarray
    ion_type: np.ndarray
    ion_number: np.ndarray
    fragment_charge: np.ndarray
    intensity: np.ndarray
    loss: np.ndarray | None = None
    loss_ids: tuple[str, ...] = ()
    mz_reported: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.request)


@dataclass(frozen=True)
class ValueArray:
    """One value per predicted request (NaN where the model gave none):
    retention time in seconds, CCS in Å², or a charge probability."""

    values: np.ndarray
    unit: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Predictor(Protocol):
    """A model that predicts one property for requests."""

    name: str
    capabilities: Capabilities

    def identity(self) -> dict[str, Any]:
        """Which model this is, exactly: backend, model, files and SHA-256s."""
        ...

    def check(self, requests: Requests) -> Refusals:
        """Which requests it cannot take, without running the model."""
        ...

    def predict(self, requests: Requests) -> FragmentArrays | ValueArray:
        """The property for ``requests`` (all of which ``check`` took)."""
        ...


def sites_key(sites: Sites) -> str:
    """The canonical ProForma string of unlabelled sites: the sort key and
    the identity of a request's peptide."""
    head = "".join(f"[{a}]" for a in sites.n_term)
    body = "".join(r + "".join(f"[{a}]" for a in m) for r, m in zip(sites.residues, sites.residue_mods))
    tail = "".join(f"[{a}]" for a in sites.c_term)
    return (f"{head}-" if head else "") + body + (f"-{tail}" if tail else "")


def merge_refusals(n: int, *parts: Mapping[int, tuple[str, str]]) -> Refusals:
    """Refusals of ``n`` requests from ``{request: (reason, detail)}``
    maps, the first map to name a request winning."""
    reasons: list[str | None] = [None] * n
    details: list[str | None] = [None] * n
    for part in parts:
        for row, (reason, detail) in part.items():
            if reasons[row] is None:
                reasons[row], details[row] = reason, detail
    return Refusals(tuple(reasons), tuple(details))
