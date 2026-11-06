"""The collision energy a fragment predictor is run at, per precursor charge.

A fragment-intensity model takes a normalised collision energy (NCE) for
each precursor. On the instrument that energy depends on the charge: in a
DIA method every precursor is fragmented as if it had the method's
default charge, so a precursor of another charge receives a different
effective energy. EncyclopeDIA models this with one NCE and a per-charge
factor table (Searle et al. 2020): the effective energy of charge z is
``NCE · f(z_ref) / f(z)``, ``z_ref`` the method's default charge
(``PrositCSVWriter.convertNCE`` in 6.9.23, used by its JChronologer and
Koina library builds alike).

That rule is a first approximation. The energy is a property of each
charge, to be tuned empirically per charge and per instrument (PI,
2026-11-05), so :class:`CollisionEnergy` is a resolved ``charge → NCE``
table together with how it was made:

- :meth:`CollisionEnergy.uniform`: one energy for every charge.
- :meth:`CollisionEnergy.scaled_from_reference`: EncyclopeDIA's rule, in
  its float32 order of operations, so a library predicted here is
  predicted at the jar's energies to the last bit.
- :meth:`CollisionEnergy.explicit`: a table given charge by charge.

Every predicted library records the table it was predicted at
(:meth:`CollisionEnergy.describe`), whichever way it was given.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

#: Searle et al. 2020's per-charge factors, as EncyclopeDIA 6.9.23 holds
#: them (``PrositCSVWriter.getChargeFactor``, float32).
SEARLE_2020_FACTORS: Mapping[int, float] = MappingProxyType(
    {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75}
)

#: The factor EncyclopeDIA gives a charge outside its table (any charge
#: above 5): the last entry, never an extrapolation.
OUTSIDE_FACTOR = 0.75

#: EncyclopeDIA's default charge for its DIA library builds
#: (``-defaultCharge``, recorded as ``LibraryBuildDefaultCharge``).
DEFAULT_REFERENCE_CHARGE = 3

#: Charges :meth:`CollisionEnergy.describe` lists for a rule-based energy
#: when no charges are named: the precursor charges the models take.
DESCRIBED_CHARGES = (1, 2, 3, 4, 5, 6)

Origin = Literal["uniform", "scaled_from_reference", "explicit"]


def _positive(value: float, what: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be a positive finite number, got {value!r}")
    return value


def _charge(value: Any, what: str = "charge") -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{what} must be a whole number, got {value!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{what} must be at least 1, got {value}")
    return value


@dataclass(frozen=True)
class CollisionEnergy:
    """The NCE a fragment predictor is run at, as a function of charge.

    Build it with :meth:`uniform`, :meth:`scaled_from_reference` or
    :meth:`explicit`; the fields record how it was made."""

    origin: Origin
    #: The energy the rule starts from (``uniform``, ``scaled_from_reference``).
    nce: float | None = None
    #: The charge at which ``nce`` applies unchanged (``scaled_from_reference``).
    reference_charge: int | None = None
    #: The per-charge factors, sorted by charge (``scaled_from_reference``).
    factors: tuple[tuple[int, float], ...] = ()
    #: The factor of a charge outside ``factors`` (``scaled_from_reference``).
    outside_factor: float | None = None
    #: The energies, sorted by charge (``explicit``).
    table: tuple[tuple[int, float], ...] = ()

    # ── constructors ────────────────────────────────────────────────────

    @classmethod
    def uniform(cls, nce: float) -> CollisionEnergy:
        """One energy for every charge."""
        return cls(origin="uniform", nce=_positive(nce, "collision energy"))

    @classmethod
    def scaled_from_reference(
        cls,
        nce: float,
        reference_charge: int = DEFAULT_REFERENCE_CHARGE,
        *,
        factors: Mapping[int, float] = SEARLE_2020_FACTORS,
        outside_factor: float = OUTSIDE_FACTOR,
    ) -> CollisionEnergy:
        """``nce`` at ``reference_charge``, scaled to charge z by
        ``f(reference_charge) / f(z)``: EncyclopeDIA's DIA rule."""
        table = tuple(
            sorted((_charge(z, "factor charge"), _positive(f, f"factor of charge {z}"))
                   for z, f in factors.items())
        )
        if not table:
            raise ValueError("the factor table is empty")
        return cls(
            origin="scaled_from_reference",
            nce=_positive(nce, "collision energy"),
            reference_charge=_charge(reference_charge, "reference charge"),
            factors=table,
            outside_factor=_positive(outside_factor, "outside factor"),
        )

    @classmethod
    def explicit(cls, table: Mapping[int, float]) -> CollisionEnergy:
        """An energy given for each charge; any other charge is refused."""
        rows = tuple(
            sorted((_charge(z), _positive(e, f"collision energy of charge {z}"))
                   for z, e in table.items())
        )
        if not rows:
            raise ValueError("the collision-energy table is empty")
        if len({z for z, _ in rows}) != len(rows):
            raise ValueError("the collision-energy table names a charge twice")
        return cls(origin="explicit", table=rows)

    @classmethod
    def from_description(cls, description: Mapping[str, Any]) -> CollisionEnergy:
        """The inverse of :meth:`describe`."""
        origin = description.get("origin")
        if origin == "uniform":
            return cls.uniform(description["nce"])
        if origin == "scaled_from_reference":
            return cls.scaled_from_reference(
                description["nce"],
                description["reference_charge"],
                factors={int(z): f for z, f in description["factors"].items()},
                outside_factor=description["outside_factor"],
            )
        if origin == "explicit":
            return cls.explicit({int(z): e for z, e in description["table"].items()})
        raise ValueError(f"unknown collision-energy origin {origin!r}")

    # ── use ─────────────────────────────────────────────────────────────

    def resolve(self, charges: Any) -> np.ndarray:
        """The energy for each charge in ``charges``, as float32 in the
        shape of ``charges``. A charge an explicit table lacks is refused,
        and so is a charge below 1."""
        z = np.asarray(charges)
        flat = z.ravel()
        if flat.size and (flat.min() < 1 or not np.all(np.equal(np.mod(flat, 1), 0))):
            bad = flat[(flat < 1) | (np.mod(flat, 1) != 0)][0]
            raise ValueError(f"charges must be whole numbers of at least 1, got {bad!r}")
        distinct, where = np.unique(flat.astype(np.int64, copy=False), return_inverse=True)
        if self.origin == "uniform":
            per = np.full(distinct.shape, np.float32(self.nce), dtype=np.float32)
        elif self.origin == "scaled_from_reference":
            factors = dict(self.factors)
            # Java: (float nce * float f(zref)) / float f(z), rounded to
            # float at each step; numpy's float32 arithmetic is the same.
            numerator = np.float32(self.nce) * np.float32(
                factors.get(self.reference_charge, self.outside_factor)
            )
            f_z = np.array([factors.get(int(c), self.outside_factor) for c in distinct],
                           dtype=np.float32)
            per = numerator / f_z
        else:
            table = dict(self.table)
            missing = [int(c) for c in distinct if int(c) not in table]
            if missing:
                raise ValueError(
                    f"no collision energy for charge {', '.join(map(str, missing))} in the "
                    f"table given (charges {', '.join(str(c) for c, _ in self.table)})"
                )
            per = np.array([table[int(c)] for c in distinct], dtype=np.float32)
        return per[where].astype(np.float32, copy=False).reshape(z.shape)

    def by_charge(self, charges: Iterable[int] | None = None) -> dict[int, float]:
        """``{charge: energy}`` for ``charges`` (default: the explicit
        table's charges, or charges 1–6 for a rule), each the float32
        value :meth:`resolve` gives, widened exactly."""
        if charges is None:
            charges = [c for c, _ in self.table] if self.origin == "explicit" else DESCRIBED_CHARGES
        wanted = sorted({_charge(c) for c in charges})
        values = self.resolve(np.array(wanted, dtype=np.int64)) if wanted else []
        return {c: float(v) for c, v in zip(wanted, values, strict=True)}

    def describe(self, charges: Iterable[int] | None = None) -> dict[str, Any]:
        """JSON-ready: how the energy was made, and the table it gives for
        ``charges`` (see :meth:`by_charge`). Keys of the nested tables are
        charge strings, as JSON requires; :meth:`from_description` reads
        it back."""
        out: dict[str, Any] = {"origin": self.origin}
        if self.nce is not None:
            out["nce"] = self.nce
        if self.origin == "scaled_from_reference":
            out["reference_charge"] = self.reference_charge
            out["factors"] = {str(z): f for z, f in self.factors}
            out["outside_factor"] = self.outside_factor
        out["table"] = {str(z): e for z, e in self.by_charge(charges).items()}
        return out

    @property
    def label(self) -> str:
        """A short name for directory names and messages: the energy for
        a rule (``33``), ``by-charge`` for an explicit table."""
        return f"{self.nce:g}" if self.nce is not None else "by-charge"


def parse_energy_table(text: str) -> dict[int, float]:
    """``"1=28,2=31,3=33"`` → ``{1: 28.0, 2: 31.0, 3: 33.0}``, the
    command line's spelling of an explicit table."""
    table: dict[int, float] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        charge, sep, energy = token.partition("=")
        if not sep:
            raise ValueError(f"expected CHARGE=ENERGY, got {token!r}")
        try:
            z = _charge(int(charge.strip()))
            e = _positive(float(energy.strip()), f"collision energy of charge {z}")
        except ValueError as exc:
            raise ValueError(f"bad entry {token!r}: {exc}") from exc
        if z in table:
            raise ValueError(f"charge {z} is given twice")
        table[z] = e
    if not table:
        raise ValueError("the collision-energy table is empty")
    return table


__all__ = [
    "DEFAULT_REFERENCE_CHARGE",
    "OUTSIDE_FACTOR",
    "SEARLE_2020_FACTORS",
    "CollisionEnergy",
    "parse_energy_table",
]
