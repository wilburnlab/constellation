"""Charge-dependent normalised-collision-energy adjustment (kept for callers).

New code uses :class:`~constellation.massspec.library.prediction.CollisionEnergy`,
which treats the energy as a property of each precursor charge and can
also take a table given charge by charge.

Until 2026-11-05 this module scaled an energy by ``f(z)`` alone, as if
every method's default charge were 1, and said EncyclopeDIA did the
same. EncyclopeDIA scales by ``f(z_ref) / f(z)``, ``z_ref`` the method's
default charge (3 in its library builds): at NCE 33 a 3+ precursor is
predicted at 33, where this module gave 28.05. :func:`adjust_nce` now
applies EncyclopeDIA's rule; libraries predicted before then were made
at the lower energies and record ``x.koina.adjust_nce_for_dia`` without
a reference charge.
"""

from __future__ import annotations

import numpy as np

from constellation.massspec.library.prediction.collision_energy import (
    DEFAULT_REFERENCE_CHARGE,
    OUTSIDE_FACTOR,
    SEARLE_2020_FACTORS,
    CollisionEnergy,
)

#: Searle et al. 2020, as EncyclopeDIA holds them. Charges above the
#: table take the last entry rather than an extrapolated factor.
NCE_CHARGE_FACTORS: dict[int, float] = dict(SEARLE_2020_FACTORS)


def nce_factor(charge: int) -> float:
    """The Searle 2020 factor of one precursor charge."""
    if charge < 1:
        raise ValueError(f"charge must be >= 1, got {charge}")
    return NCE_CHARGE_FACTORS.get(charge, OUTSIDE_FACTOR)


def adjust_nce(
    collision_energy: float | np.ndarray,
    charges: np.ndarray,
    *,
    enabled: bool = True,
    reference_charge: int = DEFAULT_REFERENCE_CHARGE,
) -> np.ndarray:
    """``collision_energy`` scaled to each charge by EncyclopeDIA's rule,
    ``f(reference_charge) / f(z)``, as float32 in the shape of ``charges``;
    with ``enabled=False`` the energy is broadcast unchanged. A per-
    precursor array of energies is scaled element by element."""
    charges = np.asarray(charges)
    energies = np.broadcast_to(np.asarray(collision_energy, dtype=np.float32), charges.shape)
    if not enabled:
        return energies.astype(np.float32)
    if np.ndim(collision_energy) == 0:
        return CollisionEnergy.scaled_from_reference(
            float(collision_energy), reference_charge
        ).resolve(charges)
    out = np.empty(charges.shape, dtype=np.float32)
    for index in np.ndindex(charges.shape):
        out[index] = CollisionEnergy.scaled_from_reference(
            float(energies[index]), reference_charge
        ).resolve(np.array([charges[index]]))[0]
    return out


__all__ = ["NCE_CHARGE_FACTORS", "adjust_nce", "nce_factor"]
