"""Charge-dependent normalised-collision-energy adjustment.

Instrument NCE is calibrated at a reference charge, so the *effective*
energy a precursor experiences falls as its charge rises. Searle et al.
(2020) published the per-charge factors below; EncyclopeDIA applies the
same correction by default when building DIA libraries, which is what
``--no-adjust-nce-for-dia`` disables. Cartographer's Koina integration
carries an identical table, so predictions stay comparable across all
three code paths.
"""

from __future__ import annotations

import numpy as np

#: Searle et al. 2020. Charges above the table clamp to the last entry
#: rather than extrapolating a factor nobody measured.
NCE_CHARGE_FACTORS: dict[int, float] = {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75}

_MAX_TABULATED = max(NCE_CHARGE_FACTORS)


def nce_factor(charge: int) -> float:
    """Multiplicative NCE factor for one precursor charge."""
    if charge < 1:
        raise ValueError(f"charge must be >= 1, got {charge}")
    return NCE_CHARGE_FACTORS[min(charge, _MAX_TABULATED)]


def adjust_nce(
    collision_energy: float | np.ndarray,
    charges: np.ndarray,
    *,
    enabled: bool = True,
) -> np.ndarray:
    """Apply the per-charge factor to a scalar or per-precursor energy.

    With ``enabled=False`` the energy is broadcast unchanged, so callers
    need no branch of their own.
    """
    charges = np.asarray(charges)
    energies = np.broadcast_to(
        np.asarray(collision_energy, dtype=np.float32), charges.shape
    ).astype(np.float32)
    if not enabled:
        return energies
    factors = np.array(
        [nce_factor(int(z)) for z in charges.ravel()], dtype=np.float32
    ).reshape(charges.shape)
    return energies * factors


__all__ = ["NCE_CHARGE_FACTORS", "adjust_nce", "nce_factor"]
