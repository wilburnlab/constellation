"""Physical-biochemistry constants and unit conversions.

Re-exports the fundamental constants from `core.chem.atoms` (single
source of truth — never redefined here) and adds the broader CODATA
2018 set plus common biochemistry conversions. All values are plain
Python floats; callers wrap in `torch.tensor(...)` at the call site
when they need autograd participation.

Categories:
    Mass / atomic           AVOGADRO, PROTON_MASS, NEUTRON_MASS,
                            ELECTRON_MASS, ISOTOPE_MASS_DIFF, DA_TO_KG
    Thermodynamic           BOLTZMANN_K_J_PER_K, GAS_CONSTANT_J_PER_MOL_K,
                            KT_298K_J, KT_298K_KCAL_PER_MOL
    Quantum / EM            PLANCK_H_J_S, LIGHT_SPEED_M_PER_S,
                            VACUUM_PERMITTIVITY_F_PER_M, ELEMENTARY_CHARGE_C
    Energy conversions      KCAL_PER_MOL_TO_KJ_PER_MOL, EV_TO_J
    Functions               ppm_to_da, da_to_ppm, k_to_kt_kcal_per_mol

Sources: NIST CODATA 2018 (https://physics.nist.gov/cuu/Constants/).
The four "exact" SI defining constants (k_B, h, c, e) are exact by
the 2019 SI redefinition; others carry CODATA uncertainties.
"""

from __future__ import annotations

import torch

# ──────────────────────────────────────────────────────────────────────
# Re-exports (must NOT redefine — single source of truth in core.chem)
# ──────────────────────────────────────────────────────────────────────

from constellation.core.chem.atoms import (  # noqa: F401  (re-exports)
    AVOGADRO,
    ELECTRON_MASS,
    ISOTOPE_MASS_DIFF,
    NEUTRON_MASS,
    PROTON_MASS,
)


# ──────────────────────────────────────────────────────────────────────
# CODATA 2018 — fundamental constants
# ──────────────────────────────────────────────────────────────────────

# Exact (SI defining constants, 2019 redefinition)
BOLTZMANN_K_J_PER_K: float = 1.380649e-23
PLANCK_H_J_S: float = 6.62607015e-34
LIGHT_SPEED_M_PER_S: float = 299792458.0
ELEMENTARY_CHARGE_C: float = 1.602176634e-19

# Derived / measured
GAS_CONSTANT_J_PER_MOL_K: float = 8.314462618  # R = N_A · k_B
VACUUM_PERMITTIVITY_F_PER_M: float = 8.8541878128e-12  # ε₀

# Convenience: thermal energy at 298.15 K (room temperature)
KT_298K_J: float = BOLTZMANN_K_J_PER_K * 298.15
KT_298K_KCAL_PER_MOL: float = (
    KT_298K_J * AVOGADRO / 4184.0
)  # ≈ 0.5925 kcal/mol


# ──────────────────────────────────────────────────────────────────────
# Unit conversions
# ──────────────────────────────────────────────────────────────────────

# Mass — Da (unified atomic mass unit) ↔ kg (CODATA 2018)
DA_TO_KG: float = 1.66053906660e-27
KG_TO_DA: float = 1.0 / DA_TO_KG

# Energy
KCAL_PER_MOL_TO_KJ_PER_MOL: float = 4.184
KJ_PER_MOL_TO_KCAL_PER_MOL: float = 1.0 / KCAL_PER_MOL_TO_KJ_PER_MOL

EV_TO_J: float = ELEMENTARY_CHARGE_C  # by definition: 1 eV = e · 1 V
J_TO_EV: float = 1.0 / EV_TO_J


# ──────────────────────────────────────────────────────────────────────
# Conversion functions
# ──────────────────────────────────────────────────────────────────────


def ppm_to_da(
    ppm: float | torch.Tensor, mz: float | torch.Tensor
) -> float | torch.Tensor:
    """Mass tolerance in Da given a ppm tolerance and an m/z reference.

    `Δm = ppm · m/z · 1e-6`. Works element-wise on tensors. Used at
    the boundary between MS data with ppm-quoted tolerances and
    physical Da-domain matching.
    """
    return ppm * mz * 1e-6


def da_to_ppm(
    da: float | torch.Tensor, mz: float | torch.Tensor
) -> float | torch.Tensor:
    """Inverse of `ppm_to_da` — convert an absolute Da tolerance to ppm
    relative to a reference m/z."""
    return da / mz * 1e6


def k_to_kt_kcal_per_mol(temperature_k: float | torch.Tensor) -> float | torch.Tensor:
    """Thermal energy `k_B · T` at temperature `T` (Kelvin), expressed
    in kcal/mol. Useful for free-energy comparisons in molecular
    thermodynamics."""
    return BOLTZMANN_K_J_PER_K * temperature_k * AVOGADRO / 4184.0
