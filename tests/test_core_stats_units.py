"""Tests for core.stats.units."""

from __future__ import annotations

import math

import pytest

from constellation.core.chem import elements as chem_elements
from constellation.core.stats import units


# ──────────────────────────────────────────────────────────────────────
# Re-export identity (no shadowing of core.chem.elements constants)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["AVOGADRO", "PROTON_MASS", "ELECTRON_MASS", "NEUTRON_MASS", "ISOTOPE_MASS_DIFF"],
)
def test_reexport_is_identity(name):
    """Re-exports must be the same object as in core.chem.elements — proves
    we don't have a shadow definition that could drift."""
    assert getattr(units, name) is getattr(chem_elements, name)


# ──────────────────────────────────────────────────────────────────────
# CODATA 2018 constants — sanity values
# ──────────────────────────────────────────────────────────────────────


def test_boltzmann_exact():
    assert units.BOLTZMANN_K_J_PER_K == 1.380649e-23


def test_planck_exact():
    assert units.PLANCK_H_J_S == 6.62607015e-34


def test_light_speed_exact():
    assert units.LIGHT_SPEED_M_PER_S == 299792458.0


def test_elementary_charge_exact():
    assert units.ELEMENTARY_CHARGE_C == 1.602176634e-19


def test_gas_constant_value():
    # R = N_A · k_B should round-trip to the same value
    derived = units.AVOGADRO * units.BOLTZMANN_K_J_PER_K
    assert math.isclose(derived, units.GAS_CONSTANT_J_PER_MOL_K, rel_tol=1e-9)


def test_kt_298k_kcal_per_mol_value():
    # k_B · T at 298.15 K ≈ 0.5925 kcal/mol — standard biochem reference
    assert math.isclose(units.KT_298K_KCAL_PER_MOL, 0.5925, abs_tol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Conversions
# ──────────────────────────────────────────────────────────────────────


def test_da_kg_roundtrip():
    assert math.isclose(units.DA_TO_KG * units.KG_TO_DA, 1.0, rel_tol=1e-15)


def test_kcal_kj_per_mol_roundtrip():
    assert math.isclose(
        units.KCAL_PER_MOL_TO_KJ_PER_MOL * units.KJ_PER_MOL_TO_KCAL_PER_MOL,
        1.0,
        rel_tol=1e-15,
    )


def test_ev_j_roundtrip():
    assert math.isclose(units.EV_TO_J * units.J_TO_EV, 1.0, rel_tol=1e-15)


def test_ev_to_j_equals_elementary_charge():
    assert units.EV_TO_J == units.ELEMENTARY_CHARGE_C


def test_kcal_kj_value():
    assert units.KCAL_PER_MOL_TO_KJ_PER_MOL == 4.184


# ──────────────────────────────────────────────────────────────────────
# Conversion functions
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ppm,mz,expected_da",
    [
        (20.0, 1000.0, 0.02),
        (5.0, 500.0, 0.0025),
        (10.0, 2000.0, 0.02),
        (0.0, 1000.0, 0.0),
    ],
)
def test_ppm_to_da_known(ppm, mz, expected_da):
    assert math.isclose(units.ppm_to_da(ppm, mz), expected_da, rel_tol=1e-12)


def test_ppm_da_roundtrip():
    mz = 1234.5678
    ppm = 7.5
    da = units.ppm_to_da(ppm, mz)
    ppm_back = units.da_to_ppm(da, mz)
    assert math.isclose(ppm_back, ppm, rel_tol=1e-12)


def test_k_to_kt_kcal_per_mol_at_298k():
    # Should match the cached KT_298K_KCAL_PER_MOL
    assert math.isclose(
        units.k_to_kt_kcal_per_mol(298.15),
        units.KT_298K_KCAL_PER_MOL,
        rel_tol=1e-12,
    )


def test_k_to_kt_doubles_at_double_temperature():
    base = units.k_to_kt_kcal_per_mol(298.15)
    doubled = units.k_to_kt_kcal_per_mol(596.30)
    assert math.isclose(doubled / base, 2.0, rel_tol=1e-12)
