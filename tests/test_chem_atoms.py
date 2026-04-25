"""Tests for constellation.core.chem.atoms — periodic-table data integrity."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.chem.atoms import (
    ATOM_SYMBOLS,
    ATOM_TYPES,
    ATOMS,
    FEATURE_COLUMNS,
    ISOTOPE_MASS_DIFF,
    MASSES,
    PROTON_MASS,
    WEIGHTS,
    Atom,
    AtomTable,
    Isotope,
)


# ──────────────────────────────────────────────────────────────────────
# Coverage and structure
# ──────────────────────────────────────────────────────────────────────


def test_full_periodic_table_loaded():
    assert len(ATOMS) == 118
    assert len(ATOM_SYMBOLS) == 118
    # Every atomic number 1..118 present.
    zs = sorted(a.atomic_number for a in ATOMS)
    assert zs == list(range(1, 119))


def test_dual_indexing():
    """ATOMS supports both symbol and atomic-number lookup."""
    assert ATOMS["C"].atomic_number == 6
    assert ATOMS[6].symbol == "C"
    assert ATOMS["U"].atomic_number == 92


def test_symbols_ordered_by_atomic_number():
    for i, sym in enumerate(ATOM_SYMBOLS):
        assert ATOMS[sym].atomic_number == i + 1


def test_chnops_legacy_alias_present():
    """ATOM_TYPES is the cartographer-equivalent CHNOPS-6 view."""
    assert ATOM_TYPES == ("C", "H", "N", "O", "S", "P")
    for s in ATOM_TYPES:
        assert s in ATOMS


# ──────────────────────────────────────────────────────────────────────
# Cartographer cross-check — the load-bearing values must match
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "symbol,expected,tol",
    [
        ("H", 1.00782503223, 1e-7),
        ("C", 12.0, 0.0),
        ("N", 14.00307400443, 1e-7),
        ("O", 15.99491461957, 1e-7),
        ("P", 30.97376199842, 1e-7),
        (
            "S",
            31.9720711744,
            1e-6,
        ),  # cartographer used 31.9720707; NIST AME2020 is more precise
        ("F", 18.99840316273, 1e-7),
        ("Cl", 34.968852682, 1e-6),
    ],
)
def test_monoisotopic_masses_match_nist(symbol: str, expected: float, tol: float):
    got = ATOMS[symbol].monoisotopic_mass
    assert abs(got - expected) <= tol, f"{symbol}: got {got}, expected {expected}"


def test_proton_mass_constant():
    # CODATA 2018 value to the precision cartographer uses.
    assert abs(PROTON_MASS - 1.00727646688) < 1e-11


def test_isotope_mass_diff_is_13c_minus_12c():
    """The conventional ¹³C-spacing constant should equal ¹³C exact mass - 12.0."""
    c12 = next(i for i in ATOMS["C"].isotopes if i.mass_number == 12)
    c13 = next(i for i in ATOMS["C"].isotopes if i.mass_number == 13)
    assert abs((c13.exact_mass - c12.exact_mass) - ISOTOPE_MASS_DIFF) < 1e-7


# ──────────────────────────────────────────────────────────────────────
# Isotope data integrity
# ──────────────────────────────────────────────────────────────────────


def test_natural_abundances_sum_to_one_for_natural_elements():
    """For elements with naturally-occurring isotopes, abundances should
    sum to 1.0 (within float tolerance). Radioactive-only elements
    (Tc, Pm, all transuranics) sum to 0."""
    for atom in ATOMS:
        total = sum(i.abundance for i in atom.isotopes)
        if atom.atomic_number in (43, 61) or atom.atomic_number >= 84:
            # Radioactive-only or trace — abundances sum to 0 or close to it.
            assert total <= 1.0 + 1e-6
        else:
            assert abs(total - 1.0) < 1e-6, f"{atom.symbol}: sum={total}"


def test_carbon_isotopes_present():
    """¹²C and ¹³C must both be present and in correct abundance ratio."""
    isotopes = {i.mass_number: i for i in ATOMS["C"].isotopes}
    assert 12 in isotopes and 13 in isotopes
    assert isotopes[12].exact_mass == 12.0  # ¹²C is the mass standard
    assert abs(isotopes[12].abundance - 0.9893) < 1e-4
    assert abs(isotopes[13].abundance - 0.0107) < 1e-4


def test_15n_distinct_from_13c_spacing():
    """Regression test for the isotope-resolution concern: ¹⁵N must NOT
    be coincident with ¹³C-spacing. The mass difference (¹⁵N-¹⁴N) -
    (¹³C-¹²C) should be ≈ -0.00633 Da."""
    n14 = next(i for i in ATOMS["N"].isotopes if i.mass_number == 14)
    n15 = next(i for i in ATOMS["N"].isotopes if i.mass_number == 15)
    c12 = next(i for i in ATOMS["C"].isotopes if i.mass_number == 12)
    c13 = next(i for i in ATOMS["C"].isotopes if i.mass_number == 13)
    n_diff = n15.exact_mass - n14.exact_mass
    c_diff = c13.exact_mass - c12.exact_mass
    assert abs((n_diff - c_diff) - (-0.00633)) < 1e-4


# ──────────────────────────────────────────────────────────────────────
# Derived MASSES / WEIGHTS dict views — pinning test
# ──────────────────────────────────────────────────────────────────────


def test_masses_pinned_to_atoms():
    """MASSES is a view, never a parallel copy — must agree with ATOMS for every symbol."""
    for s in ATOM_SYMBOLS:
        assert MASSES[s] == ATOMS[s].monoisotopic_mass


def test_weights_pinned_to_atoms():
    for s in ATOM_SYMBOLS:
        assert WEIGHTS[s] == ATOMS[s].standard_atomic_weight


# ──────────────────────────────────────────────────────────────────────
# Tensor views
# ──────────────────────────────────────────────────────────────────────


def test_mass_tensor_indexed_by_composition_axis():
    """mass_tensor[ATOMS.index(s)] should equal ATOMS[s].monoisotopic_mass."""
    mt = ATOMS.mass_tensor
    assert mt.shape == (len(ATOM_SYMBOLS),)
    assert mt.dtype == torch.float64
    for s in ATOM_SYMBOLS:
        assert mt[ATOMS.index(s)].item() == ATOMS[s].monoisotopic_mass


def test_feature_tensor_z_indexed():
    """feature_tensor row Z holds the features for atomic_number=Z. Row 0 is all-NaN padding."""
    ft = ATOMS.feature_tensor
    assert ft.shape == (119, len(FEATURE_COLUMNS))  # 0..118
    assert ft.dtype == torch.float32
    # Padding row.
    assert torch.isnan(ft[0]).all()
    # Carbon (Z=6).
    z_col = FEATURE_COLUMNS.index("atomic_number")
    mass_col = FEATURE_COLUMNS.index("monoisotopic_mass")
    assert ft[6, z_col].item() == 6.0
    assert ft[6, mass_col].item() == 12.0


def test_feature_tensor_columns_in_documented_order():
    expected = (
        "atomic_number",
        "monoisotopic_mass",
        "standard_atomic_weight",
        "pauling_electronegativity",
        "covalent_radius_pm",
        "vdw_radius_pm",
        "valence_electrons",
        "group",
        "period",
    )
    assert FEATURE_COLUMNS == expected


# ──────────────────────────────────────────────────────────────────────
# AtomTable error semantics
# ──────────────────────────────────────────────────────────────────────


def test_unknown_symbol_raises_keyerror():
    with pytest.raises(KeyError):
        ATOMS["Unobtainium"]
    with pytest.raises(KeyError):
        ATOMS[200]


def test_index_inverse_of_symbols():
    for i, sym in enumerate(ATOM_SYMBOLS):
        assert ATOMS.index(sym) == i


# ──────────────────────────────────────────────────────────────────────
# Atom dataclass behavior
# ──────────────────────────────────────────────────────────────────────


def test_atom_is_frozen():
    c = ATOMS["C"]
    with pytest.raises((AttributeError, TypeError)):
        c.symbol = "X"  # type: ignore[misc]


def test_atom_average_mass_uses_standard_weight():
    """For natural-abundance elements, .average_mass returns the IUPAC
    standard atomic weight."""
    for s in ("C", "H", "N", "O", "S", "P"):
        atom = ATOMS[s]
        assert atom.average_mass == atom.standard_atomic_weight


def test_isotope_dataclass_immutable():
    iso = Isotope(mass_number=12, exact_mass=12.0, abundance=0.9893)
    with pytest.raises((AttributeError, TypeError)):
        iso.abundance = 0.5  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Group / period assignment
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "symbol,group,period",
    [
        ("H", 1, 1),
        ("He", 18, 1),
        ("Li", 1, 2),
        ("C", 14, 2),
        ("F", 17, 2),
        ("Ne", 18, 2),
        ("Na", 1, 3),
        ("S", 16, 3),
        ("Cl", 17, 3),
        ("Ar", 18, 3),
        ("K", 1, 4),
        ("Fe", 8, 4),
        ("Br", 17, 4),
        ("Cs", 1, 6),
        ("Hf", 4, 6),
        ("Au", 11, 6),
        ("I", 17, 5),
        ("Fr", 1, 7),
        ("U", -1, 7),  # actinide
        ("La", -1, 6),  # lanthanide
    ],
)
def test_group_period(symbol: str, group: int, period: int):
    a = ATOMS[symbol]
    assert (a.group, a.period) == (group, period)
