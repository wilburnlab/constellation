"""Tests for constellation.core.chem.composition."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.chem.atoms import ATOM_SYMBOLS
from constellation.core.chem.composition import (
    Composition,
    batched_average_mass,
    batched_mass,
    parse_formula,
    stack,
)


# ──────────────────────────────────────────────────────────────────────
# parse_formula
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("H2O", {"H": 2, "O": 1}),
        ("C6H12O6", {"C": 6, "H": 12, "O": 6}),
        ("CO2", {"C": 1, "O": 2}),
        ("NaCl", {"Na": 1, "Cl": 1}),
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("Fe(CN)6", {"Fe": 1, "C": 6, "N": 6}),
        ("", {}),
    ],
)
def test_parse_formula(formula: str, expected: dict[str, int]):
    assert parse_formula(formula) == expected


# ──────────────────────────────────────────────────────────────────────
# Composition construction
# ──────────────────────────────────────────────────────────────────────


def test_zeros():
    z = Composition.zeros()
    assert z.total_atoms == 0
    assert z.formula == ""
    assert z.atoms == {}


def test_from_dict_and_atoms_view_are_inverse():
    spec = {"C": 6, "H": 12, "O": 6}
    c = Composition.from_dict(spec)
    assert c.atoms == spec


def test_from_formula_parses_hill():
    c = Composition.from_formula("C6H12O6")
    assert c.atoms == {"C": 6, "H": 12, "O": 6}


def test_unknown_symbol_raises():
    with pytest.raises(KeyError):
        Composition.from_dict({"Xx": 1})


# ──────────────────────────────────────────────────────────────────────
# Mass / formula / Hill notation
# ──────────────────────────────────────────────────────────────────────


def test_water_mass_matches_cartographer():
    """Cartographer's masses['H2O'] derived from H/O monoisotopic masses."""
    h2o = Composition.from_formula("H2O").mass
    expected = 2 * 1.00782503223 + 15.99491461957
    assert abs(h2o - expected) < 1e-9


def test_glucose_mass_matches_published():
    """Glucose monoisotopic mass: 180.0633881..."""
    g = Composition.from_formula("C6H12O6").mass
    assert abs(g - 180.063388) < 1e-5


def test_formula_uses_hill_order_with_carbon():
    """C first, H second, then alphabetical."""
    c = Composition.from_dict({"O": 6, "H": 12, "C": 6})
    assert c.formula == "C6H12O6"


def test_formula_alphabetical_when_no_carbon():
    c = Composition.from_dict({"O": 1, "H": 2})
    # No carbon → alphabetical: H before O
    assert c.formula == "H2O"


def test_formula_omits_count_of_one():
    c = Composition.from_dict({"Na": 1, "Cl": 1})
    assert c.formula == "ClNa"  # no carbon, alphabetical: Cl, Na


def test_average_mass_for_water():
    """Standard atomic weight: H=1.008, O=15.999 → H2O ≈ 18.015."""
    avg = Composition.from_formula("H2O").average_mass
    assert abs(avg - 18.015) < 0.01


# ──────────────────────────────────────────────────────────────────────
# Arithmetic operators
# ──────────────────────────────────────────────────────────────────────


def test_add_compositions():
    h2 = Composition.from_dict({"H": 2})
    o = Composition.from_dict({"O": 1})
    h2o = h2 + o
    assert h2o == Composition.from_formula("H2O")


def test_sub_can_yield_negative_counts():
    """Modification deltas may be negative — e.g. dehydration."""
    h2o = Composition.from_formula("H2O")
    h2 = Composition.from_dict({"H": 2})
    delta = h2 - h2o
    assert delta.atoms == {"O": -1}
    assert not delta.is_physical()
    assert h2o.is_physical()


def test_mul_by_integer():
    h2 = Composition.from_dict({"H": 2})
    h6 = h2 * 3
    assert h6.atoms == {"H": 6}
    h6_rmul = 3 * h2
    assert h6_rmul == h6


def test_subtract_back_to_zero():
    g = Composition.from_formula("C6H12O6")
    assert g - g == Composition.zeros()


# ──────────────────────────────────────────────────────────────────────
# Equality / hashing — load-bearing for dict-key use
# ──────────────────────────────────────────────────────────────────────


def test_equality_returns_bool():
    a = Composition.from_formula("H2O")
    b = Composition.from_dict({"H": 2, "O": 1})
    result = a == b
    assert isinstance(result, bool)
    assert result is True


def test_equality_distinguishes_isomers():
    a = Composition.from_formula("CH4")
    b = Composition.from_formula("NH3")
    assert (a == b) is False


def test_hashable_can_be_dict_key():
    h2o = Composition.from_formula("H2O")
    cache: dict = {h2o: "water"}
    # Equal composition must hit the same bucket.
    assert cache[Composition.from_dict({"H": 2, "O": 1})] == "water"


def test_hash_stable_across_construction_paths():
    a = Composition.from_formula("C6H12O6")
    b = Composition.from_dict({"C": 6, "H": 12, "O": 6})
    assert hash(a) == hash(b)


# ──────────────────────────────────────────────────────────────────────
# Tensor backing and batched free functions
# ──────────────────────────────────────────────────────────────────────


def test_counts_tensor_int32():
    c = Composition.from_formula("H2O")
    assert c.counts.dtype == torch.int32
    assert c.counts.shape == (len(ATOM_SYMBOLS),)


def test_stack_returns_2d_tensor():
    comps = [Composition.from_formula(f) for f in ("H2O", "CO2", "CH4")]
    batch = stack(comps)
    assert batch.shape == (3, len(ATOM_SYMBOLS))
    assert batch.dtype == torch.int32


def test_stack_empty_list():
    batch = stack([])
    assert batch.shape == (0, len(ATOM_SYMBOLS))


def test_batched_mass_matches_per_composition():
    comps = [Composition.from_formula(f) for f in ("H2O", "CO2", "C6H12O6")]
    batch = stack(comps)
    masses_batched = batched_mass(batch)
    assert masses_batched.shape == (3,)
    for i, c in enumerate(comps):
        assert abs(masses_batched[i].item() - c.mass) < 1e-9


def test_batched_mass_validates_shape():
    bad = torch.zeros(5, dtype=torch.int32)
    with pytest.raises(ValueError):
        batched_mass(bad)


def test_batched_average_mass_propagates_nan_for_undefined_weight():
    """Transactinides past Am have no defined standard atomic weight in
    our table (NIST leaves the field blank, since these are
    extrapolation-only). Compositions containing them yield NaN."""
    c = Composition.from_dict({"Og": 1})  # oganesson
    batched = batched_average_mass(c.counts.unsqueeze(0))
    assert math.isnan(batched[0].item())
