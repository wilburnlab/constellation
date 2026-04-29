"""Tests for `constellation.massspec.peptide.envelope`.

Validates that:
  * binned envelope spacing equals ISOTOPE_MASS_DIFF / charge
  * intensities sum to ~1.0 in binned mode
  * heavy-isotope mods (mass_override) shift the envelope by the override
    delta
  * exact mode returns valid envelopes (mz monotonic, abundances nonneg)

All inputs flow through `parse_proforma` / `Peptidoform` per the
ProForma 2.0 migration.
"""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.chem.elements import ISOTOPE_MASS_DIFF, PROTON_MASS
from constellation.core.chem.isotopes import isotope_envelope
from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.core.sequence.protein import peptide_composition
from constellation.massspec.peptide.envelope import peptide_envelope


def test_binned_envelope_shape_and_spacing():
    mz, ints = peptide_envelope(
        parse_proforma("PEPTIDEK"), charge=2, n_peaks=5
    )
    assert mz.shape == (5,)
    assert ints.shape == (5,)
    # Binned envelope is spaced at ISOTOPE_MASS_DIFF / charge along m/z axis.
    spacing = (mz[1] - mz[0]).item()
    assert math.isclose(spacing, ISOTOPE_MASS_DIFF / 2, abs_tol=1e-9)
    spacing2 = (mz[4] - mz[3]).item()
    assert math.isclose(spacing2, ISOTOPE_MASS_DIFF / 2, abs_tol=1e-9)


def test_binned_intensities_sum_to_one():
    _, ints = peptide_envelope(
        parse_proforma("PEPTIDEK"), charge=1, n_peaks=10
    )
    # FFT-binned distribution is normalized to sum 1.0 (within float precision).
    assert math.isclose(float(ints.sum()), 1.0, abs_tol=1e-5)


def test_binned_matches_underlying_isotope_envelope():
    """At charge 1, m/z = neutral + proton; envelope intensities should
    match `core.chem.isotopes.isotope_envelope` directly."""
    p = parse_proforma("PEPTIDE")
    composition = peptide_composition(p)
    masses_neutral, ints_ref = isotope_envelope(composition, n_peaks=5)
    mz, ints = peptide_envelope(p, charge=1, n_peaks=5)
    expected_mz = masses_neutral.to(torch.float64) + PROTON_MASS
    assert torch.allclose(mz, expected_mz, atol=1e-9)
    assert torch.allclose(ints, ints_ref, atol=1e-9)


def test_heavy_isotope_silac_shifts_envelope():
    """SILAC ¹³C₆ K shifts the entire envelope by the override delta / charge."""
    silac_delta = UNIMOD["UNIMOD:188"].delta_mass
    mz_light, _ = peptide_envelope(
        parse_proforma("PEPTIDEK"), charge=2, n_peaks=5
    )
    mz_heavy, _ = peptide_envelope(
        parse_proforma("PEPTIDEK[UNIMOD:188]"), charge=2, n_peaks=5
    )
    diff = (mz_heavy - mz_light).tolist()
    expected = silac_delta / 2
    for d in diff:
        assert math.isclose(d, expected, abs_tol=1e-6)


def test_exact_envelope_returns_valid_distribution():
    """High-res mode: m/z values strictly increasing, abundances non-negative."""
    mz, ints = peptide_envelope(
        parse_proforma("PEPTIDEK"), charge=1, n_peaks=8, mode="exact"
    )
    assert mz.numel() > 0
    diffs = mz[1:] - mz[:-1]
    assert (diffs > 0).all()
    assert (ints >= 0).all()


def test_charge_from_proforma_suffix():
    # /2 suffix on the proforma string flows into peptide_envelope.
    mz_implicit, _ = peptide_envelope(parse_proforma("PEPTIDEK/2"), n_peaks=3)
    mz_explicit, _ = peptide_envelope(
        parse_proforma("PEPTIDEK"), charge=2, n_peaks=3
    )
    assert torch.allclose(mz_implicit, mz_explicit, atol=1e-9)


def test_no_charge_raises():
    with pytest.raises(ValueError, match="requires charge"):
        peptide_envelope(parse_proforma("PEPTIDE"))


def test_invalid_charge():
    p = parse_proforma("PEPTIDE")
    with pytest.raises(ValueError, match="charge must be positive"):
        peptide_envelope(p, charge=0)


def test_invalid_mode():
    p = parse_proforma("PEPTIDE")
    with pytest.raises(ValueError, match="unknown mode"):
        peptide_envelope(p, charge=1, mode="invalid")  # type: ignore[arg-type]


def test_bare_peptidoform_construction():
    # Bare-sequence callers wrap with Peptidoform.
    p = Peptidoform(sequence="PEPTIDE")
    mz, _ = peptide_envelope(p, charge=2, n_peaks=3)
    assert mz.shape == (3,)
