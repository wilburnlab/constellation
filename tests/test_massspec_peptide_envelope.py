"""Tests for `constellation.massspec.peptide.envelope`.

Validates that:
  * binned envelope spacing equals ISOTOPE_MASS_DIFF / charge
  * intensities sum to ~1.0 in binned mode
  * heavy-isotope mods (mass_override) shift the envelope by the override
    delta
  * exact mode returns valid envelopes (mz monotonic, abundances nonneg)
"""

from __future__ import annotations

import math

import torch

from constellation.core.chem.elements import ISOTOPE_MASS_DIFF, PROTON_MASS
from constellation.core.chem.isotopes import isotope_envelope
from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.protein import peptide_composition
from constellation.massspec.peptide.envelope import peptide_envelope


def test_binned_envelope_shape_and_spacing():
    mz, ints = peptide_envelope("PEPTIDEK", charge=2, n_peaks=5)
    assert mz.shape == (5,)
    assert ints.shape == (5,)
    # Binned envelope is spaced at ISOTOPE_MASS_DIFF / charge along m/z axis.
    spacing = (mz[1] - mz[0]).item()
    assert math.isclose(spacing, ISOTOPE_MASS_DIFF / 2, abs_tol=1e-9)
    spacing2 = (mz[4] - mz[3]).item()
    assert math.isclose(spacing2, ISOTOPE_MASS_DIFF / 2, abs_tol=1e-9)


def test_binned_intensities_sum_to_one():
    _, ints = peptide_envelope("PEPTIDEK", charge=1, n_peaks=10)
    # FFT-binned distribution is normalized to sum 1.0 (within float precision).
    assert math.isclose(float(ints.sum()), 1.0, abs_tol=1e-5)


def test_binned_matches_underlying_isotope_envelope():
    """At charge 1, m/z = neutral + proton; envelope intensities should
    match `core.chem.isotopes.isotope_envelope` directly."""
    seq = "PEPTIDE"
    composition = peptide_composition(seq)
    masses_neutral, ints_ref = isotope_envelope(composition, n_peaks=5)
    mz, ints = peptide_envelope(seq, charge=1, n_peaks=5)
    expected_mz = masses_neutral.to(torch.float64) + PROTON_MASS
    assert torch.allclose(mz, expected_mz, atol=1e-9)
    assert torch.allclose(ints, ints_ref, atol=1e-9)


def test_heavy_isotope_silac_shifts_envelope():
    """SILAC ¹³C₆ K shifts the entire envelope by the override delta / charge."""
    seq = "PEPTIDEK"
    silac_delta = UNIMOD["UNIMOD:188"].delta_mass
    mz_light, _ = peptide_envelope(seq, charge=2, n_peaks=5)
    mz_heavy, _ = peptide_envelope(
        seq, charge=2, n_peaks=5, modifications={7: "UNIMOD:188"}
    )
    diff = (mz_heavy - mz_light).numpy()
    expected = silac_delta / 2
    for d in diff:
        assert math.isclose(d, expected, abs_tol=1e-6)


def test_exact_envelope_returns_valid_distribution():
    """High-res mode: m/z values strictly increasing, abundances non-negative."""
    mz, ints = peptide_envelope("PEPTIDEK", charge=1, n_peaks=8, mode="exact")
    assert mz.numel() > 0
    diffs = mz[1:] - mz[:-1]
    assert (diffs > 0).all()
    assert (ints >= 0).all()


def test_invalid_charge():
    import pytest

    with pytest.raises(ValueError, match="charge must be positive"):
        peptide_envelope("PEPTIDE", charge=0)


def test_invalid_mode():
    import pytest

    with pytest.raises(ValueError, match="unknown mode"):
        peptide_envelope("PEPTIDE", charge=1, mode="invalid")  # type: ignore[arg-type]
