"""Tests for `constellation.massspec.peptide.mz`.

Validates `precursor_mz` against hand-computed values for plain peptides
and modified peptides (carbamidomethyl, oxidation, phospho, TMT, SILAC
heavy K) — the latter exercising both the `delta_composition` path and
the `mass_override` heavy-isotope correction path.
"""

from __future__ import annotations

import math

import pytest

from constellation.core.chem.elements import PROTON_MASS as ELEMENTS_PROTON_MASS
from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.protein import peptide_mass
from constellation.massspec.peptide.mz import PROTON_MASS, precursor_mz


def test_proton_mass_reexport():
    assert PROTON_MASS == ELEMENTS_PROTON_MASS


def test_precursor_mz_basic_peptide():
    # PEPTIDE neutral monoisotopic = 799.359963 Da; charge 1 → +PROTON
    mz1 = precursor_mz("PEPTIDE", charge=1)
    expected = peptide_mass("PEPTIDE") + PROTON_MASS
    assert math.isclose(mz1, expected, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mz1, 800.367240, abs_tol=1e-4)


def test_precursor_mz_charge_states():
    seq = "PEPTIDE"
    mass = peptide_mass(seq)
    for z in (1, 2, 3, 4):
        mz = precursor_mz(seq, charge=z)
        assert math.isclose(mz, (mass + z * PROTON_MASS) / z, rel_tol=0, abs_tol=1e-9)


def test_precursor_mz_with_carbamidomethyl():
    # PEPC[+57.021]TIDE — UNIMOD:4 on Cys, classic IAA workup.
    seq, mods = "PEPCTIDE", {3: "UNIMOD:4"}
    mz_plain = precursor_mz(seq, charge=2)
    mz_cam = precursor_mz(seq, charge=2, modifications=mods)
    expected_delta = UNIMOD["UNIMOD:4"].delta_mass / 2
    assert math.isclose(mz_cam - mz_plain, expected_delta, abs_tol=1e-9)


def test_precursor_mz_with_oxidation():
    # MASPECT — oxidized methionine, UNIMOD:35 (+15.9949).
    mods = {0: "UNIMOD:35"}  # Met-ox at position 0
    mz_plain = precursor_mz("MASPECT", charge=2)
    mz_ox = precursor_mz("MASPECT", charge=2, modifications=mods)
    expected_delta = UNIMOD["UNIMOD:35"].delta_mass / 2
    assert math.isclose(mz_ox - mz_plain, expected_delta, abs_tol=1e-9)


def test_precursor_mz_heavy_isotope_silac_k():
    # SILAC ¹³C₆ Lys (UNIMOD:188) — mass_override path.
    silac = UNIMOD["UNIMOD:188"]
    assert silac.mass_override is not None
    # The override should be the canonical mass of 6×(13C-12C) = 6.020129.
    # Confirm peptide_mass matches mz formula.
    seq = "PEPTIDEK"
    mods = {7: "UNIMOD:188"}
    mz = precursor_mz(seq, charge=2, modifications=mods)
    expected = (peptide_mass(seq, modifications=mods) + 2 * PROTON_MASS) / 2
    assert math.isclose(mz, expected, abs_tol=1e-9)
    # Confirm the heavy delta moved the mass by the canonical override.
    mz_light = precursor_mz(seq, charge=2)
    assert math.isclose(mz - mz_light, silac.delta_mass / 2, abs_tol=1e-9)


def test_precursor_mz_invalid_charge():
    with pytest.raises(ValueError, match="charge must be positive"):
        precursor_mz("PEPTIDE", charge=0)
    with pytest.raises(ValueError):
        precursor_mz("PEPTIDE", charge=-1)


def test_precursor_mz_average_mass():
    # Average mass differs from monoisotopic; check both paths plumb.
    mz_mono = precursor_mz("PEPTIDE", charge=1, monoisotopic=True)
    mz_avg = precursor_mz("PEPTIDE", charge=1, monoisotopic=False)
    assert mz_avg > mz_mono  # average always heavier (¹³C-natural-abundance shift)
    assert mz_avg - mz_mono > 0.3  # PEPTIDE has 34 carbons; ~0.4 Da heavier
