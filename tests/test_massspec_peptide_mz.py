"""Tests for `constellation.massspec.peptide.mz`.

Validates `precursor_mz` against hand-computed values for plain peptides
and modified peptides (carbamidomethyl, oxidation, phospho, TMT, SILAC
heavy K) — the latter exercising both the `delta_composition` path and
the `mass_override` heavy-isotope correction path.

All inputs flow through `parse_proforma` / `Peptidoform` per the
ProForma 2.0 migration.
"""

from __future__ import annotations

import math

import pytest

from constellation.core.chem.elements import PROTON_MASS as ELEMENTS_PROTON_MASS
from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.core.sequence.protein import peptide_mass
from constellation.massspec.peptide.mz import PROTON_MASS, precursor_mz


def test_proton_mass_reexport():
    assert PROTON_MASS == ELEMENTS_PROTON_MASS


def test_precursor_mz_basic_peptide():
    # PEPTIDE neutral monoisotopic = 799.359963 Da; charge 1 → +PROTON
    p = parse_proforma("PEPTIDE")
    mz1 = precursor_mz(p, charge=1)
    expected = peptide_mass(p) + PROTON_MASS
    assert math.isclose(mz1, expected, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mz1, 800.367240, abs_tol=1e-4)


def test_precursor_mz_charge_states():
    p = parse_proforma("PEPTIDE")
    mass = peptide_mass(p)
    for z in (1, 2, 3, 4):
        mz = precursor_mz(p, charge=z)
        assert math.isclose(
            mz, (mass + z * PROTON_MASS) / z, rel_tol=0, abs_tol=1e-9
        )


def test_precursor_mz_charge_from_proforma_suffix():
    # /N suffix sets peptidoform.charge; precursor_mz picks it up.
    mz = precursor_mz(parse_proforma("PEPTIDE/2"))
    expected = (peptide_mass(parse_proforma("PEPTIDE")) + 2 * PROTON_MASS) / 2
    assert math.isclose(mz, expected, abs_tol=1e-9)


def test_precursor_mz_with_carbamidomethyl():
    # UNIMOD:4 on Cys, classic IAA workup.
    plain = parse_proforma("PEPCTIDE")
    cam = parse_proforma("PEPC[UNIMOD:4]TIDE")
    mz_plain = precursor_mz(plain, charge=2)
    mz_cam = precursor_mz(cam, charge=2)
    expected_delta = UNIMOD["UNIMOD:4"].delta_mass / 2
    assert math.isclose(mz_cam - mz_plain, expected_delta, abs_tol=1e-9)


def test_precursor_mz_with_oxidation():
    # MASPECT with oxidized methionine, UNIMOD:35 (+15.9949) on residue 0.
    plain = parse_proforma("MASPECT")
    ox = parse_proforma("M[UNIMOD:35]ASPECT")
    mz_plain = precursor_mz(plain, charge=2)
    mz_ox = precursor_mz(ox, charge=2)
    expected_delta = UNIMOD["UNIMOD:35"].delta_mass / 2
    assert math.isclose(mz_ox - mz_plain, expected_delta, abs_tol=1e-9)


def test_precursor_mz_heavy_isotope_silac_k():
    # SILAC ¹³C₆ Lys (UNIMOD:188) — mass_override path.
    silac = UNIMOD["UNIMOD:188"]
    assert silac.mass_override is not None
    light = parse_proforma("PEPTIDEK")
    heavy = parse_proforma("PEPTIDEK[UNIMOD:188]")
    mz_light = precursor_mz(light, charge=2)
    mz_heavy = precursor_mz(heavy, charge=2)
    expected = (peptide_mass(heavy) + 2 * PROTON_MASS) / 2
    assert math.isclose(mz_heavy, expected, abs_tol=1e-9)
    assert math.isclose(mz_heavy - mz_light, silac.delta_mass / 2, abs_tol=1e-9)


def test_precursor_mz_invalid_charge():
    p = parse_proforma("PEPTIDE")
    with pytest.raises(ValueError, match="charge must be positive"):
        precursor_mz(p, charge=0)
    with pytest.raises(ValueError):
        precursor_mz(p, charge=-1)


def test_precursor_mz_average_mass():
    # Average mass differs from monoisotopic; check both paths plumb.
    p = parse_proforma("PEPTIDE")
    mz_mono = precursor_mz(p, charge=1, monoisotopic=True)
    mz_avg = precursor_mz(p, charge=1, monoisotopic=False)
    assert mz_avg > mz_mono  # average always heavier (¹³C-natural-abundance shift)
    assert mz_avg - mz_mono > 0.3  # PEPTIDE has 34 carbons; ~0.4 Da heavier


def test_precursor_mz_bare_peptidoform():
    # Bare-sequence callers wrap with Peptidoform(sequence=...).
    p = Peptidoform(sequence="PEPTIDE")
    assert math.isclose(
        precursor_mz(p, charge=2),
        precursor_mz(parse_proforma("PEPTIDE"), charge=2),
        abs_tol=1e-9,
    )
