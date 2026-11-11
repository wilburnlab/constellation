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


@pytest.mark.parametrize("mode", ["binned", "exact"])
@pytest.mark.parametrize("modseq", ["PEPTIDEK", "AC[UNIMOD:4]DM[UNIMOD:35]K", "<13C><15N>LGEYGFQNALIVR"])
def test_a_neutral_envelope_put_at_a_charge_is_the_charged_envelope(mode, modseq):
    from constellation.massspec.peptide.envelope import envelope_mz, neutral_envelope

    pf = parse_proforma(modseq)
    masses, ab = neutral_envelope(pf, n_peaks=4, mode=mode)
    for z in (1, 2, 3, 4):
        mz, ab_z = peptide_envelope(pf, charge=z, n_peaks=4, mode=mode)
        assert torch.equal(envelope_mz(masses, z), mz)
        assert torch.equal(ab, ab_z)


def test_batched_envelopes_are_the_single_ones_to_the_bit():
    """neutral_envelopes repeats neutral_envelope's arithmetic (the
    compensated sum Python makes of a composition's mass) so its masses
    are the same bits, for labelled, SILAC and modified peptides too."""
    import numpy as np

    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.peptide.envelope import neutral_envelope, neutral_envelopes

    texts = ["PEPTIDEK", "LVNELTEFAK", "<13C><15N>ACDEFGHIK", "PEPTK[UNIMOD:259]IDEK", "C[UNIMOD:4]PEPM[UNIMOD:35]TIDER",
             "<15N>PEPTIDEKR", "[UNIMOD:1]-SAMPLER", "HLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRK"]  # fmt: skip
    peptides = [parse_proforma(t) for t in texts]
    masses, shares = neutral_envelopes(peptides, n_peaks=6)
    for i, p in enumerate(peptides):
        m, a = neutral_envelope(p, n_peaks=6, normalize="full")
        assert np.array_equal(m.numpy().view(np.int64), masses[i].view(np.int64)), texts[i]
        assert np.abs(a.numpy() - shares[i]).max() < 1e-6


@pytest.mark.parametrize("mode", ["binned", "exact"])
@pytest.mark.parametrize(
    "text, heavy",
    [("LVNELTEFAK[UNIMOD:259]", {"C": 6, "N": 2}),  # SILAC 13C(6) 15N(2) lysine
     ("[UNIMOD:737]-PEPTIDEK[UNIMOD:737]", {"C": 8, "N": 2})],  # TMT6plex 13C(4) 15N, twice
)  # fmt: skip
def test_the_heavy_atoms_of_silac_and_tmt_are_one_isotope_each(mode, text, heavy):
    """Review of 2026-11-10, C8: the envelope gave a SILAC or TMT label's
    heavy atoms natural variance (M+0's share 7-9 % low, M+2's 11-16 %
    high). They are fixed isotopes: the shape is the other atoms', the m/z
    as before."""
    import numpy as np

    from constellation.core.chem.composition import Composition
    from constellation.core.chem.elements import ELEMENTS
    from constellation.core.chem.isotopes import isotope_envelope_exact
    from constellation.core.sequence.protein import peptide_mass
    from constellation.massspec.peptide.envelope import neutral_envelope, neutral_envelopes

    p = parse_proforma(text)
    counts = peptide_composition(p).counts.clone()
    for symbol, n in heavy.items():
        counts[ELEMENTS.index(symbol)] -= n
    natural = Composition(counts)
    masses, shares = neutral_envelope(p, n_peaks=4, mode=mode)
    _, want = isotope_envelope(natural, n_peaks=4) if mode == "binned" else isotope_envelope_exact(natural, n_peaks=4)
    np.testing.assert_allclose(shares.numpy(), want.numpy(), rtol=1e-12)
    _, as_natural = (isotope_envelope(peptide_composition(p), n_peaks=4) if mode == "binned"
                     else isotope_envelope_exact(peptide_composition(p), n_peaks=4))  # fmt: skip
    assert float(shares[0]) > float(as_natural[0]) * 1.05  # what it was: M+0 several per cent low
    assert math.isclose(float(masses[0]), peptide_mass(p), abs_tol=1e-9)
    if mode == "binned":
        _, batched = neutral_envelopes([p], n_peaks=4)
        _, full = neutral_envelope(p, n_peaks=4, normalize="full")
        np.testing.assert_allclose(batched[0], full.numpy(), atol=1e-6)
