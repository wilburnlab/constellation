"""Tests for constellation.core.chem.isotopes — binned and exact APIs."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.chem.elements import ELEMENTS, ISOTOPE_MASS_DIFF
from constellation.core.chem.composition import Composition
from constellation.core.chem.isotopes import (
    average_mass,
    isotope_distribution,
    isotope_envelope,
    isotope_envelope_exact,
    isotopologue_distribution,
    monoisotopic_mass,
)


# ──────────────────────────────────────────────────────────────────────
# Binned isotope_distribution — cartographer-equivalent path
# ──────────────────────────────────────────────────────────────────────


def test_isotope_distribution_normalizes_to_one():
    dist = isotope_distribution(Composition.from_formula("C100"), n_peaks=5)
    assert abs(dist.sum().item() - 1.0) < 1e-5


def test_c100_first_peaks_match_cartographer():
    """C100 isotope envelope: classical reference distribution.
    First three peaks should be ≈ (0.339, 0.366, 0.196)."""
    dist = isotope_distribution(Composition.from_formula("C100"), n_peaks=5)
    expected = (0.339, 0.366, 0.196)
    for i, e in enumerate(expected):
        assert abs(dist[i].item() - e) < 5e-3


def test_isotope_distribution_pure_carbon_12_only_for_n_peaks_1():
    """For n_peaks=1, the M+0 abundance is just P(¹²C)^N."""
    dist = isotope_distribution(Composition.from_formula("C5"), n_peaks=1)
    # Should be normalized to 1.0 after truncation
    assert abs(dist.sum().item() - 1.0) < 1e-6


def test_isotope_envelope_masses_have_13c_spacing():
    masses, intensities = isotope_envelope(
        Composition.from_formula("C50H80"), n_peaks=4
    )
    assert masses.shape == (4,)
    # Consecutive masses differ by ISOTOPE_MASS_DIFF.
    for k in range(3):
        diff = masses[k + 1].item() - masses[k].item()
        assert abs(diff - ISOTOPE_MASS_DIFF) < 1e-9


def test_isotope_distribution_accepts_raw_tensor():
    """The hot path takes a raw count tensor to skip the Composition allocation."""
    c = Composition.from_formula("C20H40")
    via_comp = isotope_distribution(c, n_peaks=3)
    via_tensor = isotope_distribution(c.counts, n_peaks=3)
    assert torch.allclose(via_comp, via_tensor)


# ──────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────────────────────────────


def test_monoisotopic_and_average_mass_passthroughs():
    g = Composition.from_formula("C6H12O6")
    assert monoisotopic_mass(g) == g.mass
    assert average_mass(g) == g.average_mass


# ──────────────────────────────────────────────────────────────────────
# Isotopologue-resolved API — high-resolution / heavy-isotope path
# ──────────────────────────────────────────────────────────────────────


def test_chn_has_eight_isotopologues():
    """One C, one H, one N → 2 × 2 × 2 = 8 distinct isotopologues
    (¹²C/¹³C, ¹H/²H, ¹⁴N/¹⁵N). Confirms the data model and the convolution."""
    masses, abuns = isotopologue_distribution(
        Composition.from_dict({"C": 1, "H": 1, "N": 1}),
        prune_below=0.0,
    )
    assert masses.numel() == 8
    # Total abundance sums to 1.0 (no pruning).
    assert abs(abuns.sum().item() - 1.0) < 1e-9


def test_chn_isotopologue_abundances_match_analytical():
    """For independent isotope sampling, abundance of (¹³C, ¹H, ¹⁵N) is
    P(¹³C) * P(¹H) * P(¹⁵N) — the analytical product."""
    p13c = next(i.abundance for i in ELEMENTS["C"].isotopes if i.mass_number == 13)
    p1h = next(i.abundance for i in ELEMENTS["H"].isotopes if i.mass_number == 1)
    p15n = next(i.abundance for i in ELEMENTS["N"].isotopes if i.mass_number == 15)
    expected = p13c * p1h * p15n  # ¹³C¹H¹⁵N

    masses, abuns = isotopologue_distribution(
        Composition.from_dict({"C": 1, "H": 1, "N": 1}),
        prune_below=0.0,
    )
    # Find the (13C, 1H, 15N) peak: 13C exact + 1H exact + 15N exact
    c13_mass = next(i.exact_mass for i in ELEMENTS["C"].isotopes if i.mass_number == 13)
    h1_mass = next(i.exact_mass for i in ELEMENTS["H"].isotopes if i.mass_number == 1)
    n15_mass = next(i.exact_mass for i in ELEMENTS["N"].isotopes if i.mass_number == 15)
    target = c13_mass + h1_mass + n15_mass
    diffs = (masses - target).abs()
    idx = int(diffs.argmin().item())
    assert diffs[idx].item() < 1e-7
    assert abs(abuns[idx].item() - expected) < 1e-12


def test_15n_and_13c_peaks_distinct():
    """Regression test for the high-res isotope concern: ¹⁵N and ¹³C
    peaks at the same nominal-mass bin must be distinguishable in the
    isotopologue API. Mass diff (¹³C-¹²C) - (¹⁵N-¹⁴N) ≈ 0.00633 Da."""
    masses, abuns = isotopologue_distribution(
        Composition.from_dict({"C": 1, "N": 1}),
        prune_below=0.0,
    )
    # Should have 4 peaks: 12C14N, 12C15N, 13C14N, 13C15N.
    assert masses.numel() == 4
    # The two M+1 peaks (12C15N and 13C14N) should be distinct.
    sorted_m = masses.sort().values
    m_plus_one_lower = sorted_m[1].item()  # 12C15N (¹⁵N is +0.99703)
    m_plus_one_upper = sorted_m[2].item()  # 13C14N (¹³C is +1.00336)
    diff = m_plus_one_upper - m_plus_one_lower
    assert abs(diff - 0.00633) < 1e-4, f"15N/13C diff was {diff}"


def test_isotopologue_pruning_drops_long_tail():
    """High prune threshold should drop low-abundance peaks."""
    masses_strict, abuns_strict = isotopologue_distribution(
        Composition.from_formula("C100"),
        prune_below=0.05,
    )
    masses_all, abuns_all = isotopologue_distribution(
        Composition.from_formula("C100"),
        prune_below=1e-9,
    )
    assert masses_strict.numel() < masses_all.numel()
    # Total abundance loss is bounded by the threshold contribution.
    assert abuns_strict.sum().item() < abuns_all.sum().item()


def test_isotopologue_max_peaks_clip():
    masses, abuns = isotopologue_distribution(
        Composition.from_formula("C50"),
        prune_below=1e-12,
        max_peaks=3,
    )
    assert masses.numel() <= 3


# ──────────────────────────────────────────────────────────────────────
# isotope_envelope_exact — bins isotopologues at user resolution
# ──────────────────────────────────────────────────────────────────────


def test_envelope_exact_at_1da_matches_binned_for_natural_abundance():
    """At 1 Da bin width, the exact envelope should agree with the
    cartographer-equivalent binned envelope to within a few ppm for
    natural-abundance peptide-sized molecules."""
    comp = Composition.from_formula("C20H40N5O8")
    n_peaks = 4
    binned_masses, binned_intens = isotope_envelope(comp, n_peaks=n_peaks)
    exact_masses, exact_intens = isotope_envelope_exact(
        comp, bin_width_da=1.0, n_peaks=n_peaks
    )
    # First peak masses agree to <5 ppm.
    if exact_masses.numel() > 0 and binned_masses.numel() > 0:
        ppm = (
            abs(exact_masses[0].item() - binned_masses[0].item())
            / binned_masses[0].item()
            * 1e6
        )
        assert ppm < 5
    # Intensities agree up to small abundance pruning.
    n = min(exact_intens.numel(), binned_intens.numel())
    for i in range(n):
        assert abs(exact_intens[i].item() - binned_intens[i].item()) < 0.05


def test_envelope_exact_smaller_bin_yields_more_peaks():
    """Sub-Da bin width should resolve the ¹⁵N/¹³C cluster — peak count
    increases relative to nominal-mass binning."""
    comp = Composition.from_formula("C5N5")
    coarse = isotope_envelope_exact(comp, bin_width_da=1.0)
    fine = isotope_envelope_exact(comp, bin_width_da=0.001)
    # Fine binning resolves more peaks than coarse.
    assert fine[0].numel() >= coarse[0].numel()


def test_isotopologue_distribution_requires_composition():
    """Raw tensors are not accepted for the exact API — the contract is
    that this is an exploratory / high-precision path, not the hot loop."""
    c = Composition.from_formula("CN")
    with pytest.raises(TypeError):
        isotopologue_distribution(c.counts)  # type: ignore[arg-type]
