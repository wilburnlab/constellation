"""Tests for ``massspec.spectra.consensus`` — fragment basis, peak→basis
alignment, and the replicate consensus builder (sum/median, dispersion, and
the per-replicate deviance-from-bulk outlier signal)."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.spectra.consensus import (
    align_to_basis,
    build_consensus,
    fragment_basis,
)


def _basis(seq: str = "PEPTIDE"):
    return fragment_basis(
        parse_proforma(seq), ion_types=(IonType.B, IonType.Y), max_fragment_charge=2
    )


# ──────────────────────────────────────────────────────────────────────
# fragment_basis
# ──────────────────────────────────────────────────────────────────────


def test_fragment_basis_deterministic_and_sized():
    b1 = _basis()
    b2 = _basis()
    assert b1.K == b2.K and b1.K > 0
    assert torch.equal(b1.mz_theoretical, b2.mz_theoretical)
    assert torch.equal(b1.ion_type, b2.ion_type)
    # PEPTIDE (len 7) → 6 b + 6 y per charge, charges 1..2 → 24 channels.
    assert b1.K == 24
    assert b1.mz_theoretical.shape == (b1.K,)
    assert len(b1.loss_id) == b1.K


# ──────────────────────────────────────────────────────────────────────
# align_to_basis
# ──────────────────────────────────────────────────────────────────────


def test_align_places_peak_at_right_channel():
    b = _basis()
    j = 5
    obs_mz = b.mz_theoretical[j : j + 1].clone()
    obs_int = torch.tensor([123.0])
    out = align_to_basis(obs_mz, obs_int, b, tolerance=20.0, tolerance_unit="ppm")
    assert out.shape == (b.K,)
    assert math.isclose(out[j].item(), 123.0, abs_tol=1e-9)
    assert math.isclose(out.sum().item(), 123.0, abs_tol=1e-9)


def test_align_unmatched_peak_drops():
    b = _basis()
    out = align_to_basis(
        torch.tensor([50.0]), torch.tensor([999.0]), b, tolerance=20.0
    )
    assert math.isclose(out.sum().item(), 0.0, abs_tol=1e-12)


def test_align_tolerance_unit_ppm_vs_da():
    b = _basis()
    j = 8
    # 0.4 Da off: outside 20 ppm (~0.01 Da at m/z ~500) but inside 0.5 Da.
    obs_mz = b.mz_theoretical[j : j + 1].clone() + 0.4
    obs_int = torch.tensor([10.0])
    ppm = align_to_basis(obs_mz, obs_int, b, tolerance=20.0, tolerance_unit="ppm")
    da = align_to_basis(obs_mz, obs_int, b, tolerance=0.5, tolerance_unit="Da")
    assert math.isclose(ppm.sum().item(), 0.0, abs_tol=1e-12)
    assert math.isclose(da[j].item(), 10.0, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# build_consensus
# ──────────────────────────────────────────────────────────────────────


def _spectrum_on(b, channel_intensities: dict[int, float]):
    """An (obs_mz, obs_intensity) pair placing intensity on given channels."""
    idx = sorted(channel_intensities)
    mz = b.mz_theoretical[idx].clone()
    inten = torch.tensor([channel_intensities[i] for i in idx], dtype=torch.float64)
    return mz, inten


def test_consensus_identical_replicates():
    b = _basis()
    spec = _spectrum_on(b, {0: 6.0, 1: 4.0, 2: 2.0})
    cons = build_consensus([spec, spec, spec], b, aggregate="sum")
    assert cons.n_replicates == 3
    assert cons.per_replicate.shape == (3, b.K)
    # sum aggregate = 3× the single spectrum
    assert math.isclose(cons.intensity[0].item(), 18.0, abs_tol=1e-9)
    assert math.isclose(cons.intensity[1].item(), 12.0, abs_tol=1e-9)
    # identical replicates → zero dispersion and zero deviance-from-bulk
    assert torch.allclose(cons.dispersion, torch.zeros(b.K, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(
        cons.deviance_from_bulk, torch.zeros(3, dtype=torch.float64), atol=1e-9
    )


def test_consensus_median_and_normalize():
    b = _basis()
    s = _spectrum_on(b, {0: 6.0, 1: 4.0})
    cons = build_consensus([s, s, s], b, aggregate="median", normalize=True)
    # median of identical replicates = the spectrum; normalized → simplex
    assert math.isclose(cons.intensity.sum().item(), 1.0, abs_tol=1e-9)
    assert math.isclose(cons.intensity[0].item(), 0.6, abs_tol=1e-9)
    assert math.isclose(cons.intensity[1].item(), 0.4, abs_tol=1e-9)


def test_consensus_dispersion_matches_known_value():
    b = _basis()
    a = _spectrum_on(b, {0: 6.0, 1: 4.0})  # proportions [0.6, 0.4]
    c = _spectrum_on(b, {0: 4.0, 1: 6.0})  # proportions [0.4, 0.6]
    cons = build_consensus([a, c], b)
    # population std of {0.6, 0.4} = 0.1 on both channels; 0 elsewhere.
    assert math.isclose(cons.dispersion[0].item(), 0.1, abs_tol=1e-9)
    assert math.isclose(cons.dispersion[1].item(), 0.1, abs_tol=1e-9)
    assert math.isclose(cons.dispersion[2].item(), 0.0, abs_tol=1e-12)


def test_consensus_outlier_has_high_deviance_from_bulk():
    b = _basis()
    inlier = _spectrum_on(b, {0: 8.0, 1: 1.0, 2: 1.0})
    outlier = _spectrum_on(b, {0: 1.0, 1: 1.0, 2: 8.0})  # mass shifted to channel 2
    cons = build_consensus([inlier, inlier, inlier, outlier], b, aggregate="median")
    dev = cons.deviance_from_bulk
    assert dev[3].item() > 5.0 * max(dev[0].item(), 1e-6)


def test_consensus_single_replicate_is_nan_free():
    b = _basis()
    cons = build_consensus([_spectrum_on(b, {0: 5.0, 3: 5.0})], b)
    assert cons.n_replicates == 1
    assert torch.isfinite(cons.dispersion).all()
    assert torch.allclose(cons.dispersion, torch.zeros(b.K, dtype=torch.float64))
    assert torch.isfinite(cons.deviance_from_bulk).all()


def test_consensus_empty_raises():
    b = _basis()
    with pytest.raises(ValueError, match="at least one"):
        build_consensus([], b)


def test_consensus_invalid_aggregate_raises():
    b = _basis()
    with pytest.raises(ValueError, match="aggregate"):
        build_consensus([_spectrum_on(b, {0: 1.0})], b, aggregate="mean")  # type: ignore[arg-type]
