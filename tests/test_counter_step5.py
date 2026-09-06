"""Step 5 foundations: the N-weighted seed (PR-A) and the L6 N_total prior term
(PR-B), the latter wired into the single-progenitor `estimate_n` VB path (the safe
home; the multi-progenitor panel prior awaits VB-on-panel — a post-DE MAP polish
drifts/diverges/collapses on the same-grid-blend surface). See
docs/plans/counter-real-data-deconstruction.md §9."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.counter import (
    CounterObservation,
    GlobalCalibration,
    Progenitor,
    estimate_n,
    make_log_prior,
    seed_peak_from_observation,
    simulate_observation,
)

_PEP = Peptidoform(sequence="PEPTIDEKR")


def _cal() -> GlobalCalibration:
    return GlobalCalibration(
        n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0, alpha_mz=0.6, nu_mz=7.8
    )


def _progenitor(cal: GlobalCalibration, charges=(2, 3), n_total: float = 1e5) -> Progenitor:
    peak = HyperEMGPeak(
        N_total=n_total, mu=1.8e6, sigma=6000.0, tau_r=6000.0, tau_l=3000.0, eta=0.85
    )
    return Progenitor.for_peptide(
        _PEP, list(charges), cal, n_isotopes=3, peak=peak, nu_intensity=6.0, c_mz_init=200.0
    )


# ──────────────────────────────────────────────────────────────────────
# PR-A — the seed μ-centroid is weighted by the recovered ion count, not raw I
# ──────────────────────────────────────────────────────────────────────


def _flat_obs(prog: Progenitor, rt: torch.Tensor, iit: torch.Tensor) -> CounterObservation:
    s, c = rt.numel(), prog.channel_z.numel()
    return CounterObservation(
        rt=rt,
        iit=iit,
        intensity=torch.ones((s, c), dtype=torch.float64),
        mz_error=torch.zeros((s, c), dtype=torch.float64),
        mask=torch.ones((s, c), dtype=torch.bool),
        channel_z=prog.channel_z,
        channel_isotope=prog.channel_isotope,
        channel_mz=prog.channel_mz,
    )


def test_seed_mu_is_recovered_count_weighted() -> None:
    cal = _cal()
    # single-charge progenitor (3 isotope channels, all z=2 → one gain), so the
    # per-scan recovered-count weight is ∝ iit (the constant gain + isotope sum
    # cancel) and the divergence from intensity-weighting is isolated to τ.
    prog = _progenitor(cal, charges=(2,))
    rt = torch.tensor([1.0e3, 2.0e3, 3.0e3, 4.0e3, 5.0e3], dtype=torch.float64)
    intensity_mean = float(rt.mean())  # 3000 — the intensity-weighted centroid (uniform I)

    iit_ramp = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64)
    seed_peak_from_observation(prog, _flat_obs(prog, rt, iit_ramp))
    expected = float((iit_ramp * rt).sum() / iit_ramp.sum())  # N(iit)-weighted = 3666.7
    assert float(prog.peak.mu.detach()) == pytest.approx(expected, rel=1e-9)
    assert abs(float(prog.peak.mu.detach()) - intensity_mean) > 100.0  # ≠ intensity-weighted

    # constant τ → reduces exactly to the intensity-weighted mean (no regression)
    seed_peak_from_observation(prog, _flat_obs(prog, rt, torch.full((5,), 20.0, dtype=torch.float64)))
    assert float(prog.peak.mu.detach()) == pytest.approx(intensity_mean, rel=1e-9)


# ──────────────────────────────────────────────────────────────────────
# PR-B — the N_total prior term in make_log_prior
# ──────────────────────────────────────────────────────────────────────


def test_n_total_prior_term_math() -> None:
    lp = make_log_prior(n_total_center=10.0, n_total_sigma=2.0)
    assert lp is not None
    assert float(lp({"peak.log_N_total": torch.tensor(10.0)})) == 0.0  # zero at center
    # quadratic: 2σ off → -0.5·(4/2)² = -2
    assert float(lp({"peak.log_N_total": torch.tensor(14.0)})) == pytest.approx(-2.0)
    # panel-namespaced key resolves (the load-bearing multi-progenitor seam)
    lp2 = make_log_prior(
        n_total_center=5.0, n_total_sigma=1.0, n_total_key="progenitors.0.peak.log_N_total"
    )
    assert float(lp2({"progenitors.0.peak.log_N_total": torch.tensor(5.0)})) == 0.0
    # leading sample dim preserved (the VB batched-dict contract)
    out = lp({"peak.log_N_total": torch.full((7,), 10.0, dtype=torch.float64)})
    assert out.shape == (7,)
    # no active term → None
    assert make_log_prior() is None


def test_n_total_prior_composes_with_rt_term() -> None:
    lp = make_log_prior(rt_prior_ms=1000.0, rt_sigma_ms=500.0, n_total_center=8.0, n_total_sigma=1.0)
    params = {"peak.mu": torch.tensor(1000.0), "peak.log_N_total": torch.tensor(8.0)}
    assert float(lp(params)) == 0.0  # both terms at their centers
    off = {"peak.mu": torch.tensor(1000.0), "peak.log_N_total": torch.tensor(9.0)}
    assert float(lp(off)) == pytest.approx(-0.5)  # only the n_total term contributes


# ──────────────────────────────────────────────────────────────────────
# PR-B wiring — estimate_n VB path accepts n_total_prior and still recovers N
# ──────────────────────────────────────────────────────────────────────


def test_estimate_n_n_total_prior_vb_runs() -> None:
    cal = _cal()
    prog = _progenitor(cal, n_total=1e5)
    obs = simulate_observation(prog, n_scans=40, generator=torch.Generator().manual_seed(5))
    res = estimate_n(
        prog,
        obs,
        inference="vb",
        n_total_prior=True,
        seed=0,
        pop_size=12,
        max_evals=400,
        max_iter=20,
        vb_max_iter=300,
        n_elbo_samples=8,
        n_ci_samples=512,
    )
    assert math.isfinite(res["n_total"]) and res["n_total"] > 0.0
    assert res["n_total_lo"] <= res["n_total"] <= res["n_total_hi"]
    # the prior is centered on the (data-derived) seed, so recovery is not broken
    assert abs(res["n_total"] - 1e5) / 1e5 < 0.6
