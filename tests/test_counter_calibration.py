"""Tests for Counter staged calibration (StagedCalibration + hyperprior)."""

from __future__ import annotations

import pytest
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.counter import (
    GlobalCalibration,
    Progenitor,
    StagedCalibration,
    calibration_to_table,
    simulate_observation,
)


def _cal(**kw) -> GlobalCalibration:
    base = dict(
        n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0, alpha_mz=0.6, nu_mz=7.8
    )
    base.update(kw)
    return GlobalCalibration(**base)


def _calibrants(true_cal, seqs, ns, *, seed0=400):
    obs = []
    for i, (seq, n) in enumerate(zip(seqs, ns)):
        pk = HyperEMGPeak(
            N_total=n, mu=1.8e6, sigma=6000.0, tau_r=6000.0, tau_l=3000.0, eta=0.85
        )
        tp = Progenitor.for_peptide(
            Peptidoform(sequence=seq), [2, 3], true_cal, n_isotopes=3, peak=pk,
            nu_intensity=6.0, c_mz_init=200.0,
        )
        obs.append(
            simulate_observation(
                tp, n_scans=60, half_window_ms=30000.0, iit_ms=20.0,
                generator=torch.Generator().manual_seed(seed0 + i),
            )
        )
    return obs


def _fresh_progs(fit_cal, seqs):
    return [
        Progenitor.for_peptide(
            Peptidoform(sequence=s), [2, 3], fit_cal, n_isotopes=3,
            nu_intensity=5.0, c_mz_init=150.0,
        )
        for s in seqs
    ]


def test_staged_calibration_recovers_mz_offset_and_n() -> None:
    """Headline: from calibrants simulated under a perturbed m/z calibration,
    StagedCalibration recovers the global m/z offset + per-peptide N and
    promotes a peak-shape hyperprior — with the gain frozen (so no α↔N drift)
    and d_mz_0 pinned (so it isn't degenerate with the offset)."""
    seqs = ["PEPTIDEKR", "ELVISLIVEK", "ANALYTICR"]
    ns = [8e4, 1.5e5, 2e5]
    true_cal = _cal(mz_offset_ppm=3.0, d_mz_da=[0.0, 0.001, 0.0])
    obs = _calibrants(true_cal, seqs, ns)

    fit_cal = _cal(mz_offset_ppm=0.0, d_mz_da=[0.0, 0.0, 0.0])
    progs = _fresh_progs(fit_cal, seqs)
    res = StagedCalibration(progs, obs, fit_cal).run(global_iter=300, joint_iter=200)

    assert [r["stage"] for r in res.stage_reports] == ["per_peptide", "global", "joint"]
    # m/z offset recovered (cleanly identifiable); d_mz_0 stays pinned at 0.
    assert abs(float(fit_cal.mz_offset_ppm.detach()) - 3.0) < 0.7
    assert abs(float(fit_cal.d_mz_da.detach()[0])) < 1e-9
    # per-peptide N recovered (gain frozen ⇒ no α/N degeneracy).
    for n, q in zip(ns, progs):
        assert abs(float(q.peak.N_total.detach()) - n) / n < 0.12
    # a peak-shape hyperprior was promoted and installed on the calibration.
    assert res.peak_shape_prior is not None
    assert fit_cal.peak_shape_prior is res.peak_shape_prior
    for name in ("peak.log_sigma", "peak.log_tau_r", "peak.log_tau_l", "peak.logit_eta"):
        mean, std = res.peak_shape_prior[name]
        assert torch.isfinite(torch.tensor([mean, std])).all() and std > 0.0


def test_staged_calibration_freezes_gain_by_default() -> None:
    """The default global_params excludes the gain (identifiable only via the
    weak shot-noise curvature), so a run leaves α(z) untouched while still
    moving the cleanly-identifiable m/z offset."""
    seqs = ["PEPTIDEKR", "ELVISLIVEK"]
    ns = [1e5, 1.5e5]
    true_cal = _cal(mz_offset_ppm=2.0)
    obs = _calibrants(true_cal, seqs, ns)

    fit_cal = _cal(mz_offset_ppm=0.0)
    g_before = float(fit_cal.gain(torch.tensor(2.0)).detach())
    progs = _fresh_progs(fit_cal, seqs)
    StagedCalibration(progs, obs, fit_cal).run(
        global_iter=200, joint_iter=150, promote=False
    )
    assert float(fit_cal.gain(torch.tensor(2.0)).detach()) == pytest.approx(
        g_before, abs=1e-9
    )
    assert float(fit_cal.mz_offset_ppm.detach()) > 0.5  # identifiable, moved


def test_gain_calibration_converges_with_jacobian() -> None:
    """The log(τ/α) change-of-variables Jacobian makes the likelihood proper in
    α: co-fitting the gain from a mis-set start now converges TOWARD the truth
    (and N stays bounded) instead of diverging (α↑, N→0) as it would without
    the Jacobian. Recovery is partial — the shot-noise curvature is weak."""
    seqs = ["PEPTIDEKR", "ELVISLIVEK", "ANALYTICR", "GHISTIDINEK"]
    ns = [8e4, 1.5e5, 2e5, 1.2e5]
    true_cal = _cal(alpha1=15.0)  # true gain slope
    obs = _calibrants(true_cal, seqs, ns, seed0=500)

    fit_cal = _cal(alpha1=22.0)  # mis-set high
    progs = _fresh_progs(fit_cal, seqs)
    res = StagedCalibration(progs, obs, fit_cal).run(
        global_params=(
            "mz_offset_ppm", "d_mz_da", "log_alpha_mz", "log_nu_mz", "rho",
            "alpha0", "alpha1",
        ),
        global_iter=500,
        joint_iter=300,
    )
    a1 = float(fit_cal.alpha1.detach())
    # moved down from 22 toward 15 (not diverging upward), and well bounded.
    assert 13.0 < a1 < 21.0
    assert a1 < 22.0
    for n, q in zip(ns, progs):
        nf = float(q.peak.N_total.detach())
        assert 1e3 < nf < 1e6  # no collapse / blow-up
        assert abs(nf - n) / n < 0.30
    assert res.peak_shape_prior is not None


def test_staged_calibration_validates_inputs() -> None:
    cal1, cal2 = _cal(), _cal()
    obs = _calibrants(cal1, ["PEPTIDEKR"], [1e5])
    p1 = Progenitor.for_peptide(
        Peptidoform(sequence="PEPTIDEKR"), [2, 3], cal1, n_isotopes=3
    )
    with pytest.raises(ValueError):  # progenitor references a different calibration
        StagedCalibration([p1], obs, cal2)
    with pytest.raises(ValueError):  # progenitor / observation count mismatch
        StagedCalibration([p1], obs + obs, cal1)


def test_promote_hyperprior_installs_on_calibration() -> None:
    cal = _cal()
    obs = _calibrants(cal, ["PEPTIDEKR", "ELVISLIVEK"], [1e5, 1.5e5])
    progs = _fresh_progs(cal, ["PEPTIDEKR", "ELVISLIVEK"])
    sc = StagedCalibration(progs, obs, cal)
    # seed so the peak params are sane, then promote directly.
    sc.run(stages=("per_peptide",))
    prior = sc.promote_peak_shape_hyperprior()
    assert set(prior) == {
        "peak.log_sigma", "peak.log_tau_r", "peak.log_tau_l", "peak.logit_eta"
    }
    assert cal.peak_shape_prior == prior


def test_calibration_table_persists_hyperprior() -> None:
    cal = _cal()
    cal.set_peak_shape_prior(
        {"peak.log_sigma": (8.7, 0.2), "peak.log_tau_r": (8.6, 0.3)}
    )
    tbl = calibration_to_table(cal, acquisition_id=0)
    assert tbl.column("prior_log_sigma_mean")[0].as_py() == pytest.approx(8.7)
    assert tbl.column("prior_log_sigma_std")[0].as_py() == pytest.approx(0.2)
    assert tbl.column("prior_log_tau_mean")[0].as_py() == pytest.approx(8.6)
    assert tbl.column("prior_log_tau_std")[0].as_py() == pytest.approx(0.3)
    # unset prior → null columns.
    tbl0 = calibration_to_table(_cal(), acquisition_id=0)
    assert tbl0.column("prior_log_sigma_mean")[0].as_py() is None


_FULL_PRIOR = {
    "peak.log_sigma": (8.7, 0.2),
    "peak.log_tau_r": (8.6, 0.3),
    "peak.log_tau_l": (8.4, 0.25),
    "peak.logit_eta": (1.1, 0.5),
}


def _assert_prior_equal(got, want) -> None:
    assert set(got) == set(want)
    for k in want:
        assert got[k][0] == pytest.approx(want[k][0])
        assert got[k][1] == pytest.approx(want[k][1])


def test_calibration_hyperprior_roundtrips_all_four_shape_slots() -> None:
    # D8: τ_l and η priors used to be silently dropped (only σ + τ_r persisted).
    from constellation.massspec.counter import calibration_from_table

    cal = _cal()
    cal.set_peak_shape_prior(dict(_FULL_PRIOR))
    tbl = calibration_to_table(cal, acquisition_id=0)
    assert tbl.column("prior_log_tau_l_mean")[0].as_py() == pytest.approx(8.4)
    assert tbl.column("prior_logit_eta_std")[0].as_py() == pytest.approx(0.5)
    _assert_prior_equal(calibration_from_table(tbl).peak_shape_prior, _FULL_PRIOR)


def test_calibration_from_table_reads_v1_parquet_without_new_slots() -> None:
    # A v1 table has no τ_l / η columns; from_table must still load it.
    from constellation.massspec.counter import calibration_from_table

    cal = _cal()
    cal.set_peak_shape_prior(dict(_FULL_PRIOR))
    v1 = calibration_to_table(cal, acquisition_id=0).drop_columns(
        ["prior_log_tau_l_mean", "prior_log_tau_l_std",
         "prior_logit_eta_mean", "prior_logit_eta_std"]
    )
    restored = calibration_from_table(v1).peak_shape_prior  # must not raise
    _assert_prior_equal(
        restored, {"peak.log_sigma": (8.7, 0.2), "peak.log_tau_r": (8.6, 0.3)}
    )
