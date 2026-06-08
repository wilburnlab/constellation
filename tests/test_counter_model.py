"""Tests for the Counter Progenitor, Panel, attribution, and simulation."""

from __future__ import annotations

import pytest
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.counter import (
    CounterObservation,
    GlobalCalibration,
    Panel,
    Progenitor,
    observation_to_trace,
    simulate_observation,
)
from constellation.massspec.counter.attribution import responsibilities

_PEP = Peptidoform(sequence="PEPTIDEKR")


def _cal() -> GlobalCalibration:
    return GlobalCalibration(
        n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0, alpha_mz=0.6, nu_mz=7.8
    )


def _progenitor(cal: GlobalCalibration, n_total: float = 1e5) -> Progenitor:
    peak = HyperEMGPeak(
        N_total=n_total, mu=1.8e6, sigma=6000.0, tau_r=6000.0, tau_l=3000.0, eta=0.85
    )
    return Progenitor.for_peptide(
        _PEP, [2, 3], cal, n_isotopes=3, peak=peak, nu_intensity=6.0, c_mz_init=200.0
    )


def _dummy_obs(prog: Progenitor, iit_ms: float = 20.0) -> CounterObservation:
    s, c = 10, prog.channel_z.numel()
    rt = torch.linspace(1.79e6, 1.81e6, s, dtype=torch.float64)
    return CounterObservation(
        rt=rt,
        iit=torch.full((s,), iit_ms, dtype=torch.float64),
        intensity=torch.ones((s, c), dtype=torch.float64),
        mz_error=torch.zeros((s, c), dtype=torch.float64),
        mask=torch.ones((s, c), dtype=torch.bool),
        channel_z=prog.channel_z,
        channel_isotope=prog.channel_isotope,
        channel_mz=prog.channel_mz,
    )


# ──────────────────────────────────────────────────────────────────────
# Forward physics — IIT in the variance, not the mean (dimensional)
# ──────────────────────────────────────────────────────────────────────


def test_iit_cancels_in_mean_scales_count() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    i20, n20 = prog.predict(_dummy_obs(prog, iit_ms=20.0))
    i40, n40 = prog.predict(_dummy_obs(prog, iit_ms=40.0))
    # τ cancels in the mean intensity (Eq. 14).
    assert torch.allclose(i20, i40)
    # τ scales the accumulated count linearly (Eq. 2).
    assert torch.allclose(n40, 2.0 * n20)


def test_predict_dimensional_relation() -> None:
    # Eq. 11 at the predicted level: I_pred·τ/α(z) == N_count.
    cal = _cal()
    prog = _progenitor(cal)
    obs = _dummy_obs(prog)
    i_pred, n_count = prog.predict(obs)
    gain = cal.gain(prog.channel_z.to(torch.float64))
    assert torch.allclose(i_pred * obs.iit[:, None] / gain[None, :], n_count)


def test_charge_fractions_and_fixed_isotopes() -> None:
    prog = _progenitor(_cal())
    f = prog.charge_fractions()
    assert float(f.sum()) == pytest.approx(1.0)
    # isotope fractions are a fixed buffer (not a trainable parameter)
    param_names = {n for n, _ in prog.named_parameters()}
    assert "p_k" not in param_names
    assert float(prog.p_k.sum()) == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# Score — log_prob scalar, grad to peptide params (not calibration)
# ──────────────────────────────────────────────────────────────────────


def test_log_prob_scalar_grad_excludes_calibration() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    obs = simulate_observation(prog, n_scans=40, generator=torch.Generator().manual_seed(0))
    lp = prog.log_prob(obs)
    assert lp.dim() == 0
    lp.backward()
    assert prog.peak.log_N_total.grad is not None
    # calibration is shared-by-reference, NOT a progenitor parameter — so an
    # optimizer over progenitor.parameters() leaves it frozen during the fit.
    prog_param_names = {n for n, _ in prog.named_parameters()}
    assert not any(n.startswith("calibration") or "mz_offset" in n for n in prog_param_names)
    assert "channel_mz" not in prog_param_names


def test_for_peptide_builds_charge_major_grid() -> None:
    prog = _progenitor(_cal())
    # 2 charges × 3 isotopes = 6 channels, charge-major.
    assert prog.channel_z.tolist() == [2, 2, 2, 3, 3, 3]
    assert prog.channel_isotope.tolist() == [0, 1, 2, 0, 1, 2]
    assert prog.channel_mz.numel() == 6


# ──────────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────────


def test_sample_shapes_and_detection_floor() -> None:
    prog = _progenitor(_cal(), n_total=1e5)
    obs = simulate_observation(prog, n_scans=50, generator=torch.Generator().manual_seed(1))
    assert obs.intensity.shape == (50, 6)
    assert obs.mask.dtype == torch.bool
    # unobserved channels carry zero intensity
    assert float(obs.intensity[~obs.mask].abs().max()) == 0.0
    assert int(obs.mask.sum()) > 0


def test_observation_to_trace_casts_to_schema() -> None:
    from constellation.massspec.quant.schemas import XIC_TRACE_TABLE

    prog = _progenitor(_cal())
    obs = simulate_observation(prog, n_scans=30, generator=torch.Generator().manual_seed(2))
    trace, scan_meta = observation_to_trace(
        obs, acquisition_id=0, target_id=5, modified_sequence="PEPTIDEKR"
    )
    assert trace.schema.equals(XIC_TRACE_TABLE)
    assert trace.num_rows == int(obs.mask.sum())
    assert "iit" in scan_meta.column_names


# ──────────────────────────────────────────────────────────────────────
# Panel — additive, attribution, background, vmap
# ──────────────────────────────────────────────────────────────────────


def test_panel_log_prob_scalar_multi_progenitor() -> None:
    cal = _cal()
    p1, p2 = _progenitor(cal), _progenitor(cal)
    obs = simulate_observation(p1, n_scans=40, generator=torch.Generator().manual_seed(3))
    lp = Panel([p1, p2], cal).log_prob(obs)
    assert lp.dim() == 0 and torch.isfinite(lp)


def test_responsibilities_sum_to_one() -> None:
    stack = torch.rand(3, 5, 4, dtype=torch.float64) + 0.1  # (Q, S, C)
    gamma = responsibilities(stack)
    assert torch.allclose(gamma.sum(dim=0), torch.ones(5, 4, dtype=torch.float64))


def test_panel_background_parameter() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    panel = Panel([prog], cal, background=True)
    assert panel.log_background is not None
    assert float(panel.background_intensity()) > 0.0
    no_bg = Panel([prog], cal, background=False)
    assert no_bg.background_intensity() == 0.0


def test_panel_vmap_de_population_runs() -> None:
    # The score must be vmap-safe (DE's population path); just confirm it runs
    # and yields a finite loss over a few generations.
    from constellation.core.optim import DifferentialEvolution

    cal = _cal()
    prog = _progenitor(cal)
    obs = simulate_observation(prog, n_scans=30, generator=torch.Generator().manual_seed(4))
    panel = Panel([prog], cal)
    de = DifferentialEvolution(panel, pop_size=12, max_evals=120, seed=0)
    res = panel.fit(obs, optimizer=de, max_iter=8)
    import math

    assert math.isfinite(res.final_loss)
