"""Tests for the panel-shaped estimate + automatic same-grid discovery.

Synthetic only (`simulate_*`), shared-grid. PR-4 shipped the plumbing + a no-op
discovery stub; those tests lock the seam in place and prove it *enables* the
interference fix (manual add). PR-5 fills `discover_candidates` so the loop finds
the interferer automatically — its detection contract is unit-tested
deterministically (pure/no-grad), with one looser end-to-end engagement smoke
over the stochastic DE refit.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.counter import (
    DiscoverConfig,
    GlobalCalibration,
    Panel,
    Progenitor,
    discover_candidates,
    estimate_n,
    estimate_panel,
    fit_panel,
    observation_for_region,
    observation_from_trace,
    observation_to_trace,
    panel_predicted_intensity,
    panel_residual,
    simulate_observation,
    simulate_panel_observation,
)

_RT = torch.linspace(1.71e6, 2.01e6, 150, dtype=torch.float64)
_PEP = Peptidoform(sequence="PEPTIDEKR")


def _cal() -> GlobalCalibration:
    return GlobalCalibration(
        n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0, alpha_mz=0.6, nu_mz=7.8
    )


def _prog(cal, n=1e5, mu=1.8e6, sigma=5000.0) -> Progenitor:
    return Progenitor.for_peptide(
        _PEP, [2, 3], cal, n_isotopes=3,
        peak=HyperEMGPeak(N_total=n, mu=mu, sigma=sigma, tau_r=4000.0, tau_l=2000.0, eta=0.8),
        nu_intensity=6.0, c_mz_init=200.0,
    )


def _two_peak(cal, n_target=2e5, n_interferer=1.5e5, seed=4):
    """Target at 1.80e6 + a same-grid interferer at 1.92e6 (offset RT)."""
    return simulate_panel_observation(
        [_prog(cal, n_target, 1.8e6), _prog(cal, n_interferer, 1.92e6)],
        rt=_RT, iit_ms=20.0, generator=torch.Generator().manual_seed(seed),
    )


def _fit_target(cal, n_target=2e5):
    """A panel whose single progenitor is placed *at the truth* of peak A — a
    deterministic stand-in for a converged first fit, so detection can be
    unit-tested without the (stochastic) DE refit."""
    return Panel([_prog(cal, n_target, 1.8e6)], cal, background=False)


def test_panel_mutation_safety() -> None:
    """add/remove keep `log_prob` finite/0-d, the gradient finite, and the DE
    (vmap) fit path working — the path the estimate relies on."""
    cal = _cal()
    obs = simulate_observation(
        _prog(cal, 1e5), n_scans=60, half_window_ms=40000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(0),
    )
    panel = Panel([_prog(cal, 1.0)], cal, background=False)
    idx = panel.add_progenitor(_prog(cal, 1.0, mu=1.9e6))
    assert idx == 1 and len(panel.progenitors) == 2

    lp = panel.log_prob(obs)
    assert lp.dim() == 0 and bool(torch.isfinite(lp))
    g = torch.autograd.grad(panel.log_prob(obs), panel.progenitors[0].peak.log_N_total)[0]
    assert bool(torch.isfinite(g))
    fit_panel(panel, obs, optimizer="de", seed=0, max_evals=2000)  # vmap path after add

    panel.remove_progenitor(1)
    assert len(panel.progenitors) == 1
    with pytest.raises(ValueError):  # calibration-share guard
        panel.add_progenitor(_prog(_cal(), 1.0))
    with pytest.raises(IndexError):
        panel.remove_progenitor(5)


def test_panel_residual_matches_definition() -> None:
    cal = _cal()
    truth = _prog(cal, 1e5)
    obs = simulate_observation(
        truth, n_scans=50, half_window_ms=40000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(1),
    )
    panel = Panel([truth], cal, background=False)
    pred = panel_predicted_intensity(panel, obs)
    r = panel_residual(panel, obs)
    expected = torch.where(obs.mask, obs.intensity - pred, torch.zeros_like(obs.intensity))
    assert torch.allclose(r, expected)
    assert float(r[~obs.mask].abs().max()) == 0.0


def test_observation_for_region_retains_off_grid_peaks() -> None:
    """The region builder keeps in-window peaks the observation dropped (here an
    off-grid isotope) as RegionPeaks — the PR-5 discovery substrate — while the
    dense observation is unchanged."""
    cal = _cal()
    truth = _prog(cal, 1e5)
    obs0 = simulate_observation(
        truth, n_scans=40, half_window_ms=30000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(2),
    )
    trace, scan_meta = observation_to_trace(obs0, acquisition_id=0, target_id=0)
    # append an off-grid peak (isotope 7 — beyond the 3-isotope grid) at a target m/z
    tgt_mz0 = float(truth.channel_mz[0])
    a_scan = int(trace.column("scan")[0].as_py())
    extra = {n: trace.column(n)[0].as_py() for n in trace.column_names}
    extra.update(
        scan=a_scan, isotope=7, precursor_charge=2, intensity=999.0,
        mz_theoretical=tgt_mz0, mz_observed=tgt_mz0, mz_error_ppm=0.0, mz_error_da=0.0,
    )
    trace2 = pa.concat_tables([trace, pa.Table.from_pylist([extra], schema=trace.schema)])

    prog = _prog(cal, 1.0)
    obs, region = observation_for_region(trace2, scan_meta, prog, target_id=0)
    obs_ref = observation_from_trace(
        trace2, scan_meta, channel_z=prog.channel_z,
        channel_isotope=prog.channel_isotope, channel_mz=prog.channel_mz, target_id=0,
    )
    assert torch.equal(obs.mask, obs_ref.mask)  # off-grid peak does not enter the obs
    assert region.intensity.numel() >= 1  # ... but is retained for discovery
    assert float(region.intensity.max()) == pytest.approx(999.0)


def test_estimate_panel_matches_estimate_n_clean() -> None:
    """On a clean single-target obs the stub loop fits once → the panel estimate
    equals `estimate_n`; nothing is discovered."""
    cal = _cal()
    obs = simulate_observation(
        _prog(cal, 2e5), n_scans=90, half_window_ms=45000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(3),
    )
    rn = estimate_n(_prog(cal, 1.0), obs, inference="map", optimizer="de", seed=0)
    panel = Panel([_prog(cal, 1.0)], cal, background=False)
    rp = estimate_panel(panel, obs, None, config=DiscoverConfig(), rt_prior_ms=1.8e6)
    assert rp["n_discovered_interferers"] == 0
    assert rp["interference_flag"] is False
    assert abs(rp["n_total"] - rn["n_total"]) / rn["n_total"] < 0.10
    assert rp["n_total_lo"] < rp["n_total"] < rp["n_total_hi"]


def test_manual_add_corrects_target() -> None:
    """Plumbing enables the fix: a single progenitor can't own two co-eluting
    peaks (biased), but adding the interferer (narrow seeds — what PR-5 discovery
    will provide from the residual) recovers the target N."""
    cal = _cal()
    rt = torch.linspace(1.71e6, 2.01e6, 150, dtype=torch.float64)
    obs = simulate_panel_observation(
        [_prog(cal, 2e5, 1.8e6), _prog(cal, 1.5e5, 1.92e6)],
        rt=rt, iit_ms=20.0, generator=torch.Generator().manual_seed(4),
    )
    rs = estimate_n(_prog(cal, 1.0), obs, inference="map", optimizer="de", seed=0)
    single_err = abs(rs["n_total"] - 2e5) / 2e5
    assert single_err > 0.30  # one progenitor cannot own two peaks

    panel = Panel([_prog(cal, 1.0, 1.8e6), _prog(cal, 1.0, 1.92e6)], cal, background=False)
    for q, mu in ((panel.progenitors[0], 1.8e6), (panel.progenitors[1], 1.92e6)):
        with torch.no_grad():
            q.peak.log_N_total.fill_(math.log(1e5))
            q.peak.mu.fill_(mu)
            q.peak.log_sigma.fill_(math.log(5000.0))
    fit_panel(panel, obs, optimizer="de", seed=0, max_evals=10000)
    target_n = float(panel.progenitors[0].peak.N_total.detach())
    panel_err = abs(target_n - 2e5) / 2e5
    assert panel_err < 0.15  # recovered close to truth
    assert panel_err < single_err  # and far better than the single fit


# ---------------------------------------------------------------------------
# PR-5: automatic same-grid interferer discovery.
#
# `discover_candidates` is pure/no-grad and therefore deterministic, so the
# detection contract is unit-tested precisely against a target placed at the
# truth of peak A (`_fit_target` — a deterministic stand-in for the converged
# first fit). The end-to-end `estimate_panel` loop layers the (stochastic,
# global-RNG-sensitive) multi-progenitor DE refit on top, so that path is only
# an *engagement* smoke — exact N recovery there is refit-quality-bound, not a
# property of discovery (the production refit is constrained by the calibration
# peak-shape prior the synthetic fixture lacks).
# ---------------------------------------------------------------------------


def test_discover_candidates_finds_same_grid_interferer() -> None:
    """Given the target fit to peak A, discovery proposes exactly one interferer
    seeded at peak B, on the target's own grid, sharing its calibration."""
    cal = _cal()
    obs = _two_peak(cal, n_interferer=1.5e5)
    panel = _fit_target(cal)
    cands = discover_candidates(
        panel, obs, panel_residual(panel, obs), None, threshold=8.0, min_isotopes=2
    )
    assert len(cands) == 1
    c = cands[0]
    assert abs(float(c.peak.mu.detach()) - 1.92e6) < 8000.0  # seeded at B (±8 s)
    assert 0.5 * 1.5e5 < float(c.peak.N_total.detach()) < 1.5 * 1.5e5  # ~ interferer N
    assert torch.equal(c.channel_mz, panel.progenitors[0].channel_mz)  # same grid
    assert c.calibration is cal  # shares the panel calibration


def test_discover_candidates_silent_on_clean() -> None:
    """A clean single-target obs has no coherent off-core residual → no candidate."""
    cal = _cal()
    obs = simulate_observation(
        _prog(cal, 2e5, 1.8e6), n_scans=150, half_window_ms=150000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(3),
    )
    panel = _fit_target(cal)
    cands = discover_candidates(
        panel, obs, panel_residual(panel, obs), None, threshold=8.0, min_isotopes=2
    )
    assert cands == []


def test_discover_candidates_rejects_subthreshold() -> None:
    """The count gate is the real guard: a tiny second species is ignored, a
    substantial one is proposed (same fixture, only the interferer N changes)."""
    cal = _cal()
    panel = _fit_target(cal)
    tiny = _two_peak(cal, n_interferer=50.0)
    big = _two_peak(cal, n_interferer=1.5e5)
    n_tiny = len(discover_candidates(
        panel, tiny, panel_residual(panel, tiny), None, threshold=8.0, min_isotopes=2
    ))
    n_big = len(discover_candidates(
        panel, big, panel_residual(panel, big), None, threshold=8.0, min_isotopes=2
    ))
    assert n_tiny == 0
    assert n_big == 1


def test_estimate_panel_discovers_and_corrects_target() -> None:
    """End-to-end: the live loop (no manual add) discovers the same-grid interferer
    and recovers a target the single-progenitor fit gets badly wrong — the
    automatic version of `test_manual_add_corrects_target`.

    `torch.manual_seed(0)` makes it deterministic: DE draws the global RNG, so a
    seed before observation construction pins the whole pipeline (the frozen
    calibration removed the shared-state drift; `fit_panel` additionally pins its
    own search)."""
    torch.manual_seed(0)
    cal = _cal()
    obs = _two_peak(cal, n_interferer=1.5e5)  # target 2e5@1.8e6 + interferer 1.5e5@1.92e6

    single = estimate_n(_prog(cal, 1.0, 1.8e6), obs, inference="map", optimizer="de", seed=0)
    single_err = abs(single["n_total"] - 2e5) / 2e5
    assert single_err > 0.30  # one progenitor cannot own two peaks

    panel = Panel([_prog(cal, 1.0, 1.8e6)], cal, background=False)
    res = estimate_panel(panel, obs, None, config=DiscoverConfig(), rt_prior_ms=1.8e6)
    assert res["n_discovered_interferers"] == 1  # found the interferer (default cap is 1)
    assert res["interference_flag"] is True
    panel_err = abs(res["n_total"] - 2e5) / 2e5
    assert panel_err < 0.30  # target recovered
    assert panel_err < single_err  # and far better than the single fit
    assert res["n_total_lo"] < res["n_total"] < res["n_total_hi"]  # finite, bracketing CI
