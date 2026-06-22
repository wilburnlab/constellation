"""Real-data deconstruction steps 1–4: per-ion N, per-cell decomposition +
per-progenitor m/z centers, cell-level loss exclusion, and the ion→progenitor
attribution map. See docs/plans/counter-real-data-deconstruction.md."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.core.stats.units import da_to_ppm
from constellation.massspec.counter import (
    CounterObservation,
    CounterResult,
    GlobalCalibration,
    Panel,
    Progenitor,
    load_counter,
    observation_from_trace,
    observation_to_trace,
    panel_attribution_table,
    panel_cell_log_prob,
    panel_log_prob,
    save_counter,
    simulate_observation,
)
from constellation.massspec.counter.schemas import COUNTER_PEAK_ATTRIBUTION_TABLE

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


def _obs(prog: Progenitor, seed: int = 0) -> CounterObservation:
    return simulate_observation(prog, n_scans=40, generator=torch.Generator().manual_seed(seed))


# ──────────────────────────────────────────────────────────────────────
# Step 1 — per-ion recovered count N_obs = I·τ/α at the observe layer
# ──────────────────────────────────────────────────────────────────────


def test_recovered_count_matches_itau_over_alpha() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    obs = _obs(prog)
    n_obs = obs.recovered_count(cal)
    gain = cal.gain(obs.channel_z.to(torch.float64))
    expected = obs.intensity * obs.iit[:, None] / gain[None, :]
    # observed cells = I·τ/α (the accumulated-count floor is far below real counts)
    assert torch.allclose(n_obs[obs.mask], expected[obs.mask])
    # non-detections are zero, not the count floor
    assert bool(torch.all(n_obs[~obs.mask] == 0.0))


# ──────────────────────────────────────────────────────────────────────
# Step 2 — per-cell decomposition is byte-identical to the scalar likelihood
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_prog", [1, 2])
def test_panel_cell_terms_sum_to_scalar(n_prog: int) -> None:
    cal = _cal()
    progs = [_progenitor(cal) for _ in range(n_prog)]
    obs = _obs(progs[0])
    t = panel_cell_log_prob(progs, obs, cal)
    zeros = torch.zeros_like(t.intensity)
    manual = (
        torch.where(t.observed, t.intensity, zeros).sum()
        + torch.where(t.observed, t.mz, zeros).sum()
        + torch.where(t.censored_mask, t.censored, zeros).sum()
        + torch.where(t.observed, t.jacobian, zeros).sum()
    )
    scalar = panel_log_prob(progs, obs, cal)
    assert torch.allclose(manual, scalar)
    # default masks == the raw detection mask (no exclusion)
    assert torch.equal(t.observed, obs.mask)
    assert torch.equal(t.censored_mask, ~obs.mask)
    # responsibilities partition each cell among the modeled species
    assert torch.allclose(t.gamma.sum(dim=0), torch.ones_like(t.gamma[0]))


def test_mz_center_per_progenitor_reduces_and_shifts() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    mz, z, iso = prog.channel_mz, prog.channel_z, prog.channel_isotope
    # reference_mz=None (default) == reference_mz=channel_mz (the grid-defining case)
    c3 = cal.mz_center_ppm(mz, z.to(torch.float64), iso)
    c4 = cal.mz_center_ppm(mz, z.to(torch.float64), iso, reference_mz=mz)
    assert torch.allclose(c3, c4)
    # a candidate offset by Δ Da shifts the center by exactly da_to_ppm(Δ, ref)
    delta = 0.004
    c_shift = cal.mz_center_ppm(mz + delta, z.to(torch.float64), iso, reference_mz=mz)
    expected = da_to_ppm(torch.full_like(mz, delta), mz)
    assert torch.allclose(c_shift - c4, expected)


# ──────────────────────────────────────────────────────────────────────
# Step 4 — cell-level loss exclusion (L4 mask + D9 γ-purity benchmark knob)
# ──────────────────────────────────────────────────────────────────────


def test_exclude_mask_drops_cells_from_loss() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    obs = _obs(prog)
    base = panel_log_prob([prog], obs, cal)
    # all-False exclusion is a no-op
    no_ex = panel_log_prob([prog], obs, cal, exclude_mask=torch.zeros_like(obs.mask))
    assert torch.allclose(no_ex, base)
    # exclude one observed cell → it leaves both the observed and censored sets
    ex = torch.zeros_like(obs.mask)
    obs_cells = obs.mask.nonzero(as_tuple=False)
    s0, c0 = int(obs_cells[0, 0]), int(obs_cells[0, 1])
    ex[s0, c0] = True
    t = panel_cell_log_prob([prog], obs, cal, exclude_mask=ex)
    assert not bool(t.observed[s0, c0]) and not bool(t.censored_mask[s0, c0])
    # the cell is still inferred (predicted), just not scored
    assert torch.isfinite(t.i_pred[0, s0, c0])
    excluded = panel_log_prob([prog], obs, cal, exclude_mask=ex)
    assert not torch.allclose(excluded, base)


def test_gamma_loss_threshold_excludes_blended_cells() -> None:
    cal = _cal()
    # two identical progenitors → every observed cell is split γ = 0.5 / 0.5
    p1, p2 = _progenitor(cal), _progenitor(cal)
    obs = _obs(p1)
    base = panel_log_prob([p1, p2], obs, cal)
    # threshold below the 0.5 split: nothing excluded
    keep = panel_log_prob([p1, p2], obs, cal, gamma_loss_threshold=0.4)
    assert torch.allclose(keep, base)
    # threshold above 0.5: every observed (blended) cell is dropped from the loss
    t = panel_cell_log_prob([p1, p2], obs, cal, gamma_loss_threshold=0.99)
    assert int(t.observed.sum()) == 0
    # non-detections still score the censored tail (γ-threshold only touches observed)
    assert torch.equal(t.censored_mask, ~obs.mask)
    dropped = panel_log_prob([p1, p2], obs, cal, gamma_loss_threshold=0.99)
    cens_only = torch.where(~obs.mask, t.censored, torch.zeros_like(t.censored)).sum()
    assert torch.allclose(dropped, cens_only)


def test_panel_options_forwarded_from_constructor() -> None:
    cal = _cal()
    p1, p2 = _progenitor(cal), _progenitor(cal)
    obs = _obs(p1)
    panel = Panel([p1, p2], cal, gamma_loss_threshold=0.99)
    assert int(panel.cell_terms(obs).observed.sum()) == 0
    assert torch.allclose(panel.log_prob(obs), panel_log_prob([p1, p2], obs, cal, gamma_loss_threshold=0.99))


# ──────────────────────────────────────────────────────────────────────
# Step 3 — peak_id threading + the ion→progenitor attribution map
# ──────────────────────────────────────────────────────────────────────


def _round_trip_obs(prog: Progenitor) -> tuple[CounterObservation, pa.Table]:
    obs = _obs(prog, seed=7)
    trace, scan_meta = observation_to_trace(
        obs, acquisition_id=0, target_id=5, modified_sequence="PEPTIDEKR"
    )
    return observation_from_trace(
        trace,
        scan_meta,
        channel_z=prog.channel_z,
        channel_isotope=prog.channel_isotope,
        channel_mz=prog.channel_mz,
        target_id=5,
        level=1,
    ), trace


def test_source_identity_threaded_through_observe() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    obs2, _trace = _round_trip_obs(prog)
    # scan axis (S,) + observed m/z (S, C) — the stable (scan, mz_observed) key
    assert obs2.scan is not None and obs2.scan.shape == obs2.rt.shape
    assert obs2.source_mz is not None and obs2.source_mz.shape == obs2.intensity.shape
    # source_mz is finite exactly at the observed cells; NaN elsewhere
    assert torch.equal(torch.isfinite(obs2.source_mz), obs2.mask)
    # the observed m/z is a real measured m/z near its channel (not a row index /
    # zero / NaN) — within 0.1% even after the simulator's fat-tailed m/z noise
    s_i, c_i = obs2.mask.nonzero(as_tuple=True)
    rel = (obs2.source_mz[s_i, c_i] - obs2.channel_mz[c_i]).abs() / obs2.channel_mz[c_i]
    assert torch.all(rel < 1e-3)


def test_panel_attribution_table_single_progenitor() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    obs2, _ = _round_trip_obs(prog)
    panel = Panel([prog], cal)
    tbl = panel_attribution_table(panel, obs2, acquisition_id=0, target_id=5)
    assert tbl.schema.equals(COUNTER_PEAK_ATTRIBUTION_TABLE)
    # single progenitor → γ ≡ 1, one row per observed cell, all the target
    assert tbl.num_rows == int(obs2.mask.sum())
    resp = tbl.column("responsibility").to_numpy()
    assert np.allclose(resp, 1.0)
    assert pc.all(tbl.column("is_target")).as_py()
    # every emitted row carries a real observed peak identity (scan + finite m/z)
    assert pc.min(tbl.column("scan")).as_py() >= 0
    assert pc.all(pc.is_finite(tbl.column("mz_observed"))).as_py()


def test_panel_attribution_table_two_progenitors_split() -> None:
    cal = _cal()
    p1, p2 = _progenitor(cal), _progenitor(cal)
    obs2, _ = _round_trip_obs(p1)
    panel = Panel([p1, p2], cal)
    tbl = panel_attribution_table(panel, obs2, acquisition_id=0, target_id=5)
    # two identical owners → two rows per observed cell, each γ = 0.5
    assert tbl.num_rows == 2 * int(obs2.mask.sum())
    resp = tbl.column("responsibility").to_numpy()
    assert np.allclose(resp, 0.5)
    # both progenitor indices present; exactly one flagged target
    idx = set(tbl.column("progenitor_index").to_pylist())
    assert idx == {0, 1}


def test_counter_result_roundtrips_peak_attribution(tmp_path) -> None:
    cal = _cal()
    prog = _progenitor(cal)
    obs2, _ = _round_trip_obs(prog)
    attribution = panel_attribution_table(Panel([prog], cal), obs2, acquisition_id=0, target_id=5)
    counter_n = pa.table({"acquisition_id": pa.array([0], pa.int64())})
    result = CounterResult(counter_n=counter_n, peak_attribution=attribution)
    save_counter(result, tmp_path / "bundle")
    loaded = load_counter(tmp_path / "bundle")
    assert loaded.peak_attribution is not None
    assert loaded.peak_attribution.equals(attribution)


def test_attribution_emits_residual_row_for_unowned_cell() -> None:
    cal = _cal()
    prog = _progenitor(cal)
    obs2, _ = _round_trip_obs(prog)
    # move the peak far outside the window → ~0 predicted flux at every observed
    # scan, so the model owns no cell and every observed peak is residual.
    with torch.no_grad():
        prog.peak.mu.fill_(float(obs2.rt.min()) - 1.0e6)
    tbl = panel_attribution_table(Panel([prog], cal), obs2, acquisition_id=0, target_id=5)
    assert -1 in tbl.column("progenitor_index").to_pylist()
    resid = tbl.filter(pc.equal(tbl.column("progenitor_index"), -1))
    # the residual rows preserve the raw (scan, mz_observed) identity (so the
    # "what's left" anti-join never loses an unexplained peak) and are not the target
    assert pc.min(resid.column("scan")).as_py() >= 0
    assert pc.all(pc.is_finite(resid.column("mz_observed"))).as_py()
    assert not pc.any(resid.column("is_target")).as_py()


def test_source_identity_stable_across_targets() -> None:
    # The whole point of (scan, mz_observed) over a trace row index: the SAME physical
    # peak extracted under two different targets gets the SAME key, so a cross-panel
    # anti-join can tell two panels claimed one peak. A per-(target,scan,ion) row index
    # cannot — the same peak occupies distinct rows under different targets.
    cal = _cal()
    prog = _progenitor(cal)
    z, k = int(prog.channel_z[0]), int(prog.channel_isotope[0])
    mz = float(prog.channel_mz[0])  # one shared physical peak at the M+0 z-channel
    trace = pa.table(
        {
            "target_id": pa.array([5, 6], pa.int64()),  # two targets, distinct rows
            "scan": pa.array([10, 10], pa.int32()),  # same scan
            "precursor_charge": pa.array([z, z], pa.int32()),
            "isotope": pa.array([k, k], pa.int32()),
            "intensity": pa.array([1.0e4, 1.0e4], pa.float64()),
            "mz_observed": pa.array([mz, mz], pa.float64()),  # same measured m/z
            "mz_error_ppm": pa.array([0.0, 0.0], pa.float64()),
            "level": pa.array([1, 1], pa.int32()),
        }
    )
    scan_meta = pa.table(
        {
            "scan": pa.array([10], pa.int32()),
            "rt": pa.array([100.0], pa.float64()),
            "iit": pa.array([20.0], pa.float64()),
        }
    )
    kw = dict(
        channel_z=prog.channel_z, channel_isotope=prog.channel_isotope,
        channel_mz=prog.channel_mz, level=1,
    )
    o5 = observation_from_trace(trace, scan_meta, target_id=5, **kw)
    o6 = observation_from_trace(trace, scan_meta, target_id=6, **kw)
    # the shared peak lands at the same cell with a byte-identical stable key
    s, c = (int(i) for i in o5.mask.nonzero(as_tuple=False)[0])
    assert int(o5.scan[s]) == int(o6.scan[s]) == 10
    assert float(o5.source_mz[s, c]) == float(o6.source_mz[s, c]) == mz


def test_component_attribution_carries_member_identities() -> None:
    # P1 #2: a component co-fit's known members must each keep their own target_id +
    # is_target, not all inherit the reference member's id as anonymous interferers.
    cal = _cal()
    p1, p2 = _progenitor(cal), _progenitor(cal)
    obs2, _ = _round_trip_obs(p1)
    panel = Panel([p1, p2], cal)
    tbl = panel_attribution_table(
        panel, obs2, acquisition_id=0, target_id=5, progenitor_target_ids=[5, 7]
    )
    rows = tbl.to_pylist()
    # progenitor 0 → target 5, progenitor 1 → target 7; both flagged as real targets
    by_prog = {(r["progenitor_index"], r["target_id"], r["is_target"]) for r in rows}
    assert (0, 5, True) in by_prog
    assert (1, 7, True) in by_prog
    # no row mislabels member 1's peaks as belonging to the reference target 5
    assert not any(r["progenitor_index"] == 1 and r["target_id"] == 5 for r in rows)
    # a length mismatch is rejected rather than silently mis-mapping
    with pytest.raises(ValueError, match="progenitor_target_ids"):
        panel_attribution_table(
            panel, obs2, acquisition_id=0, target_id=5, progenitor_target_ids=[5]
        )


# ──────────────────────────────────────────────────────────────────────
# Regression hardening: byte-identity, near-isobaric centers, vmap exclusion
# ──────────────────────────────────────────────────────────────────────


def _pre_refactor_scalar(progs, obs, cal) -> torch.Tensor:
    """Inlined copy of the pre-decomposition `panel_log_prob` (single shared m/z
    center, direct `obs.mask` reductions) — the frozen baseline that pins the
    refactor's byte-identity contract."""
    from constellation.massspec.counter import channels as ch
    from constellation.massspec.counter.attribution import (
        mz_mixture_log_prob,
        responsibilities,
    )

    iit = obs.iit
    i_preds, n_counts, mz_scales = [], [], []
    for q in progs:
        i_pred_q, n_count_q = q.predict(obs)
        i_preds.append(i_pred_q)
        n_counts.append(n_count_q)
        c_mz_ch = q.c_mz_per_channel()
        mz_scales.append(
            (c_mz_ch[None, :] / n_count_q.clamp(min=1.0) ** cal.alpha_mz).clamp(min=1e-12).sqrt()
        )
    i_pred_stack = torch.stack(i_preds, dim=0)
    n_count_total = torch.stack(n_counts, dim=0).sum(dim=0)
    gain_ch = cal.gain(obs.channel_z.to(iit.dtype))
    n_obs = obs.intensity * iit[:, None] / gain_ch[None, :]
    int_lp = ch.poisson_count_log_prob(n_obs, n_count_total)
    zeros = torch.zeros_like(int_lp)
    int_term = torch.where(obs.mask, int_lp, zeros).sum()
    center = cal.mz_center_ppm(obs.channel_mz, obs.channel_z.to(iit.dtype), obs.channel_isotope)
    gamma = responsibilities(i_pred_stack)
    component_lp = torch.stack(
        [ch.student_t_log_prob(obs.mz_error, center[None, :], mz_scales[q], cal.nu_mz)
         for q in range(len(progs))],
        dim=0,
    )
    mz_term = torch.where(obs.mask, mz_mixture_log_prob(gamma, component_lp), zeros).sum()
    cens_term = torch.where(obs.mask, zeros, ch.censored_log_prob(n_count_total, 1.0)).sum()
    log_jac = torch.log(iit[:, None]) - torch.log(gain_ch[None, :])
    jac_term = torch.where(obs.mask, log_jac, zeros).sum()
    return int_term + mz_term + cens_term + jac_term


@pytest.mark.parametrize("n_prog", [1, 2])
def test_panel_log_prob_byte_identical_to_pre_refactor(n_prog: int) -> None:
    cal = _cal()
    progs = [_progenitor(cal) for _ in range(n_prog)]
    obs = _obs(progs[0])
    assert torch.equal(panel_log_prob(progs, obs, cal), _pre_refactor_scalar(progs, obs, cal))


def test_near_isobaric_per_progenitor_centers_diverge() -> None:
    cal = _cal()
    p1, p2 = _progenitor(cal), _progenitor(cal)
    with torch.no_grad():
        p2.channel_mz += 0.01  # shift p2's grid → a distinct per-progenitor center
    obs = _obs(p1)  # built on p1's (unshifted) grid; mz_error clusters at p1's center
    t = panel_cell_log_prob([p1, p2], obs, cal)
    cells = obs.mask
    # distinct theoretical m/z → distinct per-progenitor m/z log-densities
    assert not torch.allclose(t.component_mz[0][cells], t.component_mz[1][cells])
    # the observed errors favor the on-center species (p1)
    m0 = t.component_mz[0][cells].mean().detach()
    m1 = t.component_mz[1][cells].mean().detach()
    assert float(m0) > float(m1)


def test_exclusion_branch_vmap_safe_under_de() -> None:
    import math

    from constellation.core.optim import DifferentialEvolution

    cal = _cal()
    # γ-threshold panel: the gamma.amax / bool-& exclusion ops must survive DE's vmap
    p1, p2 = _progenitor(cal), _progenitor(cal)
    obs = _obs(p1)
    panel = Panel([p1, p2], cal, gamma_loss_threshold=0.99)
    de = DifferentialEvolution(panel, pop_size=12, max_evals=120, seed=0)
    assert math.isfinite(panel.fit(obs, optimizer=de, max_iter=6).final_loss)
    # exclude_mask panel under DE too
    ex = torch.zeros_like(obs.mask)
    ex[: ex.shape[0] // 2] = True
    panel2 = Panel([_progenitor(cal)], cal, exclude_mask=ex)
    de2 = DifferentialEvolution(panel2, pop_size=12, max_evals=120, seed=0)
    assert math.isfinite(panel2.fit(obs, optimizer=de2, max_iter=6).final_loss)
