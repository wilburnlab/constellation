"""Tests for Counter estimation, schemas, table builders, and IO."""

from __future__ import annotations

import math
import tempfile

import pytest
import torch

from constellation.core.io.schemas import get_schema
from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.counter import (
    COUNTER_GLOBAL_CALIBRATION_TABLE,
    COUNTER_N_TABLE,
    COUNTER_PEPTIDE_PARAMS_TABLE,
    CounterResult,
    GlobalCalibration,
    Progenitor,
    calibration_to_table,
    counter_n_table,
    estimate_n,
    load_counter,
    peptide_params_to_table,
    save_counter,
    simulate_observation,
    simulate_panel_observation,
)

_PEP = Peptidoform(sequence="PEPTIDEKR")


def _cal() -> GlobalCalibration:
    return GlobalCalibration(
        n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0, alpha_mz=0.6, nu_mz=7.8
    )


def _prog(cal, n_total=1e5, mu=1.8e6, **kw) -> Progenitor:
    peak = HyperEMGPeak(
        N_total=n_total, mu=mu, sigma=6000.0, tau_r=6000.0, tau_l=3000.0, eta=0.85
    )
    return Progenitor.for_peptide(
        _PEP, [2, 3], cal, n_isotopes=3, peak=peak, **kw
    )


# ──────────────────────────────────────────────────────────────────────
# Round-trip recovery (headline)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("truth_n", [5e4, 1e5, 3e5])
def test_round_trip_recovery(truth_n: float) -> None:
    cal = _cal()
    truth = _prog(cal, n_total=truth_n, nu_intensity=6.0, c_mz_init=200.0)
    obs = simulate_observation(
        truth, n_scans=60, half_window_ms=30000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(11),
    )
    prog = _prog(cal, n_total=1.0, nu_intensity=5.0, c_mz_init=150.0)
    res = estimate_n(prog, obs, inference="map", optimizer="de", seed=0)

    assert res["inference_method"] == "map"
    assert res["iit_corrected"] is True
    n = res["n_total"]
    assert n > 1.0 and n < 1e9  # no collapse / blow-up
    assert abs(n - truth_n) / truth_n < 0.10  # within 10%
    assert res["n_total_lo"] < n < res["n_total_hi"]
    assert 1.0 < res["peak_sigma"] < 30.0  # seconds; physically sane width


def test_low_n_detection_bias_corrected() -> None:
    """The complete left-censored-Poisson likelihood removes the detection-edge
    upward bias. At 1e4 ions (where the legacy unconditioned Student-t was
    +30-44% high) the count-native term recovers within ~15% and covers the
    truth -- without collapsing N toward the floor."""
    cal = _cal()
    truth = _prog(cal, n_total=1e4, nu_intensity=6.0, c_mz_init=200.0)
    obs = simulate_observation(
        truth, n_scans=60, half_window_ms=30000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(300),
    )
    prog = _prog(cal, n_total=1.0, nu_intensity=5.0, c_mz_init=150.0)
    res = estimate_n(prog, obs, inference="map", optimizer="de", seed=0)
    assert res["n_total"] > 100.0  # no collapse to the lower bound
    assert abs(res["n_total"] - 1e4) / 1e4 < 0.15
    assert res["n_total_lo"] <= 1e4 <= res["n_total_hi"]


def test_high_n_not_clipped_by_default_bound() -> None:
    """A genuinely abundant species (N ≫ 1e10, e.g. a spiked calibrant) is
    recovered, not pinned at the legacy fixed 1e10 DE ceiling. The seed-scaled
    upper bound (1000× the data seed) gives headroom; without it the MAP clamps
    at exactly 1e10 for every bright peptide."""
    cal = _cal()
    truth = _prog(cal, n_total=5e10, nu_intensity=6.0, c_mz_init=200.0)
    obs = simulate_observation(
        truth, n_scans=60, half_window_ms=30000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(13),
    )
    prog = _prog(cal, n_total=1.0, nu_intensity=5.0, c_mz_init=150.0)
    res = estimate_n(prog, obs, inference="map", optimizer="de", seed=0)
    assert res["n_total"] > 1.5e10  # off the old ceiling, not clamped at 1e10
    assert abs(res["n_total"] - 5e10) / 5e10 < 0.25  # recovers the abundant truth


def test_isotope_fraction_correction_recovered() -> None:
    """The learnable isotope-fraction correction recovers a real M+1/M+2 shift
    (the high-resolution ¹⁵N/¹³C convolution effect) without biasing N."""
    cal = _cal()
    truth = _prog(cal, n_total=2e5, nu_intensity=6.0, c_mz_init=200.0)
    with torch.no_grad():
        truth.isotope_energy_offset.copy_(
            torch.tensor([0.6, -0.3], dtype=torch.float64)
        )  # inflate M+1, deflate M+2
    truth_pk = truth.p_k.detach().clone()
    # the perturbation must actually move the fractions off theoretical
    theo = torch.softmax(truth.log_p_theo, dim=0)
    assert float((truth_pk - theo).abs().max()) > 0.1

    obs = simulate_observation(
        truth, n_scans=80, generator=torch.Generator().manual_seed(21)
    )
    prog = _prog(cal, n_total=1.0, nu_intensity=5.0, c_mz_init=150.0)
    res = estimate_n(prog, obs, inference="map", optimizer="de", seed=0)
    assert torch.allclose(prog.p_k.detach(), truth_pk, atol=0.03)
    assert abs(res["n_total"] - 2e5) / 2e5 < 0.12


def test_vb_credible_interval_recovery() -> None:
    """VB recovers N_total with a credible interval at a practical ion count."""
    cal = _cal()
    truth = _prog(cal, n_total=1e5, nu_intensity=6.0, c_mz_init=200.0)
    obs = simulate_observation(
        truth, n_scans=60, half_window_ms=30000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(200),
    )
    torch.manual_seed(0)
    prog = _prog(cal, n_total=1.0, nu_intensity=5.0, c_mz_init=150.0)
    res = estimate_n(prog, obs, inference="vb", optimizer="de", seed=0)

    assert res["inference_method"] == "vb"
    assert res["final_elbo"] is not None and math.isfinite(res["final_elbo"])
    n = res["n_total"]
    assert math.isfinite(n) and n > 1.0
    assert abs(n - 1e5) / 1e5 < 0.12
    assert res["n_total_lo"] < n < res["n_total_hi"]
    assert res["n_total_hi"] > res["n_total_lo"]  # a non-degenerate interval


def test_vb_low_signal_recovers_and_stays_finite() -> None:
    """At the detection edge the η degeneracy + extreme MC draws would NaN a
    naive VB; the optimizer NaN-guard + η prior keep it finite. With the
    count-native likelihood the low-N detection bias is gone, so VB also
    recovers a 1e4-ion peptide (within a generous bound for the wider low-count
    posterior) and brackets the truth."""
    cal = _cal()
    truth = _prog(cal, n_total=1e4, nu_intensity=6.0, c_mz_init=200.0)
    obs = simulate_observation(
        truth, n_scans=60, generator=torch.Generator().manual_seed(200)
    )
    torch.manual_seed(0)
    prog = _prog(cal, n_total=1.0, nu_intensity=5.0, c_mz_init=150.0)
    res = estimate_n(prog, obs, inference="vb", optimizer="de", seed=0)
    assert math.isfinite(res["n_total"]) and res["n_total"] > 100.0
    assert math.isfinite(res["n_total_lo"]) and math.isfinite(res["n_total_hi"])
    assert abs(res["n_total"] - 1e4) / 1e4 < 0.30
    assert res["n_total_lo"] <= 1e4 <= res["n_total_hi"]


def test_estimate_n_rejects_unknown_inference() -> None:
    cal = _cal()
    prog = _prog(cal)
    obs = simulate_observation(prog, n_scans=20, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        estimate_n(prog, obs, inference="mcmc")


def test_make_log_prior_terms() -> None:
    from constellation.massspec.counter import make_log_prior

    assert make_log_prior() is None  # nothing active
    lp = make_log_prior(rt_prior_ms=1.8e6, rt_sigma_ms=5000.0, eta_center=1.0, eta_sigma=1.0)
    params = {
        "peak.mu": torch.tensor([1.8e6, 1.8e6 + 5000.0], dtype=torch.float64),
        "peak.logit_eta": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }
    out = lp(params)
    assert out.shape == (2,)
    # at the prior centers the penalty is 0; one σ off → −0.5
    assert float(out[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(out[1]) == pytest.approx(-0.5, abs=1e-9)

    # peak-shape hyperprior term (StagedCalibration → VB)
    lp2 = make_log_prior(
        shape_centers={"peak.log_sigma": 8.0},
        shape_sigmas={"peak.log_sigma": 0.5},
    )
    p2 = {"peak.log_sigma": torch.tensor([8.0, 8.5], dtype=torch.float64)}
    o2 = lp2(p2)
    assert float(o2[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(o2[1]) == pytest.approx(-0.5, abs=1e-9)  # 1 σ off


# ──────────────────────────────────────────────────────────────────────
# Additive interference + attribution (two co-eluting progenitors)
# ──────────────────────────────────────────────────────────────────────


def test_two_coeluting_progenitors_recovered() -> None:
    from constellation.core.optim import AdamOptimizer
    from constellation.massspec.counter.orchestrate import seed_peak_from_observation
    from constellation.massspec.counter.panel import Panel

    cal = _cal()
    # Two species sharing the m/z grid, co-eluting with overlapping tails but
    # resolvable apexes (~7σ apart) so attribution can recover the split.
    mu_a, mu_b = 1.80e6, 1.842e6
    a = _prog(cal, n_total=2e5, mu=mu_a, nu_intensity=6.0, c_mz_init=200.0)
    b = _prog(cal, n_total=1e5, mu=mu_b, nu_intensity=6.0, c_mz_init=200.0)
    rt = torch.linspace(1.784e6, 1.866e6, 140, dtype=torch.float64)
    obs = simulate_panel_observation(
        [a, b], rt=rt, iit_ms=20.0, generator=torch.Generator().manual_seed(7)
    )

    # Fit a fixed two-candidate panel (calibration frozen — not a progenitor param).
    ea = _prog(cal, n_total=1.0, mu=mu_a, nu_intensity=5.0, c_mz_init=150.0)
    eb = _prog(cal, n_total=1.0, mu=mu_b, nu_intensity=5.0, c_mz_init=150.0)
    seed_peak_from_observation(ea, obs, rt_prior_ms=mu_a)
    seed_peak_from_observation(eb, obs, rt_prior_ms=mu_b)
    panel = Panel([ea, eb], cal)
    params = [p for q in (ea, eb) for p in q.parameters()]
    opt = AdamOptimizer(params, lr=0.03)
    closure = lambda: -panel.log_prob(obs)  # noqa: E731
    for _ in range(1800):
        opt.step(closure)

    na = float(ea.peak.N_total.detach())
    nb = float(eb.peak.N_total.detach())
    # Both species recovered, and the additive total is well constrained.
    assert abs(na - 2e5) / 2e5 < 0.15
    assert abs(nb - 1e5) / 1e5 < 0.20
    assert na > nb
    assert abs((na + nb) - 3e5) / 3e5 < 0.10


# ──────────────────────────────────────────────────────────────────────
# Schemas + table builders + IO
# ──────────────────────────────────────────────────────────────────────


def test_schemas_registered() -> None:
    assert get_schema("CounterN").equals(COUNTER_N_TABLE)
    assert get_schema("CounterGlobalCalibration").equals(
        COUNTER_GLOBAL_CALIBRATION_TABLE
    )
    assert get_schema("CounterPeptideParams").equals(COUNTER_PEPTIDE_PARAMS_TABLE)


def test_table_builders_and_io_roundtrip() -> None:
    cal = _cal()
    truth = _prog(cal, n_total=1e5, nu_intensity=6.0, c_mz_init=200.0)
    obs = simulate_observation(truth, n_scans=50, generator=torch.Generator().manual_seed(5))
    prog = _prog(cal, n_total=1.0, nu_intensity=5.0, c_mz_init=150.0)
    res = estimate_n(prog, obs, inference="map", optimizer="adam")

    rec = {
        **res,
        "acquisition_id": 0,
        "target_id": 9,
        "modified_sequence": "PEPTIDEKR",
        "precursor_charge": 2,
    }
    n_tbl = counter_n_table([rec])
    cal_tbl = calibration_to_table(cal, acquisition_id=0)
    pp_tbl = peptide_params_to_table(
        [prog], acquisition_id=0, target_ids=[9], modified_sequences=["PEPTIDEKR"]
    )
    assert n_tbl.schema.equals(COUNTER_N_TABLE)
    assert cal_tbl.schema.equals(COUNTER_GLOBAL_CALIBRATION_TABLE)
    assert pp_tbl.schema.equals(COUNTER_PEPTIDE_PARAMS_TABLE)
    assert cal_tbl.column("alpha_model")[0].as_py() == "linear"
    assert len(pp_tbl.column("c_mz")[0].as_py()) == 3

    with tempfile.TemporaryDirectory() as d:
        save_counter(
            CounterResult(counter_n=n_tbl, global_calibration=cal_tbl, peptide_params=pp_tbl),
            d,
        )
        loaded = load_counter(d)
    assert loaded.counter_n.schema.equals(COUNTER_N_TABLE)
    assert loaded.counter_n.column("n_total")[0].as_py() == pytest.approx(
        res["n_total"]
    )
    assert loaded.global_calibration.column("nu_mz")[0].as_py() == pytest.approx(7.8)
    assert loaded.peptide_params.num_rows == 1
