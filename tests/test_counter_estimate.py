"""Tests for Counter estimation, schemas, table builders, and IO."""

from __future__ import annotations

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


def test_estimate_n_rejects_vb_for_now() -> None:
    cal = _cal()
    prog = _prog(cal)
    obs = simulate_observation(prog, n_scans=20, generator=torch.Generator().manual_seed(0))
    with pytest.raises(NotImplementedError):
        estimate_n(prog, obs, inference="vb")


# ──────────────────────────────────────────────────────────────────────
# Additive interference + attribution (two co-eluting progenitors)
# ──────────────────────────────────────────────────────────────────────


def test_two_coeluting_progenitors_recovered() -> None:
    from constellation.core.optim import AdamOptimizer
    from constellation.massspec.counter.orchestrate import seed_peak_from_observation
    from constellation.massspec.counter.panel import Panel

    cal = _cal()
    # Two species, shared m/z grid, resolved by distinct elution apexes.
    a = _prog(cal, n_total=2e5, mu=1.80e6, nu_intensity=6.0, c_mz_init=200.0)
    b = _prog(cal, n_total=1e5, mu=1.815e6, nu_intensity=6.0, c_mz_init=200.0)
    rt = torch.linspace(1.78e6, 1.84e6, 120, dtype=torch.float64)
    obs = simulate_panel_observation(
        [a, b], rt=rt, iit_ms=20.0, generator=torch.Generator().manual_seed(7)
    )

    # Fit a fixed two-candidate panel (calibration frozen — not a progenitor param).
    ea = _prog(cal, n_total=1.0, mu=1.80e6, nu_intensity=5.0, c_mz_init=150.0)
    eb = _prog(cal, n_total=1.0, mu=1.815e6, nu_intensity=5.0, c_mz_init=150.0)
    seed_peak_from_observation(ea, obs, rt_prior_ms=1.80e6)
    seed_peak_from_observation(eb, obs, rt_prior_ms=1.815e6)
    panel = Panel([ea, eb], cal)
    params = [p for q in (ea, eb) for p in q.parameters()]
    opt = AdamOptimizer(params, lr=0.03)
    closure = lambda: -panel.log_prob(obs)  # noqa: E731
    for _ in range(1500):
        opt.step(closure)

    na = float(ea.peak.N_total.detach())
    nb = float(eb.peak.N_total.detach())
    total = na + nb
    # The summed ion count is well constrained; the split is harder but should
    # land the dominant species above the minor one and the total near truth.
    assert abs(total - 3e5) / 3e5 < 0.20
    assert na > nb


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
