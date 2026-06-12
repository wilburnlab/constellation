"""Tests for the real-data pivot: XIC_TRACE → CounterObservation."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.counter import (
    GlobalCalibration,
    Progenitor,
    estimate_n,
    observation_for_progenitor,
    observation_from_trace,
    observation_to_trace,
    simulate_observation,
)

_PEP = Peptidoform(sequence="ELVISLIVEK")


def _cal() -> GlobalCalibration:
    return GlobalCalibration(
        n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0, alpha_mz=0.6,
        nu_mz=7.8, mz_offset_ppm=2.0, d_mz_da=[0.0, 0.001, 0.0],
    )


def _truth(cal, n_total=1.2e5):
    pk = HyperEMGPeak(
        N_total=n_total, mu=1.8e6, sigma=6000.0, tau_r=6000.0, tau_l=3000.0, eta=0.85
    )
    return Progenitor.for_peptide(
        _PEP, [2, 3], cal, n_isotopes=3, peak=pk, nu_intensity=6.0, c_mz_init=200.0
    )


def _trace(seed=7, n_total=1.2e5):
    cal = _cal()
    truth = _truth(cal, n_total)
    obs = simulate_observation(
        truth, n_scans=60, half_window_ms=30000.0, iit_ms=20.0,
        generator=torch.Generator().manual_seed(seed),
    )
    trace, scan_meta = observation_to_trace(
        obs, acquisition_id=0, target_id=9, modified_sequence="ELVISLIVEK"
    )
    return obs, trace, scan_meta, truth


def test_observation_from_trace_round_trips() -> None:
    """to_trace → from_trace recovers the observation exactly on the observed
    channels (and the mask / scan axis identically)."""
    obs, trace, scan_meta, truth = _trace()
    obs2 = observation_from_trace(
        trace, scan_meta, channel_z=truth.channel_z,
        channel_isotope=truth.channel_isotope, channel_mz=truth.channel_mz,
        target_id=9,
    )
    assert obs2.n_scans == obs.n_scans
    assert obs2.n_channels == obs.n_channels
    assert torch.equal(obs2.mask, obs.mask)
    assert torch.allclose(obs2.rt, obs.rt)
    assert torch.allclose(obs2.iit, obs.iit)
    m = obs.mask
    assert torch.allclose(obs2.intensity[m], obs.intensity[m])
    assert torch.allclose(obs2.mz_error[m], obs.mz_error[m])
    # unobserved channels are zeroed
    assert float(obs2.intensity[~m].abs().max()) == 0.0


def test_estimate_n_through_pivot_recovers_n() -> None:
    """The real-data path (trace → observation_for_progenitor → estimate_n)
    recovers N_total."""
    _obs, trace, scan_meta, _truth_p = _trace(seed=7, n_total=1.2e5)
    prog = Progenitor.for_peptide(
        _PEP, [2, 3], _cal(), n_isotopes=3, nu_intensity=5.0, c_mz_init=150.0
    )
    obs_in = observation_for_progenitor(prog, trace, scan_meta, target_id=9)
    res = estimate_n(prog, obs_in, inference="map", optimizer="de", seed=0)
    assert abs(res["n_total"] - 1.2e5) / 1.2e5 < 0.10
    assert res["n_total_lo"] <= 1.2e5 <= res["n_total_hi"]


def test_scan_axis_from_metadata_includes_nondetections() -> None:
    """The scan axis comes from scan_metadata, so window scans with no detected
    channel are present (mask all-False) — required for the censored term."""
    obs, trace, scan_meta, truth = _trace()
    # Keep the full scan window but drop trace rows for the first 20 scans.
    trace_trim = trace.filter(pc.greater_equal(trace.column("scan"), 20))
    obs2 = observation_from_trace(
        trace_trim, scan_meta, channel_z=truth.channel_z,
        channel_isotope=truth.channel_isotope, channel_mz=truth.channel_mz,
        target_id=9,
    )
    assert obs2.n_scans == scan_meta.num_rows  # axis preserved
    # scans 0..19 are now all non-detection
    assert not bool(obs2.mask[:20].any())
    assert bool(obs2.mask[20:].any())


def test_dedup_keeps_most_intense() -> None:
    """Two peaks on one (scan, channel) (e.g. all_in_window interference) → the
    most intense is kept."""
    _obs, trace, scan_meta, truth = _trace()
    row = trace.slice(0, 1)
    scan0 = row.column("scan")[0].as_py()
    chg0 = row.column("precursor_charge")[0].as_py()
    iso0 = row.column("isotope")[0].as_py()
    big = float(row.column("intensity")[0].as_py()) * 5.0
    # a competing, much brighter peak at the same (scan, charge, isotope)
    dup = row.set_column(
        trace.schema.get_field_index("intensity"),
        "intensity", pa.array([big], pa.float64()),
    )
    dup = dup.set_column(
        trace.schema.get_field_index("mz_error_ppm"),
        "mz_error_ppm", pa.array([99.0], pa.float64()),
    )
    trace2 = pa.concat_tables([trace, dup.cast(trace.schema)])
    obs2 = observation_from_trace(
        trace2, scan_meta, channel_z=truth.channel_z,
        channel_isotope=truth.channel_isotope, channel_mz=truth.channel_mz,
        target_id=9,
    )
    charges = [2, 3]
    c = charges.index(chg0) * 3 + iso0
    s = scan0  # scans are 0..S-1 in order
    assert float(obs2.intensity[s, c]) == pytest.approx(big)
    assert float(obs2.mz_error[s, c]) == pytest.approx(99.0)  # the bright peak's


def test_off_grid_and_target_rows_dropped() -> None:
    """Rows for a different target, or an off-grid (charge, isotope), don't land
    in the observation."""
    obs, trace, scan_meta, truth = _trace()
    # all rows belong to target 9 → filtering to target 5 yields an empty obs
    obs_other = observation_from_trace(
        trace, scan_meta, channel_z=truth.channel_z,
        channel_isotope=truth.channel_isotope, channel_mz=truth.channel_mz,
        target_id=5,
    )
    assert not bool(obs_other.mask.any())
    # a grid that lacks charge 3 → charge-3 rows are dropped
    keep = truth.channel_z == 2
    obs_z2 = observation_from_trace(
        trace, scan_meta, channel_z=truth.channel_z[keep],
        channel_isotope=truth.channel_isotope[keep], channel_mz=truth.channel_mz[keep],
        target_id=9,
    )
    assert obs_z2.n_channels == 3  # charge 2 × 3 isotopes
    # only what the charge-2 channels held in the full observation
    assert bool(obs_z2.mask.any())


def test_empty_scan_metadata_raises() -> None:
    _obs, trace, _sm, truth = _trace()
    empty = pa.table(
        {"scan": pa.array([], pa.int32()), "rt": pa.array([], pa.float64()),
         "iit": pa.array([], pa.float32())}
    )
    with pytest.raises(ValueError):
        observation_from_trace(
            trace, empty, channel_z=truth.channel_z,
            channel_isotope=truth.channel_isotope, channel_mz=truth.channel_mz,
        )
