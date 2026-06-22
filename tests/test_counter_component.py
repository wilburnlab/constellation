"""PR-G — component co-fit (estimate_component). See §9 of the design note."""

from __future__ import annotations

import pytest
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.counter import (
    DiscoverConfig,
    GlobalCalibration,
    Panel,
    Progenitor,
    estimate_component,
    estimate_panel,
    simulate_observation,
    simulate_panel_observation,
)

_PEP = Peptidoform(sequence="PEPTIDEKR")


def _cal() -> GlobalCalibration:
    return GlobalCalibration(
        n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0, alpha_mz=0.6, nu_mz=7.8
    )


def _prog(cal: GlobalCalibration, n_total: float, mu: float) -> Progenitor:
    peak = HyperEMGPeak(
        N_total=n_total, mu=mu, sigma=6000.0, tau_r=6000.0, tau_l=3000.0, eta=0.85
    )
    return Progenitor.for_peptide(
        _PEP, [2, 3], cal, n_isotopes=3, peak=peak, nu_intensity=6.0, c_mz_init=200.0
    )


def test_single_member_matches_estimate_panel() -> None:
    # a 1-member component is estimate_panel without the discovery loop; on a clean
    # obs (discovery finds nothing) the fit + report are identical.
    cal = _cal()
    obs = simulate_observation(
        _prog(cal, 1e5, 1.8e6), n_scans=40, generator=torch.Generator().manual_seed(0)
    )
    panel_res = estimate_panel(
        Panel([_prog(cal, 1.0, 1.8e6)], cal), obs, None,
        config=DiscoverConfig(), rt_prior_ms=1.8e6,
    )
    comp_res = estimate_component(
        [_prog(cal, 1.0, 1.8e6)], obs, config=DiscoverConfig(), rt_priors_ms=[1.8e6]
    )
    assert len(comp_res) == 1
    assert comp_res[0]["n_total"] == pytest.approx(panel_res["n_total"], rel=1e-6)
    assert comp_res[0]["n_total_lo"] == pytest.approx(panel_res["n_total_lo"], rel=1e-6)


def test_two_member_blend_recovers_both() -> None:
    # an additive same-grid blend at well-separated RTs; co-fitting the (known) two
    # members recovers each N, where one progenitor cannot own both peaks.
    # Seed the global RNG before the sim: simulate_panel_observation's StudentT m/z
    # noise draws the global RNG, so the obs (hence the fit) is order-dependent
    # otherwise (fit_panel pins its OWN search, not the observation construction).
    torch.manual_seed(0)
    cal = _cal()
    a = _prog(cal, 2e5, 1.80e6)
    b = _prog(cal, 1e5, 1.92e6)
    rt = torch.linspace(1.70e6, 2.02e6, 90, dtype=torch.float64)
    obs = simulate_panel_observation(
        [a, b], rt=rt, generator=torch.Generator().manual_seed(0)
    )
    # fresh members, seeded at the two PSM RTs (breaks the co-eluting symmetry)
    res = estimate_component(
        [_prog(cal, 1.0, 1.8e6), _prog(cal, 1.0, 1.8e6)],
        obs,
        config=DiscoverConfig(),
        rt_priors_ms=[1.80e6, 1.92e6],
    )
    assert len(res) == 2
    assert abs(res[0]["n_total"] - 2e5) / 2e5 < 0.3
    assert abs(res[1]["n_total"] - 1e5) / 1e5 < 0.3
    # both members flagged as co-isobaric company
    assert res[0]["n_discovered_interferers"] == 1 and res[0]["interference_flag"] is True


def test_estimate_component_rt_prior_outside_window_does_not_crash() -> None:
    # a member whose PSM RT is outside the obs window must not invert its μ bound
    # (lo > hi would break the DE fit); it gets the default bound and fits where it can.
    cal = _cal()
    obs = simulate_observation(
        _prog(cal, 2e5, 1.8e6), n_scans=40, generator=torch.Generator().manual_seed(0)
    )
    rt_hi = float(obs.rt.max())
    res = estimate_component(
        [_prog(cal, 1.0, 1.8e6), _prog(cal, 1.0, 1.8e6)],
        obs,
        config=DiscoverConfig(),
        rt_priors_ms=[1.8e6, rt_hi + 5.0e5],  # second prior far outside the obs window
    )
    assert len(res) == 2
    assert all(r["n_total"] > 0 or r["n_total"] == 0 for r in res)  # ran, no inverted-bound crash
    import math

    assert all(math.isfinite(r["n_total"]) for r in res)


def test_estimate_component_validates_rt_priors_length() -> None:
    cal = _cal()
    obs = simulate_observation(
        _prog(cal, 1e5, 1.8e6), n_scans=30, generator=torch.Generator().manual_seed(1)
    )
    with pytest.raises(ValueError, match="rt_priors_ms"):
        estimate_component([_prog(cal, 1.0, 1.8e6)], obs, rt_priors_ms=[1.8e6, 1.9e6])
