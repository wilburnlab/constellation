"""Counter — physically-grounded Bayesian ion-count estimation.

Estimates the latent integrated ion count `N_total` (and the elution flux
`N(t)`) of a peptide species from the joint m/z + intensity likelihood of
its observed ions, decoupled from RT windowing (the score is evaluable at
any hypothesis). Built on the `core.stats` primitives (HyperEMG peak,
Student-t, VI guides, hierarchical params) and the `massspec` peptide
chemistry + acquisition schemas.

Layout:
    iit              IIT ↔ accumulated-count conversion (the one IIT site)
    calibration      GlobalCalibration — per-acquisition shared calibrants
    channels         per-channel likelihood terms (intensity / m/z / censored)
    attribution      soft intensity-weighted peak attribution
    model            Progenitor (one species) + CounterObservation
    panel            Panel — additive joint score over progenitors
    simulate         forward simulation → observations / XIC traces
    observe          real-data pivot: XIC_TRACE + scan metadata → observation
    orchestrate      estimate_n (MAP/VB) + StagedCalibration + table builders
    schemas          COUNTER_N / GLOBAL_CALIBRATION / PEPTIDE_PARAMS tables
    io               CounterResult + ParquetDir round-trip

Time base: milliseconds throughout (rt + injection time), so the flux is
[ions/ms] and `∫N(t) d(rt) = N_total` [ions] with no stray unit factors;
seconds-based RT is scaled to ms at the observation boundary.
"""

from __future__ import annotations

from .calibration import AlphaModel, GlobalCalibration
from .iit import accumulated_count
from .io import CounterResult, load_counter, save_counter
from .discover import DiscoverConfig, discover_candidates, estimate_panel, fit_panel
from .model import CounterObservation, Progenitor
from .observe import (
    RegionPeaks,
    observation_for_progenitor,
    observation_for_region,
    observation_from_trace,
)
from .orchestrate import (
    StagedCalibration,
    StagedCalibrationResult,
    calibration_from_table,
    calibration_to_table,
    counter_n_table,
    estimate_n,
    peptide_params_to_table,
    seed_peak_from_observation,
)
from .panel import Panel, PanelCellTerms, panel_cell_log_prob, panel_log_prob
from .priors import make_log_prior
from .residual import panel_predicted_intensity, panel_residual
from .candidates import (
    CandidateEntry,
    TheoreticalCandidateIndex,
    channel_overlap_components,
)
from .component import estimate_component
from .emit import panel_attribution_table
from .schemas import (
    COUNTER_GLOBAL_CALIBRATION_TABLE,
    COUNTER_N_TABLE,
    COUNTER_PEAK_ATTRIBUTION_TABLE,
    COUNTER_PEPTIDE_PARAMS_TABLE,
)
from .simulate import (
    observation_to_trace,
    simulate_observation,
    simulate_panel_observation,
)

__all__ = [
    # model
    "Progenitor",
    "Panel",
    "CounterObservation",
    "GlobalCalibration",
    "AlphaModel",
    "panel_log_prob",
    "panel_cell_log_prob",
    "PanelCellTerms",
    "accumulated_count",
    "make_log_prior",
    # simulate + estimate
    "simulate_observation",
    "simulate_panel_observation",
    "observation_to_trace",
    "observation_from_trace",
    "observation_for_progenitor",
    "estimate_n",
    "seed_peak_from_observation",
    # panel-shaped estimate + discovery seam
    "observation_for_region",
    "RegionPeaks",
    "panel_residual",
    "panel_predicted_intensity",
    "fit_panel",
    "estimate_panel",
    "discover_candidates",
    "DiscoverConfig",
    "StagedCalibration",
    "StagedCalibrationResult",
    "counter_n_table",
    "calibration_to_table",
    "calibration_from_table",
    "peptide_params_to_table",
    "panel_attribution_table",
    "TheoreticalCandidateIndex",
    "CandidateEntry",
    "channel_overlap_components",
    "estimate_component",
    # io + schemas
    "CounterResult",
    "save_counter",
    "load_counter",
    "COUNTER_N_TABLE",
    "COUNTER_GLOBAL_CALIBRATION_TABLE",
    "COUNTER_PEPTIDE_PARAMS_TABLE",
    "COUNTER_PEAK_ATTRIBUTION_TABLE",
]
