"""Emit the sparse ion→progenitor attribution map from a fitted panel.

The soft responsibilities `γ_q(s,c)` that `panel_log_prob` computes (and then
discards) ARE the per-peak ownership signal: `panel_attribution_table`
re-derives them from a fitted `Panel` and emits one row per
(observed peak, claiming progenitor) into `COUNTER_PEAK_ATTRIBUTION_TABLE`. A
peak interfered by several species gets one row per owner; "what's left" after
targets are searched is the anti-join of the full peak set against this map's
`peak_id`s. No-grad diagnostics — this feeds bookkeeping, not the optimizer.

Counter scores one (small) per-target panel at a time, so the per-cell row
assembly here is bounded and cheap; this is not a cross-dataset join (the
"no python row-loops over a partitioned dataset" rule is about those).
"""

from __future__ import annotations

import pyarrow as pa
import torch

from .model import CounterObservation
from .panel import Panel
from .schemas import COUNTER_PEAK_ATTRIBUTION_TABLE

__all__ = ["panel_attribution_table"]


@torch.no_grad()
def panel_attribution_table(
    panel: Panel,
    obs: CounterObservation,
    *,
    acquisition_id: int,
    target_id: int,
    target_index: int = 0,
    iteration: int = 0,
    weight_floor: float = 1e-3,
) -> pa.Table:
    """Sparse (peak → progenitor) soft-attribution rows for a fitted `panel`.

    One row per observed `(scan, channel)` cell × progenitor whose responsibility
    `γ_q ≥ weight_floor` there. An observed cell that NO progenitor owns at
    `≥ weight_floor` (e.g. a peak the model assigns ~0 predicted flux — the
    high-value "what's left" / interferer signal) is emitted as a **residual row**
    (`progenitor_index = -1`, `is_target = False`, `responsibility =` the unmodeled
    share `1 − Σ_q γ_q`) so its `peak_id` is never lost. Hence every observed cell
    yields ≥ 1 row. `peak_id` is the cell's source XIC_TRACE row (-1 when the
    observation carried no raw-peak identity, e.g. simulated). `weight_floor` drops
    negligible owners. Returns an empty (correctly-typed) table when there are no
    observed cells."""
    terms = panel.cell_terms(obs)
    gamma = terms.gamma  # (Q, S, C)
    observed = terms.observed  # (S, C) bool
    n_obs_count = obs.recovered_count(panel.calibration)  # (S, C)
    cz = obs.channel_z.to(torch.long)
    ck = obs.channel_isotope.to(torch.long)
    has_pid = obs.peak_id is not None

    progenitor_index: list[int] = []
    is_target: list[bool] = []
    peak_id: list[int] = []
    channel_z: list[int] = []
    channel_isotope: list[int] = []
    responsibility: list[float] = []
    n_obs: list[float] = []

    def _emit(sel: torch.Tensor, q: int, resp: torch.Tensor) -> None:
        idx = sel.nonzero(as_tuple=False)  # (M, 2) → [scan_idx, channel_idx]
        if idx.numel() == 0:
            return
        s_i, c_i = idx[:, 0], idx[:, 1]
        m = int(idx.shape[0])
        progenitor_index.extend([q] * m)
        is_target.extend([q == target_index] * m)
        peak_id.extend(obs.peak_id[s_i, c_i].tolist() if has_pid else [-1] * m)
        channel_z.extend(cz[c_i].tolist())
        channel_isotope.extend(ck[c_i].tolist())
        responsibility.extend(resp[s_i, c_i].tolist())
        n_obs.extend(n_obs_count[s_i, c_i].tolist())

    owned_any = torch.zeros_like(observed)  # (S, C) — cell owned by ≥ 1 progenitor
    for q in range(int(gamma.shape[0])):
        sel = observed & (gamma[q] >= weight_floor)  # (S, C) bool
        owned_any = owned_any | sel
        _emit(sel, q, gamma[q])

    # Residual: observed cells no progenitor owns — emit once as `progenitor_index
    # = -1` with the unmodeled share, so the "what's left" anti-join never loses a
    # raw peak the model failed to explain.
    residual = (1.0 - gamma.sum(dim=0)).clamp(0.0, 1.0)  # (S, C)
    _emit(observed & ~owned_any, -1, residual)

    n = len(peak_id)
    arrays = [
        pa.array([int(acquisition_id)] * n, pa.int64()),
        pa.array([int(target_id)] * n, pa.int64()),
        pa.array(progenitor_index, pa.int32()),
        pa.array(is_target, pa.bool_()),
        pa.array(peak_id, pa.int64()),
        pa.array(channel_z, pa.int8()),
        pa.array(channel_isotope, pa.int8()),
        pa.array(responsibility, pa.float64()),
        pa.array(n_obs, pa.float64()),
        pa.array([int(iteration)] * n, pa.int32()),
    ]
    return pa.Table.from_arrays(arrays, schema=COUNTER_PEAK_ATTRIBUTION_TABLE)
