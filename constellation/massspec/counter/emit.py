"""Emit the sparse ion→progenitor attribution map from a fitted panel.

The soft responsibilities `γ_q(s,c)` that `panel_log_prob` computes (and then
discards) ARE the per-peak ownership signal: `panel_attribution_table`
re-derives them from a fitted `Panel` and emits one row per
(observed peak, claiming progenitor) into `COUNTER_PEAK_ATTRIBUTION_TABLE`. A
peak interfered by several species gets one row per owner; "what's left" after
targets are searched is the anti-join of the full peak set against this map's
`(scan, mz_observed)` keys. No-grad diagnostics — this feeds bookkeeping, not
the optimizer.

Counter scores one (small) per-target panel at a time, so the per-cell row
assembly here is bounded and cheap; this is not a cross-dataset join (the
"no python row-loops over a partitioned dataset" rule is about those).
"""

from __future__ import annotations

from typing import Sequence

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
    progenitor_target_ids: Sequence[int | None] | None = None,
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
    share `1 − Σ_q γ_q`) so its peak is never lost. Hence every observed cell yields
    ≥ 1 row.

    Each row's `(scan, mz_observed)` is the attributed peak's stable physical-peak
    identity (`-1` / `NaN` when the observation carried no raw-peak source, e.g.
    simulated) — the anti-join / cross-panel key.

    **Per-progenitor identity.** `progenitor_target_ids[q]` is progenitor `q`'s real
    `target_id` (`None` for an anonymous discovered interferer). When given, each
    row carries its progenitor's own `target_id` with `is_target = (id is not None)`
    — so a COMPONENT co-fit's known members each keep their identity instead of all
    inheriting the reference's. When omitted (the singleton path), every row carries
    the single `target_id` with `is_target = (q == target_index)`; discovered
    interferers (`q ≥ 1`) and residuals (`q = -1`) ride under that target's id.

    `weight_floor` drops negligible owners. Returns an empty (correctly-typed) table
    when there are no observed cells."""
    terms = panel.cell_terms(obs)
    gamma = terms.gamma  # (Q, S, C)
    observed = terms.observed  # (S, C) bool
    n_obs_count = obs.recovered_count(panel.calibration)  # (S, C)
    cz = obs.channel_z.to(torch.long)
    ck = obs.channel_isotope.to(torch.long)
    has_scan = obs.scan is not None
    has_mz = obs.source_mz is not None
    n_prog = int(gamma.shape[0])
    ptids = list(progenitor_target_ids) if progenitor_target_ids is not None else None
    if ptids is not None and len(ptids) != n_prog:
        raise ValueError(
            f"progenitor_target_ids has {len(ptids)} entries != {n_prog} progenitors"
        )

    progenitor_index: list[int] = []
    row_target_id: list[int] = []
    is_target: list[bool] = []
    scan_col: list[int] = []
    mz_observed: list[float] = []
    channel_z: list[int] = []
    channel_isotope: list[int] = []
    responsibility: list[float] = []
    n_obs: list[float] = []

    def _identity(q: int) -> tuple[int, bool]:
        """`(target_id, is_target)` for progenitor `q` (`q < 0` = residual)."""
        if q >= 0 and ptids is not None and ptids[q] is not None:
            return int(ptids[q]), True  # a known component member — its own id
        if q >= 0 and ptids is None:
            return int(target_id), q == target_index  # singleton: target vs interferer
        return int(target_id), False  # anonymous interferer / residual

    def _emit(sel: torch.Tensor, q: int, resp: torch.Tensor) -> None:
        idx = sel.nonzero(as_tuple=False)  # (M, 2) → [scan_idx, channel_idx]
        if idx.numel() == 0:
            return
        s_i, c_i = idx[:, 0], idx[:, 1]
        m = int(idx.shape[0])
        tid, tgt = _identity(q)
        progenitor_index.extend([q] * m)
        row_target_id.extend([tid] * m)
        is_target.extend([tgt] * m)
        scan_col.extend(obs.scan[s_i].tolist() if has_scan else [-1] * m)
        mz_observed.extend(obs.source_mz[s_i, c_i].tolist() if has_mz else [float("nan")] * m)
        channel_z.extend(cz[c_i].tolist())
        channel_isotope.extend(ck[c_i].tolist())
        responsibility.extend(resp[s_i, c_i].tolist())
        n_obs.extend(n_obs_count[s_i, c_i].tolist())

    owned_any = torch.zeros_like(observed)  # (S, C) — cell owned by ≥ 1 progenitor
    for q in range(n_prog):
        sel = observed & (gamma[q] >= weight_floor)  # (S, C) bool
        owned_any = owned_any | sel
        _emit(sel, q, gamma[q])

    # Residual: observed cells no progenitor owns — emit once as `progenitor_index
    # = -1` with the unmodeled share, so the "what's left" anti-join never loses a
    # raw peak the model failed to explain.
    residual = (1.0 - gamma.sum(dim=0)).clamp(0.0, 1.0)  # (S, C)
    _emit(observed & ~owned_any, -1, residual)

    n = len(progenitor_index)
    arrays = [
        pa.array([int(acquisition_id)] * n, pa.int64()),
        pa.array(row_target_id, pa.int64()),
        pa.array(progenitor_index, pa.int32()),
        pa.array(is_target, pa.bool_()),
        pa.array(scan_col, pa.int64()),
        pa.array(mz_observed, pa.float64()),
        pa.array(channel_z, pa.int8()),
        pa.array(channel_isotope, pa.int8()),
        pa.array(responsibility, pa.float64()),
        pa.array(n_obs, pa.float64()),
        pa.array([int(iteration)] * n, pa.int32()),
    ]
    return pa.Table.from_arrays(arrays, schema=COUNTER_PEAK_ATTRIBUTION_TABLE)
