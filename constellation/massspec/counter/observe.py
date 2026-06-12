"""Real-data input pivot — build a `CounterObservation` from an extracted XIC.

`observation_from_trace` is the inverse of `simulate.observation_to_trace`: it
packs the long-form `XIC_TRACE_TABLE` (one row per observed `(scan, channel)`)
+ a `SCAN_METADATA`-style table (per-scan `rt`, `iit`) into the dense
`(scan × channel)` `CounterObservation` the model scores. This is the wiring
that turns Counter from simulator-validated into runnable on real
`.raw` → `massspec.quant.chromatogram` extractions.

Two design points:
  * **The scan axis comes from `scan_metadata`, not the trace.** The trace only
    carries *observed* `(scan, channel)` rows; the extraction window's
    non-detection scans live in `scan_metadata`. Using it as the scan axis is
    what feeds the censored (non-detection) term — drop it and `N` is biased.
  * **The channel grid comes from the caller** (a target `Progenitor`'s
    theoretical envelope), so channels never observed in any scan are still
    present (again, for the censored term). Trace rows are matched onto the grid
    by `(precursor_charge, isotope)`.

Units follow the canonical tables: trace/scan `rt` in **seconds** → ms (the
model's time base); `iit` in **ms**. Both overridable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from .model import CounterObservation, Progenitor

__all__ = [
    "observation_from_trace",
    "observation_for_progenitor",
    "observation_for_region",
    "RegionPeaks",
]


_RT_SCALE = {"s": 1000.0, "ms": 1.0}
_IIT_SCALE = {"s": 1000.0, "ms": 1.0}


def _scan_axis(
    scan_metadata: pa.Table, rt_unit: str, iit_unit: str, dtype: torch.dtype
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Sorted scan axis `(sm_scan, rt_ms, iit_ms)` from `scan_metadata`
    (columns `scan`, `rt`, `iit`) — the full extraction window, non-detection
    scans included. Shared by `observation_from_trace` + `observation_for_region`."""
    if scan_metadata.num_rows == 0:
        raise ValueError("scan_metadata is empty — cannot build a scan axis")
    if rt_unit not in _RT_SCALE:
        raise ValueError(f"rt_unit must be 's' or 'ms'; got {rt_unit!r}")
    if iit_unit not in _IIT_SCALE:
        raise ValueError(f"iit_unit must be 's' or 'ms'; got {iit_unit!r}")
    sm_scan = scan_metadata.column("scan").to_numpy(zero_copy_only=False)
    order = np.argsort(sm_scan, kind="stable")
    sm_scan = sm_scan[order].astype(np.int64)
    rt = scan_metadata.column("rt").to_numpy(zero_copy_only=False)[order]
    iit = scan_metadata.column("iit").to_numpy(zero_copy_only=False)[order]
    rt = torch.as_tensor(rt.astype("float64"), dtype=dtype) * _RT_SCALE[rt_unit]
    iit = torch.as_tensor(iit.astype("float64"), dtype=dtype) * _IIT_SCALE[iit_unit]
    return sm_scan, rt, iit


def observation_from_trace(
    trace: pa.Table,
    scan_metadata: pa.Table,
    *,
    channel_z: torch.Tensor,
    channel_isotope: torch.Tensor,
    channel_mz: torch.Tensor,
    target_id: int | None = None,
    level: int | None = 1,
    rt_unit: str = "s",
    iit_unit: str = "ms",
    intensity_floor: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> CounterObservation:
    """Pack an `XIC_TRACE_TABLE` + per-scan `scan_metadata` into a
    `CounterObservation` on the `(channel_z, channel_isotope, channel_mz)` grid.

    `scan_metadata` (columns `scan`, `rt`, `iit`) defines the scan axis `S`
    (sorted by `scan`) — including non-detection scans. `trace` rows
    (`scan`, `precursor_charge`, `isotope`, `intensity`, `mz_error_ppm`) are
    matched onto the grid by `(charge, isotope)` and scattered into the dense
    `(S, C)` `intensity` / `mz_error` / `mask` tensors. `target_id` / `level`
    filter the trace first (MS1 `level=1` by default). When several peaks map to
    one `(scan, channel)` (e.g. `all_in_window` interference), the most intense
    is kept. Rows whose scan is absent from `scan_metadata`, whose
    `(charge, isotope)` is off-grid, or whose intensity is `≤ intensity_floor`
    are dropped.

    Intensity is taken in the model's per-time **rate** convention (`α·flux`,
    what the Thermo reader emits as AU/ms and what `observation_to_trace`
    produces) — no per-injection-time rescaling is applied here."""
    sm_scan, rt, iit = _scan_axis(scan_metadata, rt_unit, iit_unit, dtype)
    n_scans = int(sm_scan.shape[0])

    cz = channel_z.to(torch.long)
    ck = channel_isotope.to(torch.long)
    n_chan = int(cz.numel())
    intensity = torch.zeros((n_scans, n_chan), dtype=dtype)
    mz_error = torch.zeros((n_scans, n_chan), dtype=dtype)
    mask = torch.zeros((n_scans, n_chan), dtype=torch.bool)

    # -- filter + scatter the trace -----------------------------------------
    t = trace
    if target_id is not None and "target_id" in t.column_names:
        t = t.filter(pc.equal(t.column("target_id"), target_id))
    if level is not None and "level" in t.column_names:
        t = t.filter(pc.equal(t.column("level"), level))

    if t.num_rows > 0:
        scan = t.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
        chg = t.column("precursor_charge").to_numpy(zero_copy_only=False).astype(np.int64)
        iso = t.column("isotope").to_numpy(zero_copy_only=False).astype(np.int64)
        inten = t.column("intensity").to_numpy(zero_copy_only=False).astype("float64")
        merr = t.column("mz_error_ppm").to_numpy(zero_copy_only=False).astype("float64")

        # scan → row index (only rows whose scan is in the window)
        s_idx = np.searchsorted(sm_scan, scan)
        in_range = (s_idx < n_scans) & (sm_scan[np.clip(s_idx, 0, n_scans - 1)] == scan)
        # (charge, isotope) → channel index (grid is small; build a code map)
        grid_code = (cz.numpy().astype(np.int64) << 8) + ck.numpy().astype(np.int64)
        code_to_c = {int(grid_code[c]): c for c in range(n_chan)}
        row_code = (chg << 8) + iso
        c_idx = np.array([code_to_c.get(int(rc), -1) for rc in row_code], dtype=np.int64)

        valid = in_range & (c_idx >= 0) & np.isfinite(inten) & (inten > intensity_floor)
        if valid.any():
            flat = s_idx[valid] * n_chan + c_idx[valid]
            iv, ev = inten[valid], merr[valid]
            # de-duplicate (scan, channel): keep the most intense peak. Sort by
            # (flat, intensity asc) so the last row of each flat group is the max.
            srt = np.lexsort((iv, flat))
            flat_s, iv_s, ev_s = flat[srt], iv[srt], ev[srt]
            last = np.ones(flat_s.shape[0], dtype=bool)
            last[:-1] = flat_s[1:] != flat_s[:-1]
            fk = torch.as_tensor(flat_s[last], dtype=torch.long)
            intensity.view(-1)[fk] = torch.as_tensor(iv_s[last], dtype=dtype)
            mz_error.view(-1)[fk] = torch.as_tensor(ev_s[last], dtype=dtype)
            mask.view(-1)[fk] = True

    return CounterObservation(
        rt=rt,
        iit=iit,
        intensity=intensity,
        mz_error=mz_error,
        mask=mask,
        channel_z=cz,
        channel_isotope=ck,
        channel_mz=channel_mz.to(dtype),
    )


def observation_for_progenitor(
    progenitor: Progenitor,
    trace: pa.Table,
    scan_metadata: pa.Table,
    **kwargs,
) -> CounterObservation:
    """`observation_from_trace` using a `Progenitor`'s channel grid — the
    common real-data path: build the target's `Progenitor` via
    `Progenitor.for_peptide(...)`, then pack its observation from the extracted
    trace + scan metadata before `estimate_n`."""
    return observation_from_trace(
        trace,
        scan_metadata,
        channel_z=progenitor.channel_z,
        channel_isotope=progenitor.channel_isotope,
        channel_mz=progenitor.channel_mz,
        **kwargs,
    )


@dataclass(frozen=True)
class RegionPeaks:
    """In-window raw peaks NOT placed into the target observation — the candidate
    ion nodes the fit-plus-discover loop (PR-5) can wire a new interferer
    progenitor to. Sparse (slivers near the target channels), packed flat over
    peaks. PR-4 retains them; discovery consumes them later."""

    scan_idx: torch.Tensor  # (P,) long — index into the observation's scan axis
    mz: torch.Tensor  # (P,) observed m/z [Th]
    intensity: torch.Tensor  # (P,)
    charge: torch.Tensor  # (P,) long — the extraction's charge hypothesis


def observation_for_region(
    trace: pa.Table,
    scan_metadata: pa.Table,
    target: Progenitor,
    *,
    target_id: int | None = None,
    neighborhood_ppm: float = 100.0,
    level: int | None = 1,
    rt_unit: str = "s",
    iit_unit: str = "ms",
    intensity_floor: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[CounterObservation, RegionPeaks]:
    """Build the `target`'s ion-node `CounterObservation` (via
    `observation_from_trace`) AND collect the discovery substrate: in-window raw
    peaks within `neighborhood_ppm` of a target channel that the observation did
    NOT place (a `(charge, isotope)` off the target grid, or a de-dup loser at a
    `(scan, channel)` — the chimeric / interferer evidence). Pass an
    `all_in_window=True` extraction so those peaks survive. Returns
    `(observation, RegionPeaks)`."""
    obs = observation_from_trace(
        trace,
        scan_metadata,
        channel_z=target.channel_z,
        channel_isotope=target.channel_isotope,
        channel_mz=target.channel_mz,
        target_id=target_id,
        level=level,
        rt_unit=rt_unit,
        iit_unit=iit_unit,
        intensity_floor=intensity_floor,
        dtype=dtype,
    )
    sm_scan, _rt, _iit = _scan_axis(scan_metadata, rt_unit, iit_unit, dtype)
    region = _collect_region_peaks(
        trace,
        sm_scan,
        target,
        target_id=target_id,
        level=level,
        neighborhood_ppm=neighborhood_ppm,
        intensity_floor=intensity_floor,
        dtype=dtype,
    )
    return obs, region


def _collect_region_peaks(
    trace: pa.Table,
    sm_scan: np.ndarray,
    target: Progenitor,
    *,
    target_id: int | None,
    level: int | None,
    neighborhood_ppm: float,
    intensity_floor: float,
    dtype: torch.dtype,
) -> RegionPeaks:
    """In-window peaks the observation dropped (off-grid `(charge, isotope)` or
    de-dup loser at a `(scan, channel)`), restricted to within `neighborhood_ppm`
    of a target channel m/z."""
    n_scans = int(sm_scan.shape[0])
    n_chan = int(target.channel_z.numel())
    empty = RegionPeaks(
        scan_idx=torch.zeros(0, dtype=torch.long),
        mz=torch.zeros(0, dtype=dtype),
        intensity=torch.zeros(0, dtype=dtype),
        charge=torch.zeros(0, dtype=torch.long),
    )
    t = trace
    if target_id is not None and "target_id" in t.column_names:
        t = t.filter(pc.equal(t.column("target_id"), target_id))
    if level is not None and "level" in t.column_names:
        t = t.filter(pc.equal(t.column("level"), level))
    if t.num_rows == 0:
        return empty

    scan = t.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
    chg = t.column("precursor_charge").to_numpy(zero_copy_only=False).astype(np.int64)
    iso = t.column("isotope").to_numpy(zero_copy_only=False).astype(np.int64)
    inten = t.column("intensity").to_numpy(zero_copy_only=False).astype("float64")
    mz_obs = t.column("mz_observed").to_numpy(zero_copy_only=False).astype("float64")
    mz_theo = t.column("mz_theoretical").to_numpy(zero_copy_only=False).astype("float64")
    mz = np.where(np.isfinite(mz_obs), mz_obs, mz_theo)

    s_idx = np.searchsorted(sm_scan, scan)
    in_range = (s_idx < n_scans) & (sm_scan[np.clip(s_idx, 0, n_scans - 1)] == scan)

    cz = target.channel_z.numpy().astype(np.int64)
    ck = target.channel_isotope.numpy().astype(np.int64)
    code_to_c = {int((cz[c] << 8) + ck[c]): c for c in range(n_chan)}
    c_idx = np.array([code_to_c.get(int((z << 8) + k), -1) for z, k in zip(chg, iso)], np.int64)

    tgt_mz = target.channel_mz.numpy().astype("float64")
    d_ppm = np.min(np.abs(mz[:, None] - tgt_mz[None, :]) / tgt_mz[None, :] * 1e6, axis=1)
    valid = (
        in_range
        & np.isfinite(inten)
        & (inten > intensity_floor)
        & np.isfinite(mz)
        & (d_ppm <= float(neighborhood_ppm))
    )

    # candidate = valid AND (off-grid OR a de-dup loser at its (scan, channel))
    candidate = valid & (c_idx < 0)
    on_grid = np.where(valid & (c_idx >= 0))[0]
    if on_grid.size:
        flat = s_idx[on_grid] * n_chan + c_idx[on_grid]
        order = np.lexsort((inten[on_grid], flat))  # intensity ascending within cell
        flat_s = flat[order]
        winner = np.ones(flat_s.shape[0], dtype=bool)
        winner[:-1] = flat_s[1:] != flat_s[:-1]  # last per group = max = the kept peak
        candidate[on_grid[order[~winner]]] = True  # the rest are losers

    if not candidate.any():
        return empty
    return RegionPeaks(
        scan_idx=torch.as_tensor(s_idx[candidate], dtype=torch.long),
        mz=torch.as_tensor(mz[candidate], dtype=dtype),
        intensity=torch.as_tensor(inten[candidate], dtype=dtype),
        charge=torch.as_tensor(chg[candidate], dtype=torch.long),
    )
