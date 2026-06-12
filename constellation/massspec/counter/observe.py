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

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from .model import CounterObservation, Progenitor

__all__ = ["observation_from_trace", "observation_for_progenitor"]


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
    intensity_unit: str = "rate",
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

    `intensity_unit` declares the trace intensity convention. The model scores
    intensity as a per-time **rate** `α·flux` (its simulator emits
    `α·counts/iit`, so `observation_to_trace` produces `"rate"` — the default,
    round-trip-safe). A vendor centroid intensity is typically **per-scan
    accumulated** (∝ ions integrated over the injection time); pass
    `"per_scan"` to divide each cell by its scan's `iit`, recovering the rate
    `n_obs = I·iit/α` the likelihood expects (otherwise `N` inflates ~`iit×`)."""
    if scan_metadata.num_rows == 0:
        raise ValueError("scan_metadata is empty — cannot build a scan axis")
    rt_scale = {"s": 1000.0, "ms": 1.0}
    iit_scale = {"s": 1000.0, "ms": 1.0}
    if rt_unit not in rt_scale:
        raise ValueError(f"rt_unit must be 's' or 'ms'; got {rt_unit!r}")
    if iit_unit not in iit_scale:
        raise ValueError(f"iit_unit must be 's' or 'ms'; got {iit_unit!r}")
    if intensity_unit not in ("rate", "per_scan"):
        raise ValueError(
            f"intensity_unit must be 'rate' or 'per_scan'; got {intensity_unit!r}"
        )

    # -- scan axis (sorted by scan; the full extraction window) -------------
    sm_scan = scan_metadata.column("scan").to_numpy(zero_copy_only=False)
    order = np.argsort(sm_scan, kind="stable")
    sm_scan = sm_scan[order].astype(np.int64)
    rt = scan_metadata.column("rt").to_numpy(zero_copy_only=False)[order]
    iit = scan_metadata.column("iit").to_numpy(zero_copy_only=False)[order]
    rt = torch.as_tensor(rt.astype("float64"), dtype=dtype) * rt_scale[rt_unit]
    iit = torch.as_tensor(iit.astype("float64"), dtype=dtype) * iit_scale[iit_unit]
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

    # Convert per-scan accumulated intensity → the per-time rate the model scores.
    # (Non-detection cells are 0, so 0/iit stays 0; the mask is unchanged.)
    if intensity_unit == "per_scan":
        intensity = intensity / iit.clamp(min=1e-9)[:, None]

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
