"""Forward simulation — generate `CounterObservation` fixtures and the
canonical `XIC_TRACE_TABLE` from the model run forward.

The CI/test backbone (no instrument or DLL dependency) and the prior-
calibration fixture generator. `simulate_observation` draws a count-based
observation from a progenitor (Poisson counts → intensities, Student-t m/z;
see `Progenitor.sample`). `observation_to_trace` serializes an observation
to the canonical long-form `XIC_TRACE_TABLE` (+ a synthetic
`SCAN_METADATA_TABLE` carrying `iit`), so the simulator exercises the same
output schema the real extractor produces. (The reverse pivot
`XIC_TRACE → CounterObservation` — the real-data input path — lands with
the data-wiring PR.)
"""

from __future__ import annotations

from typing import Sequence

import pyarrow as pa
import torch

from .model import CounterObservation, Progenitor

__all__ = [
    "simulate_observation",
    "simulate_panel_observation",
    "observation_to_trace",
]


def simulate_observation(
    progenitor: Progenitor,
    *,
    n_scans: int = 60,
    half_window_ms: float = 30_000.0,
    iit_ms: float = 20.0,
    floor_ions: float = 1.0,
    generator: torch.Generator | None = None,
) -> CounterObservation:
    """Sample an observation across a scan grid centred on the progenitor's
    peak apex (`peak.mu`, in ms). `half_window_ms` is the half-width of the RT
    window; `iit_ms` the (constant) injection time. Returns a
    `CounterObservation` on the progenitor's channel grid."""
    mu = float(progenitor.peak.mu.detach())
    rt = torch.linspace(
        mu - half_window_ms, mu + half_window_ms, n_scans, dtype=torch.float64
    )
    iit = torch.full((n_scans,), float(iit_ms), dtype=torch.float64)
    return progenitor.sample(rt, iit, floor_ions=floor_ions, generator=generator)


@torch.no_grad()
def simulate_panel_observation(
    progenitors: Sequence[Progenitor],
    *,
    rt: torch.Tensor,
    iit_ms: float = 20.0,
    floor_ions: float = 1.0,
    generator: torch.Generator | None = None,
) -> CounterObservation:
    """Additive multi-progenitor observation on a shared channel grid.

    Per-progenitor flux contributions add (Poisson thinning → the combined
    per-channel count is `Poisson(Σ_q λ_q)`); intensity is `α(z)·count/τ`. All
    progenitors must share the channel grid (same charges × isotopes × m/z).
    Used to exercise additive interference + soft attribution."""
    ref = progenitors[0]
    for q in progenitors:
        if not torch.equal(q.channel_z, ref.channel_z) or not torch.allclose(
            q.channel_mz, ref.channel_mz
        ):
            raise ValueError("progenitors must share the same channel grid")
    rt = rt.to(torch.float64)
    iit = torch.full((rt.shape[0],), float(iit_ms), dtype=torch.float64)
    gain_ch = ref.calibration.gain(ref.channel_z.to(torch.float64))

    lam_total = torch.zeros((rt.shape[0], ref.channel_z.numel()), dtype=torch.float64)
    for q in progenitors:
        n_flux = q.peak.forward(rt)
        f_ch = q.charge_fractions().index_select(0, q.charge_index_of_channel)
        p_ch = q.p_k.index_select(0, q.channel_isotope)
        lam_total = lam_total + n_flux[:, None] * (f_ch * p_ch)[None, :] * iit[:, None]

    counts = torch.poisson(lam_total.clamp(min=0.0), generator=generator)
    intensity = gain_ch[None, :] * counts / iit[:, None]
    mask = counts >= floor_ions
    center = ref.calibration.mz_center_ppm(
        ref.channel_mz, ref.channel_z.to(torch.float64), ref.channel_isotope
    )
    c_mz_ch = ref.c_mz_per_channel()
    scale = (
        c_mz_ch[None, :] / counts.clamp(min=1.0) ** ref.calibration.alpha_mz
    ).clamp(min=1e-12).sqrt()
    noise = torch.distributions.StudentT(df=float(ref.calibration.nu_mz)).sample(
        counts.shape
    ).to(torch.float64)
    mz_error = center[None, :] + scale * noise
    zero = torch.zeros_like(intensity)
    return CounterObservation(
        rt=rt,
        iit=iit,
        intensity=torch.where(mask, intensity, zero),
        mz_error=torch.where(mask, mz_error, zero),
        mask=mask,
        channel_z=ref.channel_z,
        channel_isotope=ref.channel_isotope,
        channel_mz=ref.channel_mz,
    )


def observation_to_trace(
    obs: CounterObservation,
    *,
    acquisition_id: int,
    target_id: int,
    modified_sequence: str | None = None,
    scan_offset: int = 0,
) -> tuple[pa.Table, pa.Table]:
    """Serialize an observation to `(XIC_TRACE_TABLE, SCAN_METADATA-like)`.

    One trace row per observed `(scan, channel)` (MS1; `level=1`). RT is
    converted ms → seconds for the canonical tables. The companion scan table
    carries `scan`, `rt` [s], and `iit` [ms]. This is a terminal/per-target
    artifact, so the modest Python assembly is acceptable (cf. the
    Arrow-native rule for dataset-scale inter-stage handoffs)."""
    from constellation.massspec.quant.schemas import XIC_TRACE_TABLE

    s_idx, c_idx = torch.nonzero(obs.mask, as_tuple=True)
    n = int(s_idx.numel())
    rt_s = (obs.rt[s_idx] / 1000.0).tolist()
    z = obs.channel_z[c_idx].to(torch.int8).tolist()
    iso = obs.channel_isotope[c_idx].to(torch.int8).tolist()
    mz_theo = obs.channel_mz[c_idx]
    mz_err_ppm = obs.mz_error[s_idx, c_idx]
    mz_obs = (mz_theo * (1.0 + mz_err_ppm / 1e6)).tolist()
    inten = obs.intensity[s_idx, c_idx].tolist()
    mz_err_da = (mz_theo * mz_err_ppm / 1e6).tolist()
    scans = (s_idx + scan_offset).to(torch.int32).tolist()

    trace = pa.table(
        {
            "acquisition_id": pa.array([acquisition_id] * n, pa.int64()),
            "target_id": pa.array([target_id] * n, pa.int64()),
            "modified_sequence": pa.array([modified_sequence] * n, pa.string()),
            "precursor_charge": pa.array(z, pa.int8()),
            "level": pa.array([1] * n, pa.int8()),
            "scan": pa.array(scans, pa.int32()),
            "rt": pa.array(rt_s, pa.float64()),
            "isolation_lower": pa.array([None] * n, pa.float64()),
            "isolation_upper": pa.array([None] * n, pa.float64()),
            "isotope": pa.array(iso, pa.int8()),
            "ion_type": pa.array([None] * n, pa.int8()),
            "position": pa.array([None] * n, pa.int32()),
            "fragment_charge": pa.array([None] * n, pa.int32()),
            "loss_id": pa.array([None] * n, pa.string()),
            "mz_theoretical": pa.array(mz_theo.tolist(), pa.float64()),
            "mz_observed": pa.array(mz_obs, pa.float64()),
            "intensity": pa.array(inten, pa.float64()),
            "mz_error_da": pa.array(mz_err_da, pa.float64()),
            "mz_error_ppm": pa.array(mz_err_ppm.tolist(), pa.float64()),
        }
    ).cast(XIC_TRACE_TABLE)

    n_scans = obs.n_scans
    scan_meta = pa.table(
        {
            "scan": pa.array(
                (torch.arange(n_scans) + scan_offset).to(torch.int32).tolist(),
                pa.int32(),
            ),
            "rt": pa.array((obs.rt / 1000.0).tolist(), pa.float64()),
            "iit": pa.array(obs.iit.to(torch.float32).tolist(), pa.float32()),
        }
    )
    return trace, scan_meta
