"""Estimation driver — fit a progenitor's `N(t)` and emit the deliverable.

`estimate_n` is the per-target driver: it MAP-fits a `Progenitor` (DE global
search over the multimodal HyperEMG + LBFGS polish, calibration frozen since
it is not a progenitor parameter) and reports the integrated ion count
`N_total` with a Laplace credible interval on `log N_total`. The richer
variational path (`inference="vb"`, true posterior credible intervals) lands
in the next PR; PR-1 ships `inference="map"`.

Table builders serialize results / model state into the `schemas.py` tables.
"""

from __future__ import annotations

import fnmatch
import math
from typing import Any, Sequence

import pyarrow as pa
import torch

from constellation.core.optim import AdamOptimizer, DifferentialEvolution
from constellation.core.stats.intervals import laplace_cov

from .calibration import GlobalCalibration
from .model import CounterObservation, Progenitor
from .priors import make_log_prior
from .schemas import (
    COUNTER_GLOBAL_CALIBRATION_TABLE,
    COUNTER_N_TABLE,
    COUNTER_PEPTIDE_PARAMS_TABLE,
)

__all__ = [
    "estimate_n",
    "seed_peak_from_observation",
    "counter_n_table",
    "calibration_to_table",
    "peptide_params_to_table",
]


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@torch.no_grad()
def seed_peak_from_observation(
    progenitor: Progenitor,
    obs: CounterObservation,
    *,
    rt_prior_ms: float | None = None,
) -> None:
    """Initialize the progenitor's `HyperEMGPeak` from the observation.

    The total observed intensity per scan, divided by `Σ_c α(z)·f_z·p_k`, is
    an estimate of the latent flux `N(t)`; its RT-centroid seeds `μ`, its
    spread seeds `σ`/`τ`, and its trapezoidal integral seeds `N_total` (which
    is ≈ the answer — the fit then refines it under the full likelihood).
    `rt_prior_ms` (e.g. a PSM RT) overrides the centroid for `μ`. A good seed
    makes the gradient fit robust without a wide global search."""
    rt = obs.rt
    i_tot = (obs.intensity * obs.mask).sum(dim=1)  # (S,)
    w = i_tot.clamp(min=0.0)
    total_w = w.sum().clamp(min=1e-12)
    centroid = (w * rt).sum() / total_w
    var = (w * (rt - centroid) ** 2).sum() / total_w
    sigma = var.clamp(min=1e4).sqrt()  # ≥ 100 ms

    f_ch = progenitor.charge_fractions().index_select(
        0, progenitor.charge_index_of_channel
    )
    p_ch = progenitor.p_k.index_select(0, progenitor.channel_isotope)
    gain_ch = progenitor.calibration.gain(progenitor.channel_z.to(rt.dtype))
    denom = (gain_ch * f_ch * p_ch).sum().clamp(min=1e-12)
    n_flux = i_tot / denom  # (S,) ≈ N(t)
    n_total = torch.trapezoid(n_flux, rt).clamp(min=1.0)

    mu = torch.as_tensor(rt_prior_ms, dtype=rt.dtype) if rt_prior_ms is not None else centroid
    progenitor.peak.log_N_total.fill_(n_total.log())
    progenitor.peak.mu.fill_(mu)
    progenitor.peak.log_sigma.fill_(sigma.log())
    half = (0.5 * sigma).clamp(min=100.0)
    progenitor.peak.log_tau_r.fill_(half.log())
    progenitor.peak.log_tau_l.fill_(half.log())


def _log_n_total_std(progenitor: Progenitor, obs: CounterObservation) -> float:
    """Marginal std of `log N_total` from a Laplace approximation over the
    N + peak-shape block (captures the N↔shape correlation, so the interval
    isn't artificially tight). Falls back to the conditional (`log_N_total`-
    only) curvature if the block Hessian is singular / non-PD."""
    try:
        cov = laplace_cov(progenitor, obs, subset=["peak.*"])
        var = float(cov[0, 0])  # log_N_total is the first peak parameter
        if math.isfinite(var) and var > 0.0:
            return math.sqrt(var)
    except Exception:
        pass
    cov = laplace_cov(progenitor, obs, subset=["peak.log_N_total"])
    return float(cov.reshape(()).clamp(min=1e-12).sqrt())


def _default_bounds(
    progenitor: Progenitor, obs: CounterObservation
) -> dict[str, tuple[float, float]]:
    """DE bounds (registered-name space) for a progenitor's per-peptide params.
    Times are in ms (the Counter time base): σ/τ span 0.1 s … 120 s."""
    rt_lo, rt_hi = float(obs.rt.min()), float(obs.rt.max())
    return {
        "peak.log_N_total": (math.log(1.0), math.log(1e10)),
        "peak.mu": (rt_lo, rt_hi),
        "peak.log_sigma": (math.log(100.0), math.log(120_000.0)),
        "peak.log_tau_r": (math.log(100.0), math.log(120_000.0)),
        "peak.log_tau_l": (math.log(100.0), math.log(120_000.0)),
        "peak.logit_eta": (_logit(0.02), _logit(0.98)),
        "charge_energy": (-12.0, 12.0),
        "isotope_energy_offset": (-1.5, 1.5),  # ≲4.5× fraction deviation; toward-0 prior in VB
        "log_nu_intensity": (math.log(2.0), math.log(200.0)),
        "log_c_mz": (math.log(1.0), math.log(1e6)),
    }


def estimate_n(
    progenitor: Progenitor,
    obs: CounterObservation,
    *,
    inference: str = "vb",
    optimizer: str = "de",
    credible_level: float = 0.95,
    rt_prior_ms: float | None = None,
    rt_sigma_ms: float = 5000.0,
    guide_params: Sequence[str] = ("peak.*",),
    isotope_offset_sigma: float = 0.5,
    eta_sigma: float = 1.0,
    n_elbo_samples: int = 16,
    n_ci_samples: int = 4096,
    vb_lr: float = 0.02,
    vb_max_iter: int = 1500,
    reseed: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
    pop_size: int = 24,
    max_evals: int = 4000,
    max_iter: int = 100,
    lr: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit `progenitor` to `obs` and return the numeric result fields of
    `COUNTER_N_TABLE` (the caller adds `acquisition_id` / `target_id` /
    identity).

    Both paths first MAP-fit the progenitor (data-seeded via
    `seed_peak_from_observation`; `optimizer` ``"de"`` (default — DE warm-start
    within bounds + LBFGS polish, robust to the multimodal HyperEMG η/τ) or
    ``"adam"``). Then:

    - **``inference="vb"``** (default): a `MeanFieldGuide` over `guide_params`
      (default ``("peak.*",)`` = `N_total` + peak shape) is warm-started at the
      MAP point and refined by `fit_vb`; the credible interval on `N_total`
      comes from the posterior samples (decoded to natural units). The RT seed
      enters as a Gaussian *prior* on `peak.mu` (not a hard window) when
      `rt_prior_ms` is given; the toward-theoretical isotope prior activates
      only when the offset is in `guide_params`. This is the calibrated-coverage
      path (the MAP point estimate carries a mild upward bias at very low ion
      counts from detection-floor censoring).
    - **``inference="map"``**: the MAP point estimate with a Laplace credible
      interval over the N + peak-shape block.

    The progenitor is left at the fitted point (VB writes the posterior mean
    back via `guide.to_model`). Bounded-LBFGS is intentionally not a driver: its
    after-step clamp breaks the line search on this surface."""
    if inference not in ("map", "vb"):
        raise ValueError(f"inference must be 'map' or 'vb'; got {inference!r}")
    if reseed:
        seed_peak_from_observation(progenitor, obs, rt_prior_ms=rt_prior_ms)
    if bounds is None:
        bounds = _default_bounds(progenitor, obs)

    # MAP fit (point estimate / VB warm-start).
    if optimizer == "de":
        de = DifferentialEvolution(
            progenitor, bounds=bounds, pop_size=pop_size, max_evals=max_evals, seed=seed
        )
        result = progenitor.fit(
            obs, optimizer=de, max_iter=max_iter, polish_on_converge=True
        )
    elif optimizer == "adam":
        adam = AdamOptimizer(progenitor, lr=lr)
        result = progenitor.fit(obs, optimizer=adam, max_iter=2000, tol=1e-8)
    else:
        raise ValueError(f"optimizer must be 'de' or 'adam'; got {optimizer!r}")

    if inference == "map":
        std = _log_n_total_std(progenitor, obs)
        log_n = float(progenitor.peak.log_N_total.detach())
        z = float(torch.special.ndtri(torch.tensor((1.0 + credible_level) / 2.0)))
        n_total = math.exp(log_n)
        n_lo, n_hi = math.exp(log_n - z * std), math.exp(log_n + z * std)
        converged, final_elbo = bool(result.converged), None
    else:  # vb
        n_total, n_lo, n_hi, converged, final_elbo = _fit_vb(
            progenitor,
            obs,
            guide_params=guide_params,
            credible_level=credible_level,
            rt_prior_ms=rt_prior_ms,
            rt_sigma_ms=rt_sigma_ms,
            isotope_offset_sigma=isotope_offset_sigma,
            eta_sigma=eta_sigma,
            n_elbo_samples=n_elbo_samples,
            n_ci_samples=n_ci_samples,
            vb_lr=vb_lr,
            vb_max_iter=vb_max_iter,
        )

    return {
        "n_total": n_total,
        "n_total_lo": n_lo,
        "n_total_hi": n_hi,
        "credible_level": float(credible_level),
        "rt_apex": float(progenitor.peak.mu.detach()) / 1000.0,  # ms → s
        "peak_sigma": float(progenitor.peak.sigma.detach()) / 1000.0,
        "peak_tau_r": float(progenitor.peak.tau_r.detach()) / 1000.0,
        "peak_tau_l": float(progenitor.peak.tau_l.detach()) / 1000.0,
        "peak_eta": float(progenitor.peak.eta.detach()),
        "inference_method": inference,
        "converged": converged,
        "final_elbo": final_elbo,
        "n_scans_used": int(obs.mask.any(dim=1).sum()),
        "iit_corrected": True,
        "interference_flag": None,
    }


def _fit_vb(
    progenitor: Progenitor,
    obs: CounterObservation,
    *,
    guide_params: Sequence[str],
    credible_level: float,
    rt_prior_ms: float | None,
    rt_sigma_ms: float,
    isotope_offset_sigma: float,
    eta_sigma: float,
    n_elbo_samples: int,
    n_ci_samples: int,
    vb_lr: float,
    vb_max_iter: int,
) -> tuple[float, float, float, bool, float | None]:
    """Variational refinement of a MAP-warm-started progenitor. Returns
    `(n_total, lo, hi, converged, final_elbo)` — the posterior-mean `N_total`
    and its credible interval from decoded posterior samples. Mutates
    `progenitor` to the posterior mean of the guided params."""
    from constellation.core.optim import AdamOptimizer as _Adam
    from constellation.core.stats import MeanFieldGuide, credible_interval, fit_vb

    guide = MeanFieldGuide(progenitor, subset=list(guide_params))
    in_guide = lambda name: any(  # noqa: E731
        fnmatch.fnmatchcase(name, pat) for pat in guide_params
    )
    # η prior toward its MAP value, but only when η is actually a guide param.
    eta_center = (
        float(progenitor.peak.logit_eta.detach())
        if in_guide("peak.logit_eta")
        else None
    )
    log_prior = make_log_prior(
        rt_prior_ms=rt_prior_ms,
        rt_sigma_ms=rt_sigma_ms,
        isotope_offset_sigma=(
            isotope_offset_sigma if in_guide("isotope_energy_offset") else None
        ),
        eta_center=eta_center,
        eta_sigma=eta_sigma,
    )
    vb = fit_vb(
        progenitor,
        guide,
        obs,
        optimizer=_Adam(guide, lr=vb_lr),
        log_prior=log_prior,
        n_samples=n_elbo_samples,
        max_iter=vb_max_iter,
    )
    n_samples = guide.named_samples(progenitor, n_ci_samples)["peak.log_N_total"].exp()
    lo, hi = credible_interval(n_samples, level=credible_level)
    guide.to_model(progenitor)  # posterior mean → model (for shape reporting)
    final_elbo = -vb.final_loss if math.isfinite(vb.final_loss) else None
    return (
        float(n_samples.mean()),
        float(lo),
        float(hi),
        bool(vb.converged),
        final_elbo,
    )


# ──────────────────────────────────────────────────────────────────────
# Table builders
# ──────────────────────────────────────────────────────────────────────


def counter_n_table(records: Sequence[dict[str, Any]]) -> pa.Table:
    """Assemble `COUNTER_N_TABLE` from full result records (each must carry the
    id/identity fields plus the numeric fields from `estimate_n`)."""
    cols = {name: [r.get(name) for r in records] for name in COUNTER_N_TABLE.names}
    return pa.table(cols).cast(COUNTER_N_TABLE)


def calibration_to_table(
    cal: GlobalCalibration, *, acquisition_id: int | None = None
) -> pa.Table:
    """Serialize a `GlobalCalibration` to a one-row
    `COUNTER_GLOBAL_CALIBRATION_TABLE`."""
    pd = cal.parameters_dict()
    row: dict[str, Any] = {
        "acquisition_id": acquisition_id,
        "analyzer": cal.analyzer,
        "alpha_model": cal.alpha_model,
        "alpha0": float(pd["alpha0"]) if "alpha0" in pd else None,
        "alpha1": float(pd["alpha1"]) if "alpha1" in pd else None,
        "alpha_z": pd["alpha_z"].tolist() if "alpha_z" in pd else None,
        "mz_offset_ppm": float(pd["mz_offset_ppm"]),
        "d_mz_da": pd["d_mz_da"].tolist(),
        "mz_precision_exponent": float(pd["alpha_mz"]),
        "nu_mz": float(pd["nu_mz"]),
        "rho_resolution": float(pd["rho"]),
        "r_ref": cal.r_ref,
        "mz_ref": cal.mz_ref,
        "prior_log_sigma_mean": None,
        "prior_log_sigma_std": None,
        "prior_log_tau_mean": None,
        "prior_log_tau_std": None,
    }
    return pa.table({k: [v] for k, v in row.items()}).cast(
        COUNTER_GLOBAL_CALIBRATION_TABLE
    )


def peptide_params_to_table(
    progenitors: Sequence[Progenitor],
    *,
    acquisition_id: int,
    target_ids: Sequence[int],
    modified_sequences: Sequence[str | None] | None = None,
    peptide_ids: Sequence[int | None] | None = None,
) -> pa.Table:
    """Serialize per-peptide tier params to `COUNTER_PEPTIDE_PARAMS_TABLE`."""
    n = len(progenitors)
    mods = modified_sequences or [None] * n
    pids = peptide_ids or [None] * n
    rows: dict[str, list[Any]] = {name: [] for name in COUNTER_PEPTIDE_PARAMS_TABLE.names}
    for q, tid, mod, pid in zip(progenitors, target_ids, mods, pids):
        rows["acquisition_id"].append(acquisition_id)
        rows["target_id"].append(tid)
        rows["peptide_id"].append(pid)
        rows["modified_sequence"].append(mod)
        rows["charge_free_energy"].append(q.charge_energy.detach().tolist())
        rows["charges"].append(q.charges.to(torch.int32).tolist())
        rows["nu_intensity"].append(float(q.nu_intensity.detach()))
        rows["c_mz"].append(q.log_c_mz.exp().detach().tolist())
        rows["isotope_fractions"].append(q.p_k.detach().tolist())
    return pa.table(rows).cast(COUNTER_PEPTIDE_PARAMS_TABLE)
