"""Fit-plus-discover loop + the progenitor-discovery seam (panel-shaped estimate).

The per-seed estimation unit is a `Panel` (one target progenitor + an additive
background), fit jointly. The loop fits the panel, inspects the residual at its
ion nodes, and — when discovery finds coherent unexplained signal — adds an
interferer progenitor and refits, so the interferer OWNS its signal and the
target's `N` is freed of it (the "ownership mapping" fix).

`discover_candidates` (PR-5) detects a coherent unexplained peak in the residual
and returns a pre-seeded **same-grid** interferer `Progenitor` — a clone of the
target's ion-node grid at a different RT (new edges only to the target's existing
nodes, no union grid). Off-m/z (union-grid) interferers + VB-on-panel credible
intervals are PR-6.

Inference: the target's `N_total` is reported with a **Laplace** credible
interval on its peak block (works for any panel membership). Variational
credible intervals over a multi-progenitor panel are a PR-6 generalization.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from constellation.core.optim import AdamOptimizer, DifferentialEvolution
from constellation.core.stats.intervals import laplace_cov

from .calibration import GlobalCalibration
from .model import CounterObservation, Progenitor
from .observe import RegionPeaks
from .orchestrate import _default_bounds, seed_peak_from_observation
from .panel import Panel
from .residual import panel_residual

__all__ = ["DiscoverConfig", "discover_candidates", "fit_panel", "estimate_panel"]

# Discovery-internal heuristics (kept out of the frozen DiscoverConfig so its
# signature stays a drop-in for the merged seam).
_RUN_FRAC = 0.1  # an RT run extends while r_scan > _RUN_FRAC × its apex
_CORE_FRAC = 0.5  # target "core" = where it predicts > _CORE_FRAC × its apex intensity
_CORE_N_SIGMA = 1.0  # ... AND within ±N·σ of the target apex (both ⇒ the target's own core)
# A candidate must clear an absolute ion floor (`threshold`) AND a fraction of the
# target's current N — the latter makes detection noise-relative (an 8-ion floor is
# far too sensitive against a bright target's Poisson residual). Faint interferers
# (≪ this fraction) barely bias N, so missing them is acceptable.
_MIN_FRAC_OF_TARGET = 0.10


@dataclass(frozen=True)
class DiscoverConfig:
    """Knobs for the fit-plus-discover loop."""

    detect_threshold: float = 8.0  # min coherent residual count to add a candidate
    # Cap on interferers added to one panel. Default 1: same-grid discovery is robust
    # for the dominant single interferer (the validated GFVDEGGLTK two-peak /
    # tail-pulling-secondary failures), but recovering 2+ co-eluting *same-grid*
    # species is weakly identifiable — the unconstrained joint refit overfits (a
    # spurious extra peak raises in-sample likelihood while collapsing the target).
    # Lifting this cap needs the parsimony / identification machinery (Λ, VB-on-panel)
    # tracked for the next phase; until then keep it at 1 for production safety.
    max_candidates: int = 1
    min_isotopes: int = 2  # a candidate must show an isotope duo/trio
    prune_n_floor: float = 1.0  # drop a refit candidate below this N_total
    fit_optimizer: str = "de"
    fit_seed: int | None = 0


def _residual_run(
    r_scan: torch.Tensor, s_star: int, *, frac: float, noise_floor: float
) -> tuple[int, int]:
    """Grow a contiguous RT run outward from `s_star` while the per-scan residual
    stays above `max(frac × r_scan[s_star], noise_floor)`."""
    n = int(r_scan.shape[0])
    thr = max(frac * float(r_scan[s_star]), noise_floor)
    lo = hi = s_star
    while lo - 1 >= 0 and float(r_scan[lo - 1]) > thr:
        lo -= 1
    while hi + 1 < n and float(r_scan[hi + 1]) > thr:
        hi += 1
    return lo, hi


def _dominant_charge(r_win: torch.Tensor, channel_z: torch.Tensor) -> int:
    """The charge whose channels carry the most residual over the run (`r_win` is
    the residual summed over the run scans, per channel `(C,)`)."""
    best_z, best = int(channel_z[0]), -1.0
    for z in channel_z.unique().tolist():
        s = float(r_win[channel_z == z].sum())
        if s > best:
            best, best_z = s, int(z)
    return best_z


def _isotope_coherent(
    r_win: torch.Tensor,
    channel_z: torch.Tensor,
    channel_isotope: torch.Tensor,
    z_star: int,
    *,
    min_isotopes: int,
) -> bool:
    """A real species shows an isotope envelope, not a single-channel spike:
    ≥ `min_isotopes` positive isotopes in charge `z_star`, with the monoisotopic
    (M+0) carrying a substantial share (≥30% of the max isotope). Strict
    monotonicity is NOT required (Poisson noise breaks it; large peptides have
    M+1 ≥ M+0)."""
    sel = channel_z == z_star
    iso, vals = channel_isotope[sel], r_win[sel]
    if int((vals > 0).sum()) < min_isotopes:
        return False
    vmax = float(vals.max())
    m0 = vals[iso == 0]
    return vmax > 0.0 and m0.numel() > 0 and float(m0.max()) >= 0.3 * vmax


@torch.no_grad()
def discover_candidates(
    panel: Panel,
    obs: CounterObservation,
    residual: torch.Tensor,
    region: RegionPeaks,
    *,
    threshold: float,
    min_isotopes: int = 2,
    calibration: GlobalCalibration | None = None,
) -> list[Progenitor]:
    """Propose an interferer progenitor from coherent unexplained residual signal —
    **the discovery seam** (same-grid scope).

    Finds the dominant residual RT-peak outside every existing progenitor's core,
    checks it has an isotope envelope and an integrated count ≥ `threshold` ions, then
    returns ONE pre-seeded `Progenitor` cloned onto the target's ion-node grid
    (the loop adds it, refits, and re-invokes discovery — interferers accrue one
    per turn). Returns `[]` when nothing coherent clears the gates. `region`
    (off-grid candidates) and `calibration` are unused at the same-grid scope
    (the clone reuses the target's grid + calibration); both feed the PR-6
    union-grid path. The signature is the fixed seam contract."""
    r = residual.clamp(min=0.0)  # (S, C)
    r_scan = r.sum(dim=1)  # (S,)
    if float(r_scan.max()) <= 0.0:
        return []

    # Mask the core of EVERY existing progenitor — the scans each already explains
    # (predicts substantial intensity AND within ±N·σ of its apex) — so the dominant
    # OFF-CORE residual peak (a genuinely new species) is found, not an existing
    # peak's own apex misfit. Masking only the target leaves a just-added interferer's
    # own residual to re-trigger discovery and split its peak (spurious over-discovery
    # that overfits and collapses the target). A tail-secondary on a decaying τ has
    # low predicted intensity there, so it survives.
    target = panel.progenitors[0]  # clone template + N reference (shared same grid)
    core = torch.zeros_like(r_scan, dtype=torch.bool)
    for q in panel.progenitors:
        i_q = q.predict(obs)[0].sum(dim=1)  # (S,)
        apex_q = i_q.max()
        in_pred = (
            i_q > _CORE_FRAC * apex_q
            if float(apex_q) > 0.0
            else torch.zeros_like(i_q, dtype=torch.bool)
        )
        in_sigma = (obs.rt - q.peak.mu.detach()).abs() <= _CORE_N_SIGMA * q.peak.sigma.detach()
        core |= in_pred & in_sigma
    r_off = r_scan.clone()
    r_off[core] = 0.0
    if float(r_off.max()) <= 0.0:
        return []  # nothing unexplained outside the existing peaks' cores
    s_star = int(torch.argmax(r_off))

    # Grow the run on the off-core residual at the candidate apex's base
    # (`_RUN_FRAC` of its height) — the seeded-N gate below, not a noise_floor here,
    # guards against spurious runs (a noise_floor from the global residual median is
    # inflated by the target's own apex misfit and collapses the run).
    lo, hi = _residual_run(r_off, s_star, frac=_RUN_FRAC, noise_floor=0.0)

    r_win = r[lo : hi + 1].sum(dim=0)  # (C,)
    z_star = _dominant_charge(r_win, obs.channel_z)
    if not _isotope_coherent(r_win, obs.channel_z, obs.channel_isotope, z_star,
                             min_isotopes=min_isotopes):
        return []

    # Same-grid interferer: a clone of the target's ion-node grid (shares the
    # channels + calibration; charge_energy left uniform — the joint refit recovers
    # the per-charge split). Pre-seed its peak from the run's residual.
    interferer = Progenitor(
        charges=target.charges.tolist(),
        isotope_fractions=target.p_k.detach(),
        calibration=panel.calibration,
        channel_mz=target.channel_mz.detach(),
    )
    r_masked = torch.zeros_like(obs.intensity)
    r_masked[lo : hi + 1, :] = r[lo : hi + 1, :]
    seed_peak_from_observation(
        interferer, replace(obs, intensity=r_masked), rt_prior_ms=float(obs.rt[s_star])
    )

    # Gate on the seeded N_total — a proper trapezoidal ion count, the SAME measure
    # as the target's N (consistent comparison), and noise-relative: clear an
    # absolute floor AND a fraction of the target's N (a bright target's Poisson
    # residual otherwise trips an 8-ion absolute floor).
    n_interferer = float(interferer.peak.N_total.detach())
    floor = max(threshold, _MIN_FRAC_OF_TARGET * float(target.peak.N_total.detach()))
    if n_interferer < floor:
        return []
    return [interferer]


def _panel_bounds(panel: Panel, obs: CounterObservation) -> dict[str, tuple[float, float]]:
    """Per-progenitor `_default_bounds` namespaced to the panel's registered
    parameter names (`progenitors.{i}.…`), plus a data-scaled background bound."""
    bounds: dict[str, tuple[float, float]] = {}
    for i, q in enumerate(panel.progenitors):
        for name, rng in _default_bounds(q, obs).items():
            bounds[f"progenitors.{i}.{name}"] = rng
    if panel.log_background is not None:
        detected = obs.intensity[obs.mask]
        i_max = float(detected.max()) if detected.numel() else 1.0
        i_max = max(i_max, 1.0)
        bounds["log_background"] = (math.log(i_max * 1e-9), math.log(i_max))
    return bounds


@contextlib.contextmanager
def _pinned_torch_rng(seed: int | None):
    """Pin torch's global RNG to `seed` for a reproducible fit, restoring the
    caller's RNG state on exit (a no-op when `seed is None`)."""
    if seed is None:
        yield
        return
    state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(state)


def fit_panel(
    panel: Panel,
    obs: CounterObservation,
    *,
    optimizer: str = "de",
    seed: int | None = 0,
    pop_size: int = 24,
    max_evals: int = 4000,
    max_iter: int = 100,
    lr: float = 0.05,
) -> Any:
    """MAP-fit the whole panel with a **fresh** optimizer (the clean reset after a
    candidate is added — never reuse an optimizer across a mutation). Reuses
    `Panel.log_prob → panel_log_prob` via `Distribution.fit`. Bounds are the
    per-progenitor `_default_bounds` namespaced to the panel (+ background)."""
    bounds = _panel_bounds(panel, obs)
    if optimizer == "de":
        # DE draws its population from the *global* torch RNG (its `seed` arg alone
        # leaves a residual dependence on the ambient RNG state). Pin the global RNG
        # to `seed` for the search and restore the caller's state after, so the fit
        # is reproducible *given its inputs* and independent of the caller's RNG
        # context — e.g. a peptide's result does not depend on how many were fit
        # before it in a worker. (Full end-to-end determinism additionally needs a
        # seed before observation construction, which also draws global RNG.)
        with _pinned_torch_rng(seed):
            de = DifferentialEvolution(
                panel, bounds=bounds, pop_size=pop_size, max_evals=max_evals, seed=seed
            )
            return panel.fit(obs, optimizer=de, max_iter=max_iter, polish_on_converge=True)
    if optimizer == "adam":
        return panel.fit(obs, optimizer=AdamOptimizer(panel, lr=lr), max_iter=2000, tol=1e-8)
    raise ValueError(f"optimizer must be 'de' or 'adam'; got {optimizer!r}")


def _target_n_from_panel(
    panel: Panel,
    obs: CounterObservation,
    *,
    target_index: int,
    credible_level: float,
    n_discovered: int,
    converged: bool,
) -> dict[str, Any]:
    """The target progenitor's `COUNTER_N_TABLE` numeric fields + a Laplace
    credible interval on its peak block (conditional on the rest of the panel),
    plus the discovered-interferer diagnostics."""
    target = panel.progenitors[target_index]
    log_n = float(target.peak.log_N_total.detach())
    block = [f"progenitors.{target_index}.peak.*"]
    std = math.nan
    try:
        cov = laplace_cov(panel, obs, subset=block)
        var = float(cov[0, 0])  # log_N_total is the first peak parameter
        if math.isfinite(var) and var > 0.0:
            std = math.sqrt(var)
    except Exception:  # noqa: BLE001 — fall back to the conditional curvature
        pass
    if not math.isfinite(std):
        cov = laplace_cov(panel, obs, subset=[f"progenitors.{target_index}.peak.log_N_total"])
        std = float(cov.reshape(()).clamp(min=1e-12).sqrt())
    z = float(torch.special.ndtri(torch.tensor((1.0 + credible_level) / 2.0)))
    return {
        "n_total": math.exp(log_n),
        "n_total_lo": math.exp(log_n - z * std),
        "n_total_hi": math.exp(log_n + z * std),
        "credible_level": float(credible_level),
        "rt_apex": float(target.peak.mu.detach()) / 1000.0,  # ms → s
        "peak_sigma": float(target.peak.sigma.detach()) / 1000.0,
        "peak_tau_r": float(target.peak.tau_r.detach()) / 1000.0,
        "peak_tau_l": float(target.peak.tau_l.detach()) / 1000.0,
        "peak_eta": float(target.peak.eta.detach()),
        "inference_method": "map",
        "converged": bool(converged),
        "final_elbo": None,
        "n_scans_used": int(obs.mask.any(dim=1).sum()),
        "iit_corrected": True,
        "interference_flag": bool(n_discovered > 0),
        "n_discovered_interferers": int(n_discovered),
    }


def estimate_panel(
    panel: Panel,
    obs: CounterObservation,
    region: RegionPeaks,
    *,
    config: DiscoverConfig | None = None,
    target_index: int = 0,
    rt_prior_ms: float | None = None,
    credible_level: float = 0.95,
    inference: str = "map",
    **fit_kw: Any,
) -> dict[str, Any]:
    """Fit-plus-discover the panel and return the target's `N_total` + CI.

    Seeds the target from the data (its PSM RT prior, if given), fits the panel,
    then loops: residual → `discover_candidates` → add + refit → until none clear
    the threshold (or `max_candidates`, or a candidate fails to survive the joint
    refit). Same-grid interferers only (a second species at the target's own m/z
    channels, different RT); off-m/z (union-grid) discovery + VB-on-panel credible
    intervals are deferred, so `inference='map'` is the only supported mode."""
    config = config or DiscoverConfig()
    if inference != "map":
        raise ValueError(
            "estimate_panel supports inference='map'; VB-on-panel is not yet implemented"
        )
    seed_peak_from_observation(panel.progenitors[target_index], obs, rt_prior_ms=rt_prior_ms)
    result = fit_panel(panel, obs, optimizer=config.fit_optimizer, seed=config.fit_seed, **fit_kw)

    while len(panel.progenitors) - 1 < config.max_candidates:
        cands = discover_candidates(
            panel,
            obs,
            panel_residual(panel, obs),
            region,
            threshold=config.detect_threshold,
            min_isotopes=config.min_isotopes,
            calibration=panel.calibration,
        )
        if not cands:
            break  # nothing coherent left to explain
        idx = panel.add_progenitor(cands[0])  # discovery returns a pre-seeded progenitor
        result = fit_panel(panel, obs, optimizer=config.fit_optimizer, seed=config.fit_seed, **fit_kw)
        if float(panel.progenitors[idx].peak.N_total.detach()) < config.prune_n_floor:
            panel.remove_progenitor(idx)  # didn't survive the joint fit
            break

    return _target_n_from_panel(
        panel,
        obs,
        target_index=target_index,
        credible_level=credible_level,
        n_discovered=len(panel.progenitors) - 1,
        converged=bool(getattr(result, "converged", True)),
    )
