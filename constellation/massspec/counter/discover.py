"""Fit-plus-discover loop + the progenitor-discovery seam (panel-shaped estimate).

The per-seed estimation unit is a `Panel` (one target progenitor + an additive
background), fit jointly. The loop fits the panel, inspects the residual at its
ion nodes, and — when discovery finds coherent unexplained signal — adds an
interferer progenitor and refits, so the interferer OWNS its signal and the
target's `N` is freed of it (the "ownership mapping" fix).

PR-4 ships the loop, the `Panel` mutability, the residual substrate, and the
discovery **hook** (`discover_candidates`), but the hook is a **no-op stub** —
so the loop fits target+background once and the estimate equals `estimate_n`'s.
PR-5 fills in `discover_candidates` (coherent-residual detection → a new
`Progenitor`), first scoped to same-grid interferers (new edges only to the
target's existing ion nodes — no union grid).

Inference: PR-4 reports the target's `N_total` with a **Laplace** credible
interval on its peak block (works for any panel membership). Variational
credible intervals over a multi-progenitor panel are a PR-5 generalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DiscoverConfig:
    """Knobs for the fit-plus-discover loop."""

    detect_threshold: float = 8.0  # min coherent residual count to add a candidate
    max_candidates: int = 4  # cap on interferers added to one panel
    min_isotopes: int = 2  # a candidate must show an isotope duo/trio
    prune_n_floor: float = 1.0  # drop a refit candidate below this N_total
    fit_optimizer: str = "de"
    fit_seed: int | None = 0


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
    """Propose interferer progenitors from coherent unexplained residual signal —
    **the discovery seam**.

    PR-4: a no-op stub returning `[]` (the loop then fits target+background once).
    PR-5 implements it: scan `residual` (and the off-grid `region` candidates) for
    a coherent peak (an RT run with an isotope partner whose integrated count ≥
    `threshold`), instantiate a `Progenitor` (pre-seeded) at the candidate m/z, and
    return it — first scoped to same-grid interferers. The signature/contract is
    fixed here so PR-5 is a drop-in."""
    return []


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
    refit). With PR-4's stub hook the loop fits once. `inference='map'` only in
    PR-4 (VB-on-panel is PR-5)."""
    config = config or DiscoverConfig()
    if inference != "map":
        raise ValueError(
            "estimate_panel supports inference='map' in PR-4; VB-on-panel is PR-5"
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
