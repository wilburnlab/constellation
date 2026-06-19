"""Prior terms for Counter fits — composable `log_prior(params) → Tensor`.

Each term reduces over its parameter's own dims while preserving any leading
sample dim, so the same callable serves the VB ELBO's batched param dict
(`(n_samples, *)`) and a single param dict. Two priors ship:

  * **RT-seed** — a Gaussian on the peak center `peak.mu` at a seed RT (e.g. a
    PSM apex). This is how the RT seed enters as an *informative prior* rather
    than a hard extraction window (the design's core reframing): it shapes the
    posterior on `μ` without gating which scans enter the likelihood.
  * **Isotope toward-theoretical** — a Gaussian pulling `isotope_energy_offset`
    toward 0 (i.e. the effective fractions toward the theoretical envelope), so
    a low-SNR peptide's fraction correction shrinks to natural abundance unless
    the data argue otherwise. Active only when the offset is being inferred.
"""

from __future__ import annotations

from typing import Callable

import torch

__all__ = ["make_log_prior", "LogPrior"]

LogPrior = Callable[[dict[str, torch.Tensor]], torch.Tensor]


def make_log_prior(
    *,
    rt_prior_ms: float | None = None,
    rt_sigma_ms: float = 5000.0,
    n_total_center: float | None = None,
    n_total_sigma: float = 2.0,
    isotope_offset_sigma: float | None = None,
    eta_center: float | None = None,
    eta_sigma: float = 1.0,
    shape_centers: dict[str, float] | None = None,
    shape_sigmas: dict[str, float] | None = None,
    mu_key: str = "peak.mu",
    n_total_key: str = "peak.log_N_total",
    offset_key: str = "isotope_energy_offset",
    eta_key: str = "peak.logit_eta",
) -> LogPrior | None:
    """Compose the active prior terms into one `log_prior(params) → Tensor`, or
    return `None` if no term is active.

    - `rt_prior_ms` → RT-seed Gaussian on `peak.mu` (σ = `rt_sigma_ms`). For the
      panel path pass the N-weighted-μ centroid; for a PSM pass its RT.
    - `n_total_center` (not `None`, in **log space**) → Gaussian on
      `peak.log_N_total` toward `n_total_center` (σ = `n_total_sigma`, log space).
      This is the L6 `N_total → log(sum(N_guess))` bias: a *soft* anti-collapse /
      anti-runaway regularizer (default σ = 2.0 ≈ a 7× band, so the likelihood
      dominates wherever it is informative and it does NOT fight the discovery
      loop's interference correction — it only matters when the likelihood is flat
      at very low N). Pass the seed's integrated-count estimate as the center.
    - `isotope_offset_sigma` (not `None`) → toward-theoretical isotope-offset
      Gaussian (toward 0).
    - `eta_center` (not `None`) → Gaussian on the HyperEMG `logit_eta` toward
      `eta_center` (σ = `eta_sigma`). This tames the η degeneracy: as η → 0/1 a
      tail weight vanishes and its `τ` becomes unconstrained and drifts to NaN
      under gradient VB; a mild prior toward the MAP η keeps both tails
      identified while still quantifying η's uncertainty.
    - `shape_centers` (`{full_param_name: mean}`, e.g. a `StagedCalibration`
      peak-shape hyperprior) → a Gaussian on each named log/logit shape param
      toward `mean` (σ from `shape_sigmas[name]`, default 1.0). This is how the
      promoted population shape distribution regularizes a per-peptide fit — the
      principled replacement for a hard `τ`-bound / RT window.

    Keys (`mu_key`, `n_total_key`, …) are constructor args so a multi-progenitor
    panel caller passes the namespaced names (`progenitors.{i}.peak.mu`). The
    active terms index those keys directly, so a name that is absent raises (a
    namespace typo is loud, not a silent no-op)."""
    terms: list[LogPrior] = []

    if shape_centers:
        centers = {k: float(v) for k, v in shape_centers.items()}
        sigmas = {k: float((shape_sigmas or {}).get(k, 1.0)) for k in centers}

        def _shape(params: dict[str, torch.Tensor]):
            total: torch.Tensor | float = 0.0
            for key, ctr in centers.items():
                val = params.get(key)
                if val is not None:
                    total = total - 0.5 * ((val - ctr) / sigmas[key]) ** 2
            return total

        terms.append(_shape)

    if eta_center is not None:
        e0, es = float(eta_center), float(eta_sigma)

        def _eta(params: dict[str, torch.Tensor]) -> torch.Tensor:
            return -0.5 * ((params[eta_key] - e0) / es) ** 2

        terms.append(_eta)

    if rt_prior_ms is not None:
        mu0, sig = float(rt_prior_ms), float(rt_sigma_ms)

        def _rt(params: dict[str, torch.Tensor]) -> torch.Tensor:
            return -0.5 * ((params[mu_key] - mu0) / sig) ** 2

        terms.append(_rt)

    if n_total_center is not None:
        ln0, lns = float(n_total_center), float(n_total_sigma)

        def _n_total(params: dict[str, torch.Tensor]) -> torch.Tensor:
            return -0.5 * ((params[n_total_key] - ln0) / lns) ** 2

        terms.append(_n_total)

    if isotope_offset_sigma is not None:
        sig = float(isotope_offset_sigma)

        def _iso(params: dict[str, torch.Tensor]):
            off = params.get(offset_key)
            if off is None or off.shape[-1] == 0:
                return torch.zeros(())
            return -0.5 * ((off / sig) ** 2).sum(dim=-1)

        terms.append(_iso)

    if not terms:
        return None

    def log_prior(params: dict[str, torch.Tensor]) -> torch.Tensor:
        total: torch.Tensor | float = 0.0
        for term in terms:
            total = total + term(params)
        return total  # type: ignore[return-value]

    return log_prior
