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
    isotope_offset_sigma: float | None = None,
    eta_center: float | None = None,
    eta_sigma: float = 1.0,
    mu_key: str = "peak.mu",
    offset_key: str = "isotope_energy_offset",
    eta_key: str = "peak.logit_eta",
) -> LogPrior | None:
    """Compose the active prior terms into one `log_prior(params) → Tensor`, or
    return `None` if no term is active.

    - `rt_prior_ms` → RT-seed Gaussian on `peak.mu` (σ = `rt_sigma_ms`).
    - `isotope_offset_sigma` (not `None`) → toward-theoretical isotope-offset
      Gaussian (toward 0).
    - `eta_center` (not `None`) → Gaussian on the HyperEMG `logit_eta` toward
      `eta_center` (σ = `eta_sigma`). This tames the η degeneracy: as η → 0/1 a
      tail weight vanishes and its `τ` becomes unconstrained and drifts to NaN
      under gradient VB; a mild prior toward the MAP η keeps both tails
      identified while still quantifying η's uncertainty."""
    terms: list[LogPrior] = []

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
