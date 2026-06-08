"""Per-channel likelihood terms — intensity (Eq. 16/18), m/z error
(Eq. 23/25), and the censored zero-observation term.

Free, `vmap`-safe functions (broadcasting torch ops only — no boolean
indexing) so they compose inside `Progenitor.log_prob` / `Panel.log_prob`
and survive DE's vmap'd population path and the VI sample batch.

Everything scores the OBSERVABLES directly (intensity, m/z error) — the
injection time `τ` and the latent count enter the *variance / precision*,
never as a pre-transform of the data (see `iit.py`). Units are the ms
convention: intensity [a.u./ms], `τ` [ms], gain [a.u./ion], m/z error
[ppm], `c_mz` [ppm²].
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "student_t_log_prob",
    "intensity_log_prob",
    "mz_error_log_prob",
    "censored_log_prob",
]


def student_t_log_prob(
    x: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """Student-t log-density (location-scale), elementwise / broadcasting.

    Matches `core.stats.StudentT.log_prob`'s stable `log1p(z²/ν)` form but as
    a free function, because Counter's scale is *data-dependent* per channel
    (a per-element tensor), not a single fitted parameter."""
    z = (x - loc) / scale
    return (
        torch.lgamma(0.5 * (nu + 1.0))
        - torch.lgamma(0.5 * nu)
        - 0.5 * torch.log(nu * math.pi)
        - torch.log(scale)
        - 0.5 * (nu + 1.0) * torch.log1p(z * z / nu)
    )


def intensity_log_prob(
    intensity_obs: torch.Tensor,
    intensity_pred: torch.Tensor,
    sum_intensity_pred: torch.Tensor,
    gain_z: torch.Tensor,
    p_k: torch.Tensor,
    iit: torch.Tensor,
    rho_r: torch.Tensor,
    nu_intensity: torch.Tensor,
    *,
    var_floor: float = 1e-12,
) -> torch.Tensor:
    """Intensity-channel log-likelihood (Eq. 16/18).

    `I_obs ~ StudentT(I_pred, √Var, ν_I)` with the multinomial shot-noise
    variance `Var = α(z)·ΣI_pred·p_k(1−p_k) / (τ·ρ_R)` — `τ` (`iit`) and the
    resolution factor `ρ_R` enter HERE, not the mean. `intensity_pred` is the
    per-channel predicted intensity (the additive total over progenitors +
    background, supplied by the Panel); `sum_intensity_pred` is `ΣI_pred`
    over isotopes for that (scan, charge)."""
    var = gain_z * sum_intensity_pred * p_k * (1.0 - p_k) / (iit * rho_r)
    scale = var.clamp(min=var_floor).sqrt()
    return student_t_log_prob(intensity_obs, intensity_pred, scale, nu_intensity)


def mz_error_log_prob(
    mz_error_obs: torch.Tensor,
    center_ppm: torch.Tensor,
    c_mz: torch.Tensor,
    n_count: torch.Tensor,
    alpha_mz: torch.Tensor,
    nu_mz: torch.Tensor,
    *,
    n_floor: float = 1.0,
) -> torch.Tensor:
    """m/z-error-channel log-likelihood (Eq. 23/25).

    `ε_obs ~ StudentT(center, √(c_mz / N_k^{α_mz}), ν_mz)`. `n_count` is the
    *latent* per-scan accumulated count `N_k = N(t)·f_z·p_k·τ` [ions] — using
    the latent (not a plug-in from the observed intensity) is what makes the
    m/z channel inform `N`. `center_ppm` carries the m/z calibration
    (`mz_offset` + `d_mz_k/z`)."""
    scale = (c_mz / n_count.clamp(min=n_floor) ** alpha_mz).clamp(min=1e-12).sqrt()
    return student_t_log_prob(mz_error_obs, center_ppm, scale, nu_mz)


def censored_log_prob(
    n_pred_count: torch.Tensor,
    floor_count: torch.Tensor | float,
) -> torch.Tensor:
    """Log-probability of *no* detection given a predicted count — the proper
    zero-observation term (replaces nb42c's `cens_weight` fudge).

    The count process is Poisson, so `P(no detection) = P(X ≤ floor | λ =
    N_pred) = Q(floor+1, λ)` via the regularized upper incomplete gamma
    (`torch.special.gammaincc`), differentiable w.r.t. the latent `λ`. This is
    the count-native CDF — it deliberately avoids the pinned `StudentT.cdf`
    gap. As `N_pred → 0` the probability → 1 (no penalty); as `N_pred` grows
    past the floor it → 0 (penalizes predicting ions where none were seen)."""
    a = torch.as_tensor(floor_count, dtype=n_pred_count.dtype) + 1.0
    a = torch.broadcast_to(a, n_pred_count.shape)
    p = torch.special.gammaincc(a, n_pred_count).clamp(min=1e-30)
    return torch.log(p)
