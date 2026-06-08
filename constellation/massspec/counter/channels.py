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
    "poisson_count_log_prob",
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
    # NOTE: legacy / high-N Gaussian-shot-noise term. The active observed-
    # channel likelihood is the count-native `poisson_count_log_prob` (below):
    # this multinomial-conditional variance under-estimates the marginal Poisson
    # variance by (1-p_k) and mis-models the discrete low-count + detection-edge
    # regime. Retained for high-N ablation / comparison.
    var = gain_z * sum_intensity_pred * p_k * (1.0 - p_k) / (iit * rho_r)
    scale = var.clamp(min=var_floor).sqrt()
    return student_t_log_prob(intensity_obs, intensity_pred, scale, nu_intensity)


def poisson_count_log_prob(
    n_obs: torch.Tensor,
    lam: torch.Tensor,
    *,
    lam_floor: float = 1e-12,
) -> torch.Tensor:
    """Poisson log-PMF of an OBSERVED (detected) channel's recovered count
    ``n_obs = I_obs*tau/alpha`` given the predicted rate ``lam = N_count`` --
    the active intensity term:

        log P(count = n_obs | lam) = n_obs*log(lam) - lam - lgamma(n_obs+1)

    Paired with `censored_log_prob` (``log P(count < floor)``) on UNOBSERVED
    channels, this is the **complete left-censored-Poisson likelihood**:
    ``Sum_detected log P(count=n_obs) + Sum_undetected log P(count < floor)`` --
    the exact, consistent (unbiased) likelihood for counts left-censored at the
    detection floor. The detection event is modelled *once*, jointly, by this
    partition: detected channels contribute the unconditional PMF, undetected
    the complementary tail mass.

    This is count-native, so Var = lam automatically (retires the Eq.16
    multinomial-conditional ``p(1-p)`` variance), and the exact PMF fixes the
    discrete low-count regime where the continuous Student-t over-/under-shoots.
    Do NOT additionally divide by ``P(count >= floor)`` (a *conditional* /
    truncated framing): combined with the censored term that double-counts the
    detection event and biases ``N`` -- the complete-likelihood partition above
    is the correct, bias-free construction.

    Bounded above by 0 (a log-PMF), so it cannot reward ``lam -> 0`` (the
    ``n_obs*log lam`` penalty -> -inf there, ``n_obs >= floor >= 1``). High-N
    equivalent: ``Poisson(lam) -> Gaussian(lam, lam)``, matching the legacy
    Student-t term up to a constant for ``lam >> 1``.

    ``n_obs`` is continuous (real ``I*tau/alpha``); ``lgamma`` admits
    non-integer counts so the term is differentiable w.r.t. ``lam`` with no
    rounding. The ``I<->count`` Jacobian ``tau/alpha`` is ``N``-independent
    (alpha frozen) and drops out -- re-add it if alpha is ever co-fit."""
    lam = lam.clamp(min=lam_floor)
    return n_obs * torch.log(lam) - lam - torch.lgamma(n_obs + 1.0)


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
    """Log-probability of *non-detection* (``count < floor``) given a predicted
    count -- the proper zero-observation term (replaces nb42c's ``cens_weight``
    fudge), and the complement of `poisson_count_log_prob`'s detection
    normalizer over the same threshold.

    The count is Poisson, so ``P(count < floor | lam=N_pred) = P(count <=
    floor-1) = gammaincc(floor, lam)`` via the regularized upper incomplete
    gamma (count-native, differentiable w.r.t. ``lam``, sidesteps the pinned
    ``StudentT.cdf`` gap). As ``N_pred -> 0`` -> 1 (no penalty); as ``N_pred``
    grows past the floor -> 0 (penalizes predicting ions where none were
    seen)."""
    # P(count < floor) = P(count <= floor-1) = gammaincc(floor, lam) (the
    # regularized UPPER incomplete gamma; verified gammaincc(1,lam)=e^-lam=P(X=0)).
    # NOT gammaincc(floor+1, ...) = P(count <= floor), which would double-assign
    # count==floor to both detection and censoring. This complements
    # poisson_count_log_prob's detection normalizer P(count >= floor) over the
    # same threshold.
    a = torch.broadcast_to(
        torch.as_tensor(float(floor_count), dtype=n_pred_count.dtype),
        n_pred_count.shape,
    )
    p = torch.special.gammaincc(a, n_pred_count).clamp(min=1e-30)
    return torch.log(p)
