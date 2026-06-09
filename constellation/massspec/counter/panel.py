"""`Panel` -- the additive joint score over co-eluting progenitors.

A panel models a `(scan x channel)` observation as an **additive** sum over
its progenitors' predicted counts plus a background, with soft
intensity-weighted attribution for the m/z channel and a proper detection
model for every channel. It is the fittable unit: `Panel.log_prob(obs)`
returns the total joint log-likelihood (a 0-d tensor), so it drops into
`Parametric.fit`, the DE population path, and the VI ELBO unchanged.

PR-1 runs a fixed small candidate set (the seed progenitor + an optional
background channel [+ library neighbours]); spike-and-slab *discovery* of
new candidates is deferred. The likelihood combines:
  * intensity -- a **Poisson PMF** on the recovered count `N_obs =
    I_obs*tau/alpha` of each OBSERVED channel: `log Poisson(N_obs | lambda)`,
    `lambda = N_count`. Count-native (Var = lambda); the exact PMF fixes the
    discrete low-count regime the continuous Student-t mis-modelled.
    (`channels.poisson_count_log_prob`.) Plus the change-of-variables Jacobian
    `log(tau/alpha)` per observed channel (the data are intensities, not
    counts): constant when alpha is frozen, but required to make the likelihood
    proper in alpha when the gain is co-fit (else the PMF's preference for
    smaller counts pulls alpha up without bound).
  * m/z (Eq. 23/25) -- soft-gamma mixture marginal over progenitors (Eq. 26/27
    conditional independence given the latent count);
  * censored (non-detection) -- Poisson tail `P(count < floor | lambda)` on
    UNOBSERVED channels. Together with the observed PMF this is the **complete
    left-censored-Poisson likelihood**: detected channels score the
    unconditional PMF, undetected the complementary tail mass, partitioning the
    Poisson over disjoint events (`>= floor` vs `< floor`) -- no double-counting
    and the correct, bias-free treatment of the detection edge. (Do NOT also
    divide the observed term by P(count >= floor): that conditional/truncated
    framing, combined with this censored term, double-models detection and
    biases N -- it is the construction that collapsed N in an earlier attempt.)
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from constellation.core.stats.parametric import Distribution

from . import channels as ch
from .attribution import mz_mixture_log_prob, responsibilities
from .calibration import GlobalCalibration
from .model import CounterObservation, Progenitor

__all__ = ["Panel", "panel_log_prob"]

_DEFAULT_FLOOR_IONS = 1.0


def panel_log_prob(
    progenitors: Sequence[Progenitor],
    obs: CounterObservation,
    calibration: GlobalCalibration,
    *,
    background_intensity: torch.Tensor | float = 0.0,
    floor_ions: float = _DEFAULT_FLOOR_IONS,
) -> torch.Tensor:
    """Total additive joint log-likelihood (0-d tensor) for a panel.

    `progenitors` share `obs`'s channel grid. The latent per-channel rate is
    the additive sum of progenitor counts (+ background as a count rate); the
    observed-channel intensity is scored count-natively (detection-conditioned
    Poisson), so there is no per-charge total / `(1-p)` variance to assemble.
    vmap-safe: only broadcasting ops + `torch.where` over a static progenitor
    list."""
    iit = obs.iit  # (S,)

    # Per-progenitor predictions + m/z scales.
    i_preds: list[torch.Tensor] = []
    n_counts: list[torch.Tensor] = []
    mz_scales: list[torch.Tensor] = []
    for q in progenitors:
        i_pred_q, n_count_q = q.predict(obs)
        i_preds.append(i_pred_q)
        n_counts.append(n_count_q)
        c_mz_ch = q.c_mz_per_channel()  # (C,)
        scale_q = (
            c_mz_ch[None, :] / n_count_q.clamp(min=1.0) ** calibration.alpha_mz
        ).clamp(min=1e-12).sqrt()
        mz_scales.append(scale_q)

    i_pred_stack = torch.stack(i_preds, dim=0)  # (Q, S, C) -- m/z responsibilities
    n_count_stack = torch.stack(n_counts, dim=0)  # (Q, S, C)
    n_count_total = n_count_stack.sum(dim=0)  # (S, C) latent Poisson rate lambda

    gain_ch = calibration.gain(obs.channel_z.to(iit.dtype))  # (C,)
    # Background as an additive count rate (unmonitored density at the channel):
    # convert the intensity-units background to counts via N = I*tau/alpha.
    if not isinstance(background_intensity, float) or background_intensity != 0.0:
        n_count_total = n_count_total + (
            background_intensity * iit[:, None] / gain_ch[None, :]
        )

    # Intensity term (observed channels): Poisson PMF on the recovered count
    # N_obs = I_obs * tau / alpha(z). With the censored tail below this is the
    # complete left-censored-Poisson likelihood (no detection normalizer here).
    n_obs = obs.intensity * iit[:, None] / gain_ch[None, :]  # (S, C)
    int_lp = ch.poisson_count_log_prob(n_obs, n_count_total)
    zeros = torch.zeros_like(int_lp)
    int_term = torch.where(obs.mask, int_lp, zeros).sum()

    # m/z term (observed channels): soft-gamma mixture marginal. The intensity
    # responsibilities are equivalent whether built from i_pred or n_count (they
    # differ only by the per-channel constant alpha/tau, which cancels).
    center = calibration.mz_center_ppm(
        obs.channel_mz, obs.channel_z.to(iit.dtype), obs.channel_isotope
    )  # (C,)
    gamma = responsibilities(i_pred_stack)  # (Q, S, C)
    component_lp = torch.stack(
        [
            ch.student_t_log_prob(
                obs.mz_error, center[None, :], mz_scales[q], calibration.nu_mz
            )
            for q in range(len(progenitors))
        ],
        dim=0,
    )  # (Q, S, C)
    mz_lp = mz_mixture_log_prob(gamma, component_lp)  # (S, C)
    mz_term = torch.where(obs.mask, mz_lp, zeros).sum()

    # Censored term (unobserved channels): Poisson non-detection tail
    # P(count < floor) -- the complement that completes the left-censored
    # Poisson likelihood over the same floor.
    cens_lp = ch.censored_log_prob(n_count_total, floor_ions)  # (S, C)
    cens_term = torch.where(obs.mask, zeros, cens_lp).sum()

    # Change-of-variables Jacobian: the DATA are intensities I, but the
    # observed-channel term scores the recovered count N_obs = I*tau/alpha, so
    # the proper data log-density adds log|dN_obs/dI| = log(tau / alpha(z)) per
    # observed channel. This is N-INDEPENDENT, so it is a constant that drops
    # out of the per-peptide N / shape fit when alpha is frozen (estimate_n).
    # It is REQUIRED when alpha is co-fit (calibration): without the -log(alpha)
    # the Poisson PMF's preference for smaller counts pulls the gain up (and N
    # down) without bound; with it, the shot-noise structure (Var ~ alpha^2 * N
    # vs Mean ~ alpha * N) identifies alpha. (mz_error, in ppm, carries no
    # alpha-dependent Jacobian.)
    log_jac = torch.log(iit[:, None]) - torch.log(gain_ch[None, :])  # (S, C)
    jac_term = torch.where(obs.mask, log_jac, zeros).sum()

    return int_term + mz_term + cens_term + jac_term


class Panel(Distribution):
    """Additive joint score over a fixed candidate set + optional background.

    Registers the shared `calibration` once and the progenitors as a
    `ModuleList`; the progenitors reference the same calibration object, so
    `functional_call` patches it correctly without duplicating its params.
    """

    def __init__(
        self,
        progenitors: Sequence[Progenitor],
        calibration: GlobalCalibration,
        *,
        background: bool = False,
        floor_ions: float = _DEFAULT_FLOOR_IONS,
        background_init: float = 1e-3,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if len(progenitors) == 0:
            raise ValueError("a Panel needs at least one progenitor")
        self.progenitors = nn.ModuleList(progenitors)
        self.calibration = calibration
        self.floor_ions = float(floor_ions)
        if background:
            self.log_background = nn.Parameter(
                torch.tensor(float(background_init), dtype=dtype).log()
            )
        else:
            self.log_background = None

    def background_intensity(self) -> torch.Tensor | float:
        if self.log_background is None:
            return 0.0
        return self.log_background.exp()

    def log_prob(self, obs: CounterObservation) -> torch.Tensor:
        return panel_log_prob(
            list(self.progenitors),
            obs,
            self.calibration,
            background_intensity=self.background_intensity(),
            floor_ions=self.floor_ions,
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Panel has no closed-form CDF.")

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for i, q in enumerate(self.progenitors):
            for name, val in q.parameters_dict().items():
                out[f"progenitor_{i}.{name}"] = val
        if self.log_background is not None:
            out["background"] = self.log_background.exp().detach()
        return out
