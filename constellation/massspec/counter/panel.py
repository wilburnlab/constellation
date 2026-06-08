"""`Panel` — the additive joint score over co-eluting progenitors.

A panel models a `(scan × channel)` observation as an **additive** sum over
its progenitors' predicted intensities plus a background term, with soft
intensity-weighted attribution for the m/z channel and a proper censored
term for non-observations. It is the fittable unit: `Panel.log_prob(obs)`
returns the total joint log-likelihood (a 0-d tensor), so it drops into
`Parametric.fit`, the DE population path, and the VI ELBO unchanged.

PR-1 runs a fixed small candidate set (the seed progenitor + an optional
background channel [+ library neighbours]); spike-and-slab *discovery* of
new candidates is deferred. The likelihood combines:
  * intensity (Eq. 16/18) — observed vs the additive total, Student-t with
    multinomial shot-noise variance (τ + resolution in the variance);
  * m/z (Eq. 23/25) — soft-γ mixture marginal over progenitors (Eq. 26/27
    conditional independence given the latent count);
  * censored (zero-observation) — Poisson tail on the total predicted count.
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

    `progenitors` share `obs`'s channel grid. The intensity-variance `p_k`
    and `ν_I` use the first (target) progenitor's values (the dominant
    species; multi-progenitor blending is a refinement). vmap-safe: only
    broadcasting ops + `torch.where` over a static progenitor list."""
    iit = obs.iit  # (S,)
    target = progenitors[0]

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

    i_pred_stack = torch.stack(i_preds, dim=0)  # (Q, S, C)
    n_count_stack = torch.stack(n_counts, dim=0)
    i_pred_total = i_pred_stack.sum(dim=0) + background_intensity  # (S, C)
    n_count_total = n_count_stack.sum(dim=0)  # (S, C)

    # ΣI_pred per (scan, charge): sum predicted intensity over same-charge channels.
    same_charge = (
        obs.channel_z[:, None] == obs.channel_z[None, :]
    ).to(i_pred_total.dtype)  # (C, C)
    sum_i_pred = i_pred_total @ same_charge.transpose(0, 1)  # (S, C)

    gain_ch = calibration.gain(obs.channel_z.to(iit.dtype))  # (C,)
    p_ch = target.p_k.index_select(0, obs.channel_isotope)  # (C,)
    rho_r = calibration.rho_R(obs.channel_mz)  # (C,)
    nu_i = target.nu_intensity

    # Intensity term (observed channels).
    int_lp = ch.intensity_log_prob(
        obs.intensity,
        i_pred_total,
        sum_i_pred,
        gain_ch[None, :],
        p_ch[None, :],
        iit[:, None],
        rho_r[None, :],
        nu_i,
    )
    zeros = torch.zeros_like(int_lp)
    int_term = torch.where(obs.mask, int_lp, zeros).sum()

    # m/z term (observed channels): soft-γ mixture marginal.
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

    # Censored term (unobserved channels): Poisson tail on the total count.
    cens_lp = ch.censored_log_prob(n_count_total, floor_ions)  # (S, C)
    cens_term = torch.where(obs.mask, zeros, cens_lp).sum()

    return int_term + mz_term + cens_term


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
