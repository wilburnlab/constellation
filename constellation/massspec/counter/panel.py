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

from typing import NamedTuple, Sequence

import torch
from torch import nn

from constellation.core.stats.parametric import Distribution

from . import channels as ch
from .attribution import mz_mixture_log_prob, responsibilities
from .calibration import GlobalCalibration
from .model import CounterObservation, Progenitor

__all__ = ["Panel", "panel_log_prob", "panel_cell_log_prob", "PanelCellTerms"]

_DEFAULT_FLOOR_IONS = 1.0


class PanelCellTerms(NamedTuple):
    """Per-cell decomposition of the panel likelihood — the unreduced `(S, C)`
    pieces `panel_log_prob` sums to a scalar.

    The four additive likelihood terms are dense `(S, C)` log-densities; the soft
    attribution and its inputs are `(Q, S, C)`. Per-ion scores (L3), logL-based
    ion selection (L4), and the ion→progenitor attribution map (L8) all read
    these. `observed` / `censored` are the *effective* per-cell masks after any
    exclusion (`exclude_mask` / `gamma_loss_threshold`): excluded cells appear in
    neither, so they are still *inferred* (predicted) but do not contribute to the
    loss. A NamedTuple (hence a vmap-safe pytree) so it can flow through the DE
    population path unchanged.
    """

    intensity: torch.Tensor  # (S, C) Poisson-PMF log-density (observed cells)
    mz: torch.Tensor  # (S, C) soft-γ mixture-marginal m/z log-density
    censored: torch.Tensor  # (S, C) non-detection tail log-prob
    jacobian: torch.Tensor  # (S, C) log(τ/α) change-of-variables
    gamma: torch.Tensor  # (Q, S, C) responsibilities, Σ_q γ_q = 1
    component_mz: torch.Tensor  # (Q, S, C) per-progenitor m/z log-density (pre-mixture)
    n_count: torch.Tensor  # (Q, S, C) per-progenitor latent count
    i_pred: torch.Tensor  # (Q, S, C) per-progenitor predicted intensity
    observed: torch.Tensor  # (S, C) bool — cells scored by intensity + mz + jac
    censored_mask: torch.Tensor  # (S, C) bool — cells scored by the censored tail


def _panel_terms(
    progenitors: Sequence[Progenitor],
    obs: CounterObservation,
    calibration: GlobalCalibration,
    *,
    background_intensity: torch.Tensor | float = 0.0,
    floor_ions: float = _DEFAULT_FLOOR_IONS,
    exclude_mask: torch.Tensor | None = None,
    gamma_loss_threshold: float | None = None,
) -> PanelCellTerms:
    """Compute every per-cell likelihood piece (no scan/channel reduction). The
    single source of truth for both `panel_log_prob` (the scalar) and
    `panel_cell_log_prob` (the structured per-cell view). vmap-safe: only
    broadcasting ops + `torch.where` over a static progenitor list."""
    iit = obs.iit  # (S,)

    # Per-progenitor predictions, m/z scales, and per-progenitor m/z centers. The
    # center is a GENERAL (observed, expected) quantity: each candidate is scored
    # at ITS OWN theoretical m/z relative to the observed grid (mz_center_ppm's
    # reference_mz = obs.channel_mz), so a near-isobaric blend resolves by the
    # candidates' distinct mass defects. For the species that defines the grid
    # (the target, and same-grid clones) the theoretical Δ is 0, so the center is
    # exactly the single-species center — backward-compatible.
    i_preds: list[torch.Tensor] = []
    n_counts: list[torch.Tensor] = []
    mz_scales: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    for q in progenitors:
        i_pred_q, n_count_q = q.predict(obs)
        i_preds.append(i_pred_q)
        n_counts.append(n_count_q)
        c_mz_ch = q.c_mz_per_channel()  # (C,)
        scale_q = (
            c_mz_ch[None, :] / n_count_q.clamp(min=1.0) ** calibration.alpha_mz
        ).clamp(min=1e-12).sqrt()
        mz_scales.append(scale_q)
        centers.append(
            calibration.mz_center_ppm(
                q.channel_mz,
                q.channel_z.to(iit.dtype),
                q.channel_isotope,
                reference_mz=obs.channel_mz,
            )
        )

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

    # m/z term (observed channels): soft-gamma mixture marginal. The intensity
    # responsibilities are equivalent whether built from i_pred or n_count (they
    # differ only by the per-channel constant alpha/tau, which cancels).
    gamma = responsibilities(i_pred_stack)  # (Q, S, C)
    component_lp = torch.stack(
        [
            ch.student_t_log_prob(
                obs.mz_error, centers[q][None, :], mz_scales[q], calibration.nu_mz
            )
            for q in range(len(progenitors))
        ],
        dim=0,
    )  # (Q, S, C)
    mz_lp = mz_mixture_log_prob(gamma, component_lp)  # (S, C)

    # Censored term (unobserved channels): Poisson non-detection tail
    # P(count < floor) -- the complement that completes the left-censored
    # Poisson likelihood over the same floor.
    cens_lp = ch.censored_log_prob(n_count_total, floor_ions)  # (S, C)

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

    # Effective per-cell masks. A cell can be excluded from the loss (but still
    # inferred) by an explicit per-cell `exclude_mask` (L4 outlier/blend hold-out)
    # and/or a γ-purity threshold (D9 benchmark: drop observed cells no single
    # progenitor owns at γ ≥ threshold). Default (neither set) → observed = mask,
    # censored = ~mask, i.e. exactly the pre-refactor likelihood.
    if exclude_mask is not None:
        ex = exclude_mask.to(torch.bool)
    else:
        ex = torch.zeros_like(obs.mask)
    if gamma_loss_threshold is not None:
        ex = ex | ((gamma.amax(dim=0) < gamma_loss_threshold) & obs.mask)
    observed = obs.mask & ~ex
    censored_mask = (~obs.mask) & ~ex

    return PanelCellTerms(
        intensity=int_lp,
        mz=mz_lp,
        censored=cens_lp,
        jacobian=log_jac,
        gamma=gamma,
        component_mz=component_lp,
        n_count=n_count_stack,
        i_pred=i_pred_stack,
        observed=observed,
        censored_mask=censored_mask,
    )


def panel_cell_log_prob(
    progenitors: Sequence[Progenitor],
    obs: CounterObservation,
    calibration: GlobalCalibration,
    *,
    background_intensity: torch.Tensor | float = 0.0,
    floor_ions: float = _DEFAULT_FLOOR_IONS,
    exclude_mask: torch.Tensor | None = None,
    gamma_loss_threshold: float | None = None,
) -> PanelCellTerms:
    """Per-cell decomposition of the panel likelihood (see `PanelCellTerms`).

    The same physics as `panel_log_prob` without the final scan/channel reduction
    — so callers can extract per-ion scores (sum a term over scans within a
    channel), select/exclude ions by their fit (L4), or read the soft `gamma`
    attribution that links each observed cell to its progenitor(s) (L8)."""
    return _panel_terms(
        progenitors,
        obs,
        calibration,
        background_intensity=background_intensity,
        floor_ions=floor_ions,
        exclude_mask=exclude_mask,
        gamma_loss_threshold=gamma_loss_threshold,
    )


def panel_log_prob(
    progenitors: Sequence[Progenitor],
    obs: CounterObservation,
    calibration: GlobalCalibration,
    *,
    background_intensity: torch.Tensor | float = 0.0,
    floor_ions: float = _DEFAULT_FLOOR_IONS,
    exclude_mask: torch.Tensor | None = None,
    gamma_loss_threshold: float | None = None,
) -> torch.Tensor:
    """Total additive joint log-likelihood (0-d tensor) for a panel.

    `progenitors` share `obs`'s channel grid. The latent per-channel rate is
    the additive sum of progenitor counts (+ background as a count rate); the
    observed-channel intensity is scored count-natively (detection-conditioned
    Poisson), so there is no per-charge total / `(1-p)` variance to assemble.
    vmap-safe: only broadcasting ops + `torch.where` over a static progenitor
    list. With no exclusion set this is byte-identical to the pre-decomposition
    likelihood; `exclude_mask` / `gamma_loss_threshold` drop the named cells from
    the loss (still inferred) — see `PanelCellTerms`."""
    t = _panel_terms(
        progenitors,
        obs,
        calibration,
        background_intensity=background_intensity,
        floor_ions=floor_ions,
        exclude_mask=exclude_mask,
        gamma_loss_threshold=gamma_loss_threshold,
    )
    zeros = torch.zeros_like(t.intensity)
    int_term = torch.where(t.observed, t.intensity, zeros).sum()
    mz_term = torch.where(t.observed, t.mz, zeros).sum()
    cens_term = torch.where(t.censored_mask, t.censored, zeros).sum()
    jac_term = torch.where(t.observed, t.jacobian, zeros).sum()
    return int_term + mz_term + cens_term + jac_term


class Panel(Distribution):
    """Additive joint score over a fixed candidate set + optional background.

    The progenitors are a `ModuleList`; the shared `calibration` is held **by
    reference** (a 1-tuple + property, like `Progenitor`), NOT a registered
    submodule. This is load-bearing: the per-peptide fit (`fit_panel`'s DE /
    `functional_call`) optimizes only the panel's *registered* parameters, so a
    by-reference calibration is frozen during the fit — as intended (its
    per-acquisition params are fit once by `StagedCalibration`, never re-fit per
    target). Registering it instead leaks `calibration.*` into `parameters()`,
    and the optimizer silently mutates the shared object across targets.
    """

    def __init__(
        self,
        progenitors: Sequence[Progenitor],
        calibration: GlobalCalibration,
        *,
        background: bool = False,
        floor_ions: float = _DEFAULT_FLOOR_IONS,
        background_init: float = 1e-3,
        exclude_mask: torch.Tensor | None = None,
        gamma_loss_threshold: float | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if len(progenitors) == 0:
            raise ValueError("a Panel needs at least one progenitor")
        self.progenitors = nn.ModuleList(progenitors)
        self._calibration_ref = (calibration,)  # by reference — frozen during fits
        self.floor_ions = float(floor_ions)
        # Loss exclusion (still inferred): a per-(scan, channel) `exclude_mask`
        # (L4 outlier/blend hold-out, matched to the obs this panel scores) and/or
        # a `gamma_loss_threshold` (D9 benchmark — drop observed cells no single
        # progenitor owns at γ ≥ threshold). Both default off → unchanged loss.
        self.exclude_mask = exclude_mask
        self.gamma_loss_threshold = (
            None if gamma_loss_threshold is None else float(gamma_loss_threshold)
        )
        if background:
            self.log_background = nn.Parameter(
                torch.tensor(float(background_init), dtype=dtype).log()
            )
        else:
            self.log_background = None

    @property
    def calibration(self) -> GlobalCalibration:
        """The shared per-acquisition calibration (by reference, not a submodule
        — see the class docstring; frozen during per-peptide fits)."""
        return self._calibration_ref[0]

    # -- candidate-set mutation (fit-plus-discover) ---------------------

    def add_progenitor(self, progenitor: Progenitor) -> int:
        """Append a discovered progenitor to the candidate set; return its index.

        The candidate set is mutable so the fit-plus-discover loop can introduce
        interferer progenitors from residual evidence. The caller MUST build a
        FRESH optimizer / VI guide after any mutation — the parameter set has
        changed, so an optimizer bound to the old `parameters()` is stale (the
        loop already constructs a new optimizer per refit, so this holds by
        construction). The progenitor must share this panel's `calibration` (the
        per-acquisition params are fit once and held by reference)."""
        if progenitor.calibration is not self.calibration:
            raise ValueError(
                "a discovered progenitor must share the panel's calibration"
            )
        self.progenitors.append(progenitor)
        return len(self.progenitors) - 1

    def remove_progenitor(self, index: int) -> None:
        """Drop a candidate (e.g. one that refit below the detection floor).
        Later progenitors shift down by one, so the loop only removes at the end
        and keeps the locked target at index 0 (so `progenitor_{i}.` parameter
        names stay stable while candidates are appended mid-loop)."""
        if not 0 <= index < len(self.progenitors):
            raise IndexError(f"progenitor index {index} out of range")
        del self.progenitors[index]

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
            exclude_mask=self.exclude_mask,
            gamma_loss_threshold=self.gamma_loss_threshold,
        )

    def cell_terms(self, obs: CounterObservation) -> "PanelCellTerms":
        """The per-cell likelihood decomposition for this panel (see
        `PanelCellTerms` / `panel_cell_log_prob`) — the substrate for per-ion
        scores, ion selection, and the soft attribution map."""
        return panel_cell_log_prob(
            list(self.progenitors),
            obs,
            self.calibration,
            background_intensity=self.background_intensity(),
            floor_ions=self.floor_ions,
            exclude_mask=self.exclude_mask,
            gamma_loss_threshold=self.gamma_loss_threshold,
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
