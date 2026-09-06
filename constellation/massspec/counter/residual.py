"""Panel residual — the `(I_obs − Σ I_pred)` substrate for progenitor discovery.

The fit-plus-discover loop (``discover.py``) inspects the residual at a fitted
panel's ion nodes: positive unexplained signal with coherent RT/isotope
structure is a discovery candidate — an interferer the additive background
can't absorb and the target shouldn't claim. These are pure, no-grad
diagnostics (they feed detection logic, not the optimizer).

Background is excluded from the predicted sum on purpose: it is diffuse,
unattributable density, so subtracting it would hide exactly the coherent peaks
discovery is meant to find (and the background can over-absorb a real interferer
during the fit — keeping it out of the residual preserves that evidence).
"""

from __future__ import annotations

import torch

from .model import CounterObservation
from .panel import Panel

__all__ = ["panel_predicted_intensity", "panel_residual"]


@torch.no_grad()
def panel_predicted_intensity(panel: Panel, obs: CounterObservation) -> torch.Tensor:
    """`Σ_q I_pred_q` over the panel's progenitors `(S, C)` — the additive
    predicted intensity, **background excluded**."""
    total = torch.zeros_like(obs.intensity)
    for q in panel.progenitors:
        i_pred_q, _ = q.predict(obs)
        total = total + i_pred_q
    return total


@torch.no_grad()
def panel_residual(panel: Panel, obs: CounterObservation) -> torch.Tensor:
    """`r = I_obs − Σ I_pred` on observed channels, 0 elsewhere `(S, C)`.

    Positive `r` is unexplained signal (a discovery candidate); negative is
    over-explained (corrected by refit, not discovery)."""
    r = obs.intensity - panel_predicted_intensity(panel, obs)
    return torch.where(obs.mask, r, torch.zeros_like(r))
