"""Ion-injection-time (IIT) ↔ accumulated-count conversion.

The one place the IIT physics lives. Reported Thermo intensity is a *rate*
(per the ms convention used throughout Counter): the number of ions that
*drove* a scan's signal — and hence its shot noise and m/z precision — is

    N_accumulated = I · τ / α(z)          [Eq. 11]

with units (DIMENSIONAL ANALYSIS, ms convention):

    [I]   = a.u.·ms⁻¹      (intensity is a rate)
    [τ]   = ms             (injection time, Thermo-native, used as-is)
    [α]   = a.u.·ion⁻¹     (gain; "a.u." = accumulated signal = I·τ)
    [N]   = a.u.·ms⁻¹ · ms / (a.u.·ion⁻¹) = ions  ✓

This is a *derived* quantity used inside the forward variance / m/z
precision (`channels.py`) — NOT a pre-transform of the observed data.
Counter scores the observables (intensity, m/z) directly; `τ` enters the
forward model here, never by destroying the raw intensity. (`τ` cancels in
the *mean* intensity, Eq. 14 — it only shows up in the count/variance.)
"""

from __future__ import annotations

import torch

__all__ = ["accumulated_count"]


def accumulated_count(
    intensity: torch.Tensor,
    iit: torch.Tensor,
    gain: torch.Tensor,
    *,
    floor: float = 1e-9,
) -> torch.Tensor:
    """Per-scan accumulated ion count `N = I·τ/α` [ions].

    `intensity` [a.u./ms], `iit` (`τ`) [ms], `gain` (`α(z)`) [a.u./ion] —
    broadcast together. `floor` clamps the result away from zero so it can
    safely feed `N^{−α_mz}` precision and `log` terms. Inverse of the
    `intensity = α·N/τ` forward relation the simulator uses.
    """
    return (intensity * iit / gain).clamp(min=floor)
