"""Fragmentation noise model — Dirichlet-multinomial with an unknown gain.

Observed fragment **intensity** vectors ``v`` (one per MS2 scan, aligned to a
peptide's fixed channel basis) are modeled as ``v = g · counts`` with the latent
fragment counts ``counts ~ DirichletMultinomial(N, p, α₀)`` and
``N = Σv/g`` (the ion count; ``g`` is the intensity-per-ion gain). Fitting one
peptide's stack of spectra jointly recovers:

  * ``p``  — the true fragmentation pattern (latent categorical);
  * ``α₀`` — the overdispersion / "effective independent-packet" count (the
    Dirichlet concentration; ``α₀ → ∞`` is pure multinomial); and
  * ``g``  — the intensity→ion gain, identified by how the per-proportion
    variance shrinks with intensity (``Var[p̂] ∝ g/I`` in the shot-noise regime).

``g`` is a free parameter here; the statistically cleaner route is to constrain
it mutually with the MS1 gain ``α(z)·IIT`` (Counter's joint inference). This v1
lets ``α₀`` absorb everything (overdispersion + residual interference + run drift)
— so the fitted ``α₀`` is a *floor* on the true fragmentation α₀ until outliers
are split out. Fit via ``Parametric.fit`` with either ``LBFGSOptimizer`` or
``DifferentialEvolution``.
"""

from __future__ import annotations

import torch
from torch import nn

from constellation.core.stats.distributions import dirichlet_multinomial_log_prob
from constellation.core.stats.parametric import Distribution

__all__ = ["FragmentationNoiseModel"]


class FragmentationNoiseModel(Distribution):
    """Per-peptide Dirichlet-multinomial fragment-intensity model. Params:
    ``log_g`` (scalar) and ``log_concentration`` (K,) ``= log(α₀·p)``. The data
    passed to ``log_prob`` / ``fit`` are intensity vectors ``v`` of shape
    ``(B, K)`` (channels on the last axis)."""

    def __init__(
        self,
        K: int,
        *,
        log_g_init: float = 7.0,
        concentration_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.log_g = nn.Parameter(torch.tensor(float(log_g_init), dtype=torch.float64))
        if concentration_init is None:
            init = torch.zeros(K, dtype=torch.float64)
        else:
            c = torch.as_tensor(concentration_init, dtype=torch.float64)
            init = torch.log(c.clamp(min=1e-9))
        self.log_concentration = nn.Parameter(init)

    @property
    def K(self) -> int:
        return int(self.log_concentration.shape[0])

    @property
    def g(self) -> torch.Tensor:
        return self.log_g.exp()

    @property
    def concentration(self) -> torch.Tensor:
        return self.log_concentration.exp()

    @property
    def alpha0(self) -> torch.Tensor:
        return self.concentration.sum()

    @property
    def probs(self) -> torch.Tensor:
        c = self.concentration
        return c / c.sum()

    def log_prob(self, v: torch.Tensor) -> torch.Tensor:
        """Per-spectrum log-likelihood of intensity vectors ``v`` (B, K).
        Counts are ``v/g``; the gain enters the count total ``N = Σv/g``."""
        return dirichlet_multinomial_log_prob(v / self.g, self.concentration)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FragmentationNoiseModel has no closed-form CDF.")

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "g": self.g.detach(),
            "alpha0": self.alpha0.detach(),
            "probs": self.probs.detach(),
        }
