"""Attribution — how shared peak intensity is split among co-eluting
progenitors.

Centroid data destroys additivity (a centroid is one intensity-weighted
blended `(m/z, intensity)` pair), so the principled treatment is **soft,
intensity-weighted attribution**: each observed peak is fractionally
attributed to the candidates by their *predicted* intensities,

    γ_q(s,c) = I_pred_q(s,c) / Σ_q' I_pred_q'(s,c).

This `γ` is the EM E-step responsibility — provably the correct limit when
candidate kernels coincide (the unresolved case) — and matches MS-Deconv's
intensity-split / MaxQuant's PIF share. The m/z likelihood of an observed
centroid is then the **mixture marginal** `logΣ_q γ_q · t(ε; center_q,
scale_q)` (a `logsumexp`), which for a single candidate reduces to the bare
Student-t. There is no hard per-peak cutoff; the only hard decision (which
candidates are *in* the model) lives at candidate-inclusion, deferred.

Multi-point (profile) generalization: when an observation carries multiple
m/z sample points instead of one centroid, the same `γ` weights a summed-
profile kernel — designed for, not built here (Orbitrap profile needs the
frequency-grid re-alignment that is out of scope; PR-1 is centroid, M=1).
"""

from __future__ import annotations

import torch

__all__ = ["responsibilities", "mz_mixture_log_prob"]


def responsibilities(
    intensity_pred_stack: torch.Tensor, *, eps: float = 1e-30
) -> torch.Tensor:
    """Intensity responsibilities `γ` `(Q, S, C)` from per-progenitor predicted
    intensities `(Q, S, C)`, normalized over the progenitor axis. Background is
    intentionally excluded from the denominator — `γ` attributes the *modeled*
    species' share of each peak."""
    denom = intensity_pred_stack.sum(dim=0).clamp(min=eps)
    return intensity_pred_stack / denom.unsqueeze(0)


def mz_mixture_log_prob(
    gamma: torch.Tensor, component_log_probs: torch.Tensor, *, eps: float = 1e-30
) -> torch.Tensor:
    """Mixture-marginal m/z log-likelihood `(S, C)` =
    `logsumexp_q( log γ_q + log t_q )`, the responsibility-weighted soft
    attribution. `gamma` and `component_log_probs` are both `(Q, S, C)`."""
    return torch.logsumexp(torch.log(gamma.clamp(min=eps)) + component_log_probs, dim=0)
