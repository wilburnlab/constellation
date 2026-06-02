"""MS2 spectral-similarity suite — the scoring comparison for Part I.

Free functions over *aligned* intensity / count vectors (the K fragment
channels on the last axis). The L2-geometry similarities (cosine /
normalized dot / Pearson / spectral angle) and the information-geometry
ones (KL divergence, spectral-entropy similarity) are reused from
``core.stats.losses``; this module adds the **multinomial deviance** that
ties them together and a single ``compare_spectra`` dispatcher the
experiment drivers call.

The thesis these support: an MS2 spectrum is a ``Multinomial(N, p)`` draw
over fragment channels, so the natural divergence is ``2·N·KL(p̂ ‖ p_ref)``
— the likelihood-ratio (G²) statistic, asymptotically ``χ²_{K−1}`` and
therefore *flat in N*. The L2 measures instead scale their apparent
disagreement with N (shot noise), over-penalizing intense ions. See
:func:`multinomial_deviance` for the exact identity
``2·N·KL = −2·(logL(x | p_ref) − logL(x | p̂))``.
"""

from __future__ import annotations

from typing import Literal

import torch

from constellation.core.stats.losses import (
    cosine_similarity,
    kld,
    normalized_dot,
    pearson_correlation,
    spectral_angle,
    spectral_entropy_similarity,
)

__all__ = ["multinomial_deviance", "compare_spectra"]

_EPS = 1e-12

Method = Literal[
    "cosine",
    "dot",
    "pearson",
    "spectral_angle",
    "spectral_entropy",
    "kld",
    "multinomial_deviance",
]


def multinomial_deviance(
    obs_counts: torch.Tensor,
    p_ref: torch.Tensor,
    *,
    pseudocount: float = 1e-3,
) -> torch.Tensor:
    """Multinomial deviance / G² likelihood-ratio statistic between observed
    fragment counts and a reference propensity vector, over the last axis.

    ``G² = 2·Σ_k x_k·log(p̂_k / p_ref,k) = 2·N·KL(p̂ ‖ p_ref)`` with
    ``N = Σ_k x_k`` and ``p̂ = x / N``. This equals
    ``−2·(logL(x | p_ref) − logL(x | p̂))`` for the multinomial
    log-likelihood (the combinatorial constant ``log N!/Πx_k!`` cancels),
    so it is the multinomial-scale comparator behind the whole Part-I
    argument. Under ``x ~ Multinomial(N, p_ref)`` it is asymptotically
    ``χ²_{K−1}`` — mean ``≈ K−1`` independent of ``N``.

    ``pseudocount`` smooths the **reference** propensities so ``log p_ref``
    is finite even at empty channels; observed-zero channels need no
    smoothing (their ``x_k·log(·)`` term is exactly ``0`` by the
    ``0·log 0 = 0`` convention, so this stays NaN-free at ``pseudocount=0``
    — unlike a naive ``kld`` call). Set ``0.0`` for the exact likelihood
    identity with strictly-positive references. Inputs are ``(..., K)``;
    returns ``obs_counts.shape[:-1]``. Lower is more similar (it is a
    divergence, not a similarity)."""
    x = obs_counts.to(torch.float64)
    p = p_ref.to(torch.float64)
    n = x.sum(dim=-1, keepdim=True)
    p_hat = x / n.clamp(min=_EPS)
    p_ref_n = p / p.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    if pseudocount > 0.0:
        k = p_ref_n.shape[-1]
        p_ref_n = (p_ref_n + pseudocount) / (1.0 + pseudocount * k)
    # ``x_k = 0`` ⇒ term 0 regardless of the log (0·finite); ``p_hat`` and
    # ``p_ref_n`` are clamped only to keep the logs finite. This is exactly
    # ``2·N·KL(p̂ ‖ p_ref)`` with the standard 0-count convention.
    term = x * (p_hat.clamp(min=_EPS).log() - p_ref_n.clamp(min=_EPS).log())
    return 2.0 * term.sum(dim=-1)


def compare_spectra(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    method: Method,
    as_counts: bool = False,
    pseudocount: float = 1e-3,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Score an aligned ``query`` spectrum against ``reference`` over the
    last axis — the single dispatch point for the experiment drivers.

    Inputs are aligned intensity / count vectors ``(K,)`` or a batch
    ``(B, K)`` (channels on the last axis); the result is
    ``query.shape[:-1]``.

    Similarities (**higher = more similar**): ``cosine``, ``dot`` (=
    normalized dot product, identical to ``cosine``), ``pearson``,
    ``spectral_angle`` (all in ``[-1, 1]`` / ``[0, 1]``),
    ``spectral_entropy`` (``[0, 1]``).

    Divergences (**lower = more similar**, ``≥ 0``): ``kld`` =
    ``KL(query ‖ reference)`` after L1 normalization; ``multinomial_deviance``
    = ``2·N·KL`` which **requires integer-like counts** — pass
    ``as_counts=True`` with ``query`` the counts and ``reference`` the
    propensities."""
    if method == "cosine":
        return cosine_similarity(query, reference, eps=eps)
    if method == "dot":
        return normalized_dot(query, reference, eps=eps)
    if method == "pearson":
        return pearson_correlation(query, reference, eps=eps)
    if method == "spectral_angle":
        # ``spectral_angle`` uses the legacy ``batch_dim`` flatten convention;
        # adapt so a ``(B, K)`` batch reduces per-row like the others.
        return spectral_angle(query, reference, eps=eps, batch_dim=query.ndim >= 2)
    if method == "spectral_entropy":
        return spectral_entropy_similarity(
            query, reference, pseudocount=pseudocount, eps=eps
        )
    if method == "kld":
        return kld(reference, query, pseudocount=pseudocount, reduce="none")
    if method == "multinomial_deviance":
        if not as_counts:
            raise ValueError(
                "method='multinomial_deviance' needs integer-like counts: pass "
                "as_counts=True with query as counts and reference as propensities."
            )
        return multinomial_deviance(query, reference, pseudocount=pseudocount)
    raise ValueError(f"unknown method {method!r}")
