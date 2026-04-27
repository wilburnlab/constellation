"""Loss functions for spectral / distributional comparisons.

Free functions only — Cartographer's `nn.Module` wrappers (`KLD`,
`Spectral_Entropy_Loss`) used `snake_case` class names against the
PascalCase invariant, and functions compose more cleanly into
`Parametric.fit`'s `loss_fn` parameter.

Sentinel convention (from Cartographer): a `-1` in an intensity tensor
flags an unobserved / masked position. The `mask` keyword overrides
sentinel-based masking; pass an explicit boolean tensor when the
sentinel is unsuitable.
"""

from __future__ import annotations

import math

import torch


# ──────────────────────────────────────────────────────────────────────
# Normalization helpers
# ──────────────────────────────────────────────────────────────────────


def l1_normalize(
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    pseudocount: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """L1-normalize along the last axis. With `pseudocount > 0`, adds
    a flat smoothing prior before normalization (Laplace-style)."""
    if mask is None:
        x_masked = x + pseudocount
    else:
        x_masked = (x + pseudocount) * mask
    denom = x_masked.sum(dim=-1, keepdim=True).clamp(min=eps)
    return x_masked / denom


def l2_normalize(
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-7,
    batch_dim: bool = False,
) -> torch.Tensor:
    """L2-normalize. With `batch_dim=True`, treats axis 0 as a batch
    axis and flattens the trailing dims for the norm computation —
    matches Cartographer's per-spectrum normalization for batched
    fragment-intensity vectors."""
    x_masked = x if mask is None else x * mask
    if batch_dim:
        n_dims = x_masked.ndim
        new_shape = [x_masked.shape[0]] + [1] * (n_dims - 1)
        sum_sq = x_masked.square().flatten(1).sum(1).reshape(new_shape)
    else:
        sum_sq = x_masked.square().sum()
    denom = sum_sq.clamp(min=eps).sqrt()
    return x_masked / denom


# ──────────────────────────────────────────────────────────────────────
# Divergences and similarities
# ──────────────────────────────────────────────────────────────────────


def kld(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pseudocount: float = 1e-3,
    mask: torch.Tensor | None = None,
    reduce: str = "mean",
    from_logits: bool = False,
) -> torch.Tensor:
    """Pseudocounted KL divergence `KL(target || pred)` between intensity vectors.

    Last axis is the distribution. Positions where `pred < 0` or
    `target < 0` are treated as masked (sentinel `-1`) unless `mask`
    is given explicitly. With `from_logits=True`, `pred` is treated as
    raw log-intensities and exponentiated before normalization.

    `reduce ∈ {'mean', 'sum', 'none'}` — applies to the leading
    (non-distribution) axes of the per-row KL.
    """
    if from_logits:
        pred = pred.exp()
    if mask is None:
        mask = (pred >= 0) & (target >= 0)
    pred_masked = pred.clamp(min=0.0) * mask + pseudocount * mask
    target_masked = target.clamp(min=0.0) * mask + pseudocount * mask
    pred_norm = pred_masked / pred_masked.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    target_norm = (
        target_masked / target_masked.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    )
    ratio = (target_norm / pred_norm.clamp(min=1e-30)).log()
    kl_terms = torch.where(mask, target_norm * ratio, torch.zeros_like(ratio))
    per_row = kl_terms.sum(dim=-1)
    if reduce == "mean":
        return per_row.mean()
    if reduce == "sum":
        return per_row.sum()
    if reduce == "none":
        return per_row
    raise ValueError(f"reduce must be 'mean'/'sum'/'none'; got {reduce!r}")


def spectral_angle(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-7,
    batch_dim: bool = False,
) -> torch.Tensor:
    """Spectral angle similarity in `[0, 1]`. `1.0` for identical
    (parallel) intensity vectors, `0.0` for orthogonal.

    `target == -1` positions are masked. The dot product is clamped
    to `[-1+eps, 1-eps]` before `arccos` (stability fix vs Cartographer,
    which let it saturate near 1)."""
    obs_mask = (target + 1.0) / (target + 1.0 + eps)
    pred_masked = pred * obs_mask
    target_masked = target * obs_mask
    pred_norm = l2_normalize(pred_masked, eps=eps, batch_dim=batch_dim)
    target_norm = l2_normalize(target_masked, eps=eps, batch_dim=batch_dim)
    product = pred_norm * target_norm
    if batch_dim:
        sum_product = product.flatten(1).sum(1)
    else:
        sum_product = product.sum()
    # Clamp at exactly ±1 — arccos handles the boundary cleanly; the
    # clamp only guards against floating-point overshoot. Pulling in by
    # `eps` (as Cartographer's TODO comment suggested) corrupts the
    # identical-vector case (similarity should be exactly 1.0).
    sum_product = sum_product.clamp(min=-1.0, max=1.0)
    return 1.0 - 2.0 * sum_product.arccos() / math.pi


def spectral_entropy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    signal_logit: torch.Tensor,
    *,
    pseudocount: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Entropy-decomposed spectral loss — completes Cartographer's
    truncated `Spectral_Entropy_Loss`.

    Splits `target` into a signal fraction (gated by `sigmoid(signal_logit)`)
    and an interference residue. Returns the observed-fraction-weighted
    sum of `kld(pred, signal)` and Shannon entropy of the interference.
    At `signal_logit → +∞` reduces to `kld(pred, target)`.
    """
    signal_frac = torch.sigmoid(signal_logit)
    signal = target * signal_frac
    interference = target - signal

    signal_kl = kld(
        pred, signal, pseudocount=pseudocount, mask=None, reduce="none"
    )

    interference_adj = interference + pseudocount
    interference_norm = interference_adj / interference_adj.sum(
        dim=-1, keepdim=True
    ).clamp(min=eps)
    interference_log = interference_norm.clamp(min=eps).log()
    interference_entropy = -(interference_norm * interference_log).sum(dim=-1)

    obs_frac = (signal.sum(dim=-1) + eps) / (target.sum(dim=-1) + eps)
    return obs_frac * signal_kl + (1.0 - obs_frac) * interference_entropy
