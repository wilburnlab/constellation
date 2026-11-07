"""Loss functions for spectral / distributional comparisons.

Free functions only — Cartographer's `nn.Module` wrappers (`KLD`,
`Spectral_Entropy_Loss`) used `snake_case` class names against the
PascalCase invariant, and functions compose more cleanly into
`Parametric.fit`'s `loss_fn` parameter.

Masking convention (from Cartographer's sentinel): a negative value — the
`-1` "unobserved" sentinel — or a NaN in either vector removes that
channel from the comparison, the same way in every function here
(:func:`_valid`). A `mask` keyword, where offered, is combined with it.

Every similarity is scale-free: multiplying either vector by a positive
constant leaves it unchanged, at any magnitude. Normalization divides by
the exact norm or total; a vector with nothing in it (all zero after
masking) is not rescaled, and a similarity with it is 0.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

#: Relative size under which a norm is taken as zero (a centred constant
#: vector leaves rounding residue about this size, not exact zeros).
_RTOL = 64.0


# ──────────────────────────────────────────────────────────────────────
# Normalization helpers
# ──────────────────────────────────────────────────────────────────────


def _valid(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The channels both vectors observe: non-negative and not NaN in each."""
    return (pred >= 0) & (target >= 0)


def _masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``x`` with masked channels set to 0 (``where``, so a NaN there does
    not leak through as it would through ``x * mask``)."""
    return torch.where(mask, x, torch.zeros_like(x))


def _safe_divide(x: torch.Tensor, denom: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """``x / denom`` where ``denom`` is not ``zero``-small; 0 where it is."""
    ok = denom > zero
    return torch.where(ok, x / torch.where(ok, denom, torch.ones_like(denom)), torch.zeros_like(x))


def _unit_lastaxis(x: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    """``x`` divided by its L2 norm over the last axis. A norm below
    ``_RTOL`` machine epsilons of ``reference``'s norm (default ``x``'s
    own, so only an exactly zero vector) counts as zero, and the vector
    is returned as zeros."""
    norm = x.square().sum(dim=-1, keepdim=True).sqrt()
    if reference is None:
        zero = torch.zeros_like(norm)
    else:
        ref = reference.square().sum(dim=-1, keepdim=True).sqrt()
        zero = ref * (_RTOL * torch.finfo(x.dtype).eps)
    return _safe_divide(x, norm, zero)


def l1_normalize(
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    pseudocount: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """L1-normalize along the last axis. With `pseudocount > 0`, adds
    a flat smoothing prior before normalization (Laplace-style), in the
    input's own units. ``eps`` only guards the division of an all-zero
    row (which stays zero)."""
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
    """L2-normalize by the exact norm: over the whole tensor, or with
    `batch_dim=True` per row of axis 0 over the flattened trailing dims
    (Cartographer's per-spectrum normalization for batched
    fragment-intensity vectors). An all-zero vector stays zero.

    ``eps`` is kept for the signature: it used to floor the *squared*
    norm, which made every comparison of small-magnitude vectors
    (intensities around 1e-4 and below) scale-dependent."""
    x_masked = x if mask is None else torch.where(mask.bool(), x, torch.zeros_like(x))
    if batch_dim:
        new_shape = [x_masked.shape[0]] + [1] * (x_masked.ndim - 1)
        norm = x_masked.square().flatten(1).sum(1).sqrt().reshape(new_shape)
    else:
        norm = x_masked.square().sum().sqrt()
    return _safe_divide(x_masked, norm, torch.zeros_like(norm))


# ──────────────────────────────────────────────────────────────────────
# Divergences and similarities
# ──────────────────────────────────────────────────────────────────────


def _proportions(
    x: torch.Tensor, mask: torch.Tensor, pseudocount: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` over the masked channels as proportions, smoothed by
    ``pseudocount`` in probability units — ``(p + c) / (1 + c·K)`` with
    ``K`` the channels kept — and whether the row had any mass at all."""
    x0 = _masked(x.clamp(min=0.0), mask)
    total = x0.sum(dim=-1, keepdim=True)
    has_mass = total > 0
    p = _safe_divide(x0, total, torch.zeros_like(total))
    if pseudocount > 0.0:
        k = mask.to(x.dtype).sum(dim=-1, keepdim=True)
        p = _masked((p + pseudocount) / (1.0 + pseudocount * k), mask)
    return p, has_mass.squeeze(-1)


def kld(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pseudocount: float = 1e-3,
    pseudocount_units: Literal["probability", "input"] = "probability",
    mask: torch.Tensor | None = None,
    reduce: str = "mean",
    from_logits: bool = False,
) -> torch.Tensor:
    """Pseudocounted KL divergence `KL(target || pred)` between intensity vectors.

    Last axis is the distribution. Channels negative or NaN in either
    vector (the `-1` sentinel) are dropped, together with any `mask`
    given. With `from_logits=True`, `pred` is treated as raw
    log-intensities and exponentiated before normalization.

    ``pseudocount_units``:

    * ``"probability"`` (default): each vector is L1-normalized first and
      then smoothed, ``(p + c) / (1 + c·K)`` — scale-free, as
      ``multinomial_deviance`` smooths. A row where either vector has no
      mass has no defined divergence and is NaN.
    * ``"input"``: ``c`` is added to the raw intensities before
      normalizing, Cartographer's form, which depends on the intensities'
      scale; :func:`spectral_entropy_loss` keeps it for training.

    `reduce ∈ {'mean', 'sum', 'none'}` — applies to the leading
    (non-distribution) axes of the per-row KL.
    """
    if from_logits:
        pred = pred.exp()
    valid = _valid(pred, target)
    if mask is not None:
        valid = valid & mask.bool()
    if pseudocount_units == "probability":
        pred_norm, pred_mass = _proportions(pred, valid, pseudocount)
        target_norm, target_mass = _proportions(target, valid, pseudocount)
        defined = pred_mass & target_mass
    elif pseudocount_units == "input":
        pred_masked = _masked(pred.clamp(min=0.0) + pseudocount, valid)
        target_masked = _masked(target.clamp(min=0.0) + pseudocount, valid)
        pred_norm = pred_masked / pred_masked.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        target_norm = target_masked / target_masked.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        defined = None
    else:
        raise ValueError(f"pseudocount_units must be 'probability' or 'input'; got {pseudocount_units!r}")
    ratio = (target_norm.clamp(min=1e-30) / pred_norm.clamp(min=1e-30)).log()
    kl_terms = torch.where(valid & (target_norm > 0), target_norm * ratio, torch.zeros_like(ratio))
    per_row = kl_terms.sum(dim=-1)
    if defined is not None:
        per_row = torch.where(defined, per_row, torch.full_like(per_row, float("nan")))
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
    (parallel) intensity vectors, `0.0` for orthogonal ones or when
    either has nothing in it.

    Channels negative or NaN in either vector are dropped. A single
    spectrum is a 1-D vector; a batch needs `batch_dim=True` (axis 0 is
    the batch, trailing dims flattened per spectrum) and is refused
    without it, where it used to be scored as one long vector. For
    channels on the last axis of any batch shape, use
    `1 - 2·arccos(cosine_similarity)/π`. The dot product is clamped to
    `[-1, 1]` before `arccos` (stability fix vs Cartographer, which let
    it saturate near 1)."""
    if pred.ndim > 1 and not batch_dim:
        raise ValueError(
            f"spectral_angle got a {tuple(pred.shape)} tensor without batch_dim=True; "
            "pass batch_dim=True to score each row, or use cosine_similarity over the last axis"
        )
    valid = _valid(pred, target)
    if batch_dim:
        p = _masked(pred, valid).flatten(1)
        t = _masked(target, valid).flatten(1)
    else:
        p = _masked(pred, valid).reshape(-1)
        t = _masked(target, valid).reshape(-1)
    sum_product = (_unit_lastaxis(p) * _unit_lastaxis(t)).sum(dim=-1)
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

    A training loss with Cartographer's semantics: its pseudocount is in
    the input's own units (``kld(..., pseudocount_units="input")``), so
    unlike the similarities it depends on the intensities' scale.
    """
    signal_frac = torch.sigmoid(signal_logit)
    signal = target * signal_frac
    interference = target - signal

    signal_kl = kld(
        pred, signal, pseudocount=pseudocount, pseudocount_units="input", mask=None, reduce="none"
    )

    interference_adj = interference + pseudocount
    interference_norm = interference_adj / interference_adj.sum(
        dim=-1, keepdim=True
    ).clamp(min=eps)
    interference_log = interference_norm.clamp(min=eps).log()
    interference_entropy = -(interference_norm * interference_log).sum(dim=-1)

    obs_frac = (signal.sum(dim=-1) + eps) / (target.sum(dim=-1) + eps)
    return obs_frac * signal_kl + (1.0 - obs_frac) * interference_entropy


# ──────────────────────────────────────────────────────────────────────
# Cosine-family similarities (raw, un-angularized)
# ──────────────────────────────────────────────────────────────────────
#
# Reduction is over the LAST axis: inputs are ``(..., K)`` with the K
# spectral channels on the trailing axis, and the result has shape
# ``input.shape[:-1]`` (a 0-d scalar for a single ``(K,)`` spectrum,
# ``(B,)`` for a ``(B, K)`` batch). For a 1-D pair, and row by row for
# ``spectral_angle(..., batch_dim=True)`` on a 2-D batch, the identity
# ``spectral_angle = 1 - 2·arccos(cosine_similarity)/π`` holds exactly.
# Channels negative or NaN in either vector are dropped (``_valid``),
# the same rule as ``spectral_angle`` and ``kld``.


def cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Cosine similarity over the last axis. ``1.0`` for parallel vectors;
    non-negative intensity vectors give ``[0, 1]``; ``0`` when either has
    nothing in it.

    The raw dot product of the L2-normalized vectors — the quantity
    ``spectral_angle`` feeds to ``arccos``. Scale-free at any magnitude.
    Returns ``input.shape[:-1]``."""
    valid = _valid(pred, target)
    pn = _unit_lastaxis(_masked(pred, valid))
    tn = _unit_lastaxis(_masked(target, valid))
    return (pn * tn).sum(dim=-1).clamp(min=-1.0, max=1.0)


def normalized_dot(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """The "normalized dot product" library-search score — identical to
    :func:`cosine_similarity` (dot product of L2-normalized intensity
    vectors). Exposed under its spectral-library name for the scoring
    comparison; its L2 geometry over-weights intense-ion shot noise (the
    Part-I point)."""
    return cosine_similarity(pred, target, eps=eps)


def pearson_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Pearson correlation over the last axis = cosine of the mean-centered
    vectors. ``+1`` for ``pred = a·target + b`` (a > 0), ``-1`` for
    anti-correlation, ``0`` when either is constant over the channels
    kept (its centred residue is rounding, taken as zero). The mean is
    over the channels both vectors observe. Scale-free at any magnitude.
    Returns ``input.shape[:-1]``."""
    valid = _valid(pred, target)
    w = valid.to(pred.dtype)
    n = w.sum(dim=-1, keepdim=True)
    pm, tm = _masked(pred, valid), _masked(target, valid)
    pred_mean = _safe_divide(pm.sum(dim=-1, keepdim=True), n, torch.zeros_like(n))
    target_mean = _safe_divide(tm.sum(dim=-1, keepdim=True), n, torch.zeros_like(n))
    pc = _masked(pred - pred_mean, valid)
    tc = _masked(target - target_mean, valid)
    pn = _unit_lastaxis(pc, reference=pm)
    tn = _unit_lastaxis(tc, reference=tm)
    return (pn * tn).sum(dim=-1).clamp(min=-1.0, max=1.0)


def spectral_entropy_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pseudocount: float = 1e-3,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Spectral-entropy similarity in ``[0, 1]`` (Li et al., Nat. Methods
    2021) over the last axis. ``1.0`` for identical spectra, ``0.0`` for
    disjoint support or when either has nothing in it.

    Equals ``1 - JSD(A, B)/ln 2`` where ``A``, ``B`` are the L1-normalized
    intensity vectors and ``JSD`` is the Jensen–Shannon divergence; written
    in Li's entropy form as
    ``1 - (2·S_AB - S_A - S_B)/ln 4`` with ``S`` the Shannon entropy and
    ``S_AB`` the entropy of ``(A + B)/2``. (The optional low-entropy
    intensity reweighting from the paper is omitted — this is the core
    similarity.) Each vector is normalized first and then smoothed by
    ``pseudocount`` in probability units, so the score is scale-free.
    Channels negative or NaN in either are dropped. Returns
    ``input.shape[:-1]``."""
    valid = _valid(pred, target)
    a, a_mass = _proportions(pred, valid, pseudocount)
    b, b_mass = _proportions(target, valid, pseudocount)
    m = 0.5 * (a + b)

    def _entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * p.clamp(min=1e-300).log()).sum(dim=-1)

    sim = 1.0 - (2.0 * _entropy(m) - _entropy(a) - _entropy(b)) / math.log(4.0)
    return torch.where(a_mass & b_mass, sim, torch.zeros_like(sim))
