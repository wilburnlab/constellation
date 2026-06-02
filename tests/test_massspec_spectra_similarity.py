"""Tests for ``massspec.spectra.similarity`` — multinomial deviance and the
``compare_spectra`` dispatcher.

The load-bearing test is ``test_multinomial_deviance_equals_loglik_identity``:
it proves ``multinomial_deviance ≡ −2·Δ(multinomial log-likelihood) ≡ 2·N·KL``.
That identity is the analytical spine of the Part-I (MS2) story — it is why
"KL beats cosine" (exp01) and "fragmentation is multinomial" (exp02) are the
same claim.
"""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.stats.distributions import Multinomial
from constellation.core.stats.losses import cosine_similarity, kld
from constellation.massspec.spectra.similarity import (
    compare_spectra,
    multinomial_deviance,
)


# ──────────────────────────────────────────────────────────────────────
# multinomial_deviance — the spine identity
# ──────────────────────────────────────────────────────────────────────


def test_multinomial_deviance_equals_loglik_identity():
    """deviance(x, p_ref) == −2·(logL(x|p_ref) − logL(x|p̂)) via Multinomial."""
    x = torch.tensor([10.0, 5.0, 3.0, 2.0], dtype=torch.float64)  # N = 20
    p_ref = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)

    dev = multinomial_deviance(x, p_ref, pseudocount=0.0).item()

    p_hat = x / x.sum()
    m_ref = Multinomial(logits=torch.log(p_ref))
    m_hat = Multinomial(logits=torch.log(p_hat))
    expected = -2.0 * (m_ref.log_prob(x) - m_hat.log_prob(x)).item()

    assert math.isclose(dev, expected, rel_tol=1e-9)


def test_multinomial_deviance_equals_2N_kld():
    """deviance == 2·N·KL(p̂ ‖ p_ref) (no zero channels → kld is well-defined)."""
    x = torch.tensor([10.0, 5.0, 3.0, 2.0], dtype=torch.float64)
    p_ref = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    n = x.sum()
    expected = 2.0 * n * kld(p_ref, x, pseudocount=0.0, reduce="none")
    assert torch.allclose(
        multinomial_deviance(x, p_ref, pseudocount=0.0), expected, atol=1e-9
    )


def test_multinomial_deviance_identical_is_zero():
    p_ref = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    x = 100.0 * p_ref  # p̂ == p_ref exactly
    assert math.isclose(
        multinomial_deviance(x, p_ref, pseudocount=0.0).item(), 0.0, abs_tol=1e-9
    )


def test_multinomial_deviance_zero_counts_no_nan_at_pseudocount_zero():
    """Observed-zero channels must not produce NaN even at pseudocount=0
    (0·log 0 = 0); this is the reason for the direct G² implementation."""
    x = torch.tensor([8.0, 0.0, 2.0, 0.0], dtype=torch.float64)
    p_ref = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    dev = multinomial_deviance(x, p_ref, pseudocount=0.0)
    assert torch.isfinite(dev).all()


def test_multinomial_deviance_mean_approaches_K_minus_1():
    """Under x ~ Multinomial(N, p), E[2N·KL(p̂‖p)] ≈ K−1 (χ²_{K−1}),
    independent of N — the N-calibration that the L2 measures lack."""
    torch.manual_seed(0)
    K, N, n_sim = 6, 800, 6000
    p = torch.tensor([0.30, 0.25, 0.20, 0.13, 0.08, 0.04], dtype=torch.float64)
    samples = torch.distributions.Multinomial(total_count=N, probs=p).sample(
        (n_sim,)
    )  # (n_sim, K)
    dev = multinomial_deviance(samples, p.expand_as(samples), pseudocount=0.0)
    assert dev.shape == (n_sim,)
    assert abs(dev.mean().item() - (K - 1)) < 0.4


def test_multinomial_deviance_batched_shape():
    x = torch.tensor([[10.0, 5.0, 5.0], [1.0, 1.0, 8.0]], dtype=torch.float64)
    p_ref = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    out = multinomial_deviance(x, p_ref.expand_as(x))
    assert out.shape == (2,)


# ──────────────────────────────────────────────────────────────────────
# compare_spectra — dispatch parity + guards
# ──────────────────────────────────────────────────────────────────────


def test_compare_spectra_dispatch_parity():
    q = torch.tensor([5.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    r = torch.tensor([0.45, 0.30, 0.15, 0.10], dtype=torch.float64)
    assert torch.allclose(
        compare_spectra(q, r, method="cosine"), cosine_similarity(q, r)
    )
    assert torch.allclose(compare_spectra(q, r, method="dot"), cosine_similarity(q, r))
    assert torch.allclose(
        compare_spectra(q, r, method="kld"),
        kld(r, q, pseudocount=1e-3, reduce="none"),
    )
    assert torch.allclose(
        compare_spectra(q, r, method="multinomial_deviance", as_counts=True),
        multinomial_deviance(q, r),
    )


def test_compare_spectra_multinomial_requires_counts():
    q = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    r = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float64)
    with pytest.raises(ValueError, match="as_counts"):
        compare_spectra(q, r, method="multinomial_deviance")


def test_compare_spectra_unknown_method():
    q = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown method"):
        compare_spectra(q, q, method="bogus")  # type: ignore[arg-type]


def test_compare_spectra_spectral_angle_batched():
    q = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    r = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    out = compare_spectra(q, r, method="spectral_angle")
    assert out.shape == (2,)
    assert math.isclose(out[0].item(), 1.0, abs_tol=1e-6)


def test_compare_spectra_broadcasts_single_reference():
    """Replicate-vs-consensus: a (B, K) query against a single (K,) reference
    must work for every method (the common case). Regression for the
    spectral_angle batch_dim path that broke on a 1-D reference."""
    torch.manual_seed(0)
    B, K = 4, 6
    ref = torch.rand(K, dtype=torch.float64) + 0.1  # a single (K,) consensus
    counts = torch.randint(1, 50, (B, K)).to(torch.float64)
    for method in [
        "cosine",
        "dot",
        "pearson",
        "spectral_angle",
        "spectral_entropy",
        "kld",
    ]:
        out = compare_spectra(counts, ref, method=method)
        assert out.shape == (B,), f"{method} did not broadcast (K,) reference"
        assert torch.isfinite(out).all(), method
    md = compare_spectra(counts, ref, method="multinomial_deviance", as_counts=True)
    assert md.shape == (B,)


def test_compare_spectra_spectral_angle_matches_losses_on_1d():
    """The cosine-derived spectral_angle path equals losses.spectral_angle on
    1-D inputs (the identity it relies on)."""
    from constellation.core.stats.losses import spectral_angle

    q = torch.tensor([5.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    r = torch.tensor([4.0, 4.0, 1.0, 2.0], dtype=torch.float64)
    assert math.isclose(
        compare_spectra(q, r, method="spectral_angle").item(),
        spectral_angle(q, r).item(),
        abs_tol=1e-9,
    )
