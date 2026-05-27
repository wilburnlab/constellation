"""Classical DSP windowing functions for 1D signals.

These return *bare* window arrays (length-N tensors) rather than applying
the window to a signal — the multiplication step is `core.signal.apodize`.
Separating the window-shape definition from the apply-step lets callers
mix and match: build a custom window once, apply it multiple times; sample
a window at non-standard length; combine windows by elementwise product.

Window shapes:
    sine_bell   classical NMR-style sine-bell window with offset, end, power
    hann        0.5 − 0.5 cos(2πi/(N-1))                       — smooth taper, 0 at both ends
    hamming     0.54 − 0.46 cos(2πi/(N-1))                     — Hann variant, ~0.08 at endpoints
    blackman    0.42 − 0.5 cos(2πi/(N-1)) + 0.08 cos(4πi/(N-1)) — narrower lobes than Hann
    tukey       cosine-taper of width αN/2 at each end, flat in the middle

Pattern: domain-specific apodization (NMR's `em` and `gaussian`, future
chromatography prefilters, etc.) wraps these — distribution-shaped windows
go through `core.stats` (Gaussian, Lorentzian-derived exponential decay),
classical DSP windows live here.
"""

from __future__ import annotations

import torch


def sine_bell(
    n: int,
    off: float = 0.0,
    end: float = 1.0,
    power: int = 1,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sine-bell (and squared sine-bell) window.

    Sweeps the sine argument from `off·π` to `end·π` across the N points,
    then raises to `power`:

        w[i] = sin(off·π + i/(N-1) · (end - off)·π) ** power

    Classical NMR usages:
        * `off=0, end=1, power=1` — pure sine bell (rises from zero, peaks
          at midpoint, returns to zero).
        * `off=0.5, end=1, power=1` — shifted sine bell (starts at peak,
          decays to zero) — similar shape to an exponential apodization.
        * `power=2` — squared sine bell, common in 2D processing for
          additional taper.

    Parameters
    ----------
    n : int
        Window length in samples.
    off : float
        Starting phase as a fraction of π. Default 0.0.
    end : float
        Ending phase as a fraction of π. Default 1.0.
    power : int
        Exponent applied to the sine values. Default 1.
    dtype : torch.dtype
        Result dtype. Default float64.

    Returns
    -------
    torch.Tensor
        1D real tensor of length `n`.
    """
    if n < 1:
        raise ValueError(f"`n` must be a positive integer, got {n}.")
    i = torch.arange(n, dtype=dtype)
    if n == 1:
        # avoid division by zero in the (i / (n-1)) factor
        return torch.full((1,), float(torch.sin(torch.tensor(off * torch.pi, dtype=dtype)) ** power), dtype=dtype)
    phase = off * torch.pi + (i / (n - 1)) * (end - off) * torch.pi
    return torch.sin(phase) ** power


def hann(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Hann (raised-cosine) window: ``0.5 − 0.5 cos(2πi/(N-1))``.

    Zero at both endpoints, peak of 1.0 at the centre.
    """
    if n < 1:
        raise ValueError(f"`n` must be a positive integer, got {n}.")
    if n == 1:
        return torch.zeros(1, dtype=dtype)
    i = torch.arange(n, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * i / (n - 1))


def hamming(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Hamming window: ``0.54 − 0.46 cos(2πi/(N-1))``.

    Non-zero at the endpoints (~0.08) — lower first sidelobe than Hann
    at the cost of small endpoint discontinuities.
    """
    if n < 1:
        raise ValueError(f"`n` must be a positive integer, got {n}.")
    if n == 1:
        return torch.full((1,), 0.08, dtype=dtype)
    i = torch.arange(n, dtype=dtype)
    return 0.54 - 0.46 * torch.cos(2.0 * torch.pi * i / (n - 1))


def blackman(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Blackman window: ``0.42 − 0.5 cos(2πi/(N-1)) + 0.08 cos(4πi/(N-1))``.

    Higher-order taper than Hann/Hamming; lower sidelobes at the cost
    of a slightly wider main lobe.
    """
    if n < 1:
        raise ValueError(f"`n` must be a positive integer, got {n}.")
    if n == 1:
        return torch.zeros(1, dtype=dtype)
    i = torch.arange(n, dtype=dtype)
    return (
        0.42
        - 0.5 * torch.cos(2.0 * torch.pi * i / (n - 1))
        + 0.08 * torch.cos(4.0 * torch.pi * i / (n - 1))
    )


def tukey(
    n: int, alpha: float = 0.5, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Tukey (cosine-tapered rectangular) window.

    Cosine-taper of length ``α·(N-1)/2`` at each end, flat in the middle.
    Special cases:
        * `alpha = 0`   → rectangular window (all ones).
        * `alpha = 1`   → Hann window.
        * `0 < alpha < 1` → flat middle of width `(1-α)·(N-1)`, cosine
          tapers on the ends.

    Parameters
    ----------
    n : int
        Window length.
    alpha : float
        Fraction of the window covered by the cosine taper (split equally
        between the two ends). Default 0.5.
    dtype : torch.dtype
        Result dtype. Default float64.
    """
    if n < 1:
        raise ValueError(f"`n` must be a positive integer, got {n}.")
    if alpha <= 0.0:
        return torch.ones(n, dtype=dtype)
    if alpha >= 1.0:
        return hann(n, dtype=dtype)
    if n == 1:
        return torch.ones(1, dtype=dtype)

    i = torch.arange(n, dtype=dtype)
    width = alpha * (n - 1) / 2.0  # length of each cosine taper

    w = torch.ones(n, dtype=dtype)
    # Left taper: i in [0, width]
    left_mask = i < width
    w = torch.where(
        left_mask,
        0.5 * (1.0 + torch.cos(torch.pi * (i / width - 1.0))),
        w,
    )
    # Right taper: i in [n - 1 - width, n - 1]
    right_mask = i > (n - 1 - width)
    w = torch.where(
        right_mask,
        0.5
        * (1.0 + torch.cos(torch.pi * ((i - (n - 1 - width)) / width))),
        w,
    )
    return w
