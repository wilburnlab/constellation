"""Isotope distributions — binned and isotopologue-resolved.

Two complementary APIs because real isotope chemistry has two natural
representations:

    binned (M+0, M+1, M+2, ...)
        One number per integer mass offset, peaks placed at
        `monoisotopic + k * ISOTOPE_MASS_DIFF` (¹³C-spacing approximation).
        Fast, the cartographer-equivalent path, what every low/medium-res
        MS scoring loop wants. Implemented by FFT polynomial multiplication
        of per-element abundance vectors keyed by mass-number offset.

    isotopologue-resolved
        Every distinct combination of isotopes as a separate
        `(exact_mass, abundance)` peak. ¹⁵N and ¹³C produce DISTINCT peaks
        at the same nominal mass. Used at high-res MS, in heavy-isotope
        labeled samples, and by NMR-adjacent code that needs exact masses.

Approximation contract
----------------------
`isotope_distribution` and `isotope_envelope` use the binned ¹³C-spacing
approximation: peak abundances are exact (FFT of per-element abundance
polynomials), but peak *masses* are placed at `mono + k · 1.0033548378 Da`.
This is correct to within a few ppm for natural-abundance organic
molecules at low/medium-res MS, and matches cartographer bit-for-bit.
For high-res MS (≥30K resolving power), heavy-isotope-labeled samples
(¹⁵N SILAC, ²H labeling), or NMR-adjacent applications needing exact
isotopologue masses, use `isotopologue_distribution` or
`isotope_envelope_exact`.

Both APIs accept `Composition` objects or raw count tensors.
"""

from __future__ import annotations

import torch

from constellation.core.chem.elements import ELEMENT_SYMBOLS, ELEMENTS, ISOTOPE_MASS_DIFF
from constellation.core.chem.composition import Composition

# ──────────────────────────────────────────────────────────────────────
# Binned (M+k) API — FFT polynomial multiplication
# ──────────────────────────────────────────────────────────────────────


def _poly_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Polynomial multiplication via FFT (convolution of coefficient vectors)."""
    n = len(a) + len(b) - 1
    fa = torch.fft.rfft(a.double(), n)
    fb = torch.fft.rfft(b.double(), n)
    return torch.fft.irfft(fa * fb, n).float()


def _poly_pow(base: torch.Tensor, exponent: int) -> torch.Tensor:
    """Raise a polynomial to an integer power via repeated squaring."""
    result = torch.ones(1)
    while exponent > 0:
        if exponent % 2 == 1:
            result = _poly_mul(result, base)
        base = _poly_mul(base, base)
        exponent //= 2
    return result


def _binned_abundances(symbol: str) -> torch.Tensor:
    """Per-element abundance polynomial keyed by mass-number offset from
    the lightest isotope. Empty for symbols with no isotopes.

    Example: O has ¹⁶O (0.99757), ¹⁷O (0.00038), ¹⁸O (0.00205) →
    polynomial [0.99757, 0.00038, 0.00205] of length 3.
    """
    isotopes = ELEMENTS[symbol].isotopes
    if not isotopes:
        return torch.tensor([1.0], dtype=torch.float32)
    # Filter to non-zero-abundance isotopes (radioactive-only entries are
    # not part of the natural-abundance distribution).
    natural = [i for i in isotopes if i.abundance > 0]
    if not natural:
        return torch.tensor([1.0], dtype=torch.float32)
    base_mn = min(i.mass_number for i in natural)
    max_mn = max(i.mass_number for i in natural)
    poly = torch.zeros(max_mn - base_mn + 1, dtype=torch.float32)
    for i in natural:
        poly[i.mass_number - base_mn] = i.abundance
    return poly


def _coerce_counts_dict(
    composition: Composition | torch.Tensor,
) -> dict[str, int]:
    """Common ingress: accept either a Composition or a raw count tensor."""
    if isinstance(composition, Composition):
        return composition.atoms
    counts = composition
    if counts.dim() != 1 or counts.shape[0] != len(ELEMENT_SYMBOLS):
        raise ValueError(
            f"raw counts tensor must be 1-D length {len(ELEMENT_SYMBOLS)}; "
            f"got {tuple(counts.shape)}"
        )
    return {
        ELEMENT_SYMBOLS[i]: int(counts[i])
        for i in range(len(ELEMENT_SYMBOLS))
        if int(counts[i]) > 0
    }


def isotope_distribution(
    composition: Composition | torch.Tensor,
    n_peaks: int = 5,
) -> torch.Tensor:
    """Binned (M+0, M+1, ..., M+n_peaks-1) abundance distribution.

    Peak abundances are exact (FFT-convolution of per-element abundance
    polynomials); peak *masses* are not returned here — see
    `isotope_envelope` for the matching mass tensor.

    Returns float32 of shape `(n_peaks,)`, normalized to sum to 1.0.
    """
    counts = _coerce_counts_dict(composition)
    result = torch.ones(1)
    for sym, n in counts.items():
        if n <= 0:
            continue
        element_abun = _binned_abundances(sym)
        if len(element_abun) == 1:
            # mono-isotopic element (P, F, I, Na, ...) — no convolution needed
            continue
        element_dist = _poly_pow(element_abun, n)
        result = _poly_mul(result, element_dist)
    result = result[:n_peaks]
    result = result.clamp(min=0.0)  # numerical noise can produce tiny negatives
    s = result.sum()
    if s.item() <= 0:
        return result
    return result / s


def isotope_envelope(
    composition: Composition | torch.Tensor,
    n_peaks: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binned envelope as `(masses, intensities)`, both shape `(n_peaks,)`.

    Masses are placed at `monoisotopic_mass + k * ISOTOPE_MASS_DIFF` for
    `k = 0..n_peaks-1` (¹³C-spacing approximation). Intensities are
    normalized abundances from `isotope_distribution`.
    """
    counts = _coerce_counts_dict(composition)
    mono = sum(n * ELEMENTS[s].monoisotopic_mass for s, n in counts.items())
    masses = torch.tensor(
        [mono + k * ISOTOPE_MASS_DIFF for k in range(n_peaks)],
        dtype=torch.float64,
    )
    intensities = isotope_distribution(composition, n_peaks)
    return masses, intensities


# ──────────────────────────────────────────────────────────────────────
# Isotopologue-resolved API — sieved convolution
# ──────────────────────────────────────────────────────────────────────


def _isotopologue_convolve(
    masses_a: torch.Tensor,
    abun_a: torch.Tensor,
    masses_b: torch.Tensor,
    abun_b: torch.Tensor,
    prune_below: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two isotopologue distributions: every (mass_a, mass_b) pair
    yields a peak at `mass_a + mass_b` with abundance `abun_a * abun_b`.
    Peaks below `prune_below` are dropped; identical masses are merged.

    Result is sorted by mass.
    """
    # Cartesian-product sums → (Ka * Kb,)
    cross_mass = (masses_a[:, None] + masses_b[None, :]).reshape(-1)
    cross_abun = (abun_a[:, None] * abun_b[None, :]).reshape(-1)
    # Prune
    keep = cross_abun >= prune_below
    cross_mass = cross_mass[keep]
    cross_abun = cross_abun[keep]
    # Merge identical masses (rare in floating point, but possible when
    # multiple atoms of the same element are convolved). Sort then group.
    if cross_mass.numel() == 0:
        return cross_mass, cross_abun
    order = torch.argsort(cross_mass)
    cross_mass = cross_mass[order]
    cross_abun = cross_abun[order]
    # Group exactly-equal floats. We use a tight tolerance because each
    # element's isotope masses are exact NIST values; any "equal" peaks
    # come from genuinely equivalent isotopologue combinations.
    same = torch.zeros_like(cross_mass, dtype=torch.bool)
    same[1:] = cross_mass[1:] == cross_mass[:-1]
    if not bool(same.any()):
        return cross_mass, cross_abun
    grouped_mass: list[float] = []
    grouped_abun: list[float] = []
    for i in range(cross_mass.numel()):
        if same[i]:
            grouped_abun[-1] += cross_abun[i].item()
        else:
            grouped_mass.append(cross_mass[i].item())
            grouped_abun.append(cross_abun[i].item())
    return (
        torch.tensor(grouped_mass, dtype=cross_mass.dtype),
        torch.tensor(grouped_abun, dtype=cross_abun.dtype),
    )


def _element_isotopologues(
    symbol: str, count: int, prune_below: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """All isotopologues of `count` atoms of `symbol`, after pruning.

    Self-convolves the per-element single-atom distribution `count` times
    (via repeated squaring) so that the per-step pruning keeps the
    distribution bounded.
    """
    natural = [i for i in ELEMENTS[symbol].isotopes if i.abundance > 0]
    if not natural:
        return torch.tensor([0.0], dtype=torch.float64), torch.tensor(
            [1.0], dtype=torch.float64
        )
    base_mass = torch.tensor([i.exact_mass for i in natural], dtype=torch.float64)
    base_abun = torch.tensor([i.abundance for i in natural], dtype=torch.float64)

    # Repeated squaring: build the n-atom distribution
    #   single-atom: (base_mass, base_abun)
    #   double-atom: convolve(single, single)
    #   etc.
    result_mass = torch.tensor([0.0], dtype=torch.float64)
    result_abun = torch.tensor([1.0], dtype=torch.float64)
    cur_mass = base_mass.clone()
    cur_abun = base_abun.clone()
    n = count
    while n > 0:
        if n & 1:
            result_mass, result_abun = _isotopologue_convolve(
                result_mass, result_abun, cur_mass, cur_abun, prune_below
            )
        n >>= 1
        if n > 0:
            cur_mass, cur_abun = _isotopologue_convolve(
                cur_mass, cur_abun, cur_mass, cur_abun, prune_below
            )
    return result_mass, result_abun


def isotopologue_distribution(
    composition: Composition,
    prune_below: float = 1e-6,
    max_peaks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full isotopologue-resolved distribution.

    Returns `(exact_masses, abundances)`, each shape `(K,)`, sorted by
    mass. K is bounded by `prune_below` (peaks with abundance below this
    threshold are discarded). For a typical tryptic peptide with
    `prune_below=1e-6`, K is in the low hundreds — tractable.

    The abundances DO NOT necessarily sum to 1.0: pruning can remove
    long-tail mass. Renormalize at the call site if needed.
    """
    if not isinstance(composition, Composition):
        raise TypeError("isotopologue_distribution requires a Composition")
    counts = composition.atoms
    result_mass = torch.tensor([0.0], dtype=torch.float64)
    result_abun = torch.tensor([1.0], dtype=torch.float64)
    for sym, n in counts.items():
        if n <= 0:
            continue
        elem_mass, elem_abun = _element_isotopologues(sym, n, prune_below)
        result_mass, result_abun = _isotopologue_convolve(
            result_mass, result_abun, elem_mass, elem_abun, prune_below
        )
    # Already sorted by mass from the convolve helper.
    if max_peaks is not None and result_mass.numel() > max_peaks:
        # Keep the top-`max_peaks` by abundance, then re-sort by mass.
        top = torch.topk(result_abun, max_peaks).indices
        order = torch.argsort(result_mass[top])
        return result_mass[top][order], result_abun[top][order]
    return result_mass, result_abun


def isotope_envelope_exact(
    composition: Composition,
    bin_width_da: float = 0.001,
    n_peaks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """High-resolution envelope: bin `isotopologue_distribution` at the
    requested mass resolution.

    Set `bin_width_da ~= 1/(2·resolving_power · m/z)` for an Orbitrap-style
    profile (e.g. 0.001 Da at m/z 500 with R=500K). Larger values
    approach the nominal-mass binned envelope.

    If `n_peaks` is provided, returns the lowest `n_peaks` bins by mass.
    """
    masses, abuns = isotopologue_distribution(composition)
    if masses.numel() == 0:
        return masses, abuns
    base = masses.min().item()
    bin_idx = ((masses - base) / bin_width_da).round().to(torch.int64)
    n_bins = int(bin_idx.max().item()) + 1
    binned_abun = torch.zeros(n_bins, dtype=abuns.dtype)
    binned_mass = torch.zeros(n_bins, dtype=masses.dtype)
    weight = torch.zeros(n_bins, dtype=abuns.dtype)
    binned_abun.index_add_(0, bin_idx, abuns)
    # Abundance-weighted mean mass within each bin.
    binned_mass.index_add_(0, bin_idx, masses * abuns)
    weight.index_add_(0, bin_idx, abuns)
    nonempty = weight > 0
    binned_mass[nonempty] = binned_mass[nonempty] / weight[nonempty]
    out_mass = binned_mass[nonempty]
    out_abun = binned_abun[nonempty]
    if n_peaks is not None and out_mass.numel() > n_peaks:
        out_mass = out_mass[:n_peaks]
        out_abun = out_abun[:n_peaks]
    return out_mass, out_abun


# ──────────────────────────────────────────────────────────────────────
# Convenience wrappers (also live as Composition properties)
# ──────────────────────────────────────────────────────────────────────


def monoisotopic_mass(composition: Composition) -> float:
    """Convenience: same as `Composition.mass`."""
    return composition.mass


def average_mass(composition: Composition) -> float:
    """Convenience: same as `Composition.average_mass`."""
    return composition.average_mass


__all__ = [
    "isotope_distribution",
    "isotope_envelope",
    "isotopologue_distribution",
    "isotope_envelope_exact",
    "monoisotopic_mass",
    "average_mass",
]
