"""Peptide isotope envelopes — charge-aware wrappers over `core.chem.isotopes`.

`peptide_envelope` builds the elemental composition of a (modified) peptide
via `core.sequence.protein.peptide_composition`, runs it through either the
binned (`isotope_envelope`) or exact (`isotope_envelope_exact`) path in
`core.chem.isotopes`, then converts neutral masses to m/z at the requested
charge.

Use:
    binned   — Cartographer-style M+0/M+1/M+2 envelope at ¹³C-spacing.
               Fast, correct to a few ppm at low/medium-res MS.
    exact    — Resolves ¹³C vs ¹⁵N (and other co-occurring isotopes).
               Required for Orbitrap-class resolution and labelled samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch

from constellation.core.chem.isotopes import (
    isotope_envelope,
    isotope_envelope_exact,
)
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import requires_canonical
from constellation.core.sequence.protein import peptide_composition, peptide_mass
from constellation.massspec.peptide.mz import PROTON_MASS

EnvelopeMode = Literal["binned", "exact"]


@requires_canonical
def peptide_envelope(
    seq: str,
    *,
    modifications: Mapping[int, str] | None = None,
    charge: int = 1,
    n_peaks: int = 5,
    mode: EnvelopeMode = "binned",
    bin_width_da: float = 1e-3,
    vocab: ModVocab = UNIMOD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isotope envelope at the given charge for a (modified) peptide.

    Returns ``(mz, intensities)``. Both are torch tensors. Intensities
    come straight from the underlying isotope routine and are not
    re-normalized (the binned path normalizes to sum 1.0; the exact
    path's pruning may leave a slightly less-than-unit total).

    Heavy-isotope mods (mass_override) shift the monoisotopic peak via
    `peptide_mass`; the light-skeleton `delta_composition` participates
    in the abundance polynomial. This matches the convention in
    `peptide_composition` / `peptide_mass`.
    """
    if charge <= 0:
        raise ValueError(f"charge must be positive; got {charge}")

    composition = peptide_composition(
        seq, modifications=modifications, vocab=vocab
    )

    if mode == "binned":
        masses_neutral, intensities = isotope_envelope(composition, n_peaks=n_peaks)
    elif mode == "exact":
        masses_neutral, intensities = isotope_envelope_exact(
            composition, bin_width_da=bin_width_da, n_peaks=n_peaks
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'binned' or 'exact'")

    # Heavy-isotope correction: `isotope_envelope`'s mono mass uses the
    # light-atom skeleton, so a peptide with mass_override mods needs an
    # offset added uniformly to the entire envelope.
    if modifications:
        canonical_neutral = peptide_mass(
            seq, modifications=modifications, vocab=vocab
        )
        skeleton_neutral = composition.mass
        offset = canonical_neutral - skeleton_neutral
        if offset != 0.0:
            masses_neutral = masses_neutral + offset

    masses_mz = (
        masses_neutral.to(torch.float64) + charge * PROTON_MASS
    ) / charge
    return masses_mz, intensities


__all__ = ["peptide_envelope", "EnvelopeMode"]
