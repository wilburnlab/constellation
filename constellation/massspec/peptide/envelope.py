"""Peptide isotope envelopes — charge-aware wrappers over `core.chem.isotopes`.

`peptide_envelope` builds the elemental composition of a peptidoform
via `core.sequence.protein.peptide_composition`, runs it through either
the binned (`isotope_envelope`) or exact (`isotope_envelope_exact`) path
in `core.chem.isotopes`, then converts neutral masses to m/z at the
requested charge.

Use:
    binned   — Cartographer-style M+0/M+1/M+2 envelope at ¹³C-spacing.
               Fast, correct to a few ppm at low/medium-res MS.
    exact    — Resolves ¹³C vs ¹⁵N (and other co-occurring isotopes).
               Required for Orbitrap-class resolution and labelled samples.

Heavy-isotope mods (`mass_override` on TMT / SILAC) shift the centerline
uniformly via `peptide_mass`. Global ProForma isotope labels (``<13C>``,
``<15N>``) shift the centerline through the same path; the binned-mode
envelope shape is unaffected (¹³C-spacing approximation brackets the
labeled centerline). Exact-mode global isotope labeling is deferred —
it would require swapping the natural-abundance distribution with a
labeled distribution at the `core.chem.isotopes` layer.
"""

from __future__ import annotations

from typing import Literal

import torch

from constellation.core.chem.isotopes import (
    isotope_envelope,
    isotope_envelope_exact,
)
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import requires_canonical
from constellation.core.sequence.proforma import Peptidoform
from constellation.core.sequence.protein import peptide_composition, peptide_mass
from constellation.massspec.peptide.mz import PROTON_MASS

EnvelopeMode = Literal["binned", "exact"]


@requires_canonical
def peptide_envelope(
    peptidoform: Peptidoform,
    *,
    charge: int | None = None,
    n_peaks: int = 5,
    mode: EnvelopeMode = "binned",
    bin_width_da: float = 1e-3,
    vocab: ModVocab = UNIMOD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isotope envelope at the given charge for a peptidoform.

    Returns ``(mz, intensities)``. Both are torch tensors. Intensities
    come straight from the underlying isotope routine and are not
    re-normalized (the binned path normalizes to sum 1.0; the exact
    path's pruning may leave a slightly less-than-unit total).

    `charge` falls back to ``peptidoform.charge`` (parsed from the
    ProForma ``/N`` suffix) when omitted.
    """
    z = charge if charge is not None else peptidoform.charge
    if z is None:
        raise ValueError(
            "peptide_envelope requires charge — pass charge=N or use a "
            "Peptidoform parsed from a /N-suffixed ProForma string"
        )
    if z <= 0:
        raise ValueError(f"charge must be positive; got {z}")
    if peptidoform.global_isotopes and mode == "exact":
        raise NotImplementedError(
            "global isotope labels (<13C>, <15N>, ...) in exact-envelope "
            "mode require a labeled isotope distribution at the "
            "core.chem.isotopes layer; deferred until a concrete consumer "
            "drives the design"
        )

    composition = peptide_composition(peptidoform, vocab=vocab)

    if mode == "binned":
        masses_neutral, intensities = isotope_envelope(composition, n_peaks=n_peaks)
    elif mode == "exact":
        masses_neutral, intensities = isotope_envelope_exact(
            composition, bin_width_da=bin_width_da, n_peaks=n_peaks
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'binned' or 'exact'")

    # Centerline correction: peptide_mass already accounts for heavy-isotope
    # `mass_override` and global isotope labeling. The skeleton mass from
    # composition.mass underestimates by exactly that delta, so we shift
    # the entire envelope uniformly.
    canonical_neutral = peptide_mass(peptidoform, vocab=vocab)
    skeleton_neutral = composition.mass
    offset = canonical_neutral - skeleton_neutral
    if offset != 0.0:
        masses_neutral = masses_neutral + offset

    masses_mz = (
        masses_neutral.to(torch.float64) + z * PROTON_MASS
    ) / z
    return masses_mz, intensities


__all__ = ["peptide_envelope", "EnvelopeMode"]
