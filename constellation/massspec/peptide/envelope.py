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
uniformly via `peptide_mass`; their heavy atoms are still given natural
variance (UNIMOD stores no per-atom isotopes for them). Global ProForma
isotope labels (``<13C>``, ``<15N>``) are taken as full incorporation: the
atoms a label covers (`core.sequence.protein.labelled_compositions`, the
chain's) are each exactly that isotope, so they move the envelope without
widening it, and the envelope's shape comes from the remaining atoms only,
in both modes. Partial incorporation (a distribution of labelled atoms) is
not represented, as ProForma 2.0 does not represent it either (§5.6).
"""

from __future__ import annotations

from typing import Literal

import torch

from constellation.core.chem.composition import Composition
from constellation.core.chem.isotopes import (
    isotope_envelope,
    isotope_envelope_exact,
)
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import requires_canonical
from constellation.core.sequence.proforma import Peptidoform
from constellation.core.sequence.protein import (
    labelled_compositions,
    peptide_composition,
    peptide_mass,
)
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

    masses_neutral, intensities = neutral_envelope(
        peptidoform, n_peaks=n_peaks, mode=mode, bin_width_da=bin_width_da, vocab=vocab
    )
    return envelope_mz(masses_neutral, z), intensities


@requires_canonical
def neutral_envelope(
    peptidoform: Peptidoform,
    *,
    n_peaks: int = 5,
    mode: EnvelopeMode = "binned",
    bin_width_da: float = 1e-3,
    vocab: ModVocab = UNIMOD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A peptidoform's isotope envelope as neutral masses and intensities,
    before any charge: what `peptide_envelope` puts at each charge with
    `envelope_mz`, so the envelope of several charges is computed once."""
    composition = peptide_composition(peptidoform, vocab=vocab)
    if peptidoform.global_isotopes:
        # Labelled atoms are one isotope each: no variance, only mass.
        labelled = labelled_compositions([peptidoform], vocab=vocab)[0]
        composition = Composition(composition.counts - labelled)

    if mode == "binned":
        masses_neutral, intensities = isotope_envelope(composition, n_peaks=n_peaks)
    elif mode == "exact":
        masses_neutral, intensities = isotope_envelope_exact(
            composition, bin_width_da=bin_width_da, n_peaks=n_peaks
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'binned' or 'exact'")

    # Centerline correction: peptide_mass already accounts for heavy-isotope
    # `mass_override` and global isotope labeling (and the labelled atoms
    # left out of the composition above). The composition's mass falls
    # short by exactly that, so the whole envelope is shifted.
    canonical_neutral = peptide_mass(peptidoform, vocab=vocab)
    skeleton_neutral = composition.mass
    offset = canonical_neutral - skeleton_neutral
    if offset != 0.0:
        masses_neutral = masses_neutral + offset
    return masses_neutral, intensities


def envelope_mz(masses_neutral: torch.Tensor, charge: int) -> torch.Tensor:
    """Neutral envelope masses as m/z at ``charge`` (protonated)."""
    return (masses_neutral.to(torch.float64) + charge * PROTON_MASS) / charge


__all__ = ["peptide_envelope", "neutral_envelope", "envelope_mz", "EnvelopeMode"]
