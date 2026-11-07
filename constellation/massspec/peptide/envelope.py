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

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from constellation.core.chem.composition import Composition
from constellation.core.chem.elements import ELEMENT_SYMBOLS, ELEMENTS, ISOTOPE_MASS_DIFF
from constellation.core.chem.isotopes import (
    isotope_distributions,
    isotope_envelope,
    isotope_envelope_exact,
)
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import requires_canonical
from constellation.core.sequence.proforma import Peptidoform
from constellation.core.sequence.protein import (
    labelled_compositions,
    peptide_composition,
    peptide_compositions,
    peptide_mass,
    peptide_masses,
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
    normalize: Literal["kept", "full"] = "kept",
) -> tuple[torch.Tensor, torch.Tensor]:
    """A peptidoform's isotope envelope as neutral masses and intensities,
    before any charge: what `peptide_envelope` puts at each charge with
    `envelope_mz`, so the envelope of several charges is computed once.
    ``normalize="full"`` (binned) gives each peak's share of the whole
    envelope rather than of the peaks kept."""
    composition = peptide_composition(peptidoform, vocab=vocab)
    if peptidoform.global_isotopes:
        # Labelled atoms are one isotope each: no variance, only mass.
        labelled = labelled_compositions([peptidoform], vocab=vocab)[0]
        composition = Composition(composition.counts - labelled)

    if mode == "binned":
        masses_neutral, intensities = isotope_envelope(composition, n_peaks=n_peaks, normalize=normalize)
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


def neutral_envelopes(
    peptidoforms: Sequence[Peptidoform],
    *,
    n_peaks: int = 5,
    vocab: ModVocab = UNIMOD,
) -> tuple[np.ndarray, np.ndarray]:
    """Many peptidoforms' binned envelopes at once, before any charge:
    ``(masses, shares)``, both ``(N, n_peaks)`` float64. Row ``i``'s masses
    are ``neutral_envelope(peptidoforms[i], n_peaks=n_peaks)``'s to the bit
    (labels and ``mass_override`` alike); its shares are each peak's share
    of the whole envelope (``normalize="full"``), to float32 rounding.

    No work is done per peptidoform in Python: compositions and masses are
    batched (``peptide_compositions``, ``peptide_masses``), the
    distributions are one Fourier transform (``isotope_distributions``),
    and the monoisotopic mass repeats, element column by element column,
    the compensated summation Python's ``sum`` makes of the single
    function's terms, so the bits agree. One envelope at a time cost about
    1.4 ms a peptidoform, most of compiling a proteome's MS1 queries."""
    n = len(peptidoforms)
    if n == 0:
        return np.zeros((0, n_peaks)), np.zeros((0, n_peaks))
    counts = peptide_compositions(peptidoforms, vocab=vocab).numpy().astype(np.int64)
    if any(getattr(pf, "global_isotopes", ()) for pf in peptidoforms):
        # Labelled atoms are one isotope each: no variance, only mass.
        counts = counts - labelled_compositions(peptidoforms, vocab=vocab).numpy().astype(np.int64)
    element_mono = np.array([ELEMENTS[s].monoisotopic_mass for s in ELEMENT_SYMBOLS], dtype=np.float64)
    # sum(n * m for each element present) as Python 3.12 sums floats
    # (Neumaier's compensated summation), element by element in
    # ELEMENT_SYMBOLS order; an absent element adds nothing to either term.
    total = np.zeros(n)
    comp = np.zeros(n)
    for e in np.flatnonzero((counts != 0).any(axis=0)).tolist():
        x = counts[:, e] * element_mono[e]
        t = total + x
        comp += np.where(np.abs(total) >= np.abs(x), (total - t) + x, (x - t) + total)
        total = t
    mono = np.where((comp != 0) & np.isfinite(comp), total + comp, total)
    k = np.arange(n_peaks)
    masses = mono[:, None] + (k * ISOTOPE_MASS_DIFF)[None, :]
    # The centreline correction neutral_envelope makes: peptide_mass less
    # the skeleton's own mass (Composition.mass's reduction, batched).
    skeleton = (torch.from_numpy(counts.astype(np.float64)) * ELEMENTS.mass_tensor).sum(dim=-1).numpy()
    offset = peptide_masses(peptidoforms, vocab=vocab).numpy() - skeleton
    masses = np.where((offset != 0.0)[:, None], masses + offset[:, None], masses)
    return masses, isotope_distributions(counts, n_peaks)


__all__ = ["peptide_envelope", "neutral_envelope", "neutral_envelopes", "envelope_mz", "EnvelopeMode"]
