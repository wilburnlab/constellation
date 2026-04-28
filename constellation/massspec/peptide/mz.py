"""Charge-aware peptide m/z helpers.

Scalar entry points for peptide-level mass-to-charge conversions. The
heavy lifting (residue + modification mass arithmetic, heavy-isotope
correction) is in `core.sequence.protein.peptide_mass`; this module
just adds the charge term.

Fragment-ion m/z lives in `peptide.ions` because it intimately depends
on `IonType` and per-ion-type composition offsets.

Re-exports `PROTON_MASS` from `core.chem.elements` so MS callers don't
have to reach across `core.stats.units` for it.
"""

from __future__ import annotations

from collections.abc import Mapping

from constellation.core.chem.elements import PROTON_MASS  # noqa: F401  (re-export)
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.protein import peptide_mass


def precursor_mz(
    seq: str,
    charge: int,
    *,
    modifications: Mapping[int, str] | None = None,
    monoisotopic: bool = True,
    vocab: ModVocab = UNIMOD,
) -> float:
    """Charged-precursor m/z for a (modified) peptide.

    `m/z = (peptide_mass + charge * PROTON_MASS) / charge`. Delegates the
    neutral-mass calculation to `core.sequence.protein.peptide_mass`,
    which handles the heavy-isotope correction for mods carrying
    `mass_override` (TMT, SILAC ¹³C₆, ...).
    """
    if charge <= 0:
        raise ValueError(f"charge must be positive; got {charge}")
    mass = peptide_mass(
        seq, modifications=modifications, monoisotopic=monoisotopic, vocab=vocab
    )
    return (mass + charge * PROTON_MASS) / charge


__all__ = ["PROTON_MASS", "precursor_mz"]
