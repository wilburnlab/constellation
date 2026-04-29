"""Charge-aware peptide m/z helpers.

Scalar entry points for peptide-level mass-to-charge conversions. The
heavy lifting (residue + modification mass arithmetic, heavy-isotope
correction, global isotope labeling) is in
`core.sequence.protein.peptide_mass`; this module just adds the charge
term.

Fragment-ion m/z lives in `peptide.ions` because it intimately depends
on `IonType` and per-ion-type composition offsets.

Re-exports `PROTON_MASS` from `core.chem.elements` so MS callers don't
have to reach across `core.stats.units` for it.
"""

from __future__ import annotations

from constellation.core.chem.elements import PROTON_MASS  # noqa: F401  (re-export)
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import Peptidoform
from constellation.core.sequence.protein import peptide_mass


def precursor_mz(
    peptidoform: Peptidoform,
    *,
    charge: int | None = None,
    monoisotopic: bool = True,
    vocab: ModVocab = UNIMOD,
) -> float:
    """Charged-precursor m/z for a peptidoform.

    `m/z = (peptide_mass + charge * PROTON_MASS) / charge`. Delegates the
    neutral-mass calculation to `core.sequence.protein.peptide_mass`,
    which handles heavy-isotope corrections (`mass_override` on TMT /
    SILAC mods) and global isotope labels (``<13C>``, ``<15N>``, ...).

    `charge` falls back to ``peptidoform.charge`` (parsed from the
    ProForma ``/N`` suffix) when omitted; passing ``charge=`` explicitly
    overrides whatever is on the peptidoform.

    Adducts (``peptidoform.adducts``) are not yet supported and raise
    ``NotImplementedError`` — adduct mass arithmetic lands in a focused
    PR when concrete library data drives the requirement.
    """
    z = charge if charge is not None else peptidoform.charge
    if z is None:
        raise ValueError(
            "precursor_mz requires charge — pass charge=N or use a "
            "Peptidoform parsed from a /N-suffixed ProForma string"
        )
    if z <= 0:
        raise ValueError(f"charge must be positive; got {z}")
    if peptidoform.adducts:
        raise NotImplementedError(
            "precursor_mz with adducts is not yet implemented; adduct "
            "mass arithmetic lands in a focused PR when concrete library "
            "data drives the requirement"
        )
    mass = peptide_mass(peptidoform, monoisotopic=monoisotopic, vocab=vocab)
    return (mass + z * PROTON_MASS) / z


__all__ = ["PROTON_MASS", "precursor_mz"]
