"""The single-peptidoform ``peptide_composition`` / ``peptide_mass`` as they
were before 2026-11-05 (``constellation.core.sequence.protein`` at the
commit before the batched rewrite), frozen as the oracle the batched
implementation is held to bit for bit. Do not edit; it is a record."""

from __future__ import annotations

from constellation.core.chem.composition import Composition
from constellation.core.chem.elements import ELEMENTS
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import AA
from constellation.core.sequence.ops import validate
from constellation.core.sequence.proforma import MultiPeptidoform, Peptidoform, parse_isotope_label
from constellation.core.sequence.protein import (
    _H2O,
    _count_fixed_mod_locations,
    _is_bare_mass_delta,
    _modref_contribution,
    _residue_index_check,
    _walk_modrefs,
)

__all__ = ["peptide_composition", "peptide_mass"]


def peptide_composition(
    peptidoform: Peptidoform,
    *,
    vocab: ModVocab = UNIMOD,
) -> Composition:
    """Sum residue compositions + terminal H₂O + every modification's
    composition delta into a single ``Composition``.

    Modification handling:

    - Residue mods, terminal mods, labile mods, unknown-position mods
      (multiplied by their count), range mods, and fixed mods all
      contribute their ``delta_composition``.
    - Cross-link / ambiguity / branch groups (TaggedMod ``group_id``)
      contribute exactly once across all anchor positions — the chemistry
      is shared between participants.
    - Heavy-isotope modifications (``mass_override``) contribute the
      *light-atom skeleton*; ``peptide_mass`` resolves the canonical mass.
    - Global isotope labels (``<13C>``, ``<15N>``, ...) contribute zero
      composition; they shift mass only and are handled in ``peptide_mass``.

    Raises:

    - ``KeyError`` for residues outside canonical AA or unknown vocab keys.
    - ``ValueError`` for ModRefs that carry only a bare mass delta (no
      composition info available — call ``peptide_mass`` instead).
    - ``IndexError`` for residue-mod positions outside the sequence.
    - ``NotImplementedError`` for ``MultiPeptidoform`` inputs (iterate
      ``.chains`` for per-chain compositions) and for glycan-payload mods.
    """
    if isinstance(peptidoform, MultiPeptidoform):
        raise NotImplementedError(
            "joint multi-chain composition is not yet supported; iterate "
            "peptidoform.chains for per-chain compositions"
        )

    seq = peptidoform.sequence
    validate(seq, AA)
    assert AA.compositions is not None  # canonical alphabet always carries them
    _residue_index_check(peptidoform)

    comp = Composition.zeros()
    for residue in seq:
        comp = comp + AA.compositions[residue]
    comp = comp + _H2O

    for modref in _walk_modrefs(peptidoform):
        if _is_bare_mass_delta(modref):
            raise ValueError(
                f"cannot derive composition from bare mass-delta ModRef "
                f"({modref.mass_delta:+f}) — peptide_mass accepts mass-only "
                "mods, but peptide_composition needs a UNIMOD/name/formula "
                "payload to determine element counts"
            )
        d_comp, _ = _modref_contribution(modref, vocab, monoisotopic=True)
        comp = comp + d_comp

    for modref, locations in peptidoform.fixed_mods:
        d_comp, _ = _modref_contribution(modref, vocab, monoisotopic=True)
        for loc in locations:
            count = _count_fixed_mod_locations(seq, loc, peptidoform)
            if count:
                comp = comp + d_comp * count

    return comp


def peptide_mass(
    peptidoform: Peptidoform,
    *,
    monoisotopic: bool = True,
    vocab: ModVocab = UNIMOD,
) -> float:
    """Monoisotopic (default) or standard-weight mass of a peptidoform.

    Aggregates: residue + terminal-water masses + every modification's
    canonical mass contribution + global-isotope label corrections.

    Heavy-isotope modifications (``mass_override``) and global isotope
    labels (``<13C>``) only adjust the result in ``monoisotopic=True``
    mode — average-mass calculations operate on standard atomic weights,
    where these distinctions are not meaningful.
    """
    if isinstance(peptidoform, MultiPeptidoform):
        raise NotImplementedError(
            "joint multi-chain mass is not yet supported; iterate "
            "peptidoform.chains for per-chain masses"
        )

    seq = peptidoform.sequence
    validate(seq, AA)
    assert AA.compositions is not None
    _residue_index_check(peptidoform)

    comp = Composition.zeros()
    for residue in seq:
        comp = comp + AA.compositions[residue]
    comp = comp + _H2O

    extra_mass = 0.0
    for modref in _walk_modrefs(peptidoform):
        d_comp, d_mass = _modref_contribution(modref, vocab, monoisotopic=monoisotopic)
        comp = comp + d_comp
        extra_mass += d_mass

    for modref, locations in peptidoform.fixed_mods:
        d_comp, d_mass = _modref_contribution(modref, vocab, monoisotopic=monoisotopic)
        for loc in locations:
            count = _count_fixed_mod_locations(seq, loc, peptidoform)
            if count:
                comp = comp + d_comp * count
                extra_mass += d_mass * count

    base = comp.mass if monoisotopic else comp.average_mass

    if peptidoform.global_isotopes:
        if not monoisotopic:
            raise ValueError(
                "global isotope labels are only meaningful in monoisotopic "
                "mode — pass monoisotopic=True or strip global_isotopes"
            )
        atoms = comp.atoms
        for label in peptidoform.global_isotopes:
            mass_n, sym = parse_isotope_label(label)
            atom_count = atoms.get(sym, 0)
            if atom_count == 0:
                continue
            elem = ELEMENTS[sym]
            extra_mass += (elem.isotope_mass(mass_n) - elem.monoisotopic_mass) * atom_count

    return base + extra_mass
