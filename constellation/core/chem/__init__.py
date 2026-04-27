"""Elemental primitives — elements, compositions, isotopes, modifications.

Purely chemical; knows nothing about residues, sequences, or alphabets.
Downstream `core.sequence` builds residue/peptide compositions *from*
`Composition` primitives defined here.

Modules:
    elements       — periodic-table data: Element, Isotope, ElementTable,
                     ELEMENTS, ELEMENT_SYMBOLS, ELEMENT_TYPES, MASSES,
                     WEIGHTS, plus PROTON_MASS / ISOTOPE_MASS_DIFF / etc.
    composition    — Composition class (1-D int32 tensor wrapper),
                     parse_formula (Hill notation in/out), batched helpers.
    isotopes       — isotope_distribution / isotope_envelope (binned, fast)
                     and isotopologue_distribution / isotope_envelope_exact
                     (high-resolution, isotopologue-resolved).
    modifications  — Modification, ModVocab (with subset/supports/
                     find_by_mass/register_custom), built-in UNIMOD vocab
                     populated from packaged JSON.
"""

from constellation.core.chem.elements import (
    AVOGADRO,
    ELECTRON_MASS,
    ELEMENT_SYMBOLS,
    ELEMENT_TYPES,
    ELEMENTS,
    Element,
    ElementTable,
    FEATURE_COLUMNS,
    ISOTOPE_MASS_DIFF,
    Isotope,
    MASSES,
    NEUTRON_MASS,
    PROTON_MASS,
    WEIGHTS,
)
from constellation.core.chem.composition import (
    Composition,
    batched_average_mass,
    batched_mass,
    parse_formula,
    stack,
)
from constellation.core.chem.isotopes import (
    average_mass,
    isotope_distribution,
    isotope_envelope,
    isotope_envelope_exact,
    isotopologue_distribution,
    monoisotopic_mass,
)
from constellation.core.chem.modifications import (
    Modification,
    ModVocab,
    UNIMOD,
)

__all__ = [
    # elements
    "Element",
    "ElementTable",
    "Isotope",
    "ELEMENTS",
    "ELEMENT_SYMBOLS",
    "ELEMENT_TYPES",
    "MASSES",
    "WEIGHTS",
    "FEATURE_COLUMNS",
    "PROTON_MASS",
    "ELECTRON_MASS",
    "NEUTRON_MASS",
    "ISOTOPE_MASS_DIFF",
    "AVOGADRO",
    # composition
    "Composition",
    "parse_formula",
    "stack",
    "batched_mass",
    "batched_average_mass",
    # isotopes
    "isotope_distribution",
    "isotope_envelope",
    "isotopologue_distribution",
    "isotope_envelope_exact",
    "monoisotopic_mass",
    "average_mass",
    # modifications
    "Modification",
    "ModVocab",
    "UNIMOD",
]
