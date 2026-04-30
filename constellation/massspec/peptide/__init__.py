"""Peptide-level MS chemistry — m/z, fragment ions, isotope envelopes,
biochemical neutral-loss rules.

Pure physics: nothing here is model- or pipeline-specific. The encoding
side (per-model peptide tokenizers / vocabularies) lives in the sibling
`constellation.massspec.tokenize` module — keeps this layer free of any
single tool's vocabulary commitments.

Submodules:
    mz                m/z helpers — `precursor_mz`, `PROTON_MASS` re-export
    ions              `IonType`, `FragmentIon`, `fragment_mz`, `fragment_ladder`
    envelope          `peptide_envelope` (charge-aware isotope envelopes)
    neutral_losses    `NeutralLoss`, `LOSS_REGISTRY`, `loss_applies`
"""

from constellation.massspec.peptide.envelope import peptide_envelope
from constellation.massspec.peptide.ions import (
    ION_OFFSET_MASSES,
    ION_OFFSETS,
    FragmentIon,
    IonType,
    fragment_ladder,
    fragment_mz,
)
from constellation.massspec.peptide.match import (
    IonAssignment,
    MzMatch,
    assign_fragments,
    match_mz,
)
from constellation.massspec.peptide.mz import PROTON_MASS, precursor_mz
from constellation.massspec.peptide.neutral_losses import (
    LOSS_REGISTRY,
    LossRegistry,
    NeutralLoss,
    loss_applies,
)

__all__ = [
    "PROTON_MASS",
    "precursor_mz",
    "IonType",
    "ION_OFFSETS",
    "ION_OFFSET_MASSES",
    "FragmentIon",
    "fragment_mz",
    "fragment_ladder",
    "peptide_envelope",
    "NeutralLoss",
    "LossRegistry",
    "LOSS_REGISTRY",
    "loss_applies",
    "MzMatch",
    "IonAssignment",
    "match_mz",
    "assign_fragments",
]
