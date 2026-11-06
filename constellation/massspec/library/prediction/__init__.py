"""Predicted properties put onto a library, with interchangeable predictors.

A *predictor* produces one property of a library's precursors (fragment
intensities, retention time, CCS, charge-state probability) behind one
contract: prepare inputs, run a model, decode into a common form. Koina
(over the network) and JChronologer's models (read from the installed
EncyclopeDIA jar, run in-process) are predictors; an in-house model
joins the same way. See ``docs/plans/library-prediction.md``.

Built so far:

    collision_energy   the energy a fragment predictor is run at, as a
                       property of each precursor charge

This package imports neither torch models nor koinapy on import.
"""

from constellation.massspec.library.prediction.collision_energy import (
    DEFAULT_REFERENCE_CHARGE,
    OUTSIDE_FACTOR,
    SEARLE_2020_FACTORS,
    CollisionEnergy,
    parse_energy_table,
)

__all__ = [
    "DEFAULT_REFERENCE_CHARGE",
    "OUTSIDE_FACTOR",
    "SEARLE_2020_FACTORS",
    "CollisionEnergy",
    "parse_energy_table",
]
