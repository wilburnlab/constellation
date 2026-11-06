"""Predicted properties put onto a library, with interchangeable predictors.

A *predictor* produces one property of a library's precursors (fragment
intensities, retention time, CCS, charge-state probability) behind one
contract: say which requests it cannot take, then predict the rest into
plain arrays. Koina (over the network) and JChronologer's models (read
from the installed EncyclopeDIA jar, run in-process) are predictors; an
in-house model joins the same way. See ``docs/plans/library-prediction.md``.

    base               the contract: ``Predictor``, ``Capabilities``,
                       ``Requests``, ``Refusals``, ``FragmentArrays``,
                       ``ValueArray``
    collision_energy   the energy a fragment predictor is run at, as a
                       property of each precursor charge
    requests           a library's precursors as unique unlabelled
                       (peptide, charge, energy) requests, each precursor's
                       request and its target/decoy group
    torch_runner       batched TorchScript inference whose output does not
                       depend on the thread count
    assemble           predicted fragments onto the library's own
                       precursors, m/z from their (labelled) peptides
    run                ``predict_onto``: check, drop refused precursors with
                       their partners, predict, one library and a report
    registry           ``backend:model`` → predictor, imported lazily

Importing this package loads no model and not koinapy.
"""

from constellation.massspec.library.prediction.base import (
    PROPERTIES,
    Capabilities,
    FragmentArrays,
    Predictor,
    Property,
    Refusals,
    Requests,
    ValueArray,
)
from constellation.massspec.library.prediction.collision_energy import (
    DEFAULT_REFERENCE_CHARGE,
    OUTSIDE_FACTOR,
    SEARLE_2020_FACTORS,
    CollisionEnergy,
    parse_energy_table,
)
from constellation.massspec.library.prediction.registry import default_predictors, predictor
from constellation.massspec.library.prediction.run import PredictionReport, predict_onto

__all__ = [
    "DEFAULT_REFERENCE_CHARGE",
    "OUTSIDE_FACTOR",
    "PROPERTIES",
    "SEARLE_2020_FACTORS",
    "Capabilities",
    "CollisionEnergy",
    "FragmentArrays",
    "PredictionReport",
    "Predictor",
    "Property",
    "Refusals",
    "Requests",
    "ValueArray",
    "default_predictors",
    "parse_energy_table",
    "predict_onto",
    "predictor",
]
