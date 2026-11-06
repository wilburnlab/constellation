"""JChronologer's models (EncyclopeDIA's library predictor), run in PyTorch.

EncyclopeDIA 6.9.23 ships its predictor as TorchScript models inside
``lib/jchronologer-1.0.10.jar`` (searlelab/jchronologer, Apache-2.0).
They load in plain PyTorch and reproduce the jar's own predictions, so
Constellation runs them in-process, read from the user's install; it
never ships them.

    bundle  - the jar EncyclopeDIA's manifest names, its members read and
              checked against ``data/jchronologer_models.json``, models
              loaded from bytes (cached per process)
    models  - the four predictors: ``ChronologerRetentionTime``,
              ``CartographerFragments``, ``ElectricianChargeState``,
              ``SculptorCcs`` (the ``prediction.base.Predictor`` contract)

Importing this package loads neither torch nor a model; the predictors
load their model when first asked to predict.
"""

from constellation.massspec.library.jchronologer.bundle import JChronologerBundle, locate_bundle

__all__ = ["JChronologerBundle", "locate_bundle"]
