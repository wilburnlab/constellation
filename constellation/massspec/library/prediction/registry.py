"""Predictors by name: ``backend:model`` → a predictor, imported lazily.

``jchronologer:{chronologer,cartographer,electrician,sculptor}`` are
EncyclopeDIA's models in PyTorch; ``koina:<model>`` is any model the
Koina server serves (``koina:Prosit_2020_intensity_HCD``,
``koina:Chronologer_RT``), its property read from the overlay
(``koina.models``) or given.

``predictor("jchronologer:cartographer", threads=8)`` builds one;
``default_predictors("jchronologer")`` builds a backend's default set,
one per property asked for. Nothing here imports torch, a model or
koinapy until a predictor is built, so ``import constellation`` stays
light.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Any

from constellation.massspec.library.prediction.base import PROPERTIES, Predictor, Property

__all__ = ["BACKENDS", "default_predictors", "predictor"]

#: backend → model → (module, class, property).
BACKENDS: dict[str, dict[str, tuple[str, str, Property]]] = {
    "jchronologer": {
        "chronologer": ("constellation.massspec.library.jchronologer.models", "ChronologerRetentionTime", "retention_time"),
        "cartographer": ("constellation.massspec.library.jchronologer.models", "CartographerFragments", "fragments"),
        "electrician": ("constellation.massspec.library.jchronologer.models", "ElectricianChargeState", "charge_state"),
        "sculptor": ("constellation.massspec.library.jchronologer.models", "SculptorCcs", "ccs"),
    },
}


#: Koina serves any model by name: the class for each property it can predict.
KOINA = {
    "fragments": ("constellation.massspec.library.koina.predictor", "KoinaFragments"),
    "retention_time": ("constellation.massspec.library.koina.predictor", "KoinaRetentionTime"),
}


def _koina_property(model: str, prop: Property | None) -> Property:
    from constellation.massspec.library.koina.models import MS2_MODELS, RT_MODELS

    known = "fragments" if model in MS2_MODELS else "retention_time" if model in RT_MODELS else None
    if prop is not None:
        if prop not in KOINA:
            raise ValueError(f"Koina predictors here give {', '.join(KOINA)}, not {prop}")
        if known is not None and known != prop:
            raise ValueError(f"koina:{model} predicts {known}, not {prop}")
        return prop
    if known is None:
        raise ValueError(f"koina:{model} is not in the overlay (koina.models); say which property it predicts")
    return known


def _entry(spec: str, prop: Property | None = None) -> tuple[str, str, tuple[str, str, Property]]:
    backend, sep, model = spec.partition(":")
    if sep and backend == "koina" and model:
        found = _koina_property(model, prop)
        return backend, model, (*KOINA[found], found)
    if not sep or backend not in BACKENDS or model not in BACKENDS[backend]:
        known = ", ".join(f"{b}:{m}" for b, models in BACKENDS.items() for m in models)
        raise ValueError(f"unknown predictor {spec!r} (known: {known}, or koina:<model>)")
    entry = BACKENDS[backend][model]
    if prop is not None and entry[2] != prop:
        raise ValueError(f"{spec} predicts {entry[2]}, not {prop}")
    return backend, model, entry


def predictor(spec: str, *, property: Property | None = None, **options: Any) -> Predictor:
    """The predictor ``backend:model`` (``koina:<any model>`` too), built
    with ``options``: runtime settings such as ``threads``,
    ``batch_size``, ``device`` and a shared ``bundle`` for JChronologer;
    ``server`` / ``client`` / ``min_intensity`` for Koina. ``property``
    says what a Koina model the overlay does not know predicts, and is
    checked against the model otherwise."""
    backend, model, (module, name, _) = _entry(spec, property)
    cls = getattr(import_module(module), name)
    return cls(model, **options) if backend == "koina" else cls(**options)


def property_of(spec: str) -> Property:
    return _entry(spec)[2][2]


def default_predictors(
    backend: str, properties: Iterable[Property] = PROPERTIES, **options: Any
) -> dict[Property, Predictor]:
    """One predictor per property from ``backend``'s models, sharing ``options``."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (known: {', '.join(BACKENDS)})")
    wanted = set(properties)
    unknown = wanted - set(PROPERTIES)
    if unknown:
        raise ValueError(f"unknown properties {sorted(unknown)}")
    if backend == "jchronologer" and "bundle" not in options:
        from constellation.massspec.library.jchronologer.bundle import locate_bundle

        options = {**options, "bundle": locate_bundle()}
    out: dict[Property, Predictor] = {}
    for model, (_, _, prop) in BACKENDS[backend].items():
        if prop in wanted:
            out[prop] = predictor(f"{backend}:{model}", **options)
    return out
