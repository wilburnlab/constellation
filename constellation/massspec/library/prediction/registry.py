"""Predictors by name: ``backend:model`` → a predictor, imported lazily.

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


def _entry(spec: str) -> tuple[str, str, tuple[str, str, Property]]:
    backend, sep, model = spec.partition(":")
    if not sep or backend not in BACKENDS or model not in BACKENDS[backend]:
        known = ", ".join(f"{b}:{m}" for b, models in BACKENDS.items() for m in models)
        raise ValueError(f"unknown predictor {spec!r} (known: {known})")
    return backend, model, BACKENDS[backend][model]


def predictor(spec: str, **options: Any) -> Predictor:
    """The predictor ``backend:model``, built with ``options`` (runtime
    settings such as ``threads``, ``batch_size``, ``device``; a shared
    ``bundle`` for JChronologer)."""
    _, _, (module, name, _) = _entry(spec)
    return getattr(import_module(module), name)(**options)


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
