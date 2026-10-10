"""Modality registry for the viz layer.

A *modality* is one kind of browser the viz server can host: the genome
browser today, a mass-spec data browser next. Each one is a subpackage
here (``viz/modalities/<name>/``) that owns everything specific to it —
its session classes, its track kernels and their query models, its extra
endpoints — and registers a :class:`Modality` descriptor so the generic
server core can open its sessions without knowing what they contain.

Adding a modality is: one subpackage here that calls
:func:`register_modality`, an import of it in ``constellation.viz``, a
``src/modalities/<name>/`` folder in the frontend, and a ``viz <name>``
CLI subcommand.

**Import boundary.** The viz core imports no domain module. A modality
subpackage may import a small, pure-function surface from one, and must
declare it as ``DOMAIN_IMPORTS`` in its ``__init__``; the imports
themselves go inside function bodies so that ``import constellation.viz``
stays cheap and works on a base install. ``tests/test_viz_import_boundary.py``
enforces all of this.

This module must import nothing beyond the standard library:
``constellation.viz`` imports it eagerly, and that has to work without
the ``[viz]`` extras (fastapi, pydantic).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


#: Modality assumed when a request or a saved-session file does not name
#: one: clients and files from before modalities existed carry no
#: ``modality`` and are genome-browser ones.
DEFAULT_MODALITY = "genome"


def _no_routers() -> Sequence[Any]:
    return ()


@dataclass(frozen=True)
class Modality:
    """How the server core reaches one modality."""

    #: Registry key; also the ``modality`` value on sessions, kernels,
    #: request bodies and saved-session files.
    name: str

    #: Stdlib dataclass describing the body of ``POST /api/sessions/open``
    #: for this modality. The endpoint validates the request against it
    #: (HTTP 422 on failure) and passes the instance to ``open_session``.
    open_request: type

    #: Build a session from a validated ``open_request`` instance. Raises
    #: ``ValueError`` for an unusable request (HTTP 400).
    open_session: Callable[[Any], Any]

    #: Describe a candidate source directory for the entry form (its kind,
    #: plus whatever lets the form warn about a mismatch). Raises
    #: ``ValueError`` when the directory is not a usable source.
    inspect_source: Callable[[Path], Mapping[str, Any]]

    #: Rebuild a session from a ``viz.sessions.SavedSession``.
    session_from_saved: Callable[[Any], Any]

    #: Stdlib dataclass describing the body of ``POST /api/saved-sessions``.
    save_request: type

    #: Zero-argument factory returning this modality's own FastAPI routers.
    #: It must do its fastapi import inside the call: the descriptor is
    #: built at ``import constellation.viz`` time, which has to work
    #: without fastapi installed.
    routers: Callable[[], Sequence[Any]] = _no_routers


_REGISTRY: dict[str, Modality] = {}


def register_modality(modality: Modality) -> Modality:
    """Register a modality by name. Re-registration raises."""
    if not modality.name:
        raise ValueError("modality must have a non-empty name")
    if modality.name in _REGISTRY:
        raise ValueError(f"modality {modality.name!r} already registered")
    _REGISTRY[modality.name] = modality
    return modality


def get_modality(name: str) -> Modality:
    if name not in _REGISTRY:
        raise KeyError(
            f"modality {name!r} not registered (known: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[name]


def registered_modalities() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "DEFAULT_MODALITY",
    "Modality",
    "get_modality",
    "register_modality",
    "registered_modalities",
]
