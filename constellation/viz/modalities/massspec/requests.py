"""Request shapes for the mass-spec modality's session endpoints.

Stdlib dataclasses (see ``viz.server.validation``): this module is
imported when the modality registers, which happens on
``import constellation.viz`` and must not need pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class SourceEntry:
    """One source: a converted acquisition (bundle) directory. ``kind`` is
    read from the directory's manifest when omitted."""

    path: str
    kind: str | None = None
    label: str | None = None


@dataclass(frozen=True, kw_only=True)
class MassSpecOpenRequest:
    """Body of ``POST /api/sessions/open`` for a mass-spec session."""

    sources: list[SourceEntry] = field(default_factory=list)
    label: str | None = None
    saved_as: str | None = None


@dataclass(frozen=True, kw_only=True)
class MassSpecSaveRequest:
    """Body of ``POST /api/saved-sessions`` for a mass-spec session."""

    label: str
    sources: list[dict[str, Any]]
    track_layout: list[dict[str, Any]] | None = None
    options: dict[str, Any] | None = None
    slug: str | None = None


__all__ = ["MassSpecOpenRequest", "MassSpecSaveRequest", "SourceEntry"]
