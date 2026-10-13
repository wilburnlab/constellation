"""Request shapes for the genome modality's session endpoints.

Stdlib dataclasses (see ``viz.server.validation``): this module is
imported when the modality registers, which happens on
``import constellation.viz`` and must not need pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class SourceEntry:
    """One source row: an ``align`` / ``cluster`` output directory. ``kind``
    is auto-detected from the directory's manifest when omitted."""

    path: str
    kind: str | None = None
    label: str | None = None


@dataclass(frozen=True, kw_only=True)
class GenomeOpenRequest:
    """Body of ``POST /api/sessions/open`` for a genome session."""

    reference_handle: str
    sources: list[SourceEntry] = field(default_factory=list)
    label: str | None = None
    saved_as: str | None = None


@dataclass(frozen=True, kw_only=True)
class GenomeSaveRequest:
    """Body of ``POST /api/saved-sessions`` for a genome session."""

    label: str
    reference_handle: str
    sources: list[dict[str, Any]]
    last_viewed_locus: dict[str, Any] | None = None
    track_layout: list[dict[str, Any]] | None = None
    options: dict[str, Any] | None = None
    slug: str | None = None


__all__ = ["GenomeOpenRequest", "GenomeSaveRequest", "SourceEntry"]
