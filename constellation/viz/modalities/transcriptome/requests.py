"""Request shapes for the transcriptome modality's session endpoints.

Stdlib dataclasses (see ``viz.server.validation``): imported when the
modality registers, on ``import constellation.viz``, which must not need
pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class SourceEntry:
    """One source: an EM run directory, and optionally where its
    demultiplexing output is (for sample names)."""

    path: str
    kind: str | None = None
    label: str | None = None
    demux_dir: str | None = None


@dataclass(frozen=True, kw_only=True)
class TranscriptomeOpenRequest:
    """Body of ``POST /api/sessions/open`` for a transcriptome session."""

    sources: list[SourceEntry] = field(default_factory=list)
    label: str | None = None
    saved_as: str | None = None


@dataclass(frozen=True, kw_only=True)
class TranscriptomeSaveRequest:
    """Body of ``POST /api/saved-sessions`` for a transcriptome session."""

    label: str
    sources: list[dict[str, Any]]
    track_layout: list[dict[str, Any]] | None = None
    options: dict[str, Any] | None = None
    slug: str | None = None


__all__ = ["SourceEntry", "TranscriptomeOpenRequest", "TranscriptomeSaveRequest"]
