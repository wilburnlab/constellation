"""Modality-neutral session contracts for the viz server.

A *session* is what one browser view looks at: a set of attached data
sources plus whatever its modality needs to interpret them (a reference
genome for the genome browser; an acquisition for a mass-spec browser).

The server core — the app's session registry, the session and track
endpoints, the binding cache — depends only on the two Protocols here.
Each modality supplies its own concrete, typed classes
(:class:`constellation.viz.modalities.genome.session.GenomeSession`, ...)
and registers how to build them with :mod:`constellation.viz.modalities`.

Protocols rather than base dataclasses on purpose: the concrete classes
are ``frozen=True, slots=True`` dataclasses, and zero-argument
``super()`` raises inside a slots-dataclass subclass on Python 3.12, so a
base class whose methods subclasses extend is a trap.

The two id helpers live here because their output is persisted — saved
``[[track_layout]]`` entries and the browser's ``localStorage`` keys are
built from them — and every modality must derive ids the same way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol


class SourceLike(Protocol):
    """One data source attached to a session (an output directory)."""

    path: Path
    kind: str
    label: str

    @property
    def source_id(self) -> str:
        """Stable id for this source; see :func:`derive_source_id`."""
        ...


class SessionLike(Protocol):
    """What the server core needs from a session, whatever its modality."""

    #: Registry key of the owning modality (``"genome"``, ...). Kernels are
    #: only ever asked about sessions of their own modality.
    modality: str
    session_id: str
    label: str
    sources: tuple[Any, ...]
    warnings: tuple[str, ...]
    saved_as: str | None

    def with_sources(self, sources: Iterable[dict[str, Any]]) -> "SessionLike":
        """Rebuild with a new source list, keeping ``session_id``."""
        ...

    def summary(self) -> dict[str, Any]:
        """Small JSON record for ``GET /api/sessions``."""
        ...

    def to_manifest(self) -> dict[str, Any]:
        """Full JSON description for ``GET /api/sessions/{id}/manifest``."""
        ...


def derive_source_id(path: Path, kind: str) -> str:
    """Stable client-side identifier for a source, from ``(path, kind)``.

    Survives add/remove cycles, so per-binding layout state keyed by it
    (visibility, display order, height) stays valid across session
    rebuilds. The value is persisted; do not change the derivation.
    """
    payload = f"{path}|{kind}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).hexdigest()
    return f"src-{digest}"


def derive_session_id(anchor: Path | str, label: str) -> str:
    """Stable, URL-safe short id for an ``(anchor, label)`` pair.

    ``anchor`` is whatever fixes the session's identity for its modality
    (the reference release directory for the genome browser). Deterministic,
    so a rebuilt session can be swapped into the registry in place.
    """
    payload = f"{anchor}|{label}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).hexdigest()
    return f"{slugify(label)}-{digest}"


def slugify(s: str) -> str:
    out: list[str] = []
    for ch in s.lower():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "session"


__all__ = [
    "SessionLike",
    "SourceLike",
    "derive_session_id",
    "derive_source_id",
    "slugify",
]
