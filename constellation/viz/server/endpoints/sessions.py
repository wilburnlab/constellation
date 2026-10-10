"""Session endpoints — open, inspect, and mutate sessions of any modality.

These routes know nothing about what a session contains. A request names
its modality (``"genome"`` when omitted, which is what every existing
client sends); the route looks that modality up in
``constellation.viz.modalities`` and hands the work to it. Endpoints
that only make sense for one modality live with that modality — the
genome browser's ``contigs`` and feature ``search`` are in
``viz.modalities.genome.endpoints``.

Endpoints:

- ``GET /api/sessions``                                  → list summaries
- ``POST /api/sessions/open``                            → register a new session
- ``POST /api/sessions/inspect-source``                  → describe a candidate source
- ``GET /api/sessions/{session_id}/manifest``            → resolved manifest JSON
- ``POST /api/sessions/{session_id}/sources``            → attach a source
- ``DELETE /api/sessions/{session_id}/sources/{id}``     → detach a source
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel

from constellation.viz.modalities import Modality, get_modality
from constellation.viz.server.endpoints.tracks import invalidate_binding_cache
from constellation.viz.server.session import SessionLike
from constellation.viz.server.validation import validate


router = APIRouter(prefix="/api/sessions", tags=["sessions"])

#: Modality assumed when a request does not name one. Clients written
#: before modalities existed send no ``modality`` field.
DEFAULT_MODALITY = "genome"


def _modality_or_400(name: str) -> Modality:
    try:
        return get_modality(name)
    except KeyError as exc:
        raise HTTPException(400, str(exc.args[0])) from exc


# ----------------------------------------------------------------------
# Summaries + manifest
# ----------------------------------------------------------------------


@router.get("")
def list_sessions(request: Request) -> list[dict]:
    """Return a small summary record per registered session."""
    sessions: dict = request.app.state.sessions
    return [s.summary() for s in sessions.values()]


@router.get("/{session_id}/manifest")
def get_manifest(session_id: str, request: Request) -> dict:
    sessions: dict = request.app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    return session.to_manifest()


# ----------------------------------------------------------------------
# Open
# ----------------------------------------------------------------------


@router.post("/open", status_code=201)
def open_session(request: Request, body: dict[str, Any] = Body(...)) -> dict:
    """Construct and register a session.

    ``body["modality"]`` selects the modality; the rest of the body is
    validated against that modality's ``open_request`` model (HTTP 422
    when it does not fit) and handed to its ``open_session``, which
    raises ``ValueError`` for a request it cannot satisfy (HTTP 400).

    For the genome browser the body is ``{reference_handle, sources,
    label?, saved_as?}``: the handle is resolved against the per-user
    reference cache and each source's ``manifest.json`` is read. Sources
    whose assembly differs from the chosen reference add an entry to the
    response's ``warnings`` — surfaced by the dashboard, not blocking.
    """
    modality = _modality_or_400(str(body.get("modality") or DEFAULT_MODALITY))
    open_request = validate(modality.open_request, body, where="body")
    try:
        session = modality.open_session(open_request)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    _replace_session(request, session)
    return session.summary()


# ----------------------------------------------------------------------
# Runtime source mutation — add / remove a source on a live session
# ----------------------------------------------------------------------


class AddSourceRequest(BaseModel):
    path: str
    kind: str | None = None
    label: str | None = None


def _sources_to_entries(sources: tuple) -> list[dict[str, Any]]:
    """Turn a session's frozen source tuple back into a list of input
    dicts suitable for the session's ``with_sources``."""
    return [
        {"path": str(src.path), "kind": src.kind, "label": src.label}
        for src in sources
    ]


def _replace_session(request: Request, session: SessionLike) -> None:
    """Atomically install a (re)built session in the registry and evict
    the per-kind binding cache for that ``session_id``."""
    request.app.state.sessions[session.session_id] = session
    invalidate_binding_cache(
        request.app.state.track_bindings_cache, session.session_id
    )


@router.post("/{session_id}/sources", status_code=201)
def add_source(
    session_id: str, body: AddSourceRequest, request: Request
) -> dict:
    """Append a data source to a live session.

    The session is rebuilt via its ``with_sources``, which validates the
    new source the same way opening does; the ``session_id`` is
    preserved (it is derived deterministically) so clients don't need to
    re-bind. The next ``GET /api/tracks?...`` call returns the bindings
    for the new source.
    """
    sessions: dict = request.app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    entries = _sources_to_entries(session.sources)
    entries.append(
        {"path": body.path, "kind": body.kind, "label": body.label}
    )
    try:
        rebuilt = session.with_sources(entries)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _replace_session(request, rebuilt)
    return rebuilt.to_manifest()


@router.delete("/{session_id}/sources/{source_id}")
def delete_source(session_id: str, source_id: str, request: Request) -> dict:
    """Remove the source with the given ``source_id`` from a live session.

    Returns the rebuilt session manifest. 404 if either the session or
    the source is unknown.
    """
    sessions: dict = request.app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    kept = [src for src in session.sources if src.source_id != source_id]
    if len(kept) == len(session.sources):
        raise HTTPException(
            404, f"unknown source_id {source_id!r} on session {session_id}"
        )
    entries = _sources_to_entries(tuple(kept))
    try:
        rebuilt = session.with_sources(entries)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _replace_session(request, rebuilt)
    return rebuilt.to_manifest()


# ----------------------------------------------------------------------
# Inspect-source — autofill an entry form's per-source fields
# ----------------------------------------------------------------------


class InspectSourceRequest(BaseModel):
    path: str
    modality: str = DEFAULT_MODALITY


@router.post("/inspect-source")
def inspect_source(body: InspectSourceRequest) -> dict[str, Any]:
    """Describe a candidate source directory before the user submits.

    The modality reads whatever identifies the directory (for the genome
    browser, its ``manifest.json``) and returns its kind plus the fields
    the entry form uses to warn about a mismatch.
    """
    modality = _modality_or_400(body.modality)
    path = Path(body.path).expanduser()
    if not path.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    try:
        return dict(modality.inspect_source(path))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
