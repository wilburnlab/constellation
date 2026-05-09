"""Track endpoints — list, metadata, and Arrow IPC data streams.

For each registered kernel, the server exposes:

- `GET /api/tracks?session=<id>`
    → list of `{kind, binding_id, label, ...}` for every binding the
      kernels can produce against the named session.
- `GET /api/tracks/{kind}/metadata?session=<id>&binding=<binding_id>`
    → the per-binding metadata JSON the renderer uses to set up the
      track (palette, samples, height, ...).
- `GET /api/tracks/{kind}/data?session=<id>&binding=<binding_id>&contig=&start=&end=...`
    → Arrow IPC stream. The `X-Track-Mode` response header carries the
      resolved mode (`vector` or `hybrid`) so the renderer branches
      without inspecting the payload schema.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from constellation.viz.server.arrow_stream import batches_to_response
from constellation.viz.server.session import Session
from constellation.viz.tracks.base import (
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackBinding,
    TrackQuery,
    get_kernel,
    registered_kinds,
)


router = APIRouter(prefix="/api/tracks", tags=["tracks"])


# In-process cache: each session's discovered bindings are computed
# lazily on first reference and re-used. Keyed by `(session_id, kind)`.
# Discovery is cheap (filesystem stat + small parquet reads) but bindings
# carry resolved Path objects that don't need to be recomputed per query.
def _bindings_for(
    session: Session, kind: str, cache: dict[tuple[str, str], list[TrackBinding]]
) -> list[TrackBinding]:
    key = (session.session_id, kind)
    if key not in cache:
        cache[key] = get_kernel(kind).discover(session)
    return cache[key]


def _find_binding(
    session: Session,
    kind: str,
    binding_id: str,
    cache: dict[tuple[str, str], list[TrackBinding]],
) -> TrackBinding | None:
    for binding in _bindings_for(session, kind, cache):
        if binding.binding_id == binding_id:
            return binding
    return None


@router.get("")
def list_tracks(session: str, request: Request) -> list[dict]:
    """List all bindings the registered kernels can produce against the
    named session. The frontend uses this to populate the "add track"
    picker."""
    sessions: dict = request.app.state.sessions
    cache: dict = request.app.state.track_bindings_cache
    s = sessions.get(session)
    if s is None:
        raise HTTPException(404, f"unknown session_id: {session}")

    out: list[dict] = []
    for kind in registered_kinds():
        for binding in _bindings_for(s, kind, cache):
            out.append(
                {
                    "kind": kind,
                    "binding_id": binding.binding_id,
                    "label": binding.label,
                }
            )
    return out


@router.get("/{kind}/metadata")
def get_metadata(
    kind: str,
    session: str,
    binding: str,
    request: Request,
) -> dict:
    sessions: dict = request.app.state.sessions
    cache: dict = request.app.state.track_bindings_cache
    s = sessions.get(session)
    if s is None:
        raise HTTPException(404, f"unknown session_id: {session}")
    try:
        kernel = get_kernel(kind)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    track_binding = _find_binding(s, kind, binding, cache)
    if track_binding is None:
        raise HTTPException(404, f"binding {binding!r} not found for kind {kind!r}")
    return kernel.metadata(track_binding)


@router.get("/{kind}/data")
def get_data(
    kind: str,
    session: str,
    binding: str,
    contig: str,
    start: int = Query(..., ge=0),
    end: int = Query(..., ge=0),
    samples: list[str] | None = Query(None),
    viewport_px: int = Query(1200, ge=1, le=8192),
    max_glyphs: int = Query(50_000, ge=100),
    force: str | None = Query(None, pattern="^(vector|hybrid)$"),
    request: Request = None,  # type: ignore[assignment]
) -> StreamingResponse:
    """Stream Arrow IPC for the requested track + viewport."""
    if end <= start:
        raise HTTPException(400, "end must be greater than start")

    sessions: dict = request.app.state.sessions
    cache: dict = request.app.state.track_bindings_cache
    s = sessions.get(session)
    if s is None:
        raise HTTPException(404, f"unknown session_id: {session}")
    try:
        kernel = get_kernel(kind)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    track_binding = _find_binding(s, kind, binding, cache)
    if track_binding is None:
        raise HTTPException(404, f"binding {binding!r} not found for kind {kind!r}")

    forced = ThresholdDecision(force) if force else None
    query = TrackQuery(
        contig=contig,
        start=int(start),
        end=int(end),
        samples=tuple(samples) if samples else (),
        viewport_px=int(viewport_px),
        max_glyphs=int(max_glyphs),
        force=forced,
    )

    mode = kernel.threshold(track_binding, query)
    schema = kernel.schema if mode is ThresholdDecision.VECTOR else HYBRID_SCHEMA
    batches = kernel.fetch(track_binding, query, mode)
    return batches_to_response(
        schema,
        batches,
        headers={
            "X-Track-Mode": mode.value,
            "X-Track-Kind": kind,
        },
    )
