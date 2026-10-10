"""Track endpoints — list, metadata, and Arrow IPC data streams.

For each registered kernel, the server exposes:

- `GET /api/tracks?session=<id>`
    → list of `{kind, binding_id, label, ...}` for every binding the
      kernels can produce against the named session.
- `GET /api/tracks/{kind}/metadata?session=<id>&binding=<binding_id>`
    → the per-binding metadata JSON the renderer uses to set up the
      track (palette, samples, height, ...).
- `GET /api/tracks/{kind}/data?session=<id>&binding=<binding_id>&<query fields>`
    → Arrow IPC stream. The query fields are whatever the kernel's
      `query_model` declares (`contig`, `start`, `end`, ... for the
      genome kernels); nothing kernel-specific is named here. The
      `X-Track-Mode` response header carries the resolved mode (`vector`
      or `hybrid`) so the renderer branches without inspecting the
      payload schema.
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from typing import Any, get_origin, get_type_hints

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import QueryParams

from constellation.viz.server.arrow_stream import batches_to_response
from constellation.viz.server.session import SessionLike
from constellation.viz.server.validation import validate
from constellation.viz.tracks.base import (
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
    session: SessionLike, kind: str, cache: dict[tuple[str, str], list[TrackBinding]]
) -> list[TrackBinding]:
    kernel = get_kernel(kind)
    # A kernel only understands sessions of its own modality: it reads
    # modality-specific attributes off the session in ``discover``. For
    # any other session it simply has no bindings, on every route.
    if kernel.modality != session.modality:
        return []
    key = (session.session_id, kind)
    if key not in cache:
        cache[key] = kernel.discover(session)
    return cache[key]


def _find_binding(
    session: SessionLike,
    kind: str,
    binding_id: str,
    cache: dict[tuple[str, str], list[TrackBinding]],
) -> TrackBinding | None:
    for binding in _bindings_for(session, kind, cache):
        if binding.binding_id == binding_id:
            return binding
    return None


def invalidate_binding_cache(
    cache: dict[tuple[str, str], list[TrackBinding]], session_id: str
) -> None:
    """Evict every ``(session_id, kind)`` entry for the given session.

    Called whenever a session's source list mutates (open, add-source,
    delete-source). The next /api/tracks request re-runs discovery
    against the rebuilt session.
    """
    for key in list(cache):
        if key[0] == session_id:
            del cache[key]


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
                    "source_id": binding.config.get("source_id"),
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


#: Query-string names the data endpoint reads itself; a query model must
#: not declare a field with either name.
_RESERVED = frozenset({"session", "binding"})


@lru_cache(maxsize=None)
def _query_plan(model: type[TrackQuery]) -> tuple[tuple[str, ...], frozenset[str]]:
    """``(field names, names of the sequence-typed fields)`` for a query
    model. Cached per class; kernels (and so models) can register late."""
    names = tuple(f.name for f in fields(model))
    clash = _RESERVED.intersection(names)
    if clash:
        raise TypeError(
            f"{model.__name__} declares {sorted(clash)}, which the data "
            f"endpoint reserves"
        )
    hints = get_type_hints(model)
    sequences = frozenset(
        name for name in names if get_origin(hints.get(name)) in (tuple, list)
    )
    return names, sequences


def _parse_query(model: type[TrackQuery], params: QueryParams) -> TrackQuery:
    """Build the kernel's query from the request's query string.

    Only the fields the model declares are read — any other parameter is
    ignored, as FastAPI ignores undeclared ones. A repeated parameter
    feeds a sequence field in full and a scalar field with its last
    value. Type and range failures are HTTP 422; a failed cross-field
    ``check()`` is HTTP 400.
    """
    names, sequences = _query_plan(model)
    raw: dict[str, Any] = {}
    for name in names:
        if name not in params:
            continue
        raw[name] = params.getlist(name) if name in sequences else params[name]
    query = validate(model, raw, where="query")
    try:
        query.check()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return query


@router.get("/{kind}/data")
def get_data(
    kind: str,
    session: str,
    binding: str,
    request: Request,
) -> StreamingResponse:
    """Stream Arrow IPC for the requested track + view."""
    try:
        kernel = get_kernel(kind)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    # The kernel decides what its query looks like, so it is resolved
    # before the query string can be read.
    query = _parse_query(kernel.query_model, request.query_params)

    sessions: dict = request.app.state.sessions
    cache: dict = request.app.state.track_bindings_cache
    s = sessions.get(session)
    if s is None:
        raise HTTPException(404, f"unknown session_id: {session}")
    track_binding = _find_binding(s, kind, binding, cache)
    if track_binding is None:
        raise HTTPException(404, f"binding {binding!r} not found for kind {kind!r}")

    mode = kernel.threshold(track_binding, query)
    schema = kernel.schema_for(query, mode)
    batches = kernel.fetch(track_binding, query, mode)
    return batches_to_response(
        schema,
        batches,
        headers={
            "X-Track-Mode": mode.value,
            "X-Track-Kind": kind,
            **kernel.response_headers(query, mode),
        },
    )
