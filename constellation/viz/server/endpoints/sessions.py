"""Session endpoints — register, inspect, and search.

The dashboard registers a session via ``POST /api/sessions/open`` with a
reference handle plus a list of data-source directories. The legacy
``POST /api/sessions {path}`` endpoint (which assumed a session-root
directory layout) was removed in the reference-cache-first cutover.

Endpoints:

- ``GET /api/sessions``                            → list summaries
- ``POST /api/sessions/open``                      → register a new session
- ``POST /api/sessions/inspect-source``            → auto-detect kind + handle
- ``GET /api/sessions/{session_id}/manifest``      → resolved manifest JSON
- ``GET /api/sessions/{session_id}/contigs``       → contig list for the
                                                     reference genome
- ``GET /api/sessions/{session_id}/search``        → annotation-feature search
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from constellation.viz.server.endpoints.tracks import invalidate_binding_cache
from constellation.viz.server.session import Session


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# ----------------------------------------------------------------------
# Summaries + manifest
# ----------------------------------------------------------------------


def _summarize(session: Session) -> dict:
    return {
        "session_id": session.session_id,
        "label": session.label,
        "reference_handle": session.reference_handle,
        "reference_path": str(session.reference_path),
        "n_sources": len(session.sources),
        "stages_present": session.stages_present(),
        "warnings": list(session.warnings),
        "saved_as": session.saved_as,
    }


@router.get("")
def list_sessions(request: Request) -> list[dict]:
    """Return a small summary record per registered session."""
    sessions: dict = request.app.state.sessions
    return [_summarize(s) for s in sessions.values()]


# ----------------------------------------------------------------------
# Open (new reference-cache-first flow)
# ----------------------------------------------------------------------


class SourceEntry(BaseModel):
    path: str
    kind: str | None = None
    label: str | None = None


class OpenSessionRequest(BaseModel):
    reference_handle: str
    sources: list[SourceEntry] = []
    label: str | None = None
    saved_as: str | None = None


@router.post("/open", status_code=201)
def open_session(body: OpenSessionRequest, request: Request) -> dict:
    """Construct a session from a reference handle + data-source list.

    The reference handle is resolved against the per-user reference
    cache; each source's ``manifest.json`` is read to populate the
    per-stage artifact slots. Sources whose ``assembly_accession``
    differs from the chosen reference emit an entry in the response's
    ``warnings`` list — the dashboard surfaces those inline but does not
    block the open.
    """
    try:
        session = Session.open(
            reference_handle=body.reference_handle,
            sources=[s.model_dump() for s in body.sources],
            label=body.label,
            saved_as=body.saved_as,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    sessions: dict = request.app.state.sessions
    sessions[session.session_id] = session
    invalidate_binding_cache(
        request.app.state.track_bindings_cache, session.session_id
    )

    return _summarize(session)


# ----------------------------------------------------------------------
# Runtime source mutation — add / remove a source on a live session
# ----------------------------------------------------------------------


class AddSourceRequest(BaseModel):
    path: str
    kind: str | None = None
    label: str | None = None


def _sources_to_entries(sources: tuple) -> list[dict[str, Any]]:
    """Turn a session's frozen source tuple back into a list of input
    dicts suitable for :meth:`Session.with_sources`."""
    return [
        {"path": str(src.path), "kind": src.kind, "label": src.label}
        for src in sources
    ]


def _replace_session(request: Request, session: Session) -> None:
    """Atomically install a rebuilt Session in the registry and evict
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

    The source path's ``manifest.json`` supplies the kind + assembly
    automatically (mirrors ``POST /api/sessions/inspect-source``). The
    session is rebuilt via :meth:`Session.with_sources`; the
    ``session_id`` is preserved (deterministic over reference + label)
    so clients don't need to re-bind. The next ``GET /api/tracks?...``
    call returns the bindings for the new source.
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
# Inspect-source — autofill the multi-row form's kind/handle fields
# ----------------------------------------------------------------------


class InspectSourceRequest(BaseModel):
    path: str


@router.post("/inspect-source")
def inspect_source(body: InspectSourceRequest) -> dict[str, Any]:
    """Read a source dir's ``manifest.json`` and return key descriptive
    fields. Used by the dashboard form to auto-detect kind and surface
    reference-handle mismatch warnings before the user submits.
    """
    from constellation.sequencing.transcriptome.manifest import (
        read_manifest_dir,
    )

    path = Path(body.path).expanduser()
    if not path.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    try:
        manifest = read_manifest_dir(path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "path": str(path.resolve()),
        "kind": manifest.kind,
        "reference_handle": manifest.reference_handle,
        "reference_path": getattr(manifest, "reference_path", None),
        "assembly_accession": manifest.assembly_accession,
        "samples": list(manifest.samples or ()),
    }


# ----------------------------------------------------------------------
# Manifest + contigs (existing endpoints, repointed at new shape)
# ----------------------------------------------------------------------


@router.get("/{session_id}/manifest")
def get_manifest(session_id: str, request: Request) -> dict:
    sessions: dict = request.app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    return session.to_manifest()


@router.get("/{session_id}/contigs")
def get_contigs(session_id: str, request: Request) -> list[dict]:
    """Return ``[{contig, length}]`` for the session's reference genome.

    The frontend uses this to populate the locus picker. Returns 404 if
    the session has no reference attached.
    """
    sessions: dict = request.app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    if session.reference_genome is None:
        raise HTTPException(404, "session has no reference genome attached")
    contigs_path = session.reference_genome / "contigs.parquet"
    if not contigs_path.exists():
        raise HTTPException(
            500,
            f"reference genome ParquetDir is incomplete: {contigs_path} missing",
        )
    table = pq.read_table(contigs_path, columns=["contig_id", "name", "length"])
    out: list[dict] = []
    for row in table.to_pylist():
        out.append(
            {
                "contig_id": row["contig_id"],
                "name": row["name"],
                "length": row["length"],
            }
        )
    return out


# ----------------------------------------------------------------------
# Feature search (unchanged — uses the new Session shape's reference
# slots and per-source derived annotations)
# ----------------------------------------------------------------------


@router.get("/{session_id}/search")
def search_features(
    session_id: str,
    request: Request,
    q: str = Query(default="", description="Substring to match (case-insensitive)."),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    """Case-insensitive substring match against annotation features.

    Scans the curated ``reference_annotation/features.parquet`` plus
    each align source's ``derived_annotation/features.parquet`` when
    present; tags each hit with a ``source`` field so the client can
    show which annotation bundle produced it. A purely-numeric ``q``
    also matches ``feature_id`` exactly. Returns up to ``limit`` rows,
    reference hits first.
    """
    sessions: dict = request.app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")

    query = (q or "").strip()
    if not query:
        return []
    if session.reference_genome is None:
        return []

    contig_name_by_id = _load_contig_name_map(session.reference_genome)
    if not contig_name_by_id:
        return []

    numeric_id: int | None = None
    try:
        numeric_id = int(query)
    except ValueError:
        numeric_id = None

    out: list[dict] = []
    # Reference annotation first (curated wins display order), then each
    # source's derived annotation in the order the user added them.
    ordered: list[tuple[Path | None, str]] = [
        (session.reference_annotation, "reference"),
    ]
    for src in session.sources:
        if src.derived_annotation is not None:
            ordered.append((src.derived_annotation, f"derived ({src.label})"))

    for annotation_dir, source_tag in ordered:
        if annotation_dir is None:
            continue
        if len(out) >= limit:
            break
        features_path = annotation_dir / "features.parquet"
        if not features_path.exists():
            continue
        remaining = limit - len(out)
        rows = _search_features_in_parquet(
            features_path=features_path,
            query=query,
            numeric_id=numeric_id,
            limit=remaining,
        )
        for row in rows:
            contig_name = contig_name_by_id.get(int(row["contig_id"]))
            if contig_name is None:
                continue
            out.append(
                {
                    "feature_id": int(row["feature_id"]),
                    "name": row["name"],
                    "type": row["type"],
                    "strand": row["strand"],
                    "contig_name": contig_name,
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "source": source_tag,
                }
            )
            if len(out) >= limit:
                break
    return out


def _load_contig_name_map(genome_dir: Path) -> dict[int, str]:
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return {}
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    ids = table.column("contig_id").to_pylist()
    names = table.column("name").to_pylist()
    return {int(i): str(n) for i, n in zip(ids, names) if i is not None and n is not None}


def _search_features_in_parquet(
    *,
    features_path: Path,
    query: str,
    numeric_id: int | None,
    limit: int,
) -> list[dict]:
    """Run the substring (+ optional feature_id) filter against one
    annotation parquet, returning at most ``limit`` rows as plain dicts."""
    dataset = pa_ds.dataset(str(features_path), format="parquet")
    name_match = pc.match_substring(pc.field("name"), query, ignore_case=True)
    predicate = name_match
    if numeric_id is not None:
        predicate = predicate | (
            pc.field("feature_id") == pa.scalar(numeric_id, pa.int64())
        )
    scanner = dataset.scanner(
        columns=["feature_id", "contig_id", "start", "end", "strand", "type", "name"],
        filter=predicate,
    )
    table = scanner.head(limit)
    if table.num_rows == 0:
        return []
    return table.to_pylist()
