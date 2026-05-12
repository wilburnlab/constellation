"""Session-discovery endpoints.

A viz server holds a small in-memory registry of `Session` objects keyed
by `session_id`. The registry is populated at app startup from the CLI's
`--session DIR` argument (PR 1 mounts exactly one); the dashboard boots
empty and registers sessions on demand via `POST /api/sessions`.

Endpoints:

- `GET /api/sessions`                          → list of session summaries
- `POST /api/sessions {path}`                  → register a session by directory path
- `GET /api/sessions/{session_id}/manifest`    → resolved manifest JSON
- `GET /api/sessions/{session_id}/contigs`     → CONTIG_TABLE summary
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constellation.viz.server.session import Session


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _summarize(session: Session) -> dict:
    return {
        "session_id": session.session_id,
        "label": session.label,
        "root": str(session.root),
        "stages_present": session.stages_present(),
    }


class RegisterSessionRequest(BaseModel):
    path: str


@router.get("")
def list_sessions(request: Request) -> list[dict]:
    """Return a small summary record per registered session."""
    sessions: dict = request.app.state.sessions
    return [_summarize(s) for s in sessions.values()]


@router.post("", status_code=201)
def register_session(body: RegisterSessionRequest, request: Request) -> dict:
    """Register (or re-register) a session by directory path.

    The path is resolved via `Session.from_root`, which prefers an
    explicit `session.toml` and falls back to a directory-walk discovery.
    Re-POSTing the same path is idempotent — `session_id` is derived from
    the resolved root, so the registry entry is overwritten in place. Any
    cached track bindings for that `session_id` are evicted so the next
    discovery call reflects the freshly-registered session contents.
    """
    raw = (body.path or "").strip()
    if not raw:
        raise HTTPException(400, "path must be a non-empty string")
    try:
        session = Session.from_root(Path(raw).expanduser())
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    sessions: dict = request.app.state.sessions
    sessions[session.session_id] = session

    cache: dict = request.app.state.track_bindings_cache
    for key in list(cache):
        if key[0] == session.session_id:
            del cache[key]

    return _summarize(session)


@router.get("/{session_id}/manifest")
def get_manifest(session_id: str, request: Request) -> dict:
    sessions: dict = request.app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    return session.to_manifest()


@router.get("/{session_id}/contigs")
def get_contigs(session_id: str, request: Request) -> list[dict]:
    """Return `[{contig, length}]` for the session's reference genome.

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
