"""Session-discovery endpoints.

A viz server holds a small in-memory registry of `Session` objects keyed
by `session_id`. The registry is populated at app startup from the CLI's
`--session DIR` argument (PR 1 mounts exactly one); the dashboard PR 2
extends this with a discovery walk over a parent directory.

Endpoints:

- `GET /api/sessions`                          → list of session summaries
- `GET /api/sessions/{session_id}/manifest`    → resolved manifest JSON
- `GET /api/sessions/{session_id}/contigs`     → CONTIG_TABLE summary
"""

from __future__ import annotations

import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Request


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions(request: Request) -> list[dict]:
    """Return a small summary record per registered session."""
    sessions: dict = request.app.state.sessions
    return [
        {
            "session_id": s.session_id,
            "label": s.label,
            "root": str(s.root),
            "stages_present": s.stages_present(),
        }
        for s in sessions.values()
    ]


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
