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
- `GET /api/sessions/{session_id}/search`      → annotation-feature search
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query, Request
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


# v1 hardcodes (table=annotation, column=name); v2 (next PR) will accept
# `table` + `column` query params for an advanced column-picker mode.
# Bindings the v1 endpoint scans, in display-priority order — reference
# annotations land before constellation-derived ones because users
# searching by gene name typically want the curated hit first.
_ANNOTATION_BINDINGS: tuple[tuple[str, str], ...] = (
    ("reference_annotation", "reference"),
    ("derived_annotation", "derived"),
)


@router.get("/{session_id}/search")
def search_features(
    session_id: str,
    request: Request,
    q: str = Query(default="", description="Substring to match (case-insensitive)."),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    """Case-insensitive substring match against annotation features.

    Scans both `reference_annotation/features.parquet` and
    `derived_annotation/features.parquet` when present; tags each hit
    with a `source` field so the client can show which annotation
    bundle produced it. A purely-numeric `q` also matches `feature_id`
    exactly. Returns up to `limit` rows, reference hits first.
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
    for slot, source in _ANNOTATION_BINDINGS:
        if len(out) >= limit:
            break
        annotation_dir = getattr(session, slot, None)
        if annotation_dir is None:
            continue
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
                    "source": source,
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
