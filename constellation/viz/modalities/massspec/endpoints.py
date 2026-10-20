"""Mass-spec-only endpoints.

The session and track endpoints in ``viz.server.endpoints`` are
modality-neutral. These describe a mass-spec session's acquisitions and
serve what a browser navigates by:

- ``GET /api/massspec/sessions/{sid}/acquisitions``                 → each run, described
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/windows``   → its DIA isolation windows
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/scans``     → its scan table (Arrow)
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/scans/{n}`` → one scan's full header

All under ``/api/massspec/``: the genome modality's routes are mounted
first and own paths such as ``/api/sessions/{id}/search``.

Imported only through the modality's ``routers()`` factory — it needs
fastapi, which ``import constellation.viz`` must not.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
from fastapi import APIRouter, HTTPException, Query, Request

from constellation.viz.modalities.massspec.cache import scan_table_for
from constellation.viz.modalities.massspec.session import MassSpecSession, MsAcquisition
from constellation.viz.server.arrow_stream import batches_to_response

router = APIRouter(prefix="/api/massspec/sessions/{session_id}", tags=["massspec"])

#: Scan-table columns sent to the client for navigation.
SCAN_ROUTE_COLUMNS = (
    "scan",
    "level",
    "rt",
    "window",
    "parent_scan",
    "precursor_mz",
    "precursor_charge",
    "isolation_lower",
    "isolation_upper",
    "tic",
)


def _session(request: Request, session_id: str) -> MassSpecSession:
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    if getattr(session, "modality", None) != MassSpecSession.modality:
        raise HTTPException(404, f"session {session_id} is not a mass-spec session")
    return session


def _acquisition(request: Request, session_id: str, key: str) -> MsAcquisition:
    try:
        return _session(request, session_id).acquisition(key)
    except KeyError:
        raise HTTPException(404, f"session {session_id} has no acquisition {key!r}") from None


def _plain(value: Any) -> Any:
    """A JSON-safe copy of one Arrow row value (NaN → null, maps → dicts)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list) and value and isinstance(value[0], tuple):
        return {k: _plain(v) for k, v in value}
    return value


@router.get("/acquisitions")
def list_acquisitions(request: Request, session_id: str) -> list[dict[str, Any]]:
    return [a.describe() for a in _session(request, session_id).acquisitions]


@router.get("/acquisitions/{key}/windows")
def list_windows(request: Request, session_id: str, key: str) -> list[dict[str, Any]]:
    from constellation.massspec.scans import isolation_window_table

    acquisition = _acquisition(request, session_id, key)
    return isolation_window_table(scan_table_for(acquisition.bundle)).to_pylist()


@router.get("/acquisitions/{key}/scans")
def scan_rows(
    request: Request,
    session_id: str,
    key: str,
    level: int | None = Query(None, ge=1, le=10),
):
    """The scan table a client steps through and finds the nearest scan in."""
    scans = scan_table_for(_acquisition(request, session_id, key).bundle)
    if level is not None:
        scans = scans.filter(pc.equal(scans.column("level"), level))
    table = scans.select([c for c in SCAN_ROUTE_COLUMNS if c in scans.column_names])
    return batches_to_response(table.schema, table.to_batches())


@router.get("/acquisitions/{key}/scans/{scan}")
def scan_header(request: Request, session_id: str, key: str, scan: int) -> dict[str, Any]:
    """Everything the bundle recorded about one scan (filter string,
    analyzer, activation, injection time, trailer values, ...)."""
    acquisition = _acquisition(request, session_id, key)
    dataset = pa_ds.dataset(acquisition.bundle / "scan_metadata.parquet")
    rows = dataset.to_table(filter=pa_ds.field("scan") == scan).to_pylist()
    if not rows:
        raise HTTPException(404, f"acquisition {key!r} has no scan {scan}")
    return {k: _plain(v) for k, v in rows[0].items() if v is not None}


__all__ = ["SCAN_ROUTE_COLUMNS", "router"]
