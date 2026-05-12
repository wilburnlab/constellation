"""Command-execution endpoints for the dashboard.

- ``POST   /api/commands``                   spawn a subprocess (acquires lock)
- ``GET    /api/commands/{job_id}``          job status snapshot
- ``WS     /api/commands/{job_id}/stream``   live stdout/stderr stream
- ``DELETE /api/commands/{job_id}``          SIGTERM → SIGKILL after grace
- ``GET    /api/commands/active``            current running job, if any

The single-compute-job lock from :mod:`constellation.viz.runner.lock`
serializes spawns — concurrent attempts return HTTP 409. The lock is
held for the *lifetime* of the running job (not just the spawn call),
released by the runner's pump task when the subprocess fully exits.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from constellation.viz.runner import (
    Job,
    JobLockError,
    acquire_or_409,
    active_job,
    get_job,
    register_job,
    release_lock,
    spawn_job,
)


router = APIRouter(prefix="/api/commands", tags=["commands"])


class CommandRequest(BaseModel):
    """Body for ``POST /api/commands``."""

    argv: list[str] = Field(..., min_length=1)


class CommandResponse(BaseModel):
    """Reply for ``POST /api/commands``."""

    job_id: str
    argv: list[str]
    started_at: str
    state: str


class JobSnapshot(BaseModel):
    """Reply for ``GET /api/commands/{job_id}`` and ``/active``."""

    job_id: str
    argv: list[str]
    started_at: str
    ended_at: str | None
    exit_code: int | None
    state: str


def _to_response(job: Job) -> CommandResponse:
    return CommandResponse(
        job_id=str(job.id),
        argv=list(job.argv),
        started_at=job.started_at.isoformat(),
        state=job.state.value,
    )


def _to_snapshot(job: Job) -> JobSnapshot:
    return JobSnapshot(
        job_id=str(job.id),
        argv=list(job.argv),
        started_at=job.started_at.isoformat(),
        ended_at=job.ended_at.isoformat() if job.ended_at else None,
        exit_code=job.exit_code,
        state=job.state.value,
    )


@router.post("", response_model=CommandResponse)
async def start_command(body: CommandRequest) -> CommandResponse:
    """Spawn a CLI subprocess. Returns 409 if another job is running."""
    try:
        acquire_or_409()
    except JobLockError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    try:
        job = await spawn_job(body.argv)
    except Exception:
        # Spawn itself failed — release the lock we just acquired so
        # the dashboard doesn't wedge.
        release_lock()
        raise
    register_job(job)
    return _to_response(job)


@router.get("/active", response_model=JobSnapshot | None)
def get_active() -> JobSnapshot | None:
    """Return the currently-running job, or ``null``."""
    job = active_job()
    return _to_snapshot(job) if job is not None else None


@router.get("/{job_id}", response_model=JobSnapshot)
def get_command(job_id: UUID) -> JobSnapshot:
    """Snapshot of a job's lifecycle state."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job_id: {job_id}")
    return _to_snapshot(job)


@router.delete("/{job_id}", response_model=JobSnapshot)
async def cancel_command(job_id: UUID) -> JobSnapshot:
    """Send SIGTERM; SIGKILL after grace if still alive."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job_id: {job_id}")
    await job.terminate()
    return _to_snapshot(job)


@router.websocket("/{job_id}/stream")
async def stream_command(websocket: WebSocket, job_id: UUID) -> None:
    """Stream a job's stdout/stderr line-by-line, then close.

    Wire frames: newline-delimited JSON, each ``{"stream", "line"}``.
    The final frame uses ``stream='exit'`` with ``line`` set to the
    exit code as a string. After the exit frame is sent, the server
    closes the WS.

    The endpoint replays the job's history first so a slow-to-connect
    client doesn't miss output that flowed before the WS opened.
    """
    await websocket.accept()
    job = get_job(job_id)
    if job is None:
        await websocket.close(code=4404, reason=f"unknown job_id: {job_id}")
        return
    queue = job.subscribe()
    try:
        # Replay the history snapshot so the terminal renders earlier
        # output before live frames start flowing.
        for frame in list(job.history):
            await websocket.send_json({"stream": frame.stream, "line": frame.line})
            if frame.stream == "exit":
                # Job already completed before the WS opened — close out.
                return
        while True:
            frame = await queue.get()
            await websocket.send_json({"stream": frame.stream, "line": frame.line})
            if frame.stream == "exit":
                return
    except WebSocketDisconnect:
        return
    finally:
        job.unsubscribe(queue)
        # WS close is automatic on handler return; explicit close is
        # only needed if we returned early via the unknown-job_id path
        # above (which already called close()).
        try:
            await websocket.close()
        except RuntimeError:
            # Already closed — fine.
            pass
