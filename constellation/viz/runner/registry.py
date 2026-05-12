"""In-memory job registry.

Holds the currently-running and recently-completed :class:`Job` objects
keyed by UUID so the WS endpoint, the cancellation endpoint, and the
StatusBar's ``/api/commands/active`` probe can all find them.

The registry is process-local — restarting the dashboard server clears
it. Persistence isn't part of v1; a future "job history" panel would
add a SQLite-backed log here.
"""

from __future__ import annotations

from uuid import UUID

from constellation.viz.runner.runner import Job, JobState

_JOBS: dict[UUID, Job] = {}


def register_job(job: Job) -> None:
    """Add a job to the registry."""
    _JOBS[job.id] = job


def get_job(job_id: UUID) -> Job | None:
    """Look up a job by id. Returns None when unknown."""
    return _JOBS.get(job_id)


def remove_job(job_id: UUID) -> None:
    """Remove a job from the registry. Idempotent."""
    _JOBS.pop(job_id, None)


def active_job() -> Job | None:
    """Return the currently-running job, if any.

    Used by the StatusBar's ``/api/commands/active`` endpoint. There is
    at most one running job at a time (enforced by the global lock), so
    this returns the first match rather than a list.
    """
    for job in _JOBS.values():
        if job.state == JobState.RUNNING:
            return job
    return None
