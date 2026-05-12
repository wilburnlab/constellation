"""Subprocess runner for dashboard-initiated CLI invocations.

The dashboard's "Run" button POSTs to ``/api/commands`` which:
1. Acquires the single-job lock (or returns HTTP 409 if held)
2. Spawns ``python -m constellation.cli.__main__ <argv>`` via :mod:`runner`
3. Registers the resulting :class:`Job` in :mod:`registry`
4. Returns the job id; the client opens a WebSocket to stream output

The lock enforces the project's single-compute-job rule (visualization
endpoints stay unrestricted). The runner uses asyncio subprocess so the
event loop stays responsive while jobs run; output is streamed line-by-
line into per-job queues that the WS endpoint drains.

This module is stdlib-only (``asyncio``, ``subprocess``, ``signal``,
``uuid``) — the ``[viz]`` extras are needed only for the surrounding
FastAPI endpoints, not for the runner itself.
"""

from constellation.viz.runner.lock import (
    JobLockError,
    acquire_or_409,
    is_locked,
    release_lock,
)
from constellation.viz.runner.registry import (
    active_job,
    get_job,
    register_job,
    remove_job,
)
from constellation.viz.runner.runner import (
    GRACE_TERMINATE_SECONDS,
    Job,
    JobState,
    OutputFrame,
    spawn_job,
)

__all__ = [
    "GRACE_TERMINATE_SECONDS",
    "Job",
    "JobLockError",
    "JobState",
    "OutputFrame",
    "acquire_or_409",
    "active_job",
    "get_job",
    "is_locked",
    "register_job",
    "release_lock",
    "remove_job",
    "spawn_job",
]
