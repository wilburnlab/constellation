"""Single-compute-job lock for dashboard-initiated CLI runs.

A module-level :class:`threading.Lock` enforces the project rule from
[constellation/viz/CLAUDE.md] that at most one compute job runs at a
time. Visualization endpoints (``/api/tracks/...``, ``/api/sessions``)
never touch this lock — only ``POST /api/commands``.

Acquisition is non-blocking: :func:`acquire_or_409` raises
:class:`JobLockError` immediately when the lock is held, which the
endpoint maps to HTTP 409 Conflict. The lock is held for the duration
of the job (across the spawn → run → exit lifecycle, which spans many
HTTP requests), so :class:`threading.Lock` is the right primitive — we
need acquire/release semantics, not async-aware waiting.
"""

from __future__ import annotations

import threading


class JobLockError(RuntimeError):
    """Raised when the single-job lock is already held."""


_LOCK = threading.Lock()


def is_locked() -> bool:
    """Return True when a job currently holds the lock."""
    return _LOCK.locked()


def acquire_or_409() -> None:
    """Acquire the global job lock immediately or raise :class:`JobLockError`.

    The caller is responsible for calling :func:`release_lock` exactly
    once when the job ends (success, failure, or cancel). The
    ``/api/commands`` endpoint wires this around the job's lifecycle via
    a try / finally in the spawn-and-track helper.
    """
    if not _LOCK.acquire(blocking=False):
        raise JobLockError("another compute job is already running")


def release_lock() -> None:
    """Release the global job lock.

    Idempotent: releasing an unlocked lock is a no-op so cleanup paths
    that might double-release (subprocess reaper + WS finally block)
    don't crash.
    """
    if _LOCK.locked():
        try:
            _LOCK.release()
        except RuntimeError:
            # Another thread released between our check and our call.
            pass
