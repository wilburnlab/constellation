"""Single-compute-job lock — sanity tests.

These verify the lock primitive in isolation. The end-to-end HTTP 409
path lives in ``test_viz_dashboard_endpoints.py`` (it needs the
``[viz]`` extras for FastAPI / httpx).
"""

from __future__ import annotations

import threading
import time

import pytest

from constellation.viz.runner.lock import (
    JobLockError,
    acquire_or_409,
    is_locked,
    release_lock,
)


@pytest.fixture(autouse=True)
def _reset_lock():
    """Ensure each test starts from an unlocked state."""
    if is_locked():
        release_lock()
    yield
    if is_locked():
        release_lock()


def test_initial_state_unlocked():
    assert not is_locked()


def test_acquire_changes_state():
    acquire_or_409()
    assert is_locked()
    release_lock()
    assert not is_locked()


def test_second_acquire_raises():
    acquire_or_409()
    with pytest.raises(JobLockError) as exc_info:
        acquire_or_409()
    assert "already running" in str(exc_info.value)


def test_release_is_idempotent():
    release_lock()  # nothing held
    release_lock()  # still nothing held
    acquire_or_409()
    release_lock()
    release_lock()  # double-release is fine
    assert not is_locked()


def test_lock_serializes_across_threads():
    """The lock must hold across thread boundaries — the FastAPI WS
    pump task and the cancel endpoint run on different async tasks
    that may execute on different threads in some uvicorn configs."""
    results: list[str] = []
    errors: list[Exception] = []

    acquire_or_409()

    def _attempt():
        try:
            acquire_or_409()
            results.append("acquired")
        except JobLockError:
            results.append("rejected")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_attempt) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    assert not errors
    # All five concurrent attempts should have been rejected — the
    # holder is the main thread.
    assert results == ["rejected"] * 5
    release_lock()


def test_lock_released_by_one_thread_acquired_by_another():
    """The lock has no owner-thread semantics — release is global."""
    acquire_or_409()

    holder_released = threading.Event()
    waiter_acquired = threading.Event()

    def _release_after_delay():
        time.sleep(0.05)
        release_lock()
        holder_released.set()

    def _try_after_release():
        holder_released.wait(timeout=1.0)
        try:
            acquire_or_409()
            waiter_acquired.set()
        finally:
            if is_locked():
                release_lock()

    releaser = threading.Thread(target=_release_after_delay)
    waiter = threading.Thread(target=_try_after_release)
    releaser.start()
    waiter.start()
    releaser.join(timeout=2.0)
    waiter.join(timeout=2.0)

    assert holder_released.is_set()
    assert waiter_acquired.is_set()
