"""Subprocess runner — spawn, stream, terminate, exit code.

Tests use a controlled Python subprocess (monkeypatched via
``SPAWN_PREFIX``) rather than the real ``constellation`` CLI so output
is deterministic. The runner has no fastapi dependency, so this file
runs under the base install.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from constellation.viz.runner import (
    Job,
    JobLockError,
    JobState,
    OutputFrame,
    acquire_or_409,
    is_locked,
    register_job,
    release_lock,
    spawn_job,
)
from constellation.viz.runner import runner as runner_mod


def _use_python_c(monkeypatch):
    """Substitute the spawn prefix so jobs run as ``python -c <code>``.

    The job's ``argv`` becomes the Python source code (one positional).
    """
    monkeypatch.setattr(runner_mod, "SPAWN_PREFIX", (sys.executable, "-c"))


async def _spawn_and_drain(code: str) -> tuple[Job, list[OutputFrame]]:
    """Acquire lock → spawn → drain to exit sentinel. All inside one loop."""
    acquire_or_409()
    job = await spawn_job([code])
    register_job(job)
    q = job.subscribe()
    frames: list[OutputFrame] = []
    while True:
        frame = await q.get()
        frames.append(frame)
        if frame.stream == "exit":
            break
    job.unsubscribe(q)
    # Give the pump task's `finally` block a tick to release the lock.
    await asyncio.sleep(0)
    return job, frames


def test_spawn_collects_stdout_and_exit_code(monkeypatch):
    """Spawn a tiny Python script via the runner; assert stdout is
    captured and the exit code propagates."""
    _use_python_c(monkeypatch)
    job, frames = asyncio.run(
        _spawn_and_drain("import sys; print('hello'); sys.exit(0)")
    )
    stdout_lines = [f.line for f in frames if f.stream == "stdout"]
    assert "hello" in stdout_lines
    exit_frames = [f for f in frames if f.stream == "exit"]
    assert len(exit_frames) == 1
    assert exit_frames[0].line == "0"
    assert job.exit_code == 0
    assert job.state == JobState.COMPLETED
    assert not is_locked()


def test_spawn_separates_stdout_and_stderr(monkeypatch):
    """The runner labels every frame with its source stream so the
    Terminal renders stderr in red without parsing prefixes."""
    _use_python_c(monkeypatch)
    _job, frames = asyncio.run(
        _spawn_and_drain(
            "import sys; "
            "print('to stdout'); "
            "print('to stderr', file=sys.stderr); "
            "sys.exit(0)"
        )
    )
    stdout_lines = [f.line for f in frames if f.stream == "stdout"]
    stderr_lines = [f.line for f in frames if f.stream == "stderr"]
    assert "to stdout" in stdout_lines
    assert "to stderr" in stderr_lines
    assert not is_locked()


def test_nonzero_exit_marks_failed(monkeypatch):
    """Subprocess exits non-zero → state is FAILED, exit_code preserved."""
    _use_python_c(monkeypatch)
    job, frames = asyncio.run(_spawn_and_drain("import sys; sys.exit(7)"))
    exit_frames = [f for f in frames if f.stream == "exit"]
    assert exit_frames[0].line == "7"
    assert job.exit_code == 7
    assert job.state == JobState.FAILED
    assert not is_locked()


def test_terminate_kills_long_running_job(monkeypatch):
    """Cancellation sends SIGTERM; the subprocess exits within the
    grace window and the job ends up CANCELLED."""
    _use_python_c(monkeypatch)
    # Shrink the grace window so the test runs quickly if SIGTERM is
    # ignored (it shouldn't be for Python, but defensive).
    monkeypatch.setattr(runner_mod, "GRACE_TERMINATE_SECONDS", 1.0)

    async def main():
        acquire_or_409()
        # 30s sleep — the test wouldn't tolerate this without termination.
        job = await spawn_job(["import time; time.sleep(30)"])
        register_job(job)
        # Give the subprocess a moment to start running
        await asyncio.sleep(0.2)
        start = time.monotonic()
        await job.terminate()
        # Drain until the exit sentinel arrives
        q = job.subscribe()
        while True:
            frame = await q.get()
            if frame.stream == "exit":
                break
        elapsed = time.monotonic() - start
        return job, elapsed

    job, elapsed = asyncio.run(main())
    assert elapsed < 5.0, f"job took {elapsed:.2f}s to terminate"
    assert job.state == JobState.CANCELLED
    assert not is_locked()


def test_subscriber_receives_history_replay_on_late_subscribe(monkeypatch):
    """A WS that connects after a job has already produced output must
    still see the earlier lines (history replay)."""
    _use_python_c(monkeypatch)

    async def main():
        acquire_or_409()
        job = await spawn_job(
            ["print('line1'); print('line2'); print('line3')"]
        )
        register_job(job)
        # Wait for the pump task to finish so the exit sentinel is in
        # history before we subscribe — simulates a slow WS connect.
        assert job._pump_task is not None
        await job._pump_task
        # History should now contain at least the three stdout frames
        # plus the exit sentinel.
        return job

    job = asyncio.run(main())
    history_lines = [f.line for f in job.history if f.stream == "stdout"]
    assert history_lines == ["line1", "line2", "line3"]
    exit_frame = next((f for f in job.history if f.stream == "exit"), None)
    assert exit_frame is not None
    assert exit_frame.line == "0"
    assert not is_locked()


def test_release_lock_is_idempotent():
    """Cleanup paths might double-release (subprocess reaper + WS
    finally block); the runner depends on release being a no-op when
    already free."""
    if is_locked():
        release_lock()
    acquire_or_409()
    release_lock()
    release_lock()  # No exception
    assert not is_locked()


def test_acquire_or_409_rejects_when_held():
    """Concurrent spawn attempts must fail fast rather than hang."""
    if is_locked():
        release_lock()
    acquire_or_409()
    with pytest.raises(JobLockError):
        acquire_or_409()
    release_lock()
