"""Async subprocess runner for ``constellation`` CLI invocations.

Public surface:

- :class:`Job` — handle for a single in-flight command
- :class:`OutputFrame` — one stdout/stderr line, or the final exit
  sentinel
- :func:`spawn_job` — start a CLI subprocess and return its Job

The runner spawns ``python -m constellation.cli.__main__ <argv>`` (not
the ``constellation`` entry-point shim) so the subprocess inherits the
parent's Python interpreter regardless of how the parent was launched.
This keeps editable installs, venvs, and direct ``python -m`` invocations
all behaving identically.

Output is streamed line-by-line into a bounded history deque and
fanned out to per-subscriber queues. The WS endpoint replays history
on connect, then async-iterates the subscriber queue for live frames.
A sentinel frame with ``stream='exit'`` and the exit code as ``line``
signals subscribers to close the connection.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

GRACE_TERMINATE_SECONDS = 10.0
"""Seconds between SIGTERM and SIGKILL when a job is cancelled."""

_HISTORY_MAX_FRAMES = 10_000
"""Cap per-job history retention so a noisy long-running job doesn't
hold an unbounded buffer in memory. Frames beyond this drop from the
*replay* path; live subscribers still see them as they happen."""

SPAWN_PREFIX = (sys.executable, "-m", "constellation.cli.__main__")
"""The argv prefix prepended to every job's ``argv``.

Using ``python -m constellation.cli.__main__`` rather than the
``constellation`` entry-point shim preserves venv correctness: the
subprocess uses the same interpreter as the parent regardless of how
the parent was launched. Tests substitute this via monkeypatch to
spawn controlled subprocesses without going through the real CLI."""


FrameStream = Literal["stdout", "stderr", "exit"]


@dataclass(frozen=True)
class OutputFrame:
    """One line of subprocess output, or the terminal exit sentinel.

    ``stream='exit'`` marks the final frame; ``line`` is the exit code
    as a string (matches the wire shape the WS endpoint serializes).
    """

    stream: FrameStream
    line: str


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Job:
    """Handle for a single in-flight CLI subprocess.

    Lifecycle:

    1. :func:`spawn_job` constructs a Job, starts the subprocess, and
       schedules :meth:`_pump_output` as a background task.
    2. The WS endpoint calls :meth:`subscribe` to receive live frames;
       it replays :attr:`history` first, then awaits the subscriber
       queue for new ones.
    3. When the subprocess exits, :meth:`_pump_output` sets
       :attr:`exit_code` + :attr:`ended_at`, broadcasts the sentinel,
       and releases the global job lock.
    """

    id: UUID
    argv: list[str]
    started_at: datetime
    proc: asyncio.subprocess.Process | None = None
    state: JobState = JobState.PENDING
    history: deque[OutputFrame] = field(
        default_factory=lambda: deque(maxlen=_HISTORY_MAX_FRAMES)
    )
    subscribers: list[asyncio.Queue[OutputFrame]] = field(default_factory=list)
    exit_code: int | None = None
    ended_at: datetime | None = None
    _pump_task: asyncio.Task[None] | None = None
    _kill_task: asyncio.Task[None] | None = None

    def subscribe(self) -> asyncio.Queue[OutputFrame]:
        """Register a new subscriber queue and return it.

        The caller is responsible for calling :meth:`unsubscribe` when
        done (typically in a WS handler's ``finally`` block).
        """
        q: asyncio.Queue[OutputFrame] = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[OutputFrame]) -> None:
        """Remove a subscriber queue. Safe to call after the job ends."""
        if q in self.subscribers:
            self.subscribers.remove(q)

    def is_alive(self) -> bool:
        """Return True while the subprocess is still running."""
        return self.proc is not None and self.proc.returncode is None

    async def terminate(self) -> None:
        """Send SIGTERM; SIGKILL after :data:`GRACE_TERMINATE_SECONDS`.

        Marks the job state as CANCELLED. Idempotent — repeated calls
        after the process has exited are no-ops.
        """
        if not self.is_alive():
            return
        # If the user explicitly cancels, record that intent before the
        # pump task observes the non-zero exit code (which would
        # otherwise misclassify the run as FAILED).
        if self.state == JobState.RUNNING:
            self.state = JobState.CANCELLED
        assert self.proc is not None
        try:
            self.proc.terminate()
        except ProcessLookupError:
            return
        if self._kill_task is None:
            self._kill_task = asyncio.create_task(self._kill_after_grace())

    async def _kill_after_grace(self) -> None:
        await asyncio.sleep(GRACE_TERMINATE_SECONDS)
        if self.is_alive():
            assert self.proc is not None
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass


async def spawn_job(argv: list[str]) -> Job:
    """Start a ``python -m constellation.cli.__main__ <argv>`` subprocess.

    Returns the :class:`Job` immediately; the subprocess runs in the
    background, with output pumped through :meth:`Job._pump_output`.
    The caller (the FastAPI endpoint) is responsible for having
    acquired the global lock via :func:`acquire_or_409` *before*
    calling this function; the runner releases the lock when the pump
    task finishes.
    """
    job = Job(
        id=uuid4(),
        argv=list(argv),
        started_at=datetime.now(timezone.utc),
    )
    proc = await asyncio.create_subprocess_exec(
        *SPAWN_PREFIX,
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    job.proc = proc
    job.state = JobState.RUNNING
    job._pump_task = asyncio.create_task(_pump_output(job))
    return job


async def _pump_output(job: Job) -> None:
    """Stream stdout/stderr into history + live subscribers; reap on exit.

    Releases the global lock when the subprocess is fully reaped, even
    if the pump itself raises — see the surrounding ``try / finally``.
    """
    assert job.proc is not None
    proc = job.proc

    async def _reader(stream: asyncio.StreamReader, label: FrameStream) -> None:
        while True:
            raw = await stream.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            frame = OutputFrame(stream=label, line=line)
            _broadcast(job, frame)

    try:
        readers: list[asyncio.Task[None]] = []
        if proc.stdout is not None:
            readers.append(asyncio.create_task(_reader(proc.stdout, "stdout")))
        if proc.stderr is not None:
            readers.append(asyncio.create_task(_reader(proc.stderr, "stderr")))
        if readers:
            await asyncio.gather(*readers, return_exceptions=True)
        await proc.wait()
        job.exit_code = proc.returncode
        job.ended_at = datetime.now(timezone.utc)
        if job.state == JobState.RUNNING:
            job.state = (
                JobState.COMPLETED if job.exit_code == 0 else JobState.FAILED
            )
        sentinel = OutputFrame(stream="exit", line=str(job.exit_code))
        _broadcast(job, sentinel)
    finally:
        # Release the global lock regardless of how the pump terminated
        # so a stuck subscriber or unexpected exception never wedges the
        # dashboard. Import here to avoid a circular import at module
        # load time.
        from constellation.viz.runner.lock import release_lock

        release_lock()


def _broadcast(job: Job, frame: OutputFrame) -> None:
    """Append a frame to history and push to every active subscriber."""
    job.history.append(frame)
    # ``list(...)`` snapshot so unsubscribe-during-iteration is safe.
    for q in list(job.subscribers):
        q.put_nowait(frame)
