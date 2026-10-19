"""An exclusive lock on a file, across threads and processes.

For read-modify-write of a small shared record (an experiment's manifest
and acquisitions table, say) by several searches running at once,
whether as threads of one command or as separate commands. ``flock``
locks belong to an open file description, so two threads that each open
the lock file block each other just as two processes do.

POSIX only in substance (Linux, WSL, macOS). Where ``fcntl`` is missing
the lock is a per-process ``threading.Lock``: threads are still kept
apart, separate processes are not.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

_FALLBACK: dict[str, threading.Lock] = {}
_FALLBACK_GUARD = threading.Lock()


class LockHeld(BlockingIOError):
    """:func:`file_lock` with ``wait=False`` found the lock taken."""


@contextmanager
def file_lock(path: Path | str, *, wait: bool = True) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if missing) for the
    block. The file itself is left in place afterwards; its contents are
    never used.

    ``wait=False``: raise :class:`LockHeld` at once, before the block
    runs, if someone else holds it (for "one at a time, and tell the
    second one so" rather than "take turns").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _fcntl is None:  # pragma: no cover - Windows
        with _FALLBACK_GUARD:
            lock = _FALLBACK.setdefault(str(path.resolve()), threading.Lock())
        if not lock.acquire(blocking=wait):
            raise LockHeld(f"{path} is held")
        try:
            yield
        finally:
            lock.release()
        return
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX if wait else _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockHeld(f"{path} is held") from None
        try:
            yield
        finally:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
    finally:
        os.close(fd)


__all__ = ["LockHeld", "file_lock"]
