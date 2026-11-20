"""Per-process caches for the transcriptome browser.

Two things are slow enough to keep:

- a run's cluster index (``em.results.cluster_index``: ~23 s and ~128 MB
  for 1.67M clusters, the time it takes to stream every consensus once to
  measure it), keyed by ``clusters.parquet``'s path and modification time;
- replays and the support tables made from them. A replay re-runs one
  template's M-step (a second for a typical cluster, a minute for 20,000
  reads), and the support view asks for six things about it at once (a
  summary and five tables), so the work is **single-flight**: the first
  request starts it, the others wait on the same future, and the result is
  kept in a small LRU. It runs on a dedicated one-worker executor, so a
  slow replay never holds a thread of the server's own pool, and one
  replay at a time keeps peak memory to one template's.

Every key carries the run's revision (:func:`run_revision`): the identity,
modification time and size of the files a pipeline rewrites whenever it
changes what a replay reads — each round's records and the final stage's
outputs — so a run finalized again, or replaced, is never served (or
certified ``verified``) from what was computed on the one before.

A request that is cancelled stops waiting; it does not cancel the work,
which another request may be waiting on too. Work nobody waits on any more
is cancelled if it has not started, and its key is free again either way.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa


class SingleFlight:
    """An LRU of computed values where concurrent requests for one key
    share one computation."""

    def __init__(self, max_entries: int, executor: ThreadPoolExecutor) -> None:
        self._max = max_entries
        self._executor = executor
        self._entries: OrderedDict[Any, Any] = OrderedDict()
        self._running: dict[Any, Future] = {}
        #: Requests waiting on each running future.
        self._waiters: dict[Future, int] = {}
        self._lock = threading.Lock()
        #: How many computations ran, for the tests.
        self.computed = 0

    def _future(self, key: Any, compute: Callable[[], Any]) -> Future:
        """The future for ``key``, with one more waiter counted on it."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                done: Future = Future()
                done.set_result(self._entries[key])
                return done
            running = self._running.get(key)
            if running is None:
                running = self._executor.submit(self._run, key, compute)
                self._running[key] = running
                self._waiters[running] = 0
                # A future cancelled before it started never runs `_run`,
                # so its key is released here rather than there.
                running.add_done_callback(lambda f, k=key: self._forget(k, f))
            self._waiters[running] = self._waiters.get(running, 0) + 1
            return running

    def _forget(self, key: Any, future: Future) -> None:
        with self._lock:
            if self._running.get(key) is future:
                del self._running[key]
            self._waiters.pop(future, None)

    def _release(self, future: Future) -> None:
        """One waiter stopped waiting; cancel the work if it was the last
        and the work has not started."""
        with self._lock:
            left = self._waiters.get(future, 0) - 1
            if left > 0:
                self._waiters[future] = left
                return
            self._waiters.pop(future, None)
        future.cancel()  # a no-op once running or done

    def _run(self, key: Any, compute: Callable[[], Any]) -> Any:
        # The key's running future is released by its done callback
        # (`_forget`), which also covers a future that never ran.
        value = compute()
        with self._lock:
            self.computed += 1
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return value

    def get(self, key: Any, compute: Callable[[], Any]) -> Any:
        """Blocking: from a thread that is not this cache's executor's."""
        future = self._future(key, compute)
        try:
            return future.result()
        finally:
            self._done_waiting(future)

    def _done_waiting(self, future: Future) -> None:
        with self._lock:
            if future in self._waiters:
                self._waiters[future] -= 1

    def get_here(self, key: Any, compute: Callable[[], Any]) -> Any:
        """The cached value, or computed in the calling thread. For a
        computation already running on the shared one-worker executor (a
        support table needing its replay), where submitting would wait on
        the worker it occupies."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
        return self._run(key, compute)

    async def aget(self, key: Any, compute: Callable[[], Any]) -> Any:
        """From an ``async`` endpoint, waiting without holding a thread.

        Shielded: cancelling this request does not cancel the shared work,
        which only stops if no request is left waiting for it."""
        future = self._future(key, compute)
        try:
            value = await asyncio.shield(asyncio.wrap_future(future))
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            self._release(future)
            raise
        self._done_waiting(future)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


#: One replay at a time: a 20,000-read template takes ~4 GB.
_REPLAY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tx-replay")
_INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tx-index")

REPLAYS = SingleFlight(max_entries=3, executor=_REPLAY_EXECUTOR)
SUPPORT = SingleFlight(max_entries=16, executor=_REPLAY_EXECUTOR)
INDEXES = SingleFlight(max_entries=4, executor=_INDEX_EXECUTOR)


#: Files a pipeline rewrites whenever it changes what a replay or the
#: cluster index reads: the final stage's outputs, and each round's records.
_RUN_FILES = (
    "manifest.json",
    "clusters.parquet",
    "cluster_nodes.parquet",
    "cluster_membership.parquet",
)
_ROUND_FILES = (
    "_SUCCESS",
    "round.json",
    "params.json",
    "refine.json",
    "final.json",
    "final_params.json",
    "merged_final.parquet",
    "templates/_SUCCESS",
)


def _stat(path: Path) -> tuple[int, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


def run_revision(run_dir: Path) -> tuple:
    """What a run's evidence is, as of now: the identity, modification time
    and size of each file in ``_RUN_FILES`` and of each round's
    ``_ROUND_FILES``. A pipeline never changes a round's shards or the
    final outputs without rewriting these, and a run replaced at the same
    path has new files, so a changed revision is a changed run. Some 60
    ``stat`` calls for a twelve-round run."""
    run_dir = Path(run_dir)
    parts: list[Any] = [str(run_dir)]
    parts.extend(_stat(run_dir / name) for name in _RUN_FILES)
    rounds = run_dir / "rounds"
    try:
        names = sorted(p.name for p in rounds.iterdir() if p.is_dir())
    except OSError:
        names = []
    for name in names:
        parts.append(name)
        parts.extend(_stat(rounds / name / f) for f in _ROUND_FILES)
    return tuple(parts)


async def cluster_index_for(run_dir: Path) -> pa.Table:
    def load() -> pa.Table:
        from constellation.sequencing.transcriptome.cluster.denovo.em.results import (
            cluster_index,
        )

        return cluster_index(run_dir)

    revision = await asyncio.to_thread(run_revision, run_dir)
    return await INDEXES.aget(revision, load)


def clear_all() -> None:
    for cache in (REPLAYS, SUPPORT, INDEXES):
        cache.clear()


__all__ = [
    "INDEXES",
    "REPLAYS",
    "SUPPORT",
    "SingleFlight",
    "clear_all",
    "cluster_index_for",
    "run_revision",
]
