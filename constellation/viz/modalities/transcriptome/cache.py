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
        self._lock = threading.Lock()
        #: How many computations ran, for the tests.
        self.computed = 0

    def _future(self, key: Any, compute: Callable[[], Any]) -> Future:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                done: Future = Future()
                done.set_result(self._entries[key])
                return done
            running = self._running.get(key)
            if running is not None:
                return running
            future = self._executor.submit(self._run, key, compute)
            self._running[key] = future
            return future

    def _run(self, key: Any, compute: Callable[[], Any]) -> Any:
        try:
            value = compute()
            with self._lock:
                self.computed += 1
                self._entries[key] = value
                self._entries.move_to_end(key)
                while len(self._entries) > self._max:
                    self._entries.popitem(last=False)
            return value
        finally:
            with self._lock:
                self._running.pop(key, None)

    def get(self, key: Any, compute: Callable[[], Any]) -> Any:
        """Blocking: from a thread that is not this cache's executor's."""
        return self._future(key, compute).result()

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
        """From an ``async`` endpoint, waiting without holding a thread."""
        return await asyncio.wrap_future(self._future(key, compute))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


#: One replay at a time: a 20,000-read template takes ~4 GB.
_REPLAY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tx-replay")
_INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tx-index")

REPLAYS = SingleFlight(max_entries=3, executor=_REPLAY_EXECUTOR)
SUPPORT = SingleFlight(max_entries=16, executor=_REPLAY_EXECUTOR)
INDEXES = SingleFlight(max_entries=4, executor=_INDEX_EXECUTOR)


def _file_key(path: Path) -> tuple[str, int]:
    return str(path), path.stat().st_mtime_ns


async def cluster_index_for(run_dir: Path) -> pa.Table:
    def load() -> pa.Table:
        from constellation.sequencing.transcriptome.cluster.denovo.em.results import (
            cluster_index,
        )

        return cluster_index(run_dir)

    return await INDEXES.aget(_file_key(Path(run_dir) / "clusters.parquet"), load)


def clear_all() -> None:
    for cache in (REPLAYS, SUPPORT, INDEXES):
        cache.clear()


__all__ = ["INDEXES", "REPLAYS", "SUPPORT", "SingleFlight", "clear_all", "cluster_index_for"]
