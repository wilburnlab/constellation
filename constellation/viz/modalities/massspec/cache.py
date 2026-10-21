"""Per-process caches for what the mass-spec kernels read repeatedly.

A browser asks for the same acquisition's scan table on every
chromatogram, every scan step and every spectrum, so the table is read
once and kept. Entries are keyed by the file's path and modification time
(a bundle rebuilt in place is read again) and evicted least recently used.
Loads happen outside the lock, so a slow read does not hold up requests for
other entries; two concurrent misses on one key both load and the second
result wins, which is harmless for immutable data.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa


class LruCache:
    """A small thread-safe least-recently-used map."""

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._entries: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_load(self, key: Any, load: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
        value = load()
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _file_key(path: Path) -> tuple[str, int]:
    return str(path), path.stat().st_mtime_ns


_SCAN_TABLES = LruCache(max_entries=16)


def scan_table_for(bundle: Path) -> pa.Table:
    """The bundle's :func:`massspec.scans.scan_table`, read once per process."""

    def load() -> pa.Table:
        from constellation.massspec.scans import scan_table

        return scan_table(bundle)

    return _SCAN_TABLES.get_or_load(_file_key(bundle / "scan_metadata.parquet"), load)


_IDENTIFICATIONS = LruCache(max_entries=16)


def identifications_for(experiment: Path, acquisition_id: int, max_qvalue: float) -> pa.Table:
    """``massspec.results.identifications`` of one run of an experiment,
    read once per process. Keyed by the experiment manifest's modification
    time, which every search added to the experiment rewrites."""

    def load() -> pa.Table:
        from constellation.massspec.results import identifications

        return identifications(experiment, acquisition_id=acquisition_id, max_qvalue=max_qvalue)

    key = (*_file_key(experiment / "manifest.json"), acquisition_id, max_qvalue)
    return _IDENTIFICATIONS.get_or_load(key, load)


__all__ = ["LruCache", "identifications_for", "scan_table_for"]
