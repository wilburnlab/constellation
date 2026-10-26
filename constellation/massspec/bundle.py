"""The peaks of a converted acquisition (a ``massspec convert`` bundle).

A bundle is a directory holding ``manifest.json``, ``scan_metadata.parquet``,
``acquisition_metadata.parquet`` and the peaks, one row per peak
(:data:`MS_PEAK_TABLE`) in scan order. This module is where peak rows are
made and written:

- :func:`peak_rows` turns one block of scans (their metadata as an Arrow
  table, their peaks as numpy arrays) into peak rows, repeating each scan's
  columns over its peaks inside Arrow. No Python object exists per peak.
- :class:`PeakWriter` writes peak rows to one Parquet file, a row group per
  retention-time bin, encoding in a background thread.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.schemas import MS_PEAK_TABLE

DEFAULT_RT_BIN_WIDTH_S = 60.0
# pyarrow's own ceiling on a row group when write_table is given none.
MAX_ROW_GROUP_ROWS = 1 << 20

# Peak-table columns that hold the scan's value on every one of its peaks.
_SCAN_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in MS_PEAK_TABLE if f.name not in ("mz", "intensity")
    and not f.name.startswith("peak_")
)


def peak_rows(
    scans: pa.Table,
    counts: np.ndarray,
    mz: np.ndarray,
    intensity: np.ndarray,
    aux: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> pa.Table:
    """Peak rows (:data:`MS_PEAK_TABLE`) for a block of scans.

    ``scans`` has one row per scan and any of the peak table's per-scan
    columns (``scan``, ``level``, ``rt``, ``isolation_lower``, ...); a column
    it lacks is null. ``counts[i]`` is the number of peaks of scan ``i``,
    whose m/z and intensities are the next ``counts[i]`` values of ``mz`` and
    ``intensity``. ``aux`` maps a ``peak_*`` column to ``(values, valid)``,
    two arrays as long as ``mz``; peaks where ``valid`` is false are null.
    """
    n = int(mz.size)
    index = pa.array(np.repeat(np.arange(len(counts), dtype=np.int64), counts))
    columns: list[pa.Array] = []
    for field in MS_PEAK_TABLE:
        name = field.name
        if name == "mz":
            columns.append(pa.array(mz, pa.float64()))
        elif name == "intensity":
            columns.append(pa.array(intensity, pa.float64()))
        elif aux is not None and name in aux:
            values, valid = aux[name]
            columns.append(
                pa.array(values.astype(np.float32, copy=False), field.type, mask=~valid)
            )
        elif name in _SCAN_COLUMNS and name in scans.column_names:
            columns.append(scans.column(name).cast(field.type).take(index))
        else:
            columns.append(pa.nulls(n, field.type))
    return pa.Table.from_arrays(columns, schema=MS_PEAK_TABLE)


class PeakWriter:
    """Write peak rows to one Parquet file, scan-major.

    A row group ends where retention time crosses a multiple of
    ``rt_bin_width_s`` and at ``max_row_group_rows``, so reading one scan or
    one retention-time window touches few row groups. Rows must arrive in
    scan order. Encoding and compression run in a background thread, which
    pyarrow lets run without the GIL, so the caller goes on reading scans
    while a row group is written.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        rt_bin_width_s: float = DEFAULT_RT_BIN_WIDTH_S,
        max_row_group_rows: int = MAX_ROW_GROUP_ROWS,
    ):
        if rt_bin_width_s <= 0:
            raise ValueError(f"rt_bin_width_s must be positive, got {rt_bin_width_s}")
        self._writer = pq.ParquetWriter(
            str(path), MS_PEAK_TABLE, compression="zstd", use_dictionary=False
        )
        self._width = float(rt_bin_width_s)
        self._max_rows = int(max_row_group_rows)
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self._bin: int | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peak-writer")
        self._inflight: deque[Future] = deque()
        self.rows_written = 0

    def write(self, table: pa.Table) -> None:
        """Append peak rows (in scan order, after every row written before)."""
        n = table.num_rows
        if n == 0:
            return
        bins = np.floor(table.column("rt").to_numpy() / self._width).astype(np.int64)
        cuts = np.flatnonzero(bins[1:] != bins[:-1]) + 1
        for start, stop in zip((0, *cuts.tolist()), (*cuts.tolist(), n)):
            if self._bin is not None and bins[start] != self._bin:
                self._flush()
            self._bin = int(bins[start])
            piece = table.slice(start, stop - start)
            while piece.num_rows:
                take = piece.slice(0, self._max_rows - self._pending_rows)
                self._pending.append(take)
                self._pending_rows += take.num_rows
                piece = piece.slice(take.num_rows)
                if self._pending_rows >= self._max_rows:
                    self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        table = pa.concat_tables(self._pending)
        self._pending = []
        self._pending_rows = 0
        # Two row groups in flight at most: one being encoded, one queued.
        while len(self._inflight) >= 2:
            self._inflight.popleft().result()
        self._inflight.append(
            self._pool.submit(self._writer.write_table, table, row_group_size=self._max_rows)
        )
        self.rows_written += table.num_rows

    def close(self) -> None:
        """Write what is buffered and close the file."""
        try:
            self._flush()
            while self._inflight:
                self._inflight.popleft().result()
        finally:
            self._pool.shutdown(wait=True)
            self._writer.close()

    def __enter__(self) -> "PeakWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        # On an error, stop without writing what is buffered.
        self._pending = []
        try:
            for future in self._inflight:
                future.cancel()
        finally:
            self._pool.shutdown(wait=True)
            self._writer.close()


__all__ = [
    "DEFAULT_RT_BIN_WIDTH_S",
    "MAX_ROW_GROUP_ROWS",
    "PeakWriter",
    "peak_rows",
]
