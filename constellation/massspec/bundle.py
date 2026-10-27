"""The peaks of a converted acquisition (a ``massspec convert`` bundle).

A bundle is a directory holding ``manifest.json``, ``scan_metadata.parquet``,
``acquisition_metadata.parquet`` and the peaks, one row per peak
(:data:`MS_PEAK_TABLE`) in scan order. The peaks are either

- ``peaks/part-NNNNN.parquet``: parts whose names sort in scan order, each a
  contiguous range of scans (bundles written since manifest v2), or
- ``peaks.parquet``: one file (earlier bundles).

This module is the one definition of where a bundle's peaks are and how
peak rows are made and written. Code that reads peaks goes through
:func:`peaks_dataset` / :func:`iter_peak_batches` (or :func:`open_peaks` for
"a bundle, a directory of parts, or a Parquet file") rather than naming
files:

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
from typing import Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation.massspec.schemas import MS_PEAK_TABLE

PEAKS_DIR = "peaks"
LEGACY_PEAKS_FILE = "peaks.parquet"
DEFAULT_RT_BIN_WIDTH_S = 60.0
# pyarrow's own ceiling on a row group when write_table is given none.
MAX_ROW_GROUP_ROWS = 1 << 20

# Peak-table columns that hold the scan's value on every one of its peaks.
_SCAN_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in MS_PEAK_TABLE if f.name not in ("mz", "intensity")
    and not f.name.startswith("peak_")
)


# ── where a bundle's peaks are ────────────────────────────────────────


def part_name(index: int) -> str:
    """The file name of a bundle's ``index``-th peak part."""
    return f"part-{index:05d}.parquet"


def peak_paths(bundle_dir: Path | str) -> list[Path]:
    """A bundle's peak files, in scan order.

    Raises ``FileNotFoundError`` for a directory with neither layout.
    """
    bundle = Path(bundle_dir)
    parts = bundle / PEAKS_DIR
    if parts.is_dir():
        return sorted(parts.glob("part-*.parquet"))
    legacy = bundle / LEGACY_PEAKS_FILE
    if legacy.is_file():
        return [legacy]
    raise FileNotFoundError(
        f"{bundle} holds no peaks (neither {PEAKS_DIR}/ nor {LEGACY_PEAKS_FILE})"
    )


def has_peaks(bundle_dir: Path | str) -> bool:
    """Whether ``bundle_dir`` holds peaks in either layout."""
    bundle = Path(bundle_dir)
    return (bundle / PEAKS_DIR).is_dir() or (bundle / LEGACY_PEAKS_FILE).is_file()


def peaks_dataset(bundle_dir: Path | str) -> ds.Dataset:
    """A bundle's peaks as one dataset, its files in scan order.

    Built from the file list, so nothing else in the bundle directory
    (scan metadata, a peak index) is mistaken for peaks.
    """
    paths = peak_paths(bundle_dir)
    return ds.dataset([str(p) for p in paths], schema=_schema_of(paths), format="parquet")


def _schema_of(paths: Sequence[Path]) -> pa.Schema:
    return pq.read_schema(paths[0]) if paths else MS_PEAK_TABLE


def open_peaks(path: Path | str) -> ds.Dataset:
    """Peaks from a bundle, a directory of peak parts, or one Parquet file."""
    p = Path(path)
    if p.is_dir() and has_peaks(p):
        return peaks_dataset(p)
    return ds.dataset(p)


def iter_peak_batches(
    bundle_dir: Path | str,
    *,
    columns: Sequence[str] | None = None,
    batch_rows: int = 1 << 20,
) -> Iterator[pa.RecordBatch]:
    """A bundle's peak rows in stored (scan) order, in record batches."""
    for path in peak_paths(bundle_dir):
        yield from pq.ParquetFile(path).iter_batches(
            batch_size=batch_rows, columns=list(columns) if columns is not None else None
        )


# ── making and writing peak rows ──────────────────────────────────────


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

    def abort(self) -> None:
        """Stop without writing what is buffered (after an error)."""
        self._pending = []
        try:
            for future in self._inflight:
                future.cancel()
        finally:
            self._pool.shutdown(wait=True)
            self._writer.close()

    def __enter__(self) -> "PeakWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


__all__ = [
    "DEFAULT_RT_BIN_WIDTH_S",
    "LEGACY_PEAKS_FILE",
    "MAX_ROW_GROUP_ROWS",
    "PEAKS_DIR",
    "PeakWriter",
    "has_peaks",
    "iter_peak_batches",
    "open_peaks",
    "part_name",
    "peak_paths",
    "peak_rows",
    "peaks_dataset",
]
