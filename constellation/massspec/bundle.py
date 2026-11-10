"""What a converted acquisition is on disk (a ``massspec convert`` bundle).

A bundle is a directory holding ``manifest.json``, ``scan_metadata.parquet``,
``acquisition_metadata.parquet`` and one set of points, one row per point
(:data:`MS_PEAK_TABLE`) in scan order, as parts whose names sort in scan
order, each a contiguous range of scans. There are two kinds:

- a **peak bundle** holds ``peaks/part-NNNNN.parquet``: centroids, always. A
  run can have any number of them, one per way of centroiding it;
- a **profile bundle** holds ``profile/part-NNNNN.parquet``: what each scan
  stored, kept so that peak lists can be made from it later. Its manifest's
  kind is ``thermo_profile``.

They are separate directories, and every consumer treats a bundle as one
list of points, so nothing has to ask which layer it was given. Bundles of
one run are tied together by the ``source_sha256`` their manifests record.

This module is the one definition of what a bundle holds and how its rows
are made and written. Code that reads peaks goes through
:func:`peaks_dataset` / :func:`iter_peak_batches` (or :func:`open_peaks` for
"a bundle, a directory of parts, or a Parquet file"), which refuse a profile
bundle; code that reads either kind goes through :func:`points_dataset`.
:func:`iter_scan_blocks` streams either kind as blocks of whole scans, and
:func:`scan_range_shards` cuts a bundle into scan ranges for workers to
stream separately. None of them names files. Only the current format opens: an older bundle is
refused with :class:`OutdatedBundleError` rather than read as if it were
current, because what its files mean has changed.

- :func:`peak_rows` turns one block of scans (their metadata as an Arrow
  table, their peaks as numpy arrays) into peak rows, repeating each scan's
  columns over its peaks inside Arrow. No Python object exists per peak.
- :class:`PeakWriter` writes peak rows to one Parquet file, a row group per
  retention-time bin, encoding in a background thread.
"""

from __future__ import annotations

import functools
import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation.massspec.schemas import MS_PEAK_TABLE

PEAKS_DIR = "peaks"
PROFILE_DIR = "profile"
#: Where a bundle keeps its measured m/z curves (``massspec.mz_axis``).
MZ_AXES_DIR = "mz_axes"
MANIFEST_FILENAME = "manifest.json"
#: A run's profile bundle sits beside its peak bundle, named by this suffix.
PROFILE_SUFFIX = ".profile"
PEAK_BUNDLE_KINDS = ("thermo_acquisition", "encyclopedia_dia_acquisition")
PROFILE_BUNDLE_KIND = "thermo_profile"
BUNDLE_KINDS = (*PEAK_BUNDLE_KINDS, PROFILE_BUNDLE_KIND)
#: The only format that opens. v3 (2026-10-31): ``peaks/`` always means
#: centroids and profile points have their own bundle; a blank trailer is
#: null rather than zero; ``profile_mode`` is gone. Before that, ``peaks/``
#: could hold either, so an older bundle cannot be read as a current one.
BUNDLE_SCHEMA_VERSION = 3
DEFAULT_RT_BIN_WIDTH_S = 60.0
# pyarrow's own ceiling on a row group when write_table is given none.
MAX_ROW_GROUP_ROWS = 1 << 20

# Peak-table columns that hold the scan's value on every one of its peaks.
_SCAN_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in MS_PEAK_TABLE if f.name not in ("mz", "intensity")
    and not f.name.startswith("peak_")
)


# ── what a bundle is ──────────────────────────────────────────────────


class OutdatedBundleError(ValueError):
    """A bundle written in a format that no longer opens."""


def _refuse_outdated(bundle: Path, manifest: Mapping[str, Any]) -> None:
    version = manifest.get("schema_version")
    if version != BUNDLE_SCHEMA_VERSION:
        raise OutdatedBundleError(
            f"{bundle} was written by an older Constellation (bundle format {version!r}; "
            f"this one reads {BUNDLE_SCHEMA_VERSION}). Convert the run again with "
            f"`constellation massspec convert`."
        )


def read_bundle_manifest(bundle_dir: Path | str) -> dict[str, Any]:
    """A bundle's manifest, checked to be a bundle in the current format.

    Everything that takes a bundle reads its manifest through this.
    """
    bundle = Path(bundle_dir)
    path = bundle / MANIFEST_FILENAME
    if not path.is_file():
        raise ValueError(f"{bundle} is not a converted acquisition (no {MANIFEST_FILENAME})")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in BUNDLE_KINDS:
        raise ValueError(
            f"{bundle} holds a {manifest.get('kind')!r}, not a converted acquisition "
            f"({' or '.join(BUNDLE_KINDS)})"
        )
    _refuse_outdated(bundle, manifest)
    return manifest


@functools.lru_cache(maxsize=512)
def _manifest_version(path: str, mtime_ns: int) -> Any:
    # A Thermo manifest runs to 100 KB and every spectrum read checks it.
    return json.loads(Path(path).read_text(encoding="utf-8")).get("schema_version")


def _check_current(bundle: Path) -> None:
    """Refuse a bundle in an older format. A directory with no manifest has
    no format to check: an unfinished conversion, or a directory of parts."""
    path = bundle / MANIFEST_FILENAME
    try:
        mtime_ns = path.stat().st_mtime_ns
    except (FileNotFoundError, NotADirectoryError):
        return
    _refuse_outdated(bundle, {"schema_version": _manifest_version(str(path), mtime_ns)})


def list_mz_axes(bundle_dir: Path | str) -> list[str]:
    """The keys of a bundle's measured m/z curves, oldest first."""
    directory = Path(bundle_dir) / MZ_AXES_DIR
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("*.parquet"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    return [p.stem for p in files]


def peak_list_name(manifest: Mapping[str, Any]) -> str:
    """What to call a bundle's list of points: ``profile``, the name a derived
    peak list was given, or else what made it (``thermo``, ``imported``)."""
    if manifest.get("kind") == PROFILE_BUNDLE_KIND:
        return PROFILE_DIR
    derived = manifest.get("derived_from") or {}
    if derived.get("name"):
        return str(derived["name"])
    return str((manifest.get("parameters") or {}).get("centroider") or "imported")


def _names_each_other(a: Path, a_manifest: Mapping[str, Any], b: Path, b_manifest: Mapping[str, Any]) -> bool:
    for source, manifest, other in ((a, a_manifest, b), (b, b_manifest, a)):
        parameters = manifest.get("parameters") or {}
        if other.name in (parameters.get("profile_bundle"), parameters.get("peak_bundle")):
            return True
        derived_from = (manifest.get("derived_from") or {}).get("source")
        if derived_from and (source.parent / str(derived_from)).resolve() == other.resolve():
            return True
    return False


def related_bundles(bundle_dir: Path | str) -> list[dict[str, Any]]:
    """The run's other bundles beside this one, without recursing: directories
    named for the run whose manifest has the same ``source_sha256``, or which
    are tied to this bundle by naming each other, directly or through another
    of the run's bundles (a conversion with no hash: a derived list names the
    profile, and the profile names the conversion's peak list). Each is
    ``name``, ``bundle``, ``holds_profile`` and ``mz_axes``. A bundle in an
    older format is passed over."""
    bundle = Path(bundle_dir).resolve()
    manifest = read_bundle_manifest(bundle)
    source = str(manifest.get("source_file") or bundle.name).replace("\\", "/").rsplit("/", 1)[-1]
    run = Path(source).stem
    sha = manifest.get("source_sha256")
    candidates: dict[Path, Mapping[str, Any]] = {}
    for candidate in sorted(bundle.parent.iterdir()):
        if candidate == bundle or not candidate.is_dir():
            continue
        if candidate.name != run and not candidate.name.startswith(run + "."):
            continue
        try:
            candidates[candidate] = read_bundle_manifest(candidate)
        except (ValueError, OSError):
            continue

    tied = {c for c, other in candidates.items() if sha and other.get("source_sha256") == sha}
    reached = [(bundle, manifest), *((c, candidates[c]) for c in tied)]
    while reached:
        member, member_manifest = reached.pop()
        for candidate, other in candidates.items():
            if candidate not in tied and _names_each_other(member, member_manifest, candidate, other):
                tied.add(candidate)
                reached.append((candidate, other))
    return [
        {
            "name": peak_list_name(candidates[c]),
            "bundle": c,
            "holds_profile": is_profile_bundle(c),
            "mz_axes": list_mz_axes(c),
        }
        for c in sorted(tied)
    ]


def part_name(index: int) -> str:
    """The file name of a bundle's ``index``-th part."""
    return f"part-{index:05d}.parquet"


def is_profile_bundle(bundle_dir: Path | str) -> bool:
    """Whether ``bundle_dir`` holds profile points rather than peaks."""
    return (Path(bundle_dir) / PROFILE_DIR).is_dir()


def peak_paths(bundle_dir: Path | str) -> list[Path]:
    """A peak bundle's files, in scan order.

    Raises ``ValueError`` for a profile bundle, whose points are not peaks,
    and ``FileNotFoundError`` for a directory with neither.
    """
    bundle = Path(bundle_dir)
    _check_current(bundle)
    parts = bundle / PEAKS_DIR
    if parts.is_dir():
        return sorted(parts.glob("part-*.parquet"))
    if is_profile_bundle(bundle):
        raise ValueError(
            f"{bundle} is a profile bundle: it holds profile points, not peaks. Make a "
            f"peak list from it with `constellation massspec centroid`."
        )
    raise FileNotFoundError(f"{bundle} holds no peaks (no {PEAKS_DIR}/)")


def point_paths(bundle_dir: Path | str) -> list[Path]:
    """A bundle's files of either kind, in scan order: for reading a scan as
    it is, whatever the bundle holds."""
    bundle = Path(bundle_dir)
    if is_profile_bundle(bundle):
        _check_current(bundle)
        return sorted((bundle / PROFILE_DIR).glob("part-*.parquet"))
    return peak_paths(bundle)


def has_peaks(bundle_dir: Path | str) -> bool:
    """Whether ``bundle_dir`` holds peaks."""
    return (Path(bundle_dir) / PEAKS_DIR).is_dir()


def _dataset_of(paths: Sequence[Path]) -> ds.Dataset:
    # Built from the file list, so nothing else in the bundle directory
    # (scan metadata, a peak index) is mistaken for points.
    return ds.dataset([str(p) for p in paths], schema=_schema_of(paths), format="parquet")


def peaks_dataset(bundle_dir: Path | str) -> ds.Dataset:
    """A peak bundle's peaks as one dataset, its files in scan order."""
    return _dataset_of(peak_paths(bundle_dir))


def points_dataset(bundle_dir: Path | str) -> ds.Dataset:
    """A bundle's points of either kind as one dataset, in scan order."""
    return _dataset_of(point_paths(bundle_dir))


def _schema_of(paths: Sequence[Path]) -> pa.Schema:
    return pq.read_schema(paths[0]) if paths else MS_PEAK_TABLE


def open_peaks(path: Path | str) -> ds.Dataset:
    """Peaks from a bundle, a directory of peak parts, or one Parquet file."""
    p = Path(path)
    if p.is_dir() and (has_peaks(p) or is_profile_bundle(p)):
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


# ── whole scans, in blocks ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScanBlock:
    """Consecutive whole scans in stored order. Scan ``scan[i]``'s points are
    ``columns[name][offsets[i]:offsets[i + 1]]``, in the order they are stored."""

    scan: np.ndarray  # (n_scans,) int64
    offsets: np.ndarray  # (n_scans + 1,) int64
    columns: dict[str, np.ndarray]

    @property
    def n_scans(self) -> int:
        return int(self.scan.size)

    @property
    def n_points(self) -> int:
        return int(self.offsets[-1])


def _scan_column_index(file: pq.ParquetFile) -> int:
    return file.schema_arrow.get_field_index("scan")


def _row_group_scan_bounds(file: pq.ParquetFile) -> list[tuple[int, int] | None]:
    """Each row group's (min, max) scan number from the footer, None where the
    footer has no statistics for it."""
    column = _scan_column_index(file)
    bounds: list[tuple[int, int] | None] = []
    for i in range(file.metadata.num_row_groups):
        stats = file.metadata.row_group(i).column(column).statistics
        if stats is None or not stats.has_min_max:
            bounds.append(None)
        else:
            bounds.append((int(stats.min), int(stats.max)))
    return bounds


def _row_groups_in(file: pq.ParquetFile, lo: int, hi: int) -> list[int]:
    """Row groups that can hold scans in ``[lo, hi)``."""
    return [
        i
        for i, b in enumerate(_row_group_scan_bounds(file))
        if b is None or (b[1] >= lo and b[0] < hi)
    ]


def iter_scan_blocks(
    bundle_dir: Path | str,
    *,
    columns: Sequence[str] = ("mz", "intensity"),
    block_rows: int = 1 << 20,
    scan_range: tuple[int, int] | None = None,
    points: bool = False,
) -> Iterator[ScanBlock]:
    """A bundle's points as blocks of whole scans, in stored order.

    Row groups are cut on retention-time bins and on a row cap, neither of
    which respects scan boundaries, so the trailing scan of each record batch
    is carried into the next. Blocks hold about ``block_rows`` points; a
    scan larger than that is still yielded whole. ``scan_range=(lo, hi)``
    keeps scans ``lo <= scan < hi``, reading only the row groups whose footer
    statistics allow such scans. A scan with no points has none to yield.
    ``points=True`` reads a profile bundle's points as well as peaks.
    """
    paths = point_paths(bundle_dir) if points else peak_paths(bundle_dir)
    wanted = ["scan", *(c for c in columns if c != "scan")]
    lo, hi = scan_range if scan_range is not None else (None, None)
    carry: dict[str, np.ndarray] | None = None

    def block_of(rows: dict[str, np.ndarray]) -> ScanBlock:
        scan = rows["scan"]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(scan)) + 1))
        offsets = np.append(starts, scan.size).astype(np.int64)
        return ScanBlock(
            scan=scan[starts].astype(np.int64),
            offsets=offsets,
            columns={c: rows[c] for c in wanted[1:]},
        )

    for path in paths:
        file = pq.ParquetFile(path)
        groups = (
            list(range(file.metadata.num_row_groups))
            if scan_range is None
            else _row_groups_in(file, lo, hi)
        )
        if not groups:
            continue
        for batch in file.iter_batches(batch_size=block_rows, row_groups=groups, columns=wanted):
            rows = {c: batch.column(i).to_numpy(zero_copy_only=False) for i, c in enumerate(wanted)}
            if scan_range is not None:
                scan = rows["scan"]
                a, b = np.searchsorted(scan, lo, "left"), np.searchsorted(scan, hi, "left")
                if a or b < scan.size:
                    rows = {c: v[a:b] for c, v in rows.items()}
            if rows["scan"].size == 0:
                continue
            if carry is not None:
                rows = {c: np.concatenate((carry[c], rows[c])) for c in wanted}
            scan = rows["scan"]
            last = int(np.searchsorted(scan, scan[-1], "left"))
            if last == 0:  # one scan so far, not known to be complete
                carry = rows
                continue
            carry = {c: v[last:] for c, v in rows.items()}
            yield block_of({c: v[:last] for c, v in rows.items()})
    if carry is not None and carry["scan"].size:
        yield block_of(carry)


#: A scan range is cut into at least this many shards when it holds the
#: points for them (a data-dependent number, so the cut never depends on
#: how many workers read it).
RANGE_SHARDS = 16


def scan_range_shards(
    bundle_dir: Path | str,
    *,
    target_rows: int = 16 << 20,
    points: bool = False,
    scan_range: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Cut a bundle into contiguous scan ranges ``[lo, hi)`` of about
    ``target_rows`` points each, at row-group starts, from footer statistics
    alone. The cut depends on the data, not on how many workers will read it.
    A bundle whose footers lack scan statistics is one range.

    With ``scan_range`` only that range is cut, by the points of the row
    groups it overlaps, into pieces of at most ``target_rows`` and at most
    1/:data:`RANGE_SHARDS` of the range; a row group is split at scans
    interpolated by its rows where it alone holds more than a piece."""
    paths = point_paths(bundle_dir) if points else peak_paths(bundle_dir)
    groups: list[tuple[int, int, int]] = []  # (first scan, last scan, rows) per row group
    for path in paths:
        file = pq.ParquetFile(path)
        for i, b in enumerate(_row_group_scan_bounds(file)):
            if b is None:
                whole = (np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                return [scan_range if scan_range is not None else whole]
            groups.append((b[0], b[1], file.metadata.row_group(i).num_rows))
    if not groups:
        return []
    if scan_range is not None:
        return _cut_range(groups, target_rows, scan_range)
    cuts = [groups[0][0]]
    rows = 0
    for first, _last, n in groups:
        if rows >= target_rows and first > cuts[-1]:
            cuts.append(first)
            rows = 0
        rows += n
    # The last scan of any row group: the last part's, unless it is empty
    # (a centroid of a partial scan list, or a range with no peaks), which
    # has no row group and raised (until 2026-11-10).
    end = max(last for _first, last, _n in groups) + 1
    return list(zip(cuts, [*cuts[1:], end]))


def _cut_range(groups: list[tuple[int, int, int]], target_rows: int, scan_range: tuple[int, int]) -> list[tuple[int, int]]:
    lo, hi = int(scan_range[0]), int(scan_range[1])
    # Each overlapping row group's share of the range, as (first, end, rows),
    # rows taken in proportion to the scans of it inside the range.
    pieces = []
    for first, last, n in groups:
        a, b = max(first, lo), min(last + 1, hi)
        if a >= b:
            continue
        pieces.append((a, b, n * (b - a) / max(last + 1 - first, 1)))
    if not pieces:
        return []
    total = sum(r for _, _, r in pieces)
    piece = max(1.0, min(float(target_rows), total / RANGE_SHARDS))
    cuts = [lo]
    rows = 0.0
    for a, b, r in pieces:
        if rows >= piece and a > cuts[-1]:
            cuts.append(a)
            rows = 0.0
        # A row group holding more than a piece is split at scans by its rows.
        while rows + r > piece and b - a > 1:
            step = max(1, int((piece - rows) / r * (b - a)))
            at = a + step
            if at >= b:
                break
            if at > cuts[-1]:
                cuts.append(at)
            r -= r * step / (b - a)
            a, rows = at, 0.0
        rows += r
    return list(zip(cuts, [*cuts[1:], hi]))


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
    while a row group is written. ``byte_stream_split`` stores ``mz`` and
    ``intensity`` with that encoding, which on Stellar profile points gave
    3.13 against 4.90 MiB per million rows.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        rt_bin_width_s: float = DEFAULT_RT_BIN_WIDTH_S,
        max_row_group_rows: int = MAX_ROW_GROUP_ROWS,
        byte_stream_split: bool = False,
    ):
        if rt_bin_width_s <= 0:
            raise ValueError(f"rt_bin_width_s must be positive, got {rt_bin_width_s}")
        self._writer = pq.ParquetWriter(
            str(path),
            MS_PEAK_TABLE,
            compression="zstd",
            use_dictionary=False,
            use_byte_stream_split=["mz", "intensity"] if byte_stream_split else False,
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
    "BUNDLE_KINDS",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_RT_BIN_WIDTH_S",
    "MANIFEST_FILENAME",
    "MAX_ROW_GROUP_ROWS",
    "MZ_AXES_DIR",
    "OutdatedBundleError",
    "PEAKS_DIR",
    "PEAK_BUNDLE_KINDS",
    "PROFILE_BUNDLE_KIND",
    "PROFILE_DIR",
    "PROFILE_SUFFIX",
    "PeakWriter",
    "has_peaks",
    "is_profile_bundle",
    "iter_peak_batches",
    "list_mz_axes",
    "open_peaks",
    "part_name",
    "peak_list_name",
    "peak_paths",
    "peak_rows",
    "peaks_dataset",
    "point_paths",
    "points_dataset",
    "read_bundle_manifest",
    "related_bundles",
]
