"""Build a derived, mass-sorted peak index from an MS_PEAK_TABLE bundle.

The index re-lays-out the canonical (scan-major, RT-binned) peaks into a
dataset partitioned by ``(ms_level, isolation_window)`` and sorted by
``mz`` then ``rt`` within each partition — so a query m/z resolves to a
contiguous slice of (rt, scan, intensity). This is the structure that
makes XIC-at-scale (and a future spectrum-centric search) scale with
query count rather than n_scans.

Partitioning regime (the load-bearing practical decision): MS1 has one
partition (null isolation bounds). DIA / SWATH MS2 has a handful of wide,
recurring windows — each becomes its own partition, and DIA extraction
falls out for free. DDA MS2, by contrast, has a near-unique isolation
window *per scan* — physically partitioning by every window would emit
tens of thousands of tiny files, so when a level's distinct-window count
exceeds ``max_isolation_windows`` the level is binned onto a synthetic
coarse precursor-m/z grid (``dda_window_width_mz``-wide cells, quasi-DIA):
the partition count stays bounded and a fragment query prunes to a couple
of bins, while the real narrow ``isolation_lower`` / ``isolation_upper``
stay as columns for exact query-time gating.

Precision is a per-axis, function-driven knob. By default the index is a
lossless f64 mirror; ``downcast`` (a subset of ``{"mz", "rt",
"intensity"}``) casts the listed axes to f32 after the sort
(``isolation_lower`` / ``isolation_upper`` follow ``mz``). Because the
canonical MS_PEAK_TABLE stays f64, downcast *values* are recoverable (the
extractor's ``exact_error`` path re-derives f64 m/z error from the canonical
table for matched rows). One caveat: downcasting ``mz`` can shift a peak
across the match-tolerance edge (~0.06 ppm), changing row *presence* — that is
NOT recoverable by ``exact_error``, so keep ``mz`` at f64 (the default) when
exact scan-major parity matters. The realized per-axis dtypes are recorded in
the manifest's ``x.massspec.precision``.

The build streams: it never holds the run. A first pass reads three
columns to find every isolation window (which fixes the partitioning); a
second assigns each batch of peaks its window and appends it to that
partition's spill file; a third sorts each partition on its own and writes
it. Memory is a batch plus the largest partition, where sorting the whole
table at once passed 23 GB on a 195M-peak DDA run (the mass-spec browser's
payload survey, docs/plans/ms-browser.md). Only numeric columns are read,
so the wide-sort hazard does not arise either.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation.core.io.schemas import pack_metadata
from constellation.massspec.schemas import (
    MS_PEAK_INDEX_SCHEMA_VERSION,
    MS_PEAK_INDEX_TABLE,
)

__all__ = [
    "PeakIndexManifest",
    "build_peak_index",
    "find_peak_index",
    "index_partition",
    "index_slice",
    "INDEX_DIRNAME",
    "MANIFEST_FILENAME",
    "DEFAULT_MAX_ISOLATION_WINDOWS",
]

MANIFEST_FILENAME = "manifest.json"
BUNDLE_KIND = "ms_peak_index"
#: Where a bundle's index lives by convention: ``<bundle>/mz_index/``.
INDEX_DIRNAME = "mz_index"
DEFAULT_MAX_ISOLATION_WINDOWS = 512

# m/z-valued axes downcast together (same physical units).
_MZ_AXES = ("mz", "isolation_lower", "isolation_upper")
# Columns read from the source. `precursor_mz` is build-time only (the
# quasi-DIA binning axis for DDA levels) and is not stored in the index.
_INDEX_COLUMNS = [
    "scan",
    "level",
    "rt",
    "mz",
    "intensity",
    "isolation_lower",
    "isolation_upper",
]
_BIN_COLUMN = "precursor_mz"
DEFAULT_DDA_WINDOW_WIDTH_MZ = 10.0
#: Peaks read per batch while building.
DEFAULT_BATCH_ROWS = 1 << 21
#: Where partitions are gathered before each is sorted; removed when done.
_SPILL_DIRNAME = ".spill"
# Sentinel for null isolation bounds during the (float-equality) window
# assignment + boundary walk — real bounds are positive m/z.
_NULL_BOUND = -1.0


@dataclass(frozen=True, slots=True)
class PeakIndexManifest:
    """Manifest for one MS_PEAK_TABLE → peak-index build."""

    schema_version: int
    kind: Literal["ms_peak_index"]
    created_at: str
    source_peaks: str
    # Build-time parameters: ``downcast`` (the f32 axis list),
    # ``max_isolation_windows`` (the partition-collapse threshold).
    parameters: dict[str, Any]
    # Realized per-axis dtypes ("f64" | "f32"), keyed mz / rt / intensity.
    precision: dict[str, str]
    # Per-level partition scheme: "native" (one partition per real
    # isolation window — MS1, DIA/SWATH) or "binned" (DDA — partitioned on
    # a synthetic coarse precursor-m/z grid; the per-row narrow isolation
    # bounds stay as columns for exact query-time gating). Keyed by level
    # (str, JSON-friendly).
    window_scheme: dict[str, str]
    # Per-binned-level maximum real isolation half-width (m/z), keyed by level
    # (str). The query layer sizes its bin-neighbor search from this so a wide
    # window covering a precursor several bins from its own center is never
    # pruned. Empty when no level is binned.
    dda_max_halfwidth_mz: dict[str, float]
    # One entry per emitted partition: level, isolation_window_id, the
    # window bounds (the real window for "native"; the synthetic bin range
    # for "binned"; null for MS1), n_rows.
    partitions: list[dict[str, Any]]
    n_windows: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _open_peaks(peaks_dir: Path) -> ds.Dataset:
    """Open a peaks source: a convert bundle (peaks.parquet), a shard
    directory, or a direct parquet file."""
    p = Path(peaks_dir)
    if p.is_dir() and (p / "peaks.parquet").exists():
        return ds.dataset(p / "peaks.parquet")
    return ds.dataset(p)


def _index_schema(downcast: frozenset[str]) -> pa.Schema:
    """MS_PEAK_INDEX_TABLE with the requested axes cast to f32."""
    fields = []
    for field in MS_PEAK_INDEX_TABLE:
        ftype = field.type
        if field.name in _MZ_AXES and "mz" in downcast:
            ftype = pa.float32()
        elif field.name == "rt" and "rt" in downcast:
            ftype = pa.float32()
        elif field.name == "intensity" and "intensity" in downcast:
            ftype = pa.float32()
        fields.append(pa.field(field.name, ftype, nullable=field.nullable))
    meta = {
        "schema_name": "MSPeakIndex",
        "schema_version": str(MS_PEAK_INDEX_SCHEMA_VERSION),
        "x.massspec.index_sort": "mz,rt",
        "x.massspec.precision": _precision_dict(downcast),
    }
    return pa.schema(fields, metadata=pack_metadata(meta))


def _precision_dict(downcast: Iterable[str]) -> dict[str, str]:
    dc = set(downcast)
    return {
        axis: ("f32" if axis in dc else "f64") for axis in ("mz", "rt", "intensity")
    }


def _bounds_keyed(table: pa.Table) -> pa.Table:
    """``table`` with ``lo_f`` / ``hi_f``: the isolation bounds with null
    (MS1) as a sentinel, so a window can be a join key."""
    lo_f = pc.fill_null(table.column("isolation_lower"), _NULL_BOUND)
    hi_f = pc.fill_null(table.column("isolation_upper"), _NULL_BOUND)
    return table.append_column("lo_f", lo_f).append_column("hi_f", hi_f)


def _distinct_windows(dataset: ds.Dataset, batch_rows: int) -> set[tuple[int, float, float]]:
    """Every distinct ``(level, isolation_lower, isolation_upper)`` of the
    peaks (null bounds as the sentinel), read three columns at a time."""
    seen: set[tuple[int, float, float]] = set()
    columns = ["level", "isolation_lower", "isolation_upper"]
    for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
        distinct = (
            _bounds_keyed(pa.Table.from_batches([batch]))
            .select(["level", "lo_f", "hi_f"])
            .group_by(["level", "lo_f", "hi_f"])
            .aggregate([])
        )
        seen.update(
            zip(
                (int(v) for v in distinct.column("level").to_pylist()),
                distinct.column("lo_f").to_pylist(),
                distinct.column("hi_f").to_pylist(),
            )
        )
    return seen


def _window_map(
    distinct: set[tuple[int, float, float]],
    level_type: pa.DataType,
    max_isolation_windows: int,
) -> tuple[pa.Table, dict[int, str]]:
    """Number the windows and decide each level's scheme.

    A level is "native" when its distinct non-null window count is ≤
    ``max_isolation_windows`` (MS1 → one null window; DIA/SWATH → a handful
    of real windows): each distinct ``(isolation_lower, isolation_upper)``
    becomes its own window. Above the cap the level is "binned" (DDA; see
    :func:`_with_window_ids`). Windows are numbered in sorted
    ``(level, lower, upper)`` order across all levels, binned ones included,
    so the ids do not depend on how the peaks were read.
    """
    per_level: dict[int, list[tuple[float, float]]] = {}
    for level, lo, hi in sorted(distinct):
        per_level.setdefault(level, []).append((lo, hi))
    scheme: dict[int, str] = {}
    map_level, map_lo, map_hi, map_cid = [], [], [], []
    combo_id = 0
    for level, combos in per_level.items():
        scheme[level] = "binned" if len(combos) > max_isolation_windows else "native"
        for lo, hi in combos:
            map_level.append(level)
            map_lo.append(lo)
            map_hi.append(hi)
            map_cid.append(combo_id)
            combo_id += 1
    map_tbl = pa.table(
        {
            "level": pa.array(map_level, level_type),
            "lo_f": pa.array(map_lo, pa.float64()),
            "hi_f": pa.array(map_hi, pa.float64()),
            "combo_id": pa.array(map_cid, pa.int32()),
        }
    )
    return map_tbl, scheme


def _with_window_ids(
    peaks: pa.Table,
    map_tbl: pa.Table,
    scheme: dict[int, str],
    dda_window_width_mz: float,
) -> tuple[pa.Table, dict[int, float]]:
    """Append an ``isolation_window_id`` column to a batch of peaks.

    Native levels take the window's number from ``map_tbl`` (an Arrow join,
    no per-row Python loop). A binned level's id is
    ``floor(precursor_mz / dda_window_width_mz)`` — a synthetic coarse
    precursor-m/z grid that bounds the partition count and lets a fragment
    query prune to a couple of bins, while the real narrow isolation bounds
    stay as columns for exact query-time gating. Binning falls back to the
    isolation-window center when ``precursor_mz`` is absent.

    Also returns, per binned level, the largest real isolation half-width
    in the batch: the query layer sizes its bin-neighbor search from it, so
    a wide window covering a precursor several bins away is never pruned.
    """
    # NOTE: join does not preserve row order — derive every array below
    # from `joined`, not from the input.
    joined = _bounds_keyed(peaks).join(map_tbl, keys=["level", "lo_f", "hi_f"], join_type="left outer")

    levels = joined.column("level").to_numpy(zero_copy_only=False)
    combo = joined.column("combo_id").to_numpy(zero_copy_only=False).astype(np.int64)
    lo = joined.column("isolation_lower").to_numpy(zero_copy_only=False)
    hi = joined.column("isolation_upper").to_numpy(zero_copy_only=False)
    center = (lo + hi) / 2.0  # NaN where bounds are null (MS1)
    if _BIN_COLUMN in joined.column_names:
        prec = joined.column(_BIN_COLUMN).to_numpy(zero_copy_only=False)
        bin_value = np.where(np.isnan(prec), center, prec)
    else:
        bin_value = center
    binned = np.where(np.isnan(bin_value), 0, np.floor(bin_value / dda_window_width_mz))
    binned = binned.astype(np.int64)

    binned_levels = np.array([lvl for lvl, s in scheme.items() if s == "binned"], dtype=np.int64)
    levels_i = levels.astype(np.int64)
    is_binned = np.isin(levels_i, binned_levels)
    window_id = np.where(is_binned, binned, combo).astype(np.int32)

    halfwidth = (hi - lo) / 2.0  # NaN where bounds null (MS1)
    max_halfwidth: dict[int, float] = {}
    for lvl in binned_levels:
        hw = halfwidth[levels_i == lvl]
        hw = hw[~np.isnan(hw)]
        if hw.size:
            max_halfwidth[int(lvl)] = float(hw.max())

    drop = ["lo_f", "hi_f", "combo_id"]
    if _BIN_COLUMN in joined.column_names:
        drop.append(_BIN_COLUMN)
    out = joined.drop_columns(drop).append_column(
        "isolation_window_id", pa.array(window_id, pa.int32())
    )
    return out, max_halfwidth


def _partition_boundaries(
    level: np.ndarray, window_id: np.ndarray
) -> list[tuple[int, int]]:
    """Contiguous (start, size) runs of constant (level, window_id)."""
    n = level.shape[0]
    if n == 0:
        return []
    combined = level.astype(np.int64) * (2**32) + window_id.astype(np.int64)
    change = np.flatnonzero(combined[1:] != combined[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def build_peak_index(
    peaks_dir: Path | str,
    index_root: Path | str,
    *,
    downcast: Sequence[str] = (),
    max_isolation_windows: int = DEFAULT_MAX_ISOLATION_WINDOWS,
    dda_window_width_mz: float = DEFAULT_DDA_WINDOW_WIDTH_MZ,
    row_group_size: int | None = None,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    created_at: str | None = None,
) -> PeakIndexManifest:
    """Build a mass-sorted peak index from an MS_PEAK_TABLE bundle.

    Parameters
    ----------
    peaks_dir : Path | str
        A convert bundle (containing ``peaks.parquet``), a shard
        directory, or a direct parquet file of MS_PEAK_TABLE rows.
    index_root : Path | str
        Output root; partitions land at
        ``<root>/ms_level=<L>/isolation_window=<W>/part-00000.parquet``.
    downcast : sequence of {"mz", "rt", "intensity"}
        Axes to store as f32 (default none → lossless f64 mirror).
        ``mz`` also downcasts the isolation bounds (same units).
    max_isolation_windows : int
        Per-level distinct-window cap above which a level switches from
        native (one partition per real window) to binned (a synthetic
        coarse precursor-m/z grid). Default 512.
    dda_window_width_mz : float
        Grid width (m/z) for binned (DDA) levels. Default 10.0.
    row_group_size : int, optional
        Rows per row group of each partition file (pyarrow's default when
        omitted). Smaller groups make an m/z slice read less.
    batch_rows : int
        Peaks read per batch. Memory is about a batch plus the largest
        partition.

    Returns
    -------
    PeakIndexManifest
        Also written to ``<index_root>/manifest.json``, last, so a finished
        index is one with a manifest.
    """
    invalid = set(downcast) - {"mz", "rt", "intensity"}
    if invalid:
        raise ValueError(
            f"downcast axes must be subset of mz/rt/intensity, got {invalid}"
        )
    downcast_set = frozenset(downcast)
    index_root = Path(index_root)
    index_root.mkdir(parents=True, exist_ok=True)
    (index_root / MANIFEST_FILENAME).unlink(missing_ok=True)

    dataset = _open_peaks(Path(peaks_dir))
    columns = list(_INDEX_COLUMNS)
    if _BIN_COLUMN in dataset.schema.names:
        columns.append(_BIN_COLUMN)
    out_schema = _index_schema(downcast_set)
    parameters = {
        "downcast": sorted(downcast_set),
        "max_isolation_windows": max_isolation_windows,
        "dda_window_width_mz": dda_window_width_mz,
    }

    # Pass 1: the windows, which fix the partitions and their ids.
    distinct = _distinct_windows(dataset, batch_rows)
    map_tbl, scheme = _window_map(distinct, dataset.schema.field("level").type, max_isolation_windows)
    max_halfwidth = {lvl: 0.0 for lvl, s in scheme.items() if s == "binned"}

    # Pass 2: each batch's peaks appended to their partition's spill file.
    spill = index_root / _SPILL_DIRNAME
    shutil.rmtree(spill, ignore_errors=True)
    spill.mkdir()
    writers: dict[tuple[int, int], pq.ParquetWriter] = {}
    try:
        for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
            if batch.num_rows == 0:
                continue
            keyed, halfwidths = _with_window_ids(
                pa.Table.from_batches([batch]), map_tbl, scheme, dda_window_width_mz
            )
            for lvl, hw in halfwidths.items():
                max_halfwidth[lvl] = max(max_halfwidth.get(lvl, 0.0), hw)
            keyed = keyed.sort_by([("level", "ascending"), ("isolation_window_id", "ascending")])
            level_np = keyed.column("level").to_numpy(zero_copy_only=False)
            wid_np = keyed.column("isolation_window_id").to_numpy(zero_copy_only=False)
            for start, size in _partition_boundaries(level_np, wid_np):
                key = (int(level_np[start]), int(wid_np[start]))
                part = keyed.slice(start, size)
                writer = writers.get(key)
                if writer is None:
                    writer = pq.ParquetWriter(spill / f"{key[0]}-{key[1]}.parquet", part.schema, compression="lz4")
                    writers[key] = writer
                writer.write_table(part)
    finally:
        for writer in writers.values():
            writer.close()

    # Pass 3: each partition sorted on its own and written.
    partitions: list[dict[str, Any]] = []
    for level, wid in sorted(writers):
        part = pq.read_table(spill / f"{level}-{wid}.parquet")
        ordered = part.sort_by([("mz", "ascending"), ("rt", "ascending")])
        if scheme.get(level) == "binned":
            # The recorded bounds are the synthetic grid cell, not the
            # per-row narrow window (which varies within the bin).
            lo = wid * dda_window_width_mz
            hi = (wid + 1) * dda_window_width_mz
        else:
            lo = ordered.column("isolation_lower")[0].as_py()
            hi = ordered.column("isolation_upper")[0].as_py()
        out = ordered.select(
            ["mz", "rt", "scan", "intensity", "isolation_window_id", "isolation_lower", "isolation_upper"]
        ).cast(out_schema)
        out_dir = index_root / f"ms_level={level}" / f"isolation_window={wid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, out_dir / "part-00000.parquet", row_group_size=row_group_size)
        partitions.append(
            {
                "level": level,
                "isolation_window_id": wid,
                "isolation_lower": lo,
                "isolation_upper": hi,
                "n_rows": out.num_rows,
            }
        )
        (spill / f"{level}-{wid}.parquet").unlink()
    shutil.rmtree(spill, ignore_errors=True)

    manifest = PeakIndexManifest(
        schema_version=MS_PEAK_INDEX_SCHEMA_VERSION,
        kind=BUNDLE_KIND,
        created_at=created_at or _now_iso(),
        source_peaks=str(peaks_dir),
        parameters=parameters,
        precision=_precision_dict(downcast_set),
        window_scheme={str(lvl): s for lvl, s in scheme.items()} if partitions else {},
        dda_max_halfwidth_mz={str(lvl): hw for lvl, hw in max_halfwidth.items()} if partitions else {},
        partitions=partitions,
        n_windows=len(partitions),
    )
    _write_manifest(index_root, manifest)
    return manifest


def _write_manifest(index_root: Path, manifest: PeakIndexManifest) -> None:
    (index_root / MANIFEST_FILENAME).write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )


def find_peak_index(bundle_dir: Path | str) -> Path | None:
    """The finished peak index at ``<bundle>/mz_index``, or ``None``.

    Finished means its manifest is there: the build writes it last.
    """
    root = Path(bundle_dir) / INDEX_DIRNAME
    manifest = root / MANIFEST_FILENAME
    if not manifest.is_file():
        return None
    kind = json.loads(manifest.read_text(encoding="utf-8")).get("kind")
    return root if kind == BUNDLE_KIND else None


def index_partition(
    index_root: Path | str,
    *,
    level: int,
    isolation_lower: float | None = None,
    isolation_upper: float | None = None,
) -> int | None:
    """The partition (``isolation_window_id``) holding ``level``'s peaks in
    the isolation window ``[isolation_lower, isolation_upper]``.

    A level with one partition (MS1) needs no bounds. Bounds are compared
    at float32, as windows are everywhere else. ``None`` when nothing
    matches, or when a binned (DDA) level is asked for: its partitions are
    m/z bins, not windows.
    """
    manifest = read_manifest(index_root)
    parts = [p for p in manifest.partitions if p["level"] == level]
    if manifest.window_scheme.get(str(level)) == "binned":
        return None
    if isolation_lower is None or isolation_upper is None:
        return int(parts[0]["isolation_window_id"]) if len(parts) == 1 else None
    want = (np.float32(isolation_lower), np.float32(isolation_upper))
    for part in parts:
        lo, hi = part["isolation_lower"], part["isolation_upper"]
        if lo is not None and hi is not None and (np.float32(lo), np.float32(hi)) == want:
            return int(part["isolation_window_id"])
    return None


def index_slice(
    index_root: Path | str,
    *,
    level: int,
    window_id: int,
    mz_lo: float,
    mz_hi: float,
    rt_lo: float | None = None,
    rt_hi: float | None = None,
    columns: Sequence[str] = ("mz", "rt", "scan", "intensity"),
) -> pa.Table:
    """The peaks of one partition with m/z in ``[mz_lo, mz_hi]`` (and RT in
    ``[rt_lo, rt_hi]`` when given), in m/z order.

    A partition is sorted by m/z, so its row groups' m/z statistics let the
    read skip all but the groups the range touches. An absent partition
    gives an empty table.
    """
    path = Path(index_root) / f"ms_level={level}" / f"isolation_window={window_id}" / "part-00000.parquet"
    if not path.exists():
        return _index_schema(frozenset()).empty_table().select(list(columns))
    condition = (ds.field("mz") >= mz_lo) & (ds.field("mz") <= mz_hi)
    if rt_lo is not None:
        condition &= ds.field("rt") >= rt_lo
    if rt_hi is not None:
        condition &= ds.field("rt") <= rt_hi
    return ds.dataset(path).to_table(columns=list(columns), filter=condition)


def read_manifest(index_root: Path | str) -> PeakIndexManifest:
    """Load a peak-index manifest."""
    data = json.loads(
        (Path(index_root) / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if data.get("schema_version") != MS_PEAK_INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"peak-index manifest version {data.get('schema_version')!r} unsupported; rebuild"
        )
    return PeakIndexManifest(**data)
