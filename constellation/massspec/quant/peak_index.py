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
canonical MS_PEAK_TABLE stays f64, any downcast is recoverable. The
realized per-axis dtypes are recorded in the manifest's
``x.massspec.precision``.

The build reads the projected (all-numeric) peak columns once and does a
single global ``sort_by`` — no string columns enter the sort, so the
wide-sort hazard does not apply. At extreme (≫1e9-peak) scale this lifts
to an external/streaming sort; that is a single-function refactor, not a
new public API.
"""

from __future__ import annotations

import json
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
    "MANIFEST_FILENAME",
    "DEFAULT_MAX_ISOLATION_WINDOWS",
]

MANIFEST_FILENAME = "manifest.json"
BUNDLE_KIND = "ms_peak_index"
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


def _assign_window_ids(
    peaks: pa.Table, max_isolation_windows: int, dda_window_width_mz: float
) -> tuple[pa.Table, dict[int, str]]:
    """Append an ``isolation_window_id`` column.

    Returns the augmented table and a ``{level: scheme}`` map. A level is
    "native" when its distinct non-null window count is ≤
    ``max_isolation_windows`` (MS1 → one null window; DIA/SWATH → a handful
    of real windows): each distinct ``(isolation_lower, isolation_upper)``
    becomes its own window, assigned via an Arrow join (no per-row Python
    loop). Above the cap the level is "binned" (DDA): each row's
    ``isolation_window_id`` is ``floor(precursor_mz / dda_window_width_mz)``
    — a synthetic coarse precursor-m/z grid that bounds the partition count
    and lets a fragment query prune to a couple of bins, while the real
    narrow isolation bounds stay as columns for exact query-time gating.
    Binning falls back to the isolation-window center when ``precursor_mz``
    is absent.
    """
    lo_f = pc.fill_null(peaks.column("isolation_lower"), _NULL_BOUND)
    hi_f = pc.fill_null(peaks.column("isolation_upper"), _NULL_BOUND)
    keyed = peaks.append_column("lo_f", lo_f).append_column("hi_f", hi_f)

    distinct = (
        keyed.select(["level", "lo_f", "hi_f"])
        .group_by(["level", "lo_f", "hi_f"])
        .aggregate([])
    )
    rows = sorted(
        zip(
            distinct.column("level").to_pylist(),
            distinct.column("lo_f").to_pylist(),
            distinct.column("hi_f").to_pylist(),
        )
    )
    per_level: dict[int, list[tuple[float, float]]] = {}
    for level, lo, hi in rows:
        per_level.setdefault(int(level), []).append((lo, hi))

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
            "level": pa.array(map_level, peaks.schema.field("level").type),
            "lo_f": pa.array(map_lo, pa.float64()),
            "hi_f": pa.array(map_hi, pa.float64()),
            "combo_id": pa.array(map_cid, pa.int32()),
        }
    )
    # NOTE: join does not preserve row order — derive every array below
    # from `joined`, not from `keyed`.
    joined = keyed.join(map_tbl, keys=["level", "lo_f", "hi_f"], join_type="left outer")

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

    binned_levels = np.array(
        [lvl for lvl, s in scheme.items() if s == "binned"], dtype=np.int64
    )
    is_binned = np.isin(levels.astype(np.int64), binned_levels)
    window_id = np.where(is_binned, binned, combo).astype(np.int32)

    drop = ["lo_f", "hi_f", "combo_id"]
    if _BIN_COLUMN in joined.column_names:
        drop.append(_BIN_COLUMN)
    out = joined.drop_columns(drop).append_column(
        "isolation_window_id", pa.array(window_id, pa.int32())
    )
    return out, scheme


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
        Per-level distinct-window cap above which the level switches from
        native (one partition per real window) to binned (a synthetic
        coarse precursor-m/z grid). Default 512.
    dda_window_width_mz : float
        Grid width (m/z) for binned (DDA) levels. Default 10.0.

    Returns
    -------
    PeakIndexManifest
        Also written to ``<index_root>/manifest.json``.
    """
    invalid = set(downcast) - {"mz", "rt", "intensity"}
    if invalid:
        raise ValueError(
            f"downcast axes must be subset of mz/rt/intensity, got {invalid}"
        )
    downcast_set = frozenset(downcast)
    index_root = Path(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    dataset = _open_peaks(Path(peaks_dir))
    columns = list(_INDEX_COLUMNS)
    if _BIN_COLUMN in dataset.schema.names:
        columns.append(_BIN_COLUMN)
    peaks = dataset.to_table(columns=columns)
    out_schema = _index_schema(downcast_set)
    parameters = {
        "downcast": sorted(downcast_set),
        "max_isolation_windows": max_isolation_windows,
        "dda_window_width_mz": dda_window_width_mz,
    }

    if peaks.num_rows == 0:
        manifest = PeakIndexManifest(
            schema_version=MS_PEAK_INDEX_SCHEMA_VERSION,
            kind=BUNDLE_KIND,
            created_at=created_at or _now_iso(),
            source_peaks=str(peaks_dir),
            parameters=parameters,
            precision=_precision_dict(downcast_set),
            window_scheme={},
            partitions=[],
            n_windows=0,
        )
        _write_manifest(index_root, manifest)
        return manifest

    keyed, scheme = _assign_window_ids(
        peaks, max_isolation_windows, dda_window_width_mz
    )
    ordered = keyed.sort_by(
        [
            ("level", "ascending"),
            ("isolation_window_id", "ascending"),
            ("mz", "ascending"),
            ("rt", "ascending"),
        ]
    ).combine_chunks()

    level_np = ordered.column("level").to_numpy(zero_copy_only=False)
    wid_np = ordered.column("isolation_window_id").to_numpy(zero_copy_only=False)
    bounds = _partition_boundaries(level_np, wid_np)

    partitions: list[dict[str, Any]] = []
    for start, size in bounds:
        part = ordered.slice(start, size)
        level = int(part.column("level")[0].as_py())
        wid = int(part.column("isolation_window_id")[0].as_py())
        if scheme.get(level) == "binned":
            # The recorded bounds are the synthetic grid cell, not the
            # per-row narrow window (which varies within the bin).
            lo = wid * dda_window_width_mz
            hi = (wid + 1) * dda_window_width_mz
        else:
            lo = part.column("isolation_lower")[0].as_py()
            hi = part.column("isolation_upper")[0].as_py()

        out = part.select(
            [
                "mz",
                "rt",
                "scan",
                "intensity",
                "isolation_window_id",
                "isolation_lower",
                "isolation_upper",
            ]
        ).cast(out_schema)

        out_dir = index_root / f"ms_level={level}" / f"isolation_window={wid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, out_dir / "part-00000.parquet")
        partitions.append(
            {
                "level": level,
                "isolation_window_id": wid,
                "isolation_lower": lo,
                "isolation_upper": hi,
                "n_rows": size,
            }
        )

    manifest = PeakIndexManifest(
        schema_version=MS_PEAK_INDEX_SCHEMA_VERSION,
        kind=BUNDLE_KIND,
        created_at=created_at or _now_iso(),
        source_peaks=str(peaks_dir),
        parameters=parameters,
        precision=_precision_dict(downcast_set),
        window_scheme={str(lvl): s for lvl, s in scheme.items()},
        partitions=partitions,
        n_windows=len(partitions),
    )
    _write_manifest(index_root, manifest)
    return manifest


def _write_manifest(index_root: Path, manifest: PeakIndexManifest) -> None:
    (index_root / MANIFEST_FILENAME).write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )


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
