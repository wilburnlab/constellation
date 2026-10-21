"""Random access to one converted acquisition, for browsing it.

A bundle (``massspec convert`` from a Thermo ``.raw``, or ``massspec
encyclopedia import-spectra`` from an EncyclopeDIA spectra file) is a
directory of ``peaks.parquet`` (one row per peak, in scan order),
``scan_metadata.parquet`` (one row per scan) and a manifest. The batch code
in ``quant`` reads it whole; this module reads the pieces an interactive
view needs, quickly:

- :func:`open_acquisition` — what the run is (kind, DIA or DDA, analyzers,
  levels, RT range, whether a peak index was built);
- :func:`scan_table` — the scans a view navigates by, with each scan's
  isolation-window id and parent scan derived once;
- :func:`isolation_window_table`, :func:`chromatogram`,
  :func:`nearest_scan`, :func:`parent_scan`, :func:`dependent_scans` —
  pure functions over that table;
- :func:`spectrum` / :func:`spectra` — peaks of one or several scans, read
  through row-group pruning on the scan number (rows are in scan order, so
  a scan sits in one or two row groups);
- :func:`xic` — extracted-ion chromatograms of a set of m/z values in one
  level (and DIA window), from the peak index when the bundle has one and
  otherwise from ``peaks.parquet`` within a retention-time window;
- :func:`locate_precursor` — the window a precursor was isolated in and
  the scans nearest its apex.

Retention times are seconds throughout, as in the bundle.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation.massspec.acquisition_structure import (
    MIN_VISITS_PER_DIA_WINDOW,
    AcquisitionType,
    infer_acquisition_structure,
)
from constellation.massspec.quant.peak_index import (
    find_peak_index,
    index_partition,
    index_slice,
)

#: The bundle kinds this module reads.
BUNDLE_KINDS = ("thermo_acquisition", "encyclopedia_dia_acquisition")

#: Scan-metadata columns :func:`scan_table` reads by default: enough to draw
#: total-ion and base-peak chromatograms and to move between scans.
NAV_COLUMNS = (
    "scan",
    "level",
    "rt",
    "tic",
    "base_peak_mz",
    "base_peak_intensity",
    "isolation_lower",
    "isolation_upper",
    "precursor_mz",
    "precursor_charge",
    "precursor_scan",
)

#: The match tolerance to assume for a scan when nothing better is known (a
#: search records its own). Low-resolution analyzers get the lab's ion-trap
#: setting; anything else the project default.
DEFAULT_TOLERANCES: dict[str, tuple[float, Literal["ppm", "Da"]]] = {
    "ITMS": (0.35, "Da"),
    "TQMS": (0.35, "Da"),
    "SQMS": (0.35, "Da"),
    "FTMS": (10.0, "ppm"),
    "TOFMS": (20.0, "ppm"),
}
FALLBACK_TOLERANCE: tuple[float, Literal["ppm", "Da"]] = (20.0, "ppm")


@dataclass(frozen=True, slots=True)
class AcquisitionInfo:
    """What a bundle holds, read from its manifest and scan table."""

    bundle: Path
    #: ``thermo_acquisition`` or ``encyclopedia_dia_acquisition``.
    kind: str
    #: The run's name as a search records it (``SourceFile``), and its
    #: SHA-256 where the bundle was converted from the instrument file.
    source_file: str
    source_sha256: str | None
    #: The peak index root, if one was built at ``<bundle>/mz_index``.
    index: Path | None
    #: ``None`` for a run with no MS2 scans.
    acquisition_type: AcquisitionType | None
    staggered: bool
    profile_mode: bool
    levels: tuple[int, ...]
    scans_per_level: dict[int, int]
    #: Analyzers per MS level. A bundle imported from a spectra file records
    #: them only for the whole run, so every level gets that list.
    analyzers: dict[int, tuple[str, ...]]
    rt_range: tuple[float, float]


def _manifest(bundle: Path) -> dict:
    path = bundle / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{bundle} is not a converted acquisition (no manifest.json)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in BUNDLE_KINDS:
        raise ValueError(
            f"{bundle} holds a {manifest.get('kind')!r}, not a converted acquisition "
            f"({' or '.join(BUNDLE_KINDS)})"
        )
    return manifest


def open_acquisition(bundle_dir: Path | str) -> AcquisitionInfo:
    """Describe the converted acquisition in ``bundle_dir``.

    Raises ``ValueError`` for a directory that is not one.
    """
    bundle = Path(bundle_dir)
    manifest = _manifest(bundle)
    meta = pq.read_table(
        bundle / "scan_metadata.parquet",
        columns=["scan", "level", "rt", "analyzer", "isolation_lower", "isolation_upper"],
    )
    acq = pq.read_table(bundle / "acquisition_metadata.parquet").to_pylist()
    run = acq[0] if acq else {}

    levels = tuple(sorted(int(v) for v in pc.unique(meta.column("level")).to_pylist()))
    counts = {
        int(row["values"]): int(row["counts"])
        for row in pc.value_counts(meta.column("level")).to_pylist()
    }
    run_analyzers = tuple(run.get("analyzers") or ())
    analyzers: dict[int, tuple[str, ...]] = {}
    for level in levels:
        at_level = meta.filter(pc.equal(meta.column("level"), level)).column("analyzer")
        seen = tuple(sorted(a for a in pc.unique(at_level).to_pylist() if a))
        analyzers[level] = seen or run_analyzers

    try:
        structure = infer_acquisition_structure(meta)
        acquisition_type, staggered = structure.acquisition_type, structure.is_staggered
    except ValueError:  # no MS2 scans, or none with isolation bounds
        acquisition_type, staggered = None, False

    rt = meta.column("rt")
    rt_range = (
        (float(pc.min(rt).as_py()), float(pc.max(rt).as_py())) if meta.num_rows else (0.0, 0.0)
    )
    profile = bool((manifest.get("parameters") or {}).get("profile_mode", run.get("profile_mode")))
    return AcquisitionInfo(
        bundle=bundle,
        kind=manifest["kind"],
        source_file=str(manifest.get("source_file") or bundle.name),
        source_sha256=manifest.get("source_sha256"),
        index=find_peak_index(bundle),
        acquisition_type=acquisition_type,
        staggered=staggered,
        profile_mode=profile,
        levels=levels,
        scans_per_level=counts,
        analyzers=analyzers,
        rt_range=rt_range,
    )


def default_tolerance(analyzers: Iterable[str]) -> tuple[float, Literal["ppm", "Da"]]:
    """The tolerance to assume for scans from ``analyzers`` (the loosest if mixed)."""
    known = [DEFAULT_TOLERANCES[a] for a in analyzers if a in DEFAULT_TOLERANCES]
    if not known:
        return FALLBACK_TOLERANCE
    in_da = [t for t in known if t[1] == "Da"]
    return max(in_da) if in_da else max(known)


# ── the scan table ────────────────────────────────────────────────────────


def scan_table(
    bundle_dir: Path | str,
    *,
    columns: Sequence[str] = NAV_COLUMNS,
    levels: Iterable[int] | None = None,
) -> pa.Table:
    """The bundle's scans, in scan order, with two derived columns.

    ``window`` (int32) numbers the distinct isolation windows of each DIA
    level in order of their lower bound, compared at float32 as a spectra
    file stores them; it is null for MS1 and for a level whose windows are
    not revisited (DDA, where nearly every scan has its own). ``parent_scan``
    (int32) is the scan a higher-level scan was taken from: its recorded
    precursor scan, else the last scan of the level below before it. Columns
    a bundle predates are null.
    """
    path = Path(bundle_dir) / "scan_metadata.parquet"
    available = set(pq.read_schema(path).names)
    wanted = list(dict.fromkeys(["scan", "level", "rt", "isolation_lower", "isolation_upper",
                                 "precursor_scan", *columns]))  # fmt: skip
    table = pq.read_table(path, columns=[c for c in wanted if c in available])
    for name in wanted:
        if name not in available:
            table = table.append_column(name, pa.nulls(table.num_rows, pa.float64()))
    if levels is not None:
        table = table.filter(pc.is_in(table.column("level"), pa.array(list(levels), pa.int8())))
    table = table.sort_by("scan")

    level = table.column("level").to_numpy(zero_copy_only=False).astype(np.int64)
    window = _window_ids(table, level)
    parent = _parent_scans(table, level)
    keep = [c for c in dict.fromkeys(columns) if c in table.column_names]
    for required in ("scan", "level", "rt"):
        if required not in keep:
            keep.insert(0, required)
    out = table.select(keep)
    return out.append_column("window", window).append_column("parent_scan", parent)


def _window_ids(table: pa.Table, level: np.ndarray) -> pa.Array:
    lower = table.column("isolation_lower").to_numpy(zero_copy_only=False).astype(np.float32)
    upper = table.column("isolation_upper").to_numpy(zero_copy_only=False).astype(np.float32)
    ids = np.full(level.size, -1, dtype=np.int32)
    for lvl in np.unique(level[level > 1]):
        at = (level == lvl) & ~(np.isnan(lower) | np.isnan(upper))
        if not at.any():
            continue
        pairs, inverse = np.unique(np.stack([lower[at], upper[at]], axis=1), axis=0, return_inverse=True)
        if at.sum() / len(pairs) < MIN_VISITS_PER_DIA_WINDOW:
            continue  # DDA: a window per scan names nothing worth navigating by
        ids[at] = inverse.reshape(-1)
    return pa.array(ids, pa.int32(), mask=ids < 0)


def _parent_scans(table: pa.Table, level: np.ndarray) -> pa.Array:
    scan = table.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
    recorded = table.column("precursor_scan").to_numpy(zero_copy_only=False)
    parent = np.full(scan.size, -1, dtype=np.int64)
    for lvl in np.unique(level[level > 1]):
        below = np.flatnonzero(level == lvl - 1)
        at = np.flatnonzero(level == lvl)
        if below.size:
            # Rows are in scan order, so the last lower-level row before each
            # scan is found by position.
            before = np.searchsorted(below, at, side="left") - 1
            ok = before >= 0
            parent[at[ok]] = scan[below[before[ok]]]
        given = ~np.isnan(recorded[at]) & (recorded[at] > 0)
        parent[at[given]] = recorded[at[given]].astype(np.int64)
    return pa.array(parent.astype(np.int32), pa.int32(), mask=parent < 0)


def isolation_window_table(scans: pa.Table) -> pa.Table:
    """The DIA isolation windows of a :func:`scan_table`: one row per ``(level, window)``.

    Columns ``level``, ``window``, ``isolation_lower``, ``isolation_upper``
    (float32 values, as compared), ``center``, ``n_scans``; sorted by level
    then lower bound. Empty for a DDA run.
    """
    have = scans.filter(pc.is_valid(scans.column("window")))
    if have.num_rows == 0:
        schema = pa.schema(
            [("level", pa.int8()), ("window", pa.int32()), ("isolation_lower", pa.float64()),
             ("isolation_upper", pa.float64()), ("center", pa.float64()), ("n_scans", pa.int64())]
        )  # fmt: skip
        return schema.empty_table()
    grouped = have.group_by(["level", "window"]).aggregate(
        [("isolation_lower", "min"), ("isolation_upper", "min"), ("scan", "count")]
    )
    lower = grouped.column("isolation_lower_min").to_numpy().astype(np.float32).astype(np.float64)
    upper = grouped.column("isolation_upper_min").to_numpy().astype(np.float32).astype(np.float64)
    out = pa.table(
        {
            "level": grouped.column("level"),
            "window": grouped.column("window"),
            "isolation_lower": lower,
            "isolation_upper": upper,
            "center": (lower + upper) / 2.0,
            "n_scans": grouped.column("scan_count"),
        }
    )
    return out.sort_by([("level", "ascending"), ("isolation_lower", "ascending")])


# ── chromatograms and navigation ──────────────────────────────────────────


def _select(scans: pa.Table, level: int, window: int | None) -> pa.Table:
    mask = pc.equal(scans.column("level"), level)
    if window is not None:
        mask = pc.and_(mask, pc.equal(scans.column("window"), window))
    return scans.filter(mask)


def chromatogram(
    scans: pa.Table,
    *,
    level: int,
    kind: Literal["tic", "bpc"] = "tic",
    window: int | None = None,
    rt_range: tuple[float, float] | None = None,
) -> pa.Table:
    """A total-ion or base-peak chromatogram from a :func:`scan_table`.

    Columns ``scan``, ``rt``, ``intensity``, plus ``mz`` (the base peak's)
    for ``kind="bpc"``. ``window`` restricts to one DIA window. A bundle's
    TIC is the instrument's for a Thermo conversion and the sum of the
    stored peaks for one imported from a spectra file. Raises
    ``ValueError`` for a base-peak chromatogram of a bundle that records no
    base peaks (imported before they were derived; import it again).
    """
    sub = _select(scans, level, window)
    if rt_range is not None:
        rt = sub.column("rt")
        sub = sub.filter(pc.and_(pc.greater_equal(rt, rt_range[0]), pc.less_equal(rt, rt_range[1])))
    if kind == "tic":
        return pa.table({"scan": sub.column("scan"), "rt": sub.column("rt"),
                         "intensity": sub.column("tic").cast(pa.float64())})  # fmt: skip
    if kind != "bpc":
        raise ValueError(f"kind must be 'tic' or 'bpc', not {kind!r}")
    if "base_peak_intensity" not in sub.column_names or (
        sub.num_rows and sub.column("base_peak_intensity").null_count == sub.num_rows
    ):
        raise ValueError(
            "this bundle records no base peaks; a bundle imported from a spectra file "
            "before they were derived has to be imported again"
        )
    return pa.table(
        {
            "scan": sub.column("scan"),
            "rt": sub.column("rt"),
            "intensity": sub.column("base_peak_intensity").cast(pa.float64()),
            "mz": sub.column("base_peak_mz").cast(pa.float64()),
        }
    )


def nearest_scan(
    scans: pa.Table, rt: float, *, level: int, window: int | None = None
) -> int | None:
    """The scan of ``level`` (and ``window``) closest in time to ``rt`` seconds."""
    sub = _select(scans, level, window)
    if sub.num_rows == 0:
        return None
    times = sub.column("rt").to_numpy(zero_copy_only=False)
    return int(sub.column("scan")[int(np.argmin(np.abs(times - rt)))].as_py())


def _row(scans: pa.Table, scan: int) -> int:
    hits = np.flatnonzero(scans.column("scan").to_numpy(zero_copy_only=False) == scan)
    if hits.size == 0:
        raise KeyError(f"no scan {scan}")
    return int(hits[0])


def parent_scan(scans: pa.Table, scan: int) -> int | None:
    """The scan ``scan`` was taken from (see :func:`scan_table`); ``None`` for MS1."""
    return scans.column("parent_scan")[_row(scans, scan)].as_py()


def dependent_scans(scans: pa.Table, scan: int) -> list[int]:
    """The scans taken from ``scan``, in scan order (DDA: what it triggered)."""
    _row(scans, scan)
    hits = scans.filter(pc.equal(scans.column("parent_scan"), scan))
    return [int(s) for s in hits.column("scan").to_pylist()]


# ── peaks ─────────────────────────────────────────────────────────────────


def _peaks(bundle_dir: Path | str) -> ds.Dataset:
    return ds.dataset(Path(bundle_dir) / "peaks.parquet")


def spectrum(bundle_dir: Path | str, scan: int) -> pa.Table:
    """The peaks of one scan: ``mz`` and ``intensity``, in m/z order.

    A scan with no peaks gives an empty table (bundles keep such scans in
    the scan table but have no peak rows for them).
    """
    table = _peaks(bundle_dir).to_table(
        columns=["mz", "intensity"], filter=ds.field("scan") == int(scan)
    )
    return table.sort_by("mz")


def spectra(bundle_dir: Path | str, scans: Iterable[int]) -> pa.Table:
    """The peaks of several scans in one read: ``scan``, ``mz``, ``intensity``."""
    wanted = pa.array(sorted({int(s) for s in scans}), pa.int32())
    table = _peaks(bundle_dir).to_table(
        columns=["scan", "mz", "intensity"], filter=ds.field("scan").isin(wanted)
    )
    return table.sort_by([("scan", "ascending"), ("mz", "ascending")])


# ── extracted-ion chromatograms ───────────────────────────────────────

#: One row per (trace, scan); every scan of the level and window in range,
#: so a trace drops to zero where its ion is absent.
XIC_SCHEMA = pa.schema(
    [
        pa.field("trace", pa.int16(), nullable=False),
        pa.field("mz_target", pa.float64(), nullable=False),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),
        #: The peak within tolerance nearest the target; 0 when there is none.
        pa.field("intensity", pa.float64(), nullable=False),
        #: That peak's m/z; null when there is none.
        pa.field("mz_observed", pa.float64()),
    ],
    metadata={b"schema_name": b"MsXic"},
)


def _half_widths(mz: np.ndarray, tolerance: float, unit: str) -> np.ndarray:
    if unit == "ppm":
        return mz * tolerance * 1e-6
    if unit == "Da":
        return np.full_like(mz, tolerance)
    raise ValueError(f"tolerance unit must be 'ppm' or 'Da', not {unit!r}")


def xic(
    bundle_dir: Path | str,
    mz: Sequence[float],
    *,
    level: int,
    tolerance: float,
    tolerance_unit: Literal["ppm", "Da"],
    window: int | None = None,
    rt_range: tuple[float, float] | None = None,
    scans: pa.Table | None = None,
    use_index: bool = True,
) -> pa.Table:
    """Extracted-ion chromatograms of each m/z in ``mz`` (``XIC_SCHEMA``).

    The traces run over every scan of ``level`` (and DIA ``window``) within
    ``rt_range`` (seconds; the whole run when omitted). In each scan a trace
    takes the peak within tolerance nearest its m/z (the more intense of two
    equally near), as ``quant.chromatogram``'s extraction does. ``scans`` is
    the bundle's :func:`scan_table`, when the caller already holds it.

    With a peak index (and ``use_index``), each trace is one m/z slice of
    its partition. Without one, the read is ``peaks.parquet`` restricted to
    the scans in range: well under a second for a few minutes of a 100-min
    run, several seconds for the whole of it (the browser's payload survey).
    """
    bundle = Path(bundle_dir)
    targets = np.asarray(mz, dtype=np.float64)
    half = _half_widths(targets, tolerance, tolerance_unit)
    table = scans if scans is not None else scan_table(bundle)
    sub = _select(table, level, window)
    if rt_range is not None:
        rt_col = sub.column("rt")
        sub = sub.filter(pc.and_(pc.greater_equal(rt_col, rt_range[0]), pc.less_equal(rt_col, rt_range[1])))
    grid_scans = sub.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
    grid_rt = sub.column("rt").to_numpy(zero_copy_only=False).astype(np.float64)
    order = np.argsort(grid_scans, kind="stable")
    grid_scans, grid_rt = grid_scans[order], grid_rt[order]
    n_t, n_s = targets.size, grid_scans.size
    best_int = np.zeros((n_t, n_s))
    best_mz = np.full((n_t, n_s), np.nan)

    if n_t and n_s:
        lo_rt, hi_rt = float(grid_rt.min()), float(grid_rt.max())
        index = find_peak_index(bundle) if use_index else None
        partition = None
        if index is not None:
            # The index numbers its partitions itself; find this window's by
            # its bounds (any scan of the window carries them).
            bounds: tuple[float | None, float | None] = (None, None)
            if window is not None:
                first = sub.slice(0, 1).to_pylist()[0]
                bounds = (first["isolation_lower"], first["isolation_upper"])
            partition = index_partition(index, level=level, isolation_lower=bounds[0], isolation_upper=bounds[1])
        for t in range(n_t):
            lo, hi = targets[t] - half[t], targets[t] + half[t]
            if index is not None and partition is not None:
                peaks = index_slice(
                    index, level=level, window_id=partition, mz_lo=lo, mz_hi=hi, rt_lo=lo_rt, rt_hi=hi_rt
                )
            else:
                peaks = _peaks(bundle).to_table(
                    columns=["scan", "mz", "intensity"],
                    filter=(ds.field("rt") >= lo_rt) & (ds.field("rt") <= hi_rt)
                    & (ds.field("mz") >= lo) & (ds.field("mz") <= hi)
                    & ds.field("scan").isin(pa.array(grid_scans, pa.int32())),
                )
            if peaks.num_rows == 0:
                continue
            p_scan = peaks.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
            p_mz = peaks.column("mz").to_numpy(zero_copy_only=False).astype(np.float64)
            p_int = peaks.column("intensity").to_numpy(zero_copy_only=False).astype(np.float64)
            # The peak nearest the target in each scan (as the XIC extraction
            # in ``quant.chromatogram`` takes it), the more intense on a tie:
            # sort by scan, distance, then intensity descending, and keep the
            # first of each scan.
            o = np.lexsort((-p_int, np.abs(p_mz - targets[t]), p_scan))
            p_scan, p_mz, p_int = p_scan[o], p_mz[o], p_int[o]
            first = np.concatenate([[True], p_scan[1:] != p_scan[:-1]])
            p_scan, p_mz, p_int = p_scan[first], p_mz[first], p_int[first]
            at = np.searchsorted(grid_scans, p_scan)
            ok = (at < n_s) & (grid_scans[np.minimum(at, n_s - 1)] == p_scan)
            best_int[t, at[ok]] = p_int[ok]
            best_mz[t, at[ok]] = p_mz[ok]

    observed = best_mz.reshape(-1)
    return pa.table(
        {
            "trace": pa.array(np.repeat(np.arange(n_t), n_s), pa.int16()),
            "mz_target": pa.array(np.repeat(targets, n_s), pa.float64()),
            "scan": pa.array(np.tile(grid_scans, n_t), pa.int32()),
            "rt": pa.array(np.tile(grid_rt, n_t), pa.float64()),
            "intensity": pa.array(best_int.reshape(-1), pa.float64()),
            "mz_observed": pa.array(observed, pa.float64(), mask=np.isnan(observed)),
        },
        schema=XIC_SCHEMA,
    )


def locate_precursor(
    scans: pa.Table,
    *,
    precursor_mz: float,
    rt: float,
    level: int = 2,
) -> dict[str, object]:
    """Where a precursor at ``precursor_mz`` eluting at ``rt`` was measured.

    Returns ``window`` (the DIA window of ``level`` isolating it, the one
    centred nearest when several do; ``None`` for DDA), ``isolation``
    (that window's bounds, or those of the DDA scan found), ``ms_scan``
    (the scan of ``level`` nearest ``rt`` that isolated the precursor) and
    ``parent_scan`` (the scan of the level below nearest ``rt``). A
    peptide-centric search reports the apex at a scan of the covering
    window, so ``ms_scan`` is that apex scan exactly.
    """
    windows = isolation_window_table(scans)
    windows = windows.filter(pc.equal(windows.column("level"), level))
    found: dict[str, object] = {"window": None, "isolation": None, "ms_scan": None}
    if windows.num_rows:
        rows = [w for w in windows.to_pylist() if w["isolation_lower"] <= precursor_mz <= w["isolation_upper"]]
        if rows:
            w = min(rows, key=lambda r: abs(r["center"] - precursor_mz))
            found["window"] = w["window"]
            found["isolation"] = (w["isolation_lower"], w["isolation_upper"])
            found["ms_scan"] = nearest_scan(scans, rt, level=level, window=w["window"])
    else:
        sub = _select(scans, level, None)
        lo = sub.column("isolation_lower").to_numpy(zero_copy_only=False)
        hi = sub.column("isolation_upper").to_numpy(zero_copy_only=False)
        covers = (lo <= precursor_mz) & (hi >= precursor_mz)
        if covers.any():
            idx = np.flatnonzero(covers)
            times = sub.column("rt").to_numpy(zero_copy_only=False)[idx]
            best = int(idx[int(np.argmin(np.abs(times - rt)))])
            found["ms_scan"] = int(sub.column("scan")[best].as_py())
            found["isolation"] = (float(lo[best]), float(hi[best]))
    found["parent_scan"] = nearest_scan(scans, rt, level=level - 1) if level > 1 else None
    return found


__all__ = [
    "BUNDLE_KINDS",
    "DEFAULT_TOLERANCES",
    "FALLBACK_TOLERANCE",
    "NAV_COLUMNS",
    "AcquisitionInfo",
    "chromatogram",
    "default_tolerance",
    "dependent_scans",
    "XIC_SCHEMA",
    "isolation_window_table",
    "locate_precursor",
    "nearest_scan",
    "open_acquisition",
    "parent_scan",
    "scan_table",
    "spectra",
    "spectrum",
    "xic",
]
