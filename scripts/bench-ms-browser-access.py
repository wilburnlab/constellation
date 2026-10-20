"""Measure what a mass-spec browser would read, on real converted runs.

Usage:
    python scripts/bench-ms-browser-access.py --bundle BUNDLE [--bundle ...] \
        [--experiment EXP --experiment-bundle BUNDLE] [--library LIB.dlib] \
        [--samples 30] [--seed 0] --out RESULTS.json

Reads only what it is given and writes only ``--out``. Answers the
questions the browser plan (``docs/plans/ms-browser.md``, PR A) leaves to
measurement:

- **payload sizes:** scans and peaks per scan by MS level, isolation windows
  and scans per window (the length of a chromatogram), the size of the scan
  table a client would hold for navigation;
- **bundle layout:** rows and row groups of ``peaks.parquet``, the RT span of
  a row group, whether MS2 scans name a parent scan, whether base-peak
  values exist;
- **latency, cold and warm:** one spectrum by scan number, an MS1 XIC read
  from ``peaks.parquet`` with and without an RT window, and the same m/z
  slice from the peak index when ``<bundle>/mz_index`` exists. "Cold" evicts
  the bundle's files from the page cache first (``posix_fadvise``), which
  works on a local filesystem and is a no-op on a Windows drive;
- **identifications (with ``--experiment``):** rows per table, and how far
  each identification's ``rt_observed`` lies from the nearest MS1 scan and
  the nearest MS2 scan of the window covering its precursor (nothing stores
  an apex scan, so the browser derives it); which input each search read;
- **library lookups (with ``--library``):** whether the EncyclopeDIA
  library has its per-precursor index on ``entries``, and the latency of
  fetching one precursor's spectrum by modified sequence and charge.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

NAV_COLUMNS = [
    "scan",
    "level",
    "rt",
    "tic",
    "base_peak_intensity",
    "isolation_lower",
    "isolation_upper",
    "precursor_mz",
    "precursor_scan",
]


# ── helpers ───────────────────────────────────────────────────────────────


def evict(paths: list[Path]) -> None:
    """Drop the files' pages from the page cache (no-op where unsupported)."""
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError):
            pass
        finally:
            os.close(fd)


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn()
    return (time.perf_counter() - start) * 1000.0, out


def summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "max": round(float(arr.max()), 2),
    }


def cold_warm(
    fn: Callable[[], Any], files: list[Path], repeats: int
) -> dict[str, Any]:
    cold, warm = [], []
    for _ in range(repeats):
        evict(files)
        cold.append(timed(fn)[0])
        warm.append(timed(fn)[0])
    return {"cold_ms": summary(cold), "warm_ms": summary(warm)}


def bundle_files(bundle: Path) -> list[Path]:
    return [p for p in bundle.rglob("*.parquet")]


def row_groups_touched(path: Path, column: str, lo: float, hi: float) -> int:
    """Row groups whose statistics for ``column`` overlap ``[lo, hi]``."""
    meta = pq.ParquetFile(path).metadata
    idx = meta.schema.to_arrow_schema().get_field_index(column)
    touched = 0
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(idx).statistics
        if stats is None or not stats.has_min_max:
            touched += 1
        elif stats.max >= lo and stats.min <= hi:
            touched += 1
    return touched


def windows_of(scans: pa.Table, level: int) -> np.ndarray:
    sub = scans.filter(pc.equal(scans.column("level"), level))
    lo = sub.column("isolation_lower").to_numpy(zero_copy_only=False)
    hi = sub.column("isolation_upper").to_numpy(zero_copy_only=False)
    keep = ~(np.isnan(lo) | np.isnan(hi))
    pairs = np.stack([lo[keep], hi[keep]], axis=1).astype(np.float32)
    return np.unique(pairs, axis=0) if pairs.size else pairs.reshape(0, 2)


# ── per-bundle measurements ───────────────────────────────────────────────


def survey_bundle(bundle: Path, *, samples: int, rng: np.random.Generator) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    peaks_path = bundle / "peaks.parquet"
    meta_path = bundle / "scan_metadata.parquet"
    files = bundle_files(bundle)
    out: dict[str, Any] = {
        "bundle": str(bundle),
        "kind": manifest.get("kind"),
        "profile_mode": (manifest.get("parameters") or {}).get("profile_mode"),
        "rt_bin_width_s": (manifest.get("parameters") or {}).get("rt_bin_width_s"),
        "bytes": {p.name: p.stat().st_size for p in files if p.parent == bundle},
    }

    # Layout of peaks.parquet.
    pmeta = pq.ParquetFile(peaks_path).metadata
    rt_idx = pmeta.schema.to_arrow_schema().get_field_index("rt")
    spans = []
    for i in range(pmeta.num_row_groups):
        st = pmeta.row_group(i).column(rt_idx).statistics
        if st is not None and st.has_min_max:
            spans.append(st.max - st.min)
    out["peaks_parquet"] = {
        "rows": pmeta.num_rows,
        "row_groups": pmeta.num_row_groups,
        "rows_per_row_group_median": int(
            statistics.median(
                pmeta.row_group(i).num_rows for i in range(pmeta.num_row_groups)
            )
        ),
        "row_group_rt_span_s_median": round(statistics.median(spans), 2) if spans else None,
    }

    # Scan table: what a client would hold to navigate.
    ms_load = timed(lambda: pq.read_table(meta_path, columns=NAV_COLUMNS))
    scans: pa.Table = ms_load[1]
    out["scan_table"] = {
        "rows": scans.num_rows,
        "nav_columns_bytes": scans.nbytes,
        "load_ms_first": round(ms_load[0], 1),
    }
    out["scan_table"]["load"] = cold_warm(
        lambda: pq.read_table(meta_path, columns=NAV_COLUMNS), [meta_path], 3
    )

    # Per-level payload sizes.
    full_meta = pq.read_table(
        meta_path,
        columns=["scan", "level", "peak_count", "analyzer", "base_peak_intensity", "master_scan"],
    )
    levels = sorted(set(full_meta.column("level").to_pylist()))
    per_level = {}
    for level in levels:
        sub = full_meta.filter(pc.equal(full_meta.column("level"), level))
        counts = sub.column("peak_count").to_numpy(zero_copy_only=False)
        analyzers = pc.value_counts(sub.column("analyzer")).to_pylist()
        entry: dict[str, Any] = {
            "scans": sub.num_rows,
            "peaks_per_scan_mean": round(float(np.nanmean(counts)), 1) if counts.size else None,
            "peaks_per_scan_max": int(np.nanmax(counts)) if counts.size else None,
            "analyzers": {str(a["values"]): a["counts"] for a in analyzers},
            "base_peak_null_frac": round(
                sub.column("base_peak_intensity").null_count / max(sub.num_rows, 1), 4
            ),
        }
        if level > 1:
            nav = scans.filter(pc.equal(scans.column("level"), level))
            entry["precursor_scan_present_frac"] = round(
                1 - nav.column("precursor_scan").null_count / max(nav.num_rows, 1), 4
            )
            entry["master_scan_present_frac"] = round(
                1 - sub.column("master_scan").null_count / max(sub.num_rows, 1), 4
            )
            wins = windows_of(scans, level)
            entry["distinct_windows"] = int(wins.shape[0])
            if 0 < wins.shape[0] <= 1024:
                lo = nav.column("isolation_lower").to_numpy(zero_copy_only=False).astype(np.float32)
                per_window = [int(np.sum(lo == w[0])) for w in wins]
                entry["scans_per_window_median"] = int(statistics.median(per_window))
                entry["scans_per_window_max"] = int(max(per_window))
        per_level[str(level)] = entry
    out["levels"] = per_level

    ms1 = scans.filter(pc.equal(scans.column("level"), 1))
    ms1_rt = np.sort(ms1.column("rt").to_numpy(zero_copy_only=False))
    out["ms1_cycle_s_median"] = round(float(np.median(np.diff(ms1_rt))), 3) if ms1_rt.size > 1 else None

    # Spectrum by scan number.
    dataset = ds.dataset(peaks_path)
    spectrum_ms: dict[str, Any] = {}
    for level in levels:
        pool = scans.filter(pc.equal(scans.column("level"), level)).column("scan").to_numpy()
        picks = rng.choice(pool, size=min(samples, pool.size), replace=False)
        cold, warm, sizes = [], [], []
        for scan in picks:
            fn = lambda s=int(scan): dataset.to_table(  # noqa: E731
                columns=["mz", "intensity"], filter=ds.field("scan") == s
            )
            evict([peaks_path])
            t, tbl = timed(fn)
            cold.append(t)
            sizes.append(tbl.num_rows)
            warm.append(timed(fn)[0])
        spectrum_ms[str(level)] = {
            "cold_ms": summary(cold),
            "warm_ms": summary(warm),
            "peaks": summary([float(s) for s in sizes]),
        }
    out["spectrum_by_scan"] = spectrum_ms

    # MS1 XIC from peaks.parquet: ±2 min window and full run.
    ms1_scans = ms1.column("scan").to_numpy()
    xic_pm2, xic_full, rg_pm2 = [], [], []
    for scan in rng.choice(ms1_scans, size=min(samples, ms1_scans.size), replace=False):
        spec = dataset.to_table(
            columns=["mz", "intensity", "rt"], filter=ds.field("scan") == int(scan)
        )
        if spec.num_rows == 0:
            continue
        top = int(np.argmax(spec.column("intensity").to_numpy()))
        mz = float(spec.column("mz")[top].as_py())
        rt = float(spec.column("rt")[top].as_py())
        lo, hi = mz - 0.35, mz + 0.35
        window = (
            (ds.field("level") == 1)
            & (ds.field("rt") >= rt - 120)
            & (ds.field("rt") <= rt + 120)
            & (ds.field("mz") >= lo)
            & (ds.field("mz") <= hi)
        )
        evict([peaks_path])
        xic_pm2.append(timed(lambda f=window: dataset.to_table(columns=["scan", "intensity"], filter=f))[0])
        rg_pm2.append(row_groups_touched(peaks_path, "rt", rt - 120, rt + 120))
        if len(xic_full) < 3:
            full = (ds.field("level") == 1) & (ds.field("mz") >= lo) & (ds.field("mz") <= hi)
            evict([peaks_path])
            xic_full.append(timed(lambda f=full: dataset.to_table(columns=["scan", "intensity"], filter=f))[0])
    out["ms1_xic_from_peaks"] = {
        "rt_pm_2min_cold_ms": summary(xic_pm2),
        "rt_pm_2min_row_groups": summary([float(r) for r in rg_pm2]),
        "full_run_cold_ms": summary(xic_full),
    }

    # The same slice from the peak index, if one exists.
    index_root = bundle / "mz_index"
    if (index_root / "manifest.json").exists():
        out["index"] = survey_index(index_root, scans, samples=samples, rng=rng)
    return out


def survey_index(index_root: Path, scans: pa.Table, *, samples: int, rng: np.random.Generator) -> dict:
    manifest = json.loads((index_root / "manifest.json").read_text())
    parts = manifest["partitions"]
    files = sorted(index_root.rglob("part-*.parquet"))
    out: dict[str, Any] = {
        "partitions": len(parts),
        "bytes_total": sum(p.stat().st_size for p in files),
        "window_scheme": manifest.get("window_scheme"),
    }
    ms1_part = next((p for p in parts if p["level"] == 1), None)
    targets = [("ms1", ms1_part)] if ms1_part else []
    ms2_parts = [p for p in parts if p["level"] == 2]
    if ms2_parts:
        targets.append(("ms2_window", ms2_parts[len(ms2_parts) // 2]))
    for name, part in targets:
        path = index_root / f"ms_level={part['level']}" / f"isolation_window={part['isolation_window_id']}" / "part-00000.parquet"
        meta = pq.ParquetFile(path).metadata
        mzs = pq.read_table(path, columns=["mz"]).column("mz").to_numpy()
        picks = rng.choice(mzs, size=min(samples, mzs.size), replace=False)
        cold, warm, groups, rows = [], [], [], []
        part_ds = ds.dataset(path)
        for mz in picks:
            lo, hi = float(mz) - 0.35, float(mz) + 0.35
            flt = (ds.field("mz") >= lo) & (ds.field("mz") <= hi)
            fn = lambda f=flt: part_ds.to_table(columns=["rt", "scan", "intensity"], filter=f)  # noqa: E731
            evict([path])
            t, tbl = timed(fn)
            cold.append(t)
            rows.append(float(tbl.num_rows))
            warm.append(timed(fn)[0])
            groups.append(float(row_groups_touched(path, "mz", lo, hi)))
        out[name] = {
            "rows": meta.num_rows,
            "row_groups": meta.num_row_groups,
            "slice_cold_ms": summary(cold),
            "slice_warm_ms": summary(warm),
            "slice_rows": summary(rows),
            "slice_row_groups": summary(groups),
        }
    return out


# ── experiment ────────────────────────────────────────────────────────────


def survey_experiment(exp: Path, bundle: Path | None) -> dict:
    from constellation.massspec.manifest import open_table

    out: dict[str, Any] = {"experiment": str(exp), "tables": {}}
    for table_dir in sorted((exp / "tables").glob("*")):
        if table_dir.is_dir():
            out["tables"][table_dir.name] = open_table(exp, table_dir.name).count_rows()

    searches = []
    for unit_manifest in sorted((exp / "units").glob("*/manifest.json")):
        data = json.loads(unit_manifest.read_text())
        inputs = data.get("inputs") or {}
        source = inputs.get("source") or {}
        searches.append(
            {
                "unit": unit_manifest.parent.name,
                "status": data.get("status"),
                "source_kind": source.get("kind"),
                "source_path": source.get("path"),
                "library": (inputs.get("library") or {}).get("path"),
                "acquisition_ids": data.get("acquisition_ids"),
            }
        )
    out["searches"] = searches

    # Size of the table the identifications drawer would hold.
    quant = open_table(exp, "precursor_quant").to_table()
    precursor = open_table(exp, "precursor").to_table(
        columns=["precursor_id", "peptide_id", "charge", "precursor_mz"]
    )
    precursor = precursor.group_by("precursor_id").aggregate(
        [("peptide_id", "min"), ("charge", "min"), ("precursor_mz", "min")]
    ).rename_columns(["precursor_id", "peptide_id", "charge", "precursor_mz"])
    joined = quant.join(precursor, "precursor_id")
    out["identifications_rows"] = joined.num_rows
    out["identifications_bytes"] = joined.nbytes

    if bundle is None:
        return out

    # rt_observed → nearest scan, per level, against the covering window.
    scans = pq.read_table(bundle / "scan_metadata.parquet", columns=NAV_COLUMNS)
    level = scans.column("level").to_numpy()
    rt = scans.column("rt").to_numpy(zero_copy_only=False)
    ms1_rt = np.sort(rt[level == 1])
    iso_lo = scans.column("isolation_lower").to_numpy(zero_copy_only=False)
    iso_hi = scans.column("isolation_upper").to_numpy(zero_copy_only=False)
    wins = windows_of(scans, 2)
    ms2_rt_by_window = [
        np.sort(rt[(level == 2) & (iso_lo.astype(np.float32) == w[0]) & (iso_hi.astype(np.float32) == w[1])])
        for w in wins
    ]

    def nearest(sorted_rt: np.ndarray, value: float) -> float:
        i = int(np.searchsorted(sorted_rt, value))
        best = np.inf
        for j in (i - 1, i):
            if 0 <= j < sorted_rt.size:
                best = min(best, abs(float(sorted_rt[j]) - value))
        return best

    rt_obs = joined.column("rt_observed").to_numpy(zero_copy_only=False)
    pmz = joined.column("precursor_mz").to_numpy(zero_copy_only=False)
    d1, d2, uncovered, multi = [], [], 0, 0
    for r, m in zip(rt_obs, pmz):
        if not np.isfinite(r) or r < 0:
            continue
        d1.append(nearest(ms1_rt, float(r)))
        covering = [k for k, w in enumerate(wins) if w[0] <= m <= w[1]]
        if not covering:
            uncovered += 1
            continue
        multi += len(covering) > 1
        d2.append(min(nearest(ms2_rt_by_window[k], float(r)) for k in covering))
    out["apex_scan_distance_s"] = {
        "ms1": summary(d1),
        "ms2_covering_window": summary(d2),
        "precursors_in_no_window": uncovered,
        "precursors_in_several_windows": int(multi),
    }
    return out


# ── library ───────────────────────────────────────────────────────────────


def survey_library(path: Path, *, samples: int, rng: np.random.Generator) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        indexes = [row[1] for row in con.execute("PRAGMA index_list('entries')")]
        (max_rowid,) = con.execute("SELECT max(rowid) FROM entries").fetchone()
        keys = []
        for rowid in rng.integers(1, max_rowid + 1, size=samples):
            row = con.execute(
                "SELECT PeptideModSeq, PrecursorCharge FROM entries WHERE rowid = ?",
                (int(rowid),),
            ).fetchone()
            if row:
                keys.append(row)
        lookup = []
        for modseq, charge in keys:
            lookup.append(
                timed(
                    lambda m=modseq, z=charge: con.execute(
                        "SELECT MassArray, IntensityArray FROM entries "
                        "WHERE PeptideModSeq = ? AND PrecursorCharge = ?",
                        (m, z),
                    ).fetchall()
                )[0]
            )
        plan = con.execute(
            "EXPLAIN QUERY PLAN SELECT MassArray FROM entries "
            "WHERE PeptideModSeq = ? AND PrecursorCharge = ?",
            ("X", 2),
        ).fetchall()
    finally:
        con.close()
    return {
        "library": str(path),
        "bytes": path.stat().st_size,
        "entries_indexes": indexes,
        "query_plan": [row[-1] for row in plan],
        "lookup_ms": summary(lookup),
    }


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, action="append", default=[])
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--experiment-bundle", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results: dict[str, Any] = {"bundles": [], "settings": {"samples": args.samples, "seed": args.seed}}
    for bundle in args.bundle:
        print(f"bundle {bundle}", flush=True)
        results["bundles"].append(survey_bundle(bundle, samples=args.samples, rng=rng))
    if args.experiment:
        print(f"experiment {args.experiment}", flush=True)
        results["experiment"] = survey_experiment(args.experiment, args.experiment_bundle)
    if args.library:
        print(f"library {args.library}", flush=True)
        results["library"] = survey_library(args.library, samples=args.samples, rng=rng)
    args.out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
