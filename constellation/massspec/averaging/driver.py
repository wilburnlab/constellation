"""Average a direct-infusion profile run into one spectrum per kind of scan.

With an analyte infused at constant concentration, every repeat of a scan is a
replicate of it, and what a run says is best read from their mean. Scans are
grouped by what was acquired, from the typed scan metadata: MS level, polarity,
activation, collision energy, precursor, isolation window and scan range. A run
that repeats one scan gives one spectrum; one that cycles through several gives
one for each.

The ion trap stores a profile scan's points on a fixed m/z grid whose step
depends on the scan rate (1/30 m/z and 1/15 m/z on a Stellar). Each group's grid
is read from its own points, the step from a scan's whole span (a median of
neighbouring differences drifts over tens of thousands of points), and every
point is placed on it by index: nothing is interpolated or re-binned. A point
that does not land on the grid is counted and reported, never merged into a
neighbour.

Every scan is averaged by default. ``skip_first`` can leave out the first scans
of each group, the first fills of that kind of scan before the AGC has any
history: on one Stellar infusion series every scan of the first cycle ran at
the injection-time ceiling, but on another the first scans filled exactly like
the rest, and with ~100 replicates one scan barely moves the mean. The summary
reports the injection time of any skipped scans beside the rest, so the choice
can be checked.

Intensities are the instrument's, which are normalised by injection time, so the
mean of scans with different fills is a mean of rates, not of ion counts. Spray
drifts over a run; a mean preserves the shape of each spectrum and, in a run
that cycles through its scans, the ratios between them, since every kind of scan
is sampled across the same stretch of time. Absolute intensity is a time
average and is reported as one.

What it writes is a profile bundle like any other (``profile/``, scan and
acquisition metadata, a manifest written last naming its source), one scan per
averaged spectrum, so the commands that read a profile bundle read it unchanged:
``massspec centroid`` makes a peak list from it. Beside that it keeps what has
no column in the peak table: each point's standard deviation and number of
replicates (``average_points/``) and each spectrum's replicate count,
injection-time and TIC summaries (``average_summary.parquet``).

Only ion-trap profile points are averaged: an Orbitrap's profile is not stored
on a grid, and a scan acquired as centroids has no profile to average.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.acquisition_structure import scan_kinds
from constellation.massspec.bundle import (
    BUNDLE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    PROFILE_BUNDLE_KIND,
    PROFILE_DIR,
    PeakWriter,
    is_profile_bundle,
    iter_scan_blocks,
    part_name,
    peak_rows,
    read_bundle_manifest,
)

SCAN_METADATA_FILENAME = "scan_metadata.parquet"
ACQUISITION_METADATA_FILENAME = "acquisition_metadata.parquet"
#: Per point: standard deviation and number of replicates.
POINTS_DIR = "average_points"
#: Per averaged spectrum: replicates, injection time, TIC, time span.
SUMMARY_FILENAME = "average_summary.parquet"
#: What an averaged bundle is named for, after its run.
SUFFIX = "average"
#: Version of the averaging, recorded in the manifest.
VERSION = 1
#: The analyzer whose profile is stored on a grid.
ION_TRAP = "ITMS"
#: How far from a grid index, as a fraction of the step, a point may sit.
OFF_GRID_TOLERANCE = 0.05

POINTS_SCHEMA = pa.schema(
    [
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("intensity_sd", pa.float64()),
        pa.field("n_scans", pa.int32(), nullable=False),
    ]
)


def group_scans(scans: pa.Table) -> tuple[np.ndarray, int]:
    """Kind of scan per scan row (``acquisition_structure.scan_kinds``)."""
    return scan_kinds(scans)


def _ordinal_in_group(group: np.ndarray, n_groups: int) -> np.ndarray:
    """0 for a group's first scan (in stored order), 1 for its second, ..."""
    order = np.argsort(group, kind="stable")
    counts = np.bincount(group, minlength=n_groups)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ordinal = np.empty(group.size, np.int64)
    ordinal[order] = np.arange(group.size) - np.repeat(starts, counts)
    return ordinal


def _refuse(scans: pa.Table) -> None:
    kinds = set(scans.column("spectrum_type").to_pylist())
    if kinds - {"profile"}:
        raise ValueError(
            f"this bundle holds {', '.join(sorted(k or 'unknown' for k in kinds - {'profile'}))} "
            f"scans; averaging places profile points on the ion trap's m/z grid"
        )
    analyzers = set(scans.column("analyzer").to_pylist())
    if analyzers - {ION_TRAP}:
        raise ValueError(
            f"this run has profile scans from {', '.join(sorted(a or 'unknown' for a in analyzers - {ION_TRAP}))}; "
            f"only ion-trap profile points sit on a fixed m/z grid"
        )


def _drift(tic: np.ndarray) -> float:
    """Mean TIC of the last five scans over the first five; NaN for fewer than ten."""
    if tic.size < 10 or not tic[:5].mean() > 0:
        return float("nan")
    return float(tic[-5:].mean() / tic[:5].mean())


def _grids(bundle: Path, row_of: np.ndarray, group: np.ndarray, n_groups: int):
    """Per group: origin, step and number of grid points, from its own points."""
    lo = np.full(n_groups, np.inf)
    hi = np.full(n_groups, -np.inf)
    step = np.full(n_groups, np.nan)
    for block in iter_scan_blocks(bundle, columns=("mz",), points=True):
        mz = block.columns["mz"]
        offsets = block.offsets
        g = group[row_of[block.scan]]
        sizes = np.diff(offsets)
        has = sizes > 0
        np.minimum.at(lo, g[has], mz[offsets[:-1][has]])
        np.maximum.at(hi, g[has], mz[offsets[1:][has] - 1])
        # The step, from the first scan of two or more points of each group
        # that has none yet: one scan per group, not one per scan.
        candidates = np.flatnonzero((sizes >= 2) & np.isnan(step[g]))
        _, first = np.unique(g[candidates], return_index=True)
        for i in candidates[first]:
            m = mz[offsets[i]:offsets[i + 1]]
            n_steps = int(np.rint((m[-1] - m[0]) / np.median(np.diff(m))))
            if n_steps > 0:
                step[g[i]] = (m[-1] - m[0]) / n_steps
    seen = np.isfinite(lo)
    n_points = np.zeros(n_groups, np.int64)
    ok = seen & np.isfinite(step)
    n_points[ok] = np.rint((hi[ok] - lo[ok]) / step[ok]).astype(np.int64) + 1
    return np.where(seen, lo, np.nan), step, n_points


def _run_name(source: Path, manifest: dict) -> str:
    return Path(str(manifest.get("source_file") or source.name).replace("\\", "/").rsplit("/", 1)[-1]).stem


def averaged_path(profile_bundle: Path | str, parent: Path | str | None = None) -> Path:
    """Where :func:`average_bundle` writes by default: ``<run>.average`` in
    ``parent``, or beside the profile bundle when ``parent`` is None."""
    source = Path(profile_bundle)
    run = _run_name(source, read_bundle_manifest(source))
    return (Path(parent) if parent is not None else source.parent) / f"{run}.{SUFFIX}"


def find_profile_bundles(directory: Path | str) -> list[Path]:
    """The run profile bundles directly inside ``directory`` (not recursive),
    sorted by name. Peak bundles and bundles made from another bundle (an
    earlier ``.average``) are not runs to average and are left out."""
    found = []
    for child in sorted(Path(directory).iterdir(), key=lambda p: p.name):
        if not (child / "manifest.json").is_file():
            continue
        try:
            if is_profile_bundle(child) and not read_bundle_manifest(child).get("derived_from"):
                found.append(child)
        except Exception:  # an unreadable manifest is not a run we can average
            continue
    return found


@dataclass(frozen=True)
class AverageResult:
    """One run of :func:`average_directory`: ``status`` is ``ok``,
    ``skipped`` (already averaged; ``force`` remakes it) or ``error``."""

    bundle: Path
    status: str
    output: Path | None
    detail: str = ""


def average_directory(
    directory: Path | str,
    *,
    output_parent: Path | str | None = None,
    skip_first: int = 0,
    force: bool = False,
) -> list[AverageResult]:
    """Average every run profile bundle directly inside ``directory`` (see
    :func:`find_profile_bundles`), each to ``<run>.average`` in
    ``output_parent`` (default: beside its source). A run that is already
    averaged is skipped unless ``force``; a run that fails is reported and the
    rest carry on."""
    if skip_first < 0:
        raise ValueError(f"skip_first must be 0 or more, got {skip_first}")
    results = []
    for bundle in find_profile_bundles(directory):
        try:
            out = averaged_path(bundle, output_parent)
            if out.exists() and any(out.iterdir()) and not force:
                results.append(AverageResult(bundle, "skipped", out, "already averaged"))
                continue
            made = average_bundle(bundle, output_dir=out, skip_first=skip_first, force=force)
            results.append(AverageResult(bundle, "ok", made))
        except Exception as exc:  # one bad run must not stop the rest
            results.append(AverageResult(bundle, "error", None, f"{type(exc).__name__}: {exc}"))
    return results


def average_bundle(
    profile_bundle: Path | str,
    *,
    output_dir: Path | str | None = None,
    skip_first: int = 0,
    force: bool = False,
) -> Path:
    """Average ``profile_bundle`` into one spectrum per kind of scan; returns
    the new bundle's path, ``<run>.average`` beside its source unless
    ``output_dir`` says otherwise.

    ``skip_first`` leaves out the first that many scans of each group (the
    AGC's first fills); the default, 0, keeps every scan.
    """
    source = Path(profile_bundle)
    manifest = read_bundle_manifest(source)
    if not is_profile_bundle(source):
        raise ValueError(
            f"{source} holds peaks, not profile points; average a run's profile "
            f"bundle (<run>.profile, from `massspec convert`)"
        )
    if skip_first < 0:
        raise ValueError(f"skip_first must be 0 or more, got {skip_first}")
    out = Path(output_dir) if output_dir is not None else averaged_path(source)
    if out.resolve() == source.resolve():
        raise ValueError("the averaged bundle cannot replace its source")
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"averaged bundle already there: {out} (pass --force to make it again)")
        shutil.rmtree(out)

    scans = pq.read_table(source / SCAN_METADATA_FILENAME)
    _refuse(scans)
    group, n_groups = group_scans(scans)
    if n_groups == 0:
        raise ValueError(f"{source} has no scans")
    numbers = scans.column("scan").to_numpy().astype(np.int64)
    row_of = np.full(int(numbers.max()) + 1, -1, np.int64)
    row_of[numbers] = np.arange(numbers.size)
    excluded = _ordinal_in_group(group, n_groups) < skip_first
    kept_per_group = np.bincount(group[~excluded], minlength=n_groups)
    if (kept_per_group == 0).any():
        raise ValueError(
            f"skip_first={skip_first} leaves {int((kept_per_group == 0).sum())} kind(s) of scan "
            f"with nothing to average; pass a smaller skip_first"
        )

    origin, step, n_points = _grids(source, row_of, group, n_groups)
    if np.isnan(step).any():
        raise ValueError(f"{int(np.isnan(step).sum())} kind(s) of scan have no profile points to average")

    # ── accumulate ────────────────────────────────────────────────────
    offset = np.concatenate(([0], np.cumsum(n_points)[:-1]))
    total = int(n_points.sum())
    acc = np.zeros(total)
    acc2 = np.zeros(total)
    count = np.zeros(total, np.int64)
    n_read = 0
    n_off = 0
    for block in iter_scan_blocks(source, columns=("mz", "intensity"), points=True):
        rows = row_of[block.scan]
        keep_scan = ~excluded[rows]
        if not keep_scan.any():
            continue
        sizes = np.diff(block.offsets)
        keep = np.repeat(keep_scan, sizes)
        g = np.repeat(group[rows], sizes)[keep]
        mz = block.columns["mz"][keep]
        intensity = block.columns["intensity"][keep].astype(np.float64)
        x = (mz - origin[g]) / step[g]
        k = np.rint(x).astype(np.int64)
        on = (np.abs(x - k) <= OFF_GRID_TOLERANCE) & (k >= 0) & (k < n_points[g])
        n_read += int(on.size)
        n_off += int((~on).sum())
        flat = offset[g[on]] + k[on]
        acc += np.bincount(flat, intensity[on], total)
        acc2 += np.bincount(flat, intensity[on] ** 2, total)
        count += np.bincount(flat, minlength=total)

    present = count > 0
    mean = np.where(present, acc / np.maximum(count, 1), 0.0)
    var = np.where(count > 1, (acc2 - count * mean**2) / np.maximum(count - 1, 1), np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    owner = np.repeat(np.arange(n_groups), n_points)
    mz_grid = origin[owner] + (np.arange(total) - offset[owner]) * step[owner]
    incomplete = np.bincount(owner, (count != kept_per_group[owner]) & present, n_groups).astype(np.int64)

    # ── one row of scan metadata per averaged spectrum ────────────────
    kept_rows = np.flatnonzero(~excluded)
    first_kept = np.full(n_groups, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first_kept, group[kept_rows], kept_rows)
    rep = scans.take(pa.array(first_kept))
    g_kept = group[kept_rows]

    def per_group_mean(name: str) -> np.ndarray:
        v = scans.column(name).to_numpy(zero_copy_only=False).astype(np.float64)[kept_rows]
        good = ~np.isnan(v)
        n = np.bincount(g_kept[good], minlength=n_groups)
        return np.where(n > 0, np.bincount(g_kept[good], v[good], n_groups) / np.maximum(n, 1), np.nan)

    def per_group_quantile(name: str, q: float) -> np.ndarray:
        v = scans.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
        return np.array([np.nanquantile(v[kept_rows][g_kept == i], q) if np.isfinite(v[kept_rows][g_kept == i]).any() else np.nan for i in range(n_groups)])

    rt_mean = per_group_mean("rt")
    tic_mean = per_group_mean("tic")
    iit_median = per_group_quantile("iit", 0.5)
    # Base peak of the mean spectrum.
    bp_mz = np.full(n_groups, np.nan)
    bp_int = np.zeros(n_groups)
    for i in range(n_groups):  # one argmax per averaged spectrum
        a, b = offset[i], offset[i] + n_points[i]
        sel = present[a:b]
        if sel.any():
            j = int(np.argmax(np.where(sel, mean[a:b], -np.inf)))
            bp_mz[i], bp_int[i] = mz_grid[a + j], mean[a + j]
    n_present = np.bincount(owner, present, n_groups).astype(np.int64)

    replace = {
        "scan": pa.array(np.arange(1, n_groups + 1, dtype=np.int32)),
        "rt": pa.array(rt_mean),
        "tic": pa.array(tic_mean),
        "iit": pa.array(iit_median, pa.float32()),
        "base_peak_mz": pa.array(bp_mz),
        "base_peak_intensity": pa.array(bp_int),
        "peak_count": pa.array(n_present, pa.uint32()),
        "intensity_floor": pa.nulls(n_groups, pa.float64()),
    }
    if "trailer_extras" in rep.column_names:  # a verbatim trailer is one scan's; an average has none
        replace["trailer_extras"] = pa.nulls(n_groups, rep.schema.field("trailer_extras").type)
    for name, values in replace.items():
        if name not in rep.column_names:
            continue
        at = rep.schema.get_field_index(name)
        rep = rep.set_column(at, rep.schema.field(name), values.cast(rep.schema.field(name).type))

    # ── per-spectrum summary ──────────────────────────────────────────
    iit_all = scans.column("iit").to_numpy(zero_copy_only=False).astype(np.float64)
    tic_all = scans.column("tic").to_numpy(zero_copy_only=False).astype(np.float64)
    rt_all = scans.column("rt").to_numpy(zero_copy_only=False).astype(np.float64)
    iit_ceiling = np.nanmax(iit_all) if np.isfinite(iit_all).any() else np.nan

    def kept_of(i: int, v: np.ndarray) -> np.ndarray:
        return v[kept_rows][g_kept == i]

    def skipped_of(i: int, v: np.ndarray) -> np.ndarray:
        return v[excluded & (group == i)]

    def stat(fn, arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        return float(fn(arr)) if arr.size else float("nan")

    summary = pa.table(
        {
            "scan": pa.array(np.arange(1, n_groups + 1, dtype=np.int32)),
            "level": rep.column("level"),
            "filter_string": rep.column("filter_string"),
            "n_scans": pa.array(kept_per_group.astype(np.int32)),
            "n_skipped": pa.array(np.bincount(group[excluded], minlength=n_groups).astype(np.int32)),
            "first_scan": pa.array([int(numbers[group == i].min()) for i in range(n_groups)], pa.int32()),
            "last_scan": pa.array([int(numbers[group == i].max()) for i in range(n_groups)], pa.int32()),
            "rt_first_s": pa.array([stat(np.min, kept_of(i, rt_all)) for i in range(n_groups)]),
            "rt_last_s": pa.array([stat(np.max, kept_of(i, rt_all)) for i in range(n_groups)]),
            "iit_median_ms": pa.array(iit_median),
            "iit_p95_ms": pa.array(per_group_quantile("iit", 0.95)),
            "iit_skipped_median_ms": pa.array([stat(np.median, skipped_of(i, iit_all)) for i in range(n_groups)]),
            "frac_iit_at_ceiling": pa.array(
                [stat(np.mean, (kept_of(i, iit_all) >= iit_ceiling - 1e-3).astype(float)) for i in range(n_groups)]
            ),
            "tic_mean": pa.array(tic_mean),
            "tic_cv": pa.array(
                [stat(np.std, kept_of(i, tic_all)) / tic_mean[i] if tic_mean[i] > 0 else np.nan for i in range(n_groups)]
            ),
            "tic_last_over_first": pa.array([_drift(kept_of(i, tic_all)) for i in range(n_groups)]),
            "grid_origin_mz": pa.array(origin),
            "grid_step_mz": pa.array(step),
            "n_points": pa.array(n_present),
            "n_points_incomplete": pa.array(incomplete),
        }
    )

    # ── write ─────────────────────────────────────────────────────────
    (out / PROFILE_DIR).mkdir(parents=True, exist_ok=True)
    (out / POINTS_DIR).mkdir(parents=True, exist_ok=True)
    try:
        keep_rows = np.flatnonzero(present)
        counts = n_present
        with PeakWriter(out / PROFILE_DIR / part_name(0), byte_stream_split=True) as writer:
            writer.write(peak_rows(rep, counts, mz_grid[keep_rows], mean[keep_rows]))
        pq.write_table(
            pa.table(
                {
                    "scan": pa.array((owner[keep_rows] + 1).astype(np.int32)),
                    "mz": pa.array(mz_grid[keep_rows]),
                    "intensity_sd": pa.array(sd[keep_rows], mask=count[keep_rows] < 2),
                    "n_scans": pa.array(count[keep_rows].astype(np.int32)),
                },
                schema=POINTS_SCHEMA,
            ),
            out / POINTS_DIR / part_name(0),
            compression="zstd",
        )
        pq.write_table(rep, out / SCAN_METADATA_FILENAME, compression="zstd")
        pq.write_table(summary, out / SUMMARY_FILENAME, compression="zstd")
        shutil.copy2(source / ACQUISITION_METADATA_FILENAME, out / ACQUISITION_METADATA_FILENAME)
        parameters = {"skip_first": skip_first, "off_grid_tolerance": OFF_GRID_TOLERANCE}
        levels: dict[int, dict[str, Any]] = {}
        for lvl, n in zip(rep.column("level").to_pylist(), n_present.tolist()):
            entry = levels.setdefault(int(lvl), {"level": int(lvl), "n_scans": 0, "n_peaks": 0, "peaks_from": {"profile": 0}})
            entry["n_scans"] += 1
            entry["n_peaks"] += int(n)
            entry["peaks_from"]["profile"] += 1
        record = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": PROFILE_BUNDLE_KIND,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_file": manifest.get("source_file"),
            "source_sha256": manifest.get("source_sha256"),
            "parameters": {**parameters, "peak_bundle": None},
            "outputs": {
                "profile": PROFILE_DIR,
                "average_points": POINTS_DIR,
                "average_summary": SUMMARY_FILENAME,
                "scan_metadata": SCAN_METADATA_FILENAME,
                "acquisition_metadata": ACQUISITION_METADATA_FILENAME,
            },
            "run_metadata": {
                **(manifest.get("run_metadata") or {}),
                "n_scans": n_groups,
                "levels": [levels[k] for k in sorted(levels)],
                "n_scans_averaged": int(kept_per_group.sum()),
                "n_scans_skipped": int(excluded.sum()),
                "n_points_read": n_read,
                "n_points_off_grid": n_off,
            },
            "derived_from": {
                "name": SUFFIX,
                "source": os.path.relpath(source, out.parent),
                "algorithm": "mean",
                "version": VERSION,
                "parameters": parameters,
            },
        }
        (out / MANIFEST_FILENAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out
