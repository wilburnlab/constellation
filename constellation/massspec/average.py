"""One spectrum per kind of scan from a direct-infusion profile run.

With an analyte infused at constant concentration, every repeat of a scan is a
replicate of it, and what a run says is best read from their mean. A kind of
scan is what ``acquisition_structure.scan_kinds`` says it is (filter string,
level, FAIMS voltage, activation, energy, precursor, isolation window, scan
range): a run that repeats one scan gives one spectrum, one that cycles
through several gives one for each.

The ion trap stores a profile scan's points on a fixed m/z grid whose step
depends on the scan rate (1/30 and 1/15 m/z on a Stellar), every point of the
range, zeros included. Each kind's grid is read from one of its scans (the
step from the scan's whole span, since a median of neighbouring differences
drifts over tens of thousands of points), and every point is placed on it by
index: nothing is interpolated or re-binned. A point that does not land on the
grid is counted and reported, never merged into a neighbour. The average is
then a group-by on (kind, grid index), done by index: one pass finds each
kind's extent, a second adds every point into its place.

Every scan is averaged by default. ``skip_first`` leaves out the first scans
of each kind, the first fills before the AGC has any history: on one Stellar
infusion series every scan of the first cycle ran at the injection-time
ceiling, on another the first scans filled like the rest, and with ~100
replicates one scan barely moves the mean. The summary reports the injection
time of skipped scans beside the rest, so the choice can be checked.

Intensities are the instrument's, normalised by injection time, so the mean of
scans with different fills is a mean of rates, not of ion counts. Spray drifts
over a run; a mean preserves the shape of each spectrum and, in a run that
cycles through its scans, the ratios between them, since every kind of scan is
sampled across the same stretch of time.

What it writes is a profile bundle (``profile/``, scan and acquisition
metadata, a manifest written last naming its source), one spectrum per kind
of scan, so the commands that read a profile bundle read it: ``massspec
centroid`` makes a peak list from it. Its spectra are numbered by kind of
scan, not by scan, and its manifest says so (``spectrum_index``): it is not
one of the run's bundles (``bundle.related_bundles``), it is not an
acquisition to register, and a sweep, a spectra file or a precursor-major
layout refuses it. Its scan metadata holds what is true of each average: the
columns that define the kind, the mean time and TIC, the median injection
time and the mean spectrum's base peak, method settings that every replicate
shares, and null for one scan's readings. Beside that it keeps what has no column
in the peak table: each point's standard deviation and number of replicates
(``average_points/``, :data:`AVERAGE_POINTS_TABLE`) and each spectrum's
replicates, injection-time and TIC summaries (``average_summary.parquet``,
:data:`AVERAGE_SUMMARY_TABLE`).

Only ion-trap profile points are averaged: an Orbitrap's profile is not stored
on a grid, and a scan acquired as centroids has no profile to average.
Ported from the lab's infusion-averaging script.
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core.io.schemas import register_schema
from constellation.massspec.acquisition_structure import KIND_NUMBER_COLUMNS, KIND_TEXT_COLUMNS, scan_kinds
from constellation.massspec.bundle import (
    BUNDLE_SCHEMA_VERSION,
    KIND_OF_SCAN_INDEX,
    MANIFEST_FILENAME,
    PROFILE_BUNDLE_KIND,
    PROFILE_DIR,
    SPECTRUM_INDEX_KEY,
    OutdatedBundleError,
    PeakWriter,
    derived_bundle_writer,
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
#: Per averaged spectrum: replicates, injection time, TIC, grid.
SUMMARY_FILENAME = "average_summary.parquet"
#: What an averaged bundle is named for, after its run.
SUFFIX = "average"
#: Version of the averaging, recorded in the manifest.
VERSION = 1
#: The analyzer whose profile is stored on a grid.
ION_TRAP = "ITMS"
#: How far from a grid index, as a fraction of the step, a point may sit.
OFF_GRID_TOLERANCE = 0.05
#: How many grid points the average is finished and written at a time: a
#: group of whole kinds of scan (a kind with more is a group of its own), so
#: what is held beside the sums does not grow with the number of kinds.
FINISH_POINTS = 1 << 20

AVERAGE_POINTS_TABLE: pa.Schema = pa.schema(
    [
        pa.field("scan", pa.int32(), nullable=False),  # the averaged spectrum
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("intensity_sd", pa.float64()),  # null under two replicates
        pa.field("n_scans", pa.int32(), nullable=False),  # replicates with a point here
    ]
)

AVERAGE_SUMMARY_TABLE: pa.Schema = pa.schema(
    [
        pa.field("scan", pa.int32(), nullable=False),  # the averaged spectrum
        pa.field("level", pa.int8()),
        pa.field("filter_string", pa.string()),
        pa.field("n_scans", pa.int32(), nullable=False),  # averaged
        pa.field("n_skipped", pa.int32(), nullable=False),  # left out by skip_first
        pa.field("first_scan", pa.int32()),  # first and last scan averaged
        pa.field("last_scan", pa.int32()),
        pa.field("rt_first_s", pa.float64()),
        pa.field("rt_last_s", pa.float64()),
        pa.field("iit_median_ms", pa.float64()),
        pa.field("iit_p95_ms", pa.float64()),
        pa.field("iit_skipped_median_ms", pa.float64()),
        # The fill limit: the method's (max_ion_time_ms) where the run records
        # it, else the longest fill seen at the level (an estimate).
        pa.field("iit_ceiling_ms", pa.float64()),
        pa.field("iit_ceiling_recorded", pa.bool_()),
        pa.field("frac_iit_at_ceiling", pa.float64()),
        pa.field("tic_mean", pa.float64()),
        pa.field("tic_cv", pa.float64()),  # sample SD over mean
        pa.field("tic_last_over_first", pa.float64()),  # last five over first five
        pa.field("grid_origin_mz", pa.float64()),
        pa.field("grid_step_mz", pa.float64()),
        pa.field("n_points", pa.int64()),
        pa.field("n_points_incomplete", pa.int64()),  # points not in every replicate
        pa.field("n_points_off_grid", pa.int64()),
    ]
)

register_schema("AveragePointsTable", AVERAGE_POINTS_TABLE)
register_schema("AverageSummaryTable", AVERAGE_SUMMARY_TABLE)


# ── what is averaged ──────────────────────────────────────────────────


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


def _ordinal_in_kind(kind: np.ndarray, n_kinds: int) -> np.ndarray:
    """0 for a kind's first scan (in stored order), 1 for its second, ..."""
    order = np.argsort(kind, kind="stable")
    counts = np.bincount(kind, minlength=n_kinds)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ordinal = np.empty(kind.size, np.int64)
    ordinal[order] = np.arange(kind.size) - np.repeat(starts, counts)
    return ordinal


def _rows_of(row_of: np.ndarray, scan: np.ndarray) -> np.ndarray:
    """Each scan number's row of scan metadata. Points of a scan the metadata
    does not hold are refused: ``row_of`` holds -1 there, which as an index
    is the last row, and would put the points in that row's kind."""
    inside = (scan >= 0) & (scan < row_of.size)
    rows = np.full(scan.size, -1, np.int64)
    rows[inside] = row_of[scan[inside]]
    if (rows < 0).any():
        missing = np.unique(scan[rows < 0])
        raise ValueError(
            f"the bundle has points of scan {int(missing[0])} ({missing.size} scan(s) in all) with no "
            f"scan metadata: it is inconsistent; convert the run again"
        )
    return rows


def _reference_grid(m: np.ndarray) -> tuple[float, float]:
    """One scan's grid: the m/z of the grid point nearest its middle, and the
    step. The step is first read from the scan's whole span (a median of
    neighbouring differences drifts over tens of thousands of points); every
    point is then numbered by whole steps from the middle, on the phase most
    points share, and the line through them is fitted by least squares, so
    a stray point (an endpoint off the grid) barely moves either."""
    span = m[-1] - m[0]
    n_steps = int(np.rint(span / np.median(np.diff(m))))
    if n_steps <= 0:
        return np.nan, np.nan
    x = (m - m[m.size // 2]) / (span / n_steps)
    k = np.rint(x - np.median(x - np.rint(x)))
    step, at_middle = np.polyfit(k, m, 1)
    return float(at_middle), float(step)


def _grids(bundle: Path, row_of: np.ndarray, kind: np.ndarray, kept: np.ndarray, n_kinds: int):
    """Each kind's grid, from the scans averaged and no others: its phase and
    step from the first of them with two or more points
    (:func:`_reference_grid`), and its extent in whole steps from that
    phase to the lowest and highest first and last point on it of every
    averaged scan. An endpoint off the phase extends nothing: it is off the
    grid, and counted. Returns each kind's origin (its grid's first point),
    step and number of grid points."""
    phase = np.full(n_kinds, np.nan)
    step = np.full(n_kinds, np.nan)
    firsts, lasts, kinds = [], [], []
    for block in iter_scan_blocks(bundle, columns=("mz",), points=True):
        rows = _rows_of(row_of, block.scan)
        k = kind[rows]
        sizes = np.diff(block.offsets)
        use = kept[rows] & (sizes > 0)
        mz = block.columns["mz"]
        firsts.append(mz[block.offsets[:-1][use]])
        lasts.append(mz[block.offsets[1:][use] - 1])
        kinds.append(k[use])
        candidates = np.flatnonzero(use & (sizes >= 2) & np.isnan(step[k]))
        _, first = np.unique(k[candidates], return_index=True)
        for i in candidates[first]:  # one scan per kind, not one per scan
            phase[k[i]], step[k[i]] = _reference_grid(mz[block.offsets[i] : block.offsets[i + 1]])
    g = np.concatenate(kinds) if kinds else np.zeros(0, np.int64)
    low = np.full(n_kinds, np.inf)
    high = np.full(n_kinds, -np.inf)
    for ends, extreme, into in ((firsts, np.minimum, low), (lasts, np.maximum, high)):
        values = np.concatenate(ends) if ends else np.zeros(0)
        x = (values - phase[g]) / step[g]
        k = np.rint(x)
        on = np.isfinite(x) & (np.abs(x - k) <= OFF_GRID_TOLERANCE)
        extreme.at(into, g[on], k[on])
    known = np.isfinite(low) & np.isfinite(high) & (high >= low)
    extent = np.zeros(n_kinds, np.int64)
    extent[known] = (high[known] - low[known]).astype(np.int64) + 1
    origin = np.where(known, phase + np.where(known, low, 0.0) * step, np.nan)
    return origin, step, extent


def _accumulate(bundle: Path, row_of: np.ndarray, kind: np.ndarray, kept: np.ndarray, origin, step, extent):
    """Sum, sum of squares and count of the averaged scans' points at each
    grid point of each kind (kinds laid end to end), and each kind's number
    of points that sit on no grid point.

    A point is placed by index and added in place (``np.add.at``, in stored
    order, so every run gives the same bytes); nothing proportional to the
    grid is made per block of points.
    """
    offset = np.concatenate(([0], np.cumsum(extent)[:-1]))
    total = int(extent.sum())
    acc = np.zeros(total)
    acc2 = np.zeros(total)
    count = np.zeros(total, np.int64)
    off_grid = np.zeros(origin.size, np.int64)
    for block in iter_scan_blocks(bundle, columns=("mz", "intensity"), points=True):
        rows = _rows_of(row_of, block.scan)
        keep_scan = kept[rows]
        if not keep_scan.any():
            continue
        sizes = np.diff(block.offsets)
        keep = np.repeat(keep_scan, sizes)
        g = np.repeat(kind[rows], sizes)[keep]
        mz = block.columns["mz"][keep]
        intensity = block.columns["intensity"][keep].astype(np.float64)
        x = (mz - origin[g]) / step[g]
        k = np.rint(x)
        on = np.isfinite(x) & (np.abs(x - k) <= OFF_GRID_TOLERANCE) & (k >= 0) & (k < extent[g])
        off_grid += np.bincount(g[~on], minlength=off_grid.size)
        flat = offset[g[on]] + k[on].astype(np.int64)
        w = intensity[on]
        np.add.at(acc, flat, w)
        np.add.at(acc2, flat, w * w)
        np.add.at(count, flat, 1)
    return acc, acc2, count, off_grid


def _groups(extent: np.ndarray, limit: int):
    """Consecutive kinds ``(first, end)`` whose grids hold at most ``limit``
    points together (a kind larger than that is a group of its own)."""
    first = 0
    while first < extent.size:
        end, held = first + 1, int(extent[first])
        while end < extent.size and held + int(extent[end]) <= limit:
            held += int(extent[end])
            end += 1
        yield first, end
        first = end


# ── per spectrum ──────────────────────────────────────────────────────


def _per_kind(kind: np.ndarray, values: np.ndarray, n_kinds: int):
    """``values`` grouped by kind: (sorted values, start of each kind,
    count of each kind), NaN left out."""
    good = np.isfinite(values)
    order = np.lexsort((values[good], kind[good]))
    sorted_values = values[good][order]
    counts = np.bincount(kind[good], minlength=n_kinds)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return sorted_values, starts, counts


def _quantile(kind: np.ndarray, values: np.ndarray, n_kinds: int, q: float) -> np.ndarray:
    """Each kind's ``q`` quantile (numpy's linear rule), NaN for none."""
    v, starts, counts = _per_kind(kind, values, n_kinds)
    out = np.full(n_kinds, np.nan)
    has = counts > 0
    at = starts[has] + q * (counts[has] - 1)
    below = np.floor(at).astype(np.int64)
    above = np.minimum(below + 1, starts[has] + counts[has] - 1)
    out[has] = v[below] + (at - below) * (v[above] - v[below])
    return out


def _extreme(kind: np.ndarray, values: np.ndarray, n_kinds: int, fn) -> np.ndarray:
    """Each kind's smallest (``np.minimum``) or largest (``np.maximum``)
    finite value, NaN for none."""
    good = np.isfinite(values)
    acc = np.full(n_kinds, np.inf if fn is np.minimum else -np.inf)
    fn.at(acc, kind[good], values[good])
    return np.where(np.isfinite(acc), acc, np.nan)


def _mean(kind: np.ndarray, values: np.ndarray, n_kinds: int, *, ddof: int | None = None) -> np.ndarray:
    """Each kind's mean, or with ``ddof`` its standard deviation; NaN for too few values."""
    good = np.isfinite(values)
    n = np.bincount(kind[good], minlength=n_kinds)
    total = np.bincount(kind[good], values[good], n_kinds)
    mean = np.divide(total, n, out=np.full(n_kinds, np.nan), where=n > 0)
    if ddof is None:
        return mean
    dev = np.bincount(kind[good], (values[good] - mean[kind[good]]) ** 2, n_kinds)
    return np.sqrt(np.divide(dev, n - ddof, out=np.full(n_kinds, np.nan), where=n > ddof))


def _drift(kind: np.ndarray, order_key: np.ndarray, tic: np.ndarray, n_kinds: int) -> np.ndarray:
    """Each kind's mean TIC over its last five scans over its first five, in
    scan order; NaN for fewer than ten scans or a first five summing to 0."""
    order = np.lexsort((order_key, kind))
    counts = np.bincount(kind, minlength=n_kinds)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    out = np.full(n_kinds, np.nan)
    for i in np.flatnonzero(counts >= 10):
        series = tic[order[starts[i] : starts[i] + counts[i]]]
        first = series[:5].mean()
        if first > 0:
            out[i] = series[-5:].mean() / first
    return out


def _known(values: np.ndarray) -> pa.Array:
    """Floats with NaN as null: a value no replicate recorded, or one there
    is nothing to take from (the median of no skipped scans), is unknown."""
    return pa.array(values, mask=~np.isfinite(values))


def _column(scans: pa.Table, name: str) -> np.ndarray:
    if name not in scans.column_names:
        return np.full(scans.num_rows, np.nan)
    return scans.column(name).cast(pa.float64()).fill_null(np.nan).to_numpy(zero_copy_only=False)


def _summary(scans, kind, kept, n_kinds, rep, origin, step, n_points, incomplete, off_per_kind) -> pa.Table:
    numbers = scans.column("scan").to_numpy().astype(np.float64)
    level = scans.column("level").to_numpy().astype(np.int64)
    iit = _column(scans, "iit")
    tic = _column(scans, "tic")
    rt = _column(scans, "rt")
    k_kept, k_skip = kind[kept], kind[~kept]

    # The fill limit of each kind: the method's, else the longest fill seen at its level.
    recorded = _extreme(kind, _column(scans, "max_ion_time_ms"), n_kinds, np.maximum)
    longest_at_level = np.full(int(level.max()) + 1, np.nan)
    for lvl in np.unique(level):
        at = iit[level == lvl]
        if np.isfinite(at).any():
            longest_at_level[lvl] = np.nanmax(at)
    kind_level = np.zeros(n_kinds, np.int64)
    kind_level[kind] = level
    ceiling = np.where(np.isfinite(recorded), recorded, longest_at_level[kind_level])
    known = np.isfinite(iit) & np.isfinite(ceiling[kind])
    at_ceiling = np.where(known, (iit >= ceiling[kind] - 1e-3).astype(np.float64), np.nan)

    tic_mean = _mean(k_kept, tic[kept], n_kinds)
    tic_sd = _mean(k_kept, tic[kept], n_kinds, ddof=1)
    tic_cv = np.divide(tic_sd, tic_mean, out=np.full(n_kinds, np.nan), where=tic_mean > 0)
    return pa.Table.from_arrays(
        [
            pa.array(np.arange(1, n_kinds + 1, dtype=np.int32)),
            rep.column("level"),
            rep.column("filter_string"),
            pa.array(np.bincount(k_kept, minlength=n_kinds).astype(np.int32)),
            pa.array(np.bincount(k_skip, minlength=n_kinds).astype(np.int32)),
            pa.array(_extreme(k_kept, numbers[kept], n_kinds, np.minimum)).cast(pa.int32()),
            pa.array(_extreme(k_kept, numbers[kept], n_kinds, np.maximum)).cast(pa.int32()),
            _known(_extreme(k_kept, rt[kept], n_kinds, np.minimum)),
            _known(_extreme(k_kept, rt[kept], n_kinds, np.maximum)),
            _known(_quantile(k_kept, iit[kept], n_kinds, 0.5)),
            _known(_quantile(k_kept, iit[kept], n_kinds, 0.95)),
            _known(_quantile(k_skip, iit[~kept], n_kinds, 0.5)),
            _known(ceiling),
            pa.array(np.isfinite(recorded)),
            _known(_mean(k_kept, at_ceiling[kept], n_kinds)),
            _known(tic_mean),
            _known(tic_cv),
            _known(_drift(k_kept, numbers[kept], tic[kept], n_kinds)),
            _known(origin),
            _known(step),
            pa.array(n_points),
            pa.array(incomplete),
            pa.array(off_per_kind),
        ],
        schema=AVERAGE_SUMMARY_TABLE,
    )


#: Columns no averaged spectrum has a value for: one scan's trailer as the
#: instrument wrote it, the lowest intensity one scan stored, and the scans
#: a scan was taken from, by a number that means nothing among kinds of scan.
_NOT_OF_AN_AVERAGE = ("trailer_extras", "intensity_floor", "precursor_scan", "master_scan")


def _agrees(a: pa.Array, b: pa.Array) -> np.ndarray:
    """Element by element, whether ``a`` and ``b`` hold the same value (two
    nulls do)."""
    both_null = a.is_null().to_numpy(zero_copy_only=False) & b.is_null().to_numpy(zero_copy_only=False)
    try:
        same = pc.fill_null(pc.equal(a, b), False).to_numpy(zero_copy_only=False)
    except (pa.ArrowNotImplementedError, pa.ArrowTypeError):  # a nested column (scan windows)
        return np.array([x == y for x, y in zip(a.to_pylist(), b.to_pylist())], dtype=bool)
    return same | both_null


def _kind_metadata(scans: pa.Table, kind: np.ndarray, kept: np.ndarray, n_kinds: int, measured: dict) -> pa.Table:
    """One row of scan metadata per kind of scan, every value true of its
    average: the columns that define a kind (``acquisition_structure``'s
    key) from its first averaged scan; ``measured`` (the spectrum's number,
    mean time and TIC, median injection time, base peak, point count) as
    given; :data:`_NOT_OF_AN_AVERAGE` null; and every other column only where
    all the averaged scans of the kind hold the same value (a method setting),
    else null (an AGC reading, a scan event's index)."""
    kept_rows = np.flatnonzero(kept)
    k_kept = kind[kept_rows]
    first = np.full(n_kinds, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, k_kept, kept_rows)
    rep = scans.take(pa.array(first))
    replicates = scans.take(pa.array(kept_rows))
    defining = set(KIND_TEXT_COLUMNS) | {name for name, _ in KIND_NUMBER_COLUMNS}
    for at, field in enumerate(rep.schema):
        name = field.name
        if name in measured:
            values = measured[name].cast(field.type)
        elif name in _NOT_OF_AN_AVERAGE:
            values = pa.nulls(n_kinds, field.type)
        elif name in defining:
            continue
        else:
            column = rep.column(name).combine_chunks()
            agree = _agrees(replicates.column(name).combine_chunks(), column.take(pa.array(k_kept)))
            same = np.bincount(k_kept, ~agree, n_kinds) == 0
            if same.all():
                continue
            values = pa.array([v if s else None for v, s in zip(column.to_pylist(), same.tolist())], field.type)
        rep = rep.set_column(at, field, values)
    return rep


# ── one run ───────────────────────────────────────────────────────────


def _run_name(source: Path, manifest: dict) -> str:
    return Path(str(manifest.get("source_file") or source.name).replace("\\", "/").rsplit("/", 1)[-1]).stem


def averaged_path(profile_bundle: Path | str, parent: Path | str | None = None) -> Path:
    """Where :func:`average_bundle` writes by default: ``<run>.average`` in
    ``parent``, or beside the profile bundle when ``parent`` is None."""
    source = Path(profile_bundle)
    run = _run_name(source, read_bundle_manifest(source))
    return (Path(parent) if parent is not None else source.parent) / f"{run}.{SUFFIX}"


def average_bundle(
    profile_bundle: Path | str,
    *,
    output_parent: Path | str | None = None,
    skip_first: int = 0,
    force: bool = False,
) -> Path:
    """Average ``profile_bundle`` into one spectrum per kind of scan; returns
    the new bundle, ``<run>.average`` in ``output_parent`` (default: beside
    its source).

    ``skip_first`` leaves out the first that many scans of each kind (the
    AGC's first fills); the default, 0, keeps every scan. An average already
    there is made again only with ``force``, and replaced only once the new
    one is whole; nothing else there is ever removed, and only one process
    writes it at a time (``bundle.derived_bundle_writer``).
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
    out = averaged_path(source, output_parent)
    with derived_bundle_writer(out, source, force=force) as staged:
        _average_into(staged, source, manifest, out, skip_first)
    return out


def _average_into(partial: Path, source: Path, manifest: dict, out: Path, skip_first: int) -> None:
    """Average ``source`` into the directory ``partial``, the manifest last,
    as the bundle that will be put at ``out``."""
    scans = pq.read_table(source / SCAN_METADATA_FILENAME)
    _refuse(scans)
    kind, n_kinds = scan_kinds(scans)
    if n_kinds == 0:
        raise ValueError(f"{source} has no scans")
    kept = _ordinal_in_kind(kind, n_kinds) >= skip_first
    n_scans = np.bincount(kind[kept], minlength=n_kinds)
    if (n_scans == 0).any():
        raise ValueError(
            f"skip_first={skip_first} leaves {int((n_scans == 0).sum())} kind(s) of scan "
            f"with nothing to average; pass a smaller skip_first"
        )

    numbers = scans.column("scan").to_numpy().astype(np.int64)
    row_of = np.full(int(numbers.max()) + 1, -1, np.int64)
    row_of[numbers] = np.arange(numbers.size)
    origin, step, extent = _grids(source, row_of, kind, kept, n_kinds)
    if np.isnan(step).any():
        raise ValueError(f"{int(np.isnan(step).sum())} kind(s) of scan have no profile points to average")

    # ── the average ───────────────────────────────────────────────────
    acc, acc2, count, off_per_kind = _accumulate(source, row_of, kind, kept, origin, step, extent)
    n_read = int(count.sum() + off_per_kind.sum())
    placeholder = np.zeros(n_kinds)
    measured = {
        "scan": pa.array(np.arange(1, n_kinds + 1, dtype=np.int32)),
        "rt": pa.array(_mean(kind[kept], _column(scans, "rt")[kept], n_kinds)),
        "tic": _known(_mean(kind[kept], _column(scans, "tic")[kept], n_kinds)),
        "iit": _known(_quantile(kind[kept], _column(scans, "iit")[kept], n_kinds, 0.5)),
        "base_peak_mz": pa.array(placeholder),  # from the averages, below
        "base_peak_intensity": pa.array(placeholder),
        "peak_count": pa.array(placeholder),
    }
    rep = _kind_metadata(scans, kind, kept, n_kinds, measured)

    # ── finished and written a group of kinds at a time ───────────────
    offset = np.concatenate(([0], np.cumsum(extent)[:-1]))
    n_points = np.zeros(n_kinds, np.int64)
    incomplete = np.zeros(n_kinds, np.int64)
    bp_mz = np.full(n_kinds, np.nan)
    bp_int = np.zeros(n_kinds)
    (partial / PROFILE_DIR).mkdir()
    (partial / POINTS_DIR).mkdir()
    with (
        PeakWriter(partial / PROFILE_DIR / part_name(0), byte_stream_split=True) as writer,
        pq.ParquetWriter(str(partial / POINTS_DIR / part_name(0)), AVERAGE_POINTS_TABLE, compression="zstd") as points,
    ):
        for first, end in _groups(extent, FINISH_POINTS):
            a, b = int(offset[first]), int(offset[end - 1] + extent[end - 1])
            present = count[a:b] > 0
            owner = np.repeat(np.arange(first, end), extent[first:end])
            index = (np.arange(a, b) - offset[owner])[present]
            owner = owner[present]
            n_ = count[a:b][present]
            mean = acc[a:b][present] / n_
            var = np.where(n_ > 1, (acc2[a:b][present] - n_ * mean**2) / np.maximum(n_ - 1, 1), np.nan)
            sd = np.sqrt(np.maximum(var, 0.0))
            mz = origin[owner] + index * step[owner]
            per = np.bincount(owner - first, minlength=end - first).astype(np.int64)
            n_points[first:end] = per
            incomplete[first:end] = np.bincount(owner - first, n_ != n_scans[owner], end - first)
            starts = np.concatenate(([0], np.cumsum(per)[:-1]))
            for i in np.flatnonzero(per):  # one argmax per averaged spectrum
                j = starts[i] + int(np.argmax(mean[starts[i] : starts[i] + per[i]]))
                bp_mz[first + i], bp_int[first + i] = mz[j], mean[j]
            writer.write(peak_rows(rep.slice(first, end - first), per, mz, mean))
            points.write_table(
                pa.Table.from_arrays(
                    [pa.array((owner + 1).astype(np.int32)), pa.array(mz), pa.array(sd, mask=n_ < 2),
                     pa.array(n_.astype(np.int32))],
                    schema=AVERAGE_POINTS_TABLE,
                )
            )
    for name, values in (("base_peak_mz", _known(bp_mz)), ("base_peak_intensity", pa.array(bp_int)),
                         ("peak_count", pa.array(n_points))):
        at = rep.schema.get_field_index(name)
        if at >= 0:
            rep = rep.set_column(at, rep.schema.field(name), values.cast(rep.schema.field(name).type))
    summary = _summary(scans, kind, kept, n_kinds, rep, origin, step, n_points, incomplete, off_per_kind)
    pq.write_table(rep, partial / SCAN_METADATA_FILENAME, compression="zstd")
    pq.write_table(summary, partial / SUMMARY_FILENAME, compression="zstd")
    shutil.copy2(source / ACQUISITION_METADATA_FILENAME, partial / ACQUISITION_METADATA_FILENAME)
    parameters = {"skip_first": skip_first, "off_grid_tolerance": OFF_GRID_TOLERANCE}
    levels: dict[int, dict[str, Any]] = {}
    for lvl, n in zip(rep.column("level").to_pylist(), n_points.tolist()):
        entry = levels.setdefault(int(lvl), {"level": int(lvl), "n_scans": 0, "n_peaks": 0, "peaks_from": {"profile": 0}})
        entry["n_scans"] += 1
        entry["n_peaks"] += int(n)
        entry["peaks_from"]["profile"] += 1
    record = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": PROFILE_BUNDLE_KIND,
        # Its spectra are kinds of scan, not the run's scans: it is no
        # member of the run's family, and sweeps and searches refuse it.
        SPECTRUM_INDEX_KEY: KIND_OF_SCAN_INDEX,
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
            "n_scans": n_kinds,
            "levels": [levels[k] for k in sorted(levels)],
            "n_scans_averaged": int(n_scans.sum()),
            "n_scans_skipped": int((~kept).sum()),
            "n_points_read": n_read,
            "n_points_off_grid": int(off_per_kind.sum()),
            # Of the averaged spectra, as summarize reads it: kinds whose
            # scans recorded no injection time (the run's own count of
            # scans is not the average's).
            "n_scans_without_injection_time": int(rep.column("iit").null_count),
        },
        "derived_from": {
            "name": SUFFIX,
            "source": os.path.relpath(source, out.parent),
            "algorithm": "mean",
            "version": VERSION,
            "parameters": parameters,
        },
    }
    (partial / MANIFEST_FILENAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


# ── a directory of runs ───────────────────────────────────────────────


def _survey(directory: Path | str) -> tuple[list[Path], list[tuple[Path, str]]]:
    """The run profile bundles directly inside ``directory`` (not recursive),
    sorted by name, and the bundles there that cannot be read, with why.
    Peak bundles, bundles made from another (an earlier ``.average``) and
    directories that are not bundles are left out."""
    runs: list[Path] = []
    problems: list[tuple[Path, str]] = []
    for child in sorted(Path(directory).iterdir(), key=lambda p: p.name):
        if child.name.startswith(".") or not (child / MANIFEST_FILENAME).is_file():
            continue
        try:
            manifest = read_bundle_manifest(child)
        except OutdatedBundleError as exc:
            problems.append((child, str(exc)))
            continue
        except ValueError as exc:  # another kind of directory; or a manifest that does not parse
            if is_profile_bundle(child):
                problems.append((child, f"its manifest cannot be read: {exc}"))
            continue
        if is_profile_bundle(child) and not manifest.get("derived_from"):
            runs.append(child)
    return runs, problems


def _is_average_of(out: Path, source: Path) -> bool:
    """Whether ``out`` is a finished average of ``source``: a bundle in the
    current format whose manifest says it is an average (its spectra kinds
    of scan) made from ``source``, with what an average holds there."""
    try:
        manifest = read_bundle_manifest(out)
    except (ValueError, OSError):
        return False
    made = manifest.get("derived_from") or {}
    return (
        manifest.get(SPECTRUM_INDEX_KEY) == KIND_OF_SCAN_INDEX
        and made.get("name") == SUFFIX
        and bool(made.get("source"))
        and (out.parent / str(made["source"])).resolve() == source.resolve()
        and (out / PROFILE_DIR).is_dir()
        and (out / SCAN_METADATA_FILENAME).is_file()
        and (out / SUMMARY_FILENAME).is_file()
    )


def find_profile_bundles(directory: Path | str) -> list[Path]:
    """The run profile bundles directly inside ``directory`` (not recursive),
    sorted by name. Peak bundles and bundles made from another bundle (an
    earlier ``.average``) are not runs to average and are left out."""
    return _survey(directory)[0]


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
    ``output_parent`` (default: beside its source). A run whose average is
    there and finished is skipped unless ``force``; a run that fails, or a
    bundle that cannot be read (an older format), is reported and the rest
    carry on."""
    if skip_first < 0:
        raise ValueError(f"skip_first must be 0 or more, got {skip_first}")
    runs, problems = _survey(directory)
    results = [AverageResult(bundle, "error", None, detail) for bundle, detail in problems]
    for bundle in runs:
        try:
            out = averaged_path(bundle, output_parent)
            if _is_average_of(out, bundle) and not force:
                results.append(AverageResult(bundle, "skipped", out, "already averaged"))
                continue
            made = average_bundle(bundle, output_parent=output_parent, skip_first=skip_first, force=force)
            results.append(AverageResult(bundle, "ok", made))
        except Exception as exc:  # one bad run must not stop the rest
            results.append(AverageResult(bundle, "error", None, f"{type(exc).__name__}: {exc}"))
    return sorted(results, key=lambda r: r.bundle.name)


__all__ = [
    "AVERAGE_POINTS_TABLE",
    "AVERAGE_SUMMARY_TABLE",
    "OFF_GRID_TOLERANCE",
    "POINTS_DIR",
    "SUFFIX",
    "SUMMARY_FILENAME",
    "AverageResult",
    "average_bundle",
    "average_directory",
    "averaged_path",
    "find_profile_bundles",
]
