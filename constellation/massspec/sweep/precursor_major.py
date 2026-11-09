"""The sweep read precursor-major: a unit of queries at a time, from the
bundle's precursor-major layout (``massspec.precursor_major``), reduced
where it is read.

The scan-major sweep cuts a run into scan ranges, writes every match of
every query and shuffles them to the reducers, because a query's trace
crosses scan ranges. A query is only ever matched in scans whose isolation
window holds its precursor (and at MS1 only near its own m/z), so read the
other way the trace does not cross anything: a unit of queries is read,
matched and reduced in one task, and only the reducers' outputs are
written. At proteome scale with no retention-time window, that is the
difference between some 100 GB of matches for one GPF fraction and the
reducers' rows (``docs/plans/sweep.md``, "Precursor-major layout").

A unit is a run of queries in (precursor m/z, query) order, cut by the
cells its queries are eligible for (``SweepSettings.unit_cells``), and at
a window bound where one is near. What it reads is every scan any of its
queries is eligible in, so the cut only decides how often a window is read
(at most twice), never what a query is matched against:

- at MS2 and above, the scans of each window that holds a query's
  precursor m/z (inclusive, as the scan-major sweep's test), every scan
  without bounds, and for a query with no precursor m/z every scan;
- at MS1, each isotope channel's m/z plus its tolerance and the largest
  correction the m/z axis makes, so every point that could be matched, or
  could be the nearer of two, is there.

A unit's queries are matched and reduced in batches of ascending query
(``SweepSettings.batch_cells``) by the scan-major sweep's own code
(``_map_block``, ``_batch``, ``_reduce``) on a plan taken for the batch
(``QueryPlan.take``), so a query's rows are the rows the scan-major sweep
gives it; each output is one part per unit, rows in query order within it.
Refused here: claims (a point is claimed by every unit whose queries
matched it, MS1 points by every unit), the gate's MS1 isotope evidence
(it ranks a whole scan's points, of which a unit reads a slice), a table
of points, and a bundle without a current layout.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core.progress import emit_done, emit_progress, emit_start
from constellation.massspec.precursor_major import PrecursorMajor, find_precursor_major
from constellation.massspec.sweep.queries import QueryPlan
from constellation.massspec.sweep.reduce import Ledger, eligible_counts

__all__ = ["LAYOUT_KEY", "sweep_precursor_major", "precursor_units"]

#: Where an output says the sweep that made it read precursor-major.
LAYOUT_KEY = b"x.massspec.sweep.layout"

_CACHE: dict[str, Any] = {}


def _refuse(source, plan: QueryPlan, settings) -> PrecursorMajor:
    if isinstance(source, pa.Table):
        raise ValueError("a table of points has no precursor-major layout: give a bundle, or sweep scan-major")
    if settings.claims:
        raise ValueError(
            "claims need every query that matched a point, and a precursor-major sweep holds one unit of "
            "queries at a time (MS1 points belong to every unit): sweep with layout='scan_major' for claims"
        )
    if settings.gate is not None and settings.gate.min_isotopes is not None and 1 in plan.levels:
        raise ValueError(
            "the gate's MS1 isotope evidence ranks a whole MS1 scan's points, and a precursor-major sweep "
            "reads each unit's slice of them: sweep with layout='scan_major' for min_isotopes"
        )
    bundle = Path(source)
    layout = find_precursor_major(bundle)
    if layout is None:
        raise ValueError(
            f"{bundle} has no current precursor-major layout: write it with `constellation massspec "
            f"precursor-major {bundle}` (convert writes it unless told not to), or sweep scan-major"
        )
    return layout


def _ledger(bundle: Path, layout: PrecursorMajor, window: tuple[int, int] | None) -> Ledger:
    """The scan-major sweep's ledger: every scan, with the points of those
    the sweep reads (all levels, in the range when there is one)."""
    from constellation.massspec.sweep.engine import _scan_metadata

    ledger = _scan_metadata(bundle)
    counts = layout.scans()
    scan = counts.column("scan").to_numpy().astype(np.int64)
    n = counts.column("n_points").to_numpy().astype(np.int64)
    if window is not None:
        keep = (scan >= window[0]) & (scan < window[1])
        scan, n = scan[keep], n[keep]
    ledger.n_points[ledger.rows_of(scan)] = n
    return ledger


# ── units ─────────────────────────────────────────────────────────────


def precursor_units(plan: QueryPlan, ledger: Ledger, unit_cells: int) -> list[np.ndarray]:
    """The plan's queries in units: runs in (precursor m/z, query) order of
    about ``unit_cells`` eligible cells each, cut at a window bound where
    one lies in the last half of a unit; queries without a precursor m/z
    after the rest. Each unit's queries ascending. Every query is in one."""
    n = plan.n_queries
    if n == 0:
        return []
    every = np.arange(n, dtype=np.int64)
    cells = np.zeros(n, np.int64)
    for level in plan.levels:
        channels = plan.channels_for_level(level)
        if channels is not None:
            cells += eligible_counts(plan, ledger, level, every) * channels.counts()
    order = plan.precursor_order
    pmz = plan.precursor_mz[order]
    finite = int(np.count_nonzero(~np.isnan(pmz)))
    frag = next((lv for lv in plan.levels if lv > 1), None)
    bounds = np.empty(0)
    if frag is not None:
        index = ledger.window_index(frag)
        bounded = ~index.unbounded
        bounds = np.unique(np.concatenate([index.lower[bounded], index.upper[bounded]]))
    # Which side of every window bound a query lies on: a cut between two
    # queries of different segments is a cut at a bound.
    segment = np.searchsorted(bounds, pmz[:finite], side="right")
    weight = np.cumsum(cells[order] + 1)
    units = []
    for lo, hi, snap in ((0, finite, True), (finite, n, False)):
        start = lo
        while start < hi:
            base = weight[start - 1] if start else 0
            end = min(max(int(np.searchsorted(weight, base + unit_cells, side="right")), start + 1), hi)
            if snap and end < hi:
                half = start + (end - start) // 2
                changes = np.flatnonzero(segment[half + 1:end + 1] != segment[half:end]) + half + 1
                if changes.size:
                    end = int(changes[-1])
            units.append(np.sort(order[start:end]))
            start = end
    return units


# ── reading a unit's points ──────────────────────────────────────────


def _row_group_ranges(f: pq.ParquetFile, column: str) -> list[tuple[Any, Any]]:
    meta = f.metadata
    col = f.schema_arrow.get_field_index(column)
    out = []
    for g in range(meta.num_row_groups):
        stats = meta.row_group(g).column(col).statistics
        out.append((stats.min, stats.max) if stats is not None and stats.has_min_max else (None, None))
    return out


def _eligible_scans(plan: QueryPlan, ledger: Ledger, level: int, window, scans_of_windows=None) -> np.ndarray:
    """Scan numbers of ``level`` the unit can be matched in: with points and
    a time, in the range, inside the span of its queries' intervals, and
    (when given) in one of ``scans_of_windows`` (ledger rows)."""
    has = plan.channels_for_level(level).counts() > 0
    rows = np.flatnonzero((ledger.level == level) & (ledger.n_points > 0) & ~np.isnan(ledger.rt)) \
        if scans_of_windows is None else scans_of_windows  # fmt: skip
    if not has.any():
        return np.empty(0, np.int64)
    t0, t1 = float(plan.rt_lo[has].min()), float(plan.rt_hi[has].max())
    rt = ledger.rt[rows]
    rows = rows[(rt >= t0) & (rt <= t1)]
    scan = ledger.scan[rows]
    if window is not None:
        scan = scan[(scan >= window[0]) & (scan < window[1])]
    return np.sort(scan)


def _read_fragment_level(layout: PrecursorMajor, plan: QueryPlan, ledger: Ledger, level: int, window):
    """The points of every scan of ``level`` (≥ 2) in a window holding one
    of the unit's precursors: (scan, mz, intensity, stored position) per
    point, each scan's points together and in stored order."""
    index = ledger.window_index(level)
    has = plan.channels_for_level(level).counts() > 0
    pmz = plan.precursor_mz[has]
    finite = np.sort(pmz[~np.isnan(pmz)])
    every = bool(np.isnan(pmz).any())
    lo = np.searchsorted(finite, index.lower, side="left")
    hi = np.searchsorted(finite, index.upper, side="right")
    chosen = index.unbounded | (hi > lo) | every if pmz.size else np.zeros(index.n_windows, bool)
    if not chosen.any():
        return None
    windows = np.flatnonzero(chosen)
    rows = np.concatenate([index.rows[index.starts[w]:index.starts[w + 1]] for w in windows])
    scans = _eligible_scans(plan, ledger, level, window, rows)
    if scans.size == 0:
        return None
    lowers = np.where(index.unbounded[windows], -np.inf, index.lower[windows])
    lo_min, lo_max = float(lowers.min()), float(lowers.max())
    wanted = pa.array(scans)
    parts = []
    for path in layout.files(level):
        f = pq.ParquetFile(path)
        lower_ranges = _row_group_ranges(f, "lower")
        scan_ranges = _row_group_ranges(f, "scan")
        groups = []
        for g, ((a, b), (s0, s1)) in enumerate(zip(lower_ranges, scan_ranges, strict=True)):
            if a is not None and (b < lo_min or a > lo_max):
                continue
            if s0 is not None and np.searchsorted(scans, s1, "right") == np.searchsorted(scans, s0, "left"):
                continue
            groups.append(g)
        if not groups:
            continue
        t = f.read_row_groups(groups, columns=["scan", "mz", "intensity"])
        t = t.filter(pc.is_in(t.column("scan"), value_set=wanted))
        if t.num_rows:
            parts.append(t)
    if not parts:
        return None
    t = pa.concat_tables(parts)  # one unit's points: what it is matched against
    scan = t.column("scan").to_numpy().astype(np.int64)
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    point = np.arange(scan.size, dtype=np.int64) - np.repeat(starts, np.diff(np.r_[starts, scan.size]))
    return scan, t.column("mz").to_numpy(), t.column("intensity").to_numpy(), point


def _ms1_intervals(plan: QueryPlan, settings, axis) -> np.ndarray:
    """Sorted, merged m/z intervals holding every point an isotope channel
    could match or be decided against: the tolerance (at each channel's own
    m/z) and the largest correction the m/z axis makes at MS1."""
    channels = plan.isotopes
    mz = channels.mz
    if mz.size == 0:
        return np.empty((0, 2))
    value, unit = settings.tolerance(1)
    half = mz * value * 1e-6 if unit == "ppm" else np.full(mz.size, float(value))
    shift = 0.0
    if axis is not None:
        _, ys = axis.knots(1)
        shift = float(np.abs(ys).max()) if ys.size else 0.0
    pad = half * (1 + 1e-9) + shift + 1e-9 * np.maximum(mz, 1.0)
    lo, hi = mz - pad, mz + pad
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    reach = np.maximum.accumulate(hi)
    new = np.r_[True, lo[1:] > reach[:-1]]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], lo.size] - 1
    return np.stack([lo[starts], reach[ends]], axis=1)


def _read_ms1(layout: PrecursorMajor, plan: QueryPlan, ledger: Ledger, settings, axis, window):
    """The MS1 points the unit's isotope channels can reach, each scan's in
    stored order: (scan, mz, intensity, stored position)."""
    intervals = _ms1_intervals(plan, settings, axis)
    scans = _eligible_scans(plan, ledger, 1, window)
    if intervals.size == 0 or scans.size == 0:
        return None
    starts, ends = intervals[:, 0], intervals[:, 1]
    wanted = pa.array(scans)
    parts = []
    for path in layout.files(1):
        f = pq.ParquetFile(path)
        groups = []
        for g, (a, b) in enumerate(_row_group_ranges(f, "mz")):
            # A group is read if an interval starts before its last m/z and ends after its first.
            if a is None or np.searchsorted(starts, b, "right") > np.searchsorted(ends, a, "left"):
                groups.append(g)
        if not groups:
            continue
        t = f.read_row_groups(groups, columns=["mz", "intensity", "scan", "point"])
        mz = t.column("mz").to_numpy()
        k = np.searchsorted(starts, mz, side="right") - 1
        inside = (k >= 0) & (mz <= ends[np.maximum(k, 0)])
        t = t.filter(pa.array(inside))
        t = t.filter(pc.is_in(t.column("scan"), value_set=wanted))
        if t.num_rows:
            parts.append(t)
    if not parts:
        return None
    t = pa.concat_tables(parts)
    scan = t.column("scan").to_numpy().astype(np.int64)
    point = t.column("point").to_numpy().astype(np.int64)
    by = np.lexsort((point, scan))
    return scan[by], t.column("mz").to_numpy()[by], t.column("intensity").to_numpy()[by], point[by]


def _blocks(rows, ledger: Ledger, axis, block_rows: int):
    """Blocks of whole scans from a level's points (``_make_block``)."""
    from constellation.massspec.sweep.engine import _make_block

    if rows is None:
        return
    scan, mz, intensity, point = rows
    by = np.argsort(scan, kind="stable")  # each scan's points stay in stored order
    scan, mz, intensity, point = scan[by], mz[by], intensity[by], point[by]
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    edges = np.r_[starts, scan.size]
    numbers = scan[starts]
    at = ledger.rows_of(numbers)
    first = 0
    while first < numbers.size:
        last = int(np.searchsorted(edges, edges[first] + block_rows, side="right")) - 1
        last = min(max(last, first + 1), numbers.size)
        a, b = int(edges[first]), int(edges[last])
        yield _make_block(ledger, at[first:last], numbers[first:last], edges[first:last + 1] - a,
                          mz[a:b], intensity[a:b], point[a:b], axis)  # fmt: skip
        first = last


# ── a unit, in a worker ──────────────────────────────────────────────


def _worker_state(bundle: str, window, settings):
    from constellation.massspec.sweep.engine import _load_axis

    key = (bundle, window, settings.mz_axis)
    if _CACHE.get("key") != key:
        _CACHE.clear()
        b = Path(bundle)
        layout = find_precursor_major(b)
        if layout is None:
            raise ValueError(f"{b}: its precursor-major layout went stale during the sweep")
        ledger = _ledger(b, layout, window)
        _CACHE.update(key=key, layout=layout, ledger=ledger, order=ledger.order(), axis=_load_axis(b, settings))
    return _CACHE["layout"], _CACHE["ledger"], _CACHE["order"], _CACHE["axis"]


def _unit_task(bundle: str, work: str, index: int, unit: QueryPlan, settings, reducers, window) -> dict[str, str]:
    """Match and reduce one unit; write each output as one part, a row
    group per batch. Returns each output's part."""
    from constellation.massspec.sweep.engine import _batch, _buckets, _map_block, _MapOut, _reduce, outputs_of

    layout, ledger, order, axis = _worker_state(bundle, window, settings)
    blocks = []
    for level in unit.levels:
        channels = unit.channels_for_level(level)
        if channels is None or channels.n == 0:
            continue
        rows = (_read_ms1(layout, unit, ledger, settings, axis, window) if level == 1
                else _read_fragment_level(layout, unit, ledger, level, window))  # fmt: skip
        blocks.extend(_blocks(rows, ledger, axis, settings.block_rows))
    every = np.arange(unit.n_queries, dtype=np.int64)
    cells = np.zeros(unit.n_queries, np.int64)
    for level in unit.levels:
        channels = unit.channels_for_level(level)
        if channels is not None:
            cells += eligible_counts(unit, ledger, level, every) * channels.counts()
    schemas = {name: schema for r in reducers for name, schema in outputs_of(r).items()}
    paths = {name: Path(work) / "reduce" / name / f"part-{index:05d}.parquet" for name in schemas}
    writers = {}
    try:
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[name] = pq.ParquetWriter(path.with_name(f".{path.name}.tmp"), schemas[name])
        for q0, q1 in _buckets(cells, settings.batch_cells):
            batch_plan = unit.take(np.arange(q0, q1, dtype=np.int64))
            out = _MapOut.concat([_map_block(block, batch_plan, settings) for block in blocks])
            queries = np.arange(batch_plan.n_queries, dtype=np.int64)
            batch = _batch(batch_plan, queries, out, ledger, order, settings, None)
            for tables in _reduce(batch, reducers):
                for name, table in tables.items():
                    writers[name].write_table(table, row_group_size=max(table.num_rows, 1))
    finally:
        for w in writers.values():
            w.close()
    for name, path in paths.items():
        os.replace(path.with_name(f".{path.name}.tmp"), path)
    return {name: str(p) for name, p in paths.items()}


# ── the parent ───────────────────────────────────────────────────────


def sweep_precursor_major(source, plan: QueryPlan, settings, reducers: Sequence, workers: int,
                          work_dir, progress, window):  # fmt: skip
    """The sweep, read precursor-major (module docstring)."""
    from constellation.massspec.bundle import read_bundle_manifest
    from constellation.massspec.sweep.engine import SweepResult, _Inline, _init_worker, _load_axis, outputs_of

    layout = _refuse(source, plan, settings)
    bundle = Path(source)
    read_bundle_manifest(bundle)
    axis = _load_axis(bundle, settings)
    ledger = _ledger(bundle, layout, window)
    units = precursor_units(plan, ledger, settings.unit_cells)
    if work_dir is None:
        work = Path(tempfile.mkdtemp(prefix="sweep-"))
        temporary = True
    else:
        work, temporary = Path(work_dir), False
        if (work / "reduce").exists():
            shutil.rmtree(work / "reduce")
    try:
        pool = (
            ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker)
            if workers > 1
            else _Inline()
        )
        done: dict[int, dict[str, str]] = {}
        emit_start(progress, "sweep: precursor-major", total=len(units))
        with pool as executor:
            # Units are taken from the plan as they are sent, a few ahead of
            # the workers, so the parent never holds the plan twice.
            pending: dict[Any, int] = {}
            ahead = max(2 * workers, 1)
            next_unit = 0
            while next_unit < len(units) or pending:
                while next_unit < len(units) and len(pending) < ahead:
                    future = executor.submit(_unit_task, str(bundle), str(work), next_unit, plan.take(units[next_unit]),
                                             settings, tuple(reducers), window)  # fmt: skip
                    pending[future] = next_unit
                    next_unit += 1
                finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for future in finished:
                    done[pending.pop(future)] = future.result()
                    emit_progress(progress, "sweep: precursor-major", len(done), len(units))
        emit_done(progress, "sweep: precursor-major", completed=len(done), total=len(units))
    except BaseException:
        if temporary:
            shutil.rmtree(work, ignore_errors=True)
        raise
    result = SweepResult(scans=ledger, matches=None, mz_axis=None if axis is None else axis.key)
    for reducer in reducers:
        for name, schema in outputs_of(reducer).items():
            result.parts[name] = [Path(done[i][name]) for i in range(len(units))]
            result.schemas[name] = schema
            result.metadata[name] = {LAYOUT_KEY: b"precursor_major"}
    if temporary:
        weakref.finalize(result, shutil.rmtree, work, True)
    return result
