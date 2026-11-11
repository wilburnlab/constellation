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
a window bound where one is near. A segment (the queries between two
window bounds) too large for one unit is cut into several, which read the
same points: they are one task, read once and reduced one after another
(halved into more tasks only to give every worker work). What a task reads
is every scan any of its queries is eligible in, so the cuts only decide
how often a window is read, never what a query is matched against:

- at MS2 and above, the scans of each window that holds a query's
  precursor m/z (inclusive, as the scan-major sweep's test), every scan
  without bounds, and for a query with no precursor m/z every scan;
- at MS1, each isotope channel's m/z plus its tolerance and the largest
  correction the m/z axis makes, so every point that could be matched, or
  could be the nearer of two, is there.

A unit's queries are matched and reduced in batches in precursor m/z order
(``SweepSettings.batch_cells``) by the scan-major sweep's own code
(``_map_block``, ``_batch``, ``_reduce``) on a plan taken for the batch
(``QueryPlan.take``), so a query's rows are the rows the scan-major sweep
gives it; each output is one part per unit, its queries in precursor m/z
order (query order, when the targets came in precursor m/z order).
Refused here: claims (a point is claimed by every unit whose queries
matched it, MS1 points by every unit), the gate's MS1 isotope evidence
(it ranks a whole scan's points, of which a unit reads a slice), a table
of points, and a bundle without a current layout.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import weakref
from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core import stages
from constellation.core.progress import emit_done, emit_progress, emit_start
from constellation.massspec.precursor_major import PrecursorMajor, find_precursor_major
from constellation.massspec.sweep.queries import QueryPlan
from constellation.massspec.sweep.reduce import Ledger, eligible_counts

__all__ = ["LAYOUT_KEY", "sweep_precursor_major", "precursor_units"]

#: Where an output says the sweep that made it read precursor-major.
LAYOUT_KEY = b"x.massspec.sweep.layout"

_CACHE: dict[str, Any] = {}
#: What the last units in this process did (for measurement and tests).
_STATS: dict[str, int] = {}


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

    ledger = _scan_metadata(bundle, window)
    counts = layout.scans()
    scan = counts.column("scan").to_numpy().astype(np.int64)
    n = counts.column("n_points").to_numpy().astype(np.int64)
    if window is not None:
        keep = (scan >= window[0]) & (scan < window[1])
        scan, n = scan[keep], n[keep]
    ledger.n_points[ledger.rows_of(scan)] = n
    return ledger


# ── units ─────────────────────────────────────────────────────────────


#: A unit's cell budget is at most the plan's cells over this, so a plan
#: of few cells (RT-windowed targets) is still spread over units (and so
#: workers), never one task; at least ``_MIN_UNIT_CELLS``.
UNIT_SPLIT = 16
_MIN_UNIT_CELLS = 1 << 20
#: The points a unit's windows may hold before it stops at a window bound:
#: a batch's point-side work is its windows' points, so a unit grows by
#: whole windows only while that stays small (one DIA window, or thousands
#: of DDA windows).
UNIT_POINTS = 1 << 22


def precursor_units(plan: QueryPlan, ledger: Ledger, unit_cells: int,
                    unit_points: int = UNIT_POINTS) -> list[np.ndarray]:  # fmt: skip
    """The plan's queries in units, in (precursor m/z, query) order: each
    unit a run of whole window segments (the queries between two
    consecutive window bounds) while its cells stay within the budget
    (``unit_cells``, or the plan's cells over :data:`UNIT_SPLIT` when that
    is less) and its segments' windows' points within ``unit_points``; a
    segment over the budget is cut by cells alone. Queries without a
    precursor m/z come after the rest, cut by cells. The cut depends on
    the plan and the run, never on the workers. A unit's batches keep its
    order, so a batch touches a window or two. Every query is in one."""
    return _units_and_groups(plan, ledger, unit_cells, unit_points)[0]


def _units_and_groups(plan: QueryPlan, ledger: Ledger, unit_cells: int,
                      unit_points: int = UNIT_POINTS) -> tuple[list[np.ndarray], list[list[int]]]:  # fmt: skip
    """:func:`precursor_units`, and the units in groups that read the same
    points: the pieces one segment was cut into by cells (or the queries
    without a precursor m/z were), each group a run of consecutive units.
    A group is read once and its units reduced one after another, so a
    window is read once per task (until 2026-11-10, once per unit: a
    segment cut in ten was read ten times, and units of queries without a
    precursor m/z each read every MS2 point of the run)."""
    n = plan.n_queries
    if n == 0:
        return [], []
    every = np.arange(n, dtype=np.int64)
    cells = np.zeros(n, np.int64)
    for level in plan.levels:
        channels = plan.channels_for_level(level)
        if channels is not None:
            cells += eligible_counts(plan, ledger, level, every) * channels.counts()
    budget = int(min(unit_cells, max(-(-int(cells.sum()) // UNIT_SPLIT), _MIN_UNIT_CELLS)))
    order = plan.precursor_order
    pmz = plan.precursor_mz[order]
    finite = int(np.count_nonzero(~np.isnan(pmz)))
    weight = cells[order] + 1
    frag = next((lv for lv in plan.levels if lv > 1), None)
    segment = np.zeros(finite, np.int64)
    points = np.zeros(finite, np.float64)
    if frag is not None:
        index = ledger.window_index(frag)
        bounded = ~index.unbounded
        lower, upper = index.lower[bounded], index.upper[bounded]
        held = np.add.reduceat(ledger.n_points[index.rows], index.starts[:-1]) if index.rows.size else np.zeros(0)
        held = held[bounded] if held.size == index.n_windows else np.zeros(bounded.sum())
        bounds = np.unique(np.concatenate([lower, upper]))
        segment = np.searchsorted(bounds, pmz[:finite], side="right")
        # Points of the windows holding each query's precursor: those with
        # lower <= pmz, less those with upper < pmz (prefix sums over each).
        by_lower, by_upper = np.argsort(lower), np.argsort(upper)
        cum_lower = np.r_[0.0, np.cumsum(held[by_lower])]
        cum_upper = np.r_[0.0, np.cumsum(held[by_upper])]
        p = pmz[:finite]
        points = (cum_lower[np.searchsorted(lower[by_lower], p, side="right")]
                  - cum_upper[np.searchsorted(upper[by_upper], p, side="left")])  # fmt: skip
    units: list[np.ndarray] = []
    groups: list[list[int]] = []
    # Segments, in m/z order: [starts[k], starts[k + 1]).
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]]) if finite else np.empty(0, np.int64)
    seg_end = np.r_[starts[1:], finite].astype(np.int64)
    seg_cells = np.add.reduceat(weight[:finite], starts) if finite else np.empty(0)
    unit_start, unit_cells_held, unit_points_held = 0, 0, 0.0

    def close(at: int) -> None:
        nonlocal unit_start, unit_cells_held, unit_points_held
        if at > unit_start:
            groups.append([len(units)])
            units.append(order[unit_start:at])
        unit_start, unit_cells_held, unit_points_held = at, 0, 0.0

    def cut(lo: int, hi: int) -> None:
        first = len(units)
        _cut_by_cells(order, weight, lo, hi, budget, units)
        if len(units) > first:
            groups.append(list(range(first, len(units))))

    for k in range(starts.size):
        a, b = int(starts[k]), int(seg_end[k])
        c, pts = int(seg_cells[k]), float(points[a])
        if unit_start < a and (unit_cells_held + c > budget or unit_points_held + pts > unit_points):
            close(a)
        if c > budget:  # a segment of its own, cut by cells
            close(a)
            cut(a, b)
            unit_start = b
            continue
        unit_cells_held += c
        unit_points_held += pts
    close(finite)
    cut(finite, n)
    return units, groups


def _cut_by_cells(order, weight, lo: int, hi: int, budget: int, units: list) -> None:
    cum = np.cumsum(weight[lo:hi])
    start = 0
    while start < hi - lo:
        base = cum[start - 1] if start else 0
        end = min(max(int(np.searchsorted(cum, base + budget, side="right")), start + 1), hi - lo)
        units.append(order[lo + start:lo + end])
        start = end


def _tasks(groups: list[list[int]], workers: int) -> list[list[int]]:
    """Groups of units as tasks, in order. With fewer tasks than about two
    per worker, the largest groups are halved until there are enough (or
    none has two units): reading a window twice is cheaper than leaving
    workers idle."""
    tasks = [list(g) for g in groups]
    if workers <= 1:
        return tasks
    while len(tasks) < 2 * workers:
        i = max(range(len(tasks)), key=lambda t: len(tasks[t]), default=None)
        if i is None or len(tasks[i]) < 2:
            break
        g = tasks[i]
        tasks[i:i + 1] = [g[: len(g) // 2], g[len(g) // 2:]]
    return tasks


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
    """Scan numbers of ``level`` the unit can be matched in: slots (with
    points, or read and empty) with a time, in the range, inside the span
    of its queries' intervals, and (when given) in one of
    ``scans_of_windows`` (ledger rows)."""
    has = plan.channels_for_level(level).counts() > 0
    rows = np.flatnonzero((ledger.level == level) & ledger.present & ~np.isnan(ledger.rt)) \
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
    _STATS["fragment_reads"] = _STATS.get("fragment_reads", 0) + 1
    # The bounded windows' range of lower bounds, and apart from it the
    # scans without bounds, stored with lower = -inf (first in the layout):
    # with them in the range, its minimum was -inf and every row group
    # below the unit's windows was decoded only to be dropped (9.8× the
    # rows on a test; until 2026-11-10).
    bounded = windows[~index.unbounded[windows]]
    lo_min = float(index.lower[bounded].min()) if bounded.size else np.inf
    lo_max = float(index.lower[bounded].max()) if bounded.size else -np.inf
    wanted = pa.array(scans)
    parts = []
    for path in layout.files(level):
        f = pq.ParquetFile(path)
        lower_ranges = _row_group_ranges(f, "lower")
        scan_ranges = _row_group_ranges(f, "scan")
        groups = []
        for g, ((a, b), (s0, s1)) in enumerate(zip(lower_ranges, scan_ranges, strict=True)):
            if a is not None and a != -np.inf and (b < lo_min or a > lo_max):
                continue
            if s0 is not None and np.searchsorted(scans, s1, "right") == np.searchsorted(scans, s0, "left"):
                continue
            groups.append(g)
        if not groups:
            continue
        t = f.read_row_groups(groups, columns=["scan", "mz", "intensity"])
        _STATS["fragment_rows_decoded"] = _STATS.get("fragment_rows_decoded", 0) + t.num_rows
        t = t.filter(pc.is_in(t.column("scan"), value_set=wanted))
        if t.num_rows:
            parts.append(t)
    if not parts:
        return None
    t = pa.concat_tables(parts)  # one unit's points: what it is matched against
    scan = t.column("scan").to_numpy().astype(np.int64)
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    point = np.arange(scan.size, dtype=np.int64) - np.repeat(starts, np.diff(np.r_[starts, scan.size]))
    intensity = t.column("intensity").to_numpy()
    # A fragment scan is read whole: its lowest point is its floor, where
    # the bundle records none (MS1 is read in m/z slices, so has none here).
    ledger.fill_floor(ledger.rows_of(scan[starts]), np.minimum.reduceat(intensity.astype(np.float64), starts))
    return scan, t.column("mz").to_numpy(), intensity, point


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


def _blocks(rows, ledger: Ledger, axis, block_rows: int, rt_span: float | None = None):
    """Blocks of whole scans from a level's points (``_make_block``), of at
    most ``block_rows`` points and, with ``rt_span``, at most that many
    seconds: a unit holds one window over the whole gradient, and a block
    spanning much more than its queries' retention-time windows matches
    every query in every scan of it, only for the matches to be dropped as
    ineligible (the scan-major sweep's blocks are a few minutes of a run)."""
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
    rt = ledger.rt[at]
    first = 0
    while first < numbers.size:
        by_rows = int(np.searchsorted(edges, edges[first] + block_rows, side="right")) - 1
        last = by_rows
        if rt_span is not None and np.isfinite(rt[first]):
            # Scans are in scan order, which is time order within a run.
            last = min(last, int(np.searchsorted(rt, rt[first] + rt_span, side="right")))
            # At least MIN_BLOCK_POINTS points (within block_rows): a unit
            # reads a window or two, whose scans are a cycle apart, so a floor
            # of one second held one scan a block, and narrow windows ran 8×
            # slower than scan-major on the cost of each block (until
            # 2026-11-10). A floor in points, not scans, is the same work per
            # block whatever a scan holds.
            enough = int(np.searchsorted(edges, edges[first] + MIN_BLOCK_POINTS, side="left"))
            last = max(last, min(enough, by_rows))
        last = min(max(last, first + 1), numbers.size)
        a, b = int(edges[first]), int(edges[last])
        yield _make_block(ledger, at[first:last], numbers[first:last], edges[first:last + 1] - a,
                          mz[a:b], intensity[a:b], point[a:b], axis)  # fmt: skip
        first = last


#: The fewest points a block of a unit holds when its span is cut by time
#: (``_blocks``): about 64 scans of a Stellar run's MS2, where a block's
#: own cost is small beside its matching.
MIN_BLOCK_POINTS = 1 << 16


def _block_span(plan: QueryPlan) -> float | None:
    """Seconds a block may span: half the unit's typical retention-time
    window, when its queries have finite ones (none otherwise); ``_blocks``
    widens a block to :data:`MIN_BLOCK_POINTS`."""
    width = plan.rt_hi - plan.rt_lo
    finite = width[np.isfinite(width)]
    if finite.size < max(1, plan.n_queries // 2):
        return None
    return float(np.median(finite)) / 2.0


# ── a unit, in a worker ──────────────────────────────────────────────


def _worker_state(bundle: str, window, settings, sweep_id: str, layout: PrecursorMajor):
    """What every unit of one sweep shares, read once per process: the
    ledger, its order and the m/z axis. Kept for one sweep only (its id),
    with the layout and the curve the parent resolved: until 2026-11-09 it
    was kept by bundle, range and the literal ``mz_axis``, so a later sweep
    in the same process with ``latest`` after a new curve was saved used
    the old curve and named the new one (review by Codex)."""
    from constellation.massspec.sweep.engine import _load_axis

    if _CACHE.get("sweep") != sweep_id:
        _CACHE.clear()
        b = Path(bundle)
        # The parent checked the layout against the peaks; a layout rebuilt
        # since then has another key in its marker.
        if not stages.is_done(layout.root, layout.key):
            raise ValueError(f"{b}: its precursor-major layout changed during the sweep")
        ledger = _ledger(b, layout, window)
        _CACHE.update(sweep=sweep_id, ledger=ledger, order=ledger.order(), axis=_load_axis(b, settings))
    return _CACHE["ledger"], _CACHE["order"], _CACHE["axis"]


def _read_blocks(layout: PrecursorMajor, plan: QueryPlan, ledger: Ledger, settings, axis, window,
                 levels=None) -> list:  # fmt: skip
    """The blocks of every point ``plan``'s queries can be matched against,
    at each of its levels (or of ``levels``)."""
    blocks = []
    for level in plan.levels if levels is None else levels:
        channels = plan.channels_for_level(level)
        if channels is None or channels.n == 0:
            continue
        rows = (_read_ms1(layout, plan, ledger, settings, axis, window) if level == 1
                else _read_fragment_level(layout, plan, ledger, level, window))  # fmt: skip
        blocks.extend(_blocks(rows, ledger, axis, settings.block_rows, _block_span(plan)))
    return blocks


def _group_task(bundle: str, work: str, first: int, group: QueryPlan, cuts: list[tuple[int, int]], settings,
                reducers, window, sweep_id: str, layout: PrecursorMajor) -> list[dict[str, str]]:  # fmt: skip
    """Read once what a group of units can be matched against at MS2 and
    above (``group``, the plan of all their queries: one window segment, so
    the same scans for each), then match and reduce each unit (``group``'s
    queries ``cuts[k]``, unit ``first + k``) on it and on its own MS1 read.
    A unit is matched against the group's points, a superset of its own,
    which gives the same rows: a match outside a query's interval or
    windows is not eligible. MS1 is read unit by unit because it is read
    by m/z, each unit its own slices: an MS1-only plan is one segment, and
    read for the group its 40 units' slices were 40 times the points each
    unit needed (Codex, 2026-11-10). Returns each unit's parts."""
    ledger, order, axis = _worker_state(bundle, window, settings, sweep_id, layout)
    shared = _read_blocks(layout, group, ledger, settings, axis, window, levels=[v for v in group.levels if v > 1])
    out = []
    for k, (a, b) in enumerate(cuts):
        unit = group.take(np.arange(a, b, dtype=np.int64))
        own = _read_blocks(layout, unit, ledger, settings, axis, window, levels=[v for v in unit.levels if v == 1])
        _STATS["ms1_points_read"] = max(_STATS.get("ms1_points_read", 0), sum(len(blk.mz) for blk in own))
        out.append(_unit_task(work, first + k, unit, shared + own, ledger, order, settings, reducers))
        del own
    return out


def _unit_task(work: str, index: int, unit: QueryPlan, blocks: list, ledger: Ledger, order, settings,
               reducers) -> dict[str, str]:  # fmt: skip
    """Match and reduce one unit on ``blocks``; write each output as one
    part, a row group per batch. Returns each output's part."""
    from constellation.massspec.sweep.engine import _batch, _buckets, _map_block, _MapOut, _reduce, outputs_of

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
        gate_first = settings.gate is not None and settings.gate_first and all(r.gated for r in reducers)
        for q0, q1 in _buckets(cells, settings.batch_cells):
            batch_plan = unit.take(np.arange(q0, q1, dtype=np.int64))
            queries = np.arange(batch_plan.n_queries, dtype=np.int64)
            if gate_first:
                # Every reducer sees only the queries the gate passes: decide
                # it from the evidence, then match those queries alone.
                ev = _MapOut.concat([_map_block(block, batch_plan, settings, matches=False) for block in blocks])
                decided = _batch(batch_plan, queries, ev, ledger, order, settings, None)
                passing = np.zeros(batch_plan.n_queries, dtype=bool)
                passing[queries[decided.decision.passed]] = True
                _STATS["gate_first_matched"] = _STATS.get("gate_first_matched", 0) + int(passing.sum())
                found = _MapOut.concat([_map_block(block, batch_plan, settings, evidence=False, only=passing)
                                        for block in blocks]) if passing.any() else _MapOut.concat([])  # fmt: skip
                out = _MapOut(found.matches, ev.evidence, ev.claims)
            else:
                out = _MapOut.concat([_map_block(block, batch_plan, settings) for block in blocks])
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
                          work_dir, progress, window, *, spill_dir=None):  # fmt: skip
    """The sweep, read precursor-major (module docstring)."""
    from constellation.massspec.bundle import read_bundle_manifest
    from constellation.massspec.sweep.engine import (
        SweepResult,
        _Inline,
        _init_worker,
        _load_axis,
        stop_pool,
        outputs_of,
        own_dir,
    )

    layout = _refuse(source, plan, settings)
    bundle = Path(source)
    read_bundle_manifest(bundle)
    axis = _load_axis(bundle, settings)
    # Workers use the curve resolved here (``latest`` named once), and keep
    # what they read for this sweep only.
    settings = replace(settings, mz_axis=None if axis is None else axis.key)
    sweep_id = uuid.uuid4().hex
    ledger = _ledger(bundle, layout, window)
    units, groups = _units_and_groups(plan, ledger, settings.unit_cells)
    tasks = _tasks(groups, workers)
    if work_dir is None:
        # Only the reducers' outputs are written, so no room is estimated.
        where = Path(spill_dir) if spill_dir is not None else Path(tempfile.gettempdir())
        where.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="sweep-", dir=where))
        temporary = True
    else:
        # A directory of this sweep's own: until 2026-11-10 a later sweep
        # cleared ``work_dir/reduce`` and wrote its parts where an earlier
        # result read its own.
        work, temporary = own_dir(work_dir), False
    try:
        pool = (
            ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker)
            if workers > 1
            else _Inline()
        )
        done: dict[int, dict[str, str]] = {}
        emit_start(progress, "sweep: precursor-major", total=len(units))
        with pool as executor:
            try:
                # Tasks are taken from the plan as they are sent, a few ahead of
                # the workers, so the parent never holds the plan twice.
                pending: dict[Any, list[int]] = {}
                ahead = max(2 * workers, 1)
                next_task = 0
                while next_task < len(tasks) or pending:
                    while next_task < len(tasks) and len(pending) < ahead:
                        members = tasks[next_task]
                        sizes = np.array([units[i].size for i in members], np.int64)
                        edges = np.r_[0, np.cumsum(sizes)]
                        group = plan.take(np.concatenate([units[i] for i in members]))
                        future = executor.submit(_group_task, str(bundle), str(work), members[0], group,
                                                 list(zip(edges[:-1].tolist(), edges[1:].tolist())), settings,
                                                 tuple(reducers), window, sweep_id, layout)  # fmt: skip
                        pending[future] = members
                        next_task += 1
                    finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for future in finished:
                        for i, parts in zip(pending.pop(future), future.result(), strict=True):
                            done[i] = parts
                        emit_progress(progress, "sweep: precursor-major", len(done), len(units))
            except BaseException:
                stop_pool(executor)
                raise
        emit_done(progress, "sweep: precursor-major", completed=len(done), total=len(units))
    except BaseException:
        # Nothing will read a failed sweep's directory, temporary or its own
        # ``sweep-<id>/`` (kept until 2026-11-11; review E12).
        shutil.rmtree(work, ignore_errors=True)
        raise
    result = SweepResult(scans=ledger, matches=None, mz_axis=None if axis is None else axis.key, work=work)
    for reducer in reducers:
        for name, schema in outputs_of(reducer).items():
            result.parts[name] = [Path(done[i][name]) for i in range(len(units))]
            result.schemas[name] = schema
            result.metadata[name] = {LAYOUT_KEY: b"precursor_major"}
    if temporary:
        weakref.finalize(result, shutil.rmtree, work, True)
    return result
