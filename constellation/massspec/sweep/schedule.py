"""Which queries a scan is matched against.

Per block of scans, the active queries are those whose retention-time
interval overlaps the block and that have channels at the level. Within
the block, scans are grouped by isolation window (a window group): every
scan of a group is matched against one set of queries, the active ones
whose precursor m/z the window covers, and each (scan, query) pair is then
held to its own scan's retention time (and, for fragments, the assigned
scan) after matching; a pair's verdict holds for each of its cells.

The isolation test is the scan-major extractor's, inclusive at both ends
on the scan's own float64 bounds: ``lower <= precursor_mz <= upper``,
computed as two ``searchsorted`` calls over the active queries sorted by
precursor m/z. A scan without bounds is matched against every active
query, and so is a query without a precursor m/z. Pure numpy, shared by
the map (to match) and the reduce (to know which cells a match could have
filled, for zero-filling). Terms as in ``docs/plans/sweep.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ragged_arange",
    "active_queries",
    "ScanGroup",
    "isolation_groups",
    "pair_eligible",
    "WindowIndex",
    "window_index",
    "eligible_pairs",
]


def ragged_arange(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``starts[i], ..., starts[i] + counts[i] - 1`` for each ``i``, concatenated."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(counts)
    return np.arange(total, dtype=np.int64) + np.repeat(
        np.asarray(starts, dtype=np.int64) - (ends - counts), counts
    )


def order_by(*keys: np.ndarray) -> np.ndarray:
    """The stable order that sorts rows by ``keys``, the first the most
    significant: ``np.lexsort(keys[::-1])``, row for row. Integer keys are
    combined in mixed radix over the ranges they hold into one int64 when
    that fits in 62 bits, which sorts about six times faster than a
    lexsort of five keys (the match sort cost more than matching did on a
    real run, 2026-11-09); otherwise it is the lexsort."""
    keys = tuple(np.asarray(k) for k in keys)
    n = keys[0].size if keys else 0
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if any(k.dtype.kind not in "iub" for k in keys):
        return np.lexsort(keys[::-1])
    lows = [int(k.min()) for k in keys]
    bases = [int(k.max()) - lo + 1 for k, lo in zip(keys, lows, strict=True)]
    if float(np.prod([float(b) for b in bases])) >= 2.0**62:
        return np.lexsort(keys[::-1])
    combined = np.zeros(n, dtype=np.int64)
    for k, lo, base in zip(keys, lows, bases, strict=True):
        combined *= base
        combined += k.astype(np.int64) - lo
    return np.argsort(combined, kind="stable")


def active_queries(
    rt_lo: np.ndarray, rt_hi: np.ndarray, has_channels: np.ndarray, t0: float, t1: float
) -> np.ndarray:
    """Queries whose inclusive interval overlaps ``[t0, t1]`` and that have
    channels at the level, in query order."""
    return np.flatnonzero((rt_lo <= t1) & (rt_hi >= t0) & has_channels)


@dataclass(frozen=True)
class ScanGroup:
    """A window group: scans of a block matched against one set of queries."""

    rows: np.ndarray  # the scans' rows in the block, in stored order
    queries: np.ndarray  # the queries, in query order


def isolation_groups(
    active: np.ndarray,
    precursor_mz: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    *,
    gate: bool,
    order: np.ndarray | None = None,
) -> list[ScanGroup]:
    """Group a level's scans (``rows`` of the block, with isolation bounds
    ``lower`` / ``upper``) by window, each with the active queries the window
    covers. With ``gate`` false (MS1) every scan is in one group with every
    active query. ``order`` is every query by precursor m/z (a stable sort,
    NaN last: ``QueryPlan.precursor_order``), so the active ones need no
    sort of their own; until 2026-11-09 they were sorted at every block,
    about 2 s a block at 9.56M queries."""
    if rows.size == 0 or active.size == 0:
        return []
    if not gate:
        return [ScanGroup(rows=rows, queries=active)]
    if order is None:
        order = np.argsort(precursor_mz, kind="stable")
    is_active = np.zeros(precursor_mz.size, dtype=bool)
    is_active[active] = True
    by_mz = order[is_active[order]]
    pmz_sorted = precursor_mz[by_mz]
    n_finite = int(np.count_nonzero(~np.isnan(pmz_sorted)))
    gated, gated_pmz = by_mz[:n_finite], pmz_sorted[:n_finite]
    passall = np.sort(by_mz[n_finite:])
    unbounded = np.isnan(lower) | np.isnan(upper)
    # Windows compared exactly: group scans whose float64 bounds are equal.
    key_lo = np.where(unbounded, -np.inf, lower)
    key_hi = np.where(unbounded, -np.inf, upper)
    pairs, inverse = np.unique(np.stack([key_lo, key_hi], axis=1), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    # Each window's members in stored order, by one stable sort (selecting
    # them window by window cost windows × scans: quadratic on DDA, where
    # nearly every scan is a window of its own).
    by_window = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[by_window], np.arange(len(pairs) + 1))
    groups = []
    for g, (lo, hi) in enumerate(pairs):
        members = rows[by_window[bounds[g]:bounds[g + 1]]]
        if lo == -np.inf and hi == -np.inf:
            queries = active
        else:
            a = np.searchsorted(gated_pmz, lo, side="left")
            b = np.searchsorted(gated_pmz, hi, side="right")
            queries = np.sort(np.concatenate((gated[a:b], passall)))
        if queries.size:
            groups.append(ScanGroup(rows=members, queries=queries))
    return groups


def pair_eligible(
    query: np.ndarray,
    scan_rt: np.ndarray,
    scan_id: np.ndarray,
    rt_lo: np.ndarray,
    rt_hi: np.ndarray,
    assigned_scan: np.ndarray | None,
) -> np.ndarray:
    """Whether each (scan, query) pair may be matched: the scan's retention
    time inside the query's inclusive interval and, when ``assigned_scan``
    is given (fragments), the scan the query is assigned to if it has one."""
    ok = (rt_lo[query] <= scan_rt) & (scan_rt <= rt_hi[query])
    if assigned_scan is not None:
        assigned = assigned_scan[query]
        ok &= (assigned < 0) | (assigned == scan_id)
    return ok


@dataclass(frozen=True)
class WindowIndex:
    """A level's scans that can be slots (with points, or read and empty,
    and a retention time), by isolation window: what eligibility is decided on, built once
    per ledger and level instead of regrouped at every call.

    ``rows`` are ledger rows sorted by (window, retention time, scan), the
    rows of window ``w`` being ``rows[starts[w]:starts[w + 1]]``. Windows
    are compared exactly on their float64 bounds; a scan without bounds is
    in an unbounded window (as is every MS1 scan, ``gate=False``)."""

    rows: np.ndarray  # int64 ledger rows, sorted
    window: np.ndarray  # int64, each sorted row's window
    rt: np.ndarray  # float64, each sorted row's retention time
    starts: np.ndarray  # (n_windows + 1,) offsets into rows
    lower: np.ndarray  # per window
    upper: np.ndarray
    unbounded: np.ndarray  # per window
    position: np.ndarray  # per ledger row, its place in rows; -1 if not in it

    @property
    def n_windows(self) -> int:
        return int(self.starts.size - 1)


def window_index(
    scan: np.ndarray, level_of: np.ndarray, rt: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    present: np.ndarray, level: int, *, gate: bool,
) -> WindowIndex:
    """The :class:`WindowIndex` of ``level``'s scans (ledger columns) that
    can be slots (``present``, ``Ledger.present``)."""
    rows = np.flatnonzero((level_of == level) & present & ~np.isnan(rt)).astype(np.int64)
    if gate:
        unbounded_row = np.isnan(lower[rows]) | np.isnan(upper[rows])
        key = np.stack([np.where(unbounded_row, -np.inf, lower[rows]),
                        np.where(unbounded_row, -np.inf, upper[rows])], axis=1)  # fmt: skip
        if rows.size:
            pairs, inverse = np.unique(key, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
        else:
            pairs, inverse = np.empty((0, 2)), np.empty(0, np.int64)
    else:
        pairs = np.array([[-np.inf, -np.inf]]) if rows.size else np.empty((0, 2))
        inverse = np.zeros(rows.size, np.int64)
    order = np.lexsort((scan[rows], rt[rows], inverse))
    rows, window = rows[order], inverse[order].astype(np.int64)
    starts = np.searchsorted(window, np.arange(len(pairs) + 1)).astype(np.int64)
    position = np.full(scan.size, -1, np.int64)
    position[rows] = np.arange(rows.size, dtype=np.int64)
    unbounded = (pairs[:, 0] == -np.inf) & (pairs[:, 1] == -np.inf) if len(pairs) else np.zeros(0, bool)
    return WindowIndex(rows=rows, window=window, rt=rt[rows].astype(np.float64), starts=starts,
                       lower=pairs[:, 0].astype(np.float64), upper=pairs[:, 1].astype(np.float64),
                       unbounded=unbounded, position=position)  # fmt: skip


def eligible_pairs(
    index: WindowIndex,
    queries: np.ndarray,
    precursor_mz: np.ndarray,
    rt_lo: np.ndarray,
    rt_hi: np.ndarray,
    assigned_scan: np.ndarray | None,
    *,
    gate: bool,
    scan_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (window, query) pair of ``queries`` and, for each, the first
    of the window's ``index.rows`` inside the query's interval and how
    many are: ``(query, start, count)``, ``start`` an offset into
    ``index.rows``. The interval test is :func:`pair_eligible`'s,
    inclusive at both ends. A window holds the queries whose precursor
    m/z it covers, with those that have none; an unbounded window (and
    every scan when not ``gate``) holds every query. With
    ``assigned_scan`` (fragments), a query with one has that scan's row or
    none: ``scan_rows`` gives each query's assigned scan as a ledger row
    (-1 where it has none or it is not in the ledger). Vectorized: no work
    per window or per query in Python."""
    empty = np.empty(0, np.int64)
    queries = np.asarray(queries, dtype=np.int64)
    if queries.size == 0 or index.n_windows == 0:
        return empty, empty, empty
    if not gate:
        pair_w = np.zeros(queries.size, np.int64)
        pair_q = queries
    else:
        pmz = precursor_mz[queries]
        finite = ~np.isnan(pmz)
        by_mz = np.argsort(pmz[finite], kind="stable")
        gated, gated_pmz = queries[finite][by_mz], pmz[finite][by_mz]
        passall = queries[~finite]
        bounded = np.flatnonzero(~index.unbounded)
        a = np.searchsorted(gated_pmz, index.lower[bounded], side="left")
        b = np.searchsorted(gated_pmz, index.upper[bounded], side="right")
        n = np.maximum(b - a, 0)
        free = np.flatnonzero(index.unbounded)
        parts_w = [np.repeat(bounded, n), np.repeat(bounded, passall.size), np.repeat(free, queries.size)]
        parts_q = [gated[ragged_arange(a, n)], np.tile(passall, bounded.size), np.tile(queries, free.size)]
        pair_w = np.concatenate(parts_w).astype(np.int64)
        pair_q = np.concatenate(parts_q).astype(np.int64)
    # Only windows whose scans' times overlap the query's interval can
    # hold a pair; the rest are dropped before any search.
    first, last = index.starts[pair_w], index.starts[pair_w + 1] - 1
    keep = (last >= first) & (rt_lo[pair_q] <= index.rt[np.maximum(last, 0)]) & (
        rt_hi[pair_q] >= index.rt[np.minimum(first, max(index.rt.size - 1, 0))]
    )
    pair_w, pair_q = pair_w[keep], pair_q[keep]
    # Exact integer keys: a row's time as its rank among the distinct
    # times, after its window, so one search over all windows finds each
    # pair's range with the float comparisons of pair_eligible, inclusive
    # at both ends.
    times = np.unique(index.rt)
    span = np.int64(times.size + 1)
    keys = index.window * span + np.searchsorted(times, index.rt, side="left")
    lo = pair_w * span + np.searchsorted(times, rt_lo[pair_q], side="left")
    hi = pair_w * span + np.searchsorted(times, rt_hi[pair_q], side="right")
    start = np.searchsorted(keys, lo, side="left").astype(np.int64)
    stop = np.searchsorted(keys, hi, side="left").astype(np.int64)
    count = np.maximum(stop - start, 0)
    if assigned_scan is not None and scan_rows is not None:
        fixed = np.flatnonzero(assigned_scan[pair_q] >= 0)
        if fixed.size:
            row = scan_rows[pair_q[fixed]]
            at = np.where(row >= 0, index.position[np.maximum(row, 0)], -1)
            ok = (at >= 0) & (index.window[np.maximum(at, 0)] == pair_w[fixed]) & (start[fixed] <= at) & (at < stop[fixed])
            start[fixed] = np.maximum(at, 0)
            count[fixed] = ok.astype(np.int64)
    return pair_q, start, count
