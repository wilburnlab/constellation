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
) -> list[ScanGroup]:
    """Group a level's scans (``rows`` of the block, with isolation bounds
    ``lower`` / ``upper``) by window, each with the active queries the window
    covers. With ``gate`` false (MS1) every scan is in one group with every
    active query."""
    if rows.size == 0 or active.size == 0:
        return []
    if not gate:
        return [ScanGroup(rows=rows, queries=active)]
    pmz = precursor_mz[active]
    finite = ~np.isnan(pmz)
    order = np.argsort(pmz[finite], kind="stable")
    gated = active[finite][order]
    gated_pmz = pmz[finite][order]
    passall = active[~finite]
    unbounded = np.isnan(lower) | np.isnan(upper)
    # Windows compared exactly: group scans whose float64 bounds are equal.
    key_lo = np.where(unbounded, -np.inf, lower)
    key_hi = np.where(unbounded, -np.inf, upper)
    pairs, inverse = np.unique(np.stack([key_lo, key_hi], axis=1), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    groups = []
    for g, (lo, hi) in enumerate(pairs):
        members = rows[inverse == g]
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
