"""After the sweep: a bucket of queries with everything matched for them.

The sweep's map reads scans and writes matches; once it is done, each
query's matches are complete, whichever scans (and workers) they came
from, and a :class:`Reducer` turns a :class:`QueryBatch` of them into one
table or several. This is the hook the scorer plugs into. What a reducer
reads: :meth:`QueryBatch.cells`, every cell a query could have matched in
(its channels in each scan it was eligible for) with what matched there,
each channel's identity and abundance, and each scan's time, TIC and
injection time; :meth:`QueryBatch.eligible` for the (scan, query) pairs
alone, from the same schedule the map used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pyarrow as pa

from constellation.massspec.sweep.evidence import Decision, Evidence
from constellation.massspec.sweep.queries import Channels, QueryPlan
from constellation.massspec.sweep.schedule import (
    WindowIndex,
    eligible_pairs,
    pair_eligible,
    ragged_arange,
    window_index,
)
from constellation.massspec.sweep.schemas import (
    SWEEP_CELL_TABLE,
    SWEEP_MATCH_TABLE,
    SWEEP_QUERY_SUMMARY_TABLE,
    SWEEP_SCAN_TABLE,
)

if TYPE_CHECKING:
    from constellation.massspec.sweep.engine import SweepSettings

__all__ = ["Cells", "Ledger", "Matches", "QueryBatch", "Reducer", "QuerySummaryReducer"]


@dataclass(frozen=True)
class Ledger:
    """The scans the sweep read, in stored order, those without points included."""

    scan: np.ndarray  # int64
    level: np.ndarray  # int8
    rt: np.ndarray  # float64
    lower: np.ndarray  # float64, NaN without bounds
    upper: np.ndarray
    n_points: np.ndarray  # int64
    #: Each scan's total ion current and ion injection time (ms), NaN where
    #: the bundle records none (and for a table of points).
    tic: np.ndarray | None = None
    iit: np.ndarray | None = None
    #: Built on first use: each level's WindowIndex (eligibility is
    #: decided on it at every call, so it is built once, not regrouped).
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("tic", "iit"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(self.scan.size, np.nan))

    @property
    def n(self) -> int:
        return int(self.scan.size)

    def window_index(self, level: int) -> WindowIndex:
        """The level's scans with points and a time, by isolation window
        (MS1 as one unbounded window). Keyed by the points the map has
        recorded, so an index is never used before they are all in."""
        key = ("windows", int(level), int(self.n_points.sum()), self.n)
        found = self._cache.get(key)
        if found is None:
            found = window_index(self.scan, self.level, self.rt, self.lower, self.upper, self.n_points,
                                 int(level), gate=level > 1)  # fmt: skip
            self._cache[key] = found
        return found

    def scan_rows(self, scans: np.ndarray) -> np.ndarray:
        """Ledger row of each scan number, -1 for one not in the ledger or
        negative (no scan)."""
        scans = np.asarray(scans, dtype=np.int64)
        by = np.argsort(self.scan, kind="stable")
        pos = np.searchsorted(self.scan[by], scans)
        at = by[np.minimum(pos, max(by.size - 1, 0))] if by.size else np.zeros(scans.size, np.int64)
        found = (scans >= 0) & (pos < by.size) & (self.scan[at] == scans) if by.size else np.zeros(scans.size, bool)
        return np.where(found, at, -1).astype(np.int64)

    def order(self) -> np.ndarray:
        """Each scan's rank by (retention time, scan number): the order a
        query's rows come in."""
        rank = np.empty(self.n, dtype=np.int64)
        rank[np.lexsort((self.scan, self.rt))] = np.arange(self.n, dtype=np.int64)
        return rank

    def rows_of(self, scans: np.ndarray) -> np.ndarray:
        """Ledger rows of scan numbers (all of which must be in it)."""
        by = np.argsort(self.scan, kind="stable")
        return by[np.searchsorted(self.scan[by], scans)]

    def table(self) -> pa.Table:
        return pa.table(
            {
                "scan": pa.array(self.scan.astype(np.int32)),
                "level": pa.array(self.level.astype(np.int8)),
                "rt": pa.array(self.rt),
                "isolation_lower": pa.array(self.lower, mask=np.isnan(self.lower)),
                "isolation_upper": pa.array(self.upper, mask=np.isnan(self.upper)),
                "n_points": pa.array(self.n_points.astype(np.int32)),
                "tic": pa.array(self.tic, mask=np.isnan(self.tic)),
                "iit": pa.array(self.iit, mask=np.isnan(self.iit)),
            },
            schema=SWEEP_SCAN_TABLE,
        )


_MATCH_FIELDS = ("query", "level", "scan", "channel", "point_index", "mz", "intensity", "n_points")


@dataclass(frozen=True)
class Matches:
    """Matched cells, one row per (cell, point). ``channel`` is the channel's
    position among its query's channels of the level's kind; ``point_index``
    the point's position in its scan's stored order, -1 when a rule
    aggregated several."""

    query: np.ndarray  # int64
    level: np.ndarray  # int8
    scan: np.ndarray  # int64
    channel: np.ndarray  # int64
    point_index: np.ndarray  # int64
    mz: np.ndarray  # float64
    intensity: np.ndarray  # float64
    n_points: np.ndarray  # int64

    @property
    def n(self) -> int:
        return int(self.query.size)

    @classmethod
    def empty(cls) -> Matches:
        i, f = np.empty(0, np.int64), np.empty(0, np.float64)
        return cls(i, np.empty(0, np.int8), i, i, i, f, f, i)

    @classmethod
    def concat(cls, parts: list[Matches]) -> Matches:
        parts = [p for p in parts if p.n]
        if not parts:
            return cls.empty()
        if len(parts) == 1:
            return parts[0]
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in _MATCH_FIELDS))

    def take(self, idx: np.ndarray) -> Matches:
        return Matches(*(getattr(self, f)[idx] for f in _MATCH_FIELDS))

    def sorted(self, scan_order: np.ndarray) -> Matches:
        """Sorted by (query, level, scan order, channel, point); ``scan_order``
        is each match's scan's rank."""
        by = np.lexsort((self.point_index, self.channel, scan_order, self.level, self.query))
        return self.take(by)

    def table(self) -> pa.Table:
        return pa.table(
            {
                "query": pa.array(self.query.astype(np.int32)),
                "level": pa.array(self.level.astype(np.int8)),
                "scan": pa.array(self.scan.astype(np.int32)),
                "channel": pa.array(self.channel.astype(np.int32)),
                "point_index": pa.array(self.point_index.astype(np.int32), mask=self.point_index < 0),
                "mz_observed": pa.array(self.mz),
                "intensity": pa.array(self.intensity),
                "n_points": pa.array(self.n_points.astype(np.int32)),
            },
            schema=SWEEP_MATCH_TABLE,
        )

    @classmethod
    def from_table(cls, t: pa.Table) -> Matches:
        def col(name, dtype, fill=None):
            c = t.column(name)
            if fill is not None:
                c = c.fill_null(fill)
            return c.to_numpy().astype(dtype)

        return cls(
            query=col("query", np.int64),
            level=col("level", np.int8),
            scan=col("scan", np.int64),
            channel=col("channel", np.int64),
            point_index=col("point_index", np.int64, -1),
            mz=col("mz_observed", np.float64),
            intensity=col("intensity", np.float64),
            n_points=col("n_points", np.int64),
        )


def _eligible_ranges(plan: QueryPlan, s: Ledger, level: int, queries: np.ndarray):
    """The (scan, query) pairs ``level`` could match ``queries`` in, as
    ranges: the level's ledger rows by window and (retention time, scan)
    (:meth:`Ledger.window_index`), and for each (window, query) pair its
    query, the first of those rows inside its interval and how many are.
    A query with an assigned scan (fragments) has that one row or none.
    So the pairs are never built as a queries × scans product: what is
    built is only what is eligible, and counting needs nothing built at
    all. Until 2026-11-07 the windows were regrouped at every call, which
    cost windows × scans (5.4 s a call on a 100,000-scan DDA run).

    The interval test is :func:`pair_eligible`'s, inclusive at both ends
    (bounds are never NaN; a scan without a time is not in the index)."""
    channels = plan.channels_for_level(level)
    if channels is None or queries.size == 0:
        return
    queries = queries[channels.counts()[queries] > 0]
    index = s.window_index(level)
    fragments = level > 1
    assigned = plan.assigned_scan if fragments else None
    scan_rows = s.scan_rows(plan.assigned_scan) if fragments and (plan.assigned_scan >= 0).any() else None
    q, start, count = eligible_pairs(index, queries, plan.precursor_mz, plan.rt_lo, plan.rt_hi, assigned,
                                     gate=fragments, scan_rows=scan_rows)  # fmt: skip
    if q.size:
        yield q, index.rows, start, count


#: The most (window, query) pairs :func:`eligible_counts` builds at once.
PAIR_BUDGET = 1 << 22


def eligible_counts(plan: QueryPlan, s: Ledger, level: int, queries: np.ndarray) -> np.ndarray:
    """Per query of ``queries`` (ascending), how many (scan, query) pairs
    ``level`` could match it in, without building them. The (window,
    query) pairs the count is made from are built for chunks of queries of
    at most about :data:`PAIR_BUDGET` pairs: on a DDA run every scan is a
    window of its own, and building them for every query at once was
    killed at 10.3 GB (until 2026-11-09)."""
    out = np.zeros(queries.size, np.int64)
    if queries.size == 0 or plan.channels_for_level(level) is None:
        return out
    windows = _windows_per_query(plan, s, level, queries)
    cum = np.cumsum(np.maximum(windows, 1))
    edges = [0]
    while edges[-1] < queries.size:
        base = cum[edges[-1] - 1] if edges[-1] else 0
        edges.append(max(int(np.searchsorted(cum, base + PAIR_BUDGET, side="right")), edges[-1] + 1))
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        chunk = queries[a:b]
        for q, _, _, count in _eligible_ranges(plan, s, level, chunk):
            out[a:b] += np.bincount(np.searchsorted(chunk, q), weights=count, minlength=chunk.size).astype(np.int64)
    return out


def _windows_per_query(plan: QueryPlan, s: Ledger, level: int, queries: np.ndarray) -> np.ndarray:
    """How many of ``level``'s windows each query falls in (what building
    its (window, query) pairs costs), counted from the windows' sorted
    bounds alone."""
    index = s.window_index(level)
    if level == 1:
        return np.full(queries.size, index.n_windows, np.int64)
    free = int(index.unbounded.sum())
    lower = np.sort(index.lower[~index.unbounded])
    upper = np.sort(index.upper[~index.unbounded])
    pmz = plan.precursor_mz[queries]
    finite = ~np.isnan(pmz)
    inside = np.searchsorted(lower, pmz, side="right") - np.searchsorted(upper, pmz, side="left")
    return np.where(finite, inside, lower.size).astype(np.int64) + free


@dataclass(frozen=True)
class QueryBatch:
    """A contiguous range of queries and everything the sweep matched for
    them, sorted by (query, level, scan order, channel, point)."""

    plan: QueryPlan
    queries: np.ndarray  # the batch's queries, ascending
    matches: Matches
    scans: Ledger
    scan_order: np.ndarray  # rank of each ledger row
    settings: SweepSettings
    #: Evidence counts of these queries, sorted by (query, scan); with a gate.
    evidence: Evidence | None = None
    #: Per query, its matches whose point another query claimed; with claims.
    shared: np.ndarray | None = None
    #: The gate's verdict per query; with a gate.
    decision: Decision | None = None

    def passing(self) -> QueryBatch:
        """The batch narrowed to the queries the gate passed (all of them
        without a gate)."""
        if self.decision is None:
            return self
        keep = self.decision.passed
        kept = self.queries[keep]
        m = self.matches
        e = self.evidence
        return replace(
            self,
            queries=kept,
            matches=m.take(np.flatnonzero(np.isin(m.query, kept))),
            evidence=None if e is None else e.take(np.flatnonzero(np.isin(e.query, kept))),
            shared=None if self.shared is None else self.shared[keep],
            decision=Decision(*(getattr(self.decision, f)[keep] for f in
                                ("applicable", "passed", "n_evidence_scans", "longest_run",
                                 "best_scan", "best_rt", "best_n"))),  # fmt: skip
        )

    def match_rows(self) -> np.ndarray:
        """Ledger row of each match's scan."""
        return self.scans.rows_of(self.matches.scan)

    def eligible(self, level: int, queries: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(query, ledger row) of every (scan, query) pair the sweep matched
        for this batch at ``level`` (or for ``queries``, an ascending subset
        of it), whether or not anything was found: scans of the level with
        points, inside the query's interval and isolation window, and its
        assigned scan for fragments. Sorted by (query, scan order). Each pair
        stands for the query's cells in that scan. Only eligible pairs are
        built, so the cost is theirs, not the batch's queries × scans."""
        s = self.scans
        qs = self.queries if queries is None else queries
        q_parts, r_parts = [], []
        for q, r, start, count in _eligible_ranges(self.plan, s, level, qs):
            qq = np.repeat(q, count)
            rr = r[ragged_arange(start, count)]
            # The ranges are the map's test; held to it rather than trusted.
            ok = pair_eligible(qq, s.rt[rr], s.scan[rr], self.plan.rt_lo, self.plan.rt_hi,
                               self.plan.assigned_scan if level > 1 else None)  # fmt: skip
            q_parts.append(qq[ok])
            r_parts.append(rr[ok])
        if not q_parts:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        q = np.concatenate(q_parts)
        r = np.concatenate(r_parts)
        by = np.lexsort((self.scan_order[r], q))
        return q[by], r[by]

    def eligible_counts(self, level: int) -> np.ndarray:
        """Per query of the batch, how many (scan, query) pairs ``level``
        could match it in: the sizes of :meth:`eligible`, without building it."""
        return eligible_counts(self.plan, self.scans, level, self.queries)

    def eligible_cells(self, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(query, ledger row, channel) of every eligible cell, in output
        order: query, scan order, channel."""
        q, r = self.eligible(level)
        channels = self.plan.channels_for_level(level)
        n = channels.counts()[q]
        channel = ragged_arange(np.zeros(q.size, np.int64), n)
        return np.repeat(q, n), np.repeat(r, n), channel

    def cells(self, level: int, *, fill: bool = True, queries: np.ndarray | None = None) -> Cells:
        """The batch's cells at ``level`` (:class:`Cells`), for its queries
        or ``queries`` (an ascending subset): one row per match, and with
        ``fill`` one per eligible cell that nothing matched, so a trace is
        dense over the scans each query could have been matched in. A rule
        that gives several points a cell (``all``) gives several rows."""
        channels = self.plan.channels_for_level(level)
        if channels is None:
            raise ValueError(f"the plan has no channels at MS{level}")
        m = self.matches
        sel = m.level == level
        if queries is not None:
            sel &= np.isin(m.query, queries)
        at = np.flatnonzero(sel)
        mq, mr, mc = m.query[at], self.scans.rows_of(m.scan[at]), m.channel[at]
        parts = dict(query=mq, row=mr, channel=mc, matched=np.ones(at.size, dtype=bool),
                     point_index=m.point_index[at], n_points=m.n_points[at], mz_observed=m.mz[at],
                     intensity=m.intensity[at])  # fmt: skip
        order = self.scan_order
        # Matches come sorted by (query, scan order, channel, point) and the
        # eligible cells by (query, scan order, channel): a cell key makes
        # both one sequence, and the unmatched cells are merged in by one
        # search (until 2026-11-09 both were re-sorted on four keys, twice).
        sub = self.queries if queries is None else queries
        width = int(channels.counts().max()) + 1 if channels.n else 1
        span = int(order.max()) + 1 if order.size else 1
        fits = float(sub.size + 1) * span * width < 2.0**62

        def cell_key(q, r, c):
            return (np.searchsorted(sub, q).astype(np.int64) * span + order[r]) * width + c

        mk = cell_key(mq, mr, mc) if fits else None
        in_order = fits and bool(np.all(np.diff(mk) >= 0)) and bool(
            np.all((np.diff(mk) > 0) | (np.diff(parts["point_index"]) >= 0)))  # fmt: skip
        if fill:
            cq, cr, ci = self.eligible_cells_of(level, sub)
            if in_order:
                ck = cell_key(cq, cr, ci)
                at = np.searchsorted(mk, ck, side="left")
                hit = (at < mk.size) & (mk[np.minimum(at, mk.size - 1)] == ck) if mk.size else np.zeros(ck.size, bool)
                empty = np.flatnonzero(~hit)
                # Each unmatched cell lands before the first match after it.
                dest = at[empty] + np.arange(empty.size)
                n_out = mq.size + empty.size
                is_cell = np.zeros(n_out, dtype=bool)
                is_cell[dest] = True
                src = np.empty(n_out, dtype=np.int64)
                src[~is_cell] = np.arange(mq.size)
                src[is_cell] = mq.size + np.arange(empty.size)
                columns = {
                    "query": np.concatenate([mq, cq[empty]]), "row": np.concatenate([mr, cr[empty]]),
                    "channel": np.concatenate([mc, ci[empty]]),
                    "matched": np.concatenate([parts["matched"], np.zeros(empty.size, bool)]),
                    "point_index": np.concatenate([parts["point_index"], np.full(empty.size, -1, np.int64)]),
                    "n_points": np.concatenate([parts["n_points"], np.zeros(empty.size, np.int64)]),
                    "mz_observed": np.concatenate([parts["mz_observed"], np.full(empty.size, np.nan)]),
                    "intensity": np.concatenate([parts["intensity"], np.zeros(empty.size)]),
                }  # fmt: skip
                return Cells(level=int(level), channels=channels, scans=self.scans,
                             **{k: np.asarray(v)[src] for k, v in columns.items()})  # fmt: skip
            return self._cells_by_sort(level, channels, parts, cq, cr, ci)
        if in_order:
            return Cells(level=int(level), channels=channels, scans=self.scans,
                         **{k: np.asarray(v) for k, v in parts.items()})  # fmt: skip
        by = np.lexsort((parts["point_index"], parts["channel"], order[parts["row"]], parts["query"]))
        return Cells(level=int(level), channels=channels, scans=self.scans,
                     **{k: np.asarray(v)[by] for k, v in parts.items()})  # fmt: skip

    def _cells_by_sort(self, level, channels, parts, cq, cr, ci) -> Cells:
        """:meth:`cells` by sorting, for matches out of order or keys too
        wide to combine (the general form the merge is held to)."""
        order = self.scan_order
        mq, mr, mc = parts["query"], parts["row"], parts["channel"]
        key_q = np.concatenate([mq, cq])
        key_s = np.concatenate([order[mr], order[cr]])
        key_c = np.concatenate([mc, ci])
        is_cell = np.concatenate([np.zeros(mq.size, bool), np.ones(cq.size, bool)])
        by = np.lexsort((is_cell, key_c, key_s, key_q))
        kq, ks, kc, flag = key_q[by], key_s[by], key_c[by], is_cell[by]
        new = np.r_[True, (kq[1:] != kq[:-1]) | (ks[1:] != ks[:-1]) | (kc[1:] != kc[:-1])] if by.size else by.astype(bool)
        group = np.cumsum(new) - 1
        has_match = np.zeros(int(new.sum()), dtype=bool)
        np.logical_or.at(has_match, group[~flag], True)
        empty = flag & ~has_match[group]
        take = by[empty] - mq.size
        parts = {
            "query": np.concatenate([mq, cq[take]]), "row": np.concatenate([mr, cr[take]]),
            "channel": np.concatenate([mc, ci[take]]),
            "matched": np.concatenate([parts["matched"], np.zeros(take.size, bool)]),
            "point_index": np.concatenate([parts["point_index"], np.full(take.size, -1, np.int64)]),
            "n_points": np.concatenate([parts["n_points"], np.zeros(take.size, np.int64)]),
            "mz_observed": np.concatenate([parts["mz_observed"], np.full(take.size, np.nan)]),
            "intensity": np.concatenate([parts["intensity"], np.zeros(take.size)]),
        }  # fmt: skip
        by = np.lexsort((parts["point_index"], parts["channel"], order[parts["row"]], parts["query"]))
        return Cells(level=int(level), channels=channels, scans=self.scans,
                     **{k: np.asarray(v)[by] for k, v in parts.items()})  # fmt: skip

    def eligible_cells_of(self, level: int, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:meth:`eligible_cells` for ``queries``, an ascending subset."""
        q, r = self.eligible(level, queries)
        channels = self.plan.channels_for_level(level)
        n = channels.counts()[q]
        channel = ragged_arange(np.zeros(q.size, np.int64), n)
        return np.repeat(q, n), np.repeat(r, n), channel


@dataclass(frozen=True)
class Cells:
    """A batch's cells at one level (:meth:`QueryBatch.cells`): one row per
    match of a cell, and, with ``fill``, one per eligible cell with none
    (``matched`` false). Rows in output order: query, scan order, channel,
    point. What is stored is what the sweep found; the rest (the channel's
    identity and abundance, the scan's time and context, the m/z error) is
    looked up as asked."""

    level: int
    query: np.ndarray  # int64
    row: np.ndarray  # int64, ledger row of the scan
    channel: np.ndarray  # int64, ordinal among the query's channels of this kind
    matched: np.ndarray  # bool
    point_index: np.ndarray  # int64, -1 when unmatched or aggregated
    n_points: np.ndarray  # int64, points the match stands for (0 unmatched)
    mz_observed: np.ndarray  # float64, NaN unmatched
    intensity: np.ndarray  # float64, 0 unmatched
    channels: Channels
    scans: Ledger

    @property
    def n(self) -> int:
        return int(self.query.size)

    @property
    def channel_index(self) -> np.ndarray:
        """Each row's channel as an index into :attr:`channels`' arrays."""
        return self.channels.offsets[self.query] + self.channel

    def _of(self, name: str) -> np.ndarray:
        return getattr(self.channels, name)[self.channel_index]

    @property
    def mz_theoretical(self) -> np.ndarray:
        return self._of("mz")

    @property
    def abundance(self) -> np.ndarray:
        return self._of("abundance")

    @property
    def evidence_key(self) -> np.ndarray:
        return self._of("evidence_key")

    @property
    def mz_error_da(self) -> np.ndarray:
        """Observed − theoretical, NaN where unmatched."""
        return np.where(self.matched, self.mz_observed - self.mz_theoretical, np.nan)

    @property
    def mz_error_ppm(self) -> np.ndarray:
        theo = self.mz_theoretical
        return np.where(self.matched, (self.mz_observed - theo) / theo * 1e6, np.nan)

    @property
    def scan(self) -> np.ndarray:
        return self.scans.scan[self.row]

    @property
    def rt(self) -> np.ndarray:
        return self.scans.rt[self.row]

    @property
    def tic(self) -> np.ndarray:
        return self.scans.tic[self.row]

    @property
    def iit(self) -> np.ndarray:
        return self.scans.iit[self.row]

    def table(self) -> pa.Table:
        """The cells as :data:`SWEEP_CELL_TABLE` rows."""
        g = self.channel_index
        c = self.channels
        frag = self.level > 1
        none = np.ones(self.n, dtype=bool)
        names = np.array([None, *c.loss_ids], dtype=object)
        tic, iit = self.tic, self.iit
        return pa.table(
            {
                "query": pa.array(self.query.astype(np.int32)),
                "level": pa.array(np.full(self.n, self.level, np.int8)),
                "scan": pa.array(self.scan.astype(np.int32)),
                "rt": pa.array(self.rt),
                "tic": pa.array(tic, mask=np.isnan(tic)),
                "iit": pa.array(iit, mask=np.isnan(iit)),
                "channel": pa.array(self.channel.astype(np.int32)),
                "precursor_charge": pa.array(c.precursor_charge[g].astype(np.int8)),
                "isotope": pa.array(c.isotope[g], mask=none if frag else None),
                "ion_type": pa.array(c.ion_type[g], mask=None if frag else none),
                "position": pa.array(c.position[g], mask=None if frag else none),
                "fragment_charge": pa.array(c.fragment_charge[g], mask=None if frag else none),
                "loss_id": pa.array(names[c.loss[g]] if frag else [None] * self.n, pa.string()),
                "abundance": pa.array(c.abundance[g], mask=np.isnan(c.abundance[g])),
                "evidence_key": pa.array(c.evidence_key[g].astype(np.int32)),
                "mz_theoretical": pa.array(c.mz[g]),
                "matched": pa.array(self.matched),
                "point_index": pa.array(self.point_index.astype(np.int32), mask=self.point_index < 0),
                "n_points": pa.array(self.n_points.astype(np.int32)),
                "mz_observed": pa.array(self.mz_observed),
                "intensity": pa.array(self.intensity),
                "mz_error_da": pa.array(self.mz_error_da),
                "mz_error_ppm": pa.array(self.mz_error_ppm),
            },
            schema=SWEEP_CELL_TABLE,
        )


class Reducer(Protocol):
    """Turns a bucket of queries' matches into a table, or several: a
    reducer of one output has ``name`` and ``schema`` and returns a table;
    one of several also has ``outputs`` (output name → schema) and returns
    a mapping of output name → table (one pass of a scorer can write its
    candidates, its per-ion quantities and its verdicts). Every table is
    held to its schema by column name. ``gated`` reducers see only the
    queries that passed the evidence gate. A reducer that builds a row for
    every eligible cell (zero-filling) says so with a true ``zero_fills``
    attribute, so that a sweep on disk sizes its buckets by those rows
    rather than by matches alone."""

    name: str
    schema: pa.Schema
    gated: bool

    def reduce(self, batch: QueryBatch) -> pa.Table | Mapping[str, pa.Table]: ...


@dataclass(frozen=True)
class QuerySummaryReducer:
    """One row per query: the gate's verdict, the scans each level could
    have matched it in, the cells it matched, and how many of its matches
    another query claimed too."""

    name: str = "query_summary"
    schema: pa.Schema = SWEEP_QUERY_SUMMARY_TABLE
    gated: bool = False

    def reduce(self, batch: QueryBatch) -> pa.Table:
        plan, q = batch.plan, batch.queries
        n = q.size
        frag = next((lv for lv in plan.levels if lv > 1), None)

        def per_query(values: np.ndarray) -> np.ndarray:
            return np.bincount(np.searchsorted(q, values), minlength=n).astype(np.int32)

        def eligible(level: int | None) -> np.ndarray:
            if level is None or level not in plan.levels:
                return np.zeros(n, np.int32)
            return batch.eligible_counts(level).astype(np.int32)

        m = batch.matches
        matched = {
            lv: per_query(m.query[m.level == lv]) if lv in plan.levels else np.zeros(n, np.int32)
            for lv in (1, frag)
        }
        d = batch.decision
        none = np.ones(n, dtype=bool)
        charge = plan.precursor_charge[q]
        query = pa.array(q)

        def opt(values, dtype, missing=None):
            mask = none if d is None else (missing if missing is not None else None)
            return pa.array(np.zeros(n, dtype) if d is None else values.astype(dtype), mask=mask)

        return pa.table(
            {
                "acquisition_id": pa.array(np.full(n, batch.settings.acquisition_id, np.int64)),
                "target_id": plan.targets.column("target_id").take(query),
                "query": pa.array(q.astype(np.int32)),
                "modified_sequence": plan.targets.column("modified_sequence").take(query),
                "precursor_charge": pa.array(charge.astype(np.int8), mask=charge < 0),
                "is_decoy": plan.targets.column("is_decoy").take(query),
                "decoy_set": plan.targets.column("decoy_set").take(query),
                "gate_applicable": pa.array(d.applicable if d is not None else np.zeros(n, bool),
                                            mask=none if d is None else None),  # fmt: skip
                "passed": pa.array(d.passed if d is not None else np.zeros(n, bool),
                                   mask=none if d is None else None),  # fmt: skip
                "n_evidence_scans": opt(d.n_evidence_scans if d else None, np.int32),
                "longest_run": opt(d.longest_run if d else None, np.int32),
                "best_scan": opt(d.best_scan if d else None, np.int32,
                                 None if d is None else d.best_scan < 0),  # fmt: skip
                "best_rt": opt(d.best_rt if d else None, np.float64,
                               None if d is None else np.isnan(d.best_rt)),  # fmt: skip
                "best_n_keys": opt(d.best_n if d else None, np.int32),
                "n_eligible_ms1": pa.array(eligible(1)),
                "n_eligible_ms2": pa.array(eligible(frag)),
                "n_matched_ms1": pa.array(matched[1]),
                "n_matched_ms2": pa.array(matched[frag]),
                "n_shared_matches": pa.array(
                    np.zeros(n, np.int32) if batch.shared is None else batch.shared.astype(np.int32),
                    mask=none if batch.shared is None else None,
                ),
            },
            schema=SWEEP_QUERY_SUMMARY_TABLE,
        )
