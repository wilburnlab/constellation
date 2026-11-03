"""After the sweep: a bucket of queries with everything matched for them.

The sweep's map reads scans and writes matches; once it is done, each
query's matches are complete, whichever scans (and workers) they came
from, and a :class:`Reducer` turns a :class:`QueryBatch` of them into a
table. This is the hook the scorer plugs into. A reducer that needs to
know which cells could have matched but did not (to zero-fill a trace,
or to count a non-detection) asks :meth:`QueryBatch.eligible`, which uses
the same schedule the map did.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pyarrow as pa

from constellation.massspec.sweep.evidence import Decision, Evidence
from constellation.massspec.sweep.queries import QueryPlan
from constellation.massspec.sweep.schedule import cell_eligible, isolation_groups, ragged_arange
from constellation.massspec.sweep.schemas import (
    SWEEP_MATCH_TABLE,
    SWEEP_QUERY_SUMMARY_TABLE,
    SWEEP_SCAN_TABLE,
)

if TYPE_CHECKING:
    from constellation.massspec.sweep.engine import SweepSettings

__all__ = ["Ledger", "Matches", "QueryBatch", "Reducer", "QuerySummaryReducer"]


@dataclass(frozen=True)
class Ledger:
    """The scans the sweep read, in stored order, peakless ones included."""

    scan: np.ndarray  # int64
    level: np.ndarray  # int8
    rt: np.ndarray  # float64
    lower: np.ndarray  # float64, NaN without bounds
    upper: np.ndarray
    n_peaks: np.ndarray  # int64

    @property
    def n(self) -> int:
        return int(self.scan.size)

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
                "n_peaks": pa.array(self.n_peaks.astype(np.int32)),
            },
            schema=SWEEP_SCAN_TABLE,
        )


_MATCH_FIELDS = ("query", "level", "scan", "ion", "peak_index", "mz", "intensity", "n_peaks")


@dataclass(frozen=True)
class Matches:
    """Matched (query ion, scan) cells. ``ion`` is the ion's position among
    its query's ions of the level's kind; ``peak_index`` the peak's position
    in its scan's stored order, -1 when a rule aggregated several."""

    query: np.ndarray  # int64
    level: np.ndarray  # int8
    scan: np.ndarray  # int64
    ion: np.ndarray  # int64
    peak_index: np.ndarray  # int64
    mz: np.ndarray  # float64
    intensity: np.ndarray  # float64
    n_peaks: np.ndarray  # int64

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
        """Sorted by (query, level, scan order, ion, peak); ``scan_order`` is
        each match's scan's rank."""
        by = np.lexsort((self.peak_index, self.ion, scan_order, self.level, self.query))
        return self.take(by)

    def table(self) -> pa.Table:
        return pa.table(
            {
                "query": pa.array(self.query.astype(np.int32)),
                "level": pa.array(self.level.astype(np.int8)),
                "scan": pa.array(self.scan.astype(np.int32)),
                "ion": pa.array(self.ion.astype(np.int32)),
                "peak_index": pa.array(self.peak_index.astype(np.int32), mask=self.peak_index < 0),
                "mz_observed": pa.array(self.mz),
                "intensity": pa.array(self.intensity),
                "n_peaks": pa.array(self.n_peaks.astype(np.int32)),
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
            ion=col("ion", np.int64),
            peak_index=col("peak_index", np.int64, -1),
            mz=col("mz_observed", np.float64),
            intensity=col("intensity", np.float64),
            n_peaks=col("n_peaks", np.int64),
        )


@dataclass(frozen=True)
class QueryBatch:
    """A contiguous range of queries and everything the sweep matched for
    them, sorted by (query, level, scan order, ion, peak)."""

    plan: QueryPlan
    queries: np.ndarray  # the batch's queries, ascending
    matches: Matches
    scans: Ledger
    scan_order: np.ndarray  # rank of each ledger row
    settings: SweepSettings
    #: Evidence counts of these queries, sorted by (query, scan); with a gate.
    evidence: Evidence | None = None
    #: Per query, its matches whose peak another query claimed; with claims.
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

    def eligible(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """(query, ledger row) of every cell the sweep matched for this
        batch at ``level``, whether or not anything was found: scans of the
        level with peaks, inside the query's interval and isolation window,
        and its assigned scan for fragments. Sorted by (query, scan order)."""
        ions = self.plan.ions_for_level(level)
        if ions is None:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        s = self.scans
        rows = np.flatnonzero((s.level == level) & (s.n_peaks > 0))
        queries = self.queries[ions.counts()[self.queries] > 0]
        groups = isolation_groups(
            queries, self.plan.precursor_mz, s.lower[rows], s.upper[rows], rows, gate=level > 1
        )
        q_parts, r_parts = [], []
        for g in groups:
            q = np.tile(g.queries, g.rows.size)
            r = np.repeat(g.rows, g.queries.size)
            ok = cell_eligible(
                q, s.rt[r], s.scan[r], self.plan.rt_lo, self.plan.rt_hi,
                self.plan.assigned_scan if level > 1 else None,
            )  # fmt: skip
            q_parts.append(q[ok])
            r_parts.append(r[ok])
        if not q_parts:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        q = np.concatenate(q_parts)
        r = np.concatenate(r_parts)
        by = np.lexsort((self.scan_order[r], q))
        return q[by], r[by]

    def eligible_cells(self, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(query, ledger row, ion) of every eligible cell, in output order:
        query, scan order, ion."""
        q, r = self.eligible(level)
        ions = self.plan.ions_for_level(level)
        n = ions.counts()[q]
        ion = ragged_arange(np.zeros(q.size, np.int64), n)
        return np.repeat(q, n), np.repeat(r, n), ion


class Reducer(Protocol):
    """Turns a bucket of queries' matches into a table. ``gated`` reducers
    see only the queries that passed the evidence gate."""

    name: str
    schema: pa.Schema
    gated: bool

    def reduce(self, batch: QueryBatch) -> pa.Table: ...


@dataclass(frozen=True)
class QuerySummaryReducer:
    """One row per query: the gate's verdict, the scans each level could
    have matched it in, the ion cells it matched, and how many of its
    matches another query claimed too."""

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
            return per_query(batch.eligible(level)[0])

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
