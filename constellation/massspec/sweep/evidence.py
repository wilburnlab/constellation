"""The evidence gate: whether a query showed enough of itself to be worth scoring.

The map counts, for every scan and query, how many of the query's
distinct fragment ions (an ion type and number, whatever its charge or
loss) have a channel within tolerance of one of the scan's evidence peaks,
its ``top_peaks`` most intense (or all of them), never counting more ions
than the distinct peaks that matched them. A scan with at least
``min_fragments`` such ions is evidence. Complementary b and y ions of one
bond count twice: they sit at different m/z, so each is a match of its
own; one ion at another charge or with a loss does not (until 2026-11-07
these words said "bonds" for what was always ion type and number). The reduce then decides, over
the scans the query could have been matched in, in order: the query
passes when they hold a run of ``min_consecutive`` evidence scans, each
within ``max_gap_s`` of the last. A query without fragments is gated on
its isotopes in MS1 the same way when ``min_isotopes`` is set, and passes
otherwise.

A bare count is a weak filter on low-resolution fragments: at 0.35 Da a
scan of about a thousand peaks puts a peak within tolerance of a given
m/z nearly half the time, so two of twenty fragment channels match by
chance in most scans. That is what ``top_peaks`` and ``min_consecutive``
are for; how they should be set is measured, not assumed.

The gate is defined on peaks, centroids, which is why it says peaks where
the rest of the sweep says points: its evidence is a scan's most intense
peaks. On profile points it would take the tops of a few profile peaks
instead, which is a question for the profile rule, not settled here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellation.massspec.sweep.kernel import match_group
from constellation.massspec.sweep.queries import Channels
from constellation.massspec.sweep.rules import AllWithin
from constellation.massspec.sweep.schedule import ragged_arange

__all__ = ["EvidenceGate", "Evidence", "count_evidence", "decide"]


@dataclass(frozen=True)
class EvidenceGate:
    """When a query counts as seen.

    The defaults were measured on an ion-trap DIA run of a BSA and labelled
    lysin digest (0.35 Da, about 710 peaks a scan; ``docs/plans/sweep.md``):
    they pass 2.5 % of reversed-sequence queries and the digest's known
    peptides. With every peak as evidence, or two fragment ions in one scan, 87 %
    of reversed queries pass there."""

    #: Distinct fragment ions (ion type and number, any charge or loss) a
    #: scan must show.
    min_fragments: int = 6
    #: Evidence peaks per scan: the most intense this many; None for all.
    top_peaks: int | None = 50
    #: Evidence scans in a row the query must show.
    min_consecutive: int = 3
    #: Longest retention-time step (s) between evidence scans of one run.
    max_gap_s: float | None = None
    #: For queries without fragments: isotopes an MS1 scan must show. None
    #: lets such queries through.
    min_isotopes: int | None = None


@dataclass(frozen=True)
class Evidence:
    """Per (query, scan) evidence counts. ``group`` is 0 for fragments and
    the precursor charge for isotopes."""

    query: np.ndarray  # int64
    group: np.ndarray  # int8
    scan: np.ndarray  # int64
    n_keys: np.ndarray  # int64
    n_peaks: np.ndarray  # int64
    n_channels: np.ndarray  # int64
    #: Summed intensity of the distinct evidence peaks matched.
    intensity: np.ndarray  # float64

    @property
    def n(self) -> int:
        return int(self.query.size)

    def count(self) -> np.ndarray:
        """Evidence per row: distinct fragment ions (or isotopes), never
        more than peaks."""
        return np.minimum(self.n_keys, self.n_peaks)

    @classmethod
    def empty(cls) -> Evidence:
        i = np.empty(0, np.int64)
        return cls(i, np.empty(0, np.int8), i, i, i, i, np.empty(0, np.float64))

    @classmethod
    def concat(cls, parts: list[Evidence]) -> Evidence:
        parts = [p for p in parts if p.n]
        if not parts:
            return cls.empty()
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in _FIELDS))

    def take(self, idx: np.ndarray) -> Evidence:
        return Evidence(*(getattr(self, f)[idx] for f in _FIELDS))


_FIELDS = ("query", "group", "scan", "n_keys", "n_peaks", "n_channels", "intensity")


def _evidence_peaks(offsets: np.ndarray, rows: np.ndarray, intensity: np.ndarray, top: int | None):
    """The block positions of each row's evidence peaks, in m/z order, as a
    sub-block (offsets over ``rows``)."""
    counts = offsets[rows + 1] - offsets[rows]
    flat = ragged_arange(offsets[rows], counts)
    if top is None:
        return flat, np.r_[0, np.cumsum(counts)].astype(np.int64)
    row_of = np.repeat(np.arange(rows.size), counts)
    # Most intense first within each row; ties to the lower m/z.
    by = np.lexsort((np.arange(flat.size), -intensity[flat], row_of))
    rank = np.empty(flat.size, np.int64)
    first = np.r_[0, np.cumsum(counts)][:-1]
    rank[by] = np.arange(flat.size) - np.repeat(first, counts)
    keep = rank < top
    kept = flat[keep]
    return kept, np.r_[0, np.cumsum(np.bincount(row_of[keep], minlength=rows.size))].astype(np.int64)


def count_evidence(
    mz: np.ndarray,
    intensity: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    scan_id: np.ndarray,
    channels: Channels,
    channel_at: np.ndarray,
    channel_query: np.ndarray,
    eligible,
    tolerance: float,
    unit: str,
    top_peaks: int | None,
    *,
    by_charge: bool,
) -> Evidence:
    """Evidence of one group of scans: its rows of the block, against the
    channels ``channel_at`` (of queries ``channel_query``).
    ``eligible(query, rows)`` says which (scan, query) pairs may count.
    ``by_charge`` counts isotopes separately for each precursor charge."""
    picked, sub_offsets = _evidence_peaks(offsets, rows, intensity, top_peaks)
    if picked.size == 0:
        return Evidence.empty()
    sub_rows = np.arange(rows.size, dtype=np.int64)
    m = match_group(mz[picked], intensity[picked], sub_offsets, sub_rows, channels.mz[channel_at],
                    AllWithin(), tolerance, unit)  # fmt: skip
    if m.n == 0:
        return Evidence.empty()
    row = rows[m.row]
    query = channel_query[m.channel]
    ok = eligible(query, row)
    if not ok.any():
        return Evidence.empty()
    row, query, channel, peak = row[ok], query[ok], channel_at[m.channel][ok], m.point[ok]
    peak_intensity = m.intensity[ok]
    key = channels.evidence_key[channel].astype(np.int64)
    group = (
        channels.precursor_charge[channel].astype(np.int64) if by_charge else np.zeros(row.size, np.int64)
    )
    # (query, group, scan) entries and (entry, value) pairs as single
    # integers: one-dimensional uniques are far quicker than row-wise ones.
    # Each stride is one past the largest value packed under it, and values
    # are shifted to start at 0 first (isotope keys run negative for
    # labelled peptides), so no two pairs can share an integer.
    n_rows = int(rows.max()) + 1
    n_groups = int(group.max()) + 1
    entry = (query * n_groups + group) * n_rows + row
    entries, inverse = np.unique(entry, return_inverse=True)

    def distinct(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        values = values - values.min()
        width = int(values.max()) + 1
        _, first = np.unique(inverse.astype(np.int64) * width + values, return_index=True)
        w = None if weights is None else weights[first]
        return np.bincount(inverse[first], weights=w, minlength=entries.size)

    return Evidence(
        query=entries // (n_groups * n_rows),
        group=((entries // n_rows) % n_groups).astype(np.int8),
        scan=scan_id[entries % n_rows],
        n_keys=distinct(key).astype(np.int64),
        n_peaks=distinct(peak).astype(np.int64),
        n_channels=distinct(channel).astype(np.int64),
        intensity=distinct(peak, peak_intensity),
    )


@dataclass(frozen=True)
class Decision:
    """The gate's verdict for each query of a batch."""

    applicable: np.ndarray  # bool: the query had something to be gated on
    passed: np.ndarray  # bool
    n_evidence_scans: np.ndarray  # int64
    longest_run: np.ndarray  # int64
    best_scan: np.ndarray  # int64, -1 for none
    best_rt: np.ndarray  # float64, NaN for none
    best_n: np.ndarray  # int64


def decide(
    gate: EvidenceGate,
    queries: np.ndarray,
    has_fragments: np.ndarray,
    eligible_fragments,
    eligible_isotopes,
    evidence: Evidence,
    ledger,
) -> Decision:
    """Apply ``gate`` to ``queries``. ``eligible_*()`` give the (query,
    ledger row) pairs each query could be matched in at that kind's level,
    in scan order, for at least every query with evidence of that kind (one
    without has no run, and decides the same with or without its pairs);
    ``evidence`` holds the map's counts for these queries."""
    n = queries.size
    applicable = has_fragments.copy()
    if gate.min_isotopes is not None:
        applicable |= ~has_fragments
    passed = ~applicable
    n_ev = np.zeros(n, np.int64)
    longest = np.zeros(n, np.int64)
    best_scan = np.full(n, -1, np.int64)
    best_rt = np.full(n, np.nan)
    best_n = np.zeros(n, np.int64)
    rows_of_scan = ledger.rows_of

    for fragments in (True, False):
        if not fragments and gate.min_isotopes is None:
            continue
        threshold = gate.min_fragments if fragments else gate.min_isotopes
        sel = (evidence.group == 0) if fragments else (evidence.group != 0)
        e = evidence.take(np.flatnonzero(sel))
        q, r = (eligible_fragments if fragments else eligible_isotopes)()
        want = has_fragments[np.searchsorted(queries, q)] == fragments
        q, r = q[want], r[want]
        if q.size == 0:
            continue
        count = np.zeros(q.size, np.int64)
        signal = np.zeros(q.size, np.float64)
        if e.n:
            # Isotope evidence is counted per charge; a scan counts its best
            # charge, and that charge's intensity.
            er = rows_of_scan(e.scan)
            key_e = np.stack([e.query, er], axis=1)
            uniq, inv = np.unique(key_e, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            best = np.zeros(uniq.shape[0], np.int64)
            np.maximum.at(best, inv, e.count())
            top = e.count() == best[inv]
            best_i = np.zeros(uniq.shape[0], np.float64)
            np.maximum.at(best_i, inv[top], e.intensity[top])
            key_c = np.stack([q, r], axis=1)
            dt = np.dtype([("q", np.int64), ("r", np.int64)])
            kc = np.ascontiguousarray(key_c).view(dt).reshape(-1)
            ku = np.ascontiguousarray(uniq).view(dt).reshape(-1)
            at = np.searchsorted(ku, kc)
            hit = (at < ku.size) & (ku[np.minimum(at, ku.size - 1)] == kc)
            count[hit] = best[at[hit]]
            signal[hit] = best_i[at[hit]]
        is_ev = count >= threshold
        p = np.searchsorted(queries, q)
        # Runs of evidence scans in each query's sequence of eligible scans.
        rt = ledger.rt[r]
        new_query = np.r_[True, q[1:] != q[:-1]]
        step_ok = np.r_[False, ~new_query[1:]]
        if gate.max_gap_s is not None:
            step_ok &= np.r_[False, np.diff(rt) <= gate.max_gap_s]
        # A run continues where this scan and the one before are evidence and
        # the step between them is allowed; run lengths by counting.
        cont = is_ev & np.r_[False, is_ev[:-1]] & step_ok
        start = is_ev & ~cont
        run_id = np.cumsum(start)
        lengths = np.bincount(run_id[is_ev], minlength=int(run_id[-1]) + 1)[1:]
        np.maximum.at(longest, p[start], lengths)
        np.add.at(n_ev, p, is_ev.astype(np.int64))
        # Best scan: most evidence, then the most intense, then the earliest.
        order = np.lexsort((np.arange(q.size), -signal, -count, p))
        first = np.r_[True, p[order][1:] != p[order][:-1]]
        top = order[first]
        has = count[top] > 0
        best_scan[p[top][has]] = ledger.scan[r[top][has]]
        best_rt[p[top][has]] = rt[top][has]
        best_n[p[top][has]] = count[top][has]
        done = has_fragments if fragments else ~has_fragments
        passed = np.where(done, longest >= gate.min_consecutive, passed)
    return Decision(applicable, passed, n_ev, longest, best_scan, best_rt, best_n)
