"""The dense view of a batch: each query's evidence at one level as padded
``[queries, scans, channels]`` tensors, what a scorer computes on.

:meth:`QueryBatch.cells <constellation.massspec.sweep.reduce.QueryBatch.cells>`
gives the same cells in long format, one row per cell. Almost every
feature a scorer wants (a trace smoothed, its peaks, fragments correlated
with each other or with the precursor, the library compared scan by scan)
is an array operation on a query's scans × channels, so the view lays the
cells out that way once.

Each level keeps its own scan axis: a query's eligible scans at that level,
in scan order (retention time, scan number), so MS1 and MS2 are not forced
onto one cycle axis (methods that take MS1 at their own cadence, staggered
windows that visit a query twice a cycle, and DDA would not fit one; PI,
2026-11-10). Each MS2 scan names its parent MS1 scan
(``acquisition_structure.parent_scans``), and :meth:`Dense.parent_slots`
finds it on the query's MS1 axis; each slot carries its retention time, so
an MS1 trace can be interpolated to MS2 times.

Masks, not sentinels, say what a value means: ``slot_valid`` (a scan on the
query's axis, not padding), ``channel_valid`` (a channel of the query, not
padding), ``acquired`` (a cell the long view has: eligible, and inside its
scan's acquired m/z range or matched) and ``matched``. ``intensity`` is 0
and ``mz_error_da`` NaN wherever nothing matched, acquired or not; so a
zero is a measurement only where ``acquired`` says so. Hand a comparison
:meth:`Dense.observed` (NaN where not acquired), or ``intensity`` with
``mask=acquired``: passed bare, a never-measured cell counts as a measured
zero (review 2026-11-10, V1).

The view is built from the batch's matches, not from the long cells: a
query's scan axis is its eligible pairs (:meth:`QueryBatch.eligible`, the
map's own schedule), and each match is placed on it by one search, so the
cost is the matches' and the pairs', never every cell's twice. One value per
cell: a sweep under the ``all`` rule (several points a cell) is refused.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from constellation.massspec.sweep.schedule import ragged_arange

if TYPE_CHECKING:
    from constellation.massspec.sweep.reduce import QueryBatch

__all__ = [
    "Dense",
    "dense_view",
    "dense_chunks",
    "PARENT_FOUND",
    "PARENT_NONE",
    "PARENT_UNREAD",
    "PARENT_OFF_AXIS",
]

#: Why :meth:`Dense.parent_slots` gives a slot its parent or not (with
#: ``reasons=True``; -1 for a padding slot): found on the query's MS1 axis;
#: the scan has no parent recorded or derived; its parent scan is no slot
#: (the sweep did not read it, or it holds no points and the bundle does
#: not say it holds none; an empty scan read is a slot of zeros); its
#: parent is a slot but not on this query's MS1 axis (outside the query's
#: interval there).
PARENT_FOUND, PARENT_NONE, PARENT_UNREAD, PARENT_OFF_AXIS = 0, 1, 2, 3

#: The ``Channels`` columns a dense view carries per (query, channel), with
#: the value a padding channel gets.
_CHANNEL_COLUMNS = {
    "ion_type": -1,
    "position": -1,
    "fragment_charge": -1,
    "loss": -1,
    "isotope": -1,
    "precursor_charge": -1,
}


@dataclass(frozen=True)
class Dense:
    """One level of a batch's queries as ``[Q, S, C]`` tensors (module
    docstring): Q queries, each on its own axis of S scan slots (padded to
    the longest) and C channel slots (padded to the most)."""

    level: int
    #: The batch's query numbers (ascending), and their numbers in the whole plan.
    queries: np.ndarray
    global_queries: np.ndarray
    # [Q, S, C]
    intensity: torch.Tensor
    mz_error_da: torch.Tensor
    matched: torch.Tensor
    acquired: torch.Tensor
    # [Q, S]
    slot_valid: torch.Tensor
    row: torch.Tensor  # ledger row, -1 padding
    scan: torch.Tensor  # scan number, -1 padding
    rt: torch.Tensor  # s, NaN padding
    tic: torch.Tensor
    iit: torch.Tensor
    window: torch.Tensor  # the slot's isolation window (the ledger's window index), -1 padding
    parent_row: torch.Tensor  # the ledger row of the slot's parent scan, -1 none
    #: [Q, S]: whether the slot's injection time was estimated from its TIC
    #: (``Ledger.iit_imputed``); its scan's lowest stored intensity (NaN
    #: unknown; ``Ledger.floor``), and what made its peaks, a code into
    #: ``centroiders`` (-1 not recorded); and whether the scan records the
    #: m/z range it acquired (where not, every channel counts as acquired).
    iit_imputed: torch.Tensor
    floor: torch.Tensor
    centroider: torch.Tensor
    range_known: torch.Tensor
    #: [Q, S, C] int32: the matched point's position in its scan's stored
    #: order (``Matches.point_index``), -1 where nothing matched or a rule
    #: aggregated several points (``sum``). Two channels holding one point,
    #: or two queries sharing one, show here (review S10).
    point_index: torch.Tensor
    # [Q, C]
    channel_valid: torch.Tensor
    mz: torch.Tensor  # theoretical, NaN padding
    abundance: torch.Tensor  # NaN not modelled (and padding), 0 predicted absent
    ion_type: torch.Tensor
    position: torch.Tensor
    fragment_charge: torch.Tensor
    loss: torch.Tensor
    isotope: torch.Tensor
    precursor_charge: torch.Tensor
    #: [Q, C] int64: channels of a query whose tolerance windows overlap
    #: (chained in m/z order) share a group, named by its first channel;
    #: such channels can match the same point. -1 for padding.
    overlap_group: torch.Tensor
    #: Whether each ledger row can be a slot (``Ledger.present``: to tell an
    #: unread parent scan).
    _ledger_present: np.ndarray | None = field(default=None, repr=False, compare=False)
    #: The names ``centroider`` codes index.
    centroiders: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.intensity.shape)  # type: ignore[return-value]

    @property
    def n_cells(self) -> int:
        """The padded size, ``Q × S × C``."""
        q, s, c = self.shape
        return q * s * c

    @property
    def mz_error_ppm(self) -> torch.Tensor:
        """Observed − theoretical in ppm, NaN where unmatched."""
        return self.mz_error_da / self.mz.unsqueeze(1).to(self.mz_error_da.dtype) * 1e6

    @property
    def n_slots(self) -> torch.Tensor:
        """``[Q]``: each query's axis length (its valid slots)."""
        return self.slot_valid.sum(dim=1)

    def observed(self) -> torch.Tensor:
        """``intensity`` with NaN wherever the cell was not acquired (and in
        padding): a zero left is a measured zero. What to hand a
        comparison that drops NaN channels (module docstring)."""
        return torch.where(self.acquired, self.intensity, torch.full_like(self.intensity, float("nan")))

    def brackets(self, ms1: Dense):
        """For each slot of this view (MS2 or above), the slots of the
        query's MS1 axis just before and after it in time, and the linear
        weight between them (``core.signal.interpolate.Bracket``, slots
        ``[Q, S]`` on ``ms1``'s axis, -1 for none): the interpolation an
        MS1 trace needs to be read at MS2 times. ``ms1`` is the level below,
        built for the same queries."""
        from constellation.core.signal.interpolate import bracket

        self._check_below(ms1)
        rt = torch.where(self.slot_valid, self.rt, torch.full_like(self.rt, float("nan")))
        return bracket(ms1.rt, rt, valid=ms1.slot_valid)

    def _check_below(self, ms1: Dense) -> None:
        if not np.array_equal(self.queries, ms1.queries):
            raise ValueError("parent_slots needs the level below built for the same queries")
        if ms1.level != self.level - 1:
            raise ValueError(f"the parents of MS{self.level} scans are MS{self.level - 1} scans, not MS{ms1.level}")

    def parent_slots(self, ms1: Dense, *, reasons: bool = False):
        """``[Q, S]``: for each slot of this view (MS2 or above), the slot of
        its parent scan on the same query's axis in ``ms1`` (the level
        below, built for the same queries), -1 where it has none or the
        parent is not on that axis (outside the query's interval there).

        With ``reasons``, also ``[Q, S]`` codes saying why
        (:data:`PARENT_FOUND`, :data:`PARENT_NONE`, :data:`PARENT_UNREAD`,
        :data:`PARENT_OFF_AXIS`; -1 for padding): the four ways a slot has
        no parent are different evidence (review V8). The nearest MS1 slot
        on either side in time is :meth:`brackets`."""
        self._check_below(ms1)
        slots = self._parent_slots(ms1)
        if not reasons:
            return slots
        parent = self.parent_row.numpy()
        code = np.full(parent.shape, PARENT_OFF_AXIS, np.int64)
        code[slots.numpy() >= 0] = PARENT_FOUND
        code[parent < 0] = PARENT_NONE
        present = self._ledger_present
        if present is not None and present.size:
            unread = (parent >= 0) & ~present[np.maximum(parent, 0)]
            code[unread & (slots.numpy() < 0)] = PARENT_UNREAD
        code[~self.slot_valid.numpy()] = -1
        return slots, torch.from_numpy(code)

    def _parent_slots(self, ms1: Dense) -> torch.Tensor:
        q, s = self.parent_row.shape
        out = np.full((q, s), -1, np.int64)
        rows1 = ms1.row.numpy()
        if q == 0 or s == 0 or rows1.size == 0:
            return torch.from_numpy(out)
        span = int(max(rows1.max(), int(self.parent_row.max()), 0)) + 1
        qi1, si1 = np.nonzero(rows1 >= 0)
        key1 = qi1.astype(np.int64) * span + rows1[qi1, si1]
        by = np.argsort(key1, kind="stable")
        key1, si1 = key1[by], si1[by]
        parent = self.parent_row.numpy()
        qi, si = np.nonzero(parent >= 0)
        key = qi.astype(np.int64) * span + parent[qi, si]
        at = np.searchsorted(key1, key)
        at_c = np.minimum(at, max(key1.size - 1, 0))
        hit = (at < key1.size) & (key1[at_c] == key) if key1.size else np.zeros(key.size, bool)
        out[qi[hit], si[hit]] = si1[at_c[hit]]
        return torch.from_numpy(out)


def _overlap_groups(mz: np.ndarray, c_valid: np.ndarray, tolerance) -> np.ndarray:
    """``[Q, C]``: each channel's overlap group (``Dense.overlap_group``):
    channels taken in m/z order, a new group wherever a channel's window
    and the previous one's do not overlap; named by the group's first
    channel. -1 for padding."""
    n_q, n_c = mz.shape
    out = np.full((n_q, n_c), -1, np.int64)
    if n_q == 0 or n_c == 0:
        return out
    value, unit = tolerance
    key = np.where(c_valid, mz, np.inf)
    by = np.argsort(key, axis=1, kind="stable")
    smz = np.take_along_axis(key, by, axis=1)
    half = smz * float(value) * 1e-6 if unit == "ppm" else np.full_like(smz, float(value))
    ok = np.isfinite(smz)
    with np.errstate(invalid="ignore"):  # padding is inf, and inf − inf is NaN
        gap = np.diff(smz, axis=1)
        reach = half[:, 1:] + half[:, :-1]
    new = np.ones((n_q, n_c), bool)
    new[:, 1:] = ~(ok[:, 1:] & ok[:, :-1] & (gap <= reach))
    group = np.cumsum(new, axis=1) - 1  # per query, 0..g in m/z order
    # Name each group by its lowest channel index.
    first = np.full((n_q, n_c), n_c, np.int64)
    rows = np.repeat(np.arange(n_q), n_c)
    np.minimum.at(first, (rows, group.reshape(-1)), by.reshape(-1))
    named = np.take_along_axis(first, group, axis=1)
    np.put_along_axis(out, by, np.where(ok, named, -1), axis=1)
    return out


def dense_view(batch: QueryBatch, level: int, *, queries: np.ndarray | None = None,
               dtype: torch.dtype = torch.float32) -> Dense:  # fmt: skip
    """The dense view of ``batch``'s queries (or ``queries``, an ascending
    subset) at ``level`` (module docstring). ``dtype`` is that of
    ``intensity`` and ``mz_error_da`` (float32 or float64); times, TIC and
    m/z stay float64."""
    if dtype not in (torch.float32, torch.float64):
        # float16 turned every intensity above 65,504 into inf (review V5).
        raise ValueError(f"the dense view's values are float32 or float64, not {dtype}")
    plan, scans = batch.plan, batch.scans
    channels = plan.channels_for_level(level)
    if channels is None:
        raise ValueError(f"the plan has no channels at MS{level}")
    if batch.settings.rule.name == "all":
        raise ValueError(
            "the dense view holds one value per cell, and the 'all' rule gives a cell every point within "
            "tolerance: use the cell view (QueryBatch.cells), or sweep with 'nearest' or 'sum'"
        )
    qs = np.asarray(batch.queries if queries is None else queries, dtype=np.int64)
    if queries is not None and qs.size:
        # Unsorted queries raised an obscure error and a repeated one came
        # back empty (review V4).
        if qs.ndim != 1 or (qs.size > 1 and not (np.diff(qs) > 0).all()):
            raise ValueError("dense(queries=...) takes the batch's queries in strictly ascending order")
        at = np.searchsorted(batch.queries, qs)
        if (at >= batch.queries.size).any() or (batch.queries[np.minimum(at, batch.queries.size - 1)] != qs).any():
            raise ValueError("dense(queries=...) takes queries of the batch only")
    n_q = qs.size
    # The scan axis: each query's eligible pairs, in (query, scan order).
    pq_, pr = batch.eligible(level, qs)
    qpos = np.searchsorted(qs, pq_)
    per_query = np.bincount(qpos, minlength=n_q).astype(np.int64)
    n_s = int(per_query.max()) if n_q and per_query.size else 0
    slot = ragged_arange(np.zeros(n_q, np.int64), per_query)
    # The channel axis.
    counts = channels.counts()[qs] if n_q else np.zeros(0, np.int64)
    n_c = int(counts.max()) if n_q else 0
    first = channels.offsets[qs] if n_q else np.zeros(0, np.int64)
    c_idx = first[:, None] + np.arange(n_c)[None, :]
    c_valid = np.arange(n_c)[None, :] < counts[:, None]
    c_at = np.where(c_valid, c_idx, 0)

    def per_channel(values: np.ndarray, pad, dtype_) -> np.ndarray:
        if values.size == 0:
            return np.full((n_q, n_c), pad, dtype_)
        return np.where(c_valid, values[c_at], pad).astype(dtype_)

    mz = per_channel(channels.mz, np.nan, np.float64)
    abundance = per_channel(channels.abundance, np.nan, np.float32)

    # Each slot: its ledger row and what the ledger says of it.
    row = np.full((n_q, n_s), -1, np.int64)
    row[qpos, slot] = pr
    valid = row >= 0
    at = np.maximum(row, 0)

    def per_slot(values: np.ndarray, pad, dtype_) -> np.ndarray:
        if values.size == 0:
            return np.full((n_q, n_s), pad, dtype_)
        return np.where(valid, values[at], pad).astype(dtype_)

    index = scans.window_index(level)
    window_of_row = np.full(scans.n, -1, np.int64)
    window_of_row[index.rows] = index.window
    parent = per_slot(scans.parent, -1, np.int64)

    # Matches, laid onto (query, slot, channel). The batch's matches are
    # sorted by query, so each query's are one range, found by a search
    # (``np.isin`` over every match cost each chunk the whole batch's; V7).
    m = batch.matches
    sel = batch._match_rows(qs, level) if m is not None and m.n else np.empty(0, np.int64)
    # Made in the asked type at once: a float64 copy cast down would double
    # the view's peak memory.
    values = torch.empty(0, dtype=dtype).numpy().dtype
    intensity = np.zeros((n_q, n_s, n_c), values)
    error = np.full((n_q, n_s, n_c), np.nan, values)
    matched = np.zeros((n_q, n_s, n_c), bool)
    point = np.full((n_q, n_s, n_c), -1, np.int32)
    if sel.size:
        mq = np.searchsorted(qs, m.query[sel])
        mr = scans.rows_of(m.scan[sel])
        order = batch.scan_order
        span = int(order.max()) + 1 if order.size else 1
        pair_key = qpos.astype(np.int64) * span + order[pr]
        match_key = mq.astype(np.int64) * span + order[mr]
        p = np.searchsorted(pair_key, match_key)
        p_c = np.minimum(p, max(pair_key.size - 1, 0))
        if not ((p < pair_key.size) & (pair_key[p_c] == match_key)).all():
            raise AssertionError("a match outside its query's eligible scans")
        ms, mc = slot[p_c], m.channel[sel]
        cell = (mq.astype(np.int64) * n_s + ms) * n_c + mc
        # The matches come sorted by (query, scan order, channel), so their
        # cells ascend and a repeat is a zero step; ``np.unique`` was 74 %
        # of the build (review V2) and stays only for matches out of order.
        step = np.diff(cell)
        repeated = (step == 0).any() if (step >= 0).all() else np.unique(cell).size != cell.size
        if repeated:
            raise ValueError("several matches of one cell: the dense view holds one value per cell")
        intensity[mq, ms, mc] = m.intensity[sel]
        error[mq, ms, mc] = m.mz[sel] - mz[mq, mc]
        matched[mq, ms, mc] = True
        point[mq, ms, mc] = m.point_index[sel]

    # Acquired: the cells the long view has (eligible; inside the scan's
    # range, or matched, which proves it was measured).
    lo = per_slot(scans.scan_lower, np.nan, np.float64)[:, :, None]
    hi = per_slot(scans.scan_upper, np.nan, np.float64)[:, :, None]
    acquired = np.less(mz[:, None, :], lo)
    acquired |= np.greater(mz[:, None, :], hi)
    np.logical_not(acquired, out=acquired)
    acquired &= valid[:, :, None]
    acquired &= c_valid[:, None, :]
    if scans.gap_offsets is not None:
        # A scan of several ranges did not acquire between them.
        gapped = valid & (np.diff(scans.gap_offsets)[at] > 0)
        qi, si = np.nonzero(gapped)
        if qi.size:
            acquired[qi, si] &= ~scans.in_gap(row[qi, si][:, None], mz[qi])
    acquired |= matched

    def t(a: np.ndarray, dtype_: torch.dtype | None = None) -> torch.Tensor:
        out = torch.from_numpy(np.ascontiguousarray(a))
        return out if dtype_ is None else out.to(dtype_)

    return Dense(
        level=int(level),
        queries=qs,
        global_queries=plan.global_query(qs),
        intensity=t(intensity),
        mz_error_da=t(error),
        matched=t(matched),
        acquired=t(acquired),
        slot_valid=t(valid),
        row=t(row),
        scan=t(per_slot(scans.scan, -1, np.int64)),
        rt=t(per_slot(scans.rt, np.nan, np.float64)),
        tic=t(per_slot(scans.tic, np.nan, np.float64)),
        iit=t(per_slot(scans.iit, np.nan, np.float64)),
        window=t(per_slot(window_of_row, -1, np.int64)),
        parent_row=t(parent),
        iit_imputed=t(per_slot(scans.iit_imputed, False, bool)),
        floor=t(per_slot(scans.floor, np.nan, np.float64)),
        centroider=t(per_slot(scans.centroider.astype(np.int64), -1, np.int64)),
        range_known=t(per_slot(scans.range_known, False, bool)),
        point_index=t(point),
        channel_valid=t(c_valid),
        mz=t(mz),
        abundance=t(abundance),
        **{name: t(per_channel(getattr(channels, name), pad, np.int64)) for name, pad in _CHANNEL_COLUMNS.items()},
        overlap_group=t(_overlap_groups(mz, c_valid, batch.settings.tolerance(level))),
        _ledger_present=scans.present,
        centroiders=scans.centroiders,
    )


def dense_chunks(batch: QueryBatch, max_cells: int, levels: Sequence[int] | None = None) -> Iterator[np.ndarray]:
    """The batch's queries in contiguous ascending runs whose dense views,
    padded (``Q × S × C``) and summed over ``levels`` (the plan's, by
    default), hold at most ``max_cells`` cells each, at least one query a
    run: for a reducer to build its dense views a run at a time. Every
    query is in one run."""
    plan = batch.plan
    levels = [lv for lv in (plan.levels if levels is None else levels) if plan.channels_for_level(lv) is not None]
    qs = np.asarray(batch.queries, dtype=np.int64)
    if qs.size == 0:
        return
    scans_at = [batch.eligible_counts(lv) for lv in levels] if levels else []
    chans_at = [plan.channels_for_level(lv).counts()[qs] for lv in levels]
    start = 0
    widest = [(0, 0)] * len(levels)
    for i in range(qs.size):
        grown = [(max(s, int(sa[i])), max(c, int(ca[i]))) for (s, c), sa, ca in zip(widest, scans_at, chans_at)]
        size = (i - start + 1) * sum(s * c for s, c in grown)
        if i > start and size > max_cells:
            yield qs[start:i]
            start = i
            grown = [(int(sa[i]), int(ca[i])) for sa, ca in zip(scans_at, chans_at)]
        widest = grown
    yield qs[start:]
