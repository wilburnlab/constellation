"""Running a sweep: the map over scans and the reduce over queries.

The map streams the acquisition's scans in stored order, a block of whole
scans at a time (:func:`~constellation.massspec.bundle.iter_scan_blocks`).
For each level in a block it finds the active queries, groups the scans
by isolation window into window groups, matches each group's points
against its queries' channels with whichever kernel is cheaper for it,
and keeps the matches whose cell is eligible (its scan inside the query's
retention-time interval, and the assigned scan for fragments). It also
records every scan it read, in the ledger.

The reduce then sorts the matches by (query, level, scan order, channel,
point), where a scan's order is its rank by (retention time, scan
number), and hands them to each reducer as a :class:`~constellation.massspec.sweep.reduce.QueryBatch`.

A source is a converted acquisition (a bundle directory) or an in-memory
table of points (a peak list's rows). A table is sorted by (rt, scan,
m/z) and its scans' retention time and isolation bounds are taken from
each scan's first row, as the scan-major extractor took them. Terms
(point, peak, query, channel, cell, match) as in ``docs/plans/sweep.md``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core.progress import ProgressCallback, emit_done, emit_progress, emit_start
from constellation.massspec.bundle import (
    ScanBlock,
    iter_scan_blocks,
    read_bundle_manifest,
    scan_range_shards,
)
from constellation.massspec.sweep.claims import Claims, claims_table, find_claims
from constellation.massspec.sweep.evidence import Evidence, EvidenceGate, count_evidence, decide
from constellation.massspec.sweep.kernel import match_group
from constellation.massspec.sweep.queries import QueryPlan
from constellation.massspec.sweep.reduce import Ledger, Matches, QueryBatch, Reducer, eligible_counts
from constellation.massspec.sweep.rules import AllWithin, MatchRule, Nearest
from constellation.massspec.sweep.schemas import SWEEP_EVIDENCE_TABLE, SWEEP_POINT_CLAIM_TABLE
from constellation.massspec.sweep.schedule import (
    active_queries,
    isolation_groups,
    order_by,
    pair_eligible,
    ragged_arange,
)

__all__ = ["SweepSettings", "SweepResult", "sweep", "CLAIMS", "LAYOUTS"]

_LOG = logging.getLogger(__name__)

#: Bytes a spilled row (a match, an evidence count, a claim) takes on disk,
#: compressed, for the free-space check before a spill: about what a run
#: file of matches measures (eight columns, two of them float64).
SPILL_BYTES_PER_ROW = 48

#: How a sweep reads a bundle: ``scan_major`` (its scan-ordered peaks in
#: scan ranges, every match written and shuffled to the reducers) or
#: ``precursor_major`` (its precursor-major layout a unit of queries at a
#: time, reduced where read; ``sweep.precursor_major``).
LAYOUTS = ("scan_major", "precursor_major")

#: The name of the claims output, beside the reducers'.
CLAIMS = "point_claims"

Tolerance = tuple[float, Literal["ppm", "Da"]]


@dataclass(frozen=True)
class SweepSettings:
    """How a sweep matches."""

    #: Tolerance per MS level; a level above 2 without its own uses level 2's.
    tolerances: Mapping[int, Tolerance]
    rule: MatchRule = Nearest()
    #: Count evidence and decide which queries gated reducers see.
    gate: EvidenceGate | None = None
    #: Record points claimed by more than one cell (the ``point_claims`` output).
    claims: bool = False
    #: ``channel`` (channel-driven), ``point`` (point-driven) or ``auto``:
    #: point-driven once a group's channels exceed ``crossover`` times its
    #: scans' mean points. Measured on centroids (docs/plans/sweep.md):
    #: point-driven is as fast or faster from a quarter of a channel per
    #: point up, 2–11× at proteome DIA scale.
    kernel: Literal["auto", "channel", "point"] = "auto"
    crossover: float = 0.25
    #: A measured m/z axis of the bundle to correct observed m/z with.
    mz_axis: str | None = None
    acquisition_id: int = 0
    block_rows: int = 1 << 20
    #: Padded cells and points per call of the channel-driven kernel.
    kernel_elements: int = 1 << 21
    #: Points per map shard when the sweep runs on disk.
    shard_rows: int = 16 << 20
    #: Rows a reduce bucket builds (``_reduce_weights``: matches, gate pairs,
    #: and every eligible cell for a reducer that ``zero_fills``), on disk and
    #: in memory alike.
    reduce_rows: int = 1 << 22
    #: Matches a map shard holds before writing them out as a sorted run.
    run_rows: int = 4 << 20
    #: Rows per row group of a sorted run. A bucket reads, from every run,
    #: the row groups its query range overlaps: whole 2^18-row groups made
    #: each bucket decode far more than its own rows, growing with the
    #: square of the matches (an estimated 8-15x at 476M; review of
    #: 2026-11-06).
    run_row_group_rows: int = 1 << 15
    #: Rows (matches, evidence counts and claims) a sweep in memory holds
    #: before it carries on on disk, tested after every window group. A held
    #: match costs some 200–500 bytes at peak (sorting, the reducers' tables),
    #: so this keeps an in-memory sweep near a disk worker's 3 GB. Until
    #: 2026-11-10 it was tested after a whole block of points and counted
    #: matches alone (a GPF fraction makes about 61 matches a point).
    memory_rows: int = 8 << 20
    #: Eligible cells per unit of a precursor-major sweep: a run of queries
    #: in precursor m/z order, read and reduced in one task. Bounds the
    #: queries a task holds; what it reads is the scans they can match in.
    unit_cells: int = 1 << 28
    #: Eligible cells per batch inside a unit: what bounds a task's matches
    #: and the reducers' input at once.
    batch_cells: int = 1 << 22
    #: In a precursor-major sweep whose reducers are all gated, count each
    #: batch's evidence and decide the gate first, then match only the
    #: queries it passed (the two-pass gate, in one pass over the points it
    #: has in memory). The outputs are the same either way.
    gate_first: bool = True

    def tolerance(self, level: int) -> Tolerance:
        if level in self.tolerances:
            return self.tolerances[level]
        if level > 2 and 2 in self.tolerances:
            return self.tolerances[2]
        raise ValueError(f"no tolerance for MS{level}: given {dict(self.tolerances)}")


@dataclass
class SweepResult:
    """What a sweep found: the scans it read, its matches in reduce order
    (in memory only; on disk they stay in the work directory), and each
    reducer's output by name: a table, or on disk its parts in query order."""

    scans: Ledger
    matches: Matches | None
    tables: dict[str, pa.Table] = field(default_factory=dict)
    parts: dict[str, list[Path]] = field(default_factory=dict)
    schemas: dict[str, pa.Schema] = field(default_factory=dict)
    mz_axis: str | None = None
    #: Schema metadata of an output beyond its schema's own (such as the
    #: tolerances a trace was extracted at), carried by ``table`` and ``write``.
    metadata: dict[str, dict[bytes, bytes]] = field(default_factory=dict)
    #: The directory this sweep wrote, its own (under a ``work_dir``, a
    #: ``sweep-<id>/`` that no later sweep touches); None in memory.
    work: Path | None = None

    def table(self, name: str) -> pa.Table:
        """A reducer's whole output, read into memory from its parts if the
        sweep ran on disk: a terminal table, for a caller to keep."""
        if name not in self.tables:
            files = self.parts[name]
            schema = self.schemas[name]
            self.tables[name] = (
                pq.read_table(files[0]) if len(files) == 1
                else pa.concat_tables([pq.read_table(f) for f in files])
                if files else schema.empty_table()
            )  # fmt: skip
        table = self.tables[name]
        if self.metadata.get(name):
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), **self.metadata[name]})
        return table

    def write(self, name: str, path: Path | str, *, metadata: Mapping[bytes, bytes] | None = None) -> Path:
        """Write a reducer's output to one Parquet file, part by part, with
        the result's metadata for it and then ``metadata`` added to its
        schema's."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if name in self.tables or not self.parts.get(name):
            table = self.table(name)
            if metadata:
                table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
            pq.write_table(table, out)
            return out
        schema = self.schemas[name]
        schema = schema.with_metadata(
            {**(schema.metadata or {}), **self.metadata.get(name, {}), **(metadata or {})}
        )
        tmp = out.with_name(f".{out.name}.tmp")
        with pq.ParquetWriter(tmp, schema) as writer:
            for f in self.parts[name]:
                for batch in pq.ParquetFile(f).iter_batches():
                    # By name, not position.
                    writer.write_batch(
                        pa.RecordBatch.from_arrays([batch.column(c.name).cast(c.type) for c in schema], schema=schema)
                    )
        tmp.replace(out)
        return out


# ── blocks of scans ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _Block:
    scan: np.ndarray  # int64, per scan
    level: np.ndarray  # int8
    rt: np.ndarray  # float64
    lower: np.ndarray  # float64, NaN without bounds
    upper: np.ndarray
    offsets: np.ndarray  # (n_scans + 1,)
    mz: np.ndarray  # per point, sorted within each scan
    intensity: np.ndarray
    point_index: np.ndarray  # each point's position in its scan's stored order


def _f64(col: pa.ChunkedArray) -> np.ndarray:
    return col.to_numpy(zero_copy_only=False).astype(np.float64)


#: Scan metadata the ledger carries beyond its own columns, when a bundle has them.
_SCAN_CONTEXT = ("tic", "iit", "scan_window_lower", "scan_window_upper", "precursor_scan", "faims_cv")


def _refuse_several_voltages(where: str, cv: np.ndarray | None) -> None:
    """A run of several FAIMS compensation voltages is refused: the sweep
    keys a scan's window by its isolation bounds alone, so both voltages'
    scans would be one window, and a peptide transmitted at one shows
    evidence in every other scan of it (the gate passed 0 of 6 planted
    peptides with two voltages, 6 of 6 with one; review of 2026-11-09, A1).
    Keying windows by voltage would not be enough: a query stays eligible
    at both, and its evidence runs would still interleave."""
    if cv is None:
        return
    seen = np.unique(cv[~np.isnan(cv)])
    if seen.size > 1:
        raise ValueError(
            f"{where} is a multi-CV FAIMS run (compensation voltages {', '.join(f'{v:g}' for v in seen)}); "
            f"the sweep does not separate voltages yet. Split it by CV first."
        )


def _refuse_multiplexed(bundle: Path, path: Path) -> None:
    """A run with multiplexed isolation (MSX: several precursors isolated
    for one scan) is refused: a scan's window is one isolation pair, so the
    reader keeps its first reaction's, and the queries in its others would
    never be matched there (review of 2026-11-09, A4)."""
    if "filter_string" not in pq.read_schema(path).names:
        return
    filters = pq.read_table(path, columns=["filter_string"]).column("filter_string")
    msx = pc.fill_null(pc.match_substring_regex(filters, r"(^|\s)msx(\s|$)"), False)
    if pc.any(msx).as_py():
        raise ValueError(
            f"{bundle} has {pc.sum(msx).as_py():,} multiplexed (MSX) scans, each isolating several precursors; "
            f"the sweep holds one isolation window a scan, so they cannot be swept yet"
        )


def _scan_metadata(bundle: Path) -> Ledger:
    path = bundle / "scan_metadata.parquet"
    context = [c for c in _SCAN_CONTEXT if c in pq.read_schema(path).names]
    t = pq.read_table(path, columns=["scan", "level", "rt", "isolation_lower", "isolation_upper", *context]).sort_by("scan")
    n = t.num_rows

    def opt(name):
        return _f64(t.column(name).cast(pa.float64()).fill_null(np.nan)) if name in context else None

    cv = opt("faims_cv")
    _refuse_several_voltages(str(bundle), cv)
    _refuse_multiplexed(bundle, path)
    scan = t.column("scan").to_numpy().astype(np.int64)
    level = t.column("level").to_numpy().astype(np.int8)
    ledger = Ledger(
        scan=scan,
        level=level,
        rt=_f64(t.column("rt")),
        lower=_f64(t.column("isolation_lower")),
        upper=_f64(t.column("isolation_upper")),
        n_points=np.zeros(n, dtype=np.int64),
        tic=opt("tic"),
        iit=opt("iit"),
        scan_lower=opt("scan_window_lower"),
        scan_upper=opt("scan_window_upper"),
        parent=_parent_rows(scan, level, opt("precursor_scan"), cv),
    )
    return ledger


def _parent_rows(scan: np.ndarray, level: np.ndarray, recorded, cv) -> np.ndarray:
    """Each scan's parent (``parent_scans``) as a row of a ledger of
    ``scan`` (in scan order), -1 for none or one not in it."""
    from constellation.massspec.acquisition_structure import parent_scans

    parent = parent_scans(scan, level, recorded, cv)
    pos = np.searchsorted(scan, parent)
    pos_c = np.minimum(pos, max(scan.size - 1, 0))
    found = (parent >= 0) & (pos < scan.size) & (scan[pos_c] == parent) if scan.size else parent >= 0
    return np.where(found, pos_c, -1).astype(np.int64)


def _sorted_within_scans(offsets: np.ndarray, mz: np.ndarray) -> np.ndarray | None:
    """A permutation that sorts each scan's points by m/z (stably), or None
    when they already are."""
    if mz.size < 2:
        return None
    down = np.flatnonzero(np.diff(mz) < 0)
    if down.size == 0:
        return None
    starts = offsets[1:-1]
    # A drop across a scan boundary is not disorder.
    inner = down[~np.isin(down + 1, starts)]
    if inner.size == 0:
        return None
    row = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
    return np.lexsort((mz, row))


def _bundle_blocks(
    bundle: Path, meta: Ledger, settings: SweepSettings, axis, *, blocks: Iterable[ScanBlock] | None = None,
    levels: Sequence[int] | None = None,
) -> Iterator[_Block]:
    if blocks is None:
        blocks = iter_scan_blocks(bundle, columns=("mz", "intensity"), block_rows=settings.block_rows)
    for b in blocks:
        at = np.searchsorted(meta.scan, b.scan)
        at_c = np.minimum(at, meta.n - 1)
        missing = meta.scan[at_c] != b.scan
        if missing.any():
            raise ValueError(
                f"{bundle}: points of scan {int(b.scan[missing][0])} have no row in scan_metadata.parquet"
            )
        counts = np.diff(b.offsets)
        meta.n_points[at] = counts
        yield _make_block(meta, at, b.scan, b.offsets, b.columns["mz"], b.columns["intensity"],
                          ragged_arange(np.zeros(b.n_scans, np.int64), counts), axis, levels)  # fmt: skip


def _make_block(meta: Ledger, at: np.ndarray, scan: np.ndarray, offsets: np.ndarray, mz: np.ndarray,
                intensity: np.ndarray, point_index: np.ndarray, axis,
                levels: Sequence[int] | None = None) -> _Block:  # fmt: skip
    """A block of whole scans (ledger rows ``at``), each scan's points in
    stored order: the m/z axis corrected and each scan's points sorted by
    the corrected m/z, each keeping its stored position. Only ``levels``
    (all, when not given) are corrected, the levels a plan matches: the
    others are never matched, and correcting them refused a sweep of MS2
    whose curve could not be applied at MS1 (Codex, 2026-11-10)."""
    mz = mz.astype(np.float64, copy=False)
    intensity = intensity.astype(np.float64, copy=False)
    level = meta.level[at]
    if axis is not None:
        point_level = np.repeat(level, np.diff(offsets))
        mz = mz.copy()
        for lvl in np.unique(level):
            if levels is not None and int(lvl) not in levels:
                continue
            sel = point_level == lvl
            mz[sel] = axis.corrected(int(lvl), mz[sel])
    perm = _sorted_within_scans(offsets, mz)
    if perm is not None:
        mz, intensity, point_index = mz[perm], intensity[perm], point_index[perm]
    return _Block(
        scan=scan,
        level=level,
        rt=meta.rt[at],
        lower=meta.lower[at],
        upper=meta.upper[at],
        offsets=offsets,
        mz=mz,
        intensity=intensity,
        point_index=point_index,
    )


def _table_blocks(table: pa.Table, block_rows: int, levels: Sequence[int]) -> tuple[list[_Block], Ledger]:
    names = set(table.column_names)
    # Only the levels matched are sorted: a scan of another level is never read.
    t = table.filter(pc.is_in(table.column("level"), pa.array(list(levels), table.schema.field("level").type)))
    t = t.sort_by([("rt", "ascending"), ("scan", "ascending"), ("mz", "ascending")])
    n = t.num_rows
    if n == 0:
        empty = np.empty(0)
        return [], Ledger(empty.astype(np.int64), empty.astype(np.int8), empty, empty, empty,
                          empty.astype(np.int64))  # fmt: skip
    scan = t.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(scan[1:] != scan[:-1]) + 1]
    offsets = np.r_[starts, n].astype(np.int64)
    nan = np.full(n, np.nan)
    lower = _f64(t.column("isolation_lower")) if "isolation_lower" in names else nan
    upper = _f64(t.column("isolation_upper")) if "isolation_upper" in names else nan
    cv = _f64(t.column("faims_cv").cast(pa.float64()).fill_null(np.nan))[starts] if "faims_cv" in names else None
    _refuse_several_voltages("the table of points", cv)
    level = t.column("level").to_numpy(zero_copy_only=False).astype(np.int8)[starts]
    # The table's scans are in time order; the parent rule wants scan order.
    by_scan = np.argsort(scan[starts], kind="stable")
    parent = np.full(starts.size, -1, np.int64)
    rows = _parent_rows(scan[starts][by_scan], level[by_scan], None, None if cv is None else cv[by_scan])
    parent[by_scan] = np.where(rows >= 0, by_scan[np.maximum(rows, 0)], -1)
    ledger = Ledger(
        scan=scan[starts],
        level=level,
        rt=_f64(t.column("rt"))[starts],
        lower=lower[starts],
        upper=upper[starts],
        n_points=np.diff(offsets),
        parent=parent,
    )
    mz = _f64(t.column("mz"))
    intensity = _f64(t.column("intensity"))
    point_index = ragged_arange(np.zeros(starts.size, np.int64), np.diff(offsets))
    blocks = []
    s0 = 0
    while s0 < starts.size:
        s1 = int(np.searchsorted(offsets, offsets[s0] + block_rows, side="left"))
        s1 = min(max(s1, s0 + 1), starts.size)
        a, z = offsets[s0], offsets[s1]
        blocks.append(
            _Block(
                scan=ledger.scan[s0:s1],
                level=ledger.level[s0:s1],
                rt=ledger.rt[s0:s1],
                lower=ledger.lower[s0:s1],
                upper=ledger.upper[s0:s1],
                offsets=offsets[s0 : s1 + 1] - a,
                mz=mz[a:z],
                intensity=intensity[a:z],
                point_index=point_index[a:z],
            )
        )
        s0 = s1
    return blocks, ledger


# ── the map ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _MapOut:
    matches: Matches
    evidence: Evidence
    claims: Claims
    #: The marker that follows a block's last window group (an empty part).
    ends_block: bool = False

    @classmethod
    def concat(cls, parts: list[_MapOut]) -> _MapOut:
        return cls(
            Matches.concat([p.matches for p in parts]),
            Evidence.concat([p.evidence for p in parts]),
            Claims.concat([p.claims for p in parts]),
        )

    @property
    def rows(self) -> int:
        """What it holds, in rows: matches, evidence counts and claims."""
        return self.matches.n + self.evidence.n + self.claims.n


def _map_block(block: _Block, plan: QueryPlan, settings: SweepSettings, *, matches: bool = True,
               evidence: bool = True, only: np.ndarray | None = None) -> _MapOut:  # fmt: skip
    """A block's matches, evidence and claims, all its window groups at once."""
    return _MapOut.concat(list(_map_block_parts(block, plan, settings, matches=matches, evidence=evidence,
                                                only=only)))  # fmt: skip


def _map_parts(blocks: Iterable[_Block], plan: QueryPlan, settings: SweepSettings) -> Iterator[_MapOut]:
    """Every block's matches, evidence and claims, a window group at a time:
    one stream, so a sweep that has to stop holding them part way through a
    block hands the rest of it on (to the disk map) instead of finishing the
    block in memory first."""
    for block in blocks:
        yield from _map_block_parts(block, plan, settings)
        yield _BLOCK_END


_BLOCK_END = _MapOut(Matches.concat([]), Evidence.concat([]), Claims.empty(), ends_block=True)


def _map_block_parts(block: _Block, plan: QueryPlan, settings: SweepSettings, *, matches: bool = True,
                     evidence: bool = True, only: np.ndarray | None = None) -> Iterator[_MapOut]:  # fmt: skip
    """A block's matches, evidence and claims one window group at a time, so
    a caller can write them out before the next group's are made: a block
    is sized in points, and at proteome scale its matches (61 per point on
    a GPF run) outgrew any run size until 2026-11-09. ``matches`` and
    ``evidence`` say which to make (claims come with matches); ``only`` (a
    mask over the plan's queries) matches those queries alone, as when the
    gate has already been decided."""
    gate = settings.gate if evidence else None
    want_matches = matches
    for level in plan.levels:
        channels = plan.channels_for_level(level)
        if channels is None or channels.n == 0:
            continue
        rows = np.flatnonzero(block.level == level)
        timed = rows[~np.isnan(block.rt[rows])]
        if timed.size == 0:
            # A scan without a retention time is never eligible
            # (pair_eligible); one of them used to make min/max NaN and
            # leave every scan of the level in the block unmatched.
            continue
        tolerance, unit = settings.tolerance(level)
        counts = channels.counts()
        active = active_queries(
            plan.rt_lo, plan.rt_hi, counts > 0 if only is None else (counts > 0) & only,
            float(block.rt[timed].min()), float(block.rt[timed].max()),
        )  # fmt: skip
        groups = isolation_groups(
            active, plan.precursor_mz, block.lower[rows], block.upper[rows], rows, gate=level > 1,
            order=plan.precursor_order if level > 1 else None,
        )
        assigned = plan.assigned_scan if level > 1 else None

        def eligible(query: np.ndarray, row: np.ndarray) -> np.ndarray:
            return pair_eligible(query, block.rt[row], block.scan[row], plan.rt_lo, plan.rt_hi, assigned)

        def stored(row: np.ndarray, point: np.ndarray) -> np.ndarray:
            at = block.offsets[row] + np.maximum(point, 0)
            return np.where(point >= 0, block.point_index[at], -1)

        for g in groups:
            matches_out, evidence_out, claims = [], [], []
            n = counts[g.queries]
            channel_at = ragged_arange(channels.offsets[g.queries], n)
            channel_query = np.repeat(g.queries, n)
            channel_mz = channels.mz[channel_at]
            kw = dict(kernel=settings.kernel, crossover=settings.crossover,
                      max_elements=settings.kernel_elements)  # fmt: skip
            m = (match_group(block.mz, block.intensity, block.offsets, g.rows, channel_mz, settings.rule,
                             tolerance, unit, **kw) if want_matches else None)  # fmt: skip
            if m is not None and m.n:
                row = g.rows[m.row]
                query = channel_query[m.channel]
                ok = eligible(query, row)
                if ok.any():
                    row, query, sel = row[ok], query[ok], np.flatnonzero(ok)
                    channel = channel_at[m.channel[sel]] - channels.offsets[query]
                    point = stored(row, m.point[sel])
                    matches_out.append(
                        Matches(query=query, level=np.full(sel.size, level, np.int8),
                                scan=block.scan[row], channel=channel, point_index=point, mz=m.mz[sel],
                                intensity=m.intensity[sel], n_points=m.n_points[sel])  # fmt: skip
                    )
                    if settings.claims and settings.rule.name != "sum":
                        claims.append(find_claims(level, block.scan[row], point, query, channel,
                                                  m.mz[sel], m.intensity[sel]))  # fmt: skip
            if want_matches and settings.claims and settings.rule.name == "sum":
                # A sum is one match per window; its claims are the window's points.
                a = match_group(block.mz, block.intensity, block.offsets, g.rows, channel_mz,
                                AllWithin(), tolerance, unit, **kw)  # fmt: skip
                row = g.rows[a.row]
                query = channel_query[a.channel]
                ok = eligible(query, row)
                row, query, sel = row[ok], query[ok], np.flatnonzero(ok)
                claims.append(find_claims(level, block.scan[row], stored(row, a.point[sel]), query,
                                          channel_at[a.channel[sel]] - channels.offsets[query], a.mz[sel],
                                          a.intensity[sel]))  # fmt: skip
            if gate is not None and (level > 1 or gate.min_isotopes is not None):
                evidence_out.append(
                    count_evidence(block.mz, block.intensity, block.offsets, g.rows, block.scan, channels,
                                   channel_at, channel_query, eligible, tolerance, unit, gate.top_peaks,
                                   by_charge=level == 1)  # fmt: skip
                )
            yield _MapOut(Matches.concat(matches_out), Evidence.concat(evidence_out), Claims.concat(claims))


def _load_axis(bundle: Path, settings: SweepSettings):
    if settings.mz_axis is None:
        return None
    from constellation.massspec.mz_axis import load_mz_axis

    axis = load_mz_axis(bundle, settings.mz_axis)
    if axis is None:
        raise ValueError(f"{bundle.name} has no measured m/z axis to apply")
    return axis


def _fragment_level(plan: QueryPlan) -> int | None:
    return next((lv for lv in plan.levels if lv > 1), None)


def _batch(plan, queries, out: _MapOut, ledger, order, settings, shared, *, presorted: bool = False,
           sized_for_cells: bool = True) -> QueryBatch:  # fmt: skip
    """A bucket's batch: matches sorted for reducers, and the gate decided.
    ``presorted``: the matches and evidence already are (a slice of a
    sweep's sorted whole), so they are not sorted again.
    ``sized_for_cells``: whether the bucket was sized by its cells
    (``QueryBatch.sized_for_cells``)."""
    if presorted:
        matches, evidence = out.matches, out.evidence
    else:
        matches = out.matches.sorted(order[ledger.rows_of(out.matches.scan)])
        evidence = out.evidence.take(order_by(out.evidence.query, out.evidence.scan))
    batch = QueryBatch(plan=plan, queries=queries, matches=matches, scans=ledger, scan_order=order,
                       settings=settings, evidence=evidence, shared=shared,
                       sized_for_cells=sized_for_cells)  # fmt: skip
    if settings.gate is None:
        return batch
    frag = _fragment_level(plan)
    has = (
        plan.fragments.counts()[queries] > 0
        if plan.fragments is not None and frag is not None
        else np.zeros(queries.size, dtype=bool)
    )
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    # Only a query with evidence has a run to find; the rest decide on
    # nothing, so their pairs are not built (a global query's are every scan).
    with_fragments = np.unique(evidence.query[evidence.group == 0])
    with_isotopes = np.unique(evidence.query[evidence.group != 0])
    decision = decide(
        settings.gate, queries, has,
        (lambda: batch.eligible(frag, with_fragments)) if frag is not None else (lambda: empty),
        (lambda: batch.eligible(1, with_isotopes)) if 1 in plan.levels else (lambda: empty),
        evidence, ledger,
    )  # fmt: skip
    return replace(batch, decision=decision)


def _conform(table: pa.Table, schema: pa.Schema, who: str) -> pa.Table:
    """``table`` as ``schema``: the same column names, in its order and
    types. A missing or extra column is the reducer's mistake and is
    refused, naming it; columns are matched by name, never by position
    (a reducer returning two columns in another order had them written
    swapped until 2026-11-07)."""
    names, want = set(table.column_names), set(schema.names)
    if names != want:
        missing, extra = sorted(want - names), sorted(names - want)
        raise ValueError(
            f"reducer {who!r} returned a table that is not its schema"
            + (f"; missing {missing}" if missing else "")
            + (f"; not in its schema {extra}" if extra else "")
        )
    try:
        return pa.table([table.column(f.name).cast(f.type) for f in schema], schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise ValueError(f"reducer {who!r} returned columns its schema does not take: {exc}") from exc


def outputs_of(reducer: Reducer) -> dict[str, pa.Schema]:
    """A reducer's outputs, by name: its ``outputs``, or the one named
    ``name`` of ``schema`` (a reducer of one table)."""
    outputs = getattr(reducer, "outputs", None)
    if outputs is not None:
        return dict(outputs)
    return {reducer.name: reducer.schema}


def _reduce(batch: QueryBatch, reducers: Sequence[Reducer]) -> list[dict[str, pa.Table]]:
    """Each reducer's output tables by name, each held to its schema."""
    passing = batch.passing() if any(r.gated for r in reducers) else None
    out = []
    for r in reducers:
        got = r.reduce(passing if r.gated else batch)
        schemas = outputs_of(r)
        if isinstance(got, pa.Table):
            if len(schemas) != 1:
                raise ValueError(f"reducer {r.name!r} declares outputs {sorted(schemas)} but returned one table")
            got = {next(iter(schemas)): got}
        if set(got) != set(schemas):
            raise ValueError(f"reducer {r.name!r} returned outputs {sorted(got)}, not its {sorted(schemas)}")
        out.append({name: _conform(got[name], schemas[name], f"{r.name}:{name}" if name != r.name else name)
                    for name in schemas})  # fmt: skip
    return out


def _check_outputs(reducers: Sequence[Reducer], settings: SweepSettings) -> None:
    """Output names are unique across reducers, and not the claims'."""
    seen: dict[str, str] = {}
    for r in reducers:
        for name in outputs_of(r):
            if name == CLAIMS and settings.claims:
                raise ValueError(f"reducer {r.name!r} names an output {CLAIMS!r}, the claims' own")
            if name in seen:
                raise ValueError(f"reducers {seen[name]!r} and {r.name!r} both name an output {name!r}")
            seen[name] = r.name


def _check_gate(plan: QueryPlan, settings: SweepSettings) -> None:
    """Refuse a gate no query could pass: the gate counts distinct fragment
    ions, so a plan keeping fewer than ``min_fragments`` channels per query
    passes nothing (until 2026-11-09 a top 5 passed 0 of 1,756 targets
    with no word)."""
    gate = settings.gate
    top = getattr(plan.settings, "top_fragments", None)
    if gate is None or top is None or not any(level > 1 for level in plan.levels):
        return
    if top < gate.min_fragments:
        raise ValueError(
            f"the plan keeps the top {top} fragments of each query, fewer than the gate's min_fragments "
            f"{gate.min_fragments}: no query could pass"
        )


def sweep(
    source: Path | str | pa.Table,
    plan: QueryPlan,
    settings: SweepSettings,
    *,
    reducers: Sequence[Reducer] = (),
    workers: int = 1,
    work_dir: Path | str | None = None,
    progress: ProgressCallback | None = None,
    scan_range: tuple[int, int] | None = None,
    rt_range: tuple[float, float] | None = None,
    layout: str = "scan_major",
    spill_dir: Path | str | None = None,
) -> SweepResult:
    """Sweep ``source`` for ``plan``'s queries and reduce the matches.

    In memory unless ``workers > 1`` or a ``work_dir`` is given, or what the
    map holds outgrows ``settings.memory_rows`` (tested after every window
    group). In memory the queries are still reduced in buckets of about
    ``settings.reduce_rows`` built rows, as on disk; the outputs are held.
    On disk the bundle is cut into
    scan ranges (from its footers, so the cut does not depend on
    ``workers``), each range's matches are written to ``work_dir`` in sorted
    runs of at most ``settings.run_rows``, and each bucket of queries is
    reduced from them, ``workers`` processes at a time for both. The outputs
    are the same tables either way, and the same bytes whatever the number
    of workers; a file written from a sweep on disk keeps its parts' row
    groups, so it differs from an in-memory sweep's file in those alone.
    Each sweep writes a directory of its own, ``work_dir/sweep-<id>/``
    (``SweepResult.work``), which no later sweep into the same
    ``work_dir`` removes or rewrites; it is the caller's to remove. Without
    a ``work_dir`` a temporary one is made in ``spill_dir`` (the system's
    temporary directory by default), kept while the result is, so its
    outputs are read as asked; a spill the directory has no room for is
    refused before it starts. With ``settings.claims`` the
    claims table is the ``point_claims`` output. A reducer gives one output
    (``name``, ``schema``) or several (``outputs``, a name → schema mapping,
    with ``reduce`` returning a name → table mapping); output names are
    unique across reducers.

    ``scan_range=(first, last)`` or ``rt_range=(t0, t1)`` (inclusive) sweeps
    part of the run: only those scans are read, so only they are eligible
    (zero-filled, gated, counted); an RT range reads the scans between the
    first and last whose time falls in it. The range is recorded in every
    output's metadata (``x.massspec.sweep.scan_range``).

    ``layout='precursor_major'`` reads the bundle's precursor-major layout
    a unit of queries at a time and reduces each unit where it is read
    (``sweep.precursor_major``): no match is written, so a search's sweep
    costs its reducers' rows and not its matches. A query's rows are the
    ones the scan-major sweep gives it; an output's parts come in units of
    queries in precursor m/z order (``x.massspec.sweep.layout`` in its
    metadata), and the in-memory matches are not kept (``matches`` None).
    """
    _check_outputs(reducers, settings)
    _check_gate(plan, settings)
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
    if scan_range is not None and rt_range is not None:
        raise ValueError("give a scan_range or an rt_range, not both")
    window = _window(source, scan_range, rt_range)
    if settings.mz_axis is not None and not isinstance(source, pa.Table):
        # A curve that cannot be applied at a level the plan matches is refused
        # here, before any work, not from inside a worker.
        axis = _load_axis(Path(source), settings)
        for level in plan.levels:
            axis.check_level(level)
    if layout == "precursor_major":
        from constellation.massspec.sweep.precursor_major import sweep_precursor_major

        result = sweep_precursor_major(source, plan, settings, reducers, workers, work_dir, progress, window,
                                       spill_dir=spill_dir)  # fmt: skip
    else:
        result = _sweep(source, plan, settings, reducers, workers, work_dir, progress, window, spill_dir=spill_dir)
    if window is not None:
        stamp = {b"x.massspec.sweep.scan_range": f"{window[0]},{window[1] - 1}".encode()}
        if rt_range is not None:
            stamp[b"x.massspec.sweep.rt_range"] = f"{float(rt_range[0])!r},{float(rt_range[1])!r}".encode()
        for name in result.schemas:
            result.metadata[name] = {**result.metadata.get(name, {}), **stamp}
    return result


def _window(source, scan_range, rt_range) -> tuple[int, int] | None:
    """The half-open scan range ``[lo, hi)`` a sweep reads, or None for all."""
    if scan_range is None and rt_range is None:
        return None
    if scan_range is not None:
        first, last = (int(x) for x in scan_range)
        if last < first:
            raise ValueError(f"scan_range must be (first, last) with first <= last; got {scan_range!r}")
        return first, last + 1
    t0, t1 = (float(x) for x in rt_range)
    if t1 < t0:
        raise ValueError(f"rt_range must be (start, end) with start <= end; got {rt_range!r}")
    if isinstance(source, pa.Table):
        scan, rt = source.column("scan").to_numpy(), _f64(source.column("rt"))
    else:
        meta = pq.read_table(Path(source) / "scan_metadata.parquet", columns=["scan", "rt"])
        scan, rt = meta.column("scan").to_numpy(), _f64(meta.column("rt"))
    inside = scan[(rt >= t0) & (rt <= t1)]
    if inside.size == 0:
        return 0, 0
    return int(inside.min()), int(inside.max()) + 1


def _sweep(source, plan, settings, reducers, workers, work_dir, progress, window, *, spill_dir=None) -> SweepResult:
    if isinstance(source, pa.Table):
        if window is not None:
            scans = source.column("scan")
            source = source.filter(pc.and_(pc.greater_equal(scans, window[0]), pc.less(scans, window[1])))
        if settings.mz_axis is not None:
            raise ValueError("an m/z axis belongs to a bundle; a table of points has none to apply")
        if workers > 1 or work_dir is not None:
            raise ValueError("a table of points is swept in memory; give a bundle to use workers")
        blocks, ledger = _table_blocks(source, settings.block_rows, plan.levels)
        parts, held = [], 0
        for part in _map_parts(blocks, plan, settings):
            parts.append(part)
            held += part.rows
            if held > settings.memory_rows:
                # A table of points has no disk path to carry on along.
                raise MemoryError(
                    f"over {settings.memory_rows} rows (matches, evidence, claims) from a table of points, "
                    f"which is swept in memory: write it as a bundle to sweep it on disk, or raise memory_rows"
                )
        out = _MapOut.concat(parts)
        return _reduce_in_memory(out, plan, ledger, settings, reducers, None)
    bundle = Path(source)
    read_bundle_manifest(bundle)
    if workers <= 1 and work_dir is None:
        axis = _load_axis(bundle, settings)
        ledger = _scan_metadata(bundle)
        emit_start(progress, "sweep", total=0, message=bundle.name)
        parts, held = [], 0
        stream = (
            iter_scan_blocks(bundle, columns=("mz", "intensity"), block_rows=settings.block_rows, scan_range=window)
            if window is not None
            else None
        )
        mapped = _map_parts(_bundle_blocks(bundle, ledger, settings, axis, blocks=stream, levels=plan.levels),
                            plan, settings)  # fmt: skip
        for part in mapped:
            parts.append(part)
            held += part.rows
            if held > settings.memory_rows:
                # Too much to hold: what is held becomes the first run of a
                # sweep on disk, which carries on from the next window group
                # (until 2026-11-10, from the next whole block).
                need = _spill_estimate(bundle, ledger, held, window)
                emit_done(progress, "sweep", message=f"over {settings.memory_rows} rows; on disk")
                return _on_temporary_disk(bundle, plan, settings, reducers, 1, progress,
                                          resume=(ledger, mapped, parts), window=window, spill_dir=spill_dir,
                                          need=need)  # fmt: skip
        out = _MapOut.concat(parts)
        result = _reduce_in_memory(out, plan, ledger, settings, reducers, axis)
        emit_done(progress, "sweep", message=f"{out.matches.n} matches")
        return result
    if work_dir is None:
        return _on_temporary_disk(bundle, plan, settings, reducers, workers, progress, window=window,
                                  spill_dir=spill_dir)  # fmt: skip
    return _sweep_on_disk(bundle, plan, settings, reducers, workers, own_dir(work_dir), progress, window=window)


def own_dir(work_dir: Path | str) -> Path:
    """A new directory under ``work_dir`` for one sweep: ``sweep-<id>/``. A
    sweep writes only there, so a later sweep into the same ``work_dir``
    never removes or rewrites what an earlier result reads (until
    2026-11-10 both layouts cleared and reused ``work_dir`` itself, so an
    earlier result read the later sweep's rows)."""
    work = Path(work_dir) / f"sweep-{uuid.uuid4().hex[:16]}"
    work.mkdir(parents=True)
    return work


def _spill_estimate(bundle: Path, ledger: Ledger, held: int, window) -> int:
    """Bytes a spilled sweep is likely to write: the rows held so far, scaled
    by the points still to read (each scan's ``peak_count`` where the bundle
    records it, its peak files' rows otherwise), at
    :data:`SPILL_BYTES_PER_ROW`. An estimate for a free-space check, not a
    bound."""
    read = int(ledger.n_points.sum())
    total = None
    path = bundle / "scan_metadata.parquet"
    if "peak_count" in pq.read_schema(path).names:
        t = pq.read_table(path, columns=["scan", "peak_count"])
        scan = t.column("scan").to_numpy().astype(np.int64)
        count = t.column("peak_count").to_numpy(zero_copy_only=False)
        if not np.isnan(count.astype(np.float64)).any():
            inside = np.ones(scan.size, bool) if window is None else (scan >= window[0]) & (scan < window[1])
            total = int(count[inside].astype(np.int64).sum())
    if total is None:
        from constellation.massspec.bundle import peak_paths

        total = sum(pq.read_metadata(p).num_rows for p in peak_paths(bundle))
    scale = max(total / max(read, 1), 1.0)
    return int(held * scale * SPILL_BYTES_PER_ROW)


def _check_room(where: Path, need: int) -> None:
    """Refuse a spill ``where`` has no room for, before it starts: a spill
    at GPF scale is about 100 GB, and a disk filled part way leaves a
    sweep's files behind with nothing to show for them."""
    free = shutil.disk_usage(where).free
    if need > free:
        raise OSError(
            f"a sweep that no longer fits in memory would spill about {need / 1e9:.1f} GB to {where}, "
            f"which has {free / 1e9:.1f} GB free: pass spill_dir= (or work_dir=) on a disk with room, "
            f"a narrower plan, or a larger memory_rows"
        )


def _on_temporary_disk(bundle, plan, settings, reducers, workers, progress, resume=None, window=None, *,
                       spill_dir=None, need: int | None = None) -> SweepResult:  # fmt: skip
    """A sweep on disk in a temporary directory (in ``spill_dir``, else the
    system's), which lives as long as its result: outputs are read from it
    as asked, and it is removed after. ``resume`` is (ledger, the rest of
    the map's stream of window groups, what is held) of an in-memory sweep
    to carry on, as one shard; ``need`` the bytes it is expected to write,
    checked against the directory's free space first."""
    where = Path(spill_dir) if spill_dir is not None else Path(tempfile.gettempdir())
    where.mkdir(parents=True, exist_ok=True)
    if need is not None:
        _check_room(where, need)
    tmp = Path(tempfile.mkdtemp(prefix="sweep-", dir=where))
    _LOG.info("sweep on disk in %s", tmp)
    try:
        result = _sweep_on_disk(bundle, plan, settings, reducers, workers, tmp, progress, resume=resume, window=window)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    weakref.finalize(result, shutil.rmtree, tmp, True)
    return result


def _shared_counts(claims: Claims, n_queries: int, *, per_cell: bool) -> np.ndarray:
    """Per query, its matches whose point another query also claimed. A
    claim is a (point, cell) pair, which is one match under ``nearest`` and
    ``all``; under ``sum`` (``per_cell``) a match is a cell over several
    points, and counts once however many of them are shared. A scan's
    claims are made together, so every claim of a cell is in ``claims``."""
    shared = claims.take(np.flatnonzero(claims.n_queries >= 2))
    query = shared.query
    if per_cell and shared.n:
        by = np.lexsort((shared.channel, shared.query, shared.scan, shared.level))
        key = [v[by] for v in (shared.level, shared.scan, shared.query, shared.channel)]
        first = np.r_[True, np.any([k[1:] != k[:-1] for k in key], axis=0)]
        query = key[2][first]
    return np.bincount(query, minlength=n_queries).astype(np.int64)


def _reduce_in_memory(out: _MapOut, plan, ledger, settings, reducers, axis) -> SweepResult:
    """Reduce what an in-memory sweep matched, in buckets of queries of about
    ``settings.reduce_rows`` built rows (:func:`_reduce_weights`), as a
    sweep on disk does. Buckets are contiguous ascending query ranges and a
    query's rows depend only on that query, so each output is the one call's
    over the whole plan, row for row; what a call builds is bounded. Until
    2026-11-10 the whole plan went to the reducers at once (1,962,000 cells
    in one call where the disk path's largest was 48,000)."""
    order = ledger.order()
    shared = (_shared_counts(out.claims, plan.n_queries, per_cell=settings.rule.name == "sum")
              if settings.claims else None)  # fmt: skip
    matches = out.matches.sorted(order[ledger.rows_of(out.matches.scan)])
    evidence = out.evidence.take(order_by(out.evidence.query, out.evidence.scan))
    counts = np.bincount(matches.query, minlength=plan.n_queries).astype(np.int64)
    buckets = _buckets(_reduce_weights(plan, ledger, settings, reducers, counts), settings.reduce_rows)
    result = SweepResult(scans=ledger, matches=matches, mz_axis=None if axis is None else axis.key)
    collected: list[dict[str, list[pa.Table]]] = [{name: [] for name in outputs_of(r)} for r in reducers]
    for q0, q1 in buckets or [(0, 0)]:
        m0, m1 = np.searchsorted(matches.query, [q0, q1], side="left")
        e0, e1 = np.searchsorted(evidence.query, [q0, q1], side="left")
        part = _MapOut(matches.take(np.arange(m0, m1)), evidence.take(np.arange(e0, e1)), Claims.empty())
        batch = _batch(plan, np.arange(q0, q1, dtype=np.int64), part, ledger, order, settings,
                       None if shared is None else shared[q0:q1], presorted=True,
                       sized_for_cells=_fills(reducers))  # fmt: skip
        for into, tables in zip(collected, _reduce(batch, reducers)):
            for name, table in tables.items():
                into[name].append(table)
    for reducer, tables in zip(reducers, collected):
        schemas = outputs_of(reducer)
        for name, parts in tables.items():
            # The caller's in-memory result: a terminal table, made of one
            # bucket's table after another.
            result.tables[name] = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
            result.schemas[name] = schemas[name]
    if settings.claims:
        result.tables[CLAIMS] = claims_table(out.claims, plan, ledger, settings.acquisition_id)
        result.schemas[CLAIMS] = SWEEP_POINT_CLAIM_TABLE
    return result


# ── on disk ───────────────────────────────────────────────────────────
#
# A sweep's own directory (work_dir/sweep-<id>/, or a temporary one):
# plan/{queries,channels}.parquet        the plan, read by every worker
# matches/shard-SSSSS-RRRR.parquet       a shard's matches, in runs sorted by query
# evidence/shard-SSSSS-RRRR.parquet      its evidence counts, sorted by query
# claims/shard-SSSSS-RRRR.parquet        its claims, each run sorted by scan and point
# shared/shard-SSSSS.parquet             its shared matches per query
# scans/shard-SSSSS.parquet              the scans a shard read, with points
# scans.parquet                          the ledger, written after the map
# reduce/<name>/part-BBBBB.parquet       a bucket's output per reducer
# reduce/point_claims/part-SSSSS.parquet a shard's claims, sorted as one

_WORKER: dict[str, object] = {}


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _worker_plan(work: Path) -> QueryPlan:
    key = f"plan:{work}"
    if key not in _WORKER:
        _WORKER.clear()
        _WORKER[key] = QueryPlan.read(work / "plan")
    return _WORKER[key]  # type: ignore[return-value]


def _read_ledger(path: Path) -> Ledger:
    t = pq.read_table(path)
    scan = t.column("scan").to_numpy().astype(np.int64)
    ledger = Ledger(
        scan=scan,
        level=t.column("level").to_numpy().astype(np.int8),
        rt=t.column("rt").to_numpy(),
        lower=_f64(t.column("isolation_lower")),
        upper=_f64(t.column("isolation_upper")),
        n_points=t.column("n_points").to_numpy().astype(np.int64),
        tic=_f64(t.column("tic").fill_null(np.nan)),
        iit=_f64(t.column("iit").fill_null(np.nan)),
        scan_lower=_f64(t.column("scan_window_lower").fill_null(np.nan)),
        scan_upper=_f64(t.column("scan_window_upper").fill_null(np.nan)),
    )
    parent = t.column("parent_scan").fill_null(-1).to_numpy().astype(np.int64)
    rows = ledger.scan_rows(parent)
    object.__setattr__(ledger, "parent", np.where(parent >= 0, rows, -1).astype(np.int64))
    return ledger


def _worker_ledger(work: Path) -> tuple[Ledger, np.ndarray]:
    key = f"ledger:{work}"
    if key not in _WORKER:
        ledger = _read_ledger(work / "scans.parquet")
        _WORKER[key] = (ledger, ledger.order())
    return _WORKER[key]  # type: ignore[return-value]


def _shard_name(i: int) -> str:
    return f"shard-{i:05d}.parquet"


def _evidence_table(e: Evidence) -> pa.Table:
    return pa.table(
        {
            "query": pa.array(e.query.astype(np.int32)),
            "group": pa.array(e.group.astype(np.int8)),
            "scan": pa.array(e.scan.astype(np.int32)),
            "n_keys": pa.array(e.n_keys.astype(np.int16)),
            "n_peaks": pa.array(e.n_peaks.astype(np.int16)),
            "n_channels": pa.array(e.n_channels.astype(np.int16)),
            "intensity": pa.array(e.intensity),
        },
        schema=SWEEP_EVIDENCE_TABLE,
    )


def _evidence_from(t: pa.Table) -> Evidence:
    def col(name, dtype):
        return t.column(name).to_numpy().astype(dtype)

    return Evidence(col("query", np.int64), col("group", np.int8), col("scan", np.int64),
                    col("n_keys", np.int64), col("n_peaks", np.int64), col("n_channels", np.int64),
                    col("intensity", np.float64))  # fmt: skip


def _run_name(shard: int, run: int) -> str:
    return f"shard-{shard:05d}-{run:04d}.parquet"


def _map_shard(bundle: str, work: str, index: int, scan_range: tuple[int, int], settings: SweepSettings) -> int:
    """Sweep one scan range; write its matches, evidence and claims as
    sorted runs of at most ``settings.run_rows`` matches, and the scans it
    read. Returns the number of matches."""
    b, w = Path(bundle), Path(work)
    meta = _scan_metadata(b)
    axis = _load_axis(b, settings)
    plan = _worker_plan(w)
    blocks = iter_scan_blocks(
        b, columns=("mz", "intensity"), block_rows=settings.block_rows, scan_range=scan_range
    )
    parts = _map_parts(_bundle_blocks(b, meta, settings, axis, blocks=blocks, levels=plan.levels), plan, settings)
    return _map_stream(w, index, plan, meta, parts, settings)


def _map_stream(
    w: Path,
    index: int,
    plan: QueryPlan,
    meta: Ledger,
    parts: Iterable[_MapOut],
    settings: SweepSettings,
    pending: list[_MapOut] | None = None,
) -> int:
    """Write the stream of window groups ``parts`` (whose scans ``meta``
    counts the points of, as they are read) as shard ``index``, starting
    from ``pending`` parts already made, and the shard's scans. A run is
    written once ``run_rows`` rows are held (matches, evidence and claims
    alike). Returns the number of matches.

    Claims are written only for whole blocks: a block's window groups
    interleave scan numbers, so a claims run ending inside a block would
    be out of scan order with the next, and putting them back in order
    then needed the whole shard's claims at once (2026-11-10). A block's
    claims are held until it ends, so runs follow one another in scan
    order and are made into the table one at a time."""
    shared = np.zeros(plan.n_queries, dtype=np.int64) if settings.claims else None
    # The caller's list is emptied as it is written, so what was held in
    # memory is freed rather than kept for the rest of the shard.
    given = pending if pending is not None else []
    pending: list[_MapOut] = []
    reading: list[Claims] = []  # the claims of the block being read
    finished: list[Claims] = []  # the claims of whole blocks, not yet written
    held = total = run = 0

    def take(part: _MapOut) -> None:
        nonlocal held, total
        held += part.rows  # its claims are held too, until their block is written
        total += part.matches.n
        if part.claims.n:
            reading.append(part.claims)
            part = replace(part, claims=Claims.empty())
        if part.ends_block:
            finished.extend(reading)
            reading.clear()
        pending.append(part)

    def flush() -> None:
        nonlocal held, run
        if not pending and not finished:
            return
        out = _MapOut.concat(pending)
        claims = Claims.concat(finished)
        pending.clear()
        finished.clear()
        held = sum(c.n for c in reading)
        name = _run_name(index, run)
        m = out.matches
        m = m.take(order_by(m.query, m.level, m.scan, m.channel, m.point_index))
        # Sorted runs of at most run_rows each (one window group can hold more).
        for start in range(0, max(m.n, 1), settings.run_rows):
            stop = min(start + settings.run_rows, m.n)
            _write(m.take(np.arange(start, stop)).table(), w / "matches" / _run_name(index, run),
                   row_group_size=settings.run_row_group_rows)  # fmt: skip
            run += 1
        if settings.gate is not None:
            e = out.evidence
            _write(_evidence_table(e.take(order_by(e.query, e.group, e.scan))),
                   w / "evidence" / name, row_group_size=settings.run_row_group_rows)  # fmt: skip
        if settings.claims and claims.n:
            _write(claims.sorted().table(), w / "claims" / name, row_group_size=1 << 18)
            shared[:] += _shared_counts(claims, plan.n_queries, per_cell=settings.rule.name == "sum")

    for part in given:
        take(part)
    given.clear()
    for part in parts:
        take(part)
        if held >= settings.run_rows:
            flush()
    finished.extend(reading)  # a stream that ended inside a block
    reading.clear()
    flush()
    if settings.claims:
        at = np.flatnonzero(shared)
        _write(pa.table({"query": pa.array(at.astype(np.int32)), "n": pa.array(shared[at])}),
               w / "shared" / _shard_name(index))  # fmt: skip
    seen = meta.n_points > 0
    _write(
        pa.table(
            {
                "scan": pa.array(meta.scan[seen].astype(np.int32)),
                "n_points": pa.array(meta.n_points[seen].astype(np.int32)),
            }
        ),
        w / "scans" / _shard_name(index),
    )
    return total


def _write(table: pa.Table, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, **kw)
    tmp.replace(path)


#: Rows a reduce has decoded from sorted runs, in this process (for tests
#: and benchmarks of how much more than its own rows a bucket reads).
READ_STATS = {"rows_decoded": 0, "claims_held": 0}


def _footer(path: Path):
    """A run's footer, read once per process."""
    key = f"footer:{path}"
    if key not in _WORKER:
        _WORKER[key] = pq.read_metadata(path)
    return _WORKER[key]


def _read_range(files: list[Path], q0: int, q1: int) -> pa.Table | None:
    """The rows of queries ``[q0, q1)`` from sorted runs: in each run, only
    the row groups whose ``query`` statistics overlap the range are read."""
    if not files:
        return None
    parts = []
    schema = None
    for f in files:
        meta = _footer(f)
        schema = schema or meta.schema.to_arrow_schema()
        column = meta.schema.to_arrow_schema().get_field_index("query")
        groups = []
        for i in range(meta.num_row_groups):
            stats = meta.row_group(i).column(column).statistics
            if stats is None or not stats.has_min_max or (stats.max >= q0 and stats.min < q1):
                groups.append(i)
        if groups:
            table = pq.ParquetFile(f, metadata=meta).read_row_groups(groups)
            READ_STATS["rows_decoded"] += table.num_rows
            q = table.column("query")
            parts.append(table.filter(pc.and_(pc.greater_equal(q, q0), pc.less(q, q1))))
    # One bucket's rows, for one reducer call: not a hand-off between stages.
    return pa.concat_tables(parts) if parts else schema.empty_table()


def _reduce_bucket(
    work: str, index: int, q0: int, q1: int, reducers: Sequence[Reducer], settings: SweepSettings
) -> list[str]:
    """Reduce queries ``[q0, q1)`` from every shard's outputs; write each
    reducer's part and return their paths."""
    w = Path(work)
    plan = _worker_plan(w)
    ledger, order = _worker_ledger(w)

    def shards(kind: str) -> list[Path]:
        return sorted((w / kind).glob("shard-*.parquet"))

    m = _read_range(shards("matches"), q0, q1)
    matches = Matches.from_table(m) if m is not None else Matches.empty()  # no shard matched anything
    ev = _read_range(shards("evidence"), q0, q1)
    evidence = _evidence_from(ev) if ev is not None else Evidence.empty()
    shared = None
    if settings.claims:
        shared = np.zeros(q1 - q0, dtype=np.int64)
        t = _read_range(shards("shared"), q0, q1)
        if t is not None and t.num_rows:
            np.add.at(shared, t.column("query").to_numpy() - q0, t.column("n").to_numpy())
    queries = np.arange(q0, q1, dtype=np.int64)
    batch = _batch(plan, queries, _MapOut(matches, evidence, Claims.empty()), ledger, order, settings,
                   shared, sized_for_cells=_fills(reducers))  # fmt: skip
    out: dict[str, str] = {}
    for tables in _reduce(batch, reducers):
        for name, table in tables.items():
            path = w / "reduce" / name / f"part-{index:05d}.parquet"
            _write(table, path)
            out[name] = str(path)
    return out


def _claims_part(work: str, index: int, acquisition_id: int) -> list[str]:
    """A shard's claims with their claimants, run by run, in scan order.

    Each run holds whole blocks, sorted (``_map_stream``), and the runs
    follow one another in scan order, so a run at a time is enough: holding
    them all to sort them as one (2026-11-10, briefly) undid the memory
    bound of a sweep spilled to disk, which is one shard. Runs are taken in
    the order written, by number (by name, a run past 9,999 came first)."""
    w = Path(work)
    plan = _worker_plan(w)
    ledger, _ = _worker_ledger(w)
    out = []
    for f in _shard_runs(w / "claims", index):
        claims = Claims.from_table(pq.read_table(f))
        READ_STATS["claims_held"] = max(READ_STATS["claims_held"], claims.n)
        path = w / "reduce" / CLAIMS / f"part-{f.stem.removeprefix('shard-')}.parquet"
        _write(claims_table(claims, plan, ledger, acquisition_id), path)
        out.append(str(path))
    return out


def _shard_runs(directory: Path, index: int) -> list[Path]:
    """A shard's runs in ``directory``, in the order they were written."""
    return sorted(directory.glob(f"shard-{index:05d}-*.parquet"), key=lambda f: int(f.stem.rsplit("-", 1)[1]))


def _fills(reducers) -> bool:
    """Whether a reducer of the sweep builds every cell (declares ``zero_fills``)."""
    return any(getattr(r, "zero_fills", False) for r in reducers)


def _reduce_weights(plan: QueryPlan, ledger: Ledger, settings: SweepSettings, reducers, matches: np.ndarray) -> np.ndarray:
    """Per query, the rows reducing it builds: its matches; with a gate, the
    (scan, query) pairs of a query with matches, whose evidence is walked
    over them; with a reducer that zero-fills, every eligible cell. So a
    bucket is bounded by what is built, not only by what matched: a
    thousand unmatched global queries cost nothing to gate, and their
    zero-filled trace is sized as the trace it is."""
    weight = matches.astype(np.int64).copy()
    fills = _fills(reducers)
    if settings.gate is None and not fills:
        return weight
    every = np.arange(plan.n_queries, dtype=np.int64)
    for level in plan.levels:
        channels = plan.channels_for_level(level)
        if channels is None:
            continue
        pairs = eligible_counts(plan, ledger, level, every)
        if settings.gate is not None:
            weight += np.where(matches > 0, pairs, 0)
        if fills:
            weight += pairs * channels.counts()
    return weight


def _buckets(weights: np.ndarray, target: int) -> list[tuple[int, int]]:
    """Contiguous query ranges of about ``target`` rows each (at least one
    query each), from per-query row counts (:func:`_reduce_weights`)."""
    n = weights.size
    if n == 0:
        return []
    weight = weights.astype(np.int64) + 1
    cum = np.cumsum(weight)
    edges = [0]
    while edges[-1] < n:
        base = cum[edges[-1] - 1] if edges[-1] else 0
        nxt = int(np.searchsorted(cum, base + target, side="right"))
        edges.append(min(max(nxt, edges[-1] + 1), n))
    return list(zip(edges[:-1], edges[1:]))


def _run(executor, fn, jobs, progress, stage) -> list:
    futures = [executor.submit(fn, *job) for job in jobs]
    emit_start(progress, stage, total=len(futures))
    for done, _ in enumerate(as_completed(futures), 1):
        emit_progress(progress, stage, done, len(futures))
    results = [f.result() for f in futures]
    emit_done(progress, stage, completed=len(futures), total=len(futures))
    return results


class _Inline:
    """The pool's interface, running each task as it is submitted."""

    def submit(self, fn, *args):
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001 - handed to the caller, as a pool would
            future.set_exception(exc)
        return future

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sweep_on_disk(bundle, plan, settings, reducers, workers, work: Path, progress, resume=None,
                   window=None) -> SweepResult:  # fmt: skip
    _WORKER.clear()  # an inline sweep reads its plan through the same cache
    # ``work`` is this sweep's own new directory (``own_dir``, or a temporary
    # one): nothing in it to clear, and nothing of another sweep's to touch.
    plan.write(work / "plan")
    axis = _load_axis(bundle, settings)
    # Read (and checked: a multi-CV run is refused) before any worker starts;
    # its point counts are the shards', filled in after the map.
    ledger = _scan_metadata(bundle)
    # A range is cut into shards of its own (until 2026-11-09 the whole run's
    # shards were trimmed to it, so a range inside one shard ran in one process).
    shards = (
        scan_range_shards(bundle, target_rows=settings.shard_rows, scan_range=window) if resume is None else [None]
    )
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker)
        if workers > 1
        else _Inline()
    )
    with pool as executor:
        if resume is None:
            _run(executor, _map_shard,
                 [(str(bundle), str(work), i, r, settings) for i, r in enumerate(shards)],
                 progress, "sweep: matching")  # fmt: skip
        else:
            meta, blocks, held = resume
            _map_stream(work, 0, plan, meta, blocks, settings, pending=held)
        for f in sorted((work / "scans").glob("shard-*.parquet")):
            t = pq.read_table(f)
            ledger.n_points[ledger.rows_of(t.column("scan").to_numpy().astype(np.int64))] = t.column(
                "n_points"
            ).to_numpy()
        _write(ledger.table(), work / "scans.parquet")
        counts = np.zeros(plan.n_queries, dtype=np.int64)
        for f in sorted((work / "matches").glob("shard-*.parquet")):
            q = pq.read_table(f, columns=["query"]).column("query").to_numpy()
            counts += np.bincount(q, minlength=plan.n_queries)
        buckets = _buckets(_reduce_weights(plan, ledger, settings, reducers, counts), settings.reduce_rows)
        paths = _run(executor, _reduce_bucket,
                     [(str(work), i, q0, q1, tuple(reducers), settings) for i, (q0, q1) in enumerate(buckets)],
                     progress, "sweep: reducing")  # fmt: skip
        claim_parts = (
            _run(executor, _claims_part,
                 [(str(work), i, settings.acquisition_id) for i in range(len(shards))],
                 progress, "sweep: claims")  # fmt: skip
            if settings.claims
            else []
        )
    _WORKER.clear()
    result = SweepResult(scans=ledger, matches=None, mz_axis=None if axis is None else axis.key, work=work)
    for reducer in reducers:
        for name, schema in outputs_of(reducer).items():
            result.parts[name] = [Path(p[name]) for p in paths]
            result.schemas[name] = schema
    if settings.claims:
        result.parts[CLAIMS] = [Path(p) for part in claim_parts for p in part]
        result.schemas[CLAIMS] = SWEEP_POINT_CLAIM_TABLE
    return result
