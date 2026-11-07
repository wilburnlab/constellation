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

import shutil
import tempfile
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
    pair_eligible,
    ragged_arange,
)

__all__ = ["SweepSettings", "SweepResult", "sweep", "CLAIMS"]

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
    #: Matches per reduce bucket when the sweep runs on disk.
    reduce_rows: int = 1 << 22
    #: Matches a map shard holds before writing them out as a sorted run.
    run_rows: int = 4 << 20
    #: Matches a sweep in memory holds before it carries on on disk. A held
    #: match costs some 200–500 bytes at peak (sorting, the reducers' tables),
    #: so this keeps an in-memory sweep near a disk worker's 3 GB.
    memory_rows: int = 8 << 20

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


def _scan_metadata(bundle: Path) -> Ledger:
    t = pq.read_table(
        bundle / "scan_metadata.parquet",
        columns=["scan", "level", "rt", "isolation_lower", "isolation_upper"],
    ).sort_by("scan")
    n = t.num_rows
    return Ledger(
        scan=t.column("scan").to_numpy().astype(np.int64),
        level=t.column("level").to_numpy().astype(np.int8),
        rt=_f64(t.column("rt")),
        lower=_f64(t.column("isolation_lower")),
        upper=_f64(t.column("isolation_upper")),
        n_points=np.zeros(n, dtype=np.int64),
    )


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
    bundle: Path, meta: Ledger, settings: SweepSettings, axis, *, blocks: Iterable[ScanBlock] | None = None
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
        mz = b.columns["mz"].astype(np.float64, copy=False)
        intensity = b.columns["intensity"].astype(np.float64, copy=False)
        level = meta.level[at]
        if axis is not None:
            point_level = np.repeat(level, counts)
            mz = mz.copy()
            for lvl in np.unique(level):
                sel = point_level == lvl
                mz[sel] = axis.corrected(int(lvl), mz[sel])
        point_index = ragged_arange(np.zeros(b.n_scans, np.int64), counts)
        perm = _sorted_within_scans(b.offsets, mz)
        if perm is not None:
            mz, intensity, point_index = mz[perm], intensity[perm], point_index[perm]
        yield _Block(
            scan=b.scan,
            level=level,
            rt=meta.rt[at],
            lower=meta.lower[at],
            upper=meta.upper[at],
            offsets=b.offsets,
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
    ledger = Ledger(
        scan=scan[starts],
        level=t.column("level").to_numpy(zero_copy_only=False).astype(np.int8)[starts],
        rt=_f64(t.column("rt"))[starts],
        lower=lower[starts],
        upper=upper[starts],
        n_points=np.diff(offsets),
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

    @classmethod
    def concat(cls, parts: list[_MapOut]) -> _MapOut:
        return cls(
            Matches.concat([p.matches for p in parts]),
            Evidence.concat([p.evidence for p in parts]),
            Claims.concat([p.claims for p in parts]),
        )


def _map_block(block: _Block, plan: QueryPlan, settings: SweepSettings) -> _MapOut:
    matches, evidence, claims = [], [], []
    gate = settings.gate
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
            plan.rt_lo, plan.rt_hi, counts > 0, float(block.rt[timed].min()), float(block.rt[timed].max())
        )
        groups = isolation_groups(
            active, plan.precursor_mz, block.lower[rows], block.upper[rows], rows, gate=level > 1
        )
        assigned = plan.assigned_scan if level > 1 else None

        def eligible(query: np.ndarray, row: np.ndarray) -> np.ndarray:
            return pair_eligible(query, block.rt[row], block.scan[row], plan.rt_lo, plan.rt_hi, assigned)

        def stored(row: np.ndarray, point: np.ndarray) -> np.ndarray:
            at = block.offsets[row] + np.maximum(point, 0)
            return np.where(point >= 0, block.point_index[at], -1)

        for g in groups:
            n = counts[g.queries]
            channel_at = ragged_arange(channels.offsets[g.queries], n)
            channel_query = np.repeat(g.queries, n)
            channel_mz = channels.mz[channel_at]
            kw = dict(kernel=settings.kernel, crossover=settings.crossover,
                      max_elements=settings.kernel_elements)  # fmt: skip
            m = match_group(block.mz, block.intensity, block.offsets, g.rows, channel_mz, settings.rule,
                            tolerance, unit, **kw)  # fmt: skip
            if m.n:
                row = g.rows[m.row]
                query = channel_query[m.channel]
                ok = eligible(query, row)
                if ok.any():
                    row, query, sel = row[ok], query[ok], np.flatnonzero(ok)
                    channel = channel_at[m.channel[sel]] - channels.offsets[query]
                    point = stored(row, m.point[sel])
                    matches.append(
                        Matches(query=query, level=np.full(sel.size, level, np.int8),
                                scan=block.scan[row], channel=channel, point_index=point, mz=m.mz[sel],
                                intensity=m.intensity[sel], n_points=m.n_points[sel])  # fmt: skip
                    )
                    if settings.claims and settings.rule.name != "sum":
                        claims.append(find_claims(level, block.scan[row], point, query, channel,
                                                  m.mz[sel], m.intensity[sel]))  # fmt: skip
            if settings.claims and settings.rule.name == "sum":
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
                evidence.append(
                    count_evidence(block.mz, block.intensity, block.offsets, g.rows, block.scan, channels,
                                   channel_at, channel_query, eligible, tolerance, unit, gate.top_peaks,
                                   by_charge=level == 1)  # fmt: skip
                )
    return _MapOut(Matches.concat(matches), Evidence.concat(evidence), Claims.concat(claims))


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


def _batch(plan, queries, out: _MapOut, ledger, order, settings, shared) -> QueryBatch:
    """A bucket's batch: matches sorted for reducers, and the gate decided."""
    matches = out.matches.sorted(order[ledger.rows_of(out.matches.scan)])
    evidence = out.evidence.take(np.lexsort((out.evidence.scan, out.evidence.query)))
    batch = QueryBatch(plan=plan, queries=queries, matches=matches, scans=ledger, scan_order=order,
                       settings=settings, evidence=evidence, shared=shared)  # fmt: skip
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


def _reduce(batch: QueryBatch, reducers: Sequence[Reducer]) -> list[pa.Table]:
    passing = batch.passing() if any(r.gated for r in reducers) else None
    return [_conform(r.reduce(passing if r.gated else batch), r.schema, r.name) for r in reducers]


def sweep(
    source: Path | str | pa.Table,
    plan: QueryPlan,
    settings: SweepSettings,
    *,
    reducers: Sequence[Reducer] = (),
    workers: int = 1,
    work_dir: Path | str | None = None,
    progress: ProgressCallback | None = None,
) -> SweepResult:
    """Sweep ``source`` for ``plan``'s queries and reduce the matches.

    In memory unless ``workers > 1`` or a ``work_dir`` is given, or the
    matches outgrow ``settings.memory_rows``. Then the bundle is cut into
    scan ranges (from its footers, so the cut does not depend on
    ``workers``), each range's matches are written to ``work_dir`` in sorted
    runs of at most ``settings.run_rows``, and each bucket of queries is
    reduced from them, ``workers`` processes at a time for both. The outputs
    are the same either way, byte for byte, whatever the number of workers.
    Without a ``work_dir`` a temporary one is used, kept while the result
    is, so its outputs are read as asked. With ``settings.claims`` the
    claims table is the ``point_claims`` output.
    """
    if isinstance(source, pa.Table):
        if settings.mz_axis is not None:
            raise ValueError("an m/z axis belongs to a bundle; a table of points has none to apply")
        if workers > 1 or work_dir is not None:
            raise ValueError("a table of points is swept in memory; give a bundle to use workers")
        blocks, ledger = _table_blocks(source, settings.block_rows, plan.levels)
        parts, held = [], 0
        for b in blocks:
            parts.append(_map_block(b, plan, settings))
            held += parts[-1].matches.n
            if held > settings.memory_rows:
                # A table of points has no disk path to carry on along.
                raise MemoryError(
                    f"over {settings.memory_rows} matches from a table of points, which is swept in memory: "
                    f"write it as a bundle to sweep it on disk, or raise memory_rows"
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
        blocks = _bundle_blocks(bundle, ledger, settings, axis)
        for b in blocks:
            parts.append(_map_block(b, plan, settings))
            held += parts[-1].matches.n
            if held > settings.memory_rows:
                # Too much to hold: what is held becomes the first run of a
                # sweep on disk, which carries on from the next block.
                emit_done(progress, "sweep", message=f"over {settings.memory_rows} matches; on disk")
                return _on_temporary_disk(bundle, plan, settings, reducers, 1, progress,
                                          resume=(ledger, blocks, parts))  # fmt: skip
        out = _MapOut.concat(parts)
        result = _reduce_in_memory(out, plan, ledger, settings, reducers, axis)
        emit_done(progress, "sweep", message=f"{out.matches.n} matches")
        return result
    if work_dir is None:
        return _on_temporary_disk(bundle, plan, settings, reducers, workers, progress)
    return _sweep_on_disk(bundle, plan, settings, reducers, workers, Path(work_dir), progress)


def _on_temporary_disk(bundle, plan, settings, reducers, workers, progress, resume=None) -> SweepResult:
    """A sweep on disk in a temporary directory, which lives as long as its
    result: outputs are read from it as asked, and it is removed after.
    ``resume`` is (ledger, the rest of the block stream, matches held) of an
    in-memory sweep to carry on, as one shard."""
    tmp = Path(tempfile.mkdtemp(prefix="sweep-"))
    try:
        result = _sweep_on_disk(bundle, plan, settings, reducers, workers, tmp, progress, resume=resume)
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
    order = ledger.order()
    queries = np.arange(plan.n_queries, dtype=np.int64)
    shared = (_shared_counts(out.claims, plan.n_queries, per_cell=settings.rule.name == "sum")
              if settings.claims else None)  # fmt: skip
    batch = _batch(plan, queries, out, ledger, order, settings, shared)
    result = SweepResult(scans=ledger, matches=batch.matches, mz_axis=None if axis is None else axis.key)
    for reducer, table in zip(reducers, _reduce(batch, reducers)):
        result.tables[reducer.name] = table
        result.schemas[reducer.name] = reducer.schema
    if settings.claims:
        result.tables[CLAIMS] = claims_table(out.claims, plan, ledger, settings.acquisition_id)
        result.schemas[CLAIMS] = SWEEP_POINT_CLAIM_TABLE
    return result


# ── on disk ───────────────────────────────────────────────────────────
#
# work_dir/plan/{queries,channels}.parquet   the plan, read by every worker
# work_dir/matches/shard-SSSSS-RRRR.parquet   a shard's matches, in runs sorted by query
# work_dir/evidence/shard-SSSSS-RRRR.parquet  its evidence counts, sorted by query
# work_dir/claims/shard-SSSSS-RRRR.parquet    its claims, sorted by scan and point
# work_dir/shared/shard-SSSSS.parquet    its shared matches per query
# work_dir/scans/shard-SSSSS.parquet     the scans a shard read, with points
# work_dir/scans.parquet                 the ledger, written after the map
# work_dir/reduce/<name>/part-BBBBB.parquet   a bucket's output per reducer

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
    return Ledger(
        scan=t.column("scan").to_numpy().astype(np.int64),
        level=t.column("level").to_numpy().astype(np.int8),
        rt=t.column("rt").to_numpy(),
        lower=_f64(t.column("isolation_lower")),
        upper=_f64(t.column("isolation_upper")),
        n_points=t.column("n_points").to_numpy().astype(np.int64),
    )


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
    blocks = iter_scan_blocks(
        b, columns=("mz", "intensity"), block_rows=settings.block_rows, scan_range=scan_range
    )
    return _map_stream(w, index, _worker_plan(w), meta, _bundle_blocks(b, meta, settings, axis, blocks=blocks),
                       settings)  # fmt: skip


def _map_stream(
    w: Path,
    index: int,
    plan: QueryPlan,
    meta: Ledger,
    blocks: Iterable[_Block],
    settings: SweepSettings,
    pending: list[_MapOut] | None = None,
) -> int:
    """Match ``blocks`` (whose scans ``meta`` counts the points of) as shard
    ``index``, starting from ``pending`` matches already made, and write the
    shard's runs and scans."""
    shared = np.zeros(plan.n_queries, dtype=np.int64) if settings.claims else None
    # The caller's list is emptied as it is written, so what was held in
    # memory is freed rather than kept for the rest of the shard.
    pending = pending if pending is not None else []
    held = total = sum(p.matches.n for p in pending)
    run = 0

    def flush() -> None:
        nonlocal held, run
        if not pending:
            return
        out = _MapOut.concat(pending)
        pending.clear()
        held = 0
        name = _run_name(index, run)
        run += 1
        m = out.matches
        m = m.take(np.lexsort((m.point_index, m.channel, m.scan, m.level, m.query)))
        _write(m.table(), w / "matches" / name, row_group_size=1 << 18)
        if settings.gate is not None:
            e = out.evidence
            _write(_evidence_table(e.take(np.lexsort((e.scan, e.group, e.query)))),
                   w / "evidence" / name, row_group_size=1 << 18)  # fmt: skip
        if settings.claims:
            _write(out.claims.sorted().table(), w / "claims" / name, row_group_size=1 << 18)
            shared[:] += _shared_counts(out.claims, plan.n_queries, per_cell=settings.rule.name == "sum")

    for block in blocks:
        pending.append(_map_block(block, plan, settings))
        held += pending[-1].matches.n
        total += pending[-1].matches.n
        if held >= settings.run_rows:
            flush()
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


def _read_range(files: list[Path], q0: int, q1: int) -> pa.Table | None:
    import pyarrow.dataset as ds

    if not files:
        return None
    return ds.dataset([str(f) for f in files], format="parquet").to_table(
        filter=(ds.field("query") >= q0) & (ds.field("query") < q1)
    )


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
                   shared)  # fmt: skip
    out = []
    for reducer, table in zip(reducers, _reduce(batch, reducers)):
        path = w / "reduce" / reducer.name / f"part-{index:05d}.parquet"
        _write(table, path)
        out.append(str(path))
    return out


def _claims_part(work: str, index: int, acquisition_id: int) -> list[str]:
    """A shard's claims with their claimants, run by run, in scan order."""
    w = Path(work)
    plan = _worker_plan(w)
    ledger, _ = _worker_ledger(w)
    out = []
    for f in sorted((w / "claims").glob(f"shard-{index:05d}-*.parquet")):
        claims = Claims.from_table(pq.read_table(f))
        path = w / "reduce" / CLAIMS / f"part-{f.stem.removeprefix('shard-')}.parquet"
        _write(claims_table(claims, plan, ledger, acquisition_id), path)
        out.append(str(path))
    return out


def _reduce_weights(plan: QueryPlan, ledger: Ledger, settings: SweepSettings, reducers, matches: np.ndarray) -> np.ndarray:
    """Per query, the rows reducing it builds: its matches; with a gate, the
    (scan, query) pairs of a query with matches, whose evidence is walked
    over them; with a reducer that zero-fills, every eligible cell. So a
    bucket is bounded by what is built, not only by what matched: a
    thousand unmatched global queries cost nothing to gate, and their
    zero-filled trace is sized as the trace it is."""
    weight = matches.astype(np.int64).copy()
    fills = any(getattr(r, "zero_fills", False) for r in reducers)
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


def _sweep_on_disk(bundle, plan, settings, reducers, workers, work: Path, progress, resume=None) -> SweepResult:
    _WORKER.clear()  # an inline sweep reads its plan through the same cache
    for sub in ("matches", "evidence", "claims", "shared", "scans", "reduce"):
        if (work / sub).exists():
            shutil.rmtree(work / sub)
    plan.write(work / "plan")
    axis = _load_axis(bundle, settings)
    shards = scan_range_shards(bundle, target_rows=settings.shard_rows) if resume is None else [None]
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
        ledger = _scan_metadata(bundle)
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
    result = SweepResult(scans=ledger, matches=None, mz_axis=None if axis is None else axis.key)
    for k, reducer in enumerate(reducers):
        result.parts[reducer.name] = [Path(p[k]) for p in paths]
        result.schemas[reducer.name] = reducer.schema
    if settings.claims:
        result.parts[CLAIMS] = [Path(p) for part in claim_parts for p in part]
        result.schemas[CLAIMS] = SWEEP_POINT_CLAIM_TABLE
    return result
