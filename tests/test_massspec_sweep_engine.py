"""Running a sweep end to end (`massspec.sweep.engine`) and the XIC traces it
gives through `XicTraceReducer`.

A bundle streamed block by block must give what its peak table gives the
frozen extractor; the block size and the kernel must change nothing; and
the two behaviours changed on purpose are pinned here.
"""

from __future__ import annotations

import dataclasses

import functools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import _xic_scan_major_oracle as oracle
from _massspec_fixtures import dia_scans, drop_unacquired, planted_run, write_bundle, write_bundle_manifest
from constellation.massspec.bundle import open_peaks
from constellation.massspec.quant.chromatogram import XicTraceReducer, extract_xic_scan_major
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep import compile_queries
from constellation.massspec.sweep.engine import SweepSettings, sweep
from constellation.massspec.sweep.rules import AllWithin, Nearest, SumWithin

_RUNS = {
    "dia": dict(acquisition="dia", cycles=40, window_width=100.0),
    "staggered": dict(acquisition="dia", staggered=True, cycles=40, window_width=100.0),
    "dda": dict(acquisition="dda", cycles=40),
}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("engine"))


@functools.lru_cache(maxsize=None)
def _run(name: str, tmp_root: str):
    run = planted_run(**_RUNS[name])
    bundle = write_bundle(Path(tmp_root), name, run.scans, row_group_size=97)
    write_bundle_manifest(bundle)
    return run, bundle, open_peaks(bundle).to_table()


def _targets(run, *, nan_rt: bool = False) -> pa.Table:
    rows = run.targets.to_pylist()
    if nan_rt:
        rows.append(dict(rows[0], target_id=777, rt_center=float("nan")))
    return pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)


@pytest.mark.parametrize("name", list(_RUNS))
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("mode", ["nearest", "keep_unmatched", "all_in_window"])
def test_a_streamed_bundle_gives_what_its_table_gives_the_oracle(tmp_root, name, level, mode):
    run, bundle, peaks = _run(name, tmp_root)
    targets = _targets(run, nan_rt=True)
    kw = dict(
        acquisition_id=3, level=level, rt_window=15.0, tolerance=0.35, tolerance_unit="Da",
        all_in_window=mode == "all_in_window", drop_unmatched=mode != "keep_unmatched",
        charge_range=(1, 4),  # the oracle's default; the sweep's is each target's own charge
    )  # fmt: skip
    want = oracle.extract_xic_scan_major(peaks, targets, **kw)
    assert want.num_rows > 0
    # A bundle records each scan's acquired range, and a channel outside it
    # is not zero-filled (trace v2); a table of points records none.
    oracle.assert_traces_identical(extract_xic_scan_major(bundle, targets, **kw), drop_unacquired(want, bundle))
    oracle.assert_traces_identical(extract_xic_scan_major(peaks, targets, **kw), want)


def _traces(source, plan, **settings):
    rule = settings.pop("rule", Nearest())
    drop = settings.pop("drop_unmatched", True)
    s = SweepSettings(tolerances={1: (20.0, "ppm"), 2: (0.35, "Da")}, rule=rule, **settings)
    return sweep(source, plan, s, reducers=[XicTraceReducer(drop_unmatched=drop)]).table("xic_trace")


@pytest.mark.parametrize("rule", [Nearest(), AllWithin(), SumWithin()])
def test_block_size_and_kernel_change_nothing(tmp_root, rule):
    run, bundle, _ = _run("staggered", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    base = _traces(bundle, plan, rule=rule)
    assert base.num_rows > 0
    for kw in (dict(block_rows=50), dict(block_rows=777), dict(kernel="channel"), dict(kernel="point"),
               dict(kernel="channel", kernel_elements=64), dict(block_rows=1, kernel="point")):  # fmt: skip
        oracle.assert_traces_identical(_traces(bundle, plan, rule=rule, **kw), base)


def test_both_levels_in_one_sweep_are_each_levels_rows(tmp_root):
    run, bundle, peaks = _run("dia", tmp_root)
    targets = _targets(run)
    both = _traces(bundle, compile_queries(targets, levels=[1, 2], rt_window=15.0))
    for level, tol in ((1, (20.0, "ppm")), (2, (0.35, "Da"))):
        one = oracle.extract_xic_scan_major(
            peaks, targets, acquisition_id=0, level=level, rt_window=15.0, tolerance=tol[0],
            tolerance_unit=tol[1],
        )  # fmt: skip
        part = both.filter(pc.equal(both["level"], level))
        assert part.num_rows == one.num_rows
        for name in ("target_id", "scan", "mz_theoretical", "mz_observed", "intensity"):
            assert part[name].to_pylist() == one[name].to_pylist()
    levels = both["level"].to_pylist()
    tids = both["target_id"].to_pylist()
    # Within a target, MS1 rows come before MS2 rows.
    for t in set(tids):
        seq = [lv for lv, tid in zip(levels, tids) if tid == t]
        assert seq == sorted(seq)


def test_a_missing_precursor_mz_is_derived_and_gates_fragments(tmp_root):
    run, bundle, peaks = _run("dia", tmp_root)
    given = _targets(run)
    missing = given.set_column(
        given.schema.get_field_index("precursor_mz"), "precursor_mz", pa.nulls(given.num_rows, pa.float64())
    )
    kw = dict(acquisition_id=0, level=2, rt_window=15.0, tolerance=0.35, tolerance_unit="Da")
    gated = oracle.extract_xic_scan_major(peaks, given, **kw)
    ungated = oracle.extract_xic_scan_major(peaks, missing, **kw)
    assert ungated.num_rows > gated.num_rows  # the extractor matched every window
    got = extract_xic_scan_major(bundle, missing, **kw)
    assert got.num_rows == gated.num_rows
    assert got["mz_observed"].to_pylist() == gated["mz_observed"].to_pylist()


def test_assigned_scans_only_leaves_ms1_alone(tmp_root):
    run, bundle, peaks = _run("dda", tmp_root)
    rows = run.targets.to_pylist()
    ms2 = [s for s in run.scans if s["level"] == 2]
    for r in rows:
        hits = [s for s in ms2 if s["isolation_lower"] <= r["precursor_mz"] <= s["isolation_upper"]]
        r["scan"] = min(hits, key=lambda s: abs(s["rt"] - r["rt_center"]))["scan"]
    targets = pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)
    kw = dict(acquisition_id=0, level=1, rt_window=15.0)
    assert oracle.extract_xic_scan_major(peaks, targets, assigned_scans_only=True, **kw).num_rows == 0
    plain = oracle.extract_xic_scan_major(peaks, targets, **kw)
    oracle.assert_traces_identical(
        extract_xic_scan_major(bundle, targets, assigned_scans_only=True, **kw), plain
    )


def test_an_mz_axis_corrects_observed_mz(tmp_root, tmp_path):
    from constellation.massspec.mz_axis import MZ_AXIS_TABLE, MzAxis, save_mz_axis

    run = planted_run(acquisition="dia", cycles=30, window_width=100.0, error_ppm=0.0)
    shift = 0.004
    shifted = []
    for s in run.scans:
        s = dict(s, mz=s["mz"] + shift)
        shifted.append(s)
    exact = write_bundle(tmp_path, "exact", run.scans)
    write_bundle_manifest(exact)
    off = write_bundle(tmp_path, "off", shifted)
    write_bundle_manifest(off)
    knots = np.arange(100.0, 2100.0, 20.0)
    table = pa.Table.from_pylist(
        [dict(level=lv, mz=float(m), shift_th=shift, se_th=1e-4, coherence=0.9, half_window_th=10.0)
         for lv in (1, 2) for m in knots],
        schema=MZ_AXIS_TABLE,
    )  # fmt: skip
    save_mz_axis(off, MzAxis(key="flat0004", table=table, parameters={"method": "test"}))
    kw = dict(acquisition_id=0, level=2, rt_window=15.0, tolerance=2.0, tolerance_unit="ppm")
    targets = _targets(run)
    want = extract_xic_scan_major(exact, targets, **kw)
    assert want.num_rows > 0
    assert extract_xic_scan_major(off, targets, **kw).num_rows < want.num_rows / 4
    got = extract_xic_scan_major(off, targets, mz_axis="latest", **kw)
    assert got.num_rows == want.num_rows
    np.testing.assert_allclose(got["mz_observed"].to_numpy(), want["mz_observed"].to_numpy(), atol=1e-9)
    with pytest.raises(ValueError, match="m/z axis"):
        extract_xic_scan_major(open_peaks(off).to_table(), targets, mz_axis="latest", **kw)


def test_the_ledger_holds_peakless_scans(tmp_path):
    scans = dia_scans(cycles=4, empty_scan=6)
    bundle = write_bundle(tmp_path, "run", scans)
    write_bundle_manifest(bundle)
    targets = pa.Table.from_pylist([dict(target_id=1, precursor_mz=500.0)], schema=XIC_TARGET_TABLE)
    plan = compile_queries(targets, levels=[1])
    result = sweep(bundle, plan, SweepSettings(tolerances={1: (20.0, "ppm")}))
    assert result.scans.n == len(scans)
    assert result.scans.n_points[result.scans.scan == 6].tolist() == [0]
    assert (result.scans.n_points[result.scans.scan != 6] > 0).all()


def test_sum_rule_adds_the_peaks_in_a_window(tmp_root):
    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=15.0)
    each = _traces(bundle, plan, rule=AllWithin())
    summed = _traces(bundle, plan, rule=SumWithin())
    key = ["target_id", "scan", "ion_type", "position", "fragment_charge"]
    totals = each.group_by(key).aggregate([("intensity", "sum")])
    joined = summed.join(totals, key)
    np.testing.assert_allclose(
        joined["intensity"].to_numpy(), joined["intensity_sum"].to_numpy(), rtol=1e-12
    )
    assert joined.num_rows == summed.num_rows == totals.num_rows


@pytest.mark.parametrize("mode", ["nearest", "keep_unmatched", "all_in_window"])
def test_workers_and_shards_change_nothing(tmp_root, tmp_path, mode):
    run, bundle, _ = _run("staggered", tmp_root)
    plan = compile_queries(_targets(run, nan_rt=True), levels=[1, 2], rt_window=20.0)
    rule = AllWithin() if mode == "all_in_window" else Nearest()
    drop = mode != "keep_unmatched"
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    reducer = XicTraceReducer(drop_unmatched=drop)
    base = sweep(bundle, plan, SweepSettings(tolerances=tol, rule=rule), reducers=[reducer]).table("xic_trace")
    assert base.num_rows > 0
    small = SweepSettings(tolerances=tol, rule=rule, shard_rows=3000, reduce_rows=500, block_rows=300,
                          run_rows=400)  # fmt: skip
    for i, (workers, settings) in enumerate(((1, small), (3, small))):
        work = tmp_path / f"work{i}"
        got = sweep(bundle, plan, settings, reducers=[reducer], workers=workers, work_dir=work)
        assert len(got.parts["xic_trace"]) > 1  # several buckets
        runs = sorted(p.name for p in (got.work / "matches").glob("*.parquet"))
        assert len({r[:11] for r in runs}) > 1 and len(runs) > len({r[:11] for r in runs})  # shards, runs
        oracle.assert_traces_identical(got.table("xic_trace"), base)
        out = got.write("xic_trace", tmp_path / f"out{i}.parquet")
        oracle.assert_traces_identical(pq.read_table(out), base)


def test_a_temporary_work_dir_is_removed(tmp_root):
    run, bundle, peaks = _run("dia", tmp_root)
    kw = dict(acquisition_id=0, level=2, rt_window=15.0, tolerance=0.35, tolerance_unit="Da")
    want = oracle.extract_xic_scan_major(peaks, _targets(run), **kw)
    oracle.assert_traces_identical(extract_xic_scan_major(bundle, _targets(run), workers=2, **kw), want)


def test_too_many_matches_for_memory_go_to_disk(tmp_root):
    import gc

    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    reducer = XicTraceReducer()
    base = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[reducer])
    assert base.matches is not None  # held in memory
    tight = SweepSettings(tolerances=tol, memory_rows=100, run_rows=300, shard_rows=4000)
    spilled = sweep(bundle, plan, tight, reducers=[reducer])
    assert spilled.matches is None and spilled.parts["xic_trace"]  # done on disk
    tmp = spilled.parts["xic_trace"][0].parents[2]
    oracle.assert_traces_identical(spilled.table("xic_trace"), base.table("xic_trace"))
    del spilled
    gc.collect()
    assert not tmp.exists()


def test_an_in_memory_sweep_spills_after_the_window_group_that_passes_its_limit(tmp_root, monkeypatch, tmp_path):
    """The limit was tested after a whole block of points, counting matches
    only: with the run in one block, 942 matches were held at a limit of
    100 (review of 2026-11-09, E3). Now it is tested after every window
    group, on everything held, and the rest of the block goes on on disk."""
    from constellation.massspec.sweep import engine
    from constellation.massspec.sweep.evidence import EvidenceGate
    from constellation.massspec.sweep.reduce import QuerySummaryReducer

    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2])  # no RT window
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    reducers = [QuerySummaryReducer(), XicTraceReducer()]
    gate = EvidenceGate(min_fragments=2, min_consecutive=1)
    base = sweep(bundle, plan, SweepSettings(tolerances=tol, gate=gate), reducers=reducers)
    seen = []
    real = engine._on_temporary_disk

    def spy(*args, resume=None, **kw):
        if resume is not None:
            seen.append([p.rows for p in resume[2]])
        return real(*args, resume=resume, **kw)

    monkeypatch.setattr(engine, "_on_temporary_disk", spy)
    tight = SweepSettings(tolerances=tol, gate=gate, memory_rows=100, block_rows=1 << 20)
    got = sweep(bundle, plan, tight, reducers=reducers, spill_dir=tmp_path)
    assert got.matches is None and len(seen) == 1
    held = seen[0]
    assert sum(held[:-1]) <= tight.memory_rows < sum(held)  # spilled after the part that passed it
    assert got.work.parent == tmp_path
    for name in base.schemas:
        oracle.assert_traces_identical(got.table(name), base.table(name))


class _CellCounter:
    """A reducer that zero-fills and records the cells each call builds."""

    name = "cells_per_call"
    schema = pa.schema([("cells", pa.int64()), ("first", pa.int64()), ("last", pa.int64())])
    gated = False
    zero_fills = True

    def reduce(self, batch):
        n = sum(batch.cells(level).n for level in batch.plan.levels)
        q = batch.plan.global_query(batch.queries)
        first, last = (int(q[0]), int(q[-1])) if q.size else (-1, -1)
        return pa.table({"cells": [n], "first": [first], "last": [last]}, schema=self.schema)


def test_an_in_memory_reduce_is_bucketed_as_one_on_disk_is(tmp_root, tmp_path):
    """The in-memory reduce handed the whole plan to one call: 1,962,000
    cells where the disk path's largest call was 48,000 (review of
    2026-11-09, E3). Both now cut the same buckets, by the rows built."""
    from constellation.massspec.sweep.engine import _reduce_weights
    from constellation.massspec.sweep.evidence import EvidenceGate
    from constellation.massspec.sweep.reduce import QuerySummaryReducer

    run, bundle, _ = _run("dia", tmp_root)
    rows = run.targets.to_pylist()
    many = [dict(r, target_id=r["target_id"] + 1000 * k) for k in range(20) for r in rows]
    plan = compile_queries(pa.Table.from_pylist(many, schema=XIC_TARGET_TABLE), levels=[1, 2])
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    gate = EvidenceGate(min_fragments=2, min_consecutive=1)
    settings = SweepSettings(tolerances=tol, gate=gate, claims=True, reduce_rows=20_000)
    reducers = [_CellCounter(), QuerySummaryReducer(), XicTraceReducer(drop_unmatched=False),
                XicTraceReducer(gated=True, name="passing_trace")]  # fmt: skip
    mem = sweep(bundle, plan, settings, reducers=reducers)
    disk = sweep(bundle, plan, settings, reducers=reducers, work_dir=tmp_path / "w")
    assert mem.matches is not None  # in memory
    calls = mem.table("cells_per_call")
    assert calls.num_rows > 5
    counts = np.bincount(mem.matches.query, minlength=plan.n_queries)
    weights = _reduce_weights(plan, mem.scans, settings, reducers, counts)
    assert max(calls.column("cells").to_pylist()) <= settings.reduce_rows + int(weights.max())
    # Contiguous ascending ranges that cover every query once.
    first, last = calls.column("first").to_numpy(), calls.column("last").to_numpy()
    assert first[0] == 0 and last[-1] == plan.n_queries - 1 and (first[1:] == last[:-1] + 1).all()
    for name in disk.schemas:
        oracle.assert_traces_identical(mem.table(name), disk.table(name))


def test_claims_on_disk_are_the_in_memory_table(tmp_root, tmp_path):
    """A run can end inside a block, whose window groups interleave scan
    numbers, so claims written run by run came out of scan order on disk
    (review of 2026-11-09, D2). Claims are now written for whole blocks."""
    from constellation.massspec.sweep.engine import CLAIMS
    from constellation.massspec.sweep.reduce import QuerySummaryReducer

    run, bundle, _ = _run("dia", tmp_root)
    rows = run.targets.to_pylist()
    rows.append(dict(rows[4], target_id=4444))  # a duplicate: every one of its points claimed twice
    plan = compile_queries(pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE), levels=[1, 2], rt_window=20.0)
    tol = {1: (20.0, "ppm"), 2: (20.0, "ppm")}
    want = sweep(bundle, plan, SweepSettings(tolerances=tol, rule=AllWithin(), claims=True),
                 reducers=[QuerySummaryReducer()]).table(CLAIMS)  # fmt: skip
    assert want.num_rows > 0
    for block_rows, run_rows in ((1 << 20, 20), (1 << 20, 5), (300, 20)):
        settings = SweepSettings(tolerances=tol, rule=AllWithin(), claims=True, block_rows=block_rows,
                                 run_rows=run_rows)  # fmt: skip
        got = sweep(bundle, plan, settings, reducers=[QuerySummaryReducer()], work_dir=tmp_path / "w")
        assert len(list((got.work / "matches").glob("*.parquet"))) > 1  # runs end inside blocks
        assert got.table(CLAIMS).equals(want)


def test_claims_are_made_into_their_table_a_run_at_a_time(tmp_root, tmp_path):
    """Sorting a shard's claims as one held them all at once, and a sweep
    spilled to disk is one shard, so its claims undid the bound that made
    it spill (Codex, 2026-11-10: 1,600 held where main held 160)."""
    from constellation.massspec.sweep import engine
    from constellation.massspec.sweep.engine import CLAIMS
    from constellation.massspec.sweep.reduce import QuerySummaryReducer

    run, bundle, _ = _run("dia", tmp_root)
    rows = run.targets.to_pylist()
    rows.append(dict(rows[4], target_id=4444))
    plan = compile_queries(pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE), levels=[1, 2], rt_window=20.0)
    tol = {1: (20.0, "ppm"), 2: (20.0, "ppm")}
    want = sweep(bundle, plan, SweepSettings(tolerances=tol, rule=AllWithin(), claims=True),
                 reducers=[QuerySummaryReducer()]).table(CLAIMS)  # fmt: skip
    engine.READ_STATS["claims_held"] = 0
    settings = SweepSettings(tolerances=tol, rule=AllWithin(), claims=True, block_rows=100, run_rows=5)
    got = sweep(bundle, plan, settings, reducers=[QuerySummaryReducer()], work_dir=tmp_path / "w")
    assert got.table(CLAIMS).equals(want)
    runs = [pq.read_metadata(f).num_rows for f in (got.work / "claims").glob("*.parquet")]
    assert len(runs) > 10
    assert engine.READ_STATS["claims_held"] == max(runs) < want.num_rows / 5


def test_a_spill_with_no_room_is_refused_before_it_starts(tmp_root, tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    from constellation.massspec.sweep import engine

    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2])
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(engine.shutil, "disk_usage", lambda path: usage(10, 10, 0))
    with pytest.raises(OSError, match=str(tmp_path)):
        sweep(bundle, plan, SweepSettings(tolerances={1: (20.0, "ppm"), 2: (0.35, "Da")}, memory_rows=100),
              reducers=[XicTraceReducer()], spill_dir=tmp_path)  # fmt: skip
    assert not list(tmp_path.glob("sweep-*"))
    monkeypatch.setattr(engine.shutil, "disk_usage", shutil.disk_usage)


# ── eligibility and the reduce's buckets ─────────────────────────────


def _brute_pairs(plan, s, level, queries):
    """Every (query, ledger row) pair by the definition, one at a time."""
    channels = plan.channels_for_level(level)
    out = []
    for q in queries:
        if channels.counts()[q] == 0:
            continue
        for r in range(s.n):
            if s.level[r] != level or s.n_points[r] == 0:
                continue
            if not plan.rt_lo[q] <= s.rt[r] <= plan.rt_hi[q]:
                continue
            if level > 1:
                lo, hi, p = s.lower[r], s.upper[r], plan.precursor_mz[q]
                if not (np.isnan(lo) or np.isnan(hi) or np.isnan(p) or lo <= p <= hi):
                    continue
                a = plan.assigned_scan[q]
                if a >= 0 and a != s.scan[r]:
                    continue
            out.append((int(q), int(r)))
    order = s.order()
    return sorted(out, key=lambda qr: (qr[0], order[qr[1]]))


@pytest.mark.parametrize("name", ["staggered", "dda"])
def test_eligible_pairs_are_the_definitions(tmp_root, name):
    """The reduce's pairs, built from per-query ranges of scans in time
    order rather than a queries × scans product filtered, are exactly the
    pairs the definition gives, and their counts are their sizes."""
    from constellation.massspec.sweep.reduce import Matches, QueryBatch

    run, bundle, _ = _run(name, tmp_root)
    rows = _targets(run, nan_rt=True).to_pylist()
    ms2 = [s for s in run.scans if s["level"] == 2]
    for i, r in enumerate(rows):
        if i % 3 == 1:  # an explicit interval, edges on scans' own times
            r["rt_start"], r["rt_end"] = ms2[5 + i]["rt"], ms2[40 + i]["rt"]
        if i % 4 == 2:  # an assigned scan, inside or outside its window and interval
            r["scan"] = ms2[(7 * i) % len(ms2)]["scan"]
    rows.append(dict(rows[0], target_id=900, rt_center=None, rt_start=None, rt_end=None))  # global
    targets = pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    for assigned in (False, True):
        plan = compile_queries(targets, levels=[1, 2], rt_window=15.0, assigned_scans_only=assigned)
        settings = SweepSettings(tolerances=tol)
        ledger = sweep(bundle, plan, settings).scans
        every = np.arange(plan.n_queries, dtype=np.int64)
        batch = QueryBatch(plan=plan, queries=every, matches=Matches.empty(), scans=ledger,
                           scan_order=ledger.order(), settings=settings)  # fmt: skip
        some = every[::2]
        for level in (1, 2):
            want = _brute_pairs(plan, ledger, level, every)
            assert want
            q, r = batch.eligible(level)
            assert list(zip(q.tolist(), r.tolist())) == want
            counts = np.bincount([p[0] for p in want], minlength=plan.n_queries)
            assert batch.eligible_counts(level).tolist() == counts.tolist()
            q, r = batch.eligible(level, some)
            assert list(zip(q.tolist(), r.tolist())) == [p for p in want if p[0] in set(some.tolist())]


def test_reduce_buckets_are_bounded_by_the_rows_they_build(tmp_root, tmp_path):
    """Buckets were cut by matches alone, so unmatched queries piled into
    one; their zero-filled rows, or the pairs a gate walks, now weigh too."""
    from constellation.massspec.sweep.engine import _reduce_weights
    from constellation.massspec.sweep.evidence import EvidenceGate

    run, bundle, _ = _run("dia", tmp_root)
    far = [dict(target_id=1000 + i, precursor_mz=1900.0 + i, precursor_charge=2) for i in range(40)]
    targets = pa.Table.from_pylist(run.targets.to_pylist() + far, schema=XIC_TARGET_TABLE)
    plan = compile_queries(targets, levels=[1])  # global XICs
    tol = {1: (20.0, "ppm")}
    fill = XicTraceReducer(drop_unmatched=False)
    base = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[fill])
    matches = np.bincount(base.matches.query, minlength=plan.n_queries)
    unmatched = matches == 0
    assert unmatched.sum() == len(far)
    plain = _reduce_weights(plan, base.scans, SweepSettings(tolerances=tol), [XicTraceReducer()], matches)
    assert plain.tolist() == matches.tolist()
    gated = _reduce_weights(plan, base.scans, SweepSettings(tolerances=tol, gate=EvidenceGate()), [], matches)
    assert (gated[unmatched] == 0).all() and (gated[~unmatched] > matches[~unmatched]).all()
    filled = _reduce_weights(plan, base.scans, SweepSettings(tolerances=tol), [fill], matches)
    assert (filled[unmatched] > 0).all()
    small = SweepSettings(tolerances=tol, reduce_rows=400, shard_rows=4000)
    got = sweep(bundle, plan, small, reducers=[fill], work_dir=tmp_path / "w")
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in got.parts["xic_trace"]]
    assert max(sizes) <= small.reduce_rows + int(filled.max())
    oracle.assert_traces_identical(got.table("xic_trace"), base.table("xic_trace"))


def test_an_acquisition_without_points_sweeps_on_disk_too(tmp_path):
    from constellation.massspec.sweep.evidence import EvidenceGate
    from constellation.massspec.sweep.reduce import QuerySummaryReducer

    scans = [dict(s, mz=np.empty(0), intensity=np.empty(0)) for s in dia_scans(cycles=3)]
    bundle = write_bundle(tmp_path, "empty", scans)
    write_bundle_manifest(bundle)
    targets = pa.Table.from_pylist([dict(target_id=1, precursor_mz=500.0, precursor_charge=2)],
                                   schema=XIC_TARGET_TABLE)  # fmt: skip
    plan = compile_queries(targets, levels=[1])
    settings = SweepSettings(tolerances={1: (20.0, "ppm")}, claims=True, gate=EvidenceGate(min_isotopes=2))
    reducers = [XicTraceReducer(), QuerySummaryReducer()]
    for i, kw in enumerate(({}, dict(work_dir=tmp_path / "w"), dict(workers=2))):
        result = sweep(bundle, plan, settings, reducers=reducers, **kw)
        assert result.table("xic_trace").num_rows == 0
        assert result.table("query_summary")["passed"].to_pylist() == [False]
        assert pq.read_table(result.write("point_claims", tmp_path / f"c{i}.parquet")).num_rows == 0


@pytest.mark.parametrize("on_disk", [False, True])
def test_extract_xic_records_its_tolerances_in_memory_and_written(tmp_root, tmp_path, on_disk):
    """The Python API stamps the tolerances itself, so Counter clamps to a
    trace's own whether the CLI wrote it or a notebook did."""
    import argparse

    from constellation.massspec.cli.counter import _resolve_collide_ppm
    from constellation.massspec.quant.chromatogram import extract_xic

    run, bundle, _ = _run("dia", tmp_root)
    kw = dict(acquisition_id=0, rt_window=15.0, summary=True, claims=True)
    if on_disk:
        kw["work_dir"] = tmp_path / "w"
    one = extract_xic(bundle, run.targets, levels=[1], tolerances={1: (8.0, "ppm")}, **kw)
    two = extract_xic(bundle, run.targets, levels=[1, 2], tolerances={1: (8.0, "ppm"), 2: (0.35, "Da")}, **kw)
    for name in ("xic_trace", "query_summary", "point_claims"):
        meta = one.table(name).schema.metadata
        assert meta[b"x.massspec.extraction_tolerance"] == b"8.0", name
        assert meta[b"x.massspec.extraction_tolerance_unit"] == b"ppm", name
        written = pq.read_schema(one.write(name, tmp_path / f"{name}.parquet")).metadata
        assert written[b"x.massspec.extraction_tolerance"] == b"8.0", name
    meta = pq.read_schema(two.write("xic_trace", tmp_path / "two.parquet")).metadata
    assert meta[b"x.massspec.extraction_tolerance.ms1"] == b"8.0"
    assert meta[b"x.massspec.extraction_tolerance.ms2"] == b"0.35"
    args = argparse.Namespace(extraction_tolerance_ppm=None, trace=tmp_path / "xic_trace.parquet",
                              collide_ppm=20.0, no_progress=True)  # fmt: skip
    assert _resolve_collide_ppm(args) == 8.0


# ── the same plan on disk; outputs bound by name (review of 2026-11-06) ──


_PROBE_SCHEMA = pa.schema([("query", pa.int64()), ("rt_center", pa.float64()), ("scan", pa.int32()),
                           ("settings", pa.string())])  # fmt: skip


class _PlanProbe:
    """What a reducer sees of its plan: the targets' columns and settings."""

    name = "probe"
    schema = _PROBE_SCHEMA
    gated = False

    def reduce(self, batch):
        import json

        t = batch.plan.targets.take(pa.array(batch.queries))
        return pa.table({
            "query": pa.array(batch.queries, pa.int64()), "rt_center": t.column("rt_center"),
            "scan": t.column("scan"),
            "settings": [json.dumps(batch.plan.settings.to_dict())] * batch.queries.size,
        }, schema=_PROBE_SCHEMA)  # fmt: skip


class _Swapped:
    """Returns its schema's columns in another order."""

    name = "swapped"
    schema = pa.schema([("a", pa.float64()), ("b", pa.float64())])
    gated = False

    def reduce(self, batch):
        n = batch.queries.size
        return pa.table({"b": np.full(n, 2.0), "a": np.full(n, 1.0)})


class _Wrong:
    name = "wrong"
    schema = pa.schema([("a", pa.float64())])
    gated = False

    def reduce(self, batch):
        return pa.table({"z": np.zeros(batch.queries.size)})


def test_reducers_see_the_same_plan_in_memory_spilled_and_in_workers(tmp_root, tmp_path):
    """On disk the plan was rebuilt from the query table, without the
    targets' other columns and with default settings."""
    from constellation.massspec.sweep import ChannelSettings

    run, bundle, _ = _run("dia", tmp_root)
    targets = _targets(run).set_column(
        _targets(run).schema.get_field_index("scan"), "scan", pa.array(range(run.targets.num_rows), pa.int32())
    )
    settings = ChannelSettings(n_isotopes=2, charge_range=(2, 3), max_fragment_charge=1)
    plan = compile_queries(targets, levels=[1, 2], channels=settings, rt_window=20.0)
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    base = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[_PlanProbe()]).table("probe")
    assert base.column("scan").null_count == 0 and base.column("rt_center").null_count == 0
    spilled = sweep(bundle, plan, SweepSettings(tolerances=tol, memory_rows=10, run_rows=50),
                    reducers=[_PlanProbe()])  # fmt: skip
    assert spilled.matches is None
    assert spilled.table("probe").equals(base)
    for workers in (1, 3):
        got = sweep(bundle, plan, SweepSettings(tolerances=tol, reduce_rows=50), reducers=[_PlanProbe()],
                    workers=workers, work_dir=tmp_path / f"w{workers}")  # fmt: skip
        assert got.table("probe").equals(base)


def test_outputs_are_bound_to_their_schema_by_name(tmp_root, tmp_path):
    """Columns returned in another order were written swapped, on disk."""
    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    tol = {2: (0.35, "Da")}
    for kw in ({}, dict(workers=2, work_dir=tmp_path / "w")):
        result = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[_Swapped()], **kw)
        for table in (result.table("swapped"), pq.read_table(result.write("swapped", tmp_path / "out.parquet"))):
            assert set(table.column("a").to_pylist()) == {1.0} and set(table.column("b").to_pylist()) == {2.0}
    with pytest.raises(ValueError, match="reducer 'wrong'.*missing \\['a'\\].*not in its schema \\['z'\\]"):
        sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[_Wrong()])


def test_a_scan_without_a_retention_time_leaves_the_others_matched(tmp_root):
    """One NaN retention time made the block's min/max NaN, and no query
    was active at that level for the whole block."""
    run, _, peaks = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    base = _traces(peaks, plan)
    rt = peaks.column("rt").to_numpy().copy()
    first_ms2 = int(np.flatnonzero(peaks.column("level").to_numpy() == 2)[0])
    lost_scan = peaks.column("scan")[first_ms2].as_py()
    rt[peaks.column("scan").to_numpy() == lost_scan] = np.nan
    with_nan = peaks.set_column(peaks.schema.get_field_index("rt"), "rt", pa.array(rt))
    got = _traces(with_nan, plan, block_rows=1 << 30)
    keep = pc.not_equal(base.column("scan"), lost_scan)
    assert got.num_rows == base.filter(keep).num_rows > 0


def test_a_table_of_points_past_memory_rows_is_refused(tmp_root):
    """A table of points is swept in memory and has no disk path to spill
    to; it grew without bound."""
    run, _, peaks = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    with pytest.raises(MemoryError, match="write it as a bundle"):
        sweep(peaks, plan, SweepSettings(tolerances={2: (0.35, "Da")}, memory_rows=5, block_rows=200),
              reducers=[XicTraceReducer()])  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eligibility_on_random_ledgers_is_the_definition(seed):
    """Random ledgers with what real runs mix: shared DIA windows, one
    window per DDA scan, scans without bounds or without a time or points,
    queries without a precursor m/z, assigned scans in and out of reach.
    The vectorized index must give exactly the definition's pairs (it
    replaced a regrouping that cost windows × scans per call)."""
    from constellation.massspec.sweep.reduce import Ledger, Matches, QueryBatch

    rng = np.random.default_rng(seed)
    n = 400
    level = np.where(rng.random(n) < 0.15, 1, 2).astype(np.int8)
    dia = rng.integers(0, 6, n)
    dda = rng.random(n) < 0.4
    centre = np.where(dda, rng.uniform(400, 1000, n), 400 + 100 * dia + 50)
    width = np.where(dda, 0.7, 50.0)
    lower, upper = centre - width, centre + width
    unbounded = rng.random(n) < 0.05
    lower[(level == 1) | unbounded], upper[(level == 1) | unbounded] = np.nan, np.nan
    rt = np.sort(rng.uniform(0, 600, n))
    rt[rng.random(n) < 0.03] = np.nan
    points = np.where(rng.random(n) < 0.05, 0, 100).astype(np.int64)
    ledger = Ledger(scan=np.arange(n, dtype=np.int64) * 3 + 11, level=level, rt=rt, lower=lower, upper=upper,
                    n_points=points)  # fmt: skip
    m = 60
    rows = []
    for i in range(m):
        row = dict(target_id=i, modified_sequence="PEPTIDEK", precursor_charge=2)
        if i % 7 != 3:
            row["precursor_mz"] = float(rng.uniform(380, 1020))
        if i % 3 == 0:
            row["rt_center"] = float(rng.uniform(0, 600))
        elif i % 3 == 1:
            t0 = float(rng.uniform(0, 500))
            row["rt_start"], row["rt_end"] = t0, t0 + float(rng.uniform(0, 120))
        if i % 5 == 0:
            row["scan"] = int(ledger.scan[rng.integers(0, n)]) if i % 10 else 999_999  # one not in the ledger
        rows.append(row)
    targets = pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)
    settings = SweepSettings(tolerances={1: (20.0, "ppm"), 2: (0.35, "Da")})
    for assigned in (False, True):
        plan = compile_queries(targets, levels=[1, 2], rt_window=40.0, assigned_scans_only=assigned)
        every = np.arange(plan.n_queries, dtype=np.int64)
        batch = QueryBatch(plan=plan, queries=every, matches=Matches.empty(), scans=ledger,
                           scan_order=ledger.order(), settings=settings)  # fmt: skip
        for lv in (1, 2):
            want = _brute_pairs(plan, ledger, lv, every)
            q, r = batch.eligible(lv)
            assert list(zip(q.tolist(), r.tolist())) == want, (seed, assigned, lv)
            counts = np.bincount([p[0] for p in want], minlength=plan.n_queries)
            assert batch.eligible_counts(lv).tolist() == counts.tolist()


def test_the_window_index_is_built_once_per_ledger_and_level(tmp_root, monkeypatch):
    from constellation.massspec.sweep import reduce as reduce_module
    from constellation.massspec.sweep.reduce import Matches, QueryBatch

    calls = []
    real = reduce_module.window_index
    monkeypatch.setattr(reduce_module, "window_index", lambda *a, **k: calls.append(a[-1]) or real(*a, **k))
    run, bundle, _ = _run("dda", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=15.0)
    settings = SweepSettings(tolerances={1: (20.0, "ppm"), 2: (0.35, "Da")})
    ledger = sweep(bundle, plan, settings).scans
    calls.clear()
    every = np.arange(plan.n_queries, dtype=np.int64)
    batch = QueryBatch(plan=plan, queries=every, matches=Matches.empty(), scans=ledger,
                       scan_order=ledger.order(), settings=settings)  # fmt: skip
    for _ in range(3):
        batch.eligible(2)
        batch.eligible_counts(2)
        batch.eligible_cells(1)
    assert sorted(calls) == [1, 2]


def test_a_bucket_reads_about_its_own_rows_from_the_runs(tmp_root, tmp_path):
    """Each bucket decoded whole 2^18-row groups of every run, which grows
    with the square of the matches (review of 2026-11-06): rows decoded are
    now bounded by the matches plus two row groups per run and bucket."""
    from constellation.massspec.sweep import engine

    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    small = dict(run_rows=300, reduce_rows=200, shard_rows=4000, run_row_group_rows=16)
    engine.READ_STATS["rows_decoded"] = 0
    work = tmp_path / "w"
    got = sweep(bundle, plan, SweepSettings(tolerances=tol, **small), reducers=[XicTraceReducer()],
                workers=1, work_dir=work)  # fmt: skip
    runs = sorted((got.work / "matches").glob("*.parquet"))
    matches = sum(pq.read_metadata(f).num_rows for f in runs)
    buckets = len(got.parts["xic_trace"])
    assert len(runs) > 3 and buckets > 3
    decoded = engine.READ_STATS["rows_decoded"]
    assert matches <= decoded <= matches + 2 * len(runs) * buckets * small["run_row_group_rows"]
    want = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[XicTraceReducer()]).table("xic_trace")
    oracle.assert_traces_identical(got.table("xic_trace"), want)


# ── what a reducer reads: cells, scan context, several outputs (2026-11-07) ──


def _batch_of(bundle, plan, settings):
    """The whole plan as one batch, in memory."""
    from constellation.massspec.sweep.reduce import QueryBatch

    result = sweep(bundle, plan, settings)
    every = np.arange(plan.n_queries, dtype=np.int64)
    return QueryBatch(plan=plan, queries=every, matches=result.matches, scans=result.scans,
                      scan_order=result.scans.order(), settings=settings)  # fmt: skip


@pytest.mark.parametrize("name", ["dia", "dda"])
def test_cells_are_every_eligible_cell_with_what_matched_there(tmp_root, name):
    """A cell view for scorers: every channel in every scan a query was
    eligible in, matched or not, errors signed observed − theoretical."""
    run, bundle, _ = _run(name, tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    settings = SweepSettings(tolerances={1: (20.0, "ppm"), 2: (0.35, "Da")})
    batch = _batch_of(bundle, plan, settings)
    for level in (1, 2):
        cells = batch.cells(level)
        cq, cr, ci = batch.eligible_cells(level)
        assert cells.n == cq.size  # nearest: one row per eligible cell
        assert sorted(zip(cells.query.tolist(), cells.row.tolist(), cells.channel.tolist())) == sorted(
            zip(cq.tolist(), cr.tolist(), ci.tolist()))  # fmt: skip
        m = batch.matches
        assert int(cells.matched.sum()) == int((m.level == level).sum())
        hit = cells.matched
        assert np.allclose(cells.mz_error_da[hit], cells.mz_observed[hit] - cells.mz_theoretical[hit])
        assert np.isnan(cells.mz_error_ppm[~hit]).all() and (cells.intensity[~hit] == 0).all()
        assert np.array_equal(cells.scan, batch.scans.scan[cells.row])
        table = cells.table()
        assert table.num_rows == cells.n and table.schema.names[0] == "query"


def test_cells_keep_every_match_of_a_cell_under_rule_all(tmp_root):
    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    settings = SweepSettings(tolerances={2: (0.35, "Da")}, rule=AllWithin())
    batch = _batch_of(bundle, plan, settings)
    cells = batch.cells(2, fill=False)
    assert cells.n == batch.matches.n and cells.matched.all()
    filled = batch.cells(2)
    key = list(zip(filled.query.tolist(), filled.row.tolist(), filled.channel.tolist()))
    assert len(key) > len(set(key))  # some cell holds several matches
    eligible = batch.eligible_cells(2)
    assert set(key) == set(zip(*(a.tolist() for a in eligible)))


def test_the_ledger_carries_each_scans_tic_and_injection_time(tmp_path):
    """What a score against the whole scan needs (the share of the TIC a
    query explains, ion counts from injection time), from the bundle."""
    scans = dia_scans(cycles=3)
    for i, s in enumerate(scans):
        s["tic"], s["iit"] = 1000.0 + i, (10.0 if i % 2 else None)
    bundle = write_bundle_manifest(write_bundle(tmp_path, "ctx", scans))
    run = planted_run(acquisition="dia", cycles=3, window_width=100.0)
    plan = compile_queries(run.targets, levels=[1])
    result = sweep(bundle, plan, SweepSettings(tolerances={1: (20.0, "ppm")}))
    ledger = result.scans
    by_scan = {s["scan"]: s for s in scans}
    assert ledger.tic.tolist() == [by_scan[int(x)]["tic"] for x in ledger.scan]
    want = [by_scan[int(x)]["iit"] for x in ledger.scan]
    assert [None if np.isnan(v) else v for v in ledger.iit.tolist()] == want
    assert ledger.table().column("tic").to_pylist() == ledger.tic.tolist()
    table_run = sweep(open_peaks(bundle).to_table(), plan, SweepSettings(tolerances={1: (20.0, "ppm")}))
    assert np.isnan(table_run.scans.tic).all()  # a table of points records none


_TWO_A = pa.schema([("query", pa.int64()), ("n", pa.int64())])
_TWO_B = pa.schema([("query", pa.int64()), ("total", pa.float64())])


class _TwoOutputs:
    """One pass, two tables: per query, its cells, and its summed intensity."""

    name = "two"
    gated = False
    outputs = {"cell_counts": _TWO_A, "intensity": _TWO_B}

    def reduce(self, batch):
        cells = batch.cells(2)
        q = batch.queries
        at = np.searchsorted(q, cells.query)
        return {
            "cell_counts": pa.table({"query": q, "n": np.bincount(at, minlength=q.size)}, schema=_TWO_A),
            "intensity": pa.table({"query": q, "total": np.bincount(at, weights=cells.intensity, minlength=q.size)},
                                  schema=_TWO_B),  # fmt: skip
        }


def test_a_reducer_of_several_outputs_is_the_same_in_memory_and_in_workers(tmp_root, tmp_path):
    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    tol = {2: (0.35, "Da")}
    base = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[_TwoOutputs(), XicTraceReducer()])
    assert set(base.schemas) == {"cell_counts", "intensity", "xic_trace"}
    assert base.table("cell_counts").column("n").to_numpy().sum() > 0
    for workers in (1, 3):
        got = sweep(bundle, plan, SweepSettings(tolerances=tol, reduce_rows=100), reducers=[_TwoOutputs()],
                    workers=workers, work_dir=tmp_path / f"w{workers}")  # fmt: skip
        assert len(got.parts["intensity"]) > 1
        for name in ("cell_counts", "intensity"):
            assert got.table(name).equals(base.table(name))
            assert pq.read_table(got.write(name, tmp_path / f"{name}{workers}.parquet")).equals(base.table(name))


def test_output_names_are_unique_across_reducers(tmp_root):
    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    clash = _TwoOutputs()
    clash.outputs = {"xic_trace": _TWO_A, "intensity": _TWO_B}
    with pytest.raises(ValueError, match="both name an output 'xic_trace'"):
        sweep(bundle, plan, SweepSettings(tolerances={2: (0.35, "Da")}), reducers=[XicTraceReducer(), clash])


# ── part of a run (2026-11-07) ──────────────────────────────────────


def _unranged(table: pa.Table) -> pa.Table:
    """A trace without the range stamp a ranged sweep adds to its metadata."""
    meta = {k: v for k, v in (table.schema.metadata or {}).items() if not k.startswith(b"x.massspec.sweep.")}
    return table.replace_schema_metadata(meta)


def test_a_scan_range_sweeps_exactly_those_scans(tmp_root, tmp_path):
    """Only the scans read are eligible: the zero-filled trace and the
    summary's counts cover them alone, in memory and on disk alike."""
    from constellation.massspec.sweep.reduce import QuerySummaryReducer

    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2])  # global: every scan
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    scans = sorted(s["scan"] for s in run.scans)
    first, last = scans[len(scans) // 4], scans[len(scans) // 2]
    reducers = [XicTraceReducer(drop_unmatched=False), QuerySummaryReducer()]
    full = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=reducers).table("xic_trace")
    inside = pc.and_(pc.greater_equal(full.column("scan"), first), pc.less_equal(full.column("scan"), last))
    want = full.filter(inside)
    runs = [sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=reducers, scan_range=(first, last))]
    for workers in (1, 3):
        runs.append(sweep(bundle, plan, SweepSettings(tolerances=tol, shard_rows=3000, reduce_rows=500),
                          reducers=reducers, scan_range=(first, last), workers=workers,
                          work_dir=tmp_path / f"w{workers}"))  # fmt: skip
    for got in runs:
        trace = got.table("xic_trace")
        oracle.assert_traces_identical(_unranged(trace), want)
        assert trace.schema.metadata[b"x.massspec.sweep.scan_range"] == f"{first},{last}".encode()
        summary = got.table("query_summary")
        ms2 = sum(1 for s in run.scans if s["level"] == 2 and first <= s["scan"] <= last)
        assert max(summary.column("n_eligible_ms2").to_pylist()) <= ms2


def test_an_rt_range_is_the_scans_it_spans(tmp_root):
    run, bundle, peaks = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2])
    settings = SweepSettings(tolerances={2: (0.35, "Da")})
    times = sorted((s["rt"], s["scan"]) for s in run.scans)
    t0, t1 = times[10][0], times[60][0]
    in_range = [scan for t, scan in times if t0 <= t <= t1]
    by_rt = sweep(bundle, plan, settings, reducers=[XicTraceReducer()], rt_range=(t0, t1)).table("xic_trace")
    by_scan = sweep(bundle, plan, settings, reducers=[XicTraceReducer()],
                    scan_range=(min(in_range), max(in_range))).table("xic_trace")  # fmt: skip
    oracle.assert_traces_identical(_unranged(by_rt), _unranged(by_scan))
    on_table = sweep(peaks, plan, settings, reducers=[XicTraceReducer()], rt_range=(t0, t1)).table("xic_trace")
    oracle.assert_traces_identical(_unranged(on_table), _unranged(by_rt))
    empty = sweep(bundle, plan, settings, reducers=[XicTraceReducer()], rt_range=(1e9, 2e9)).table("xic_trace")
    assert empty.num_rows == 0
    with pytest.raises(ValueError, match="not both"):
        sweep(bundle, plan, settings, scan_range=(1, 2), rt_range=(0.0, 1.0))


# ── at scale (2026-11-09): runs capped, ranges sharded, pairs chunked, cells merged ──


def test_no_run_file_holds_more_than_run_rows(tmp_root, tmp_path):
    """A block's matches are flushed group by group and written in runs of
    at most run_rows, however many one block makes."""
    run, bundle, _ = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2])
    tol = {1: (20.0, "ppm"), 2: (0.35, "Da")}
    base = sweep(bundle, plan, SweepSettings(tolerances=tol), reducers=[XicTraceReducer()]).table("xic_trace")
    capped = SweepSettings(tolerances=tol, run_rows=97, block_rows=1 << 20)
    got = sweep(bundle, plan, capped, reducers=[XicTraceReducer()], workers=1, work_dir=tmp_path / "w")
    oracle.assert_traces_identical(got.table("xic_trace"), base)
    runs = sorted((got.work / "matches").glob("*.parquet"))
    assert len(runs) > 1
    assert max(pq.read_metadata(f).num_rows for f in runs) <= 97


def test_a_scan_range_is_cut_into_shards_of_its_own(tmp_root, tmp_path):
    from constellation.massspec.bundle import scan_range_shards

    run, bundle, _ = _run("dia", tmp_root)
    whole = scan_range_shards(bundle)
    scans = sorted(s["scan"] for s in run.scans)
    lo, hi = scans[len(scans) // 3], scans[len(scans) // 3 + 40]
    assert sum(1 for a, b in whole if a < hi and b > lo) == 1  # trimmed shards: one process
    shards = scan_range_shards(bundle, scan_range=(lo, hi))
    assert len(shards) > 1
    assert shards[0][0] == lo and shards[-1][1] == hi
    assert all(a < b for a, b in shards) and all(b == c for (_, b), (c, _) in zip(shards, shards[1:]))
    plan = compile_queries(_targets(run), levels=[2])
    settings = SweepSettings(tolerances={2: (0.35, "Da")})
    want = sweep(bundle, plan, settings, reducers=[XicTraceReducer()], scan_range=(lo, hi - 1)).table("xic_trace")
    got = sweep(bundle, plan, settings, reducers=[XicTraceReducer()], scan_range=(lo, hi - 1), workers=2,
                work_dir=tmp_path / "w")  # fmt: skip
    oracle.assert_traces_identical(got.table("xic_trace"), want)
    assert len(list((got.work / "scans").glob("*.parquet"))) > 1


def test_eligible_pairs_are_counted_in_chunks_bounded_by_the_pairs_they_build(tmp_root, monkeypatch):
    """On DDA every scan is a window of its own: the parent counts a query's
    pairs without building every (window, query) pair at once."""
    from constellation.massspec.sweep import reduce as reduce_mod

    run, bundle, _ = _run("dda", tmp_root)
    plan = compile_queries(_targets(run), levels=[2])
    result = sweep(bundle, plan, SweepSettings(tolerances={2: (0.35, "Da")}))
    every = np.arange(plan.n_queries, dtype=np.int64)
    want = reduce_mod.eligible_counts(plan, result.scans, 2, every)
    built: list[int] = []
    real = reduce_mod.eligible_pairs

    def recording(*args, **kwargs):
        out = real(*args, **kwargs)
        built.append(int(out[0].size))
        return out

    monkeypatch.setattr(reduce_mod, "eligible_pairs", recording)
    monkeypatch.setattr(reduce_mod, "PAIR_BUDGET", 1)
    got = reduce_mod.eligible_counts(plan, result.scans, 2, every)
    np.testing.assert_array_equal(got, want)
    assert len(built) > 1
    windows = reduce_mod._windows_per_query(plan, result.scans, 2, every)
    assert max(built) <= 1 + int(windows.max())


@pytest.mark.parametrize("rule", [Nearest(), AllWithin(), SumWithin()])
@pytest.mark.parametrize("name", ["dia", "dda"])
def test_cells_merge_is_the_sort_it_replaced(tmp_root, name, rule):
    """The merge of matches and unmatched cells gives what sorting both on
    four keys gave, row for row."""
    run, bundle, _ = _run(name, tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    batch = _batch_of(bundle, plan, SweepSettings(tolerances={1: (20.0, "ppm"), 2: (0.35, "Da")}, rule=rule))
    batch = dataclasses.replace(batch, matches=batch.matches.sorted(batch.scan_order[batch.match_rows()]))
    for level in (1, 2):
        for subset in (None, batch.queries[::3]):
            for fill in (True, False):
                got = batch.cells(level, fill=fill, queries=subset)
                m = batch.matches
                sel = (m.level == level) & (np.ones(m.n, bool) if subset is None else np.isin(m.query, subset))
                at = np.flatnonzero(sel)
                parts = dict(query=m.query[at], row=batch.scans.rows_of(m.scan[at]), channel=m.channel[at],
                             matched=np.ones(at.size, bool), point_index=m.point_index[at], n_points=m.n_points[at],
                             mz_observed=m.mz[at], intensity=m.intensity[at])  # fmt: skip
                cq, cr, ci = (batch.eligible_cells_of(level, batch.queries if subset is None else subset) if fill
                              else (np.empty(0, np.int64),) * 3)  # fmt: skip
                want = batch._cells_by_sort(level, plan.channels_for_level(level), parts, cq, cr, ci)
                for column in ("query", "row", "channel", "matched", "point_index", "n_points", "mz_observed",
                               "intensity"):  # fmt: skip
                    np.testing.assert_array_equal(getattr(got, column), getattr(want, column), err_msg=column)
