"""The sweep read precursor-major (`massspec.sweep.precursor_major`) is held to
the scan-major sweep: the same ledger and, for every query, the same rows
(every float bit and null, in the same order within the query); outputs
come in units of queries by precursor m/z, so they are compared sorted
by query (or target). Within the precursor-major sweep, outputs are the
same bytes whatever the worker count or batch size."""

from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute  # noqa: F401
import pyarrow.parquet as pq
import pytest

import _xic_scan_major_oracle as oracle
from _massspec_fixtures import planted_run, write_bundle, write_bundle_manifest
from constellation.massspec.mz_axis import MzAxis, save_mz_axis
from constellation.massspec.precursor_major import build_precursor_major
from constellation.massspec.quant.chromatogram import XicTraceReducer
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep import compile_queries
from constellation.massspec.sweep.engine import SweepSettings, sweep
from constellation.massspec.sweep.evidence import EvidenceGate
from constellation.massspec.sweep.precursor_major import LAYOUT_KEY, precursor_units
from constellation.massspec.sweep.reduce import QuerySummaryReducer
from constellation.massspec.sweep.rules import AllWithin, Nearest, SumWithin

_RUNS = {
    "dia": dict(acquisition="dia", cycles=30, window_width=100.0),
    "staggered": dict(acquisition="dia", staggered=True, cycles=30, window_width=100.0),
    "dda": dict(acquisition="dda", cycles=40),
}
_TOL = {1: (20.0, "ppm"), 2: (0.35, "Da")}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("precursor_major"))


def _bundle(root: Path, name: str, scans) -> Path:
    bundle = write_bundle_manifest(write_bundle(root, name, scans, row_group_size=97))
    build_precursor_major(bundle)
    return bundle


@functools.lru_cache(maxsize=None)
def _run(name: str, tmp_root: str):
    run = planted_run(**_RUNS[name])
    return run, _bundle(Path(tmp_root), name, run.scans)


def _targets(run, extra=()) -> pa.Table:
    rows = run.targets.to_pylist()
    # A copy of each with its sequence reversed: matched, but not planted.
    rows += [dict(r, target_id=r["target_id"] + 500, modified_sequence=r["modified_sequence"][-2::-1] + r["modified_sequence"][-1])
             for r in rows if "[" not in r["modified_sequence"]]  # fmt: skip
    return pa.Table.from_pylist(rows + list(extra), schema=XIC_TARGET_TABLE)


def _by(table: pa.Table, key: str) -> pa.Table:
    t = table.take(pa.array(np.argsort(table.column(key).to_numpy(), kind="stable")))
    meta = {k: v for k, v in (t.schema.metadata or {}).items() if k != LAYOUT_KEY}
    return t.replace_schema_metadata(meta)


def _same_rows(got: pa.Table, want: pa.Table, key: str) -> None:
    oracle.assert_traces_identical(_by(got, key), _by(want, key))


def _check(bundle, plan, settings, reducers, **kw):
    """Both layouts, output by output; returns the precursor-major result."""
    a = sweep(bundle, plan, settings, reducers=reducers, **kw)
    b = sweep(bundle, plan, settings, reducers=reducers, layout="precursor_major", **kw)
    for name in a.schemas:
        key = "query" if "query" in a.schemas[name].names else "target_id"
        _same_rows(b.table(name), a.table(name), key)
        assert b.table(name).schema.metadata[LAYOUT_KEY] == b"precursor_major"
    np.testing.assert_array_equal(b.scans.n_points, a.scans.n_points)
    return b


@pytest.mark.parametrize("name", list(_RUNS))
@pytest.mark.parametrize("levels", [(1,), (2,), (1, 2)])
@pytest.mark.parametrize("rule", [Nearest(), AllWithin(), SumWithin()], ids=lambda r: r.name)
def test_each_query_gets_the_rows_the_scan_major_sweep_gives_it(tmp_root, name, levels, rule):
    run, bundle = _run(name, tmp_root)
    plan = compile_queries(_targets(run), levels=list(levels), rt_window=20.0)
    settings = SweepSettings(tolerances=_TOL, rule=rule, unit_cells=300, batch_cells=120)
    got = _check(bundle, plan, settings, [XicTraceReducer(drop_unmatched=False), QuerySummaryReducer()])
    assert got.table("xic_trace").num_rows > 0


@pytest.mark.parametrize("name", list(_RUNS))
def test_options_change_nothing_between_layouts(tmp_root, name):
    run, bundle = _run(name, tmp_root)
    targets = _targets(run)
    scans = sorted(s["scan"] for s in run.scans)
    times = sorted(s["rt"] for s in run.scans)
    for plan_kw, settings_kw, sweep_kw, drop in (
        (dict(rt_window=None), {}, {}, True),
        (dict(rt_window=20.0), dict(tolerances={1: (0.02, "Da"), 2: (500.0, "ppm")}), {}, False),
        (dict(rt_window=20.0), dict(gate=EvidenceGate(min_fragments=2, min_consecutive=1)), {}, True),
        (dict(rt_window=None), {}, dict(scan_range=(scans[len(scans) // 4], scans[len(scans) // 2])), False),
        (dict(rt_window=20.0), {}, dict(rt_range=(times[10], times[-10])), True),
        (dict(rt_window=20.0), dict(unit_cells=50, batch_cells=10), {}, False),
    ):
        plan = compile_queries(targets, levels=[1, 2], **plan_kw)
        settings = replace(SweepSettings(tolerances=_TOL), **settings_kw)
        reducers = [XicTraceReducer(drop_unmatched=drop), QuerySummaryReducer()]
        if settings.gate is not None:
            reducers.append(XicTraceReducer(drop_unmatched=drop, gated=True, name="passing_trace"))
        _check(bundle, plan, settings, reducers, **sweep_kw)


@pytest.mark.parametrize("layout", ["scan_major", "precursor_major"])
def test_a_later_sweep_into_the_same_work_dir_leaves_an_earlier_result_alone(tmp_root, tmp_path, layout):
    """Both layouts cleared and reused ``work_dir``, so an earlier result
    whose table was not yet read gave the later sweep's rows (review of
    2026-11-09, E1). Each sweep now writes a directory of its own."""
    run, bundle = _run("dia", tmp_root)
    work = tmp_path / "w"
    targets = _targets(run)
    reducers = [QuerySummaryReducer(), XicTraceReducer()]
    plan_a = compile_queries(targets, levels=[1, 2], rt_window=20.0)
    plan_b = compile_queries(targets.slice(0, 2), levels=[1, 2], rt_window=20.0)
    want = sweep(bundle, plan_a, SweepSettings(tolerances=_TOL), reducers=reducers, layout=layout)
    a = sweep(bundle, plan_a, SweepSettings(tolerances=_TOL), reducers=reducers, layout=layout, work_dir=work)
    b = sweep(bundle, plan_b, SweepSettings(tolerances={1: (20.0, "ppm"), 2: (0.5, "Da")}), reducers=reducers,
              layout=layout, work_dir=work)  # fmt: skip
    assert a.work != b.work and a.work.parent == b.work.parent == work
    for name in want.schemas:
        oracle.assert_traces_identical(a.table(name), want.table(name))
    assert b.table("query_summary").num_rows == 2


def test_workers_and_batches_change_no_byte(tmp_root, tmp_path):
    run, bundle = _run("staggered", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    reducers = [XicTraceReducer(drop_unmatched=False), QuerySummaryReducer()]
    base = sweep(bundle, plan, SweepSettings(tolerances=_TOL, unit_cells=400), reducers=reducers,
                 layout="precursor_major")  # fmt: skip
    for workers, batch in ((2, 1 << 22), (1, 7), (2, 33)):
        got = sweep(bundle, plan, SweepSettings(tolerances=_TOL, unit_cells=400, batch_cells=batch), reducers=reducers,
                    layout="precursor_major", workers=workers, work_dir=tmp_path / f"w{workers}-{batch}")  # fmt: skip
        for name in base.schemas:
            oracle.assert_traces_identical(got.table(name), base.table(name))


# ── edges: overlapping windows, a precursor on a bound, no bounds, no precursor m/z ──


@functools.lru_cache(maxsize=None)
def _odd_run(tmp_root: str):
    """A DIA run whose windows overlap by 1 Th, with an MS2 scan that has no
    isolation bounds."""
    run = planted_run(acquisition="dia", cycles=30, window_width=100.0, seed=3)
    scans = [dict(s) for s in run.scans]
    for s in scans:
        if s["level"] == 2:
            s["isolation_lower"] -= 0.5
            s["isolation_upper"] += 0.5
    ms2 = [s for s in scans if s["level"] == 2]
    ms2[7]["isolation_lower"] = ms2[7]["isolation_upper"] = None
    return run, scans, _bundle(Path(tmp_root), "odd", scans)


def test_overlapping_windows_a_bound_and_no_bounds(tmp_root):
    run, scans, bundle = _odd_run(tmp_root)
    ms2 = [s for s in scans if s["level"] == 2 and s["isolation_lower"] is not None]
    bound = float(ms2[0]["isolation_upper"])
    first = run.targets.to_pylist()[0]
    extra = [dict(first, target_id=900, precursor_mz=bound),  # exactly on a window's bound
             dict(first, target_id=901, precursor_mz=bound - 0.7)]  # inside the overlap  # fmt: skip
    targets = _targets(run, extra)
    for rule in (Nearest(), AllWithin()):
        plan = compile_queries(targets, levels=[1, 2], rt_window=None)
        settings = SweepSettings(tolerances=_TOL, rule=rule, unit_cells=200, batch_cells=50)
        got = _check(bundle, plan, settings, [XicTraceReducer(drop_unmatched=False), QuerySummaryReducer()])
        summary = got.table("query_summary")
        on_bound = summary.filter(pa.compute.equal(summary.column("target_id"), 900))
        assert on_bound.column("n_eligible_ms2")[0].as_py() > 0


def test_queries_without_a_precursor_mz_are_matched_in_every_window(tmp_root):
    run, _, bundle = _odd_run(tmp_root)
    rows = run.targets.to_pylist()
    rows = [dict(r, precursor_mz=None) if i % 2 else r for i, r in enumerate(rows)]
    targets = pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)
    plan = compile_queries(targets, levels=[2], rt_window=None, derive_precursor_mz=False)
    assert np.isnan(plan.precursor_mz).any()
    _check(bundle, plan, SweepSettings(tolerances=_TOL, unit_cells=500, batch_cells=100), [XicTraceReducer()])


def test_a_scan_without_bounds_does_not_widen_what_a_unit_reads(tmp_path, monkeypatch):
    """A scan without bounds is stored with lower = -inf, first; with it in
    the range of lower bounds read, every row group below a unit's windows
    was decoded and dropped (9.8× the rows; review of 2026-11-09, A3)."""
    from constellation.massspec import precursor_major as layout_module
    from constellation.massspec.precursor_major import find_precursor_major
    from constellation.massspec.sweep import precursor_major as pm

    monkeypatch.setattr(layout_module, "MIN_GROUP_ROWS", 64)
    monkeypatch.setattr(layout_module, "MAX_GROUP_ROWS", 4096)
    run = planted_run(acquisition="dia", cycles=30, window_width=50.0)
    centres = np.arange(375.0, 1250.0, 50.0)
    targets = pa.Table.from_pylist(
        [dict(target_id=i, modified_sequence="LVNELTEFAK", precursor_charge=2, precursor_mz=float(c))
         for i, c in enumerate(centres)], schema=XIC_TARGET_TABLE)  # fmt: skip
    plan = compile_queries(targets, levels=[2])
    decoded = {}
    for name, unbounded in (("bounded", False), ("unbounded", True)):
        scans = [dict(s) for s in run.scans]
        if unbounded:
            ms2 = [i for i, s in enumerate(scans) if s["level"] == 2]
            scans[ms2[len(ms2) // 2]]["isolation_lower"] = None
        bundle = _bundle(tmp_path, name, scans)
        layout = find_precursor_major(bundle)
        ledger = pm._ledger(bundle, layout, None)
        pm._STATS["fragment_rows_decoded"] = 0
        for q in range(plan.n_queries):
            pm._read_fragment_level(layout, plan.take(np.array([q])), ledger, 2, None)
        decoded[name] = pm._STATS["fragment_rows_decoded"]
        if unbounded:
            # The rows of row groups holding the scan without bounds, which
            # every unit has to read.
            free = 0
            for path in layout.files(2):
                f = pq.ParquetFile(path)
                for g, (a, _) in enumerate(pm._row_group_ranges(f, "lower")):
                    free += f.metadata.row_group(g).num_rows if a == -np.inf else 0
    assert decoded["unbounded"] <= decoded["bounded"] + plan.n_queries * free


def test_a_segment_cut_into_units_is_read_once(tmp_root):
    """Pieces of one segment read the same points; each was read once per
    unit (review of 2026-11-09, E4), and units of queries without a
    precursor m/z each read every MS2 point of the run (E5). They are now
    one task, read once, with the same rows."""
    from constellation.massspec.sweep import precursor_major as pm

    run, bundle = _run("dia", tmp_root)
    first = run.targets.to_pylist()[0]
    many = [dict(first, target_id=30000 + k) for k in range(200)]
    many += [dict(r, target_id=40000 + i, precursor_mz=None) for i, r in enumerate(run.targets.to_pylist())]
    targets = pa.Table.from_pylist(many, schema=XIC_TARGET_TABLE)
    plan = compile_queries(targets, levels=[2], rt_window=None, derive_precursor_mz=False)
    settings = SweepSettings(tolerances=_TOL, unit_cells=2000, batch_cells=500)
    ledger = pm._ledger(bundle, pm.find_precursor_major(bundle), None)
    units, groups = pm._units_and_groups(plan, ledger, settings.unit_cells)
    assert len(units) > 4 and len(groups) == 2  # one segment, and the queries without a precursor m/z
    pm._STATS["fragment_reads"] = 0
    _check(bundle, plan, settings, [XicTraceReducer(drop_unmatched=False), QuerySummaryReducer()])
    assert pm._STATS["fragment_reads"] == len(groups)
    # More workers than tasks: groups are halved so that every worker has work.
    tasks = pm._tasks(groups, workers=4)
    assert len(tasks) >= min(8, len(units)) and [u for t in tasks for u in t] == list(range(len(units)))


def test_ms1_is_read_unit_by_unit_within_a_group(tmp_root):
    """An MS1-only plan is one segment, so all its units are one group, and
    reading the group's MS1 at once read every unit's m/z slices for each
    (Codex, 2026-11-10: 40 units, 800 points held where each needed 20).
    MS1 is read per unit; the rows are scan-major's."""
    from constellation.massspec.sweep import precursor_major as pm

    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1], rt_window=None)
    settings = SweepSettings(tolerances=_TOL, unit_cells=80, batch_cells=40)
    layout = pm.find_precursor_major(bundle)
    ledger = pm._ledger(bundle, layout, None)
    units, groups = pm._units_and_groups(plan, ledger, settings.unit_cells)
    assert len(groups) == 1 and len(units) > 4

    def points(p):
        return sum(len(b.mz) for b in pm._read_blocks(layout, p, ledger, settings, None, None))

    own = max(points(plan.take(u)) for u in units)
    assert own < points(plan) / 2  # the slices differ, unit to unit
    pm._STATS["ms1_points_read"] = 0
    _check(bundle, plan, settings, [XicTraceReducer(drop_unmatched=False), QuerySummaryReducer()])
    assert pm._STATS["ms1_points_read"] == own


def test_narrow_windows_make_blocks_of_many_points(tmp_root, monkeypatch):
    """A block of a unit spanned at least a second, which on one window is
    one scan: narrow RT windows ran 8× slower than scan-major (review of
    2026-11-09, E6). A block now holds at least MIN_BLOCK_POINTS points."""
    from constellation.massspec.sweep import precursor_major as pm

    monkeypatch.setattr(pm, "MIN_BLOCK_POINTS", 2000)  # this run's MS2 holds a few times that
    run, bundle = _run("dia", tmp_root)
    rows = run.targets.to_pylist()
    for r in rows:
        r["rt_start"], r["rt_end"] = r["rt_center"] - 0.4, r["rt_center"] + 0.4
    targets = pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)
    plan = compile_queries(targets, levels=[2])
    layout = pm.find_precursor_major(bundle)
    ledger = pm._ledger(bundle, layout, None)
    unit = plan.take(np.arange(plan.n_queries))
    every = pm._read_fragment_level(layout, unit, ledger, 2, None)
    span = pm._block_span(unit)
    assert span is not None and span < 1.0
    blocks = list(pm._blocks(every, ledger, None, 1 << 20, span))
    points = [b.mz.size for b in blocks]
    assert len(blocks) > 1 and min(points[:-1]) >= pm.MIN_BLOCK_POINTS
    _check(bundle, plan, SweepSettings(tolerances=_TOL), [XicTraceReducer(drop_unmatched=False)])


def test_a_large_m_z_correction_is_read_around(tmp_root, tmp_path):
    """At MS1 a unit reads its channels' m/z plus the tolerance plus the
    largest correction: a 0.3 Th shift must lose nothing."""
    run = planted_run(acquisition="dia", cycles=30, window_width=100.0, seed=5)
    bundle = _bundle(tmp_path, "shifted", run.scans)
    axis = MzAxis(key="synthetic", table=pa.table({"level": pa.array([1, 1, 2, 2], pa.int8()),
                                                   "mz": [300.0, 1600.0, 150.0, 1600.0],
                                                   "shift_th": [0.3, -0.25, 0.2, 0.1]}),
                  parameters={})  # fmt: skip
    save_mz_axis(bundle, axis)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    for tol in (_TOL, {1: (0.05, "Da"), 2: (0.35, "Da")}):
        settings = SweepSettings(tolerances=tol, mz_axis="synthetic", unit_cells=300)
        got = _check(bundle, plan, settings, [XicTraceReducer()])
        assert got.mz_axis == "synthetic"


def test_a_curve_undetermined_at_ms1_does_not_stop_a_sweep_of_ms2(tmp_path):
    """A curve that cannot be applied at MS1 is refused for a plan matching
    MS1, but scan-major corrected every level it read, the plan's or not,
    so a sweep of MS2 alone was refused too (Codex, 2026-11-10); the
    precursor-major read gave it. Both give it, and agree."""
    run = planted_run(acquisition="dia", cycles=30, window_width=100.0, seed=9)
    bundle = _bundle(tmp_path, "ambiguous", run.scans)
    axis = MzAxis(key="ms1-undetermined", table=pa.table({"level": pa.array([1, 1, 2, 2], pa.int8()),
                                                          "mz": [100.0, 2000.0, 100.0, 2000.0],
                                                          "shift_th": [0.2, 0.2, 0.1, 0.1]}),
                  parameters={"ambiguous_levels": [1]})  # fmt: skip
    save_mz_axis(bundle, axis)
    settings = SweepSettings(tolerances=_TOL, mz_axis="ms1-undetermined", unit_cells=300)
    got = _check(bundle, compile_queries(_targets(run), levels=[2], rt_window=20.0), settings, [XicTraceReducer()])
    assert got.mz_axis == "ms1-undetermined" and got.table("xic_trace").num_rows > 0
    with pytest.raises(ValueError, match="branch is undetermined"):
        sweep(bundle, compile_queries(_targets(run), levels=[1, 2], rt_window=20.0), settings,
              reducers=[XicTraceReducer()])  # fmt: skip


def test_every_eligible_scan_is_in_its_units_read_set(tmp_root):
    """Correctness does not depend on where units are cut: whatever the
    budget, a unit reads every scan any of its queries is eligible in."""
    from constellation.massspec.sweep import precursor_major as pm
    from constellation.massspec.sweep.reduce import QueryBatch

    run, bundle = _run("staggered", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    layout = pm.find_precursor_major(bundle)
    ledger = pm._ledger(bundle, layout, None)
    settings = SweepSettings(tolerances=_TOL)
    for budget in (1, 40, 300, 1 << 30):
        units = precursor_units(plan, ledger, budget)
        assert np.array_equal(np.sort(np.concatenate(units)), np.arange(plan.n_queries))
        for unit in units:
            sub = plan.take(unit)
            rows = pm._read_fragment_level(layout, sub, ledger, 2, None)
            read = set() if rows is None else set(rows[0].tolist())
            batch = QueryBatch(plan=sub, queries=np.arange(sub.n_queries), matches=None, scans=ledger,
                               scan_order=ledger.order(), settings=settings)  # fmt: skip
            _, eligible_rows = batch.eligible(2)
            assert set(ledger.scan[eligible_rows].tolist()) <= read


def test_units_cut_at_window_bounds_when_they_can(tmp_root):
    from constellation.massspec.sweep import precursor_major as pm

    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2])
    ledger = pm._ledger(bundle, pm.find_precursor_major(bundle), None)
    index = ledger.window_index(2)
    for unit in precursor_units(plan, ledger, 2000)[:-1]:
        pmz = plan.precursor_mz[unit]
        inside = [(lo <= pmz) & (pmz <= hi) for lo, hi in zip(index.lower, index.upper, strict=True)]
        # Each unit's precursors lie in one window, or straddle few.
        assert sum(bool(x.any()) for x in inside) <= 2


# ── refusals ─────────────────────────────────────────────────────────


def test_what_a_precursor_major_sweep_refuses(tmp_root, tmp_path):
    from _massspec_fixtures import dia_scans

    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2])
    with pytest.raises(ValueError, match="claims"):
        sweep(bundle, plan, SweepSettings(tolerances=_TOL, claims=True), layout="precursor_major")
    with pytest.raises(ValueError, match="min_isotopes"):
        sweep(bundle, plan, SweepSettings(tolerances=_TOL, gate=EvidenceGate(min_isotopes=2)), layout="precursor_major")
    from constellation.massspec.bundle import open_peaks

    with pytest.raises(ValueError, match="table of points"):
        sweep(open_peaks(bundle).to_table(), plan, SweepSettings(tolerances=_TOL), layout="precursor_major")
    bare = write_bundle_manifest(write_bundle(tmp_path, "bare", dia_scans(cycles=3)))
    with pytest.raises(ValueError, match="precursor-major layout"):
        sweep(bare, plan, SweepSettings(tolerances=_TOL), layout="precursor_major")
    with pytest.raises(ValueError, match="layout must be"):
        sweep(bundle, plan, SweepSettings(tolerances=_TOL), layout="window")


# ── the gate first ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(_RUNS))
def test_the_gate_decided_first_changes_no_output_and_matches_only_what_passed(tmp_root, name):
    """With every reducer gated, a batch counts its evidence, decides, and
    matches only the queries that passed: the two-pass gate in one pass."""
    from constellation.massspec.sweep import precursor_major as pm

    run, bundle = _run(name, tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    gate = EvidenceGate(min_fragments=3, min_consecutive=2)
    reducers = [XicTraceReducer(drop_unmatched=False, gated=True)]
    on = SweepSettings(tolerances=_TOL, gate=gate, unit_cells=300, batch_cells=120)
    pm._STATS.clear()
    first = sweep(bundle, plan, on, reducers=reducers, layout="precursor_major")
    matched = pm._STATS.get("gate_first_matched", 0)
    later = sweep(bundle, plan, replace(on, gate_first=False), reducers=reducers, layout="precursor_major")
    oracle.assert_traces_identical(first.table("xic_trace"), later.table("xic_trace"))
    _same_rows(first.table("xic_trace"),
               sweep(bundle, plan, on, reducers=reducers).table("xic_trace"), "target_id")  # fmt: skip
    summary = sweep(bundle, plan, on, reducers=[QuerySummaryReducer()], layout="precursor_major").table("query_summary")
    passed = int(np.sum(summary.column("passed").to_numpy(zero_copy_only=False)))
    assert 0 < passed < plan.n_queries
    assert matched == passed  # only what passed was matched


def test_a_plan_of_few_cells_is_still_spread_over_units_by_window(tmp_root):
    """RT-windowed targets hold few cells: they must not become one task.
    A unit stops at a window bound once its windows hold enough points,
    and its blocks span about half a target's RT window."""
    from constellation.massspec.sweep import precursor_major as pm

    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    ledger = pm._ledger(bundle, pm.find_precursor_major(bundle), None)
    one = pm.precursor_units(plan, ledger, 1 << 40, unit_points=1 << 40)
    by_window = pm.precursor_units(plan, ledger, 1 << 40, unit_points=1)
    assert len(by_window) > len(one) >= 1
    index = ledger.window_index(2)
    for unit in by_window:
        pmz = plan.precursor_mz[unit]
        windows = {w for w, (lo, hi) in enumerate(zip(index.lower, index.upper, strict=True))
                   if ((lo <= pmz) & (pmz <= hi)).any()}  # fmt: skip
        assert len(windows) <= 2
    assert pm._block_span(plan) == pytest.approx(20.0)  # half of ±20 s
    assert pm._block_span(compile_queries(_targets(run), levels=[2])) is None


# ── review by Codex (2026-11-09) ─────────────────────────────────────


def _axis(key: str, shift: float) -> MzAxis:
    return MzAxis(key=key, table=pa.table({"level": pa.array([1, 1, 2, 2], pa.int8()),
                                           "mz": [100.0, 2000.0, 100.0, 2000.0],
                                           "shift_th": [shift] * 4}), parameters={})  # fmt: skip


def test_a_later_sweep_in_the_same_process_uses_the_latest_curve(tmp_path):
    """Workers keep what they read for one sweep only: a curve saved between
    two sweeps of `latest` is the one the second sweep uses and names."""
    import os

    run = planted_run(acquisition="dia", cycles=30, window_width=100.0, seed=7)
    bundle = _bundle(tmp_path, "axes", run.scans)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    settings = SweepSettings(tolerances={1: (0.02, "Da"), 2: (0.02, "Da")}, mz_axis="latest")
    path = save_mz_axis(bundle, _axis("none", 0.0))
    first = sweep(bundle, plan, settings, reducers=[XicTraceReducer()], layout="precursor_major")
    assert first.mz_axis == "none" and first.table("xic_trace").num_rows > 0
    later = save_mz_axis(bundle, _axis("shifted", 0.3))
    os.utime(later, ns=(path.stat().st_mtime_ns + 10**9,) * 2)
    got = sweep(bundle, plan, settings, reducers=[XicTraceReducer()], layout="precursor_major")
    want = sweep(bundle, plan, settings, reducers=[XicTraceReducer()])
    assert got.mz_axis == want.mz_axis == "shifted"
    _same_rows(got.table("xic_trace"), want.table("xic_trace"), "target_id")
    assert got.table("xic_trace").num_rows != first.table("xic_trace").num_rows


def test_a_scan_missing_only_its_upper_bound_is_unbounded_in_both_layouts(tmp_path):
    """The sweep takes a scan missing either bound as unbounded; the layout
    must too, or that scan's points are never read for any query."""
    from constellation.massspec.precursor_major import find_precursor_major

    run = planted_run(acquisition="dia", cycles=30, window_width=100.0, seed=8)
    scans = [dict(s) for s in run.scans]
    ms2 = [s for s in scans if s["level"] == 2]
    for s in ms2[10:14]:
        s["isolation_upper"] = None  # the lower bound stays
    bundle = _bundle(tmp_path, "half", scans)
    lower = pa.concat_tables([pq.read_table(f) for f in find_precursor_major(bundle).files(2)])
    unbounded = set(lower.filter(pa.compute.is_inf(lower.column("lower"))).column("scan").to_pylist())
    assert {s["scan"] for s in ms2[10:14]} <= unbounded
    plan = compile_queries(_targets(run), levels=[2], rt_window=None)
    got = _check(bundle, plan, SweepSettings(tolerances=_TOL), [XicTraceReducer(), QuerySummaryReducer()])
    assert got.table("xic_trace").filter(pa.compute.is_in(got.table("xic_trace").column("scan"),
                                                          value_set=pa.array([s["scan"] for s in ms2[10:14]]))).num_rows > 0  # fmt: skip


@pytest.mark.parametrize("layout", ["scan_major", "precursor_major"])
def test_a_taken_plan_names_its_queries_the_same_on_disk_and_in_memory(tmp_root, tmp_path, layout):
    run, bundle = _run("dia", tmp_root)
    whole = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    taken = whole.take(np.array([3, 1]))
    settings = SweepSettings(tolerances=_TOL)
    names = []
    for kw in ({}, {"work_dir": tmp_path / "w", "workers": 2}):
        summary = sweep(bundle, taken, settings, reducers=[QuerySummaryReducer()], layout=layout, **kw)
        names.append(sorted(summary.table("query_summary").column("query").to_pylist()))
    assert names == [[1, 3], [1, 3]]
    taken.write(tmp_path / "plan")
    from constellation.massspec.sweep.queries import QueryPlan

    np.testing.assert_array_equal(QueryPlan.read(tmp_path / "plan").query_ids, [3, 1])
