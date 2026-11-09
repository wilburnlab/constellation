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
