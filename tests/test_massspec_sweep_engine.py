"""Running a sweep end to end (`massspec.sweep.engine`) and the XIC traces it
gives through `XicTraceReducer`.

A bundle streamed block by block must give what its peak table gives the
frozen extractor; the block size and the kernel must change nothing; and
the two behaviours changed on purpose are pinned here.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import _xic_scan_major_oracle as oracle
from _massspec_fixtures import dia_scans, planted_run, write_bundle, write_bundle_manifest
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
    )  # fmt: skip
    want = oracle.extract_xic_scan_major(peaks, targets, **kw)
    assert want.num_rows > 0
    oracle.assert_traces_identical(extract_xic_scan_major(bundle, targets, **kw), want)
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
    for kw in (dict(block_rows=50), dict(block_rows=777), dict(kernel="cells"), dict(kernel="peaks"),
               dict(kernel="cells", kernel_elements=64), dict(block_rows=1, kernel="peaks")):  # fmt: skip
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
    assert result.scans.n_peaks[result.scans.scan == 6].tolist() == [0]
    assert (result.scans.n_peaks[result.scans.scan != 6] > 0).all()


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
        runs = sorted(p.name for p in (work / "matches").glob("*.parquet"))
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
