"""What the sweep's ledger knows of each scan beyond its points: the m/z
range it acquired (a channel outside it was never measured, so it is not
zero-filled), the scan one level down it was taken from, and a refusal of
runs of several FAIMS compensation voltages (review of 2026-11-09, E2 and
A1)."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from _massspec_fixtures import dda_scans, dia_scans, planted_run, write_bundle, write_bundle_manifest
from constellation.massspec.acquisition_structure import parent_scans
from constellation.massspec.precursor_major import build_precursor_major
from constellation.massspec.quant.chromatogram import XicTraceReducer
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep import compile_queries
from constellation.massspec.sweep.engine import SweepSettings, sweep
from constellation.massspec.sweep.queries import ChannelSettings
from constellation.massspec.sweep.schemas import SWEEP_CELL_TABLE

_TOL = {1: (20.0, "ppm"), 2: (0.35, "Da")}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("scan_context"))


def _bundle(root: Path, name: str, scans) -> Path:
    bundle = write_bundle_manifest(write_bundle(root, name, scans, row_group_size=97))
    build_precursor_major(bundle)
    return bundle


@functools.lru_cache(maxsize=None)
def _ranged(tmp_root: str):
    """A planted DIA run whose MS2 scans record their 200–1500 range (MS1:
    350–1250, the fixture's)."""
    run = planted_run(acquisition="dia", cycles=30, window_width=100.0)
    scans = [dict(s) for s in run.scans]
    for s in scans:
        if s["level"] == 2:
            s["scan_window_lower"], s["scan_window_upper"] = 200.0, 1500.0
    return run, _bundle(Path(tmp_root), "ranged", scans)


class _Cells:
    """Every cell of every query, as SWEEP_CELL_TABLE rows, by level."""

    name = "cells"
    outputs = {"cells_ms1": SWEEP_CELL_TABLE, "cells_ms2": SWEEP_CELL_TABLE}
    gated = False
    zero_fills = True

    def reduce(self, batch):
        return {f"cells_ms{lv}": batch.cells(lv).table() for lv in (1, 2)}


def _windows(bundle: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    t = pq.read_table(bundle / "scan_metadata.parquet", columns=["scan", "scan_window_lower", "scan_window_upper"])
    return (t.column("scan").to_numpy(), t.column("scan_window_lower").to_numpy(zero_copy_only=False),
            t.column("scan_window_upper").to_numpy(zero_copy_only=False))  # fmt: skip


@pytest.mark.parametrize("layout", ["scan_major", "precursor_major"])
def test_a_channel_outside_its_scans_range_is_not_zero_filled(tmp_root, layout):
    run, bundle = _ranged(tmp_root)
    plan = compile_queries(pa.Table.from_pylist(run.targets.to_pylist(), schema=XIC_TARGET_TABLE), levels=[1, 2],
                           rt_window=20.0, channels=ChannelSettings(charge_range=(1, 4)))  # fmt: skip
    got = sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[_Cells()], layout=layout)
    scan, lo, hi = _windows(bundle)
    by = {int(s): (a, b) for s, a, b in zip(scan, lo, hi, strict=True)}
    for level in (1, 2):
        cells = got.table(f"cells_ms{level}")
        mz = cells.column("mz_theoretical").to_numpy()
        bounds = np.array([by[int(s)] for s in cells.column("scan").to_numpy()])
        outside = (mz < bounds[:, 0]) | (mz > bounds[:, 1])
        matched = cells.column("matched").to_numpy(zero_copy_only=False)
        assert not (outside & ~matched).any()  # never a zero where nothing was measured
        channels = plan.channels_for_level(level)
        lows = (channels.mz < (350.0 if level == 1 else 200.0)) | (channels.mz > (1250.0 if level == 1 else 1500.0))
        assert lows.any()  # there were channels outside to leave out


def test_a_matched_cell_outside_its_scans_range_is_kept(tmp_root, tmp_path):
    """A match proves the point was measured: a channel just outside the
    recorded range, matched within its tolerance, keeps its row."""
    run = planted_run(acquisition="dia", cycles=30, window_width=100.0)
    plan = compile_queries(pa.Table.from_pylist(run.targets.to_pylist(), schema=XIC_TARGET_TABLE), levels=[1],
                           rt_window=20.0)  # fmt: skip
    plain = write_bundle_manifest(write_bundle(tmp_path, "plain", run.scans, row_group_size=97))
    traces = sweep(plain, plan, SweepSettings(tolerances=_TOL), reducers=[XicTraceReducer()]).table("xic_trace")
    some = float(np.median(traces.column("mz_theoretical").to_numpy()))
    scans = [dict(s) for s in run.scans]
    for s in scans:
        if s["level"] == 1:
            s["scan_window_lower"], s["scan_window_upper"] = 350.0, some - 1e-6
    narrow = write_bundle_manifest(write_bundle(tmp_path, "narrow", scans, row_group_size=97))
    filled = sweep(narrow, plan, SweepSettings(tolerances=_TOL),
                   reducers=[XicTraceReducer(drop_unmatched=False)]).table("xic_trace")  # fmt: skip
    matched = filled.filter(pc.invert(pc.is_nan(filled.column("mz_observed"))))
    assert matched.num_rows == traces.num_rows
    assert (matched.column("mz_theoretical").to_numpy() > some).any()


def test_the_parent_rule():
    scan = np.arange(1, 11)
    level = np.array([1, 2, 2, 1, 2, 2, 1, 2, 2, 2])
    np.testing.assert_array_equal(parent_scans(scan, level), [-1, 1, 1, -1, 4, 4, -1, 7, 7, 7])
    # The instrument's own record wins where it gives one (0 or NaN: none).
    recorded = np.array([0, 1, 1, 0, 1, np.nan, 0, 0, 7, 4], dtype=np.float64)
    np.testing.assert_array_equal(parent_scans(scan, level, recorded), [-1, 1, 1, -1, 1, 4, -1, 7, 7, 4])
    # Alternating FAIMS voltages: the parent is the last MS1 at the same one.
    cv = np.array([-45, -45, -45, -65, -65, -65, -45, -45, -65, -45], dtype=np.float64)
    np.testing.assert_array_equal(parent_scans(scan, level, None, cv), [-1, 1, 1, -1, 4, 4, -1, 7, 4, 7])
    # No MS1 before, or none at all.
    np.testing.assert_array_equal(parent_scans(scan[1:3], level[1:3]), [-1, -1])


@pytest.mark.parametrize("acquisition", ["dia", "dda"])
def test_the_ledger_links_each_scan_to_its_parent_as_the_browser_does(tmp_path, acquisition):
    from constellation.massspec.scans import scan_table

    scans = dia_scans(cycles=6) if acquisition == "dia" else dda_scans()
    bundle = write_bundle_manifest(write_bundle(tmp_path, acquisition, scans))
    targets = pa.Table.from_pylist([dict(target_id=1, precursor_mz=401.4324, precursor_charge=2)],
                                   schema=XIC_TARGET_TABLE)  # fmt: skip
    ledger = sweep(bundle, compile_queries(targets, levels=[1]), SweepSettings(tolerances=_TOL)).scans
    parent = np.where(ledger.parent >= 0, ledger.scan[np.maximum(ledger.parent, 0)], -1)
    browser = scan_table(bundle).column("parent_scan").fill_null(-1).to_numpy()
    np.testing.assert_array_equal(parent, browser)
    assert (parent[ledger.level == 2] > 0).all()


def test_the_ledger_on_disk_is_the_ledger(tmp_root, tmp_path):
    from constellation.massspec.sweep.engine import _read_ledger, _write

    run, bundle = _ranged(tmp_root)
    plan = compile_queries(pa.Table.from_pylist(run.targets.to_pylist(), schema=XIC_TARGET_TABLE), levels=[1, 2])
    ledger = sweep(bundle, plan, SweepSettings(tolerances=_TOL)).scans
    _write(ledger.table(), tmp_path / "scans.parquet")
    back = _read_ledger(tmp_path / "scans.parquet")
    for name in ("scan", "level", "rt", "lower", "upper", "n_points", "tic", "iit", "scan_lower", "scan_upper",
                 "parent"):  # fmt: skip
        np.testing.assert_array_equal(getattr(back, name), getattr(ledger, name), err_msg=name)
    assert (ledger.parent[ledger.level == 2] >= 0).all() and np.isfinite(ledger.scan_lower).all()


def _faims(scans, voltages):
    out = []
    for i, s in enumerate(scans):
        cycle = sum(1 for t in scans[: i + 1] if t["level"] == 1) - 1
        out.append(dict(s, faims_cv=float(voltages[cycle % len(voltages)])))
    return out


@pytest.mark.parametrize("layout", ["scan_major", "precursor_major"])
def test_a_run_of_several_faims_voltages_is_refused(tmp_path, layout):
    run = planted_run(acquisition="dia", cycles=30, window_width=100.0)
    targets = pa.Table.from_pylist(run.targets.to_pylist(), schema=XIC_TARGET_TABLE)
    plan = compile_queries(targets, levels=[1, 2], rt_window=20.0)
    one = _bundle(tmp_path, "one", _faims(run.scans, [-45.0]))
    assert sweep(one, plan, SweepSettings(tolerances=_TOL), reducers=[XicTraceReducer()],
                 layout=layout).table("xic_trace").num_rows > 0  # fmt: skip
    two = _bundle(tmp_path, "two", _faims(run.scans, [-45.0, -65.0]))
    with pytest.raises(ValueError, match="multi-CV FAIMS"):
        sweep(two, plan, SweepSettings(tolerances=_TOL), reducers=[XicTraceReducer()], layout=layout)
    with pytest.raises(ValueError, match="multi-CV FAIMS"):
        sweep(two, plan, SweepSettings(tolerances=_TOL), reducers=[XicTraceReducer()], layout=layout,
              workers=2, work_dir=tmp_path / "w")  # fmt: skip


@pytest.mark.parametrize("layout", ["scan_major", "precursor_major"])
def test_a_run_of_multiplexed_isolation_is_refused(tmp_path, layout):
    """An MSX scan isolates several precursors and its window holds one
    (review of 2026-11-09, A4)."""
    scans = [dict(s) for s in dia_scans(cycles=4)]
    for s in scans:
        if s["level"] == 2:
            s["filter_string"] = s["filter_string"].replace("Full ms2", "Full msx ms2")
    bundle = _bundle(tmp_path, "msx", scans)
    targets = pa.Table.from_pylist([dict(target_id=1, precursor_mz=401.4324, precursor_charge=2)],
                                   schema=XIC_TARGET_TABLE)  # fmt: skip
    with pytest.raises(ValueError, match="multiplexed"):
        sweep(bundle, compile_queries(targets, levels=[1]), SweepSettings(tolerances=_TOL), layout=layout)


def test_a_table_of_points_of_several_voltages_is_refused():
    from _massspec_fixtures import MS1_FILTER  # noqa: F401  (the fixture's peaks carry faims_cv)
    from constellation.massspec.schemas import MS_PEAK_TABLE

    rows = []
    for s in _faims(dia_scans(cycles=4), [-45.0, -65.0]):
        for mz, inten in zip(s["mz"], s["intensity"], strict=True):
            rows.append({f.name: s.get(f.name) for f in MS_PEAK_TABLE} | {"mz": float(mz), "intensity": float(inten)})
    table = pa.Table.from_pylist(rows, schema=MS_PEAK_TABLE)
    targets = pa.Table.from_pylist([dict(target_id=1, precursor_mz=401.4324, precursor_charge=2)],
                                   schema=XIC_TARGET_TABLE)  # fmt: skip
    with pytest.raises(ValueError, match="multi-CV FAIMS"):
        sweep(table, compile_queries(targets, levels=[1]), SweepSettings(tolerances=_TOL))
