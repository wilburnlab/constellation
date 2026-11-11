"""The scan context a scorer reads, review 2026-11-10 (second pass): parent
scans (A10), unsupported runs (A13), injection times (S4), floors (S5),
several ranges and imported ranges (A7), the layout writer (A8), the
summary (A9), the browser's scan table (A12), .dia versions (A14)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from _massspec_fixtures import planted_run, write_bundle, write_bundle_manifest
from constellation.massspec.acquisition_structure import impute_injection_times, parent_scans, unsupported
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep import compile_queries
from constellation.massspec.sweep.engine import SweepSettings, sweep
from constellation.massspec.sweep.reduce import FLOOR_MAP, FLOOR_RECORDED

_TOL = {1: (20.0, "ppm"), 2: (0.35, "Da")}


def _keep(bundle, plan, **kw):
    seen = []

    class Probe:
        name, schema, gated, zero_fills = "probe", pa.schema([("n", pa.int64())]), False, True

        def reduce(self, batch):
            seen.append(batch)
            return pa.table({"n": [0]}, schema=self.schema)

    sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[Probe()], **kw)
    return seen


def _targets(run) -> pa.Table:
    return pa.Table.from_pylist(run.targets.to_pylist(), schema=XIC_TARGET_TABLE)


# ── A10: parents ─────────────────────────────────────────────────────


def test_a_recorded_master_scan_is_taken_only_when_it_can_be_the_parent():
    scan = np.arange(1, 9)
    level = np.array([1, 2, 2, 1, 2, 2, 1, 2])
    derived = parent_scans(scan, level)
    assert derived.tolist() == [-1, 1, 1, -1, 4, 4, -1, 7]
    # 9 is not in the run, 5 is the scan itself, 6 is later, 2 is not one level down.
    recorded = np.array([np.nan, 1, 9, np.nan, 5, 6, np.nan, 2.0])
    assert parent_scans(scan, level, recorded).tolist() == derived.tolist()
    good = np.array([np.nan, 1, 1, np.nan, 1, 4, np.nan, 4.0])  # earlier MS1 scans: taken as given
    assert parent_scans(scan, level, good).tolist() == [-1, 1, 1, -1, 1, 4, -1, 4]


def test_a_voltage_recorded_on_ms2_alone_still_finds_ms1_parents():
    """With MS1 at NaN and MS2 at one voltage, every MS2 scan lost its
    parent: NaN was a voltage group of its own."""
    scan = np.arange(1, 7)
    level = np.array([1, 2, 2, 1, 2, 2])
    cv = np.array([np.nan, -45, -45, np.nan, -45, -45])
    assert parent_scans(scan, level, None, cv).tolist() == [-1, 1, 1, -1, 4, 4]
    two = np.array([-45, -45, -60, -60, -45, -60.0])
    level2 = np.array([1, 2, 1, 2, 2, 2])
    assert parent_scans(scan, level2, None, two).tolist() == [-1, 1, -1, 3, 1, 3]


# ── A13: one test of what cannot be read ────────────────────────────


def test_unsupported_names_several_voltages_and_multiplexing():
    t = pa.table({"faims_cv": pa.array([-45.0, -60.0, None]), "filter_string": ["FTMS + p NSI Full ms", "x msx y", None]})
    reasons = unsupported(t)
    assert len(reasons) == 2 and "multi-CV FAIMS" in reasons[0] and "multiplexed" in reasons[1]
    assert unsupported(pa.table({"faims_cv": pa.array([-45.0, None])})) == []


def test_the_browser_warns_and_gives_no_parents_for_a_multi_cv_run(tmp_path):
    from constellation.massspec.scans import scan_table

    run = planted_run(acquisition="dia", cycles=4, window_width=100.0)
    scans = [dict(s, faims_cv=-45.0 if i % 2 else -60.0) for i, s in enumerate(run.scans)]
    bundle = write_bundle_manifest(write_bundle(tmp_path, "cv", scans))
    with pytest.warns(UserWarning, match="multi-CV FAIMS"):
        t = scan_table(bundle)
    assert t.column("parent_scan").null_count == t.num_rows


def test_the_dia_export_refuses_a_multiplexed_run(tmp_path):
    from constellation.massspec.io.encyclopedia import write_dia

    run = planted_run(acquisition="dia", cycles=4, window_width=100.0)
    scans = [dict(s) for s in run.scans]
    scans[1]["filter_string"] = "FTMS + p NSI msx ms2 500.0000@hcd26.00 600.0000@hcd26.00 [200.0000-1500.0000]"
    bundle = write_bundle_manifest(write_bundle(tmp_path, "msx", scans))
    with pytest.raises(ValueError, match="multiplexed"):
        write_dia([bundle], tmp_path / "msx.dia")


# ── A12: the browser's scan table ───────────────────────────────────


def test_the_scan_table_of_ms2_alone_gives_their_parents(tmp_path):
    from constellation.massspec.scans import scan_table

    run = planted_run(acquisition="dia", cycles=4, window_width=100.0)
    bundle = write_bundle_manifest(write_bundle(tmp_path, "dia", run.scans))
    every = scan_table(bundle)
    ms2 = scan_table(bundle, levels=[2])
    assert ms2.column("parent_scan").null_count == 0
    want = every.filter(pa.compute.equal(every.column("level"), 2)).column("parent_scan")
    assert ms2.column("parent_scan").equals(want)


# ── S4: injection times ─────────────────────────────────────────────


def test_missing_injection_times_are_estimated_from_the_tic_and_flagged():
    level = np.array([1, 2, 2, 1, 2, 2, 1, 2])
    iit = np.array([10, np.nan, 20, 10, 30, np.nan, np.nan, 25.0])
    tic = np.array([1e6, 2e6, 1e6, 1e6, 1e6, 5e5, np.nan, 1e6])
    got, flag = impute_injection_times(level, iit, tic)
    # MS2: N = median(20, 30, 25) × 1000 = 25,000 ions; capped at 30 ms.
    assert got[1] == pytest.approx(12.5) and got[5] == pytest.approx(30.0)
    assert np.isnan(got[6]) and not flag[6]  # no TIC: nothing to estimate from
    assert flag.tolist() == [False, True, False, False, False, True, False, False]
    np.testing.assert_array_equal(got[~flag & np.isfinite(iit)], iit[~flag & np.isfinite(iit)])


def test_the_dense_view_carries_imputed_injection_times_and_the_bundle_keeps_its_own(tmp_path):
    run = planted_run(acquisition="dia", cycles=20, window_width=100.0)
    scans = [dict(s, tic=1e6 + i) for i, s in enumerate(run.scans)]
    hidden = {s["scan"] for s in scans[5:40:3]}
    for s in scans:
        if s["scan"] in hidden:
            s["iit"] = None
    bundle = write_bundle_manifest(write_bundle(tmp_path, "iit", scans))
    plan = compile_queries(_targets(run), levels=[1, 2])
    batch = _keep(bundle, plan)[0]
    for level in (1, 2):
        d = batch.dense(level)
        valid = d.slot_valid
        assert not torch.isnan(d.iit[valid]).any()
        flagged = d.scan[d.iit_imputed].tolist()
        assert flagged and set(flagged) <= hidden
    stored = pq.read_table(bundle / "scan_metadata.parquet", columns=["scan", "iit"]).to_pylist()
    assert all(r["iit"] is None for r in stored if r["scan"] in hidden)  # never written back


# ── S5: floors ──────────────────────────────────────────────────────


def test_a_floor_is_recorded_or_found_and_precursor_major_ms1_has_none(tmp_path):
    from constellation.massspec.precursor_major import build_precursor_major

    run = planted_run(acquisition="dia", cycles=20, window_width=100.0)
    bundle = write_bundle_manifest(write_bundle(tmp_path, "floor", run.scans))
    build_precursor_major(bundle)
    lowest = {s["scan"]: float(np.min(s["intensity"])) for s in run.scans if s["intensity"].size}
    plan = compile_queries(_targets(run), levels=[1, 2])
    # No intensity_floor column (a bundle from before v5): the map finds it.
    for layout in ("scan_major", "precursor_major"):
        batch = _keep(bundle, plan, layout=layout)[0]
        for level in (1, 2):
            d = batch.dense(level)
            qi, si = torch.nonzero(d.slot_valid, as_tuple=True)
            scans_, floors = d.scan[qi, si].tolist(), d.floor[qi, si].tolist()
            if layout == "precursor_major" and level == 1:
                assert all(np.isnan(f) for f in floors)  # read in m/z slices
                continue
            assert all(f == pytest.approx(lowest[s]) for s, f in zip(scans_, floors))
            ledger = batch.scans
            rows = ledger.rows_of(np.array(scans_))
            assert (ledger.floor_source[rows] == FLOOR_MAP).all()
    # Recorded: the column wins, and is said to be recorded.
    meta = pq.read_table(bundle / "scan_metadata.parquet")
    floors = pa.array([lowest.get(s, None) for s in meta.column("scan").to_pylist()], pa.float64())
    meta = meta.set_column(meta.schema.get_field_index("intensity_floor"), "intensity_floor", floors)
    pq.write_table(meta, bundle / "scan_metadata.parquet")
    batch = _keep(bundle, plan)[0]
    d = batch.dense(1)
    rows = batch.scans.rows_of(d.scan[d.slot_valid].numpy())
    assert (batch.scans.floor_source[rows] == FLOOR_RECORDED).all()


# ── A7: several ranges, imported ranges ─────────────────────────────


def test_a_channel_between_a_scans_ranges_is_not_acquired(tmp_path):
    run = planted_run(acquisition="dia", cycles=12, window_width=100.0)
    scans = [dict(s) for s in run.scans]
    for s in scans:
        if s["level"] == 2:
            s["scan_window_lower"], s["scan_window_upper"] = 200.0, 1500.0
            s["scan_windows"] = [{"lower": 200.0, "upper": 600.0}, {"lower": 700.0, "upper": 1500.0}]
    bundle = write_bundle_manifest(write_bundle(tmp_path, "gaps", scans))
    plan = compile_queries(_targets(run), levels=[2])
    batch = _keep(bundle, plan)[0]
    d = batch.dense(2)
    mz = d.mz.unsqueeze(1).expand_as(d.acquired)
    in_gap = (mz > 600.0) & (mz < 700.0) & d.slot_valid.unsqueeze(2) & d.channel_valid.unsqueeze(1)
    assert in_gap.any()
    assert not (d.acquired & in_gap & ~d.matched).any()
    cells = batch.cells(2)
    assert not ((cells.mz_theoretical > 600.0) & (cells.mz_theoretical < 700.0) & ~cells.matched).any()
    assert d.range_known[d.slot_valid].all()


_DIA_METHOD = """Experiment 1
\tExperiment Name = MS
\t\tScan MasterScan
\t\t\tScan Range (m/z) = 350-1250
Experiment 2
\tExperiment Name = DIA
\t\tScan DIAScan
\t\t\tPrecursor Mass Range (m/z) = 900-1000
\t\t\tScan Range (m/z) = 200-1500
"""

_DDA_METHOD = """Experiment 1
\tExperiment Name = MS
\t\tScan MasterScan
\t\t\tScan Range (m/z) = 350-1250
\t\tFilter PrecursorSelectionRange
\t\t\tMass Range (m/z) = 350-1250
Scan Event 1
\t\tScan ddMSnScan
\t\t\tMSn Level = 2
\t\t\tScan Range Mode = Manual
\t\t\tScan Range (m/z) = 200-1500
"""


def test_method_text_gives_each_levels_range():
    from constellation.massspec.readers.thermo._method import scan_ranges_from_method

    assert scan_ranges_from_method([_DIA_METHOD]) == {1: (350.0, 1250.0), 2: (200.0, 1500.0)}
    assert scan_ranges_from_method(["", _DDA_METHOD]) == {1: (350.0, 1250.0), 2: (200.0, 1500.0)}
    two = _DIA_METHOD + "Experiment 3\n\tExperiment Name = DIA\n\t\tScan DIAScan\n\t\t\tScan Range (m/z) = 150-2000\n"
    assert scan_ranges_from_method([two]) == {1: (350.0, 1250.0)}  # MS2 of two ranges is not one


def test_an_imported_dia_takes_its_ms2_range_from_the_method(tmp_path):
    """An imported MS2 scan recorded no range, so every channel counted as
    acquired (A7: 33 % more MS2 cells on the real GPF run, all zeros)."""
    from constellation.massspec.io.encyclopedia import import_dia, write_dia

    run = planted_run(acquisition="dia", cycles=6, window_width=100.0)
    bundle = write_bundle_manifest(write_bundle(tmp_path, "src", run.scans))
    dia = tmp_path / "out.dia"
    write_dia([bundle], dia)
    con = sqlite3.connect(dia)
    con.execute("INSERT OR REPLACE INTO metadata (Key, Value) VALUES (?, ?)",
                ("thermo.instrument_method.1.raw_text", _DIA_METHOD))  # fmt: skip
    con.commit()
    con.close()
    (imported,) = import_dia(dia, tmp_path / "imported")
    meta = pq.read_table(imported / "scan_metadata.parquet").to_pylist()
    ms2 = [r for r in meta if r["level"] == 2]
    assert ms2 and all((r["scan_window_lower"], r["scan_window_upper"]) == (200.0, 1500.0) for r in ms2)
    assert all(r["intensity_floor"] is not None for r in meta if r["peak_count"])


def test_a_dia_of_an_unread_format_is_refused_by_name(tmp_path):
    from constellation.massspec.io.encyclopedia._dia import DiaFile

    path = tmp_path / "old.dia"
    con = sqlite3.connect(path)
    for t in ("metadata (Key string, Value string)", "precursor (x int)", "ranges (x int)", "spectra (x int)"):
        con.execute(f"CREATE TABLE {t}")
    con.execute("INSERT INTO metadata VALUES ('version', '0.4.0')")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="0.4.0"):
        DiaFile(path)


# ── A8: the layout writer ───────────────────────────────────────────


def test_a_scan_spanning_three_batches_is_written_whole():
    import tempfile

    from constellation.massspec.precursor_major import PrecursorMajorWriter

    root = Path(tempfile.mkdtemp())
    writer = PrecursorMajorWriter(root, 0, max_points=10)

    def batch(scan, n, start):
        return pa.table({"scan": pa.array([scan] * n, pa.int32()), "level": pa.array([2] * n, pa.int8()),
                         "mz": pa.array(np.arange(start, start + n, dtype=np.float64)),
                         "intensity": pa.array(np.ones(n, np.float32)),
                         "isolation_lower": pa.array([500.0] * n), "isolation_upper": pa.array([502.0] * n)})  # fmt: skip

    writer.write(batch(1, 12, 0))  # past the limit
    writer.write(batch(1, 12, 100))  # the same scan goes on: held, not split
    writer.write(batch(1, 5, 200))
    writer.write(batch(2, 3, 300))
    writer.close()
    files = sorted(root.rglob("*.parquet"))
    tables = [pq.read_table(f) for f in files]
    holding = [t for t in tables if "scan" in t.column_names and 1 in t.column("scan").to_pylist()]
    assert len(holding) == 1 and holding[0].filter(pa.compute.equal(holding[0].column("scan"), 1)).num_rows == 29


# ── A9: the summary ─────────────────────────────────────────────────


def test_summarize_takes_a_profile_curve_undetermined_at_ms1(tmp_path):
    from constellation.massspec.bundle_summary import _binned_curve
    from constellation.massspec.mz_axis import MZ_AXIS_TABLE, MzAxis

    rows = [dict(level=lv, mz=float(m), shift_th=0.01, se_th=1e-4, coherence=0.9, half_window_th=10.0)
            for lv in (1, 2) for m in np.arange(300.0, 1300.0, 50.0)]  # fmt: skip
    table = pa.Table.from_pylist(rows, schema=MZ_AXIS_TABLE)
    axis = MzAxis(key="peaks", table=table, parameters={"method": "test"})
    profile = MzAxis(key="profile", table=table, parameters={"method": "test", "ambiguous_levels": [1]})
    assert not profile.is_determined(1) and profile.is_determined(2)
    bins = _binned_curve(axis, 1, profile)
    assert bins and all(b["bias_th"] is None for b in bins.values())
    assert any(b["bias_th"] is not None for b in _binned_curve(axis, 2, profile).values())
