"""Tests for the derived mass-sorted peak index builder."""

from __future__ import annotations

import collections

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.quant.peak_index import build_peak_index, read_manifest

_SCHEMA = {
    "scan": pa.int32(),
    "level": pa.int8(),
    "rt": pa.float64(),
    "mz": pa.float64(),
    "intensity": pa.float64(),
    "isolation_lower": pa.float64(),
    "isolation_upper": pa.float64(),
    "precursor_mz": pa.float64(),
}


def _peaks(rows):
    cols = collections.defaultdict(list)
    for r in rows:
        for k in _SCHEMA:
            cols[k].append(r.get(k))
    return pa.table({k: pa.array(cols[k], _SCHEMA[k]) for k in _SCHEMA})


def _row(scan, level, rt, mz, lo=None, hi=None, prec=None):
    return dict(
        scan=scan,
        level=level,
        rt=rt,
        mz=mz,
        intensity=100.0,
        isolation_lower=lo,
        isolation_upper=hi,
        precursor_mz=prec,
    )


def _write(tmp_path, peaks):
    src = tmp_path / "bundle"
    src.mkdir()
    pq.write_table(peaks, src / "peaks.parquet")
    return src


def test_default_is_lossless_f64_mirror(tmp_path):
    peaks = _peaks([_row(1, 1, 10.0, 500.3), _row(1, 1, 10.0, 200.1)])
    man = build_peak_index(_write(tmp_path, peaks), tmp_path / "idx")
    assert man.precision == {"mz": "f64", "rt": "f64", "intensity": "f64"}
    t = pq.read_table(next((tmp_path / "idx").rglob("part-00000.parquet")))
    assert t.schema.field("mz").type == pa.float64()
    # mz ascending within partition
    assert t.column("mz").to_pylist() == sorted(t.column("mz").to_pylist())


def test_ms1_single_partition_dia_native(tmp_path):
    rows = [_row(1, 1, 10.0, 200.0), _row(1, 1, 10.0, 800.0)]
    rows += [_row(2, 2, 10.5, 150.0, 400.0, 410.0, 405.0)]
    rows += [_row(3, 2, 11.0, 160.0, 410.0, 420.0, 415.0)]
    man = build_peak_index(_write(tmp_path, _peaks(rows)), tmp_path / "idx")
    assert man.window_scheme == {"1": "native", "2": "native"}
    assert man.n_windows == 3  # MS1 + two DIA windows


def test_dda_binned_into_quasi_dia_grid(tmp_path):
    rows = []
    for i in range(20):  # 20 unique narrow windows, precursors 500..519
        p = 500.0 + i
        rows.append(_row(100 + i, 2, float(i), 300.0 + i, p - 0.7, p + 0.7, p))
    man = build_peak_index(
        _write(tmp_path, _peaks(rows)),
        tmp_path / "idx",
        max_isolation_windows=5,
        dda_window_width_mz=10.0,
    )
    assert man.window_scheme == {"2": "binned"}
    # precursors 500..519 → bins 50, 51
    assert man.n_windows == 2
    for p in man.partitions:
        assert p["isolation_lower"] == p["isolation_window_id"] * 10.0


def test_per_axis_downcast(tmp_path):
    peaks = _peaks([_row(1, 1, 10.0, 500.0), _row(1, 1, 10.0, 200.0)])
    man = build_peak_index(
        _write(tmp_path, peaks), tmp_path / "idx", downcast=["rt", "intensity"]
    )
    assert man.precision == {"mz": "f64", "rt": "f32", "intensity": "f32"}
    t = pq.read_table(next((tmp_path / "idx").rglob("part-00000.parquet")))
    assert t.schema.field("mz").type == pa.float64()  # mz stays f64
    assert t.schema.field("rt").type == pa.float32()
    assert t.schema.field("intensity").type == pa.float32()


def test_downcast_mz_also_downcasts_iso_bounds(tmp_path):
    rows = [_row(2, 2, 10.0, 150.0, 400.0, 410.0, 405.0)]
    man = build_peak_index(
        _write(tmp_path, _peaks(rows)), tmp_path / "idx", downcast=["mz"]
    )
    assert man.precision["mz"] == "f32"
    t = pq.read_table(next((tmp_path / "idx").rglob("part-00000.parquet")))
    assert t.schema.field("mz").type == pa.float32()
    assert t.schema.field("isolation_lower").type == pa.float32()


def test_empty_input(tmp_path):
    peaks = _peaks([])
    man = build_peak_index(_write(tmp_path, peaks), tmp_path / "idx")
    assert man.n_windows == 0
    assert read_manifest(tmp_path / "idx").n_windows == 0
