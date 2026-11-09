"""A bundle's points precursor-major (`massspec.precursor_major`): every
point once, each scan's points together and in stored order, every scan
counted, row groups a window prunes to, a key that follows the peak list,
and the same bytes whether `convert` writes it or it is built afterwards."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _massspec_fixtures import dia_scans, planted_run, write_bundle, write_bundle_manifest
from _thermo_fakes import IT_MS1, MS2, FakeEvent, FakeRaw, scan
from constellation.massspec import precursor_major as pm
from constellation.massspec.bundle import peaks_dataset
from constellation.massspec.precursor_major import (
    PRECURSOR_MAJOR_DIR,
    build_precursor_major,
    find_precursor_major,
)

_RUNS = {
    "dia": dict(acquisition="dia", cycles=12, window_width=100.0),
    "staggered": dict(acquisition="dia", staggered=True, cycles=12, window_width=100.0),
    "dda": dict(acquisition="dda", cycles=40),
}


def _bundle(tmp_path, name, scans, row_group_size=37):
    return write_bundle_manifest(write_bundle(tmp_path, name, scans, row_group_size=row_group_size))


def _with_odd_scans(scans):
    """A run's scans with an MS2 scan that has no isolation bounds and one
    that has no points."""
    scans = [dict(s) for s in scans]
    ms2 = [i for i, s in enumerate(scans) if s["level"] == 2]
    scans[ms2[3]]["isolation_lower"] = scans[ms2[3]]["isolation_upper"] = None
    scans[ms2[5]]["mz"] = np.empty(0)
    scans[ms2[5]]["intensity"] = np.empty(0)
    return scans


def _layout_rows(root: Path, level: int) -> pa.Table:
    return pa.concat_tables([pq.read_table(f) for f in sorted((root / pm.level_dir(level)).glob("part-*.parquet"))])


@pytest.mark.parametrize("name", list(_RUNS))
def test_every_point_is_there_once_with_its_scan_together_and_in_stored_order(tmp_path, name):
    scans = _with_odd_scans(planted_run(**_RUNS[name]).scans)
    bundle = _bundle(tmp_path, name, scans)
    build_precursor_major(bundle)
    layout = find_precursor_major(bundle)
    assert layout is not None and layout.levels == (1, 2)
    peaks = peaks_dataset(bundle).to_table(columns=["scan", "level", "mz", "intensity", "isolation_lower"])
    by_scan: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for s, mz, it in zip(*(peaks.column(c).to_numpy() for c in ("scan", "mz", "intensity")), strict=True):
        by_scan.setdefault(int(s), ([], []))
        by_scan[int(s)][0].append(mz)
        by_scan[int(s)][1].append(it)
    level_of = dict(zip(peaks.column("scan").to_pylist(), peaks.column("level").to_pylist(), strict=True))

    for f in layout.files(2):  # each part sorted by window
        part = pq.read_table(f, columns=["lower"]).column("lower").to_numpy()
        assert np.all(part[1:] >= part[:-1])
    ms2 = _layout_rows(layout.root, 2)
    lower = ms2.column("lower").to_numpy()
    seen = set()
    scan_col = ms2.column("scan").to_numpy()
    starts = np.flatnonzero(np.r_[True, scan_col[1:] != scan_col[:-1]])
    for a, b in zip(starts, [*starts[1:], scan_col.size], strict=True):
        s = int(scan_col[a])
        assert s not in seen  # a scan's points are one run
        seen.add(s)
        np.testing.assert_array_equal(ms2.column("mz").to_numpy()[a:b], by_scan[s][0])
        np.testing.assert_array_equal(ms2.column("intensity").to_numpy()[a:b], by_scan[s][1])
    assert seen == {s for s, lv in level_of.items() if lv == 2}
    no_bounds = [s["scan"] for s in scans if s["level"] == 2 and s["isolation_lower"] is None]
    assert set(scan_col[np.isneginf(lower)].tolist()) == set(no_bounds)

    for f in layout.files(1):
        assert np.all(np.diff(pq.read_table(f, columns=["mz"]).column("mz").to_numpy()) >= 0)
    ms1 = _layout_rows(layout.root, 1)
    for s, point, mz in zip(*(ms1.column(c).to_numpy() for c in ("scan", "point", "mz")), strict=True):
        assert by_scan[int(s)][0][int(point)] == mz
    assert ms1.num_rows + ms2.num_rows == peaks.num_rows

    counts = layout.scans()
    want = {s["scan"]: len(s["mz"]) for s in scans}
    assert dict(zip(counts.column("scan").to_pylist(), counts.column("n_points").to_pylist(), strict=True)) == want


def test_a_window_is_read_from_a_few_row_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "MIN_GROUP_ROWS", 8)
    monkeypatch.setattr(pm, "MAX_GROUP_ROWS", 64)
    bundle = _bundle(tmp_path, "dia", planted_run(**_RUNS["dia"]).scans, row_group_size=10_000)
    build_precursor_major(bundle)
    path = find_precursor_major(bundle).files(2)[0]
    f = pq.ParquetFile(path)
    meta = f.metadata
    col = f.schema_arrow.get_field_index("lower")
    bounds = [(meta.row_group(g).column(col).statistics.min, meta.row_group(g).column(col).statistics.max)
              for g in range(meta.num_row_groups)]  # fmt: skip
    lower = pq.read_table(path, columns=["lower"]).column("lower").to_numpy()
    windows = sorted(set(lower.tolist()))
    assert len(windows) > 2
    for w in windows:
        touched = [g for g, (lo, hi) in enumerate(bounds) if lo <= w <= hi]
        rows = sum(meta.row_group(g).num_rows for g in touched)
        # The window's rows, and at most a small group's worth either side.
        assert rows <= int(np.sum(lower == w)) + 2 * 64
        assert len(touched) < meta.num_row_groups


def test_the_layout_follows_its_peak_list_and_survives_a_copy(tmp_path):
    bundle = _bundle(tmp_path, "run", dia_scans(cycles=6))
    build_precursor_major(bundle)
    assert find_precursor_major(bundle) is not None
    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    assert find_precursor_major(copy) is not None
    part = sorted((copy / "peaks").glob("*.parquet"))[0]
    t = pq.read_table(part)
    t = t.set_column(t.schema.get_field_index("intensity"), "intensity", pa.array(t.column("intensity").to_numpy() + 1))
    pq.write_table(t, part, row_group_size=37)
    assert find_precursor_major(copy) is None  # stale: another peak list
    build_precursor_major(copy)
    assert find_precursor_major(copy) is not None


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.parquet"))}


def test_a_parallel_build_writes_the_bytes_a_serial_one_does(tmp_path):
    bundle = _bundle(tmp_path, "run", planted_run(**_RUNS["staggered"]).scans)
    assert len(list((bundle / "peaks").glob("*.parquet"))) > 1
    build_precursor_major(bundle, workers=1)
    serial = _files(bundle / PRECURSOR_MAJOR_DIR)
    build_precursor_major(bundle, workers=2, force=True)
    assert _files(bundle / PRECURSOR_MAJOR_DIR) == serial


# ── written by convert ───────────────────────────────────────────────


def _dia_raw(cycles: int) -> FakeRaw:
    """An ion-trap DIA run: an MS1, then four 25-Th windows (one MS2 per
    cycle with no isolation), a few centroids each."""
    scans, n = {}, 1
    rng = np.random.default_rng(5)
    for c in range(cycles):
        scans[n] = scan(IT_MS1.replace(" p ", " c "), rt_min=n / 600.0,
                        mz=np.sort(rng.uniform(400, 1000, 12)), intensity=rng.uniform(1, 9, 12))  # fmt: skip
        n += 1
        for w, centre in enumerate((450.0, 475.0, 500.0, 525.0)):
            event = FakeEvent(mass=None) if (c % 5 == 2 and w == 1) else FakeEvent(mass=centre, width=25.0)
            k = int(rng.integers(0, 9))
            scans[n] = scan(MS2, rt_min=n / 600.0, mz=np.sort(rng.uniform(200, 1500, k)),
                            intensity=rng.uniform(1, 9, k), event=event)  # fmt: skip
            n += 1
    return FakeRaw(scans)


@pytest.fixture
def raws(monkeypatch, fake_centroider):
    from constellation.massspec.readers.thermo import _convert, _read

    files: dict[str, object] = {}

    def open_raw(path):
        return _read._RawFileHandle(files[Path(path).stem])

    monkeypatch.setattr(_read, "_open_raw_file", open_raw)
    yield files
    _convert._close_open_files()


def _source(tmp_path: Path, stem: str) -> Path:
    p = tmp_path / f"{stem}.raw"
    p.write_bytes(stem.encode())
    return p


def test_convert_writes_the_layout_a_build_afterwards_would(tmp_path, raws):
    from _thermo_fakes import FakeRaw  # noqa: F401
    from constellation.massspec.readers.thermo import _convert
    from test_thermo_batch import PicklingExecutor

    raws["run"] = _dia_raw(900)
    src = _source(tmp_path, "run")
    job = _convert._Job(src, tmp_path / "run", _convert.ConvertOptions(), force=False)
    _convert._run_jobs([job], workers=2, executor_factory=lambda n: PicklingExecutor())
    assert job.error is None
    bundle = tmp_path / "run"
    assert len(list((bundle / "peaks").glob("*.parquet"))) > 1
    assert find_precursor_major(bundle) is not None
    assert job.manifest.outputs["precursor_major"] == PRECURSOR_MAJOR_DIR
    written = _files(bundle / PRECURSOR_MAJOR_DIR)
    scans_written = (bundle / PRECURSOR_MAJOR_DIR / "scans.parquet").read_bytes()
    build_precursor_major(bundle, force=True)
    assert _files(bundle / PRECURSOR_MAJOR_DIR) == written
    assert (bundle / PRECURSOR_MAJOR_DIR / "scans.parquet").read_bytes() == scans_written


def test_convert_can_leave_it_out_and_force_replaces_it(tmp_path, raws):
    from constellation.massspec.readers.thermo import convert

    raws["run"] = _dia_raw(30)
    src = _source(tmp_path, "run")
    convert(src, tmp_path / "plain", precursor_major=False)
    assert not (tmp_path / "plain" / PRECURSOR_MAJOR_DIR).exists()
    convert(src, tmp_path / "run")
    first = find_precursor_major(tmp_path / "run")
    (tmp_path / "run" / PRECURSOR_MAJOR_DIR / "ms2" / "stale.parquet").write_bytes(b"x")
    convert(src, tmp_path / "run", force=True)
    assert not (tmp_path / "run" / PRECURSOR_MAJOR_DIR / "ms2" / "stale.parquet").exists()
    assert find_precursor_major(tmp_path / "run").key == first.key


def test_a_failed_convert_leaves_no_finished_layout(tmp_path, raws, monkeypatch):
    from constellation.massspec.readers.thermo import _convert, convert

    raws["run"] = _dia_raw(30)

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(_convert, "build_precursor_major", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        convert(_source(tmp_path, "run"), tmp_path / "run")
    assert not (tmp_path / "run" / PRECURSOR_MAJOR_DIR).exists()
    assert find_precursor_major(tmp_path / "run") is None


def test_a_profile_bundle_gets_none(tmp_path, raws):
    from _thermo_fakes import mixed_raw
    from constellation.massspec.readers.thermo import convert

    raws["mixed"] = mixed_raw()
    convert(_source(tmp_path, "mixed"), tmp_path / "mixed")
    assert (tmp_path / "mixed.profile").is_dir()
    assert not (tmp_path / "mixed.profile" / PRECURSOR_MAJOR_DIR).exists()
    assert find_precursor_major(tmp_path / "mixed") is not None


def test_the_command_builds_it_and_says_when_it_is_current(tmp_path, capsys):
    from constellation.cli.__main__ import main

    bundle = _bundle(tmp_path, "run", dia_scans(cycles=4))
    assert main(["massspec", "precursor-major", str(bundle)]) == 0
    assert "points, levels MS1, MS2" in capsys.readouterr().out
    assert main(["massspec", "precursor-major", str(bundle)]) == 0
    assert "current" in capsys.readouterr().out
    assert main(["massspec", "precursor-major", str(tmp_path / "nothing")]) == 2


def test_convert_builds_the_layout_after_its_readers_and_before_its_manifest(tmp_path, raws, monkeypatch):
    """The reading pool (whose workers hold the .NET runtime) is shut before
    the layout is built, and the manifest that says the bundle is complete
    is written after it."""
    from constellation.massspec.readers.thermo import _convert
    from test_thermo_batch import PicklingExecutor

    events: list[str] = []

    class Recording(PicklingExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            events.append("readers shut")

    real = _convert.build_precursor_major

    def build(bundle, **kwargs):
        events.append("built" if not (bundle / "manifest.json").exists() else "built after the manifest")
        assert kwargs["workers"] <= _convert.MAX_BUILD_WORKERS
        return real(bundle, **kwargs)

    monkeypatch.setattr(_convert, "build_precursor_major", build)
    raws["run"] = _dia_raw(900)
    done: list[bool] = []
    job = _convert._Job(_source(tmp_path, "run"), tmp_path / "run", _convert.ConvertOptions(), force=False)
    _convert._run_jobs([job], workers=8, executor_factory=lambda n: Recording(),
                       on_done=lambda j: done.append((j.out_dir / "manifest.json").exists()))  # fmt: skip
    assert job.error is None and events == ["readers shut", "built"]
    assert done == [True] and find_precursor_major(tmp_path / "run") is not None
