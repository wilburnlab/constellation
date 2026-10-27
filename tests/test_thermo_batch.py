"""Tests for the Thermo conversion driver (``_convert``) and ``convert_batch``.

Two tiers:

1. **The driver on stand-in raw files** (``_thermo_fakes``, no DLLs):
   range planning; ranges written as parts by an executor that sends every
   task and result through pickle, as a process pool does, compared with
   one range in this process; ``convert_batch``'s skip / force / per-file
   errors; an interrupt clearing what it wrote; an existing bundle left
   alone when a conversion is refused.

2. **CLI integration** — drive ``_cmd_massspec_convert`` end-to-end with
   ``convert`` / ``convert_batch`` monkeypatched to canned results. Covers
   the routing, the summary lines and exit codes.

Real spawned workers on a real ``.raw`` are exercised by
``test_thermo_reader.py``'s real-file tier (gated on
``$CONSTELLATION_TEST_THERMO_RAW``).
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import Executor, Future
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from _thermo_fakes import long_raw, mixed_raw
from constellation.massspec.bundle import peaks_dataset
from constellation.massspec.readers.thermo import (
    BatchResult,
    convert,
    convert_batch,
    plan_ranges,
)
from constellation.massspec.readers.thermo import _convert, _read
from constellation.massspec.readers.thermo.manifest import MANIFEST_FILENAME


# ──────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────


def _make_raw_inputs(tmp_path: Path, stems: list[str]) -> list[Path]:
    """Create empty ``<stem>.raw`` files in ``tmp_path/in/`` and return paths."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    paths: list[Path] = []
    for stem in stems:
        p = in_dir / f"{stem}.raw"
        p.write_bytes(stem.encode())
        paths.append(p)
    return paths


@pytest.fixture
def raws(monkeypatch, fake_centroider):
    """Map a ``.raw``'s stem to the stand-in raw file it opens as.

    A stem missing from the map fails to open, as a corrupt file would.
    """
    files: dict[str, object] = {}

    def open_raw(path):
        stem = Path(path).stem
        if stem not in files:
            raise OSError(f"Thermo RawFileReader could not open {path}")
        return _read._RawFileHandle(files[stem])

    monkeypatch.setattr(_read, "_open_raw_file", open_raw)
    yield files
    _convert._close_open_files()


class PicklingExecutor(Executor):
    """Runs tasks here, but sends each task and result through pickle."""

    def __init__(self, interrupt_on: str | None = None):
        self.calls: list[str] = []
        self.interrupt_on = interrupt_on

    def submit(self, fn, /, *args, **kwargs) -> Future:
        self.calls.append(fn.__name__)
        future: Future = Future()
        if fn.__name__ == self.interrupt_on:
            raise KeyboardInterrupt
        fn, args = pickle.loads(pickle.dumps((fn, args)))
        try:
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


# ──────────────────────────────────────────────────────────────────────
# 1. The driver on stand-in raw files
# ──────────────────────────────────────────────────────────────────────


class TestPlanRanges:
    def test_one_worker_one_range(self):
        assert plan_ranges(1, 81850, 1) == [(1, 81850)]

    def test_ranges_cover_the_file_in_order(self):
        ranges = plan_ranges(5, 81854, 8)
        assert ranges[0][0] == 5 and ranges[-1][1] == 81854
        assert all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:]))
        assert len(ranges) == 8 * _convert.RANGES_PER_WORKER

    def test_ranges_not_shorter_than_the_minimum(self):
        ranges = plan_ranges(1, 5000, 8)
        assert len(ranges) == 2
        assert min(hi - lo + 1 for lo, hi in ranges) >= _convert.MIN_RANGE_SCANS

    def test_no_scans_no_ranges(self):
        assert plan_ranges(1, 0, 4) == []


class TestDriver:
    def test_parts_equal_one_range(self, tmp_path, raws):
        raws["run"] = long_raw(25_000)
        src = _make_raw_inputs(tmp_path, ["run"])[0]

        one = convert(src, tmp_path / "one", workers=1)
        executor = PicklingExecutor()
        job = _convert._Job(src, tmp_path / "many", _convert.ConvertOptions(), force=False)
        _convert._run_jobs([job], workers=4, executor_factory=lambda n: executor)
        assert job.error is None

        parts = sorted((tmp_path / "many" / "peaks").iterdir())
        assert len(parts) == len(plan_ranges(1, 25_000, 4)) == 12
        assert executor.calls.count("_run_range") == 12
        assert executor.calls[:4] == ["_warm"] * 4  # every worker started first
        for table in ("scan_metadata.parquet", "acquisition_metadata.parquet"):
            a = pq.read_table(tmp_path / "one" / table).drop_columns(["source_path"] if "acq" in table else [])
            b = pq.read_table(tmp_path / "many" / table).drop_columns(["source_path"] if "acq" in table else [])
            assert a.equals(b), table
        assert peaks_dataset(tmp_path / "one").to_table().equals(
            peaks_dataset(tmp_path / "many").to_table()
        )
        assert job.manifest.parameters["workers"] == 4
        assert one.parameters["workers"] == 1
        assert one.source_sha256 == job.manifest.source_sha256

    def test_interrupt_clears_what_was_written(self, tmp_path, raws):
        raws["run"] = long_raw(9_000)
        src = _make_raw_inputs(tmp_path, ["run"])[0]
        job = _convert._Job(src, tmp_path / "out", _convert.ConvertOptions(), force=False)
        with pytest.raises(KeyboardInterrupt):
            _convert._run_jobs(
                [job], workers=2,
                executor_factory=lambda n: PicklingExecutor(interrupt_on="_run_range"),
            )
        assert list((tmp_path / "out").iterdir()) == []

    def test_a_refused_conversion_leaves_an_existing_bundle(self, tmp_path, raws):
        raws["run"] = mixed_raw()
        src = _make_raw_inputs(tmp_path, ["run"])[0]
        convert(src, tmp_path / "out")
        before = sorted(p.name for p in (tmp_path / "out").iterdir())
        with pytest.raises(FileExistsError):
            convert(src, tmp_path / "out")
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == before


class TestConvertBatch:
    def test_each_file_its_own_bundle_and_outcome(self, tmp_path, raws):
        paths = _make_raw_inputs(tmp_path, ["a", "bad", "c"])
        raws["a"], raws["c"] = mixed_raw(), long_raw(30)
        out = tmp_path / "out"
        results = convert_batch(paths, out)

        assert [r.status for r in results] == ["ok", "error", "ok"]
        assert [r.input_path for r in results] == paths
        assert "could not open" in results[1].detail
        assert not any((out / "bad").iterdir())  # a failed file leaves nothing behind
        assert peaks_dataset(out / "a").count_rows() == 7
        assert json.loads((out / "c" / MANIFEST_FILENAME).read_text())["run_metadata"]["n_scans"] == 30

    def test_skips_finished_bundles_unless_forced(self, tmp_path, raws):
        paths = _make_raw_inputs(tmp_path, ["a", "b"])
        raws["a"], raws["b"] = mixed_raw(), mixed_raw()
        out = tmp_path / "out"
        convert_batch(paths[:1], out)
        stamp = (out / "a" / MANIFEST_FILENAME).read_text()

        results = convert_batch(paths, out)
        assert [r.status for r in results] == ["skipped", "ok"]
        assert (out / "a" / MANIFEST_FILENAME).read_text() == stamp

        results = convert_batch(paths, out, force=True)
        assert [r.status for r in results] == ["ok", "ok"]

    def test_all_skipped_converts_nothing(self, tmp_path, raws, monkeypatch):
        paths = _make_raw_inputs(tmp_path, ["a"])
        raws["a"] = mixed_raw()
        convert_batch(paths, tmp_path / "out")
        monkeypatch.setattr(
            "constellation.massspec.readers.thermo._batch._run_jobs",
            lambda *a, **k: pytest.fail("nothing should be converted"),
        )
        assert [r.status for r in convert_batch(paths, tmp_path / "out")] == ["skipped"]

    def test_options_reach_every_file(self, tmp_path, raws):
        paths = _make_raw_inputs(tmp_path, ["a"])
        raws["a"] = mixed_raw()
        convert_batch(paths, tmp_path / "out", profile=True, capture_trailer_extras=False,
                      compute_sha256=False)
        manifest = json.loads((tmp_path / "out" / "a" / MANIFEST_FILENAME).read_text())
        assert manifest["parameters"]["peaks"] == "profile"
        assert manifest["parameters"]["capture_trailer_extras"] is False
        assert manifest["source_sha256"] is None

    def test_duplicate_stem_raises(self, tmp_path):
        a = tmp_path / "x" / "run.raw"
        b = tmp_path / "y" / "run.raw"
        for p in (a, b):
            p.parent.mkdir()
            p.write_bytes(b"")
        with pytest.raises(ValueError, match="duplicate output bundle"):
            convert_batch([a, b], tmp_path / "out")

    def test_empty_paths_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            convert_batch([], tmp_path / "out")


# ──────────────────────────────────────────────────────────────────────
# 2. CLI integration tier
# ──────────────────────────────────────────────────────────────────────


def _make_args(
    *,
    input_path: Path,
    output_dir: Path | None = None,
    threads: int | None = 1,
    force: bool = False,
    no_progress: bool = True,
) -> argparse.Namespace:
    """Build a Namespace matching the argparse layout for `convert`."""
    return argparse.Namespace(
        input=input_path,
        output_dir=output_dir,
        threads=threads,
        rt_bin_width_s=60.0,
        profile=False,
        capture_trailer_extras=True,
        compute_sha256=True,
        force=force,
        batch_size=512,
        allow_empty_levels=False,
        no_progress=no_progress,
    )


class TestCliRouter:
    def test_single_file(self, tmp_path, monkeypatch, capsys):
        """A single file emits 'wrote bundle:' and the per-level summary."""
        from constellation.massspec import cli as ms_cli

        src = tmp_path / "sample.raw"
        src.write_bytes(b"")
        out = tmp_path / "out"

        class _Manifest:
            outputs = {
                "peaks": "peaks.parquet",
                "scan_metadata": "scan_metadata.parquet",
                "acquisition_metadata": "acquisition_metadata.parquet",
            }
            run_metadata = {
                "levels": [
                    {"level": 1, "n_scans": 3, "n_peaks": 30, "peaks_from": {"thermo": 3}}
                ]
            }

        def fake_thermo_convert(src_arg, bundle_dir, **kwargs):
            Path(bundle_dir).mkdir(parents=True, exist_ok=True)
            (Path(bundle_dir) / MANIFEST_FILENAME).write_text("{}")
            return _Manifest()

        monkeypatch.setattr(
            "constellation.massspec.readers.thermo._netruntime.require_thermo",
            lambda: None,
        )
        monkeypatch.setattr(
            "constellation.massspec.readers.thermo.convert",
            fake_thermo_convert,
        )

        args = _make_args(input_path=src, output_dir=out, threads=1)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 0
        assert "wrote bundle:" in captured.out
        # Every level says how its peaks were made.
        assert "MS1: 3 scans, 30 peaks (3 centroided by Thermo's centroider)" in captured.out
        assert "batch convert:" not in captured.err

    def test_batch_directory_routes_through_convert_batch(
        self, tmp_path, monkeypatch, capsys
    ):
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for stem in ("a", "b"):
            (in_dir / f"{stem}.raw").write_bytes(b"")

        def fake_convert_batch(paths, output_parent, **kwargs):
            return [
                BatchResult(
                    input_path=p,
                    bundle_dir=output_parent / p.stem,
                    status="ok",
                )
                for p in paths
            ]

        monkeypatch.setattr(
            "constellation.massspec.readers.thermo._netruntime.require_thermo",
            lambda: None,
        )
        monkeypatch.setattr(
            "constellation.massspec.readers.thermo.convert_batch",
            fake_convert_batch,
        )

        out = tmp_path / "out"
        args = _make_args(input_path=in_dir, output_dir=out, threads=2)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 0
        assert "batch convert: 2 succeeded, 0 skipped, 0 failed" in captured.err
        assert "wrote bundle:" not in captured.out

    def test_batch_exits_5_on_error(self, tmp_path, monkeypatch, capsys):
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        bad = in_dir / "bad.raw"
        good = in_dir / "good.raw"
        bad.write_bytes(b"")
        good.write_bytes(b"")

        def fake_convert_batch(paths, output_parent, **kwargs):
            return [
                BatchResult(
                    input_path=p,
                    bundle_dir=output_parent / p.stem,
                    status="error" if p.stem == "bad" else "ok",
                    detail="RuntimeError: nope" if p.stem == "bad" else None,
                )
                for p in paths
            ]

        monkeypatch.setattr(
            "constellation.massspec.readers.thermo._netruntime.require_thermo",
            lambda: None,
        )
        monkeypatch.setattr(
            "constellation.massspec.readers.thermo.convert_batch",
            fake_convert_batch,
        )

        args = _make_args(input_path=in_dir, threads=2)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 5
        assert "1 failed" in captured.err
        assert "  failed:" in captured.err
        assert "bad.raw" in captured.err
        assert "RuntimeError: nope" in captured.err

    def test_empty_directory_exits_2(self, tmp_path, monkeypatch, capsys):
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "empty"
        in_dir.mkdir()

        monkeypatch.setattr(
            "constellation.massspec.readers.thermo._netruntime.require_thermo",
            lambda: None,
        )

        args = _make_args(input_path=in_dir, threads=2)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 2
        assert "no .raw files in" in captured.err

    def test_missing_input_exits_2(self, tmp_path, capsys):
        from constellation.massspec import cli as ms_cli

        args = _make_args(input_path=tmp_path / "does-not-exist.raw")
        rc = ms_cli._cmd_massspec_convert(args)
        captured = capsys.readouterr()
        assert rc == 2
        assert "input not found" in captured.err

    @pytest.mark.parametrize("threads, expected", [(3, 3), (None, "default")])
    def test_single_file_threads_are_workers(
        self, tmp_path, monkeypatch, capsys, threads, expected
    ):
        """A single .raw is converted by --threads worker processes (the
        cores, up to 8, by default); it is not routed through the batch."""
        from constellation.massspec import cli as ms_cli
        from constellation.massspec.readers.thermo import default_workers

        src = tmp_path / "solo.raw"
        src.write_bytes(b"")
        seen: dict[str, object] = {}

        class _Manifest:
            outputs = {"peaks": "peaks", "scan_metadata": "scan_metadata.parquet",
                       "acquisition_metadata": "acquisition_metadata.parquet"}
            run_metadata = {"levels": []}

        def fake_thermo_convert(src_arg, bundle_dir, **kwargs):
            seen.update(kwargs)
            return _Manifest()

        monkeypatch.setattr(
            "constellation.massspec.readers.thermo._netruntime.require_thermo", lambda: None
        )
        monkeypatch.setattr("constellation.massspec.readers.thermo.convert", fake_thermo_convert)
        monkeypatch.setattr(
            "constellation.massspec.readers.thermo.convert_batch",
            lambda *a, **k: pytest.fail("a single file is not a batch"),
        )

        args = _make_args(input_path=src, threads=threads)
        assert ms_cli._cmd_massspec_convert(args) == 0
        assert seen["workers"] == (default_workers() if expected == "default" else expected)
        assert "wrote bundle:" in capsys.readouterr().out

    def test_directory_glob_case_insensitive_and_deduped(
        self, tmp_path, monkeypatch
    ):
        """Directory mode picks up *.raw and *.RAW without double-counting.

        On case-insensitive filesystems both globs hit the same dirent;
        we dedupe via resolve() so the input list has one entry per
        physical file regardless of casing.
        """
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        # Lowercase + uppercase + a non-.raw file.
        (in_dir / "lower.raw").write_bytes(b"")
        (in_dir / "ignore.txt").write_bytes(b"")
        # Skip an explicit uppercase fixture: case-insensitive
        # filesystems would have it collide with a sibling lowercase,
        # and case-sensitive filesystems still exercise the second
        # glob below — _collect_raw_files runs both patterns and
        # dedupes by resolved path, which is the property under test.

        found = ms_cli._collect_raw_files(in_dir)
        # Either case: one match (lower.raw); ignore.txt is filtered.
        assert len(found) == 1
        assert found[0].name == "lower.raw"
