"""One spectrum per kind of scan from a direct-infusion profile run.

Synthetic ground truth: five cycles of one MS1 (positive), one MS1 (negative,
same range) and three activations x two isolation windows of MS2. Every
replicate of a point is a known value plus a known deviation (-3, -1, +1, +3
over cycles 2-5), so the mean and standard deviation are exact; the first
cycle is garbage (an AGC first fill) that ``skip_first=1`` must leave out, m/z carries sub-step jitter, and the
points are written in small row groups across two parts so scans straddle
both.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from _massspec_fixtures import write_bundle, write_bundle_manifest

from constellation.massspec.acquisition_structure import scan_kinds
from constellation.massspec.schemas import SCAN_METADATA_TABLE
from constellation.massspec.average import (
    POINTS_DIR,
    SUMMARY_FILENAME,
    average_bundle,
    average_directory,
    find_profile_bundles,
)
from constellation.massspec.bundle import read_bundle_manifest
from constellation.massspec.scans import spectrum

STEP1 = 1 / 30  # MS1 grid
STEP2 = 1 / 15  # MS2 grid
GRID1 = 400.0 + np.arange(1, 301) * STEP1
GRID2 = 200.0 + np.arange(1, 151) * STEP2
CONDITIONS = (("HCD", 10.0), ("HCD", 20.0), ("CID", 35.0))
WINDOWS = (500.0, 516.0)
DEVIATION = (1e9, -3.0, -1.0, 1.0, 3.0)  # by cycle; the first is the AGC's first fill
SD = float(np.std([-3.0, -1.0, 1.0, 3.0], ddof=1))
RNG = np.random.default_rng(0)


def _jitter(grid: np.ndarray) -> np.ndarray:
    return grid + RNG.uniform(-1e-4, 1e-4, grid.size)


def _scans() -> list[dict]:
    scans: list[dict] = []
    for cycle, dev in enumerate(DEVIATION):
        for sign, base in (("+", 1000.0), ("-", 400.0)):
            scans.append(
                {
                    "scan": len(scans) + 1, "level": 1, "rt": cycle * 30.0 + len(scans) * 0.1,
                    "iit": 100.0 if cycle == 0 else 5.0, "tic": 1e6, "analyzer": "ITMS",
                    "filter_string": f"ITMS {sign} p ESI Full ms [400.0000-410.0000]",
                    "scan_window_lower": 400.0, "scan_window_upper": 410.0,
                    "mz": _jitter(GRID1), "intensity": np.full(GRID1.size, base) + dev,
                }
            )
        for act, energy in CONDITIONS:
            for w in WINDOWS:
                scans.append(
                    {
                        "scan": len(scans) + 1, "level": 2, "rt": cycle * 30.0 + len(scans) * 0.1,
                        "iit": 100.0 if cycle == 0 else 2.0, "tic": 2e5, "analyzer": "ITMS",
                        "filter_string": f"ITMS - p ESI r Full ms2 {w:.4f}@{act.lower()}{energy:.2f} [200.0000-210.0000]",
                        "activation_type": act, "collision_energy": energy, "precursor_mz": w,
                        "isolation_lower": w - 8, "isolation_upper": w + 8,
                        "scan_window_lower": 200.0, "scan_window_upper": 210.0,
                        "mz": _jitter(GRID2), "intensity": np.full(GRID2.size, energy * w / 100.0) + dev,
                    }
                )
    return scans


@pytest.fixture
def profile(tmp_path):
    bundle = write_bundle(tmp_path, "run.profile", _scans(), profile=True, row_group_size=100)
    return write_bundle_manifest(bundle, run="run", sha256="abc")


def _out(out):
    points = pq.read_table(sorted((out / "profile").glob("*.parquet"))).to_pydict()
    extra = pq.read_table(sorted((out / POINTS_DIR).glob("*.parquet"))).to_pydict()
    meta = pq.read_table(out / "scan_metadata.parquet").to_pylist()
    summary = pq.read_table(out / SUMMARY_FILENAME).to_pylist()
    return {k: np.asarray(v) for k, v in points.items()}, {k: np.asarray(v) for k, v in extra.items()}, meta, summary


class TestAverageBundle:
    def test_one_spectrum_per_kind_of_scan_in_order_of_appearance(self, profile, tmp_path):
        out = average_bundle(profile, skip_first=1)
        assert out == tmp_path / "run.average"
        _, _, meta, summary = _out(out)
        assert len(meta) == 8  # MS1 + and -, 3 activations x 2 windows
        assert [m["scan"] for m in meta] == list(range(1, 9))
        assert [m["level"] for m in meta] == [1, 1, 2, 2, 2, 2, 2, 2]
        assert meta[0]["filter_string"].startswith("ITMS + ") and meta[1]["filter_string"].startswith("ITMS - ")
        assert [(m["activation_type"], m["collision_energy"], m["precursor_mz"]) for m in meta[2:]] == [
            (a, e, w) for a, e in CONDITIONS for w in WINDOWS
        ]
        assert all(s["n_scans"] == 4 and s["n_skipped"] == 1 for s in summary)

    def test_mean_and_sd_are_the_planted_values_on_the_grid(self, profile):
        points, extra, meta, _ = _out(average_bundle(profile, skip_first=1))
        for m in meta:
            at = points["scan"] == m["scan"]
            if m["level"] == 1:
                base = 1000.0 if "+" in m["filter_string"] else 400.0
                grid = GRID1
            else:
                base = m["collision_energy"] * m["precursor_mz"] / 100.0
                grid = GRID2
            assert at.sum() == grid.size
            assert points["intensity"][at] == pytest.approx(base, rel=1e-9)
            assert points["mz"][at] == pytest.approx(grid, abs=2e-4)
            assert extra["intensity_sd"][at] == pytest.approx(SD, rel=1e-6)
            assert set(extra["n_scans"][at].tolist()) == {4}
        assert np.array_equal(points["mz"], extra["mz"]) and np.array_equal(points["scan"], extra["scan"])

    def test_the_first_fill_is_reported_beside_the_rest(self, profile):
        _, _, meta, summary = _out(average_bundle(profile, skip_first=1))
        for m, s in zip(meta, summary):
            want = 5.0 if m["level"] == 1 else 2.0
            assert s["iit_median_ms"] == pytest.approx(want) and m["iit"] == pytest.approx(want)
            assert s["iit_skipped_median_ms"] == pytest.approx(100.0)
            # read from a jittered scan: +/-1e-4 m/z over ~300 points
            assert s["grid_step_mz"] == pytest.approx(STEP1 if m["level"] == 1 else STEP2, abs=2e-6)

    def test_by_default_every_scan_is_kept(self, profile, tmp_path):
        out = average_bundle(profile, output_parent=tmp_path / "all")
        assert read_bundle_manifest(out)["derived_from"]["parameters"]["skip_first"] == 0
        points, extra, _, summary = _out(out)
        assert all(s["n_scans"] == 5 and s["n_skipped"] == 0 for s in summary)
        assert set(extra["n_scans"].tolist()) == {5}
        assert points["intensity"][points["scan"] == 1] == pytest.approx(1000.0 + 1e9 / 5, rel=1e-9)

    def test_it_is_a_profile_bundle_any_command_reads(self, profile):
        out = average_bundle(profile, skip_first=1)
        manifest = read_bundle_manifest(out)
        assert manifest["kind"] == "thermo_profile" and manifest["source_sha256"] == "abc"
        assert manifest["derived_from"]["name"] == "average"
        assert manifest["derived_from"]["parameters"]["skip_first"] == 1
        assert manifest["run_metadata"]["n_points_off_grid"] == 0
        assert manifest["run_metadata"]["n_scans_averaged"] == 32
        s = spectrum(out, 3)
        assert s.num_rows == GRID2.size
        assert s.column("intensity").to_numpy() == pytest.approx(10.0 * 500.0 / 100.0)

    def test_a_peak_list_from_it_lands_beside_it_not_on_the_profiles(self, profile, tmp_path):
        from constellation.massspec.centroiding import centroid_bundle

        made = centroid_bundle(average_bundle(profile), "apex", precursor_major=False)
        assert made.name.startswith("run.average.apex-")

    def test_an_off_grid_point_is_counted_not_merged(self, tmp_path):
        scans = _scans()
        scans[-1]["mz"] = scans[-1]["mz"].copy()
        scans[-1]["mz"][10] += STEP2 / 2  # half a step: on no grid point
        bundle = write_bundle_manifest(write_bundle(tmp_path, "r.profile", scans, profile=True), run="r")
        out = average_bundle(bundle)
        assert json.loads((out / "manifest.json").read_text())["run_metadata"]["n_points_off_grid"] == 1
        _, extra, meta, summary = _out(out)
        last = meta[-1]["scan"]
        assert summary[-1]["n_points_incomplete"] == 1
        assert sorted(set(extra["n_scans"][extra["scan"] == last].tolist())) == [4, 5]  # every scan kept by default

    def test_an_existing_bundle_is_refused_unless_forced(self, profile):
        average_bundle(profile)
        with pytest.raises(FileExistsError):
            average_bundle(profile)
        average_bundle(profile, force=True)

    def test_it_takes_only_a_profile_bundle(self, tmp_path):
        peaks = write_bundle_manifest(write_bundle(tmp_path, "p", _scans()[:3]))
        with pytest.raises(ValueError, match="holds peaks"):
            average_bundle(peaks)

    def test_an_orbitraps_profile_is_refused(self, tmp_path):
        scans = _scans()
        for s in scans:
            s["analyzer"] = "FTMS"
        bundle = write_bundle_manifest(write_bundle(tmp_path, "o.profile", scans, profile=True), run="o")
        with pytest.raises(ValueError, match="FTMS"):
            average_bundle(bundle)

    def test_skipping_everything_is_refused(self, profile):
        with pytest.raises(ValueError, match="nothing to average"):
            average_bundle(profile, skip_first=5)

    def test_a_failure_leaves_nothing(self, profile, tmp_path, monkeypatch):
        from constellation.massspec import average as driver

        def broken(*_, **__):
            raise RuntimeError("no")

        monkeypatch.setattr(driver.shutil, "copy2", broken)
        with pytest.raises(RuntimeError):
            average_bundle(profile, output_parent=tmp_path / "out")
        assert list((tmp_path / "out").iterdir()) == []  # neither the average nor its partial directory

    def test_the_same_run_averages_to_the_same_bytes(self, profile, tmp_path):
        a = average_bundle(profile, output_parent=tmp_path / "a", skip_first=1)
        b = average_bundle(profile, output_parent=tmp_path / "b", skip_first=1)
        files = sorted(p.relative_to(a) for p in a.rglob("*.parquet"))
        assert files and files == sorted(p.relative_to(b) for p in b.rglob("*.parquet"))
        for f in files:
            assert (a / f).read_bytes() == (b / f).read_bytes(), f

    def test_a_stellar_grid_is_followed_over_its_whole_span(self, tmp_path):
        # As the instrument stores it: a float32 step accumulated in double,
        # 19,500 points over 200-1500 m/z, zeros included (RF30's MS2 scans).
        step = float(np.float32(1 / 15))
        grid = 200.0 + np.arange(1, 19_501) * step
        rng = np.random.default_rng(1)
        scans = [
            {"scan": i + 1, "level": 2, "rt": i * 0.1, "iit": 20.0, "tic": 1.0, "analyzer": "ITMS",
             "filter_string": "ITMS + p NSI r Full ms2 500.0000@hcd24.00 [200.0000-1500.0000]",
             "activation_type": "HCD", "collision_energy": 24.0, "precursor_mz": 500.0,
             "isolation_lower": 496.0, "isolation_upper": 504.0,
             "scan_window_lower": 200.0, "scan_window_upper": 1500.0,
             "mz": grid, "intensity": np.where(rng.random(grid.size) < 0.5, 0.0, rng.gamma(1.0, 50.0, grid.size))}
            for i in range(4)
        ]
        bundle = write_bundle_manifest(write_bundle(tmp_path, "s.profile", scans, profile=True, row_group_size=7_000), run="s")
        out = average_bundle(bundle)
        points, extra, _, summary = _out(out)
        assert summary[0]["n_points_off_grid"] == 0 and summary[0]["n_points"] == grid.size
        assert summary[0]["grid_step_mz"] == pytest.approx(step, rel=1e-12)
        assert np.max(np.abs(points["mz"] - grid)) < 1e-9
        planted = np.mean([s["intensity"] for s in scans], axis=0)
        assert points["intensity"] == pytest.approx(planted, rel=1e-12, abs=1e-12)
        assert set(extra["n_scans"].tolist()) == {4}


@pytest.fixture
def run_dir(tmp_path):
    """A converted directory: two runs' profile bundles, one run's peak bundle,
    an Orbitrap profile that must fail, and an unrelated folder."""
    d = tmp_path / "data"
    d.mkdir()
    for run in ("b_run", "a_run"):
        write_bundle_manifest(write_bundle(d, f"{run}.profile", _scans(), profile=True), run=run)
    write_bundle_manifest(write_bundle(d, "a_run", _scans()[:3]))  # the centroided peak bundle
    orbi = _scans()
    for s in orbi:
        s["analyzer"] = "FTMS"
    write_bundle_manifest(write_bundle(d, "orbi.profile", orbi, profile=True), run="orbi")
    (d / "notes").mkdir()
    return d


class TestAverageDirectory:
    def test_only_run_profile_bundles_are_found(self, run_dir):
        names = [b.name for b in find_profile_bundles(run_dir)]
        assert names == ["a_run.profile", "b_run.profile", "orbi.profile"]
        average_bundle(run_dir / "a_run.profile")
        # the new a_run.average is a profile bundle too, but made from a run: not found
        assert [b.name for b in find_profile_bundles(run_dir)] == names

    def test_every_run_is_averaged_and_one_failure_does_not_stop_the_rest(self, run_dir):
        results = average_directory(run_dir)
        assert [(r.bundle.name, r.status) for r in results] == [
            ("a_run.profile", "ok"), ("b_run.profile", "ok"), ("orbi.profile", "error"),
        ]
        assert "FTMS" in results[2].detail and not (run_dir / "orbi.average").exists()
        for run in ("a_run", "b_run"):
            summary = pq.read_table(run_dir / f"{run}.average" / SUMMARY_FILENAME).to_pylist()
            assert len(summary) == 8 and all(s["n_scans"] == 5 for s in summary)

    def test_averaged_runs_are_skipped_unless_forced(self, run_dir):
        average_directory(run_dir)
        stamp = (run_dir / "a_run.average" / "manifest.json").stat().st_mtime_ns
        again = average_directory(run_dir)
        assert [r.status for r in again[:2]] == ["skipped", "skipped"]
        assert (run_dir / "a_run.average" / "manifest.json").stat().st_mtime_ns == stamp
        assert [r.status for r in average_directory(run_dir, force=True)[:2]] == ["ok", "ok"]

    def test_output_parent_and_skip_first_reach_every_run(self, run_dir, tmp_path):
        results = average_directory(run_dir, output_parent=tmp_path / "out", skip_first=1)
        assert results[0].output == tmp_path / "out" / "a_run.average"
        summary = pq.read_table(tmp_path / "out" / "b_run.average" / SUMMARY_FILENAME).to_pylist()
        assert all(s["n_scans"] == 4 and s["n_skipped"] == 1 for s in summary)
        assert not (run_dir / "a_run.average").exists()

    def test_a_negative_skip_first_is_refused_before_anything_is_written(self, run_dir):
        with pytest.raises(ValueError, match="skip_first"):
            average_directory(run_dir, skip_first=-1)
        assert not list(run_dir.glob("*.average"))


def test_scans_are_grouped_by_what_was_acquired(profile) -> None:
    table = pq.read_table(profile / "scan_metadata.parquet")
    group, n = scan_kinds(table)
    assert n == 8
    assert np.bincount(group).tolist() == [5] * 8
    assert group[:8].tolist() == list(range(8))  # first cycle sets the order


def _meta(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"scan": i + 1, "level": 1, "analyzer": "ITMS", "scan_window_lower": 400.0, "scan_window_upper": 410.0, **r}
         for i, r in enumerate(rows)],
        schema=SCAN_METADATA_TABLE,
    )


class TestScanKinds:
    def test_faims_voltages_are_kinds_of_their_own(self):
        rows = [{"faims_cv": cv, "filter_string": f"ITMS + p NSI cv={cv:.2f} r Full ms [400.0000-410.0000]"}
                for _ in range(3) for cv in (-45.0, -65.0)]
        kind, n = scan_kinds(_meta(rows))
        assert n == 2 and kind.tolist() == [0, 1] * 3

    def test_a_voltage_alone_separates_scans_with_no_filter_string(self):
        kind, n = scan_kinds(_meta([{"faims_cv": cv} for cv in (-45.0, -65.0, -45.0)]))
        assert n == 2 and kind.tolist() == [0, 1, 0]

    def test_scan_rates_are_kinds_of_their_own(self):
        # Same range and level; the filter string's rate token (r rapid, n normal) differs, and so does the grid.
        rows = [{"filter_string": f"ITMS + p NSI {rate} Full ms [400.0000-410.0000]"} for rate in "rnrn"]
        kind, n = scan_kinds(_meta(rows))
        assert n == 2 and kind.tolist() == [0, 1, 0, 1]

    def test_polarities_are_kinds_of_their_own(self):
        rows = [{"filter_string": f"ITMS {sign} p NSI r Full ms [400.0000-410.0000]"} for sign in "+-+"]
        assert scan_kinds(_meta(rows))[0].tolist() == [0, 1, 0]

    def test_last_digit_jitter_is_one_kind_and_order_is_first_appearance(self):
        rows = [
            {"level": 2, "precursor_mz": 600.0, "isolation_lower": 596.0, "isolation_upper": 604.0},
            {"level": 2, "precursor_mz": 500.0, "isolation_lower": 496.0001, "isolation_upper": 504.0},
            {"level": 2, "precursor_mz": 500.0, "isolation_lower": 496.0002, "isolation_upper": 504.0},
            {"level": 2, "precursor_mz": 600.0, "isolation_lower": 596.0, "isolation_upper": 604.0},
        ]
        kind, n = scan_kinds(_meta(rows))
        assert n == 2 and kind.tolist() == [0, 1, 1, 0]

    def test_an_empty_table_has_no_kinds(self):
        kind, n = scan_kinds(_meta([]))
        assert n == 0 and kind.size == 0


def test_a_faims_run_is_averaged_voltage_by_voltage(tmp_path):
    scans = []
    for _cycle in range(3):
        for cv, level in ((-45.0, 1000.0), (-65.0, 10.0)):
            scans.append({
                "scan": len(scans) + 1, "level": 1, "rt": len(scans) * 0.5, "iit": 5.0, "tic": 1e6, "analyzer": "ITMS",
                "faims_cv": cv, "filter_string": f"ITMS + p NSI cv={cv:.2f} r Full ms [400.0000-410.0000]",
                "scan_window_lower": 400.0, "scan_window_upper": 410.0,
                "mz": _jitter(GRID1), "intensity": np.full(GRID1.size, level),
            })
    bundle = write_bundle_manifest(write_bundle(tmp_path, "f.profile", scans, profile=True), run="f")
    points, _, meta, _ = _out(average_bundle(bundle))
    assert [m["faims_cv"] for m in meta] == [-45.0, -65.0]
    assert points["intensity"][points["scan"] == 1] == pytest.approx(1000.0)
    assert points["intensity"][points["scan"] == 2] == pytest.approx(10.0)


class TestCommand:
    def _parse(self, *argv):
        from constellation.massspec.cli import build_parser

        root = argparse.ArgumentParser()
        build_parser(root.add_subparsers(dest="command"))
        return root.parse_args(["massspec", "average", *argv])

    def test_it_writes_and_refuses_to_overwrite(self, profile, tmp_path, capsys):
        args = self._parse(str(profile))
        assert args.func(args) == 0
        assert (tmp_path / "run.average" / "manifest.json").is_file()
        assert args.func(args) == 4
        assert self._parse(str(profile), "--force").func(self._parse(str(profile), "--force")) == 0

    def test_a_peak_bundle_is_a_usage_error(self, tmp_path):
        peaks = write_bundle_manifest(write_bundle(tmp_path, "p", _scans()[:3]))
        args = self._parse(str(peaks))
        assert args.func(args) == 2

    def test_skip_first_and_output_dir_reach_the_driver(self, profile, tmp_path):
        assert self._parse(str(profile)).skip_first == 0
        args = self._parse(str(profile), "--skip-first", "1", "-o", str(tmp_path / "elsewhere"))
        assert args.func(args) == 0
        summary = pq.read_table(tmp_path / "elsewhere" / "run.average" / SUMMARY_FILENAME).to_pylist()
        assert all(s["n_scans"] == 4 and s["n_skipped"] == 1 for s in summary)

    def test_a_directory_averages_every_run_and_reports_failures(self, run_dir, capsys):
        args = self._parse(str(run_dir))
        assert args.func(args) == 5  # orbi.profile failed
        assert (run_dir / "a_run.average" / "manifest.json").is_file()
        assert (run_dir / "b_run.average" / "manifest.json").is_file()
        err = capsys.readouterr().err
        assert "2 averaged, 0 skipped, 1 failed" in err

    def test_a_directory_rerun_skips_what_is_done(self, tmp_path, capsys):
        d = tmp_path / "ok"
        d.mkdir()
        write_bundle_manifest(write_bundle(d, "r.profile", _scans(), profile=True), run="r")
        assert self._parse(str(d)).func(self._parse(str(d))) == 0
        assert self._parse(str(d)).func(self._parse(str(d))) == 0
        assert "0 averaged, 1 skipped, 0 failed" in capsys.readouterr().err

    def test_a_directory_with_nothing_to_average_is_a_usage_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        args = self._parse(str(tmp_path / "empty"))
        assert args.func(args) == 2


# ── what the bundle says it is ────────────────────────────────────────


@pytest.fixture
def family(tmp_path):
    """A run as `convert` leaves it (its peak list and its profile, tied by
    hash), a peak list made from the profile, the run's average and a peak
    list made from the average."""
    from constellation.massspec.centroiding import centroid_bundle

    d = tmp_path / "data"
    d.mkdir()
    scans = _scans()
    write_bundle_manifest(write_bundle(d, "run", [dict(s, mz=s["mz"][:5], intensity=s["intensity"][:5]) for s in scans]),
                          run="run", sha256="S")
    profile = write_bundle_manifest(write_bundle(d, "run.profile", scans, profile=True), run="run", sha256="S")
    centroid_bundle(profile, "apex", precursor_major=False)
    average = average_bundle(profile)
    average_list = centroid_bundle(average, "apex")
    return {"dir": d, "profile": profile, "average": average, "average_list": average_list}


class TestWhatTheBundleSaysItIs:
    def test_an_average_says_its_spectra_are_kinds_of_scan(self, family):
        from constellation.massspec.bundle import KIND_OF_SCAN_INDEX, spectrum_index

        assert spectrum_index(read_bundle_manifest(family["average"])) == KIND_OF_SCAN_INDEX
        assert spectrum_index(read_bundle_manifest(family["average_list"])) == KIND_OF_SCAN_INDEX
        assert spectrum_index(read_bundle_manifest(family["profile"])) == "scan"

    def test_it_is_not_one_of_the_runs_bundles(self, family):
        from constellation.massspec.bundle import related_bundles
        from constellation.massspec.bundle_summary import summarize_bundle
        from constellation.massspec.scans import open_acquisition

        run = family["dir"] / "run"
        assert sorted(r["bundle"].name for r in related_bundles(run)) == ["run.apex-7752bdef", "run.profile"]
        names = [b["name"] for b in summarize_bundle(run)["bundles"]]
        assert len(names) == len(set(names)) == 3
        assert all(".average" not in r["bundle"].name for r in open_acquisition(family["profile"]).related)
        # The average's own family is its peak list.
        assert [r["bundle"].name for r in related_bundles(family["average"])] == [family["average_list"].name]

    def test_its_peak_list_is_named_for_it_and_has_no_precursor_major_layout(self, family):
        made = family["average_list"]
        assert made.name.startswith("run.average.apex-")
        assert not (made / "precursor_major").exists()
        assert "precursor_major" not in read_bundle_manifest(made)["outputs"]

    def test_what_needs_a_runs_scans_refuses_it(self, family, tmp_path):
        from constellation.massspec.acquisitions import acquisition_record_from_bundle
        from constellation.massspec.io.encyclopedia import write_dia
        from constellation.massspec.precursor_major import build_precursor_major
        from constellation.massspec.sweep.engine import sweep

        for bundle in (family["average"], family["average_list"]):
            with pytest.raises(ValueError, match="kind of scan.*an acquisition"):
                acquisition_record_from_bundle(bundle)
        with pytest.raises(ValueError, match="kind of scan.*a sweep"):
            sweep(family["average_list"], None, None)
        with pytest.raises(ValueError, match="kind of scan.*precursor-major"):
            build_precursor_major(family["average_list"])
        with pytest.raises(ValueError, match="kind of scan.*spectra file"):
            write_dia([family["average_list"]], tmp_path / "x.dia")
        # The run's own bundles are as they were.
        assert acquisition_record_from_bundle(family["dir"] / "run")["source_sha256"] == "S"


def test_its_scan_metadata_is_true_of_each_average(tmp_path):
    scans = _scans()
    for i, s in enumerate(scans):
        s["agc_target"] = 30_000  # a method setting: every replicate shares it
        s["agc_fill"] = 0.1 + 0.01 * i  # a reading: one scan's
        s["precursor_scan"] = 1.0 if s["level"] == 2 else None  # a scan number of the run
    scans.append(dict(scans[2], scan=len(scans) + 1, precursor_mz=900.0, isolation_lower=892.0, isolation_upper=908.0,
                      filter_string="ITMS - p ESI r Full ms2 900.0000@hcd10.00 [200.0000-210.0000]"))  # a kind seen once
    bundle = write_bundle_manifest(write_bundle(tmp_path, "m.profile", scans, profile=True), run="m")
    meta = pq.read_table(average_bundle(bundle) / "scan_metadata.parquet").to_pylist()
    assert len(meta) == 9
    assert all(m["agc_target"] == 30_000 for m in meta)
    assert all(m["agc_fill"] is None for m in meta[:8]) and meta[8]["agc_fill"] is not None  # one replicate agrees with itself
    assert all(m["precursor_scan"] is None and m["master_scan"] is None for m in meta)
    assert all(m["trailer_extras"] is None and m["intensity_floor"] is None for m in meta)
    assert [m["precursor_mz"] for m in meta[2:8]] == [w for _ in CONDITIONS for w in WINDOWS]


# ── where it is written, and what is never removed ────────────────────


class TestOutputSafety:
    def test_o_names_where_the_average_goes_in_both_modes(self, profile, run_dir, tmp_path):
        assert average_bundle(profile, output_parent=tmp_path / "x") == tmp_path / "x" / "run.average"
        results = average_directory(run_dir, output_parent=tmp_path / "y")
        assert {r.output for r in results if r.status == "ok"} == {tmp_path / "y" / "a_run.average", tmp_path / "y" / "b_run.average"}

    def test_force_never_removes_what_is_not_an_average_of_this_run(self, profile, tmp_path):
        from constellation.massspec.bundle import prepare_derived_bundle

        # The profile's own parent (what `-o <parent> --force` removed before).
        with pytest.raises(ValueError, match="holds"):
            prepare_derived_bundle(tmp_path, profile, force=True)
        # Someone's directory where the average would go.
        theirs = tmp_path / "out" / "run.average"
        theirs.mkdir(parents=True)
        (theirs / "figure.svg").write_text("keep")
        with pytest.raises(FileExistsError, match="not a bundle made from"):
            average_bundle(profile, output_parent=tmp_path / "out", force=True)
        assert (theirs / "figure.svg").read_text() == "keep" and (profile / "manifest.json").is_file()

    def test_a_failed_forced_rerun_keeps_the_average_there(self, profile, tmp_path):
        out = average_bundle(profile)
        before = (out / "manifest.json").read_text()
        with pytest.raises(ValueError, match="nothing to average"):
            average_bundle(profile, skip_first=99, force=True)
        assert (out / "manifest.json").read_text() == before
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]

    def test_a_forced_rerun_replaces_it_whole(self, profile):
        out = average_bundle(profile)
        (out / "stale").write_text("x")
        assert average_bundle(profile, skip_first=1, force=True) == out
        assert not (out / "stale").exists()
        assert read_bundle_manifest(out)["derived_from"]["parameters"]["skip_first"] == 1

    def test_what_a_killed_run_leaves_is_cleared_and_the_run_averaged(self, run_dir):
        partial = run_dir / ".a_run.average.partial"
        (partial / "profile").mkdir(parents=True)  # a run killed while writing
        results = {r.bundle.name: r for r in average_directory(run_dir)}
        assert results["a_run.profile"].status == "ok"
        assert not partial.exists() and (run_dir / "a_run.average" / "manifest.json").is_file()

    def test_an_unfinished_directory_where_the_average_goes_is_reported_not_skipped(self, run_dir):
        old = run_dir / "a_run.average"
        (old / "profile").mkdir(parents=True)  # no manifest: not a finished average
        results = {r.bundle.name: r for r in average_directory(run_dir)}
        assert results["a_run.profile"].status == "error" and "not a bundle made from" in results["a_run.profile"].detail
        assert (old / "profile").is_dir()  # left as it was
        assert results["b_run.profile"].status == "ok"

    def test_a_bundle_in_an_older_format_is_reported(self, run_dir):
        old = run_dir / "old.profile"
        write_bundle_manifest(write_bundle(run_dir, "old.profile", _scans()[:4], profile=True), run="old")
        manifest = json.loads((old / "manifest.json").read_text())
        manifest["schema_version"] = 2
        (old / "manifest.json").write_text(json.dumps(manifest))
        results = {r.bundle.name: r for r in average_directory(run_dir)}
        assert results["old.profile"].status == "error" and "older" in results["old.profile"].detail


class TestCommandReports:
    def _run(self, *argv):
        from constellation.massspec.cli import build_parser

        root = argparse.ArgumentParser()
        build_parser(root.add_subparsers(dest="command"))
        args = root.parse_args(["massspec", "average", *argv])
        return args.func(args)

    def test_it_says_what_it_made(self, profile, capsys):
        assert self._run(str(profile), "--skip-first", "1") == 0
        out, err = capsys.readouterr()
        assert "8 kind(s) of scan from 32 scans (8 skipped)" in out and "warning" not in err

    def test_points_off_the_grid_are_a_warning(self, tmp_path, capsys):
        scans = _scans()
        scans[-1]["mz"] = scans[-1]["mz"].copy()
        scans[-1]["mz"][10] += STEP2 / 2
        bundle = write_bundle_manifest(write_bundle(tmp_path, "r.profile", scans, profile=True), run="r")
        assert self._run(str(bundle)) == 0
        assert "warning: r.average: 1 points sat on no grid point" in capsys.readouterr().err

    def test_something_else_where_it_goes_is_exit_4_and_kept(self, profile, tmp_path, capsys):
        theirs = tmp_path / "o" / "run.average"
        theirs.mkdir(parents=True)
        (theirs / "keep").write_text("x")
        assert self._run(str(profile), "-o", str(tmp_path / "o"), "--force") == 4
        assert (theirs / "keep").is_file()

    def test_a_directory_of_only_old_bundles_is_reported_not_called_empty(self, tmp_path, capsys):
        d = tmp_path / "d"
        d.mkdir()
        b = write_bundle_manifest(write_bundle(d, "old.profile", _scans()[:4], profile=True), run="old")
        manifest = json.loads((b / "manifest.json").read_text())
        manifest["schema_version"] = 2
        (b / "manifest.json").write_text(json.dumps(manifest))
        assert self._run(str(d)) == 5
        assert "0 averaged, 0 skipped, 1 failed" in capsys.readouterr().err


def test_summarize_describes_the_average_not_the_run(tmp_path):
    """Found averaging RF30: summarize called 101 averaged spectra a DDA run
    and quoted the run's 31,300 scans with a blank trailer."""
    from constellation.massspec.bundle_summary import summarize_bundle

    scans = _scans()
    for s in scans:
        if s["level"] == 1 and s["filter_string"].startswith("ITMS - "):
            s["iit"] = None  # a blank trailer on every scan of one kind
    bundle = write_bundle_manifest(write_bundle(tmp_path, "s.profile", scans, profile=True), run="s")
    run = json.loads((bundle / "manifest.json").read_text())
    run["run_metadata"]["n_scans_without_injection_time"] = 5
    (bundle / "manifest.json").write_text(json.dumps(run))
    summary = summarize_bundle(average_bundle(bundle))
    assert summary["acquisition_type"] is None
    assert summary["n_scans_without_injection_time"] == 1  # one kind of scan, not the run's five scans
