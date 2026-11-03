"""Peak lists made from a profile bundle, each a complete bundle of its own."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from _massspec_fixtures import write_bundle, write_bundle_manifest
from _thermo_fakes import mixed_raw

from constellation.massspec.bundle import peaks_dataset, read_bundle_manifest
from constellation.massspec.centroiding import ALGORITHMS, centroid_bundle, peak_list_key, pick_apex
from constellation.massspec.io.encyclopedia import write_dia
from constellation.massspec.scans import open_acquisition, spectrum


def _peak(grid: np.ndarray, centre: float, height: float, s: float = 0.14) -> np.ndarray:
    return height * np.exp(-0.5 * ((grid - centre) / s) ** 2)


GRID = np.arange(400.0, 420.0, 1 / 15)


class TestApex:
    def test_it_finds_each_maximum_within_a_small_fraction_of_a_grid_step(self):
        y = _peak(GRID, 403.021, 1000.0) + _peak(GRID, 409.5, 400.0)
        mz, area = pick_apex(GRID, y, None)
        assert mz == pytest.approx([403.021, 409.5], abs=1e-3)
        assert area[0] / area[1] == pytest.approx(2.5, rel=0.01)

    def test_its_areas_partition_the_scan_as_thermos_do(self):
        y = _peak(GRID, 403.0, 1000.0) + _peak(GRID, 403.8, 700.0) + _peak(GRID, 410.0, 300.0)
        _, area = pick_apex(GRID, y, None)
        assert area.sum() == pytest.approx(y.sum())

    def test_a_small_bump_on_a_peak_is_not_a_peak(self):
        y = _peak(GRID, 405.0, 1000.0)
        y[(np.abs(GRID - 405.33)).argmin()] += 2.0  # far under a quarter of its height
        assert pick_apex(GRID, y, None)[0].size == 1

    def test_the_smallest_area_needs_an_ion_scale(self):
        y = _peak(GRID, 405.0, 2.0)  # about 14 in area
        assert pick_apex(GRID, y, None)[0].size == 1  # no injection time, no rule
        assert pick_apex(GRID, y, 1000.0)[0].size == 1  # 14 ions
        assert pick_apex(GRID, y, 1.0)[0].size == 0  # 0.014 ions

    def test_nothing_in_nothing_out(self):
        assert pick_apex(np.empty(0), np.empty(0), None)[0].size == 0
        assert pick_apex(GRID, np.zeros(GRID.size), None)[0].size == 0


def _scans(n: int = 20, *, analyzer: str = "ITMS") -> list[dict]:
    y = _peak(GRID, 403.0, 1000.0) + _peak(GRID, 409.0, 300.0)
    return [
        {
            "scan": i + 1, "level": 1 if i % 5 == 0 else 2, "rt": float(i + 1), "iit": 20.0,
            "mz": GRID, "intensity": y, "analyzer": analyzer,
            "filter_string": "ITMS + p NSI r Full ms [400.0000-420.0000]",
            "isolation_lower": None if i % 5 == 0 else 700.0,
            "isolation_upper": None if i % 5 == 0 else 708.0,
            "precursor_mz": None if i % 5 == 0 else 704.0,
        }
        for i in range(n)
    ]


@pytest.fixture
def profile(tmp_path) -> Path:
    bundle = write_bundle(tmp_path, "run.profile", _scans(), profile=True, row_group_size=4096)
    return write_bundle_manifest(bundle, run="run", sha256="abc")


class TestCentroidBundle:
    def test_a_complete_bundle_named_for_what_made_it(self, profile, tmp_path):
        out = centroid_bundle(profile, "apex")
        key = peak_list_key(ALGORITHMS["apex"])
        assert out == tmp_path / f"run.{key}"
        manifest = read_bundle_manifest(out)
        assert manifest["kind"] == "thermo_acquisition"
        assert manifest["source_sha256"] == "abc"
        assert manifest["derived_from"]["name"] == key
        assert manifest["derived_from"]["algorithm"] == "apex"
        assert manifest["parameters"]["partial"] is False
        info = open_acquisition(out)
        assert not info.holds_profile and info.peak_list == key
        assert spectrum(out, 2).column("mz").to_numpy() == pytest.approx([403.0, 409.0], abs=1e-3)
        scans = pq.read_table(out / "scan_metadata.parquet")
        assert set(scans.column("spectrum_type").to_pylist()) == {"centroid"}
        assert set(scans.column("centroider").to_pylist()) == {"apex"}
        assert set(scans.column("peak_count").to_pylist()) == {2}
        assert manifest["run_metadata"]["levels"][1]["n_peaks"] == 32

    def test_it_is_a_peak_list_every_consumer_takes(self, profile, tmp_path):
        out = centroid_bundle(profile, "apex")
        assert peaks_dataset(out).count_rows() == 40
        summary = write_dia([out], tmp_path / "out.dia")
        assert summary.path.is_file()

    def test_a_scan_acquired_as_centroids_passes_through(self, tmp_path):
        scans = _scans(10)
        stored = {"mz": np.array([401.0, 402.0, 403.0]), "intensity": np.array([5.0, 6.0, 7.0])}
        scans[3] = {**scans[3], **stored}
        bundle = write_bundle(tmp_path, "run.profile", scans, profile=True, row_group_size=4096)
        table = pq.read_table(bundle / "scan_metadata.parquet")
        types = ["centroid" if n == 4 else "profile" for n in table.column("scan").to_pylist()]
        import pyarrow as pa

        at = table.schema.get_field_index("spectrum_type")
        table = table.set_column(at, table.schema.field(at), pa.array(types, pa.string()))
        pq.write_table(table, bundle / "scan_metadata.parquet")
        write_bundle_manifest(bundle, run="run", sha256="abc")
        out = centroid_bundle(bundle, "apex")
        assert spectrum(out, 4).column("mz").to_pylist() == [401.0, 402.0, 403.0]
        made = pq.read_table(out / "scan_metadata.parquet").column("centroider").to_pylist()
        assert made[3] == "instrument" and made[0] == "apex"

    def test_a_partial_list_says_so_and_cannot_be_searched(self, profile, tmp_path):
        out = centroid_bundle(profile, "apex", stride=4, output_dir=tmp_path / "some")
        manifest = read_bundle_manifest(out)
        assert manifest["parameters"]["partial"] is True and manifest["parameters"]["n_scans_picked"] == 5
        counts = pq.read_table(out / "scan_metadata.parquet").column("peak_count").to_pylist()
        assert sum(c > 0 for c in counts) == 5
        with pytest.raises(ValueError, match="partial peak list"):
            write_dia([out], tmp_path / "out.dia")

    def test_chosen_scans_are_picked_and_the_rest_are_empty(self, profile, tmp_path):
        out = centroid_bundle(profile, "apex", scans=[2, 7], output_dir=tmp_path / "two")
        assert spectrum(out, 2).num_rows == 2 and spectrum(out, 7).num_rows == 2
        assert spectrum(out, 3).num_rows == 0

    def test_workers_make_the_same_list(self, profile, tmp_path):
        one = centroid_bundle(profile, "apex", output_dir=tmp_path / "one", workers=1)
        two = centroid_bundle(profile, "apex", output_dir=tmp_path / "two", workers=2)
        assert peaks_dataset(one).to_table().equals(peaks_dataset(two).to_table())

    def test_an_existing_list_is_refused_unless_forced(self, profile):
        out = centroid_bundle(profile, "apex")
        with pytest.raises(FileExistsError, match="already there"):
            centroid_bundle(profile, "apex")
        (out / "stale").write_text("x")
        centroid_bundle(profile, "apex", force=True)
        assert not (out / "stale").exists()

    def test_it_takes_only_a_profile_bundle(self, tmp_path):
        peaks = write_bundle_manifest(write_bundle(tmp_path, "run", _scans(5), row_group_size=4096))
        with pytest.raises(ValueError, match="holds peaks"):
            centroid_bundle(peaks, "apex")

    def test_an_algorithm_is_named_or_refused(self, profile):
        with pytest.raises(ValueError, match="unknown algorithm"):
            centroid_bundle(profile, "nonesuch")

    def test_other_analyzers_profile_scans_are_refused_not_guessed_at(self, tmp_path):
        bundle = write_bundle(tmp_path, "run.profile", _scans(5, analyzer="FTMS"), profile=True, row_group_size=4096)
        write_bundle_manifest(bundle, run="run")
        with pytest.raises(ValueError, match="FTMS"):
            centroid_bundle(bundle, "apex")

    def test_a_failure_leaves_nothing(self, profile, tmp_path, monkeypatch):
        def broken(*_, **__):
            raise RuntimeError("no")

        monkeypatch.setitem(ALGORITHMS, "apex", ALGORITHMS["apex"].__class__("apex", 1, broken))
        with pytest.raises(RuntimeError):
            centroid_bundle(profile, "apex", output_dir=tmp_path / "out")
        assert not (tmp_path / "out").exists()


class TestThroughConvert:
    @pytest.fixture
    def src(self, monkeypatch, fake_centroider, tmp_path) -> Path:
        from _thermo_fakes import FakeRaw, IT_MS1, scan

        from constellation.massspec.readers.thermo import _read

        raw = FakeRaw({
            i: scan(IT_MS1, rt_min=i / 60, acquired="profile", mz=GRID, intensity=_peak(GRID, 403.0 + i, 500.0))
            for i in range(1, 6)
        })
        monkeypatch.setattr(_read, "_open_raw_file", lambda path: _read._RawFileHandle(raw))
        path = tmp_path / "run.raw"
        path.write_bytes(b"x")
        return path

    def test_an_own_algorithm_makes_the_runs_peak_list(self, src, tmp_path):
        from constellation.massspec.readers.thermo import convert

        m = convert(src, tmp_path / "run", centroid_algo="apex")
        assert m.kind == "thermo_acquisition" and m.parameters["centroider"] == "apex"
        assert (tmp_path / "run.profile" / "manifest.json").is_file()
        assert spectrum(tmp_path / "run", 1).column("mz").to_numpy() == pytest.approx([404.0], abs=1e-3)

    def test_without_the_profile_it_is_removed_once_the_list_is_made(self, src, tmp_path):
        from constellation.massspec.readers.thermo import convert

        convert(src, tmp_path / "run", centroid_algo="apex", keep_profile=False)
        assert not (tmp_path / "run.profile").exists()
        assert peaks_dataset(tmp_path / "run").count_rows() == 5

    def test_a_centroid_run_has_nothing_for_it_to_do(self, monkeypatch, tmp_path):
        from _thermo_fakes import long_raw

        from constellation.massspec.readers.thermo import _read, convert

        monkeypatch.setattr(_read, "_open_raw_file", lambda path: _read._RawFileHandle(long_raw(10)))
        src = tmp_path / "c.raw"
        src.write_bytes(b"x")
        m = convert(src, tmp_path / "c", centroid_algo="apex")
        assert m.parameters["centroider"] == "thermo"  # the asked-for default never ran
        assert peaks_dataset(tmp_path / "c").count_rows() == 30

    def test_an_unknown_name_is_refused(self, src, tmp_path):
        from constellation.massspec.readers.thermo import convert

        with pytest.raises(ValueError, match="unknown centroid_algo"):
            convert(src, tmp_path / "run", centroid_algo="nonesuch")

    def test_the_commands_names_are_the_registrys(self):
        from constellation.massspec.cli import convert as command

        assert set(command.OWN_ALGOS) == set(ALGORITHMS)


def test_the_browser_finds_a_derived_list_beside_its_source(profile, tmp_path) -> None:
    from constellation.massspec.bundle import related_bundles

    out = centroid_bundle(profile, "apex")
    assert [r["bundle"] for r in related_bundles(profile)] == [out]
    assert [r["name"] for r in related_bundles(out)] == ["profile"]


def test_a_manifest_with_no_hash_still_finds_its_source(tmp_path) -> None:
    from constellation.massspec.bundle import related_bundles

    bundle = write_bundle(tmp_path, "run.profile", _scans(5), profile=True, row_group_size=4096)
    write_bundle_manifest(bundle, run="run", sha256="placeholder")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["source_sha256"] = None
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    out = centroid_bundle(bundle, "apex")
    assert [r["name"] for r in related_bundles(out)] == ["profile"]


def test_with_no_hash_a_run_s_bundles_find_each_other_through_the_profile(tmp_path) -> None:
    """A derived list names the profile, and the profile names the conversion's
    peak list, so opening either list must find the other."""
    from constellation.massspec.bundle import related_bundles

    def without_hash(bundle: Path, **parameters) -> None:
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["source_sha256"] = None
        manifest["parameters"].update(parameters)
        (bundle / "manifest.json").write_text(json.dumps(manifest))

    source = write_bundle(tmp_path, "run.profile", _scans(5), profile=True, row_group_size=4096)
    write_bundle_manifest(source, run="run", sha256="placeholder")
    without_hash(source, peak_bundle="run")
    peaks = write_bundle(tmp_path, "run", _scans(5))
    write_bundle_manifest(peaks, sha256="placeholder")
    without_hash(peaks, profile_bundle="run.profile", centroider="thermo")
    derived = centroid_bundle(source, "apex")

    assert {r["bundle"] for r in related_bundles(peaks)} == {source, derived}
    assert {r["bundle"] for r in related_bundles(derived)} == {source, peaks}
    assert {r["bundle"] for r in related_bundles(source)} == {peaks, derived}
