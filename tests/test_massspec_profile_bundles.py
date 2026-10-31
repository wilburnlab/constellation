"""Profile bundles and peak lists: what a conversion writes, and what can read it.

A run's profile points are its data and live in a bundle of their own;
"peaks" always means centroids. These hold the conversion's table of outcomes
(profile or centroid run, by what was asked), and that neither kind of bundle
can be taken for the other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from _massspec_fixtures import dia_scans, write_bundle, write_bundle_manifest
from _thermo_fakes import FakeRaw, long_raw, mixed_raw

from constellation.massspec.bundle import (
    BUNDLE_SCHEMA_VERSION,
    OutdatedBundleError,
    open_peaks,
    peak_paths,
    peaks_dataset,
    point_paths,
    points_dataset,
    read_bundle_manifest,
)
from constellation.massspec.quant.peak_index import build_peak_index, find_peak_index
from constellation.massspec.readers.thermo import (
    DEFAULT_CENTROID_ALGO,
    NO_CENTROIDING,
    EmptyLevelError,
    convert,
    convert_batch,
    read_manifest_dir,
)
from constellation.massspec.scans import open_acquisition, spectrum


@pytest.fixture
def converting(monkeypatch, fake_centroider, tmp_path):
    """A stand-in .raw of a given make, converted without the DLLs."""
    from constellation.massspec.readers.thermo import _read

    def make(raw: FakeRaw) -> Path:
        monkeypatch.setattr(_read, "_open_raw_file", lambda path: _read._RawFileHandle(raw))
        src = tmp_path / "run.raw"
        src.write_bytes(b"not really a raw file")
        return src

    return make


class TestWhatAConversionWrites:
    def test_a_profile_run_gets_both_by_default(self, converting, tmp_path):
        src = converting(mixed_raw())
        m = convert(src, tmp_path / "run")
        assert m.kind == "thermo_acquisition"
        assert m.parameters["profile_bundle"] == "run.profile"
        assert peaks_dataset(tmp_path / "run").count_rows() == 7
        assert points_dataset(tmp_path / "run.profile").count_rows() == 12
        profile = read_manifest_dir(tmp_path / "run.profile")
        assert profile.kind == "thermo_profile"
        assert profile.parameters["peak_bundle"] == "run"
        assert profile.outputs["profile"] == "profile"
        assert profile.source_sha256 == m.source_sha256
        # The bundles say what they hold, each its own way per scan.
        peaks = pq.read_table(tmp_path / "run" / "scan_metadata.parquet")
        points = pq.read_table(tmp_path / "run.profile" / "scan_metadata.parquet")
        assert peaks.column("peak_count").to_pylist() == [2, 3, 2, 0]
        assert points.column("peak_count").to_pylist() == [3, 3, 4, 2]
        assert points.column("spectrum_type").to_pylist() == [
            "profile", "centroid", "profile", "profile",
        ]

    def test_the_peak_list_is_the_same_with_or_without_the_profile(self, converting, tmp_path):
        src = converting(mixed_raw())
        convert(src, tmp_path / "both" / "run")
        convert(src, tmp_path / "alone" / "run", keep_profile=False)
        assert not (tmp_path / "alone" / "run.profile").exists()
        assert peaks_dataset(tmp_path / "both" / "run").to_table().equals(
            peaks_dataset(tmp_path / "alone" / "run").to_table()
        )
        assert pq.read_table(tmp_path / "both" / "run" / "scan_metadata.parquet").equals(
            pq.read_table(tmp_path / "alone" / "run" / "scan_metadata.parquet")
        )

    def test_no_centroiding_keeps_the_profile_alone(self, converting, tmp_path):
        src = converting(mixed_raw())
        m = convert(src, tmp_path / "run", centroid_algo=NO_CENTROIDING)
        assert m.kind == "thermo_profile" and m.parameters["peak_bundle"] is None
        assert not (tmp_path / "run").exists()
        # A mixed run's centroid-acquired scan keeps what it stored.
        points = pq.read_table(tmp_path / "run.profile" / "scan_metadata.parquet")
        assert points.column("centroider").to_pylist() == [None, "instrument", None, None]
        assert spectrum(tmp_path / "run.profile", 2).column("mz").to_pylist() == [201.0, 305.5, 410.2]

    def test_keeping_nothing_is_refused_before_anything_is_written(self, converting, tmp_path):
        src = converting(mixed_raw())
        with pytest.raises(ValueError, match="would write nothing"):
            convert(src, tmp_path / "run", keep_profile=False, centroid_algo=NO_CENTROIDING)
        assert not (tmp_path / "run").exists() and not (tmp_path / "run.profile").exists()

    @pytest.mark.parametrize(
        "asked",
        [
            {},
            {"keep_profile": False},
            {"centroid_algo": NO_CENTROIDING},
            {"keep_profile": False, "centroid_algo": NO_CENTROIDING},
        ],
    )
    def test_a_centroid_run_just_comes_in_as_peaks(self, converting, tmp_path, asked):
        """Neither flag has anything to do: there is no profile to keep or centroid."""
        src = converting(long_raw(20))
        m = convert(src, tmp_path / "run", **asked)
        assert m.kind == "thermo_acquisition"
        assert m.parameters["profile_bundle"] is None
        assert not (tmp_path / "run.profile").exists()
        assert peaks_dataset(tmp_path / "run").count_rows() == 60

    def test_existing_bundles_are_refused_and_force_replaces_only_the_runs_own(
        self, converting, tmp_path
    ):
        src = converting(mixed_raw())
        convert(src, tmp_path / "run")
        with pytest.raises(FileExistsError):
            convert(src, tmp_path / "run")
        derived = tmp_path / "run.apex-1a2b3c4d"
        derived.mkdir()
        (derived / "manifest.json").write_text("{}")
        (tmp_path / "run" / "mz_index").mkdir()
        (tmp_path / "run.profile" / "mz_axes").mkdir()
        convert(src, tmp_path / "run", force=True)
        assert not (tmp_path / "run" / "mz_index").exists()
        assert not (tmp_path / "run.profile" / "mz_axes").exists()
        assert (derived / "manifest.json").read_text() == "{}"

    def test_a_failed_conversion_leaves_nothing(self, converting, tmp_path):
        raw = mixed_raw()
        src = converting(raw)
        raw.scans[2]["mz"] = raw.scans[2]["mz"][:0]
        raw.scans[2]["intensity"] = raw.scans[2]["intensity"][:0]
        with pytest.raises(EmptyLevelError):
            convert(src, tmp_path / "run")
        assert list((tmp_path / "run").iterdir()) == []
        assert not (tmp_path / "run.profile").exists()

    def test_the_two_are_encoded_differently(self, converting, tmp_path):
        """Byte-stream-split for what is new; the peak list exactly as it was."""
        src = converting(mixed_raw())
        convert(src, tmp_path / "run")

        def encodings(path: Path) -> set[str]:
            group = pq.ParquetFile(path).metadata.row_group(0)
            names = {group.column(i).path_in_schema: group.column(i).encodings for i in range(group.num_columns)}
            return set(names["mz"])

        assert "BYTE_STREAM_SPLIT" in encodings(point_paths(tmp_path / "run.profile")[0])
        assert "BYTE_STREAM_SPLIT" not in encodings(peak_paths(tmp_path / "run")[0])

    def test_runs_without_injection_times_are_counted(self, converting, tmp_path, caplog):
        src = converting(mixed_raw())
        with caplog.at_level("WARNING"):
            m = convert(src, tmp_path / "run")
        assert m.run_metadata["n_scans_without_injection_time"] == 3
        assert "3 of 4 scans have no injection time" in caplog.text


class TestABatch:
    def test_both_bundles_per_file_and_a_second_pass_skips(self, tmp_path, monkeypatch, fake_centroider):
        from constellation.massspec.readers.thermo import _read

        monkeypatch.setattr(
            _read, "_open_raw_file", lambda path: _read._RawFileHandle(mixed_raw())
        )
        src = tmp_path / "a.raw"
        src.write_bytes(b"x")
        out = tmp_path / "out"
        assert [r.status for r in convert_batch([src], out)] == ["ok"]
        assert (out / "a" / "manifest.json").is_file()
        assert (out / "a.profile" / "manifest.json").is_file()
        (again,) = convert_batch([src], out)
        assert again.status == "skipped" and again.detail == "already converted: a"

    def test_no_centroiding_is_judged_by_its_own_bundle(self, tmp_path, monkeypatch, fake_centroider):
        from constellation.massspec.readers.thermo import _read

        monkeypatch.setattr(
            _read, "_open_raw_file", lambda path: _read._RawFileHandle(mixed_raw())
        )
        src = tmp_path / "a.raw"
        src.write_bytes(b"x")
        out = tmp_path / "out"
        assert [r.status for r in convert_batch([src], out, centroid_algo=NO_CENTROIDING)] == ["ok"]
        assert not (out / "a").exists()
        (again,) = convert_batch([src], out, centroid_algo=NO_CENTROIDING)
        assert again.detail == "already converted: a.profile"


class TestNeitherIsTakenForTheOther:
    @pytest.fixture
    def profile(self, tmp_path) -> Path:
        return write_bundle_manifest(write_bundle(tmp_path, "run.profile", dia_scans(), profile=True))

    def test_a_profile_bundle_holds_no_peaks(self, profile):
        for read in (peak_paths, peaks_dataset, open_peaks):
            with pytest.raises(ValueError, match="profile bundle.*massspec centroid"):
                read(profile)

    def test_its_points_read_for_browsing_and_indexing(self, profile, tmp_path):
        assert points_dataset(profile).count_rows() > 0
        assert open_acquisition(profile).holds_profile
        build_peak_index(profile, profile / "mz_index")
        assert find_peak_index(profile) is not None

    def test_a_peak_bundle_reads_either_way(self, tmp_path):
        bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans()))
        assert not open_acquisition(bundle).holds_profile
        assert points_dataset(bundle).to_table().equals(peaks_dataset(bundle).to_table())

    def test_a_search_refuses_a_profile_bundle(self, profile):
        from constellation.massspec.workflows.encyclopedia import WorkflowError, _bundle_source

        with pytest.raises(WorkflowError, match="profile bundle"):
            _bundle_source(profile)


class TestOlderFormats:
    @pytest.fixture
    def old(self, tmp_path) -> Path:
        """What a v2 --profile conversion left: profile points under peaks/."""
        bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans()))
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["schema_version"] = 2
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        return bundle

    def test_every_way_in_refuses_it(self, old, tmp_path):
        from constellation.massspec.acquisitions import acquisition_record_from_bundle
        from constellation.massspec.io.encyclopedia import write_dia
        from constellation.massspec.workflows.encyclopedia import WorkflowError, _bundle_source

        for read in (
            read_bundle_manifest,
            peak_paths,
            point_paths,
            peaks_dataset,
            points_dataset,
            open_peaks,
            open_acquisition,
            acquisition_record_from_bundle,
            lambda b: write_dia([b], tmp_path / "out.dia"),
            lambda b: build_peak_index(b, tmp_path / "index"),
        ):
            with pytest.raises(OutdatedBundleError, match="massspec convert"):
                read(old)
        with pytest.raises(WorkflowError, match="massspec convert"):
            _bundle_source(old)

    def test_the_manifest_reader_refuses_it_too(self, old):
        with pytest.raises(ValueError, match="schema_version=2"):
            read_manifest_dir(old)

    def test_an_experiment_converts_again_rather_than_reuse_it(self, old, tmp_path):
        from constellation.massspec.workflows.encyclopedia import _is_conversion_of

        src = tmp_path / "run.raw"
        src.write_bytes(b"x")
        assert _is_conversion_of(old, src) is False

    def test_a_search_finds_no_bundle_in_it(self, old):
        from constellation.massspec.results import _bundle_manifest

        assert _bundle_manifest(old) is None

    def test_the_format_version_is_written(self, tmp_path):
        bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans()))
        assert read_bundle_manifest(bundle)["schema_version"] == BUNDLE_SCHEMA_VERSION


class TestTheCommand:
    def test_the_removed_flag_names_its_replacement(self, tmp_path, capsys):
        from constellation.massspec.cli.convert import _cmd_massspec_convert

        class Args:
            input = tmp_path / "run.raw"
            profile = True

        assert _cmd_massspec_convert(Args()) == 2
        err = capsys.readouterr().err
        assert "--profile was removed" in err and "--centroid-algo none" in err

    def test_the_help_says_what_a_profile_run_gets_by_default(self):
        import argparse

        from constellation.massspec.cli.convert import DEFAULT_ALGO, _build_convert_parser

        subs = argparse.ArgumentParser().add_subparsers()
        _build_convert_parser(subs)
        text = " ".join(subs.choices["convert"].format_help().split())
        assert f"holds peaks made from them by --centroid-algo (default: {DEFAULT_ALGO})" in text
        assert f"How to make peaks from profile scans. Default: {DEFAULT_ALGO}." in text

    def test_the_commands_names_are_the_readers(self):
        from constellation.massspec.cli import convert as command

        assert command.DEFAULT_ALGO == DEFAULT_CENTROID_ALGO
        assert command.NO_ALGO == NO_CENTROIDING
