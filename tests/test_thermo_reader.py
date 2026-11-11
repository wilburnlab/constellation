"""Tests for ``constellation.massspec.readers.thermo``.

Three tiers:

1. **Pure-Python unit tests** (no pythonnet / DLL dependency) — exercise
   the filter-string parser, the declarative trailer table + coercion
   helpers, the manifest round-trip and the schema registry.

2. **A stand-in raw file** (``FakeRaw``, no DLLs) — drives the range
   reader (every way a scan's peaks are made), the peak writer, and
   ``convert`` / ``ThermoReader.read`` end to end. Thermo's centroider is
   replaced by a stand-in (``fake_centroider``).

3. **Real-`.raw` smoke test** — gated on
   ``is_thermo_available()`` AND ``$CONSTELLATION_TEST_THERMO_RAW``
   pointing at a real fixture file. Skipped by default.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.massspec.readers.thermo._filter import parse_filter_string
from constellation.massspec.readers.thermo._trailer import (
    _TRAILER_FIELDS,
    TRAILER_KEY_CHARGE,
    TRAILER_KEY_IIT,
    TRAILER_KEY_MASTER_SCAN,
    TRAILER_KEY_MONOISOTOPIC,
    promote_trailer,
    resolve_trailer_field,
    safe_bool,
    safe_float,
    safe_int,
    safe_str,
    trailer_to_dict,
    try_number,
)
from constellation.massspec.readers.thermo._netruntime import is_thermo_available
from constellation.massspec.readers.thermo.manifest import (
    BUNDLE_KIND,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ThermoAcquisitionManifest,
    read_manifest,
    read_manifest_dir,
    write_manifest,
)
from _thermo_fakes import MS1, MS2, FakeEvent, FakeRaw, mixed_raw, scan
from constellation.massspec.bundle import peaks_dataset
from constellation.massspec.schemas import (
    ACQUISITION_METADATA_TABLE,
    MS_PEAK_TABLE,
    SCAN_METADATA_TABLE,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Pure-Python tier
# ──────────────────────────────────────────────────────────────────────


class TestFilterStringParser:
    def test_empty_input_returns_empty_dict(self):
        assert parse_filter_string("") == {}

    def test_ms1_full_filter(self):
        out = parse_filter_string("FTMS + p NSI Full ms [400.0-1600.0]")
        assert out["analyzer"] == "FTMS"
        assert out["polarity"] == "+"
        assert out["point_type"] == "P"
        assert out["ionization"] == "NSI"
        assert out["ms_level"] == 1
        assert out["scan_window"] == (400.0, 1600.0)
        assert "activations" not in out

    def test_ms2_hcd(self):
        out = parse_filter_string(
            "FTMS + c NSI d Full ms2 458.7660@hcd28.00 [110.0-1000.0]"
        )
        assert out["ms_level"] == 2
        assert out["point_type"] == "C"
        assert out["activations"] == [(458.7660, "hcd", 28.0)]

    def test_cid_vs_hcd_same_energy_distinguished_by_token(self):
        # ProteomeTools acquires both CID35 (ion trap) and HCD35 (Orbitrap) at
        # identical NCE 35 — collision_energy alone cannot separate them; the
        # activation token + analyzer are required.
        cid = parse_filter_string(
            "ITMS + c NSI r d Full ms2 530.7766@cid35.00 [141.0-1072.0]"
        )
        hcd = parse_filter_string(
            "FTMS + c NSI d Full ms2 530.7766@hcd35.00 [100.0-1072.0]"
        )
        assert cid["analyzer"] == "ITMS" and cid["activations"][0][1] == "cid"
        assert hcd["analyzer"] == "FTMS" and hcd["activations"][0][1] == "hcd"
        # identical energy — proof that energy is not a discriminator
        assert cid["activations"][0][2] == hcd["activations"][0][2] == 35.0

    def test_ethcd_multi_activation_shares_isolation_mz(self):
        out = parse_filter_string(
            "FTMS + c NSI d Full ms2 458.7660@etd50.00@hcd25.00 [110.0-1000.0]"
        )
        assert len(out["activations"]) == 2
        assert out["activations"][0] == (458.766, "etd", 50.0)
        assert out["activations"][1] == (458.766, "hcd", 25.0)

    def test_faims_cv(self):
        out = parse_filter_string("FTMS + p NSI cv=-45.00 Full ms [400-1600]")
        assert out["faims_cv"] == -45.0

    def test_negative_polarity(self):
        out = parse_filter_string("FTMS - p NSI Full ms [400-1600]")
        assert out["polarity"] == "-"

    def test_malformed_input_no_raise(self):
        # Random text shouldn't crash — sparser dict is acceptable.
        out = parse_filter_string("not a real filter string blah")
        assert isinstance(out, dict)


class TestCoercion:
    def test_safe_float(self):
        assert safe_float(None) is None
        assert safe_float("1.5") == 1.5
        assert safe_float(3) == 3.0
        assert safe_float("not a number") is None

    def test_safe_int(self):
        assert safe_int(None) is None
        assert safe_int("42") == 42
        assert safe_int("3.14") is None  # int('3.14') raises in CPython
        assert safe_int(7) == 7

    def test_safe_bool_recognises_on_off_yes_no(self):
        assert safe_bool("On") is True
        assert safe_bool("off") is False
        assert safe_bool("Yes") is True
        assert safe_bool("No") is False
        assert safe_bool(True) is True
        assert safe_bool(False) is False
        assert safe_bool("") is None
        assert safe_bool("unknown") is None
        assert safe_bool(None) is None

    def test_safe_str_strips_and_collapses_empty(self):
        assert safe_str("  hello  ") == "hello"
        assert safe_str("") is None
        assert safe_str(None) is None

    def test_try_number_int_float_passthrough(self):
        assert try_number("") is None
        assert try_number("42") == 42
        assert try_number("3.14") == 3.14
        assert try_number("not-a-number") == "not-a-number"
        assert try_number(None) is None
        assert try_number(7) == 7


class TestTrailerPromotion:
    def test_field_count_is_at_least_eighty(self):
        # ~80 typed columns per the docstring; rounded up.
        assert len(_TRAILER_FIELDS) >= 80

    def test_promote_with_typical_orbitrap_trailer(self):
        trailer = {
            "Scan Description:": "  HCD on m/z 458.77  ",
            "AGC:": "On",
            "Micro Scan Count:": 1,
            "HCD Energy:": "28.0",
            "FT Conversion A:": "1.23",
            "AGC Target:": 500_000,
            "Funnel RF Level:": "60",
            "FAIMS Attached:": "Yes",
            "FAIMS Voltage On:": "Yes",
            "unrelated_key:": "ignored",
        }
        promoted = promote_trailer(trailer)
        assert promoted["scan_description"] == "HCD on m/z 458.77"
        assert promoted["agc_enabled"] is True
        assert promoted["microscan_count"] == 1
        assert promoted["hcd_energy"] == 28.0
        assert promoted["ft_conversion_a"] == 1.23
        assert promoted["agc_target"] == 500_000
        assert promoted["funnel_rf_level"] == 60.0
        assert promoted["faims_attached"] is True
        assert promoted["faims_voltage_on"] is True
        # Untyped key is not promoted (only available via trailer_extras).
        assert "unrelated_key" not in promoted
        # Absent typed fields default to None.
        assert promoted["lc_fwhm"] is None
        assert promoted["mild_trapping_enabled"] is None

    def test_firmware_variant_falls_through_to_legacy_key(self):
        # "Conversion Parameter A:" is the legacy firmware name;
        # "FT Conversion A:" is the newer one. Either should resolve.
        legacy = {"Conversion Parameter A:": "9.9"}
        modern = {"FT Conversion A:": "9.9"}
        assert promote_trailer(legacy)["ft_conversion_a"] == 9.9
        assert promote_trailer(modern)["ft_conversion_a"] == 9.9

    def test_resolve_trailer_field_returns_first_nonnull(self):
        from constellation.massspec.readers.thermo._trailer import TrailerField

        f = TrailerField(("Primary Key:", "Fallback Key:"), "test_col", safe_float)
        assert resolve_trailer_field({"Primary Key:": "1.0"}, f) == 1.0
        assert resolve_trailer_field({"Fallback Key:": "2.0"}, f) == 2.0
        assert (
            resolve_trailer_field({"Primary Key:": None, "Fallback Key:": "3.0"}, f)
            == 3.0
        )
        assert resolve_trailer_field({}, f) is None


# Fake CommonCore HeaderInformation surface — gives ``trailer_to_dict``
# something to walk without a .NET handle.
class _FakeTrailer:
    def __init__(self, pairs: list[tuple[str, Any]]):
        self.Labels = [k for k, _ in pairs]
        self.Values = [v for _, v in pairs]


class TestTrailerToDict:
    def test_none_returns_empty(self):
        assert trailer_to_dict(None) == {}

    def test_label_value_pairs_round_trip_with_type_coerce(self):
        ft = _FakeTrailer(
            [
                ("HCD Energy:", "28.0"),
                ("AGC:", "On"),
                ("Master Scan Number:", "0"),
                ("Empty Field:", ""),
            ]
        )
        d = trailer_to_dict(ft)
        assert d["HCD Energy:"] == 28.0
        assert d["AGC:"] == "On"
        assert d["Master Scan Number:"] == 0
        # Empty values collapse to None.
        assert d["Empty Field:"] is None


class TestSchemaRegistration:
    def test_thermo_schemas_registered(self):
        from constellation.core.io.schemas import get_schema

        assert get_schema("MSPeakTable") is MS_PEAK_TABLE
        assert get_schema("ScanMetadataTable") is SCAN_METADATA_TABLE
        assert get_schema("AcquisitionMetadataTable") is ACQUISITION_METADATA_TABLE

    def test_peak_table_dtypes_are_64bit_where_expected(self):
        # Sanity-check the dtype decisions captured in the plan.
        assert MS_PEAK_TABLE.field("scan").type == pa.int32()
        assert MS_PEAK_TABLE.field("level").type == pa.int8()
        assert MS_PEAK_TABLE.field("rt").type == pa.float64()
        assert MS_PEAK_TABLE.field("mz").type == pa.float64()
        assert MS_PEAK_TABLE.field("precursor_scan").type == pa.float64()
        assert MS_PEAK_TABLE.field("precursor_mz").type == pa.float64()
        assert MS_PEAK_TABLE.field("isolation_lower").type == pa.float64()
        assert MS_PEAK_TABLE.field("isolation_upper").type == pa.float64()

    def test_scan_metadata_table_has_trailer_extras_map(self):
        f = SCAN_METADATA_TABLE.field("trailer_extras")
        assert pa.types.is_map(f.type)

    def test_scan_metadata_has_analyzer_and_activation_type(self):
        # Schema v2: analyzer + activation_type promoted to per-scan columns.
        names = set(SCAN_METADATA_TABLE.names)
        assert {"analyzer", "activation_type"} <= names
        assert SCAN_METADATA_TABLE.field("analyzer").type == pa.string()
        assert SCAN_METADATA_TABLE.field("activation_type").type == pa.string()
        assert SCAN_METADATA_TABLE.metadata[b"schema_version"] == b"5"

    def test_acquisition_metadata_table_has_tune_data_map(self):
        f = ACQUISITION_METADATA_TABLE.field("tune_data")
        assert pa.types.is_map(f.type)


class TestManifestRoundTrip:
    def test_write_then_read(self, tmp_path: Path):
        path = tmp_path / MANIFEST_FILENAME
        m = write_manifest(
            path,
            source_file="/data/run42.raw",
            source_sha256="deadbeef" * 8,
            parameters={"rt_bin_width_s": 60.0},
            outputs={
                "peaks": "peaks",
                "scan_metadata": "scan_metadata.parquet",
                "acquisition_metadata": "acquisition_metadata.parquet",
            },
            run_metadata={"instrument_model": "Orbitrap Astral", "n_scans": 4242},
        )
        assert m.schema_version == MANIFEST_SCHEMA_VERSION
        assert m.kind == BUNDLE_KIND

        rt = read_manifest(path)
        assert isinstance(rt, ThermoAcquisitionManifest)
        assert rt.source_file == "/data/run42.raw"
        assert rt.source_sha256 == "deadbeef" * 8
        assert rt.parameters["rt_bin_width_s"] == 60.0
        assert rt.outputs["peaks"] == "peaks"
        assert rt.run_metadata["n_scans"] == 4242

        rt2 = read_manifest_dir(tmp_path)
        assert rt2 == rt

    def test_read_rejects_wrong_schema_version(self, tmp_path: Path):
        import json

        path = tmp_path / MANIFEST_FILENAME
        write_manifest(
            path,
            source_file="/data/run42.raw",
            source_sha256=None,
            parameters={},
            outputs={},
            run_metadata={},
        )
        raw = json.loads(path.read_text())
        raw["schema_version"] = 99
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="schema_version"):
            read_manifest(path)

    def test_read_rejects_wrong_kind(self, tmp_path: Path):
        import json

        path = tmp_path / MANIFEST_FILENAME
        write_manifest(
            path,
            source_file="/data/run42.raw",
            source_sha256=None,
            parameters={},
            outputs={},
            run_metadata={},
        )
        raw = json.loads(path.read_text())
        raw["kind"] = "demux"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="kind"):
            read_manifest(path)

    def test_read_dir_missing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="manifest.json"):
            read_manifest_dir(tmp_path)



# ──────────────────────────────────────────────────────────────────────
# 2. A stand-in raw file: the range reader, the peak writer, convert()
# ──────────────────────────────────────────────────────────────────────


def _read(raw, **options):
    from constellation.massspec.readers.thermo._scans import ReadOptions, iter_scan_blocks

    blocks = list(iter_scan_blocks(raw, raw.RunHeaderEx.FirstSpectrum,
                                   raw.RunHeaderEx.LastSpectrum, ReadOptions(**options),
                                   max_block_scans=2))
    return (
        pa.concat_tables([b.scans for b in blocks]),
        pa.concat_tables([b.peak_table() for b in blocks]),
    )


class TestRangeReader:
    def _raw(self):
        return mixed_raw()

    def test_how_each_scan_is_centroided(self, fake_centroider):
        scans, peaks = _read(self._raw())
        assert scans.column("spectrum_type").to_pylist() == ["centroid"] * 4
        assert scans.column("centroider").to_pylist() == [
            "instrument", "instrument", "thermo", "instrument",
        ]
        by_scan = {s: peaks.filter(pa.compute.equal(peaks["scan"], s)) for s in (1, 2, 3, 4)}
        assert by_scan[1].column("mz").to_pylist() == [400.1, 500.0]  # the stream
        assert by_scan[2].column("mz").to_pylist() == [201.0, 305.5, 410.2]
        assert by_scan[3].column("mz").to_pylist() == [450.1, 450.2]  # the centroider
        assert by_scan[4].num_rows == 0  # an empty stream is no peaks, not a reason to centroid
        assert scans.column("peak_count").to_pylist() == [2, 3, 2, 0]

    def test_stream_values_ride_along_and_others_are_null(self, fake_centroider):
        _, peaks = _read(self._raw())
        assert peaks.column("peak_resolution").to_pylist() == [6e4, 5e4, None, None, None, None, None]
        assert peaks.column("peak_noise").null_count == 5

    def test_stored_points_come_with_the_peaks(self, fake_centroider):
        from constellation.massspec.readers.thermo._scans import ReadOptions, iter_scan_blocks

        raw = self._raw()
        blocks = list(iter_scan_blocks(raw, 1, 4, ReadOptions(keep_profile=True), max_block_scans=2))
        scans = pa.concat_tables([b.scans for b in blocks])
        points = pa.concat_tables([b.point_scans() for b in blocks])
        # `scans` says what the peaks are; `point_scans` what the points are.
        assert scans.column("peak_count").to_pylist() == [2, 3, 2, 0]
        assert scans.column("centroider").to_pylist() == [
            "instrument", "instrument", "thermo", "instrument",
        ]
        assert points.column("spectrum_type").to_pylist() == [
            "profile", "centroid", "profile", "profile",
        ]
        assert points.column("centroider").to_pylist() == [None, "instrument", None, None]
        assert points.column("peak_count").to_pylist() == [3, 3, 4, 2]
        assert sum(b.point_table().num_rows for b in blocks) == 12
        assert sum(b.peak_table().num_rows for b in blocks) == 7

    def test_no_centroider_gives_the_points_alone(self, fake_centroider):
        from constellation.massspec.readers.thermo._scans import ReadOptions, iter_scan_blocks

        raw = self._raw()
        options = ReadOptions(keep_profile=True, centroider=None)
        blocks = list(iter_scan_blocks(raw, 1, 4, options))
        assert all(b.peaks is None for b in blocks)
        scans = pa.concat_tables([b.scans for b in blocks])
        assert scans.column("peak_count").to_pylist() == [3, 3, 4, 2]
        assert scans.column("centroider").to_pylist() == [None, "instrument", None, None]

    def test_one_load_gives_both_lists_of_an_ion_trap_scan(self, fake_centroider, monkeypatch):
        from constellation.massspec.readers.thermo import _scans

        raw = self._raw()
        loads: list[int] = []
        original = _scans._load_scan
        monkeypatch.setattr(_scans, "_load_scan", lambda r, n: loads.append(n) or original(r, n))
        segmented: list[int] = []
        read = raw.GetSegmentedScanFromScanNumber
        raw.GetSegmentedScanFromScanNumber = lambda n, s: segmented.append(n) or read(n, s)
        list(_scans.iter_scan_blocks(raw, 3, 3, _scans.ReadOptions(keep_profile=True)))
        assert loads == [3] and segmented == []

    def test_blank_trailer_is_null_and_its_display_is_kept(self, fake_centroider):
        """A blank trailer's zeros are not measurements (ledger #20)."""
        blank = [
            ("Micro Scan Count:", "0"), ("Ion Injection Time (ms):", "0.000"),
            ("Elapsed Scan Time (sec):", "0.000"), ("Master Scan Number:", "0"),
            ("Monoisotopic M/Z:", "0.0000"), ("Charge State:", "0"),
            ("MS2 Isolation Width:", "0.00"), ("MS2 Isolation Offset:", "0.00"),
            ("FAIMS Voltage On:", "No"), ("HCD Energy:", "0.0"),
        ]
        normal = [
            ("Micro Scan Count:", "1"), ("Ion Injection Time (ms):", "39.591"),
            ("Elapsed Scan Time (sec):", "0.038"), ("HCD Energy:", "24.0"),
        ]
        event = FakeEvent(mass=500.25, width=8.0036)
        raw = FakeRaw({
            1: scan(MS2, rt_min=0.1, mz=[300.0, 400.0], trailer=normal, event=event),
            2: scan(MS2, rt_min=0.2, mz=[300.0, 400.0], trailer=blank, event=event),
        })
        ok, gone = _read(raw)[0].to_pylist()
        assert ok["iit"] == pytest.approx(39.591) and ok["microscan_count"] == 1
        for column in ("iit", "microscan_count", "ms2_isolation_width", "ms2_isolation_offset",
                       "hcd_energy", "faims_voltage_on", "master_scan", "precursor_scan",
                       "precursor_charge", "monoisotopic_mz_override"):
            assert gone[column] is None, column
        # The spectrum and the scan event's values are intact...
        assert gone["peak_count"] == 2 and gone["precursor_mz"] == 500.25
        assert gone["isolation_upper"] - gone["isolation_lower"] == pytest.approx(8.0036)
        # ...and the file's own strings stay readable.
        assert ("Ion Injection Time (ms):", "0.000") in gone["trailer_extras"]

    def test_a_zero_injection_time_alone_is_not_a_blank_trailer(self, fake_centroider):
        raw = FakeRaw({1: scan(MS1, rt_min=0.1, mz=[400.0], trailer=[
            ("Micro Scan Count:", "1"), ("Ion Injection Time (ms):", "0.000"),
            ("Elapsed Scan Time (sec):", "0.021"),
        ])})
        row = _read(raw)[0].to_pylist()[0]
        assert row["iit"] == 0.0 and row["microscan_count"] == 1

    def test_scan_metadata(self, fake_centroider):
        scans, peaks = _read(self._raw())
        ms1, ms2 = scans.to_pylist()[:2]
        assert ms1["rt"] == pytest.approx(30.0)
        assert ms1["iit"] == 15.0
        assert ms1["agc_enabled"] is True
        assert ms1["level"] == 1 and ms1["analyzer"] == "FTMS"
        assert (ms1["scan_window_lower"], ms1["scan_window_upper"]) == (400.0, 1600.0)
        assert ms2["level"] == 2 and ms2["analyzer"] == "ITMS"
        assert ms2["activation_type"] == "hcd" and ms2["collision_energy"] == 30.0
        assert ms2["precursor_mz"] == 500.2512  # the trailer's monoisotopic m/z
        assert ms2["precursor_charge"] == 2
        assert ms2["precursor_scan"] == ms2["master_scan"] == 1.0
        # The scan event's width (2.0009), not the trailer's display (2.00),
        # centred on the isolation target.
        assert ms2["isolation_lower"] == 500.25 - 2.0009 / 2
        assert ms2["isolation_upper"] == 500.25 + 2.0009 / 2
        # trailer_extras holds what Thermo displays.
        assert ("MS2 Isolation Width:", "2.00") in ms2["trailer_extras"]
        # Per-scan values repeated on every peak of the scan.
        ms2_peaks = peaks.filter(pa.compute.equal(peaks["scan"], 2))
        assert set(ms2_peaks.column("precursor_mz").to_pylist()) == {500.2512}
        assert set(ms2_peaks.column("level").to_pylist()) == {2}

    def test_an_ms3_scan_isolates_its_precursor_and_records_its_own_activation(self, fake_centroider):
        """An MS3 scan's window was its MS2 fragment's (reaction 1) and its
        activation its MS2 parent's (review of 2026-11-09, A4): the sweep
        matches a scan to the queries whose precursor its window holds."""
        ms3 = "FTMS + p NSI Full ms3 600.3000@cid35.00 300.1500@hcd65.00 [100.0000-1000.0000]"
        raw = FakeRaw({
            1: scan(MS1, rt_min=0.1, mz=[400.0]),
            2: scan(ms3, rt_min=0.2, mz=[126.1, 127.1],
                    event=FakeEvent(mass=(600.30, 300.15), width=(0.7, 2.0), offset=(0.0, 0.0)),
                    trailer=[("Master Scan Number:", "1")]),
        })
        row = _read(raw)[0].to_pylist()[1]
        assert row["level"] == 3
        assert row["activation_type"] == "hcd" and row["collision_energy"] == 65.0
        assert (row["isolation_lower"], row["isolation_upper"]) == (600.30 - 0.35, 600.30 + 0.35)
        assert row["precursor_mz"] == 600.30

    def test_typed_trailer_values_not_display_strings(self, fake_centroider):
        """Thermo displays a trailer entry rounded to its header's precision;
        the typed value is the instrument's."""
        iit = float(np.float32(31.0431213))
        raw = FakeRaw({
            1: scan(MS1, rt_min=0.1, mz=[400.0],
                    trailer=[("Ion Injection Time (ms):", "31.043", iit)]),
            2: scan(MS2, rt_min=0.2, mz=[300.0], event=FakeEvent(mass=500.25, width=2.0009),
                    trailer=[("Ion Injection Time (ms):", "100.000", 100.0),
                             ("MS2 Isolation Width:", "2.00", 2.0009),
                             ("Monoisotopic M/Z:", "500.2512", 500.25123456)]),
        })
        ms1, ms2 = _read(raw)[0].to_pylist()
        assert ms1["iit"] == iit
        assert ("Ion Injection Time (ms):", "31.043") in ms1["trailer_extras"]
        assert ms2["ms2_isolation_width"] == 2.0009
        assert ms2["precursor_mz"] == ms2["monoisotopic_mz_override"] == 500.25123456
        # The header lists MS2-only entries, but an MS1 scan does not have them.
        assert ms1["ms2_isolation_width"] is None and ms1["monoisotopic_mz_override"] is None

    def test_without_a_trailer_header_the_display_strings_are_read(self, fake_centroider):
        raw = FakeRaw(
            {1: scan(MS1, rt_min=0.1, mz=[400.0],
                     trailer=[("Ion Injection Time (ms):", "31.043", 31.0431213)])},
            header=False,
        )
        assert _read(raw)[0].to_pylist()[0]["iit"] == pytest.approx(31.043, abs=1e-6)

    def test_no_trailer_extras_keeps_typed_columns(self, fake_centroider):
        scans, _ = _read(self._raw(), capture_trailer_extras=False)
        ms1 = scans.to_pylist()[0]
        assert ms1["trailer_extras"] is None
        assert ms1["agc_enabled"] is True and ms1["iit"] == 15.0

    def test_negative_value_in_unsigned_column_is_null(self, fake_centroider):
        raw = FakeRaw({1: scan(MS1, rt_min=0.1, mz=[400.0],
                                 trailer=[("Master Index:", "-1"), ("AGC PS Mode:", "-1")])})
        scans, _ = _read(raw)
        row = scans.to_pylist()[0]
        assert row["master_index"] is None  # uint32
        assert row["agc_ps_mode"] == -1  # signed since scan-metadata v4

    def test_peaks_sorted_within_each_scan(self, fake_centroider):
        raw = FakeRaw({
            1: scan(MS1, rt_min=0.1, mz=[400.0, 300.0, 500.0], intensity=[1, 2, 3]),
            2: scan(MS1, rt_min=0.2, mz=[100.0, 200.0], intensity=[4, 5]),
        })
        _, peaks = _read(raw)
        assert peaks.column("mz").to_pylist() == [300.0, 400.0, 500.0, 100.0, 200.0]
        assert peaks.column("intensity").to_pylist() == [2, 1, 3, 4, 5]

    def test_unreadable_scan_is_skipped(self, fake_centroider):
        raw = self._raw()
        raw.scans[2]["stats"] = None  # attribute access fails like a broken scan
        scans, _ = _read(raw)
        assert scans.column("scan").to_pylist() == [1, 3, 4]

    def test_unknown_centroider_is_refused(self):
        from constellation.massspec.readers.thermo._scans import ReadOptions

        with pytest.raises(ValueError, match="centroider"):
            ReadOptions(centroider="nonesuch")


class TestPeakWriter:
    def _table(self, rts, scans=None):
        from constellation.massspec.bundle import peak_rows

        scans_t = pa.table({
            "scan": pa.array(scans or list(range(1, len(rts) + 1)), pa.int32()),
            "level": pa.array([1] * len(rts), pa.int8()),
            "rt": pa.array(rts, pa.float64()),
        })
        counts = np.full(len(rts), 2)
        mz = np.tile([100.0, 200.0], len(rts))
        return peak_rows(scans_t, counts, mz, mz / 10)

    def test_row_group_per_rt_bin(self, tmp_path: Path):
        from constellation.massspec.bundle import PeakWriter

        path = tmp_path / "peaks.parquet"
        with PeakWriter(path, rt_bin_width_s=60.0) as w:
            w.write(self._table([30.0, 45.0]))
            w.write(self._table([70.0, 90.0], scans=[3, 4]))
        pf = pq.ParquetFile(path)
        assert pf.num_row_groups == 2
        assert pf.read().column("scan").to_pylist() == [1, 1, 2, 2, 3, 3, 4, 4]
        assert pf.schema_arrow.equals(MS_PEAK_TABLE)

    def test_bin_boundary_inside_one_write(self, tmp_path: Path):
        from constellation.massspec.bundle import PeakWriter

        path = tmp_path / "peaks.parquet"
        with PeakWriter(path, rt_bin_width_s=60.0) as w:
            w.write(self._table([30.0, 70.0, 130.0]))
        assert pq.ParquetFile(path).num_row_groups == 3

    def test_row_groups_capped(self, tmp_path: Path):
        from constellation.massspec.bundle import PeakWriter

        path = tmp_path / "peaks.parquet"
        with PeakWriter(path, rt_bin_width_s=600.0, max_row_group_rows=3) as w:
            w.write(self._table([1.0, 2.0, 3.0, 4.0]))  # 8 rows in one bin
        pf = pq.ParquetFile(path)
        assert [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)] == [3, 3, 2]

    def test_peak_rows_repeats_scan_values_and_masks_aux(self):
        from constellation.massspec.bundle import peak_rows

        scans_t = pa.table({
            "scan": pa.array([7, 8], pa.int32()),
            "level": pa.array([1, 2], pa.int8()),
            "rt": pa.array([1.0, 2.0]),
            "precursor_mz": pa.array([None, 500.5]),
        })
        aux = {"peak_noise": (np.array([1.0, 2.0, 0.0]), np.array([True, True, False]))}
        t = peak_rows(scans_t, np.array([2, 1]), np.array([1.0, 2.0, 3.0]), np.ones(3), aux)
        assert t.column("scan").to_pylist() == [7, 7, 8]
        assert t.column("precursor_mz").to_pylist() == [None, None, 500.5]
        assert t.column("peak_noise").to_pylist() == [1.0, 2.0, None]
        assert t.column("ion_mobility").null_count == 3


class TestConvert:
    @pytest.fixture
    def raw(self, monkeypatch, fake_centroider, tmp_path):
        from constellation.massspec.readers.thermo import _read

        raw = TestRangeReader()._raw()
        monkeypatch.setattr(_read, "_open_raw_file", lambda path: _read._RawFileHandle(raw))
        src = tmp_path / "run.raw"
        src.write_bytes(b"not really a raw file")
        return raw, src

    def test_bundle_and_level_summary(self, raw, tmp_path):
        from constellation.massspec.readers.thermo import convert, describe_levels

        _, src = raw
        m = convert(src, tmp_path / "run")
        assert {p.name for p in (tmp_path / "run").iterdir()} == {
            "manifest.json", "peaks", "precursor_major", "scan_metadata.parquet",
            "acquisition_metadata.parquet",
        }
        assert [p.name for p in (tmp_path / "run" / "peaks").iterdir()] == ["part-00000.parquet"]
        assert m.outputs["precursor_major"] == "precursor_major"
        assert m.schema_version == 3
        assert m.outputs["peaks"] == "peaks"
        assert m.parameters["peaks"] == "centroid"
        assert m.parameters["centroider"] == "thermo"
        assert m.source_sha256 == hashlib.sha256(src.read_bytes()).hexdigest()
        assert describe_levels(m.run_metadata["levels"]) == [
            "MS1: 3 scans, 4 peaks (2 centroided by the instrument, 1 centroided by "
            "Thermo's centroider)",
            "MS2: 1 scans, 3 peaks (1 centroided by the instrument)",
        ]
        acq = pq.read_table(tmp_path / "run" / "acquisition_metadata.parquet").to_pylist()[0]
        assert [h["label"] for h in acq["trailer_header"]][:2] == [
            "Ion Injection Time (ms):", "AGC:",
        ]
        assert acq["analyzers"] == ["FTMS", "ITMS"]
        assert peaks_dataset(tmp_path / "run").count_rows() == 7

    def test_run_header_sample_and_method_text(self, raw, tmp_path):
        from _thermo_fakes import Obj

        from constellation.massspec.readers.thermo import convert

        fake, src = raw
        fake.RunHeaderEx = Obj(
            FirstSpectrum=1, LastSpectrum=4, StartTime=0.5, EndTime=1.3,
            ExpectedRunTime=107.0, MaxIntegratedIntensity=9.5e8, SpectraCount=4,
        )
        fake.FileHeader = Obj(CreationDate=Obj(ToString=lambda fmt: "2026-10-07T12:15:40.9362541Z"))
        fake.SampleInformation = Obj(
            Vial="G:B3", InjectionVolume=1.0,
            InstrumentMethodFile="C:\\methods\\GPF.meth", RawFileName="D:\\run.raw",
        )
        fake.InstrumentMethodsCount = 2
        fake.GetAllInstrumentNamesFromInstrumentMethod = lambda: ["SiiXcalibur", "TNG-Thorium"]
        fake.GetInstrumentMethod = lambda i: ["LC method\r\n", "MS method\r\n"][i]

        m = convert(src, tmp_path / "run", compute_sha256=False)
        acq = pq.read_table(tmp_path / "run" / "acquisition_metadata.parquet").to_pylist()[0]
        assert acq["run_start_time"] == "2026-10-07T12:15:40.9362541Z"
        assert (acq["run_start_minutes"], acq["run_end_minutes"]) == (0.5, 1.3)
        assert acq["expected_run_minutes"] == 107.0 and acq["spectra_count"] == 4
        assert acq["sample_vial"] == "G:B3" and acq["sample_method_file"] == "C:\\methods\\GPF.meth"
        assert acq["method_name"] == "GPF.meth"
        assert acq["instrument_methods"] == [
            {"name": "SiiXcalibur", "text": "LC method\r\n"},
            {"name": "TNG-Thorium", "text": "MS method\r\n"},
        ]
        assert m.run_metadata["instrument_method_names"] == ["SiiXcalibur", "TNG-Thorium"]

        # CommonCore 5.0 on Linux gives no method text: nothing is recorded.
        # Nor is a sample field the file leaves empty.
        fake.GetInstrumentMethod = lambda i: None
        fake.SampleInformation.Vial = None
        convert(src, tmp_path / "old", compute_sha256=False)
        acq = pq.read_table(tmp_path / "old" / "acquisition_metadata.parquet").to_pylist()[0]
        assert acq["instrument_methods"] is None and acq["sample_vial"] is None
        assert acq["sample_raw_file_name"] == "D:\\run.raw"

    def test_empty_level_fails_and_leaves_nothing(self, raw, tmp_path):
        from constellation.massspec.readers.thermo import EmptyLevelError, convert

        fake, src = raw
        fake.scans[2]["mz"] = np.empty(0)
        fake.scans[2]["intensity"] = np.empty(0)
        out = tmp_path / "run"
        with pytest.raises(EmptyLevelError, match="MS2"):
            convert(src, out)
        assert list(out.iterdir()) == []
        m = convert(src, out, allow_empty_levels=True)
        assert m.run_metadata["levels"][1]["n_peaks"] == 0

    def test_a_file_whose_scans_cannot_be_read_fails(self, raw, tmp_path):
        from constellation.massspec.readers.thermo import (
            ThermoReader,
            UnreadableScansError,
            convert,
        )

        fake, src = raw

        def refuse(n):
            raise RuntimeError(f"scan {n} is damaged")

        fake.GetScanStatsForScanNumber = refuse
        out = tmp_path / "run"
        with pytest.raises(UnreadableScansError, match="none of the file's 4 scans"):
            convert(src, out, allow_empty_levels=True)
        assert list(out.iterdir()) == []
        with pytest.raises(UnreadableScansError):
            ThermoReader().read(src)

    def test_force_clears_what_an_earlier_conversion_left(self, raw, tmp_path):
        from constellation.massspec.readers.thermo import convert

        _, src = raw
        out = tmp_path / "run"
        convert(src, out)
        (out / "mz_index").mkdir()  # a peak index of the old peaks
        (out / "notes.txt").write_text("mine")
        with pytest.raises(FileExistsError):
            convert(src, out)
        convert(src, out, force=True)
        assert not (out / "mz_index").exists()
        assert (out / "notes.txt").read_text() == "mine"

    def test_in_memory_reader(self, raw, monkeypatch):
        from constellation.massspec.readers.thermo import ThermoReader

        _, src = raw
        result = ThermoReader().read(src)
        assert result.primary.num_rows == 7
        assert result.companions["scan_metadata"].num_rows == 4
        assert result.run_metadata["n_scans"] == 4
        assert "tune_data" not in result.run_metadata


# ──────────────────────────────────────────────────────────────────────
# 3. Real-`.raw` integration tier (skipped without an env var)
# ──────────────────────────────────────────────────────────────────────


_REAL_RAW_PATH_ENV = "CONSTELLATION_TEST_THERMO_RAW"


@pytest.mark.skipif(
    not (is_thermo_available() and os.environ.get(_REAL_RAW_PATH_ENV)),
    reason=(
        f"Thermo DLLs not installed or {_REAL_RAW_PATH_ENV} not set "
        f"(integration test requires both)."
    ),
)
class TestRealRawConvert:
    """Skipped unless ``$CONSTELLATION_TEST_THERMO_RAW`` points at a real file."""

    @pytest.fixture
    def raw_path(self) -> Path:
        return Path(os.environ[_REAL_RAW_PATH_ENV])

    def test_convert_writes_full_bundle(self, raw_path: Path, tmp_path: Path):
        from constellation.massspec.readers.thermo import convert

        bundle_dir = tmp_path / raw_path.stem
        m = convert(raw_path, bundle_dir, compute_sha256=False)
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "peaks").is_dir()
        assert (bundle_dir / "scan_metadata.parquet").exists()
        assert (bundle_dir / "acquisition_metadata.parquet").exists()
        assert m.run_metadata["n_scans"] > 0

        peaks = peaks_dataset(bundle_dir).to_table()
        assert peaks.schema.equals(MS_PEAK_TABLE)
        # Sanity: peaks table is non-empty and m/z values are sorted ascending
        # within at least the first scan.
        assert peaks.num_rows > 0


def test_find_dotnet_root_uses_interpreter_prefix_before_conda_prefix(
    tmp_path, monkeypatch
) -> None:
    """The runtime is found when the env is not activated.

    Calling ``constellation`` by full path (an sbatch script, a wrapper)
    leaves ``$CONDA_PREFIX`` unset or pointing at the base env, which has
    no .NET runtime. The interpreter's own prefix is the env that does.
    """
    from constellation.massspec.readers.thermo import _netruntime

    env_prefix = tmp_path / "envs" / "constellation"
    (env_prefix / "lib" / "dotnet").mkdir(parents=True)
    base_prefix = tmp_path / "base"
    base_prefix.mkdir()

    monkeypatch.delenv("DOTNET_ROOT", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(base_prefix))
    monkeypatch.setattr(_netruntime.shutil, "which", lambda name: None)
    monkeypatch.setattr(_netruntime.sys, "prefix", str(env_prefix))

    assert _netruntime._find_dotnet_root() == env_prefix / "lib" / "dotnet"


class TestReviewSecondPass:
    """Review 2026-11-10: A7 (several ranges), A11 (MS3 after EThcD), S5 (floor)."""

    def test_an_ms3_after_supplemental_activation_records_its_own(self, fake_centroider):
        """The MS3 scan took its MS2 reaction's supplemental activation (hcd
        15), counting activations instead of isolations (A11)."""
        ms3 = "ITMS + c NSI r d Full ms3 600.0000@etd20.00@hcd15.00 300.0000@hcd35.00 [100.0000-1000.0000]"
        raw = FakeRaw({
            1: scan(MS1, rt_min=0.1, mz=[400.0]),
            2: scan(ms3, rt_min=0.2, mz=[126.1, 127.1],
                    event=FakeEvent(mass=(600.0, 300.0), width=(0.7, 2.0), offset=(0.0, 0.0)),
                    trailer=[("Master Scan Number:", "1")]),
        })
        row = _read(raw)[0].to_pylist()[1]
        assert row["level"] == 3
        assert row["activation_type"] == "hcd" and row["collision_energy"] == 35.0

    def test_a_scan_of_several_ranges_keeps_each(self, fake_centroider):
        """A filter of several ranges parsed to none (A7)."""
        multi = "FTMS + p NSI Full ms [200.0000-700.0000, 800.0000-1500.0000]"
        raw = FakeRaw({1: scan(multi, rt_min=0.1, mz=[400.0, 900.0], intensity=[5.0, 3.0])})
        row = _read(raw)[0].to_pylist()[0]
        assert (row["scan_window_lower"], row["scan_window_upper"]) == (200.0, 1500.0)
        assert row["scan_windows"] == [{"lower": 200.0, "upper": 700.0}, {"lower": 800.0, "upper": 1500.0}]
        assert row["intensity_floor"] == 3.0  # the lowest stored intensity (S5)
        one = _read(FakeRaw({1: scan(MS1, rt_min=0.1, mz=[400.0])}))[0].to_pylist()[0]
        assert one["scan_windows"] is None
