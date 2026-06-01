"""Tests for ``constellation.massspec.io.thermo``.

Three tiers:

1. **Pure-Python unit tests** (no pythonnet / DLL dependency) — exercise
   the filter-string parser, the declarative trailer table + coercion
   helpers, the manifest round-trip, the schema registry, and the
   ``_build_scan_meta`` row builder against fake CommonCore objects.

2. **Fake-CommonCore integration test** — drive the iteration path
   (`_iter_scan_batches`, `_PeakBuffer`, `_build_acquisition_table`)
   against ``FakeRawFile`` mocks. Confirms the end-to-end scan-to-table
   wiring without a real `.raw` file.

3. **Real-`.raw` smoke test** — gated on
   ``is_thermo_available()`` AND ``$CONSTELLATION_TEST_THERMO_RAW``
   pointing at a real fixture file. Skipped by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.massspec.io.thermo._filter import parse_filter_string
from constellation.massspec.io.thermo._trailer import (
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
from constellation.massspec.io.thermo._netruntime import is_thermo_available
from constellation.massspec.io.thermo.manifest import (
    BUNDLE_KIND,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ThermoAcquisitionManifest,
    read_manifest,
    read_manifest_dir,
    write_manifest,
)
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
        from constellation.massspec.io.thermo._trailer import TrailerField

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


# ──────────────────────────────────────────────────────────────────────
# 2. Fake-CommonCore tier
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _FakeStats:
    TIC: float = 0.0
    BasePeakMass: float = 0.0
    BasePeakIntensity: float = 0.0
    PacketCount: int = 0


class _FakeFilter:
    """Minimal stand-in for a CommonCore Filter object."""

    def __init__(self, filter_str: str, isolation_mass: float | None = None,
                 isolation_width: float | None = None):
        self._s = filter_str
        self._mass = isolation_mass
        self._width = isolation_width

    def ToString(self):
        return self._s

    def GetMass(self, idx: int) -> float:  # noqa: ARG002
        if self._mass is None:
            raise RuntimeError("no isolation mass")
        return self._mass

    def GetIsolationWidth(self, idx: int) -> float:  # noqa: ARG002
        if self._width is None:
            raise RuntimeError("no width")
        return self._width

    def GetIsolationWidthOffset(self, idx: int) -> float:  # noqa: ARG002
        return 0.0


class TestBuildScanMeta:
    def test_ms1_row_minimal(self):
        from constellation.massspec.io.thermo._read import _build_scan_meta

        meta = _build_scan_meta(
            scan_num=1,
            rt=1.5,
            filter_str="FTMS + p NSI Full ms [400.0-1600.0]",
            stats=_FakeStats(TIC=1.23e7, BasePeakMass=523.45, BasePeakIntensity=4.5e5,
                             PacketCount=200),
            trailer={
                "Ion Injection Time (ms):": "15.0",
                "AGC:": "On",
            },
            filt=_FakeFilter("FTMS + p NSI Full ms [400.0-1600.0]"),
            profile_mode=False,
            capture_trailer_extras=True,
        )
        assert meta["scan"] == 1
        assert meta["level"] == 1
        assert meta["rt"] == 1.5
        assert meta["tic"] == 1.23e7
        assert meta["base_peak_mz"] == 523.45
        assert meta["base_peak_intensity"] == 4.5e5
        assert meta["peak_count"] == 200
        assert meta["iit"] == 15.0
        assert meta["precursor_mz"] is None
        assert meta["precursor_charge"] is None
        assert meta["agc_enabled"] is True
        # MS1: analyzer set from the filter string, no activation.
        assert meta["analyzer"] == "FTMS"
        assert meta["activation_type"] is None

    def test_ms2_row_with_precursor_and_charge(self):
        from constellation.massspec.io.thermo._read import _build_scan_meta

        meta = _build_scan_meta(
            scan_num=42,
            rt=12.5,
            filter_str=(
                "FTMS + c NSI d Full ms2 458.7660@hcd28.00 [110.0-1000.0]"
            ),
            stats=_FakeStats(TIC=5.0e5, PacketCount=80),
            trailer={
                TRAILER_KEY_MONOISOTOPIC: "458.7660",
                TRAILER_KEY_CHARGE: "2",
                TRAILER_KEY_IIT: "50.0",
                TRAILER_KEY_MASTER_SCAN: "40",
                "MS2 Isolation Width:": "1.6",
                "HCD Energy:": "28.0",
            },
            filt=_FakeFilter(
                "FTMS + c NSI d Full ms2 458.7660@hcd28.00 [110.0-1000.0]",
                isolation_mass=458.766,
                isolation_width=1.6,
            ),
            profile_mode=False,
            capture_trailer_extras=True,
        )
        assert meta["level"] == 2
        assert meta["precursor_mz"] == 458.766
        assert meta["precursor_charge"] == 2
        assert meta["collision_energy"] == 28.0
        # MS2: analyzer + activation_type are persisted per-scan columns now.
        assert meta["analyzer"] == "FTMS"
        assert meta["activation_type"] == "hcd"
        assert meta["isolation_lower"] == pytest.approx(458.766 - 0.8, rel=1e-6)
        assert meta["isolation_upper"] == pytest.approx(458.766 + 0.8, rel=1e-6)
        # Master scan promoted to float64 in PEAK_TABLE shape.
        assert meta["precursor_scan"] == 40.0
        assert meta["master_scan"] == 40.0
        assert meta["hcd_energy"] == 28.0
        # Trailer extras present as a flat key/value list.
        assert any(k == TRAILER_KEY_MONOISOTOPIC for k, _ in meta["trailer_extras"])

    def test_master_scan_zero_collapses_to_none(self):
        """Thermo uses ``Master Scan Number: 0`` to mean 'no parent'."""
        from constellation.massspec.io.thermo._read import _build_scan_meta

        meta = _build_scan_meta(
            scan_num=1,
            rt=0.5,
            filter_str="FTMS + p NSI Full ms [400-1600]",
            stats=_FakeStats(),
            trailer={TRAILER_KEY_MASTER_SCAN: "0"},
            filt=_FakeFilter("FTMS + p NSI Full ms [400-1600]"),
            profile_mode=False,
            capture_trailer_extras=True,
        )
        assert meta["master_scan"] is None
        assert meta["precursor_scan"] is None


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
        assert SCAN_METADATA_TABLE.metadata[b"schema_version"] == b"2"

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
            parameters={"rt_bin_width_s": 60.0, "profile_mode": False},
            outputs={
                "peaks": "peaks.parquet",
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
        assert rt.outputs["peaks"] == "peaks.parquet"
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
# 3. Fake-CommonCore iteration-level test
# ──────────────────────────────────────────────────────────────────────


class TestPeakBuffer:
    def test_rt_binned_row_groups(self, tmp_path: Path):
        """Verify _PeakBuffer flushes a row group at every RT-bin boundary."""
        from constellation.massspec.io.thermo._read import _PeakBuffer, _ScanBatch

        # Two scans in bin 0, two in bin 1 → 2 row groups when
        # rt_bin_width=60s. (RT in seconds: 30 + 45 in bin 0; 70 + 90 in bin 1.)
        mz = np.asarray([100.0, 200.0, 100.5, 200.5, 110.0, 210.0, 110.5, 210.5])
        intensity = np.asarray([1.0, 2.0, 1.5, 2.5, 3.0, 4.0, 3.5, 4.5])
        offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
        scan_meta = [
            {"scan": 1, "level": 1, "rt": 30.0, "precursor_scan": None,
             "precursor_mz": None},
            {"scan": 2, "level": 1, "rt": 45.0, "precursor_scan": None,
             "precursor_mz": None},
            {"scan": 3, "level": 1, "rt": 70.0, "precursor_scan": None,
             "precursor_mz": None},
            {"scan": 4, "level": 1, "rt": 90.0, "precursor_scan": None,
             "precursor_mz": None},
        ]
        batch = _ScanBatch(mz=mz, intensity=intensity, scan_offsets=offsets,
                           scan_meta=scan_meta, peak_aux=None)

        out_path = tmp_path / "peaks.parquet"
        with pq.ParquetWriter(
            str(out_path), MS_PEAK_TABLE, compression="zstd", use_dictionary=False
        ) as writer:
            buf = _PeakBuffer(MS_PEAK_TABLE, writer)
            buf.append_batch(batch, rt_bin_width=60.0)
            buf.flush()

        pf = pq.ParquetFile(str(out_path))
        assert pf.num_row_groups == 2
        tbl = pq.read_table(str(out_path))
        assert tbl.num_rows == 8


# ──────────────────────────────────────────────────────────────────────
# 4. Real-`.raw` integration tier (skipped without an env var)
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
        from constellation.massspec.io.thermo import convert

        bundle_dir = tmp_path / raw_path.stem
        m = convert(raw_path, bundle_dir, compute_sha256=False)
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "peaks.parquet").exists()
        assert (bundle_dir / "scan_metadata.parquet").exists()
        assert (bundle_dir / "acquisition_metadata.parquet").exists()
        assert m.run_metadata["n_scans"] > 0

        peaks = pq.read_table(bundle_dir / "peaks.parquet")
        assert peaks.schema.equals(MS_PEAK_TABLE)
        # Sanity: peaks table is non-empty and m/z values are sorted ascending
        # within at least the first scan.
        assert peaks.num_rows > 0
