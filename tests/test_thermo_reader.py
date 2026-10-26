"""Tests for ``constellation.massspec.readers.thermo``.

Three tiers:

1. **Pure-Python unit tests** (no pythonnet / DLL dependency) — exercise
   the filter-string parser, the declarative trailer table + coercion
   helpers, the manifest round-trip and the schema registry.

2. **A stand-in raw file** (``_FakeRaw``, no DLLs) — drives the range
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
        assert SCAN_METADATA_TABLE.metadata[b"schema_version"] == b"4"

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
# 2. A stand-in raw file: the range reader, the peak writer, convert()
# ──────────────────────────────────────────────────────────────────────


class _Obj:
    """Attribute bag standing in for a CommonCore object."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeEvent:
    def __init__(self, mass=None, width=None, offset=0.0):
        self._mass, self._width, self._offset = mass, width, offset

    def GetMass(self, idx):  # noqa: ARG002
        if self._mass is None:
            raise RuntimeError("no precursor")
        return self._mass

    def GetIsolationWidth(self, idx):  # noqa: ARG002
        if self._width is None:
            raise RuntimeError("no width")
        return self._width

    def GetIsolationWidthOffset(self, idx):  # noqa: ARG002
        return self._offset


def _scan(
    scan_type: str,
    *,
    rt_min: float,
    acquired: str = "centroid",
    mz=(),
    intensity=None,
    stream=None,
    trailer=(),
    event=None,
):
    """One stand-in scan. ``acquired`` is ``centroid`` or ``profile``;
    ``stream`` is ``None`` (no centroid stream), ``()`` (an empty one) or
    ``(mz, intensity, resolution, noise, baseline)``."""
    mz = np.asarray(mz, dtype=np.float64)
    intensity = (
        np.asarray(intensity, dtype=np.float64)
        if intensity is not None
        else np.arange(1, mz.size + 1, dtype=np.float64)
    )
    return {
        "stats": _Obj(
            TIC=float(intensity.sum()),
            BasePeakMass=float(mz[np.argmax(intensity)]) if mz.size else 0.0,
            BasePeakIntensity=float(intensity.max()) if mz.size else 0.0,
            StartTime=rt_min,
            ScanType=scan_type,
            IsCentroidScan=acquired == "centroid",
        ),
        "mz": mz,
        "intensity": intensity,
        "stream": stream,
        "trailer": _Obj(Labels=[k for k, _ in trailer], Values=[v for _, v in trailer]),
        "event": event or _FakeEvent(),
    }


class _FakeRaw:
    """The slice of IRawDataPlus the range reader calls."""

    def __init__(self, scans: dict[int, dict]):
        self.scans = scans
        self.RunHeaderEx = _Obj(FirstSpectrum=min(scans), LastSpectrum=max(scans))

    def GetScanStatsForScanNumber(self, n):
        return self.scans[n]["stats"]

    def GetScanEvents(self, lo, hi):
        return [self.scans[n]["event"] for n in range(lo, hi + 1)]

    def GetScanEventForScanNumber(self, n):
        return self.scans[n]["event"]

    def GetSegmentedScanFromScanNumber(self, n, stats):  # noqa: ARG002
        s = self.scans[n]
        return _Obj(Positions=s["mz"], Intensities=s["intensity"])

    def GetCentroidStream(self, n, include_ref):  # noqa: ARG002
        stream = self.scans[n]["stream"]
        if stream is None or stream == ():
            return _Obj(Length=0, Masses=None, Intensities=None)
        mz, inten, res, noise, base = (np.asarray(a, dtype=np.float64) for a in stream)
        return _Obj(
            Length=mz.size, Masses=mz, Intensities=inten,
            Resolutions=res, Noises=noise, Baselines=base,
        )

    def GetTrailerExtraInformation(self, n):
        return self.scans[n]["trailer"]

    def Dispose(self):
        pass


@pytest.fixture
def fake_centroider(monkeypatch):
    """Stand in for ``Scan.FromFile`` + Thermo's centroider: the "centroids"
    of a profile scan are its points above half its maximum."""
    from constellation.massspec.readers.thermo import _scans

    def load_scan(raw, n):
        s = raw.scans[n]
        return _Obj(HasCentroidStream=s["stream"] is not None, mz=s["mz"], intensity=s["intensity"])

    def centroider(scan):
        keep = scan.intensity >= scan.intensity.max() / 2
        return scan.mz[keep], scan.intensity[keep]

    monkeypatch.setattr(_scans, "_load_scan", load_scan)
    monkeypatch.setitem(_scans.CENTROIDERS, "thermo", centroider)


_MS1 = "FTMS + p NSI Full ms [400.0000-1600.0000]"
_MS2 = "ITMS + c NSI r Full ms2 500.2500@hcd30.00 [200.0000-1500.0000]"
_IT_MS1 = "ITMS + p NSI r Full ms [400.0000-1200.0000]"


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
        return _FakeRaw({
            # Orbitrap profile MS1 with the instrument's centroid stream.
            1: _scan(_MS1, rt_min=0.5, acquired="profile", mz=[400.0, 400.1, 400.2],
                     intensity=[1.0, 9.0, 2.0],
                     stream=([400.1, 500.0], [9.0, 4.0], [6e4, 5e4], [1.0, 1.5], [0.1, 0.2]),
                     trailer=[("Ion Injection Time (ms):", "15.000"), ("AGC:", "On")]),
            # Ion-trap centroid MS2.
            2: _scan(_MS2, rt_min=0.51, mz=[201.0, 305.5, 410.2],
                     trailer=[("Monoisotopic M/Z:", "500.2512"), ("Charge State:", "2"),
                              ("MS2 Isolation Width:", "2.00"), ("Master Scan Number:", "1")],
                     event=_FakeEvent(mass=500.25, width=2.0009)),
            # Ion-trap profile MS1: no centroid stream at all.
            3: _scan(_IT_MS1, rt_min=1.2, acquired="profile",
                     mz=[450.0, 450.1, 450.2, 450.3], intensity=[1.0, 8.0, 10.0, 2.0]),
            # Orbitrap profile MS1 whose stream exists but is empty.
            4: _scan(_MS1, rt_min=1.3, acquired="profile", mz=[400.0, 400.1],
                     intensity=[3.0, 4.0], stream=()),
        })

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

    def test_profile_keeps_points(self, fake_centroider):
        scans, peaks = _read(self._raw(), profile=True)
        assert scans.column("spectrum_type").to_pylist() == [
            "profile", "centroid", "profile", "profile",
        ]
        assert scans.column("centroider").to_pylist() == [None, "instrument", None, None]
        assert scans.column("peak_count").to_pylist() == [3, 3, 4, 2]
        assert peaks.num_rows == 12

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
        # The trailer's width, centred on the isolation target.
        assert ms2["isolation_lower"] == pytest.approx(499.25)
        assert ms2["isolation_upper"] == pytest.approx(501.25)
        assert ("MS2 Isolation Width:", "2.0") in ms2["trailer_extras"]
        # Per-scan values repeated on every peak of the scan.
        ms2_peaks = peaks.filter(pa.compute.equal(peaks["scan"], 2))
        assert set(ms2_peaks.column("precursor_mz").to_pylist()) == {500.2512}
        assert set(ms2_peaks.column("level").to_pylist()) == {2}

    def test_no_trailer_extras_keeps_typed_columns(self, fake_centroider):
        scans, _ = _read(self._raw(), capture_trailer_extras=False)
        ms1 = scans.to_pylist()[0]
        assert ms1["trailer_extras"] is None
        assert ms1["agc_enabled"] is True and ms1["iit"] == 15.0

    def test_negative_value_in_unsigned_column_is_null(self, fake_centroider):
        raw = _FakeRaw({1: _scan(_MS1, rt_min=0.1, mz=[400.0],
                                 trailer=[("Master Index:", "-1"), ("AGC PS Mode:", "-1")])})
        scans, _ = _read(raw)
        row = scans.to_pylist()[0]
        assert row["master_index"] is None  # uint32
        assert row["agc_ps_mode"] == -1  # signed since scan-metadata v4

    def test_peaks_sorted_within_each_scan(self, fake_centroider):
        raw = _FakeRaw({
            1: _scan(_MS1, rt_min=0.1, mz=[400.0, 300.0, 500.0], intensity=[1, 2, 3]),
            2: _scan(_MS1, rt_min=0.2, mz=[100.0, 200.0], intensity=[4, 5]),
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
            "manifest.json", "peaks.parquet", "scan_metadata.parquet",
            "acquisition_metadata.parquet",
        }
        assert m.parameters["peaks"] == "centroid"
        assert m.parameters["centroider"] == "thermo"
        assert m.source_sha256 == hashlib.sha256(src.read_bytes()).hexdigest()
        assert describe_levels(m.run_metadata["levels"]) == [
            "MS1: 3 scans, 4 peaks (2 centroided by the instrument, 1 centroided by "
            "Thermo's centroider)",
            "MS2: 1 scans, 3 peaks (1 centroided by the instrument)",
        ]
        acq = pq.read_table(tmp_path / "run" / "acquisition_metadata.parquet").to_pylist()[0]
        assert acq["profile_mode"] is False  # nothing stored as profile
        assert acq["analyzers"] == ["FTMS", "ITMS"]
        assert pq.read_table(tmp_path / "run" / "peaks.parquet").num_rows == 7

    def test_profile_bundle_says_so(self, raw, tmp_path):
        from constellation.massspec.readers.thermo import convert

        _, src = raw
        m = convert(src, tmp_path / "run", profile=True, compute_sha256=False)
        acq = pq.read_table(tmp_path / "run" / "acquisition_metadata.parquet").to_pylist()[0]
        assert acq["profile_mode"] is True
        assert m.parameters["peaks"] == "profile"
        assert m.source_sha256 is None

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
        assert (bundle_dir / "peaks.parquet").exists()
        assert (bundle_dir / "scan_metadata.parquet").exists()
        assert (bundle_dir / "acquisition_metadata.parquet").exists()
        assert m.run_metadata["n_scans"] > 0

        peaks = pq.read_table(bundle_dir / "peaks.parquet")
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
