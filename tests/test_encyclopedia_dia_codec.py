"""Constellation's own reader, writer and diff for EncyclopeDIA spectra files.

(The ``.dia`` file. The name is EncyclopeDIA's; the file holds DDA runs
as readily as DIA ones, and "DIA" elsewhere in Constellation means the
acquisition method.)

Everything here runs on small bundles built in the test, so it needs no
jar, no Thermo DLLs and no data. The conventions asserted (scan numbers
as ``SpectrumIndex``, seconds, float32-rounded scalars, sequential
float32 TIC, the duty-cycle formula, the merged-file renumbering, which
metadata keys exist for DIA vs DDA) were all read off ``.dia`` files
MSRawJava wrote from real ``.raw`` files; the last test in this file
re-checks them against such a pair when one is available locally.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.massspec.acquisition_structure import infer_acquisition_structure
from constellation.massspec.io.encyclopedia import (
    DiaFile,
    DiaReader,
    SpectraWriteStopped,
    diff_dia,
    import_dia,
    write_dia,
)
from constellation.massspec.io.encyclopedia._codec import encode_f4, encode_f8
from constellation.massspec.io.encyclopedia._schema import (
    DIA_INDEXES_V08,
    DIA_TABLES,
    DIA_TABLES_V08,
)
from constellation.massspec.schemas import (
    ACQUISITION_METADATA_TABLE,
    MS_PEAK_TABLE,
    SCAN_METADATA_TABLE,
)

_MS1_FILTER = "ITMS + c NSI r Full ms [350.0000-1250.0000]"


def _f32(x: float) -> float:
    return float(np.float32(x))


def _seq_tic(intensity: np.ndarray) -> float:
    """Sequential float32 sum, written out longhand as the independent check."""
    total = np.float32(0.0)
    for v in intensity.astype(np.float32):
        total = np.float32(total + v)
    return float(total)


def _dia_scans(
    *,
    cycles: int = 12,
    windows: tuple[tuple[float, float], ...] = ((400.0, 402.0), (402.0, 404.0), (404.0, 406.0)),
    seed: int = 0,
    empty_scan: int | None = None,
) -> list[dict[str, Any]]:
    """A small DIA run: ``cycles`` × (1 MS1 + one MS2 per window)."""
    rng = np.random.default_rng(seed)
    scans: list[dict[str, Any]] = []
    number = 0
    for _cycle in range(cycles):
        for level, window in [(1, None), *[(2, w) for w in windows]]:
            number += 1
            n_peaks = 0 if number == empty_scan else int(rng.integers(3, 9))
            mz = np.sort(rng.uniform(200.0, 1500.0, n_peaks))
            scan = {
                "scan": number,
                "level": level,
                # Not representable in float32, so rounding is exercised.
                "rt": number * 0.731 + 0.0123456789,
                "iit": 31.043 if level == 2 else 0.753,
                "mz": mz,
                "intensity": rng.uniform(10.0, 5e5, n_peaks),
                "filter_string": _MS1_FILTER,
            }
            if window is not None:
                lo, hi = window
                target = (lo + hi) / 2.0 + 0.4324
                scan.update(
                    isolation_lower=target - 1.0,
                    isolation_upper=target + 1.0,
                    precursor_mz=target,
                    ms2_isolation_offset=0.0,
                    filter_string=f"ITMS + c NSI r Full ms2 {target:.4f}@hcd26.00 [200.0000-1500.0000]",
                    activation_type="hcd",
                )
            scans.append(scan)
    return scans


def _dda_scans(*, n_ms1: int = 4, per_ms1: int = 5, seed: int = 1) -> list[dict[str, Any]]:
    """A small DDA run: every MS2 isolates a different precursor."""
    rng = np.random.default_rng(seed)
    scans: list[dict[str, Any]] = []
    number = 0
    for _ in range(n_ms1):
        number += 1
        parent = number
        mz = np.sort(rng.uniform(350.0, 1250.0, 6))
        scans.append(
            {"scan": number, "level": 1, "rt": number * 0.4, "iit": 1.0, "mz": mz,
             "intensity": rng.uniform(1e3, 1e6, 6), "filter_string": _MS1_FILTER}
        )  # fmt: skip
        for _ in range(per_ms1):
            number += 1
            # Five decimals of real precision; the filter string keeps four.
            target = round(float(rng.uniform(400.0, 1000.0)), 5)
            offset = 0.25
            mz = np.sort(rng.uniform(200.0, 1500.0, 5))
            scans.append(
                {"scan": number, "level": 2, "rt": number * 0.4, "iit": 11.333, "mz": mz,
                 "intensity": rng.uniform(10.0, 1e5, 5),
                 "isolation_lower": target + offset - 0.8,
                 "isolation_upper": target + offset + 0.8,
                 "precursor_mz": target, "precursor_charge": 2, "precursor_scan": float(parent),
                 "ms2_isolation_offset": offset, "activation_type": "hcd",
                 "filter_string": f"ITMS + c NSI r d Full ms2 {target:.4f}@hcd26.00 [200.0000-1500.0000]"}
            )  # fmt: skip
    return scans


def _write_bundle(
    root: Path,
    name: str,
    scans: list[dict[str, Any]],
    *,
    profile: bool = False,
    drop_scan_window_columns: bool = False,
) -> Path:
    """Write ``scans`` as a bundle shaped like ``massspec convert`` output."""
    bundle = root / name
    bundle.mkdir(parents=True)
    peak_rows: dict[str, list[Any]] = {f.name: [] for f in MS_PEAK_TABLE}
    meta_rows: list[dict[str, Any]] = []
    for s in scans:
        n = int(s["mz"].size)
        for f in MS_PEAK_TABLE:
            if f.name == "mz":
                peak_rows["mz"].extend(s["mz"].tolist())
            elif f.name == "intensity":
                peak_rows["intensity"].extend(s["intensity"].tolist())
            else:
                peak_rows[f.name].extend([s.get(f.name)] * n)
        row = {k: v for k, v in s.items() if k not in ("mz", "intensity")}
        row["peak_count"] = n
        if s["level"] == 1 and not drop_scan_window_columns:
            row["scan_window_lower"], row["scan_window_upper"] = 350.0, 1250.0
        meta_rows.append(row)
    pq.write_table(
        pa.table(
            {k: pa.array(v, MS_PEAK_TABLE.field(k).type) for k, v in peak_rows.items()},
            schema=MS_PEAK_TABLE,
        ),
        bundle / "peaks.parquet",
        row_group_size=17,  # scans straddle row groups
    )
    scan_table = pa.Table.from_pylist(meta_rows, schema=SCAN_METADATA_TABLE)
    if drop_scan_window_columns:  # a bundle written before schema v3
        scan_table = scan_table.drop_columns(["scan_window_lower", "scan_window_upper"])
    pq.write_table(scan_table, bundle / "scan_metadata.parquet")
    acquisition = {
        "source_format": "thermo_raw",
        "source_path": f"/data/{name}.raw",
        "constellation_version": "0.0.0",
        "schema_version": 1,
        "instrument_model": "Stellar",
        "instrument_serial": "QLT-S-10199",
        "software_version": "1.0.4354",
        "creation_date": "10/07/2026 12:15:40",
        "analyzers": ["ITMS"],
        "n_scans": len(scans),
        "profile_mode": profile,
    }
    pq.write_table(
        pa.Table.from_pylist([acquisition], schema=ACQUISITION_METADATA_TABLE),
        bundle / "acquisition_metadata.parquet",
    )
    return bundle


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# ── single-run export ──────────────────────────────────────────────────


def test_export_single_run_conventions(tmp_path: Path) -> None:
    scans = _dia_scans(empty_scan=7)
    bundle = _write_bundle(tmp_path, "run_a", scans)
    out = tmp_path / "run_a.dia"
    # A tiny batch size forces scans to straddle record batches.
    summary = write_dia([bundle], out, batch_rows=5, chunk_scans=4)

    ms1 = [s for s in scans if s["level"] == 1]
    ms2 = [s for s in scans if s["level"] == 2]
    assert (summary.n_fractions, summary.n_precursor, summary.n_spectra) == (1, len(ms1), len(ms2))
    assert summary.structures[0].acquisition_type == "DIA"

    with DiaFile(out) as dia:
        assert dia.format_version == "0.8.0"
        assert dia.fractions == ()  # a single run names no fractions
        precursor = pa.Table.from_batches(list(dia.iter_precursor()))
        spectra = pa.Table.from_batches(list(dia.iter_spectra()))
        ranges = dia.ranges().to_pylist()
        md = dia.metadata

    # SpectrumIndex is the scan number; names are scan=N; times are seconds.
    assert precursor.column("spectrum_index").to_pylist() == [s["scan"] for s in ms1]
    assert spectra.column("spectrum_index").to_pylist() == [s["scan"] for s in ms2]
    assert spectra.column("spectrum_name").to_pylist() == [f"scan={s['scan']}" for s in ms2]
    assert spectra.column("scan_start_time_s").to_pylist() == [_f32(s["rt"]) for s in ms2]
    # 31.043 ms -> float32(0.031043) s, not float32(float32(31.043)/1000).
    assert set(spectra.column("ion_injection_time").to_pylist()) == {_f32(0.031043)}
    assert set(precursor.column("ion_injection_time").to_pylist()) == {_f32(0.000753)}
    # MS1 "isolation" is the scan range; MS2 target is the window centre.
    assert set(precursor.column("isolation_window_lower").to_pylist()) == {350.0}
    assert set(precursor.column("isolation_window_upper").to_pylist()) == {1250.0}
    assert spectra.column("isolation_window_target").to_pylist() == [
        _f32(s["precursor_mz"]) for s in ms2
    ]
    assert spectra.column("isolation_window_lower").to_pylist() == [
        _f32(s["isolation_lower"]) for s in ms2
    ]
    assert set(spectra.column("precursor_charge").to_pylist()) == {0}  # DIA: none reported
    assert set(spectra.column("precursor_name").to_pylist()) == {"0"}  # no parent scan
    # Peaks: m/z exact, intensity float32; the empty scan is kept, empty.
    for table, source in ((precursor, ms1), (spectra, ms2)):
        for row, s in zip(table.to_pylist(), source):
            assert row["mz_list"] == s["mz"].tolist()
            assert row["intensity_list"] == s["intensity"].astype(np.float32).tolist()
            assert row["tic"] == _seq_tic(s["intensity"])
    assert 7 in spectra.column("spectrum_index").to_pylist() or 7 in precursor.column(
        "spectrum_index"
    ).to_pylist()

    # One range per window; duty cycle from the unrounded times.
    assert len(ranges) == 3
    for r in ranges:
        mine = [s for s in ms2 if _f32(s["isolation_lower"]) == r["start"]]
        assert r["num_windows"] == len(mine) == 12
        assert r["rt_start"] == _f32(mine[0]["rt"]) and r["rt_stop"] == _f32(mine[-1]["rt"])
        assert r["duty_cycle"] == _f32((mine[-1]["rt"] - mine[0]["rt"]) / (len(mine) - 1))

    # Metadata the search reads.
    assert md["version"] == "0.8.0"
    assert md["filename"] == "run_a.dia"
    assert md["sourcename"] == "run_a.raw"  # becomes SourceFile in every result
    assert md["rawFileStructure.dataAcquisitionType"] == "DIA"
    assert md["rawFileStructure.isStaggered"] == "false"
    assert md["rawFileStructure.precursorMarginSize"] == 0
    assert md["conversion.demultiplex"] == "false"
    assert md["conversion.outputType"] == "EncyclopeDIA"
    assert md["totalPrecursorTIC"] == sum(_seq_tic(s["intensity"]) for s in ms1)
    assert md["runStartTime"] == "2026-10-07T12:15:40Z"
    assert md["acq.ms1_count"] == len(ms1) and md["acq.ms2_count"] == len(ms2)
    assert md["acq.mass_analyzers"] == "MassAnalyzerITMS"
    assert md["acq.fragmentations"] == "HigherEnergyCollisionalDissociation"
    assert md["acq.polarities"] == "Positive"
    assert md["InstrumentConfigurations"] == (
        "configurationId:IC1,accession:,name:Stellar[INSTRUMENT-ID-COMPONENT-DELIMITER]"
        "order:1,cvRef:,accessionId:,name:MassAnalyzerITMS,type:analyzer"
    )

    # Tables and indexes as the reference writer creates them.
    names = {r[0] for r in _rows(out, "select name from sqlite_master where name not like 'sqlite_%'")}
    assert {t for t, _ in DIA_TABLES_V08} <= names
    assert len([n for n in names if "_index_" in n]) == len(DIA_INDEXES_V08) == 6

    # The scan map says where every spectrum came from.
    scan_map = pq.read_table(summary.scan_map_path).to_pylist()
    assert [(m["spectrum_index"], m["scan"], m["level"]) for m in scan_map] == [
        (s["scan"], s["scan"], s["level"]) for s in scans
    ]
    assert {m["source_name"] for m in scan_map} == {"run_a.raw"}


def test_export_reads_scan_range_from_filter_string_for_older_bundles(tmp_path: Path) -> None:
    """Bundles from before SCAN_METADATA_TABLE v3 have no scan_window columns."""
    scans = _dia_scans()
    new = write_dia([_write_bundle(tmp_path / "v3", "run", scans)], tmp_path / "v3.dia")
    old = write_dia(
        [_write_bundle(tmp_path / "v2", "run", scans, drop_scan_window_columns=True)],
        tmp_path / "v2.dia",
    )
    assert diff_dia(new.path, old.path).identical


def test_export_is_atomic_and_refuses_to_overwrite(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, "run", _dia_scans())
    out = tmp_path / "run.dia"
    write_dia([bundle], out)
    with pytest.raises(FileExistsError):
        write_dia([bundle], out)
    write_dia([bundle], out, overwrite=True)
    assert [p.name for p in tmp_path.glob(".run.dia.tmp-*")] == []  # no leftovers


def test_export_stops_between_chunks_when_asked(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, "run", _dia_scans())
    out = tmp_path / "run.dia"
    asked = []

    def should_stop() -> bool:
        asked.append(1)
        return len(asked) > 2  # the third chunk

    with pytest.raises(SpectraWriteStopped, match="stopped writing run.dia at scan 8 of"):
        write_dia([bundle], out, chunk_scans=4, should_stop=should_stop)
    assert not out.exists() and list(tmp_path.glob(".run.dia.tmp-*")) == []
    write_dia([bundle], out, chunk_scans=4, should_stop=lambda: False)  # never asked to stop
    assert out.is_file()


# ── DDA ────────────────────────────────────────────────────────────────


def test_export_dda_conventions(tmp_path: Path) -> None:
    scans = _dda_scans()
    out = tmp_path / "dda.dia"
    summary = write_dia([_write_bundle(tmp_path, "dda", scans)], out)
    assert summary.structures[0].acquisition_type == "DDA"
    ms2 = [s for s in scans if s["level"] == 2]
    with DiaFile(out) as dia:
        spectra = pa.Table.from_batches(list(dia.iter_spectra())).to_pylist()
        md = dia.metadata
    for row, s in zip(spectra, ms2):
        # Full-precision target (centre less the offset), not the filter
        # string's four decimals.
        assert row["isolation_window_target"] == _f32(s["precursor_mz"])
        assert row["precursor_name"] == str(int(s["precursor_scan"]))  # parent MS1 scan
        assert row["precursor_charge"] == 2
    assert any(
        _f32(float(f"{s['precursor_mz']:.4f}")) != _f32(s["precursor_mz"]) for s in ms2
    ), "fixture should include a target the 4-decimal filter string cannot represent"
    # A DDA file states its type and nothing about staggering.
    assert md["rawFileStructure.dataAcquisitionType"] == "DDA"
    assert "rawFileStructure.isStaggered" not in md
    assert "rawFileStructure.precursorMarginSize" not in md


# ── acquisition structure ──────────────────────────────────────────────


def _structure(windows: list[tuple[float, float]], visits: int) -> Any:
    rows = [{"scan": 1, "level": 1, "rt": 0.0}]
    for v in range(visits):
        for lo, hi in windows:
            rows.append(
                {"scan": len(rows) + 1, "level": 2, "rt": float(v),
                 "isolation_lower": lo, "isolation_upper": hi}
            )  # fmt: skip
    return infer_acquisition_structure(pa.Table.from_pylist(rows, schema=SCAN_METADATA_TABLE))


def test_infer_acquisition_structure() -> None:
    contiguous = [(400.0 + 8 * i, 408.0 + 8 * i) for i in range(10)]
    s = _structure(contiguous, visits=50)
    assert (s.acquisition_type, s.is_staggered, s.n_windows, s.n_ms2) == ("DIA", False, 10, 500)
    # A 1 m/z margin between neighbours is not staggering.
    margins = [(lo - 0.5, hi + 0.5) for lo, hi in contiguous]
    assert _structure(margins, visits=50).is_staggered is False
    # Alternate cycles offset by half a window.
    staggered = contiguous + [(lo + 4.0, hi + 4.0) for lo, hi in contiguous]
    assert _structure(staggered, visits=50).is_staggered is True
    # Every scan its own window.
    dda = [(400.0 + 0.37 * i, 401.6 + 0.37 * i) for i in range(300)]
    assert _structure(dda, visits=1).acquisition_type == "DDA"
    with pytest.raises(ValueError, match="no MS2"):
        infer_acquisition_structure(
            pa.Table.from_pylist([{"scan": 1, "level": 1, "rt": 0.0}], schema=SCAN_METADATA_TABLE)
        )


# ── merged export ──────────────────────────────────────────────────────


def test_export_merged_fractions(tmp_path: Path) -> None:
    low = _dia_scans(windows=((400.0, 402.0), (402.0, 404.0)), seed=3)
    high = _dia_scans(windows=((500.0, 502.0), (502.0, 504.0), (504.0, 506.0)), seed=4)
    out = tmp_path / "combined.dia"
    summary = write_dia(
        [_write_bundle(tmp_path, "gpf_low", low), _write_bundle(tmp_path, "gpf_high", high)], out
    )
    assert summary.n_fractions == 2 and summary.n_ranges == 5

    with DiaFile(out) as dia:
        assert dia.fractions == ((0, "gpf_low.dia"), (1, "gpf_high.dia"))
        precursor = pa.Table.from_batches(list(dia.iter_precursor())).to_pylist()
        spectra = pa.Table.from_batches(list(dia.iter_spectra())).to_pylist()
        md = dia.metadata

    n1 = [sum(s["level"] == 1 for s in f) for f in (low, high)]
    n2 = [sum(s["level"] == 2 for s in f) for f in (low, high)]
    # Renumbered from 0, continuous: MS1 block then MS2 block, per fraction.
    by_fraction = {
        f: (
            [r["spectrum_index"] for r in precursor if r["fraction"] == f],
            [r["spectrum_index"] for r in spectra if r["fraction"] == f],
        )
        for f in (0, 1)
    }
    assert by_fraction[0][0] == list(range(0, n1[0]))
    assert by_fraction[0][1] == list(range(n1[0], n1[0] + n2[0]))
    base = n1[0] + n2[0]
    assert by_fraction[1][0] == list(range(base, base + n1[1]))
    assert by_fraction[1][1] == list(range(base + n1[1], base + n1[1] + n2[1]))
    # Names keep the original scan number, so they repeat across fractions.
    assert [r["spectrum_name"] for r in precursor if r["fraction"] == 1][0] == "scan=1"
    # Each fraction's MS1 bounds become that fraction's MS2 span.
    for f, source in ((0, low), (1, high)):
        ms2 = [s for s in source if s["level"] == 2]
        lo = min(_f32(s["isolation_lower"]) for s in ms2)
        hi = max(_f32(s["isolation_upper"]) for s in ms2)
        assert {(r["isolation_window_lower"], r["isolation_window_upper"])
                for r in precursor if r["fraction"] == f} == {(lo, hi)}  # fmt: skip
    # File-level metadata only.
    assert md["sourcename"] == md["filename"] == "combined.dia"
    assert md["version"] == "0.8.0" and md["conversion.demultiplex"] == "false"
    assert not any(k.startswith(("rawFileStructure.", "acq.", "run.")) for k in md)
    # The scan map resolves the renumbering.
    scan_map = pq.read_table(summary.scan_map_path).to_pylist()
    last = scan_map[-1]
    assert (last["fraction"], last["source_name"], last["scan"]) == (1, "gpf_high.raw", high[-1]["scan"])


def test_export_rejects_duplicate_fraction_names(tmp_path: Path) -> None:
    a = _write_bundle(tmp_path / "x", "run", _dia_scans())
    b = _write_bundle(tmp_path / "y", "run", _dia_scans(seed=2))
    with pytest.raises(ValueError, match="share a source name"):
        write_dia([a, b], tmp_path / "m.dia")


# ── refusals ───────────────────────────────────────────────────────────


def test_export_refusals(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="profile-mode"):
        write_dia([_write_bundle(tmp_path, "prof", _dia_scans(), profile=True)], tmp_path / "p.dia")

    ms3 = _dia_scans()
    ms3[5]["level"] = 3
    with pytest.raises(ValueError, match="MS levels"):
        write_dia([_write_bundle(tmp_path, "ms3", ms3)], tmp_path / "m.dia")

    faims = _dia_scans()
    for i, s in enumerate(faims):
        s["faims_cv"] = -45.0 if i % 2 else -65.0
    with pytest.raises(ValueError, match="multi-CV FAIMS"):
        write_dia([_write_bundle(tmp_path, "faims", faims)], tmp_path / "f.dia")

    with pytest.raises(FileNotFoundError, match="not a peak bundle"):
        write_dia([tmp_path / "nope"], tmp_path / "n.dia")
    with pytest.raises(ValueError, match="must end in .dia"):
        write_dia([tmp_path / "prof"], tmp_path / "out.sqlite")


def test_export_rejects_peaks_without_scan_metadata(tmp_path: Path) -> None:
    scans = _dia_scans()
    bundle = _write_bundle(tmp_path, "run", scans)
    table = pq.read_table(bundle / "scan_metadata.parquet")
    pq.write_table(table.slice(0, table.num_rows - 3), bundle / "scan_metadata.parquet")
    with pytest.raises(ValueError, match="does not list"):
        write_dia([bundle], tmp_path / "run.dia")


# ── import + round trip ────────────────────────────────────────────────


def test_import_round_trip_single_run(tmp_path: Path) -> None:
    scans = _dia_scans(empty_scan=7)
    first = write_dia([_write_bundle(tmp_path / "src", "run_a", scans)], tmp_path / "run_a.dia")
    (bundle,) = import_dia(first.path, tmp_path / "imported")
    assert bundle.name == "run_a"

    # The bundle has the shape `massspec convert` writes.
    peaks = pq.read_table(bundle / "peaks.parquet")
    meta = pq.read_table(bundle / "scan_metadata.parquet")
    assert peaks.schema.equals(MS_PEAK_TABLE) and meta.schema.equals(SCAN_METADATA_TABLE)
    assert peaks.num_rows == sum(s["mz"].size for s in scans)
    assert meta.column("scan").to_pylist() == [s["scan"] for s in scans]
    ms1 = meta.filter(pa.compute.equal(meta.column("level"), 1)).to_pylist()
    assert {(r["scan_window_lower"], r["scan_window_upper"]) for r in ms1} == {(350.0, 1250.0)}
    assert all(r["isolation_lower"] is None for r in ms1)  # as in a Thermo bundle
    # Seconds in the file, milliseconds in the bundle.
    assert {round(r["iit"], 3) for r in meta.to_pylist() if r["level"] == 2} == {31.043}

    again = write_dia([bundle], tmp_path / "again" / "run_a.dia")
    exact = diff_dia(first.path, again.path)
    # Everything a search reads survives exactly: every scan and peak.
    assert not exact.fields, exact.report()
    # What does not is derived from scan times, which the file holds only
    # at float32: the run's extent in the provenance metadata, and
    # (for some windows) the last digit of the duty cycle.
    assert all(f.startswith(("metadata[run.", "ranges.duty_cycle")) for f in exact.findings)
    assert diff_dia(first.path, again.path, rtol=1e-6).identical


def test_import_merged_gives_one_bundle_per_fraction(tmp_path: Path) -> None:
    low = _dia_scans(windows=((400.0, 402.0), (402.0, 404.0)), seed=3)
    high = _dia_scans(windows=((500.0, 502.0), (502.0, 504.0)), seed=4)
    merged = write_dia(
        [_write_bundle(tmp_path, "gpf_low", low), _write_bundle(tmp_path, "gpf_high", high)],
        tmp_path / "combined.dia",
    )
    bundles = import_dia(merged.path, tmp_path / "imported")
    assert [b.name for b in bundles] == ["gpf_low", "gpf_high"]
    for bundle, source in zip(bundles, (low, high)):
        meta = pq.read_table(bundle / "scan_metadata.parquet")
        assert meta.column("scan").to_pylist() == [s["scan"] for s in source]
    with pytest.raises(FileExistsError):
        import_dia(merged.path, tmp_path / "imported")


# ── the 0.5.0 generation and unsupported content ───────────────────────


def _write_v05(path: Path, *, ion_mobility: bool = False) -> None:
    """A two-scan file in the 2.12.30 layout (or 0.8.0 with ion mobility)."""
    con = sqlite3.connect(path)
    for _name, ddl in DIA_TABLES_V08 if ion_mobility else DIA_TABLES:
        con.execute(ddl)
    mz, mz_len = encode_f8(np.array([500.25, 600.5]))
    inten, int_len = encode_f4(np.array([10.0, 20.0]))
    name = "controllerType=0 controllerNumber=1 scan="
    if ion_mobility:
        con.execute(
            "INSERT INTO spectra VALUES (0,'scan=2',0,2,1.5,0.01,400,401,402,0,?,?,?,?,?,?,30.0)",
            (mz_len, mz, int_len, inten, int_len, inten),
        )
        con.execute(
            "INSERT INTO precursor VALUES (0,'scan=1',1,1.0,0.001,350,1250,?,?,?,?,NULL,NULL,30.0)",
            (mz_len, mz, int_len, inten),
        )
        con.execute("INSERT INTO metadata VALUES ('version','0.8.0')")
    else:
        con.execute(
            "INSERT INTO spectra VALUES (0,?,?,1,1.5,0.01,400,401,402,2,?,?,?,?)",
            (name + "2", name + "1", mz_len, mz, int_len, inten),
        )
        con.execute(
            "INSERT INTO precursor VALUES (0,?,0,1.0,0.001,350,1250,?,?,?,?,30.0)",
            (name + "1", mz_len, mz, int_len, inten),
        )
        con.execute("INSERT INTO ranges VALUES (400,402,0.0,1)")
        con.execute("INSERT INTO metadata VALUES ('version','0.5.0')")
    con.commit()
    con.close()


def test_reader_presents_050_files_in_the_common_shape(tmp_path: Path) -> None:
    path = tmp_path / "old.dia"
    _write_v05(path)
    with DiaFile(path) as dia:
        assert dia.format_version == "0.5.0" and dia.fractions == ()
        (row,) = pa.Table.from_batches(list(dia.iter_spectra())).to_pylist()
        (ranges,) = dia.ranges().to_pylist()
    assert row["isolation_window_target"] == 401.0  # was IsolationWindowCenter
    assert row["tic"] is None  # no per-MS2 TIC in 0.5.0
    assert row["mz_list"] == [500.25, 600.5] and row["intensity_list"] == [10.0, 20.0]
    assert ranges["rt_start"] is None and ranges["num_windows"] == 1
    # The registered whole-file reader goes through the same path.
    result = DiaReader().read(path)
    assert result.primary.num_rows == 1 and result.companions["precursor"].num_rows == 1
    assert result.run_metadata["x.encyclopedia.version"] == "0.5.0"

    # Importing recovers the scan numbers and the parent link from the names.
    (bundle,) = import_dia(path, tmp_path / "imported")
    meta = pq.read_table(bundle / "scan_metadata.parquet").to_pylist()
    assert [(r["scan"], r["level"]) for r in meta] == [(1, 1), (2, 2)]
    assert meta[1]["precursor_scan"] == 1.0 and meta[1]["precursor_charge"] == 2


def test_reader_refuses_ion_mobility_arrays(tmp_path: Path) -> None:
    path = tmp_path / "tims.dia"
    _write_v05(path, ion_mobility=True)
    with DiaFile(path) as dia, pytest.raises(NotImplementedError, match="ion-mobility"):
        list(dia.iter_spectra())


def test_reader_rejects_a_non_dia_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "other.dia"
    sqlite3.connect(path).execute("CREATE TABLE metadata (Key, Value)").connection.commit()
    with pytest.raises(ValueError, match="not a .dia file"):
        DiaFile(path)


# ── diff ───────────────────────────────────────────────────────────────


def test_diff_reports_what_differs_and_keeps_going(tmp_path: Path) -> None:
    scans = _dia_scans()
    a = write_dia([_write_bundle(tmp_path / "a", "run", scans)], tmp_path / "a.dia").path
    assert diff_dia(a, a).identical

    altered = [dict(s) for s in scans]
    altered[10]["intensity"] = altered[10]["intensity"] * 1.5  # one scan's peaks
    del altered[20]  # one scan missing
    b = write_dia([_write_bundle(tmp_path / "b", "run", altered)], tmp_path / "b.dia").path

    d = diff_dia(a, b)
    assert not d.identical
    report = d.report()
    table = "spectra" if scans[10]["level"] == 2 else "precursor"
    assert d.fields[f"{table}.intensity_list"].n_scans == 1
    assert f"fraction 0 scan={scans[10]['scan']}: peak" in report
    # The missing scan is one line, and the scans after it are still compared.
    assert f"1 scans only in a (fraction 0 scan={scans[20]['scan']})" in report
    assert d.n_scans_compared == len(scans) - 1
    assert report.endswith("DIFFERENT")
    # A tolerance wide enough for the intensity change removes that finding.
    assert f"{table}.intensity_list" not in diff_dia(a, b, intensity_rtol=0.5).fields


def test_diff_metadata_modes(tmp_path: Path) -> None:
    scans = _dia_scans()
    a = write_dia([_write_bundle(tmp_path / "a", "run", scans)], tmp_path / "a.dia").path
    b = tmp_path / "b.dia"
    b.write_bytes(a.read_bytes())
    con = sqlite3.connect(b)
    con.execute("INSERT INTO metadata VALUES ('thermo.sample.vial', 'G:B3')")
    con.execute("UPDATE metadata SET Value = 'elsewhere.dia' WHERE Key = 'filename'")
    con.commit()
    con.close()
    # Writer-specific keys are ignored; a key one side lacks is not, by
    # default, a difference between the spectra.
    assert diff_dia(a, b).identical
    strict = diff_dia(a, b, metadata_keys="all")
    assert strict.findings == ["metadata keys only in b: ['thermo.sample.vial']"]
    con = sqlite3.connect(b)
    con.execute("UPDATE metadata SET Value = 'DDA' WHERE Key = 'rawFileStructure.dataAcquisitionType'")
    con.commit()
    con.close()
    assert "metadata[rawFileStructure.dataAcquisitionType]: 'DIA' vs 'DDA'" in diff_dia(a, b).findings


def test_diff_pairs_merged_files_by_fraction_and_scan(tmp_path: Path) -> None:
    """A scan missing from one merged file must not misalign the rest.

    In a merged file SpectrumIndex is a running count, so dropping one
    scan shifts every later index. Pairing on (fraction, scan) keeps the
    comparison on the same scans and reports the shift as a count.
    """
    low = _dia_scans(windows=((400.0, 402.0), (402.0, 404.0)), seed=3)
    high = _dia_scans(windows=((500.0, 502.0), (502.0, 504.0)), seed=4)
    a = write_dia(
        [_write_bundle(tmp_path / "a", "gpf_low", low), _write_bundle(tmp_path / "a", "gpf_high", high)],
        tmp_path / "a.dia",
    ).path
    dropped = low[4]  # an MS2 scan early in the first fraction
    assert dropped["level"] == 2
    low_b = [s for s in low if s is not dropped]
    b = write_dia(
        [_write_bundle(tmp_path / "b", "gpf_low", low_b), _write_bundle(tmp_path / "b", "gpf_high", high)],
        tmp_path / "b.dia",
    ).path

    d = diff_dia(a, b)
    assert f"spectra: 1 scans only in a (fraction 0 scan={dropped['scan']})" in d.findings
    assert d.n_scans_compared == len(low) + len(high) - 1
    # Every later scan is the same scan with a shifted index, and nothing else.
    differing = {name for name in d.fields if not name.endswith(".spectrum_index")}
    assert differing == set(), d.report()
    assert d.fields["spectra.spectrum_index"].max_abs == 1.0
    assert d.fields["precursor.spectrum_index"].n_scans == sum(s["level"] == 1 for s in high)


# ── CLI ────────────────────────────────────────────────────────────────


def test_cli_export_import_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from constellation.cli.__main__ import main

    enc = ["massspec", "encyclopedia"]
    bundle = _write_bundle(tmp_path, "run", _dia_scans())
    out = tmp_path / "out" / "run.dia"
    export = [*enc, "export-spectra", "--peaks-dir", str(bundle), "--output-file", str(out)]
    assert main(export) == 0
    assert "1 fraction(s) [DIA]" in capsys.readouterr().err
    assert main(export) == 2
    assert "exists" in capsys.readouterr().err

    imported = tmp_path / "imp"
    assert main([*enc, "import-spectra", "--spectra-file", str(out), "-o", str(imported)]) == 0
    capsys.readouterr()
    again = tmp_path / "again" / "run.dia"
    assert main([*enc, "export-spectra", "--peaks-dir", str(imported / "run"),
                 "--output-file", str(again), "--no-progress"]) == 0  # fmt: skip
    assert main([*enc, "compare-spectra", str(out), str(again), "--rtol", "1e-6"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("IDENTICAL")

    con = sqlite3.connect(again)
    con.execute("UPDATE spectra SET PrecursorCharge = 3 WHERE SpectrumIndex = 2")
    con.commit()
    con.close()
    assert main([*enc, "compare-spectra", str(out), str(again), "--rtol", "1e-6"]) == 1
    assert "spectra.precursor_charge: differs in 1 scans" in capsys.readouterr().out


# ── against a reference pair, when one is available ────────────────────

_REFERENCE_DIA = os.environ.get("CONSTELLATION_DIA_REFERENCE")
_REFERENCE_BUNDLE = os.environ.get("CONSTELLATION_DIA_REFERENCE_BUNDLE")


@pytest.mark.slow
@pytest.mark.skipif(
    not (_REFERENCE_DIA and _REFERENCE_BUNDLE),
    reason=(
        "set CONSTELLATION_DIA_REFERENCE to a .dia MSRawJava wrote from a DIA .raw and "
        "CONSTELLATION_DIA_REFERENCE_BUNDLE to the `massspec convert` bundle of that .raw"
    ),
)
def test_export_matches_a_reference_dia(tmp_path: Path) -> None:
    """A Constellation-written file equals the reference writer's, scan for scan.

    Exact, no tolerances: every scalar, every peak, every isolation window
    and every metadata value both files carry.
    """
    summary = write_dia([Path(_REFERENCE_BUNDLE)], tmp_path / Path(_REFERENCE_DIA).name)
    result = diff_dia(summary.path, _REFERENCE_DIA)
    assert result.identical, result.report()
    assert result.n_scans_compared == summary.n_precursor + summary.n_spectra
