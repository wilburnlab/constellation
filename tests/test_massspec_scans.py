"""Random access to a converted acquisition (``massspec.scans``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest

from constellation.massspec.io.encyclopedia import import_dia, write_dia
from constellation.massspec.quant.peak_index import build_peak_index
from constellation.massspec.scans import (
    chromatogram,
    default_tolerance,
    dependent_scans,
    isolation_window_table,
    locate_precursor,
    nearest_scan,
    open_acquisition,
    parent_scan,
    scan_table,
    spectra,
    spectrum,
    xic,
)

from _massspec_fixtures import (
    dda_scans,
    dia_scans,
    with_scan_stats,
    write_bundle,
    write_bundle_manifest,
)

WINDOWS = ((402.0, 404.0), (400.0, 402.0), (404.0, 406.0))  # deliberately unsorted


@pytest.fixture
def dia(tmp_path: Path) -> tuple[Path, list[dict]]:
    scans = with_scan_stats(dia_scans(windows=WINDOWS, empty_scan=6))
    return write_bundle_manifest(write_bundle(tmp_path, "dia_run", scans)), scans


@pytest.fixture
def dda(tmp_path: Path) -> tuple[Path, list[dict]]:
    scans = with_scan_stats(dda_scans())
    return write_bundle_manifest(write_bundle(tmp_path, "dda_run", scans)), scans


# ── what the run is ───────────────────────────────────────────────────────


def test_a_dia_run_is_described(dia) -> None:
    bundle, scans = dia
    info = open_acquisition(bundle)
    assert info.kind == "thermo_acquisition"
    assert info.source_file == "/data/dia_run.raw" and info.source_sha256 == "sha-of-dia_run"
    assert info.acquisition_type == "DIA" and not info.staggered
    assert info.levels == (1, 2)
    assert info.scans_per_level == {1: 12, 2: 36}
    # The scans name no analyzer, so the run's list stands for every level.
    assert info.analyzers == {1: ("ITMS",), 2: ("ITMS",)}
    assert info.rt_range == (scans[0]["rt"], scans[-1]["rt"])
    assert info.index is None


def test_a_built_peak_index_is_found(dia) -> None:
    bundle, _ = dia
    build_peak_index(bundle, bundle / "mz_index")
    assert open_acquisition(bundle).index == bundle / "mz_index"


def test_a_directory_that_is_not_a_bundle_is_refused(tmp_path: Path, dia) -> None:
    with pytest.raises(ValueError, match="no manifest"):
        open_acquisition(tmp_path)
    bundle, _ = dia
    (bundle / "manifest.json").write_text(json.dumps({"kind": "massspec_experiment"}))
    with pytest.raises(ValueError, match="not a converted acquisition"):
        open_acquisition(bundle)


def test_a_dda_run_is_described(dda) -> None:
    assert open_acquisition(dda[0]).acquisition_type == "DDA"


@pytest.mark.parametrize(
    ("analyzers", "expected"),
    [
        (["ITMS"], (0.35, "Da")),
        (["FTMS"], (10.0, "ppm")),
        (["FTMS", "ITMS"], (0.35, "Da")),  # the loosest of a mixed run
        ([], (20.0, "ppm")),
        (["SOMETHING"], (20.0, "ppm")),
    ],
)
def test_default_tolerance_follows_the_analyzer(analyzers, expected) -> None:
    assert default_tolerance(analyzers) == expected


# ── the scan table ────────────────────────────────────────────────────────


def test_dia_windows_are_numbered_by_lower_bound(dia) -> None:
    bundle, scans = dia
    table = scan_table(bundle)
    assert table.column("scan").to_pylist() == [s["scan"] for s in scans]
    ms1 = table.filter(pc.equal(table.column("level"), 1))
    assert ms1.column("window").null_count == ms1.num_rows

    windows = isolation_window_table(table)
    assert windows.column("window").to_pylist() == [0, 1, 2]
    lower = windows.column("isolation_lower").to_numpy()
    assert np.all(np.diff(lower) > 0)
    assert windows.column("n_scans").to_pylist() == [12, 12, 12]
    centers = windows.column("center").to_numpy()
    assert np.allclose(centers, (lower + windows.column("isolation_upper").to_numpy()) / 2)

    for row in table.filter(pc.equal(table.column("level"), 2)).to_pylist():
        w = windows.filter(pc.equal(windows.column("window"), row["window"])).to_pylist()[0]
        assert np.float32(row["isolation_lower"]) == np.float32(w["isolation_lower"])


def test_a_dia_scans_parent_is_the_ms1_before_it(dia) -> None:
    bundle, scans = dia
    table = scan_table(bundle)
    last_ms1 = None
    for s in scans:
        if s["level"] == 1:
            last_ms1 = s["scan"]
            assert parent_scan(table, s["scan"]) is None
        else:
            assert parent_scan(table, s["scan"]) == last_ms1


def test_a_dda_scan_keeps_its_recorded_parent_and_has_no_window(dda) -> None:
    bundle, scans = dda
    table = scan_table(bundle)
    assert table.column("window").null_count == table.num_rows
    assert isolation_window_table(table).num_rows == 0
    ms1 = [s["scan"] for s in scans if s["level"] == 1]
    assert dependent_scans(table, ms1[0]) == [ms1[0] + i for i in range(1, 6)]
    assert parent_scan(table, ms1[0] + 3) == ms1[0]
    with pytest.raises(KeyError):
        parent_scan(table, 10_000)


def test_columns_a_bundle_predates_read_as_null(dia) -> None:
    bundle, _ = dia
    table = scan_table(bundle, columns=("scan", "level", "rt", "scan_window_lower", "no_such"))
    assert "no_such" in table.column_names
    assert table.column("no_such").null_count == table.num_rows


# ── chromatograms and navigation ──────────────────────────────────────────


def test_chromatograms_by_level_and_window(dia) -> None:
    bundle, scans = dia
    table = scan_table(bundle)
    tic = chromatogram(table, level=1)
    expected = [s for s in scans if s["level"] == 1]
    assert tic.column("scan").to_pylist() == [s["scan"] for s in expected]
    assert tic.column("intensity").to_pylist() == pytest.approx([s.get("tic") for s in expected])

    bpc = chromatogram(table, level=1, kind="bpc")
    assert bpc.column("mz").to_pylist() == [s.get("base_peak_mz") for s in expected]

    one_window = chromatogram(table, level=2, window=1)
    assert one_window.num_rows == 12
    lows = {round(s["isolation_lower"], 3) for s in scans
            if s["level"] == 2 and s["scan"] in one_window.column("scan").to_pylist()}  # fmt: skip
    assert len(lows) == 1

    t0, t1 = scans[10]["rt"], scans[30]["rt"]
    clipped = chromatogram(table, level=1, rt_range=(t0, t1)).column("rt").to_numpy()
    assert clipped.min() >= t0 and clipped.max() <= t1


def test_a_base_peak_chromatogram_needs_base_peaks(tmp_path: Path) -> None:
    bundle = write_bundle_manifest(write_bundle(tmp_path, "bare", dia_scans()))
    with pytest.raises(ValueError, match="no base peaks"):
        chromatogram(scan_table(bundle), level=1, kind="bpc")


def test_the_nearest_scan_of_a_level_and_window(dia) -> None:
    bundle, scans = dia
    table = scan_table(bundle)
    target = scans[17]
    assert nearest_scan(table, target["rt"] + 0.01, level=target["level"]) == target["scan"]
    ms2 = table.filter(pc.equal(table.column("scan"), target["scan"])).to_pylist()[0]
    assert ms2["level"] == 2
    found = nearest_scan(table, target["rt"], level=2, window=ms2["window"])
    assert found == target["scan"]
    assert nearest_scan(table, 0.0, level=3) is None


# ── peaks ─────────────────────────────────────────────────────────────────


def test_one_spectrum_is_read_by_scan_number(dia) -> None:
    bundle, scans = dia
    for s in (scans[0], scans[9], scans[-1]):
        peaks = spectrum(bundle, s["scan"])
        order = np.argsort(s["mz"])
        assert peaks.column("mz").to_numpy().tolist() == s["mz"][order].tolist()
        assert peaks.column("intensity").to_numpy().tolist() == s["intensity"][order].tolist()
    assert spectrum(bundle, 6).num_rows == 0  # a scan with no peaks


def test_several_spectra_in_one_read(dia) -> None:
    bundle, scans = dia
    wanted = [scans[3]["scan"], scans[20]["scan"]]
    table = spectra(bundle, wanted)
    assert sorted(set(table.column("scan").to_pylist())) == sorted(wanted)
    assert table.num_rows == sum(scans[i]["mz"].size for i in (3, 20))


# ── a bundle imported from a spectra file ─────────────────────────────────


def test_an_imported_bundle_reads_the_same_way(tmp_path: Path, dia) -> None:
    bundle, scans = dia
    spectra_file = write_dia([bundle], tmp_path / "dia_run.dia")
    (imported,) = import_dia(spectra_file.path, tmp_path / "imported")
    info = open_acquisition(imported)
    assert info.kind == "encyclopedia_dia_acquisition"
    assert info.acquisition_type == "DIA" and info.analyzers == {1: ("ITMS",), 2: ("ITMS",)}
    table = scan_table(imported)
    assert isolation_window_table(table).num_rows == 3
    # Import derives the base peak from the stored peaks.
    bpc = chromatogram(table, level=1, kind="bpc")
    expected = [s.get("base_peak_mz") for s in scans if s["level"] == 1]
    assert bpc.column("mz").to_pylist() == expected


# ── extracted-ion chromatograms and precursor location ───────────────────


def _expected_trace(scans: list[dict], target: float, half: float, keep) -> dict[int, float]:
    """Brute force: per kept scan, the peak within tolerance nearest the
    target (the more intense on a tie), 0 if none."""
    out = {}
    for s in scans:
        if not keep(s):
            continue
        hits = [(abs(m - target), -i, i) for m, i in zip(s["mz"], s["intensity"]) if abs(m - target) <= half]
        out[s["scan"]] = min(hits)[2] if hits else 0.0
    return out


def test_xic_matches_brute_force_with_and_without_the_index(dia) -> None:
    bundle, scans = dia
    ms2 = [s for s in scans if s["level"] == 2 and s["mz"].size]
    target = float(ms2[4]["mz"][1])  # a real peak of one MS2 scan
    table = scan_table(bundle)
    window = table.filter(pc.equal(table.column("scan"), ms2[4]["scan"])).column("window")[0].as_py()
    lower = round(ms2[4]["isolation_lower"], 6)

    def in_window(s):
        return s["level"] == 2 and round(s["isolation_lower"], 6) == lower

    expected = _expected_trace(scans, target, 0.5, in_window)
    plain = xic(bundle, [target], level=2, window=window, tolerance=0.5, tolerance_unit="Da")
    assert dict(zip(plain.column("scan").to_pylist(), plain.column("intensity").to_pylist())) == expected
    assert plain.column("mz_observed").null_count == sum(v == 0 for v in expected.values())

    build_peak_index(bundle, bundle / "mz_index")
    indexed = xic(bundle, [target], level=2, window=window, tolerance=0.5, tolerance_unit="Da")
    assert indexed.equals(plain)


def test_xic_zero_fills_and_respects_the_rt_window(dia) -> None:
    bundle, scans = dia
    rts = [s["rt"] for s in scans if s["level"] == 1]
    out = xic(bundle, [5000.0, 123.0], level=1, tolerance=10, tolerance_unit="ppm", rt_range=(rts[2], rts[5]))
    assert out.column("trace").to_pylist() == [0] * 4 + [1] * 4
    assert set(out.column("intensity").to_pylist()) == {0.0}
    assert out.column("mz_observed").null_count == 8
    assert min(out.column("rt").to_pylist()) == rts[2] and max(out.column("rt").to_pylist()) == rts[5]


def test_locating_a_dia_precursor(dia) -> None:
    bundle, scans = dia
    table = scan_table(bundle)
    windows = isolation_window_table(table).to_pylist()
    w = windows[1]
    target_scan = [s for s in scans if s["level"] == 2 and round(s["isolation_lower"], 4) == round(w["isolation_lower"], 4)][3]
    found = locate_precursor(table, precursor_mz=w["center"], rt=target_scan["rt"])
    assert found["window"] == w["window"]
    assert found["ms_scan"] == target_scan["scan"]
    assert found["parent_scan"] == nearest_scan(table, target_scan["rt"], level=1)
    nowhere = locate_precursor(table, precursor_mz=2000.0, rt=target_scan["rt"])
    assert nowhere["window"] is None and nowhere["ms_scan"] is None


def test_locating_a_dda_precursor(dda) -> None:
    bundle, scans = dda
    table = scan_table(bundle)
    picked = [s for s in scans if s["level"] == 2][7]
    found = locate_precursor(table, precursor_mz=picked["precursor_mz"], rt=picked["rt"])
    assert found["window"] is None
    assert found["ms_scan"] == picked["scan"]
    assert found["isolation"] == pytest.approx((picked["isolation_lower"], picked["isolation_upper"]))


def test_xic_takes_the_nearest_peak_not_the_most_intense(tmp_path) -> None:
    scans = with_scan_stats(dia_scans(cycles=3))
    target = 777.0
    ms1 = scans[0]
    ms1["mz"] = np.array([target - 0.05, target + 0.3])  # near and weak, far and strong
    ms1["intensity"] = np.array([10.0, 1000.0])
    bundle = write_bundle_manifest(write_bundle(tmp_path, "close", scans))
    trace = xic(bundle, [target], level=1, tolerance=0.35, tolerance_unit="Da")
    first = trace.slice(0, 1).to_pylist()[0]
    assert first["scan"] == ms1["scan"]
    assert first["intensity"] == 10.0 and first["mz_observed"] == pytest.approx(target - 0.05)
