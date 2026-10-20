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
    nearest_scan,
    open_acquisition,
    parent_scan,
    scan_table,
    spectra,
    spectrum,
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
