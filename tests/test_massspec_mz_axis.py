"""The m/z axis: the phase of a fold, and a run's error read from it.

Spectra are synthesised from a small digest-like lattice with a known shift.
The estimator must recover the shift with no peak picking, from profile points
or peaks alike; a literal FineTune replica (a binned map and a 2**18-point FFT)
must agree with the direct sum; and a stored curve must change nothing until it
is named.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet as pq
import pytest
import torch
from _massspec_fixtures import write_bundle, write_bundle_manifest

from constellation.core.signal.phase import PhaseFold
from constellation.massspec.bundle_summary import format_bundle_summary, summarize_bundle
from constellation.massspec.mz_axis import (
    LATEST,
    MzAxisSettings,
    describe,
    estimate_mz_axis,
    list_mz_axes,
    load_mz_axis,
    load_reference,
    save_mz_axis,
    scale_error_ppm,
)
from constellation.massspec.scans import spectrum, xic

PERIOD = 1.000495


def _reference() -> tuple[dict[str, np.ndarray], dict]:
    """Ions exactly on the lattice's zero phase, at every bin."""
    n_bins = 2000
    flat = np.ones(n_bins, dtype=np.complex128)
    return (
        {"fragment_1plus": flat, "precursor_2plus": flat},
        {"period": PERIOD, "lo": 100.0, "bin_width": 1.0, "n_bins": n_bins, "source": "test"},
    )


def _peaks(rng, lo: float, hi: float, period: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    k = rng.integers(int(lo / period) + 1, int(hi / period), n)
    return np.sort(k * period), rng.uniform(1e3, 1e5, n)


def _profile_of(mz: np.ndarray, intensity: np.ndarray, *, skew: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Points on a 1/15 Th grid, each peak Gaussian (FWHM 0.33), like a trap."""
    grid = np.arange(200.0, 1500.0, 1 / 15)
    y = np.zeros(grid.size)
    s = 0.14
    for m, i in zip(mz, intensity):
        a, b = np.searchsorted(grid, [m - 1, m + 1])
        u = grid[a:b] - m
        width = np.where(u > 0, s * (1 + skew), s)
        y[a:b] += i * np.exp(-0.5 * (u / width) ** 2)
    return grid, y


def _run(tmp: Path, name: str, shift, *, profile: bool, n_scans: int = 60, seed: int = 3, ms1: bool = True) -> Path:
    """MS1 every tenth scan on the 2+ lattice, MS2 on the 1+, read ``shift(mz)`` high."""
    rng = np.random.default_rng(seed)
    scans = []
    for number in range(1, n_scans + 1):
        level = 1 if ms1 and number % 10 == 1 else 2
        period = PERIOD / (2 if level == 1 else 1)
        lo, hi = (400.0, 1200.0) if level == 1 else (200.0, 1400.0)
        mz, intensity = _peaks(rng, lo, hi, period, 400)
        mz = mz + shift(mz)
        if profile:
            mz, intensity = _profile_of(mz, intensity)
        scan = {
            "scan": number, "level": level, "rt": float(number), "iit": 10.0,
            "mz": mz, "intensity": intensity,
            "filter_string": "ITMS + p NSI r Full ms [400.0000-1200.0000]",
        }
        if level == 2:
            scan.update(isolation_lower=700.0, isolation_upper=708.0, precursor_mz=704.0)
        scans.append(scan)
    bundle = write_bundle(tmp, name, scans, profile=profile, row_group_size=4096)
    return write_bundle_manifest(bundle, run="run", sha256="abc")


SETTINGS = MzAxisSettings(min_scans={1: 6, 2: 50}, n_blocks=8, target_se_th=0.02)


class TestPhaseFold:
    def test_the_phase_is_where_the_signal_clusters(self):
        fold = PhaseFold(1.0, 0.0, 100.0)
        x = torch.arange(1.0, 99.0, dtype=torch.float64) + 0.2
        fold.add(x, torch.ones_like(x))
        w = fold.window(0.0, 100.0)
        assert w.phase * 1.0 / (2 * math.pi) == pytest.approx(0.2)
        assert w.coherence == pytest.approx(1.0)

    def test_a_reference_is_subtracted(self):
        fold = PhaseFold(1.0, 0.0, 100.0)
        x = torch.arange(1.0, 99.0, dtype=torch.float64) + 0.3
        fold.add(x, torch.ones_like(x))
        reference = complex(np.exp(2j * np.pi * 0.1))
        assert fold.window(0.0, 100.0, reference=reference).phase / (2 * math.pi) == pytest.approx(0.2)

    def test_a_symmetric_shape_leaves_the_phase_alone(self):
        """The reason no picking is needed: smearing only shortens the sum."""
        sticks = PhaseFold(1.0, 0.0, 100.0)
        smeared = PhaseFold(1.0, 0.0, 100.0)
        centres = torch.arange(1.0, 99.0, dtype=torch.float64) + 0.2
        sticks.add(centres, torch.ones_like(centres))
        grid = torch.arange(0.0, 100.0, 1 / 64, dtype=torch.float64)
        y = torch.exp(-0.5 * ((grid[:, None] - centres[None, :]) / 0.14) ** 2).sum(1)
        smeared.add(grid, y)
        a, b = sticks.window(0.0, 100.0), smeared.window(0.0, 100.0)
        assert b.phase == pytest.approx(a.phase, abs=1e-6)
        assert b.coherence < a.coherence

    def test_unordered_arrows_have_no_coherence_and_a_wide_error(self):
        rng = np.random.default_rng(1)
        fold = PhaseFold(1.0, 0.0, 100.0, n_blocks=8)
        for block in range(8):
            x = torch.from_numpy(rng.uniform(0.0, 100.0, 2000))
            fold.add(x, torch.ones_like(x), block)
        w = fold.window(0.0, 100.0)
        assert w.coherence < 0.05
        assert w.se > 0.1

    def test_the_error_falls_with_more_data(self):
        def fold_of(n: int) -> float:
            rng = np.random.default_rng(2)
            fold = PhaseFold(1.0, 0.0, 100.0, n_blocks=16)
            for block in range(16):
                x = rng.integers(1, 99, n) + 0.2 + rng.normal(0, 0.1, n)
                x = torch.from_numpy(x)
                fold.add(x, torch.ones_like(x), block)
            return fold.window(0.0, 100.0).se

        assert fold_of(4000) < fold_of(40) / 4

    def test_it_gives_the_shift_finetune_reads_from_its_fft(self):
        """FineTune's map: 0.009995454567 m/z bins, a 2**18-point FFT, the bin
        nearest the lattice's period, and the shift from the difference of the
        run's phase and the theoretical map's at that bin. The bin is not
        exactly at the period, which is why it is only ever a difference."""
        period = 1.00045475
        width = 0.009995454567
        shift = -0.17
        rng = np.random.default_rng(4)
        theory = rng.integers(400, 800, 5000) * period + rng.normal(0, 0.04, 5000)
        observed = theory + shift
        y = rng.uniform(1.0, 10.0, 5000)

        n = 1 << 18
        k = int(round(n * width / period))

        def phase_at_bin(x: np.ndarray) -> float:
            binned = np.zeros(n)
            np.add.at(binned, np.floor(x / width).astype(int), y)
            return float(np.angle(np.fft.rfft(binned)[k]))

        # np.fft sums e^(-iwx), so a shift to higher m/z lowers its phase.
        difference = phase_at_bin(theory) - phase_at_bin(observed)
        literal = math.remainder(difference, 2 * math.pi) * period / (2 * math.pi)

        run = PhaseFold(period, 300.0, 900.0)
        run.add(torch.from_numpy(observed), torch.from_numpy(y))
        reference = PhaseFold(period, 300.0, 900.0)
        reference.add(torch.from_numpy(theory), torch.from_numpy(y))
        direct = run.window(300.0, 900.0, reference=reference.coefficient(300.0, 900.0)).phase
        direct = direct * period / (2 * math.pi)

        assert direct == pytest.approx(shift, abs=1e-6)
        assert literal == pytest.approx(shift, abs=0.01)  # what binning loses
        assert direct == pytest.approx(literal, abs=0.01)


class TestEstimate:
    def _shift(self, mz):
        return -0.12 - 0.08 * np.exp(-(np.asarray(mz) - 200.0) / 200.0)

    @pytest.mark.parametrize("profile", [False, True])
    def test_it_finds_a_curved_shift_from_points_or_peaks(self, tmp_path, profile):
        bundle = _run(tmp_path, "run.profile" if profile else "run", self._shift, profile=profile)
        axis = estimate_mz_axis(bundle, settings=SETTINGS, reference=_reference())
        xs, ys = axis.knots(2)
        assert xs.size > 20
        assert np.max(np.abs(ys - self._shift(xs))) < 0.02
        xs, ys = axis.knots(1)
        assert xs.size > 5
        assert np.max(np.abs(ys - self._shift(xs))) < 0.03

    def test_it_recovers_past_half_a_period_on_ms2_and_takes_ms1_from_it(self, tmp_path):
        """+0.3 Th: outside MS1's own +-0.25 Th, inside MS2's +-0.5 Th."""
        bundle = _run(tmp_path, "run", lambda mz: 0.3 + 0 * mz, profile=False)
        axis = estimate_mz_axis(bundle, settings=SETTINGS, reference=_reference())
        assert np.median(axis.knots(2)[1]) == pytest.approx(0.3, abs=0.01)
        assert np.median(axis.knots(1)[1]) == pytest.approx(0.3, abs=0.02)
        assert axis.parameters["ambiguous_levels"] == []

    def test_ms1_alone_is_reported_as_known_only_within_its_period(self, tmp_path):
        rng = np.random.default_rng(5)
        scans = []
        for number in range(1, 31):
            mz, intensity = _peaks(rng, 400.0, 1200.0, PERIOD / 2, 400)
            scans.append({
                "scan": number, "level": 1, "rt": float(number), "iit": 1.0, "mz": mz + 0.3,
                "intensity": intensity, "filter_string": "ITMS + c NSI r Full ms [400.0000-1200.0000]",
            })
        bundle = write_bundle_manifest(write_bundle(tmp_path, "ms1", scans, row_group_size=4096))
        axis = estimate_mz_axis(bundle, settings=MzAxisSettings(min_scans={1: 12, 2: 0}, n_blocks=8, target_se_th=0.02), reference=_reference())
        assert axis.parameters["ambiguous_levels"] == [1]
        assert any("half a period" in line for line in describe(axis))

    def test_a_window_with_nothing_to_say_is_left_out_and_interpolated(self, tmp_path):
        """Noise above 1000 m/z has no phase to read."""
        rng = np.random.default_rng(6)
        scans = []
        for number in range(1, 61):
            mz, intensity = _peaks(rng, 300.0, 1000.0, PERIOD, 400)
            noise = rng.uniform(1000.0, 1400.0, 400)
            all_mz = np.concatenate([mz - 0.1, noise])
            order = np.argsort(all_mz)
            scans.append({
                "scan": number, "level": 2, "rt": float(number), "iit": 1.0, "mz": all_mz[order],
                "intensity": np.concatenate([intensity, rng.uniform(1e3, 1e5, 400)])[order],
                "filter_string": "ITMS + c NSI r Full ms2 700.0@hcd24.00 [200.0000-1500.0000]",
                "isolation_lower": 700.0, "isolation_upper": 708.0, "precursor_mz": 704.0,
            })
        bundle = write_bundle_manifest(write_bundle(tmp_path, "gap", scans, row_group_size=4096))
        axis = estimate_mz_axis(
            bundle, settings=MzAxisSettings(min_scans={1: 0, 2: 50}, n_blocks=8, target_se_th=0.01, max_half_window_th=40.0),
            reference=_reference(),
        )
        xs, ys = axis.knots(2)
        assert xs.max() < 1100.0
        assert np.max(np.abs(ys + 0.1)) < 0.02
        # Past the last knot the shift continues flat.
        assert axis.shift(2, np.array([1400.0]))[0] == pytest.approx(ys[-1])

    def test_it_says_what_a_straight_line_through_it_comes_to(self, tmp_path):
        bundle = _run(tmp_path, "run", lambda mz: 2e-4 * (np.asarray(mz) - 800.0), profile=False)
        axis = estimate_mz_axis(bundle, settings=SETTINGS, reference=_reference())
        assert scale_error_ppm(axis, 2) == pytest.approx(200.0, abs=40.0)

    def test_the_packaged_reference_loads(self):
        coefficients, meta = load_reference()
        assert meta["period"] == pytest.approx(PERIOD)
        assert np.abs(coefficients["fragment_1plus"]).sum() > 0
        assert np.abs(coefficients["precursor_2plus"]).sum() > 0


class TestUse:
    @pytest.fixture
    def run(self, tmp_path) -> tuple[Path, Path]:
        bundle = _run(tmp_path, "run", lambda mz: -0.15 + 0 * mz, profile=False)
        axis = estimate_mz_axis(bundle, settings=SETTINGS, reference=_reference())
        return bundle, save_mz_axis(bundle, axis)

    def test_it_is_stored_beside_the_points_and_found_again(self, run):
        bundle, path = run
        assert path.parent.name == "mz_axes"
        assert list_mz_axes(bundle) == [path.stem]
        again = load_mz_axis(bundle, LATEST)
        assert again.key == path.stem and again.table.num_rows > 0
        assert load_mz_axis(bundle, path.stem).parameters == again.parameters
        assert pq.read_schema(path).metadata[b"schema_name"] == b"MzAxis"

    def test_no_curve_and_an_unknown_one_differ(self, tmp_path, run):
        bundle, _ = run
        other = _run(tmp_path, "other", lambda mz: 0 * mz, profile=False, n_scans=20)
        assert load_mz_axis(other, LATEST) is None
        with pytest.raises(ValueError, match="no m/z axis"):
            load_mz_axis(bundle, "deadbeef")

    def test_a_stored_curve_changes_nothing_until_it_is_named(self, run):
        bundle, _ = run
        plain = spectrum(bundle, 2)
        fixed = spectrum(bundle, 2, mz_axis=LATEST)
        assert plain.schema.names == fixed.schema.names == ["mz", "intensity"]
        assert fixed.column("intensity").equals(plain.column("intensity"))
        shift = plain.column("mz").to_numpy() - fixed.column("mz").to_numpy()
        assert np.median(shift) == pytest.approx(-0.15, abs=0.02)
        # The bundle's own values are untouched.
        assert spectrum(bundle, 2).equals(plain)

    def test_with_no_curve_latest_leaves_m_z_alone(self, tmp_path):
        bundle = _run(tmp_path, "bare", lambda mz: 0 * mz, profile=False, n_scans=12)
        assert spectrum(bundle, 2, mz_axis=LATEST).equals(spectrum(bundle, 2))

    def test_m_z_stays_increasing_past_the_last_knot(self, run):
        bundle, _ = run
        axis = load_mz_axis(bundle)
        beyond = np.array([50.0, 100.0, 1800.0, 1900.0, 2000.0])
        assert np.all(np.diff(axis.corrected(2, beyond)) > 0)

    def test_observed_undoes_corrected(self, run):
        bundle, _ = run
        axis = load_mz_axis(bundle)
        mz = np.array([300.0, 500.0, 900.0])
        assert axis.observed(2, axis.corrected(2, mz)) == pytest.approx(mz, abs=1e-3)

    def test_an_xic_looks_where_the_instrument_reads_a_true_m_z(self, run):
        bundle, _ = run
        peak = spectrum(bundle, 2).column("mz")[10].as_py()
        common = dict(level=2, tolerance=0.02, tolerance_unit="Da")
        axis = load_mz_axis(bundle)
        true_mz = float(axis.corrected(2, np.array([peak]))[0])
        plain = xic(bundle, [peak], **common)
        fixed = xic(bundle, [true_mz], mz_axis=LATEST, **common)
        assert fixed.column("mz_target")[0].as_py() == true_mz
        assert fixed.column("intensity").to_pylist() == plain.column("intensity").to_pylist()
        at = plain.column("scan").to_pylist().index(2)
        assert plain.column("mz_observed").to_pylist()[at] == peak
        assert fixed.column("mz_observed").to_pylist()[at] == pytest.approx(true_mz)


def test_the_command_measures_and_stores(tmp_path, capsys, monkeypatch) -> None:
    from constellation.massspec import mz_axis as module
    from constellation.massspec.cli.mz_axis import _cmd_massspec_mz_axis

    bundle = _run(tmp_path, "run", lambda mz: -0.1 + 0 * mz, profile=False)

    class Args:
        reference = None
        target_se = 0.02
        dry_run = False

    Args.bundle = bundle
    monkeypatch.setattr(module, "load_reference", lambda path=None: _reference())
    real = module.MzAxisSettings
    monkeypatch.setattr(module, "MzAxisSettings", lambda: real(min_scans={1: 6, 2: 50}, n_blocks=8))
    assert _cmd_massspec_mz_axis(Args()) == 0
    out = capsys.readouterr().out
    assert "MS2:" in out and "wrote" in out
    assert list_mz_axes(bundle)
    assert json.loads(pa.parquet.read_schema(bundle / "mz_axes" / f"{list_mz_axes(bundle)[0]}.parquet").metadata[b"x.massspec.mz_axis"])["method"] == "mass_lattice"


class TestSummarizingABundle:
    """``massspec summarize BUNDLE``: what it holds, the run's other bundles,
    and each peak list's curve against the profile's."""

    @staticmethod
    def _family(tmp: Path) -> tuple[Path, Path]:
        profile = _run(tmp, "run.profile", lambda mz: -0.10 + 0 * mz, profile=True)
        peaks = _run(tmp, "run", lambda mz: -0.13 + 0 * mz, profile=False)
        return profile, peaks

    def test_it_says_what_the_bundle_holds_and_names_the_run_s_others(self, tmp_path) -> None:
        profile, peaks = self._family(tmp_path)
        summary = summarize_bundle(peaks)
        assert summary["name"] != "profile" and not summary["holds_profile"]
        assert sum(row["n_scans"] for row in summary["levels"]) == 60
        by_name = {b["name"]: b for b in summary["bundles"]}
        assert by_name["profile"]["path"] == str(profile.resolve())
        assert by_name["profile"]["holds_profile"] and not by_name["profile"]["this"]
        assert by_name[summary["name"]]["this"]
        assert summary["bundles"][0]["holds_profile"], "the profile comes first"
        assert summary["mz_axis"] == []

    def test_with_no_curves_it_names_the_command_rather_than_measuring(self, tmp_path) -> None:
        profile, peaks = self._family(tmp_path)
        text = format_bundle_summary(summarize_bundle(peaks))
        assert f"massspec mz-axis {profile.resolve()}" in text
        assert f"massspec mz-axis {peaks.resolve()}" in text
        assert not list_mz_axes(peaks), "summarizing measures nothing"

    def test_a_peak_list_s_bias_is_its_curve_minus_the_profile_s(self, tmp_path) -> None:
        profile, peaks = self._family(tmp_path)
        reference = _reference()
        for bundle in (profile, peaks):
            save_mz_axis(bundle, estimate_mz_axis(bundle, settings=SETTINGS, reference=reference))
        summary = summarize_bundle(peaks)
        ms2 = next(entry for entry in summary["mz_axis"] if entry["level"] == 2)
        lists = {item["name"]: item for item in ms2["lists"]}
        assert ms2["lists"][0]["name"] == "profile"
        bias = [row["bias_th"] for row in lists[summary["name"]]["bins"].values()]
        assert np.median(bias) == pytest.approx(-0.03, abs=0.01)
        assert all(row["bias_th"] is None for row in lists["profile"]["bins"].values())
        assert "bias" in format_bundle_summary(summary)
        json.dumps(summary)

    def test_a_list_has_no_bias_past_the_end_of_the_profile_s_curve(self, tmp_path) -> None:
        profile, peaks = self._family(tmp_path)
        reference = _reference()
        for bundle in (profile, peaks):
            save_mz_axis(bundle, estimate_mz_axis(bundle, settings=SETTINGS, reference=reference))
        table = pq.read_table(profile / "mz_axes" / f"{list_mz_axes(profile)[0]}.parquet")
        short = table.filter(pa.compute.less(table.column("mz"), 500.0))
        pq.write_table(short, profile / "mz_axes" / f"{list_mz_axes(profile)[0]}.parquet")
        summary = summarize_bundle(peaks)
        ms2 = next(entry for entry in summary["mz_axis"] if entry["level"] == 2)
        rows = next(item for item in ms2["lists"] if item["name"] == summary["name"])["bins"]
        assert rows["1000"]["bias_th"] is None and rows["1000"]["shift_th"] is not None

    def test_the_verb_takes_a_bundle_and_refuses_follow(self, tmp_path, capsys) -> None:
        from constellation.massspec.cli.summarize import _cmd_massspec_summarize

        _profile, peaks = self._family(tmp_path)

        class Args:
            experiment_dir = peaks
            json = True
            follow = False

        assert _cmd_massspec_summarize(Args()) == 0
        assert json.loads(capsys.readouterr().out)["path"] == str(peaks.resolve())
        Args.follow = True
        assert _cmd_massspec_summarize(Args()) == 2
        assert "--follow" in capsys.readouterr().err

    def test_an_older_bundle_is_refused_with_its_message(self, tmp_path, capsys) -> None:
        from constellation.massspec.cli.summarize import _cmd_massspec_summarize

        _profile, peaks = self._family(tmp_path)
        manifest = json.loads((peaks / "manifest.json").read_text())
        manifest["schema_version"] = 2
        (peaks / "manifest.json").write_text(json.dumps(manifest))

        class Args:
            experiment_dir = peaks
            json = False
            follow = False

        assert _cmd_massspec_summarize(Args()) == 2
        assert "convert" in capsys.readouterr().err
