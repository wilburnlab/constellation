"""The scan-major XIC extractor against the frozen copy of its original
implementation (`tests/_xic_scan_major_oracle.py`).

These pass trivially until the sweep replaces the extractor's internals;
from then on they are what says the replacement changed nothing: the same
rows, in the same order, with the same NaN positions and float bits, over
planted DIA, staggered DIA and DDA runs and every option the extractor
has. The two behaviours the sweep changes on purpose (a missing precursor
m/z, the assigned-scan restriction at MS1) are left out here and tested
where they change.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import _xic_scan_major_oracle as oracle
from _massspec_fixtures import planted_run, write_bundle, write_bundle_manifest
from constellation.massspec.bundle import open_peaks
from constellation.massspec.cli import build_parser
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.quant.chromatogram import extract_xic_scan_major
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE

#: Small enough for the option matrix to stay quick, big enough to hold every case.
_RUNS = {
    "dia": dict(acquisition="dia", cycles=40, window_width=100.0),
    "staggered": dict(acquisition="dia", staggered=True, cycles=40, window_width=100.0),
    "dda": dict(acquisition="dda", cycles=40),
}


def _reversed(modseq: str) -> str:
    return modseq[-2::-1] + modseq[-1] if "[" not in modseq else modseq


@functools.lru_cache(maxsize=None)
def _run(name: str, tmp_root: str):
    run = planted_run(**_RUNS[name])
    bundle = write_bundle(Path(tmp_root), name, run.scans, row_group_size=97)
    write_bundle_manifest(bundle)
    return run, bundle, open_peaks(bundle).to_table()


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("parity"))


def _targets(run, *, level: int, name: str) -> pa.Table:
    """The planted targets, a reversed (unplanted) copy of each, one with an
    explicit RT range, and at MS1 a bare-m/z target."""
    rows = run.targets.to_pylist()
    decoys = []
    for r in rows:
        if "[" in r["modified_sequence"]:
            continue
        d = dict(r, target_id=r["target_id"] + 500, modified_sequence=_reversed(r["modified_sequence"]))
        decoys.append(d)
    rows = rows + decoys
    rows[1] = dict(rows[1], rt_start=rows[1]["rt_center"] - 7.0, rt_end=rows[1]["rt_center"] + 3.0)
    if level == 1:
        rows.append(dict(rows[0], target_id=9999, modified_sequence=None, rt_center=None))
    if name == "dda":
        ms2 = [s for s in run.scans if s["level"] == 2]
        for r in rows:
            if r["modified_sequence"] is None:
                continue
            hits = [s for s in ms2 if s["isolation_lower"] <= r["precursor_mz"] <= s["isolation_upper"]]
            if hits and r["rt_center"] is not None:
                r["scan"] = min(hits, key=lambda s: abs(s["rt"] - r["rt_center"]))["scan"]
    return pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)


def _both(peaks, targets, **kw):
    got = extract_xic_scan_major(peaks, targets, acquisition_id=7, **kw)
    want = oracle.extract_xic_scan_major(peaks, targets, acquisition_id=7, **kw)
    return got, want


_TOLERANCES = [(20.0, "ppm"), (0.35, "Da")]


@pytest.mark.parametrize("name", list(_RUNS))
@pytest.mark.parametrize("rt_window", [None, 12.0])
@pytest.mark.parametrize("tolerance", _TOLERANCES)
@pytest.mark.parametrize("mode", ["nearest", "keep_unmatched", "all_in_window"])
@pytest.mark.parametrize("charge_range", [(1, 4), None])
def test_ms1_parity(tmp_root, name, rt_window, tolerance, mode, charge_range):
    run, _, peaks = _run(name, tmp_root)
    got, want = _both(
        peaks, _targets(run, level=1, name=name), level=1, rt_window=rt_window,
        tolerance=tolerance[0], tolerance_unit=tolerance[1], charge_range=charge_range,
        all_in_window=mode == "all_in_window", drop_unmatched=mode != "keep_unmatched",
    )  # fmt: skip
    assert want.num_rows > 0
    oracle.assert_traces_identical(got, want)


@pytest.mark.parametrize("name", list(_RUNS))
@pytest.mark.parametrize("rt_window", [None, 12.0])
@pytest.mark.parametrize("tolerance", _TOLERANCES)
@pytest.mark.parametrize("mode", ["nearest", "keep_unmatched", "all_in_window"])
def test_ms2_parity(tmp_root, name, rt_window, tolerance, mode):
    run, _, peaks = _run(name, tmp_root)
    got, want = _both(
        peaks, _targets(run, level=2, name=name), level=2, rt_window=rt_window,
        tolerance=tolerance[0], tolerance_unit=tolerance[1],
        all_in_window=mode == "all_in_window", drop_unmatched=mode != "keep_unmatched",
    )  # fmt: skip
    assert want.num_rows > 0
    oracle.assert_traces_identical(got, want)


@pytest.mark.parametrize(
    "ions",
    [
        dict(ion_types=(IonType.A, IonType.B, IonType.Y), max_fragment_charge=1),
        dict(neutral_losses=("H2O", "NH3"), max_fragment_charge=3),
    ],
)
def test_ms2_ion_settings_parity(tmp_root, ions):
    run, _, peaks = _run("dia", tmp_root)
    got, want = _both(peaks, _targets(run, level=2, name="dia"), level=2, rt_window=20.0, **ions)
    assert want.num_rows > 0
    oracle.assert_traces_identical(got, want)


def test_assigned_scans_parity_at_ms2(tmp_root):
    run, _, peaks = _run("dda", tmp_root)
    targets = _targets(run, level=2, name="dda")
    assert pc.sum(pc.is_valid(targets["scan"])).as_py() > 0
    got, want = _both(peaks, targets, level=2, assigned_scans_only=True, drop_unmatched=False)
    assert want.num_rows > 0
    oracle.assert_traces_identical(got, want)


def test_rows_come_in_target_order_then_retention_time(tmp_root):
    run, _, peaks = _run("dia", tmp_root)
    targets = _targets(run, level=2, name="dia")
    trace = oracle.extract_xic_scan_major(peaks, targets, acquisition_id=0, level=2, rt_window=12.0)
    order = {t: i for i, t in enumerate(targets["target_id"].to_pylist())}
    seq = [order[t] for t in trace["target_id"].to_pylist()]
    assert seq == sorted(seq)
    for tid in set(trace["target_id"].to_pylist()):
        rt = trace.filter(pc.equal(trace["target_id"], tid))["rt"].to_pylist()
        assert rt == sorted(rt)


def test_planted_edge_and_duplicate_peaks_are_in_play(tmp_root):
    """The fixture's tolerance-edge and duplicate peaks sit where a parity
    test can see them: in a scan the first planted target is matched in."""
    run, _, peaks = _run("dia", tmp_root)
    trace = oracle.extract_xic_scan_major(
        peaks, run.targets.slice(0, 1), acquisition_id=0, level=2, all_in_window=True
    )
    scans = set(trace["scan"].to_pylist())
    assert run.edge[0] in scans and run.duplicate[0] in scans


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="constellation")
    subs = parser.add_subparsers(dest="command", required=True)
    build_parser(subs)
    args = parser.parse_args(argv)
    return args.func(args)


@pytest.mark.parametrize("level", [1, 2])
def test_cli_extract_matches_the_oracle(tmp_root, tmp_path, level):
    run, bundle, peaks = _run("dia", tmp_root)
    targets = _targets(run, level=level, name="dia")
    path = tmp_path / "targets.parquet"
    pq.write_table(targets, path)
    out = tmp_path / "xic"
    rc = _cli([
        "massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(path),
        "-o", str(out), "--level", str(level), "--acquisition-id", "7",
        "--tolerance", "0.35", "--tolerance-unit", "Da", "--rt-window", "12", "--no-progress",
    ])  # fmt: skip
    assert rc == 0
    got = pq.read_table(out / "xic_trace.parquet")
    want = oracle.extract_xic_scan_major(
        peaks, targets, acquisition_id=7, level=level, tolerance=0.35, tolerance_unit="Da",
        rt_window=12.0,
    )  # fmt: skip
    oracle.assert_traces_identical(got, want)


def test_cli_both_levels_in_one_sweep(tmp_root, tmp_path, capsys):
    import json

    run, bundle, peaks = _run("dia", tmp_root)
    targets = run.targets
    path = tmp_path / "targets.parquet"
    pq.write_table(targets, path)
    out = tmp_path / "xic"
    rc = _cli([
        "massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(path),
        "-o", str(out), "--level", "both", "--ms1-tolerance", "20", "--ms1-tolerance-unit", "ppm",
        "--ms2-tolerance", "0.35", "--ms2-tolerance-unit", "Da", "--rt-window", "12",
        "--workers", "2", "--no-progress",
    ])  # fmt: skip
    assert rc == 0
    assert "usually matched within" in capsys.readouterr().err  # ITMS MS1 at 20 ppm
    got = pq.read_table(out / "xic_trace.parquet")
    meta = got.schema.metadata
    assert meta[b"x.massspec.extraction_tolerance.ms1"] == b"20.0"
    assert meta[b"x.massspec.extraction_tolerance_unit.ms2"] == b"Da"
    assert b"x.massspec.extraction_tolerance" not in meta  # two different tolerances
    manifest = json.loads((out / "manifest.json").read_text())["metadata"]
    assert manifest["levels"] == [1, 2] and manifest["engine"] == "sweep"
    assert not (out / ".sweep").exists()
    for level, tol in ((1, (20.0, "ppm")), (2, (0.35, "Da"))):
        want = oracle.extract_xic_scan_major(
            peaks, targets, acquisition_id=0, level=level, rt_window=12.0, tolerance=tol[0],
            tolerance_unit=tol[1],
        )  # fmt: skip
        part = got.filter(pc.equal(got["level"], level))
        assert part["mz_observed"].to_pylist() == want["mz_observed"].to_pylist()


def test_cli_refuses_both_levels_from_the_index(tmp_root, tmp_path):
    run, bundle, _ = _run("dia", tmp_root)
    path = tmp_path / "targets.parquet"
    pq.write_table(run.targets, path)
    rc = _cli(["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets",
               str(path), "-o", str(tmp_path / "x"), "--level", "both", "--use-index",
               "--no-progress"])  # fmt: skip
    assert rc == 2


def test_counter_reads_the_ms1_tolerance_of_a_two_level_trace(tmp_path):
    from constellation.massspec.cli.counter import _resolve_collide_ppm

    table = pa.table({"x": [1]}).replace_schema_metadata({
        b"x.massspec.extraction_tolerance.ms1": b"8.0",
        b"x.massspec.extraction_tolerance_unit.ms1": b"ppm",
        b"x.massspec.extraction_tolerance.ms2": b"0.35",
        b"x.massspec.extraction_tolerance_unit.ms2": b"Da",
    })  # fmt: skip
    pq.write_table(table, tmp_path / "t.parquet")
    args = argparse.Namespace(extraction_tolerance_ppm=None, trace=tmp_path / "t.parquet",
                              collide_ppm=20.0, no_progress=True)  # fmt: skip
    assert _resolve_collide_ppm(args) == 8.0


def test_cli_gate_claims_and_summary(tmp_root, tmp_path):
    import json

    run, bundle, peaks = _run("dia", tmp_root)
    targets = _targets(run, level=2, name="dia")  # planted ones and reversed copies
    path = tmp_path / "targets.parquet"
    pq.write_table(targets, path)
    out = tmp_path / "xic"
    rc = _cli([
        "massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(path),
        "-o", str(out), "--level", "2", "--tolerance", "20", "--min-fragments", "4",
        "--min-consecutive-scans", "2", "--only-passing", "--claims", "--summary",
        "--workers", "1", "--no-progress",
    ])  # fmt: skip
    assert rc == 0
    trace = pq.read_table(out / "xic_trace.parquet")
    summary = pq.read_table(out / "query_summary.parquet")
    claims = pq.read_table(out / "point_claims.parquet")
    planted = set(run.targets["target_id"].to_pylist())
    assert set(trace["target_id"].to_pylist()) == planted
    passed = {t for t, p in zip(summary["target_id"].to_pylist(), summary["passed"].to_pylist()) if p}
    assert passed == planted and summary.num_rows == targets.num_rows
    assert claims.num_rows > 0
    manifest = json.loads((out / "manifest.json").read_text())["metadata"]
    assert manifest["gate"]["min_fragments"] == 4 and manifest["gate"]["only_passing"]
    assert manifest["tables"] == ["xic_trace.parquet", "point_claims.parquet", "query_summary.parquet"]
    assert _cli(["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets",
                 str(path), "-o", str(tmp_path / "y"), "--level", "2", "--only-passing",
                 "--no-progress"]) == 2  # fmt: skip


@pytest.mark.parametrize("extra", [
    ["--gate"], ["--min-fragments", "3"], ["--only-passing", "--gate"], ["--mz-axis", "nonexistent"],
    ["--claims"], ["--summary"], ["--work-dir", "WORK"],
])  # fmt: skip
def test_cli_refuses_sweep_options_with_the_index(tmp_root, tmp_path, extra, capsys):
    """The mass index applied none of these and said nothing; now it says no."""
    run, bundle, _ = _run("dia", tmp_root)
    path = tmp_path / "targets.parquet"
    pq.write_table(run.targets, path)
    extra = [str(tmp_path / "w") if a == "WORK" else a for a in extra]
    rc = _cli(["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets",
               str(path), "-o", str(tmp_path / "x"), "--level", "1", "--use-index",
               "--index-dir", str(tmp_path / "index"), "--no-progress", *extra])  # fmt: skip
    assert rc == 2
    assert "belong to the sweep" in capsys.readouterr().err
    assert not (tmp_path / "x").exists() and not (tmp_path / "index").exists()


def test_cli_takes_a_zero_tolerance_as_given(tmp_root, tmp_path):
    """`--ms1-tolerance 0` fell back to `--tolerance` (20 ppm) through `or`."""
    run, bundle, peaks = _run("dia", tmp_root)
    targets = _targets(run, level=1, name="dia")
    path = tmp_path / "targets.parquet"
    pq.write_table(targets, path)
    common = ["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(path),
              "--level", "1", "--rt-window", "12", "--no-progress"]  # fmt: skip
    assert _cli([*common, "-o", str(tmp_path / "x"), "--ms1-tolerance", "0"]) == 0
    got = pq.read_table(tmp_path / "x" / "xic_trace.parquet")
    assert got.schema.metadata[b"x.massspec.extraction_tolerance"] == b"0.0"
    want = oracle.extract_xic_scan_major(peaks, targets, acquisition_id=0, level=1, tolerance=0.0,
                                         rt_window=12.0)  # fmt: skip
    oracle.assert_traces_identical(got, want)
    wide = oracle.extract_xic_scan_major(peaks, targets, acquisition_id=0, level=1, rt_window=12.0)
    assert wide.num_rows > want.num_rows
    assert _cli([*common, "-o", str(tmp_path / "y"), "--ms1-tolerance", "-1"]) == 2
    assert _cli([*common, "-o", str(tmp_path / "z"), "--gate", "--min-consecutive-scans", "0"]) == 2
