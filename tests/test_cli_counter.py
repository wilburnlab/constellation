"""Smoke tests for the `massspec counter {calibrate, estimate}` CLI.

Synthetic only: a simulated observation → `observation_to_trace` gives the trace +
scan-metadata parquet inputs; a default calibration + a tiny target table complete
the contract. Exercises the spawn worker pool (`--workers 2`).
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.cli import build_parser
from constellation.massspec.counter import (
    GlobalCalibration,
    Progenitor,
    calibration_to_table,
    observation_to_trace,
    simulate_observation,
)
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE

_SEQ = "PEPTIDEKR"


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="constellation")
    subs = parser.add_subparsers(dest="command", required=True)
    build_parser(subs)
    args = parser.parse_args(argv)
    return args.func(args)


def _inputs(tmp_path):
    cal = GlobalCalibration(n_isotopes=3, charges=[2, 3], alpha0=1.0, alpha1=15.0,
                            alpha_mz=0.6, nu_mz=7.8)
    truth = Progenitor.for_peptide(
        Peptidoform(sequence=_SEQ), [2], cal, n_isotopes=3,
        peak=HyperEMGPeak(N_total=2e5, mu=1.8e6, sigma=5000.0, tau_r=4000.0, tau_l=2000.0, eta=0.8),
        nu_intensity=6.0, c_mz_init=200.0,
    )
    obs = simulate_observation(truth, n_scans=80, half_window_ms=45000.0, iit_ms=20.0,
                               generator=torch.Generator().manual_seed(0))
    trace, scan_meta = observation_to_trace(obs, acquisition_id=0, target_id=0, modified_sequence=_SEQ)
    pq.write_table(trace, tmp_path / "trace.parquet")
    pq.write_table(scan_meta, tmp_path / "scan_meta.parquet")
    pq.write_table(calibration_to_table(cal, acquisition_id=0), tmp_path / "cal.parquet")
    tgt = cast_to_schema(
        pa.table({"target_id": [0], "modified_sequence": [_SEQ],
                  "precursor_charge": [2], "rt_center": [1800.0]}),
        XIC_TARGET_TABLE,
    )
    pq.write_table(tgt, tmp_path / "targets.parquet")
    return tmp_path


def test_cli_counter_estimate_worker_pool(tmp_path):
    p = _inputs(tmp_path)
    out = p / "est"
    rc = _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets.parquet"),
        "-o", str(out), "--workers", "2", "--no-progress",
    ])
    assert rc == 0
    assert (out / "_SUCCESS").exists()
    n = pq.read_table(out / "counter_n.parquet")
    assert n.num_rows == 1
    row = n.to_pylist()[0]
    assert row["n_total"] > 0 and row["interference_flag"] is False
    assert {"target_id", "n_total", "n_total_lo", "n_total_hi", "rt_apex"} <= set(n.column_names)


def test_cli_counter_estimate_component_cofit(tmp_path):
    # two co-isobaric (same peptide+charge), co-eluting targets → ONE co-fit component
    # → one record per member (exercises the multi-member component worker path).
    p = _inputs(tmp_path)
    tgt = cast_to_schema(
        pa.table({"target_id": [0, 1], "modified_sequence": [_SEQ, _SEQ],
                  "precursor_charge": [2, 2], "rt_center": [1800.0, 1800.0]}),
        XIC_TARGET_TABLE,
    )
    pq.write_table(tgt, p / "targets2.parquet")
    out = p / "est2"
    rc = _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets2.parquet"),
        "-o", str(out), "--workers", "1", "--no-progress",
    ])
    assert rc == 0
    n = pq.read_table(out / "counter_n.parquet")
    assert n.num_rows == 2  # one record per component member
    rows = n.to_pylist()
    assert {r["target_id"] for r in rows} == {0, 1}
    assert all(r["n_total"] > 0 for r in rows)
    assert all(r["interference_flag"] is True for r in rows)  # co-isobaric company


def test_cli_counter_estimate_component_spawn_pool(tmp_path):
    # the load-bearing combination: multi-member component co-fit ACROSS the spawn Pool
    # (a unit list[dict] + multi-record return must pickle round-trip).
    p = _inputs(tmp_path)
    tgt = cast_to_schema(
        pa.table({"target_id": [0, 1], "modified_sequence": [_SEQ, _SEQ],
                  "precursor_charge": [2, 2], "rt_center": [1800.0, 1800.0]}),
        XIC_TARGET_TABLE,
    )
    pq.write_table(tgt, p / "targets2.parquet")
    out = p / "est_spawn"
    rc = _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets2.parquet"),
        "-o", str(out), "--workers", "2", "--no-progress",
    ])
    assert rc == 0
    n = pq.read_table(out / "counter_n.parquet")
    assert n.num_rows == 2 and {r["target_id"] for r in n.to_pylist()} == {0, 1}


def test_cli_counter_rejects_duplicate_target_ids(tmp_path):
    p = _inputs(tmp_path)
    tgt = cast_to_schema(
        pa.table({"target_id": [0, 0], "modified_sequence": [_SEQ, _SEQ],
                  "precursor_charge": [2, 3], "rt_center": [1800.0, 1800.0]}),
        XIC_TARGET_TABLE,
    )
    pq.write_table(tgt, p / "dup.parquet")
    with pytest.raises(SystemExit, match="duplicate target_id"):
        _run([
            "massspec", "counter", "estimate",
            "--trace", str(p / "trace.parquet"),
            "--scan-metadata", str(p / "scan_meta.parquet"),
            "--calibration", str(p / "cal.parquet"),
            "--targets", str(p / "dup.parquet"),
            "-o", str(p / "dup_out"), "--no-progress",
        ])


def test_resolve_collide_ppm_clamps_to_extraction_tolerance(tmp_path):
    from constellation.massspec.cli import _resolve_collide_ppm
    from constellation.massspec.quant.chromatogram import _stamp_extraction_tolerance

    p = _inputs(tmp_path)
    trace = pq.read_table(p / "trace.parquet")
    pq.write_table(_stamp_extraction_tolerance(trace, 10.0, "ppm"), p / "trace10.parquet")

    def _args(trace_path, collide, override=None):
        return argparse.Namespace(
            trace=trace_path, collide_ppm=collide,
            extraction_tolerance_ppm=override, no_progress=True,
        )

    # collide above the recorded tolerance → clamped to it
    assert _resolve_collide_ppm(_args(p / "trace10.parquet", 20.0)) == 10.0
    # collide already within tolerance → unchanged
    assert _resolve_collide_ppm(_args(p / "trace10.parquet", 5.0)) == 5.0
    # explicit override wins over the recorded value
    assert _resolve_collide_ppm(_args(p / "trace10.parquet", 20.0, override=8.0)) == 8.0
    # trace without recorded tolerance (observation_to_trace) → left as-is
    assert _resolve_collide_ppm(_args(p / "trace.parquet", 20.0)) == 20.0


def test_resolve_collide_ppm_reads_bundle_directory(tmp_path):
    # the canonical `chromatogram extract` output is a BUNDLE DIR (xic_trace.parquet +
    # manifest.json) — the clamp must read the parquet inside it, not fail on the dir.
    from constellation.massspec.cli import _resolve_collide_ppm
    from constellation.massspec.quant.chromatogram import _stamp_extraction_tolerance, save_xic

    p = _inputs(tmp_path)
    bundle = p / "trace_bundle"
    save_xic(_stamp_extraction_tolerance(pq.read_table(p / "trace.parquet"), 10.0, "ppm"), bundle)
    args = argparse.Namespace(
        trace=bundle, collide_ppm=20.0, extraction_tolerance_ppm=None, no_progress=True
    )
    assert _resolve_collide_ppm(args) == 10.0


def test_cli_counter_estimate_on_bundle_directory(tmp_path):
    # end-to-end on a bundle dir: the worker's load_xic must read it too.
    from constellation.massspec.quant.chromatogram import save_xic

    p = _inputs(tmp_path)
    save_xic(pq.read_table(p / "trace.parquet"), p / "trace_bundle")
    out = p / "est_bundle"
    rc = _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace_bundle"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets.parquet"),
        "-o", str(out), "--workers", "1", "--no-progress",
    ])
    assert rc == 0
    assert pq.read_table(out / "counter_n.parquet").num_rows == 1


def test_cli_counter_estimate_emit_attribution(tmp_path):
    p = _inputs(tmp_path)
    out = p / "est_attr"
    rc = _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets.parquet"),
        # --workers 2: the attribution pa.Table must round-trip through the spawn Pool
        "-o", str(out), "--workers", "2", "--emit-attribution", "--no-progress",
    ])
    assert rc == 0
    attr = pq.read_table(out / "peak_attribution.parquet")
    assert attr.num_rows > 0
    assert {"scan", "mz_observed", "progenitor_index", "responsibility", "target_id"} <= set(
        attr.column_names
    )
    assert set(attr.column("target_id").to_pylist()) == {0}
    # the stable physical-peak key is populated for a real (round-tripped) extraction
    import pyarrow.compute as pc

    assert pc.min(attr.column("scan")).as_py() >= 0
    assert pc.all(pc.is_finite(attr.column("mz_observed"))).as_py()

    # default (no flag) → no attribution file written
    out2 = p / "est_noattr"
    _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets.parquet"),
        "-o", str(out2), "--workers", "1", "--no-progress",
    ])
    assert not (out2 / "peak_attribution.parquet").exists()


def test_cli_counter_emit_attribution_empty_still_writes_file(tmp_path):
    # P2 #3: --emit-attribution must always write a schema-correct file, even when no
    # target produces a row (here: an RT-center far from any scan → all no_signal).
    p = _inputs(tmp_path)
    tgt = cast_to_schema(
        pa.table({"target_id": [0], "modified_sequence": [_SEQ],
                  "precursor_charge": [2], "rt_center": [9.0e6]}),  # nowhere near the trace
        XIC_TARGET_TABLE,
    )
    pq.write_table(tgt, p / "targets_far.parquet")
    out = p / "est_empty_attr"
    rc = _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets_far.parquet"),
        "-o", str(out), "--workers", "1", "--emit-attribution", "--no-progress",
    ])
    assert rc == 0
    from constellation.massspec.counter import COUNTER_PEAK_ATTRIBUTION_TABLE

    assert (out / "peak_attribution.parquet").exists()  # requested → present, not omitted
    attr = pq.read_table(out / "peak_attribution.parquet")
    assert attr.num_rows == 0
    assert attr.schema.equals(COUNTER_PEAK_ATTRIBUTION_TABLE)


def test_cli_counter_component_attribution_member_identities(tmp_path):
    # P1 #2 end-to-end: a two-member co-fit component's attribution rows carry BOTH
    # members' target_ids (each is_target=True), not just the reference's.
    p = _inputs(tmp_path)
    tgt = cast_to_schema(
        pa.table({"target_id": [0, 1], "modified_sequence": [_SEQ, _SEQ],
                  "precursor_charge": [2, 2], "rt_center": [1800.0, 1800.0]}),
        XIC_TARGET_TABLE,
    )
    pq.write_table(tgt, p / "targets2.parquet")
    out = p / "est_comp_attr"
    rc = _run([
        "massspec", "counter", "estimate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibration", str(p / "cal.parquet"),
        "--targets", str(p / "targets2.parquet"),
        "-o", str(out), "--workers", "1", "--emit-attribution", "--no-progress",
    ])
    assert rc == 0
    attr = pq.read_table(out / "peak_attribution.parquet")
    rows = attr.to_pylist()
    # both members appear as real targets (the reference 0 AND the co-member 1)
    targets_flagged = {r["target_id"] for r in rows if r["is_target"]}
    assert {0, 1} <= targets_flagged


def test_counter_cofit_units_partition_and_size_cap():
    # the parent-side partitioner: two co-isobaric, co-eluting targets group into one
    # component; the size cap splits an oversized component back to singletons.
    from constellation.massspec.cli import _counter_cofit_units

    targets = [
        {"target_id": 0, "modified_sequence": _SEQ, "precursor_charge": 2, "rt_center": 1800.0},
        {"target_id": 1, "modified_sequence": _SEQ, "precursor_charge": 2, "rt_center": 1800.0},
    ]
    opts = dict(n_isotopes=3, collide_ppm=20.0, rt_overlap_s=60.0, max_component_size=2)
    units, n_capped = _counter_cofit_units(targets, opts)
    assert n_capped == 0 and len(units) == 1 and len(units[0]) == 2  # one 2-member component

    units2, capped2 = _counter_cofit_units(targets, {**opts, "max_component_size": 1})
    assert capped2 == 1 and len(units2) == 2 and all(len(u) == 1 for u in units2)  # split

    # RT-separated targets never co-fit even though they are co-isobaric
    far = [targets[0], {**targets[1], "rt_center": 3600.0}]
    units3, _ = _counter_cofit_units(far, opts)
    assert len(units3) == 2 and all(len(u) == 1 for u in units3)


def test_cli_counter_calibrate(tmp_path):
    p = _inputs(tmp_path)
    out = p / "cal_out"
    rc = _run([
        "massspec", "counter", "calibrate",
        "--trace", str(p / "trace.parquet"),
        "--scan-metadata", str(p / "scan_meta.parquet"),
        "--calibrants", str(p / "targets.parquet"),
        "-o", str(out), "--no-progress",
    ])
    assert rc == 0
    assert (out / "_SUCCESS").exists()
    assert (out / "global_calibration.parquet").exists()
    assert (out / "peptide_params.parquet").exists()
