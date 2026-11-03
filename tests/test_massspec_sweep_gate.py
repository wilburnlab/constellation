"""The evidence gate, shared-peak claims and the query summary
(`massspec.sweep.{evidence, claims}`, `QuerySummaryReducer`)."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import _xic_scan_major_oracle as oracle
from _massspec_fixtures import planted_run, write_bundle, write_bundle_manifest
from constellation.massspec.quant.chromatogram import XicTraceReducer
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep import compile_queries
from constellation.massspec.sweep.engine import CLAIMS, SweepSettings, sweep
from constellation.massspec.sweep.evidence import EvidenceGate, count_evidence
from constellation.massspec.sweep.reduce import QuerySummaryReducer
from constellation.massspec.sweep.rules import AllWithin, Nearest, SumWithin

_TOL = {1: (20.0, "ppm"), 2: (20.0, "ppm")}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("gate"))


@functools.lru_cache(maxsize=None)
def _run(tmp_root: str):
    run = planted_run(acquisition="dia", cycles=40, window_width=100.0)
    bundle = write_bundle(Path(tmp_root), "dia", run.scans, row_group_size=97)
    write_bundle_manifest(bundle)
    return run, bundle


def _with_decoys(run) -> pa.Table:
    rows = run.targets.to_pylist()
    decoys = [dict(r, target_id=r["target_id"] + 500, modified_sequence=r["modified_sequence"][-2::-1]
                   + r["modified_sequence"][-1]) for r in rows if "[" not in r["modified_sequence"]]  # fmt: skip
    return pa.Table.from_pylist(rows + decoys, schema=XIC_TARGET_TABLE)


def test_a_bond_seen_at_two_charges_counts_once():
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.peptide.mz import precursor_mz
    from constellation.massspec.sweep import IonSettings

    targets = pa.Table.from_pylist(
        [dict(target_id=1, modified_sequence="PEPTIDEK", precursor_charge=3,
              precursor_mz=precursor_mz(parse_proforma("PEPTIDEK"), charge=3))],
        schema=XIC_TARGET_TABLE,
    )  # fmt: skip
    plan = compile_queries(targets, levels=[2], ions=IonSettings(max_fragment_charge=2))
    f = plan.fragments
    # One scan holding y5 at charges 1 and 2, and b3 at charge 1.
    want = [(int(t), int(p), int(c)) for t, p, c in zip(f.ion_type, f.position, f.fragment_charge)]
    pick = [i for i, (t, p, c) in enumerate(want) if (t == 4 and p == 2) or (t == 1 and p == 2 and c == 1)]
    mz = np.sort(f.mz[pick])
    ev = count_evidence(
        mz, np.ones(mz.size), np.array([0, mz.size]), np.array([0]), np.array([10]), f,
        np.arange(f.n), np.zeros(f.n, np.int64), lambda q, r: np.ones(q.size, bool), 20.0, "ppm", None,
        by_charge=False,
    )  # fmt: skip
    assert ev.n == 1
    assert ev.n_ions.tolist() == [len(pick)]
    assert ev.count().tolist() == [2]  # two bonds, though three ions matched


@pytest.mark.parametrize("rule", [Nearest(), SumWithin()])
def test_planted_peptides_pass_and_reversed_ones_do_not(tmp_root, rule):
    run, bundle = _run(tmp_root)
    targets = _with_decoys(run)
    plan = compile_queries(targets, levels=[1, 2])
    gate = EvidenceGate(min_fragments=4, min_consecutive=2)
    result = sweep(bundle, plan, SweepSettings(tolerances=_TOL, rule=rule, gate=gate),
                   reducers=[QuerySummaryReducer()])  # fmt: skip
    summary = result.table("query_summary")
    passed = dict(zip(summary["target_id"].to_pylist(), summary["passed"].to_pylist()))
    planted = set(run.targets["target_id"].to_pylist())
    assert all(passed[t] for t in planted)
    assert not any(passed[t] for t in passed if t not in planted)
    best = dict(zip(summary["target_id"].to_pylist(), summary["best_rt"].to_pylist()))
    apex = dict(zip(run.targets["target_id"].to_pylist(), run.targets["rt_center"].to_pylist()))
    for t in planted:
        assert abs(best[t] - apex[t]) < 10.0


def test_gated_reducers_see_only_passing_queries(tmp_root):
    run, bundle = _run(tmp_root)
    plan = compile_queries(_with_decoys(run), levels=[2])
    gate = EvidenceGate(min_fragments=4, min_consecutive=2)
    settings = SweepSettings(tolerances=_TOL, gate=gate)
    everything = sweep(bundle, plan, settings, reducers=[XicTraceReducer()]).table("xic_trace")
    only = sweep(bundle, plan, settings,
                 reducers=[XicTraceReducer(name="gated", gated=True)]).table("gated")  # fmt: skip
    planted = run.targets["target_id"]
    assert set(only["target_id"].to_pylist()) == set(planted.to_pylist())
    oracle.assert_traces_identical(only, everything.filter(pc.is_in(everything["target_id"], planted)))


def test_the_gate_reaches_across_shards_and_workers(tmp_root, tmp_path):
    run, bundle = _run(tmp_root)
    plan = compile_queries(_with_decoys(run), levels=[1, 2])
    gate = EvidenceGate(min_fragments=3, min_consecutive=3, top_peaks=10)
    base = sweep(bundle, plan, SweepSettings(tolerances=_TOL, gate=gate, claims=True),
                 reducers=[QuerySummaryReducer()])  # fmt: skip
    small = SweepSettings(tolerances=_TOL, gate=gate, claims=True, shard_rows=2000, reduce_rows=300,
                          run_rows=200, block_rows=200)  # fmt: skip
    for workers in (1, 3):
        got = sweep(bundle, plan, small, reducers=[QuerySummaryReducer()], workers=workers,
                    work_dir=tmp_path / f"w{workers}")  # fmt: skip
        assert got.table("query_summary").equals(base.table("query_summary"))
        assert got.table(CLAIMS).equals(base.table(CLAIMS))


def test_claims_record_near_neighbours_and_duplicates(tmp_root):
    run, bundle = _run(tmp_root)
    rows = run.targets.to_pylist()
    rows.append(dict(rows[4], target_id=4444))  # HLVDEPQNLIK twice
    targets = pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)
    plan = compile_queries(targets, levels=[2], rt_window=20.0)
    result = sweep(bundle, plan, SweepSettings(tolerances=_TOL, rule=AllWithin(), claims=True),
                   reducers=[XicTraceReducer(), QuerySummaryReducer()])  # fmt: skip
    claims = result.table(CLAIMS)
    assert claims.num_rows > 0
    by_peak = {}
    for scan, peak, tid in zip(claims["scan"].to_pylist(), claims["peak_index"].to_pylist(),
                               claims["target_id"].to_pylist()):  # fmt: skip
        by_peak.setdefault((scan, peak), set()).add(tid)
    # Every peak the duplicate matched is claimed by both copies.
    trace = result.table("xic_trace")
    dup = trace.filter(pc.equal(trace["target_id"], 4444))
    assert dup.num_rows > 0
    claimed_pairs = {k for k, v in by_peak.items() if {1004, 4444} <= v}
    assert len(claimed_pairs) == dup.num_rows
    # PEPTMIDEK and its oxidised form share y1–y3 and co-elute in one window.
    assert any({1002, 1003} <= v for v in by_peak.values())
    summary = result.table("query_summary")
    shared = dict(zip(summary["target_id"].to_pylist(), summary["n_shared_matches"].to_pylist()))
    assert shared[4444] == dup.num_rows and shared[1004] == dup.num_rows
    assert shared[1000] == 0
    # The table's columns are filled from the plan.
    row = claims.filter(pc.equal(claims["target_id"], 4444)).slice(0, 1).to_pylist()[0]
    assert row["modified_sequence"] == "HLVDEPQNLIK" and row["isotope"] is None
    assert abs(row["mz_error_ppm"]) < 20.0


def test_summary_counts_agree_with_the_trace(tmp_root):
    run, bundle = _run(tmp_root)
    plan = compile_queries(run.targets, levels=[1, 2], rt_window=15.0)
    result = sweep(bundle, plan, SweepSettings(tolerances=_TOL),
                   reducers=[XicTraceReducer(drop_unmatched=False), QuerySummaryReducer()])  # fmt: skip
    trace = result.table("xic_trace")
    summary = result.table("query_summary")
    assert summary["passed"].null_count == summary.num_rows  # no gate, no verdict
    for row in summary.to_pylist():
        t = trace.filter(pc.equal(trace["target_id"], row["target_id"]))
        for level in (1, 2):
            part = t.filter(pc.equal(t["level"], level))
            assert len(set(part["scan"].to_pylist())) == row[f"n_eligible_ms{level}"]
            assert pc.sum(pc.invert(pc.is_nan(part["mz_observed"]))).as_py() == \
                row[f"n_matched_ms{level}"]  # fmt: skip


def test_queries_without_fragments_can_be_gated_on_isotopes(tmp_root):
    run, bundle = _run(tmp_root)
    rows = [dict(target_id=1, precursor_mz=run.targets["precursor_mz"][0].as_py(), precursor_charge=2),
            dict(target_id=2, precursor_mz=1234.5678, precursor_charge=2)]  # fmt: skip
    plan = compile_queries(pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE), levels=[1])
    for gate, want in ((EvidenceGate(), [True, True]),
                       (EvidenceGate(min_isotopes=3, min_consecutive=2), [True, False])):  # fmt: skip
        s = sweep(bundle, plan, SweepSettings(tolerances=_TOL, gate=gate),
                  reducers=[QuerySummaryReducer()]).table("query_summary")  # fmt: skip
        assert s["passed"].to_pylist() == want
