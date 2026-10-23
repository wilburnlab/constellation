"""Retention-time alignment for mass spec: the EncyclopeDIA replay, pairs, and the API."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from constellation.massspec.manifest import table_partition_dir
from constellation.massspec.results import IDENTIFICATION_SCHEMA
from constellation.massspec.rt import (
    PAIR_SCHEMA,
    align,
    align_to_library,
    engine_pairs,
    library_pairs,
    run_pairs,
    to_rt_alignment_table,
)
from constellation.massspec.rt.encyclopedia import (
    MIN_KDE_POINTS,
    _binary_search,
    _Range,
    _round_half_up,
    _stdev,
    fit_encyclopedia,
)
from constellation.massspec.search.schemas import RT_ALIGNMENT

_FIXTURES = Path(__file__).parent / "data" / "encyclopedia_rt_kde" / "6.9.23"


def _f32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).to(torch.float32)


# ── the replay, held to what the jar returned (scripts/survey-encyclopedia-rt-kde.py) ──


@pytest.mark.parametrize("path", sorted(_FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_the_replay_returns_exactly_what_the_jar_returned(path: Path) -> None:
    record = json.loads(path.read_text())
    x = torch.tensor(record["x"], dtype=torch.float64)
    y = torch.tensor(record["y"], dtype=torch.float64)
    fit = fit_encyclopedia(x, y)
    assert torch.equal(fit.warp(x).to(torch.float32), _f32(record["warp"]))
    assert torch.equal(fit.delta(x, y).to(torch.float32), _f32(record["delta"]))
    assert fit.within_band(x, y).tolist() == record["within95"]
    assert torch.equal(fit.curve.inverse(y).to(torch.float32), _f32(record["inverse_y"]))
    query = torch.tensor(record["query_x"], dtype=torch.float64)
    assert torch.equal(fit.warp(query).to(torch.float32), _f32(record["query_warp"]))


def test_the_fixtures_reach_every_kind_of_fit() -> None:
    models = {p.stem: fit_encyclopedia(*(torch.tensor(json.loads(p.read_text())[k]) for k in ("x", "y"))).model
              for p in _FIXTURES.glob("*.json")}
    assert models["n1"] == "identity" and models["n2"] == models["n20"] == "linear"
    assert models["n21"] == "kde" and MIN_KDE_POINTS == 21


def test_java_rounding_and_binary_search() -> None:
    assert [_round_half_up(v) for v in (2.5, -2.5, 2.4999999, -0.5)] == [3, -2, 2, 0]
    rng = random.Random(3)
    for _ in range(50):
        values = sorted(rng.choice([0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.5]) for _ in range(rng.randint(1, 12)))
        keys = [rng.choice([-1.0, 0.0, 1.0, 1.5, 2.0, 3.5, 9.0]) for _ in range(8)]
        got = _binary_search(torch.tensor(values, dtype=torch.float64), torch.tensor(keys, dtype=torch.float64))
        assert got.tolist() == [_java_binary_search(values, k) for k in keys]


def _java_binary_search(a: list[float], key: float) -> int:
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        if a[mid] < key:
            lo = mid + 1
        elif a[mid] > key:
            hi = mid - 1
        else:
            return mid
    return -(lo + 1)


def test_a_zero_width_axis_maps_to_the_middle_cell() -> None:
    values = _f32([5.0, 5.0, 5.0])
    assert _Range(values).index(values, 3000).tolist() == [1500, 1500, 1500]


def test_the_bandwidths_running_sums_are_float32() -> None:
    # A sum a float32 accumulator cannot carry exactly: the result must
    # be the sequential float32 one, not the double one.
    values = [16777216.0, 1.0, 1.0, 1.0, 1.0]
    acc = np.float32(0)
    for v in values:
        acc = np.float32(acc + np.float32(v))
    mean = np.float32(acc / np.float32(len(values)))
    var = np.float32(0)
    for v in values:
        d = np.float32(np.float32(v) - mean)
        var = np.float32(var + np.float32(d * d))
    expected = float(np.float32(math.sqrt(np.float32(var / np.float32(len(values))))))
    assert _stdev(values) == expected


# ── pairs ──


def _ids(rows: list[dict]) -> pa.Table:
    defaults = {"unit": "u", "acquisition_id": 0, "modified_sequence": None, "sequence": None, "charge": 2,
                "is_decoy": False, "qvalue": 0.001}
    full = [{**defaults, **r} for r in rows]
    return pa.Table.from_pylist(full, schema=IDENTIFICATION_SCHEMA)


def test_library_pairs_keep_confident_targets_with_a_library_time() -> None:
    ids = _ids([
        {"precursor_id": 1, "peptide_id": 10, "rt": 100.0, "sequence": "PEPTIDEK", "qvalue": 0.02},  # beyond 1 %
        {"precursor_id": 2, "peptide_id": 11, "rt": 200.0, "sequence": "PEPTIDER"},
        {"precursor_id": 3, "peptide_id": 11, "rt": 210.0, "sequence": "PEPTIDER", "qvalue": 0.0001},  # better
        {"precursor_id": 4, "peptide_id": 12, "rt": 300.0, "sequence": "PEPK"},  # too short
        {"precursor_id": 5, "peptide_id": 13, "rt": 400.0, "sequence": "LONGPEPTIDEK", "is_decoy": True},
        {"precursor_id": 6, "peptide_id": 14, "rt": 500.0, "sequence": "ANOTHERONEK"},  # no library time
        {"precursor_id": 7, "peptide_id": 15, "rt": None, "sequence": "NOTIMEHEREK"},
    ])
    precursors = pa.table({
        "precursor_id": pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64()),
        "rt_predicted": [90.0, 190.0, 205.0, 280.0, 390.0, -1.0, 600.0],
    })
    pairs = library_pairs(ids, precursors, min_length=7)
    assert pairs.schema.equals(PAIR_SCHEMA, check_metadata=False)
    assert pairs.column("precursor_id").to_pylist() == [3]
    assert pairs.column("reference_rt").to_pylist() == [205.0]
    by_precursor = library_pairs(ids, precursors, per="precursor")
    assert sorted(by_precursor.column("precursor_id").to_pylist()) == [2, 3, 4]


def test_run_pairs_join_two_runs_on_the_peptide() -> None:
    reference = _ids([
        {"precursor_id": 1, "peptide_id": 10, "rt": 100.0},
        {"precursor_id": 2, "peptide_id": 11, "rt": 200.0, "qvalue": 0.005},
        {"precursor_id": 3, "peptide_id": 12, "rt": 300.0},
    ])
    target = _ids([
        {"precursor_id": 1, "peptide_id": 10, "rt": 110.0, "acquisition_id": 1},
        {"precursor_id": 22, "peptide_id": 11, "rt": 215.0, "acquisition_id": 1, "charge": 3},
        {"precursor_id": 4, "peptide_id": 13, "rt": 400.0, "acquisition_id": 1},
    ])
    pairs = run_pairs(reference, target)
    rows = sorted(zip(pairs.column("peptide_id").to_pylist(), pairs.column("reference_rt").to_pylist(),
                      pairs.column("observed_rt").to_pylist(), pairs.column("qvalue").to_pylist()))
    assert rows == [(10, 100.0, 110.0, 0.001), (11, 200.0, 215.0, 0.005)]
    assert run_pairs(reference, target, per="precursor").num_rows == 1


def _rt_alignment(x: list[float], y: list[float], *, pass_name: str = "final", decoy_at: int | None = None) -> pa.Table:
    n = len(x)
    return pa.table({
        "acquisition_id": pa.array([0] * n, pa.int64()),
        "engine": ["encyclopedia"] * n,
        "pass": [pass_name] * n,
        "modified_sequence": [f"PEP{i}" for i in range(n)],
        "peptide_id": pa.array(list(range(n)), pa.int64()),
        "charge": pa.array([None] * n, pa.int32()),
        "library_rt": [v * 60 for v in x],
        "observed_rt": [v * 60 for v in y],
        "fitted_rt": [0.0] * n,
        "residual": [0.0] * n,
        "within_95": [True] * n,
        "is_decoy": [i == decoy_at for i in range(n)],
    }, schema=RT_ALIGNMENT)


def test_engine_pairs_take_the_final_pass_in_order_with_decoys() -> None:
    first = _rt_alignment([1.0, 2.0], [1.1, 2.1], pass_name="first")
    final = _rt_alignment([5.0, 3.0, 4.0], [5.5, 3.3, 4.4], decoy_at=1)
    pairs = engine_pairs(pa.concat_tables([first, final]))
    assert pairs.column("reference_rt").to_pylist() == [300.0, 180.0, 240.0]
    assert pairs.column("is_decoy").to_pylist() == [False, True, False]
    assert engine_pairs(pa.concat_tables([first, final]), pass_name="first").num_rows == 2
    assert engine_pairs(final, include_decoys=False).num_rows == 2


# ── align ──


def _outlier_fixture() -> tuple[list[float], list[float]]:
    record = json.loads((_FIXTURES / "outliers.json").read_text())
    return record["x"], record["y"]


def test_the_default_is_the_global_path_refined_by_the_modal_objective() -> None:
    x, y = _outlier_fixture()
    fitted = align(engine_pairs(_rt_alignment(x, y)))
    assert fitted.method == "global" and fitted.encyclopedia is None
    assert fitted.alignment.diagnostics["refine"] == "modal"


def test_the_encyclopedia_method_is_the_replay_in_seconds() -> None:
    x, y = _outlier_fixture()
    pairs = engine_pairs(_rt_alignment(x, y))
    fitted = align(pairs, method="encyclopedia")
    assert fitted.method == "encyclopedia" and fitted.encyclopedia is not None
    minutes = fit_encyclopedia(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
    q = torch.tensor(x, dtype=torch.float64)
    assert torch.allclose(fitted.to_observed(q * 60), minutes.warp(q) * 60, atol=1e-3)
    assert fitted.band[0] == pytest.approx(minutes.band[0] * 60)
    inside = fitted.within_band(q * 60, torch.tensor(y, dtype=torch.float64) * 60)
    assert inside.float().mean().item() == pytest.approx(0.95, abs=0.002)


def test_align_with_the_global_method_and_soft_tails() -> None:
    x, y = _outlier_fixture()
    pairs = engine_pairs(_rt_alignment(x, y))
    fitted = align(pairs, method="global")
    lo, hi = fitted.alignment.domain
    beyond = torch.tensor([hi + 120.0], dtype=torch.float64)
    assert bool(fitted.is_extrapolated(beyond))
    at_edge = fitted.to_observed(torch.tensor([hi], dtype=torch.float64))
    assert float(fitted.to_observed(beyond)) > float(at_edge)
    round_trip = fitted.to_reference(fitted.to_observed(torch.tensor([1800.0, 3600.0])))
    assert round_trip.tolist() == pytest.approx([1800.0, 3600.0], abs=1e-6)


def test_the_replay_takes_no_generic_options() -> None:
    x, y = _outlier_fixture()
    with pytest.raises(ValueError):
        align(engine_pairs(_rt_alignment(x, y)), method="encyclopedia", refine="modal")
    with pytest.raises(ValueError):
        align(engine_pairs(_rt_alignment(x, y)), method="encyclopedia", curve="linear")


def test_to_rt_alignment_table() -> None:
    x, y = _outlier_fixture()
    pairs = engine_pairs(_rt_alignment(x, y))
    fitted = align(pairs, method="global", refine=None)
    table = to_rt_alignment_table(fitted, pairs, acquisition_id=3)
    assert table.schema.equals(RT_ALIGNMENT)
    assert set(table.column("engine").to_pylist()) == {"constellation"}
    assert set(table.column("pass").to_pylist()) == {"global"}
    residual = np.asarray(table.column("residual")) + np.asarray(table.column("fitted_rt"))
    assert np.allclose(residual, np.asarray(table.column("observed_rt")))
    irt = align(pairs, method="global", refine=None, reference_unit="iRT")
    with pytest.raises(ValueError):
        to_rt_alignment_table(irt, pairs, acquisition_id=3)


def test_align_to_library_refits_the_engines_own_points(tmp_path: Path) -> None:
    x, y = _outlier_fixture()
    directory = table_partition_dir(tmp_path, "rt_alignment", "run-0000abcd", 0)
    directory.mkdir(parents=True)
    pq.write_table(_rt_alignment(x, y), directory / "part-0.parquet")
    fitted = align_to_library(tmp_path, unit="run-0000abcd", acquisition_id=0)
    assert fitted.n_pairs == len(x)
    with pytest.raises(ValueError):
        align_to_library(tmp_path, unit="run-0000abcd", acquisition_id=7)


def _pairs_table(reference: list[float], observed: list[float]) -> pa.Table:
    n = len(reference)
    columns = {f.name: pa.nulls(n, f.type) for f in PAIR_SCHEMA}
    columns["reference_rt"] = pa.array(reference, pa.float64())
    columns["observed_rt"] = pa.array(observed, pa.float64())
    return pa.table(columns, schema=PAIR_SCHEMA)


def test_band_flags_follow_the_curve_in_use_beyond_the_fit() -> None:
    times = [300.0 * i for i in range(21)]
    fitted = align(_pairs_table(times, times), method="encyclopedia")
    beyond = torch.tensor([6100.0], dtype=torch.float64)
    assert float(fitted.to_observed(beyond)) == pytest.approx(6100.0)
    assert float(fitted.residuals(beyond, beyond)) == pytest.approx(0.0)
    assert bool(fitted.within_band(beyond, beyond))


def test_band_flags_inside_the_fit_are_the_jars() -> None:
    x, y = _outlier_fixture()
    fitted = align(engine_pairs(_rt_alignment(x, y)), method="encyclopedia", extrapolation="clamp")
    xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    jar = fit_encyclopedia(xt, yt).within_band(xt, yt)
    assert torch.equal(fitted.within_band(xt * 60, yt * 60), jar)


def test_one_pair_gives_the_jars_identity_everywhere() -> None:
    fitted = align(_pairs_table([750.0], [840.0]), method="encyclopedia")
    assert fitted.encyclopedia.model == "identity"
    assert fitted.to_observed(torch.tensor([750.0, 3000.0])).tolist() == pytest.approx([750.0, 3000.0])


@pytest.mark.parametrize("method", ["encyclopedia", "global"])
def test_queries_keep_their_shape(method: str) -> None:
    times = [300.0 * i for i in range(30)]
    fitted = align(_pairs_table(times, [t + 60.0 for t in times]), method=method, refine=None if method == "global" else "auto")
    query = torch.tensor([[600.0], [1200.0]], dtype=torch.float64)
    out = fitted.to_observed(query)
    assert out.shape == (2, 1)
    assert out.reshape(-1).tolist() == pytest.approx([660.0, 1260.0], abs=1.0)
    assert fitted.within_band(query, out).shape == (2, 1)
