"""The dense view (`massspec.sweep.dense`) is the cell view laid out as
``[queries, scans, channels]`` tensors: every cell of ``QueryBatch.cells``
is an acquired cell of the view with the same values, nothing else is, and
padding carries nothing. It is the same for every query whichever layout,
kernel, path or batching swept it, and each MS2 slot finds its parent MS1
scan on the query's own MS1 axis."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from _massspec_fixtures import planted_run, write_bundle, write_bundle_manifest
from constellation.massspec.precursor_major import build_precursor_major
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep import compile_queries
from constellation.massspec.sweep.engine import SweepSettings, sweep
from constellation.massspec.sweep.evidence import EvidenceGate
from constellation.massspec.sweep.queries import ChannelSettings
from constellation.massspec.sweep.rules import AllWithin, SumWithin

_RUNS = {
    "dia": dict(acquisition="dia", cycles=30, window_width=100.0),
    "staggered": dict(acquisition="dia", staggered=True, cycles=30, window_width=100.0),
    "dda": dict(acquisition="dda", cycles=40),
}
_TOL = {1: (20.0, "ppm"), 2: (0.35, "Da")}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("dense"))


@functools.lru_cache(maxsize=None)
def _run(name: str, tmp_root: str):
    run = planted_run(**_RUNS[name])
    scans = [dict(s) for s in run.scans]
    for s in scans:
        if s["level"] == 2:  # a recorded range at MS2 too, so some cells are not acquired
            s["scan_window_lower"], s["scan_window_upper"] = 200.0, 1500.0
    bundle = write_bundle_manifest(write_bundle(Path(tmp_root), name, scans, row_group_size=97))
    build_precursor_major(bundle)
    return run, bundle


def _targets(run) -> pa.Table:
    rows = run.targets.to_pylist()
    rows += [dict(r, target_id=r["target_id"] + 500, modified_sequence=r["modified_sequence"][-2::-1] + r["modified_sequence"][-1])
             for r in rows if "[" not in r["modified_sequence"]]  # fmt: skip
    return pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)


def _against_cells(batch, level: int) -> int:
    """Hold the batch's dense view at ``level`` to its cell view; returns the cells compared."""
    cells = batch.cells(level, fill=True)
    d = batch.dense(level, dtype=torch.float64)
    q, s, c = d.shape
    assert q == batch.queries.size
    qpos = np.searchsorted(d.queries, cells.query)
    rows = d.row.numpy()
    # Each cell's slot: the one holding its scan on its query's axis.
    span = max(int(batch.scans.n), 1)
    slot_keys = np.arange(q)[:, None] * span + np.where(rows >= 0, rows, span - 1)
    flat = slot_keys.reshape(-1)
    by = np.argsort(flat, kind="stable")
    at = by[np.searchsorted(flat[by], qpos * span + cells.row)]
    slot = at % max(s, 1)
    assert (rows[qpos, slot] == cells.row).all()
    ch = cells.channel
    acquired = d.acquired.numpy()
    assert acquired[qpos, slot, ch].all() and int(acquired.sum()) == cells.n  # the same cells, and no more
    np.testing.assert_array_equal(d.matched.numpy()[qpos, slot, ch], cells.matched)
    np.testing.assert_array_equal(d.intensity.numpy()[qpos, slot, ch], cells.intensity)
    np.testing.assert_array_equal(d.mz_error_da.numpy()[qpos, slot, ch], cells.mz_error_da)
    np.testing.assert_array_equal(d.mz.numpy()[qpos, ch], cells.mz_theoretical)
    np.testing.assert_array_equal(d.abundance.numpy()[qpos, ch], cells.abundance)
    for name in ("rt", "tic", "iit", "scan"):
        np.testing.assert_array_equal(getattr(d, name).numpy()[qpos, slot], getattr(cells, name), err_msg=name)
    # Padding and cells not acquired carry nothing.
    off = ~acquired
    assert (d.intensity.numpy()[off] == 0).all() and not d.matched.numpy()[off].any()
    assert np.isnan(d.mz_error_da.numpy()[off]).all()
    pad = ~d.slot_valid.numpy()
    assert (rows[pad] == -1).all() and np.isnan(d.rt.numpy()[pad]).all() and not acquired[pad].any()
    assert not acquired[~d.channel_valid.numpy()[:, None, :].repeat(s, axis=1)].any()
    return cells.n


class _Checked:
    """A reducer that holds every batch's dense views to its cells, and
    gives each query's dense cells back as rows (to compare paths)."""

    name = "dense"
    schema = pa.schema([("query", pa.int64()), ("level", pa.int8()), ("scan", pa.int64()), ("channel", pa.int64()),
                        ("intensity", pa.float64()), ("matched", pa.bool_()), ("parent_scan", pa.int64())])  # fmt: skip
    gated = False
    zero_fills = True

    def reduce(self, batch):
        parts = []
        dense = {lv: batch.dense(lv, dtype=torch.float64) for lv in batch.plan.levels}
        for level, d in dense.items():
            _against_cells(batch, level)
            parent = (d.parent_slots(dense[level - 1]).numpy() if level - 1 in dense
                      else np.full(d.row.shape, -1, np.int64))  # fmt: skip
            qi, si, ci = np.nonzero(d.acquired.numpy())
            if level - 1 in dense:
                below = dense[level - 1].scan.numpy()
                ps = parent[qi, si]
                parent_scan = np.where(ps >= 0, below[qi, np.maximum(ps, 0)], -1)
            else:
                parent_scan = np.full(qi.size, -1, np.int64)
            parts.append(pa.table({
                "query": d.global_queries[qi], "level": np.full(qi.size, level, np.int8),
                "scan": d.scan.numpy()[qi, si], "channel": ci, "intensity": d.intensity.numpy()[qi, si, ci],
                "matched": d.matched.numpy()[qi, si, ci], "parent_scan": parent_scan,
            }, schema=self.schema))  # fmt: skip
        return pa.concat_tables(parts) if parts else self.schema.empty_table()


def _sorted(t: pa.Table) -> pa.Table:
    return t.sort_by([("query", "ascending"), ("level", "ascending"), ("scan", "ascending"), ("channel", "ascending")])


@pytest.mark.parametrize("name", list(_RUNS))
@pytest.mark.parametrize("charges", [None, (1, 4)])
def test_the_dense_view_is_the_cell_view(tmp_root, tmp_path, name, charges):
    run, bundle = _run(name, tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0, channels=ChannelSettings(charge_range=charges))
    base = sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[_Checked()]).table("dense")
    assert base.num_rows > 0 and pc.any(base.column("matched")).as_py()
    small = SweepSettings(tolerances=_TOL, reduce_rows=800, batch_cells=300, unit_cells=600)
    for kw in (dict(settings=SweepSettings(tolerances=_TOL, kernel="point")),
               dict(settings=SweepSettings(tolerances=_TOL, kernel="channel")),
               dict(settings=small, work_dir=tmp_path / "w"),
               dict(settings=small, workers=2, work_dir=tmp_path / "w2"),
               dict(settings=small, layout="precursor_major"),
               dict(settings=SweepSettings(tolerances=_TOL), layout="precursor_major")):  # fmt: skip
        settings = kw.pop("settings")
        got = sweep(bundle, plan, settings, reducers=[_Checked()], **kw).table("dense")
        assert _sorted(got).equals(_sorted(base)), kw


def test_each_ms2_slot_finds_its_parent_ms1_scan(tmp_root):
    from constellation.massspec.acquisition_structure import parent_scans

    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    seen = {}

    class Probe:
        name, schema, gated, zero_fills = "probe", pa.schema([("n", pa.int64())]), False, True

        def reduce(self, batch):
            seen.update(batch=batch, d1=batch.dense(1), d2=batch.dense(2))
            return pa.table({"n": [0]}, schema=self.schema)

    sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[Probe()])
    batch, d1, d2 = seen["batch"], seen["d1"], seen["d2"]
    ledger = batch.scans
    want = parent_scans(ledger.scan, ledger.level)
    rows2 = d2.row.numpy()
    valid = rows2 >= 0
    np.testing.assert_array_equal(ledger.scan[np.maximum(d2.parent_row.numpy(), 0)][valid], want[rows2[valid]])
    slots = d2.parent_slots(d1).numpy()
    on_axis = slots >= 0
    assert on_axis[valid].mean() > 0.9  # interval edges aside, the parent is on the axis
    qi, si = np.nonzero(on_axis)
    np.testing.assert_array_equal(d1.scan.numpy()[qi, slots[qi, si]], want[rows2[qi, si]])
    # Each MS2 scan comes after its parent, less than a cycle later.
    lag = d2.rt.numpy()[qi, si] - d1.rt.numpy()[qi, slots[qi, si]]
    assert (lag > 0).all()
    with pytest.raises(ValueError, match="same queries"):
        d2.parent_slots(batch.dense(1, queries=batch.queries[:2]))


def test_one_value_a_cell(tmp_root):
    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    with pytest.raises(ValueError, match="one value per cell"):
        sweep(bundle, plan, SweepSettings(tolerances=_TOL, rule=AllWithin()), reducers=[_Checked()])
    got = sweep(bundle, plan, SweepSettings(tolerances=_TOL, rule=SumWithin()), reducers=[_Checked()])
    assert got.table("dense").num_rows > 0


def test_queries_without_scans_and_levels_without_channels(tmp_root):
    run, bundle = _run("dia", tmp_root)
    rows = run.targets.to_pylist()
    rows.append(dict(rows[0], target_id=9999, rt_center=None, rt_start=1e6, rt_end=1e6 + 1))  # after the run
    plan = compile_queries(pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE), levels=[1], rt_window=20.0)
    seen = {}

    class Probe:
        name, schema, gated, zero_fills = "probe", pa.schema([("n", pa.int64())]), False, True

        def reduce(self, batch):
            seen["d1"] = batch.dense(1)
            with pytest.raises(ValueError, match="no channels at MS2"):
                batch.dense(2)
            seen["chunks"] = list(batch.dense_chunks(200))
            seen["n"] = batch.queries.size
            return pa.table({"n": [0]}, schema=self.schema)

    sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[Probe()])
    d1 = seen["d1"]
    last = d1.queries.size - 1
    assert not d1.slot_valid[last].any() and not d1.acquired[last].any()
    chunks = seen["chunks"]
    assert np.array_equal(np.concatenate(chunks), np.arange(seen["n"]))
    assert len(chunks) > 1


def test_dense_chunks_stay_under_their_budget(tmp_root):
    run, bundle = _run("staggered", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    budget = 3000
    sizes = []

    class Chunked:
        name, schema, gated, zero_fills = "chunked", pa.schema([("n", pa.int64())]), False, True

        def reduce(self, batch):
            covered = []
            for chunk in batch.dense_chunks(budget):
                views = [batch.dense(lv, queries=chunk) for lv in batch.plan.levels]
                total = sum(v.n_cells for v in views)
                sizes.append((chunk.size, total))
                covered.append(chunk)
            assert np.array_equal(np.concatenate(covered), batch.queries)
            return pa.table({"n": [len(covered)]}, schema=self.schema)

    sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[Chunked()])
    assert len(sizes) > 1
    assert all(total <= budget or n == 1 for n, total in sizes)


def test_a_gated_reducer_gets_the_dense_view_of_the_queries_it_sees(tmp_root):
    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    gate = EvidenceGate(min_fragments=3, min_consecutive=2)

    class Gated(_Checked):
        name = "gated_dense"
        gated = True

    got = sweep(bundle, plan, SweepSettings(tolerances=_TOL, gate=gate), reducers=[Gated()]).table("gated_dense")
    every = sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[_Checked()]).table("dense")
    kept = set(got.column("query").to_pylist())
    assert 0 < len(kept) < len(set(every.column("query").to_pylist()))
    sub = every.filter(pc.is_in(every.column("query"), pa.array(sorted(kept), pa.int64())))
    assert _sorted(got.rename_columns(sub.column_names)).equals(_sorted(sub))


# ──────────────────────────────────────────────────────────────────────
# Review 2026-11-10: V1, V2, V4–V8, S10, D11
# ──────────────────────────────────────────────────────────────────────


def _probe(bundle, plan, settings=None, **kw):
    """Run one sweep with a reducer that keeps its batches."""
    seen: list = []

    class Probe:
        name, schema, gated, zero_fills = "probe", pa.schema([("n", pa.int64())]), False, True

        def reduce(self, batch):
            seen.append(batch)
            return pa.table({"n": [0]}, schema=self.schema)

    sweep(bundle, plan, settings or SweepSettings(tolerances=_TOL), reducers=[Probe()], **kw)
    return seen


def test_observed_is_nan_wherever_nothing_was_measured(tmp_root):
    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    batch = _probe(bundle, plan)[0]
    d = batch.dense(2)
    obs = d.observed()
    acquired = d.acquired
    assert (~acquired).any() and acquired.any()
    assert torch.isnan(obs[~acquired]).all()
    assert torch.equal(obs[acquired], d.intensity[acquired])  # a zero that was measured stays zero
    assert (obs[acquired & ~d.matched] == 0).all()


def test_dense_refuses_queries_out_of_order_repeated_or_foreign_and_odd_dtypes(tmp_root):
    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    batch = _probe(bundle, plan)[0]
    qs = batch.queries
    assert qs.size >= 3
    for bad in (qs[[2, 0]], qs[[0, 0]], np.array([qs.max() + 1000])):
        with pytest.raises(ValueError, match="dense\\(queries"):
            batch.dense(2, queries=bad)
    with pytest.raises(ValueError, match="float32 or float64"):
        batch.dense(2, dtype=torch.float16)


def test_a_query_at_a_time_gives_the_same_tensors(tmp_root):
    """Selecting a chunk's matches by search (V7) changes nothing."""
    run, bundle = _run("staggered", tmp_root)
    plan = compile_queries(_targets(run), levels=[1, 2], rt_window=20.0)
    batch = _probe(bundle, plan)[0]
    for level in (1, 2):
        whole = batch.dense(level, dtype=torch.float64)
        for i, q in enumerate(batch.queries):
            one = batch.dense(level, queries=np.array([q]), dtype=torch.float64)
            n = int(one.n_slots[0])
            assert n == int(whole.n_slots[i])
            for name in ("intensity", "matched", "acquired", "point_index"):
                a, b = getattr(one, name)[0, :n], getattr(whole, name)[i, :n, : one.shape[2]]
                assert torch.equal(torch.nan_to_num(a.to(torch.float64)), torch.nan_to_num(b.to(torch.float64))), name


def test_point_index_is_the_matched_points_position(tmp_root):
    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    batch = _probe(bundle, plan)[0]
    d = batch.dense(2)
    cells = batch.cells(2, fill=False)
    pi = d.point_index.numpy()
    assert (pi[~d.matched.numpy()] == -1).all()
    assert (pi[d.matched.numpy()] >= 0).all()
    assert sorted(pi[d.matched.numpy()].tolist()) == sorted(cells.point_index.tolist())
    summed = _probe(bundle, plan, SweepSettings(tolerances=_TOL, rule=SumWithin()))[0].dense(2)
    assert (summed.point_index.numpy()[summed.matched.numpy()] >= -1).all()


def test_a_reducer_that_builds_dense_views_must_declare_zero_fills(tmp_root):
    """Its buckets are sized by matches alone otherwise (V6)."""
    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)

    class Undeclared:
        name, schema, gated = "undeclared", pa.schema([("n", pa.int64())]), False

        def reduce(self, batch):
            batch.dense(2)
            return pa.table({"n": [0]}, schema=self.schema)

    with pytest.raises(ValueError, match="zero_fills"):
        sweep(bundle, plan, SweepSettings(tolerances=_TOL), reducers=[Undeclared()])


def test_parent_reasons_and_brackets(tmp_path):
    """Why a slot has no parent (V8), and the MS1 slots either side of it."""
    from constellation.massspec.sweep.dense import PARENT_EMPTY, PARENT_FOUND, PARENT_OFF_AXIS

    run = planted_run(acquisition="dia", cycles=30, window_width=100.0)
    scans = [dict(s) for s in run.scans]
    ms1 = [i for i, s in enumerate(scans) if s["level"] == 1]
    emptied = scans[ms1[len(ms1) // 2]]
    emptied["mz"], emptied["intensity"] = np.empty(0), np.empty(0)
    bundle = write_bundle_manifest(write_bundle(tmp_path, "reasons", scans))
    rows = _targets(run).to_pylist()
    # One query whose interval opens just after an MS1 scan: its first MS2
    # slots' parent is in the run but not on its MS1 axis.
    opens = scans[ms1[5]]["rt"] + 1e-3
    rows.append(dict(rows[0], target_id=777, rt_center=None, rt_start=opens, rt_end=opens + 40.0))
    plan = compile_queries(pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE), levels=[1, 2], rt_window=20.0)
    batch = _probe(bundle, plan)[0]
    d1, d2 = batch.dense(1), batch.dense(2)
    slots, why = d2.parent_slots(d1, reasons=True)
    valid = d2.slot_valid.numpy()
    why = why.numpy()
    assert (why[~valid] == -1).all()
    assert ((why == PARENT_FOUND) == (slots.numpy() >= 0)).all()
    assert (why == PARENT_EMPTY).any()  # the MS2 scans after the emptied MS1
    assert (why == PARENT_OFF_AXIS).any()  # at the edges of a query's interval
    b = d2.brackets(d1)
    rt1, rt2 = d1.rt.numpy(), d2.rt.numpy()
    qi, si = np.nonzero(b.inside.numpy())
    left, right = b.left.numpy()[qi, si], b.right.numpy()[qi, si]
    assert (rt1[qi, left] <= rt2[qi, si]).all() and (rt2[qi, si] <= rt1[qi, right]).all()
    w = b.weight.numpy()[qi, si]
    assert ((w >= 0) & (w <= 1)).all()


def test_overlapping_channels_share_a_group(tmp_root):
    run, bundle = _run("dia", tmp_root)
    plan = compile_queries(_targets(run), levels=[2], rt_window=20.0)
    wide = SweepSettings(tolerances={1: (20.0, "ppm"), 2: (5.0, "Da")})
    batch = _probe(bundle, plan, wide)[0]
    d = batch.dense(2)
    group, mz, ok = d.overlap_group.numpy(), d.mz.numpy(), d.channel_valid.numpy()
    assert (group[~ok] == -1).all()
    for q in range(group.shape[0]):
        idx = np.flatnonzero(ok[q])
        by = idx[np.argsort(mz[q, idx], kind="stable")]
        for a, b in zip(by[:-1], by[1:]):
            same = abs(mz[q, b] - mz[q, a]) <= 10.0
            assert (group[q, a] == group[q, b]) == same
        for g in np.unique(group[q, idx]):
            assert g == idx[group[q, idx] == g].min()
    assert (group[ok] != np.nonzero(ok)[1]).any()  # some channels are not alone at 5 Da
