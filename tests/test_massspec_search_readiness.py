"""Defects a review of the search path found on 2026-11-08, each held by a
test that fails on the code before its fix: decoy sets carried apart,
fragment coverage through a ``.dlib``, an empirical library's annotation
tolerance, I/L twins' decoys, charge 0, a gate no query can pass, and a
library with no coverage on the command line."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from constellation.massspec.library.decoys import DecoySet, il_key
from constellation.massspec.library.fragment_abundance import (
    ANNOTATION_TOLERANCE_KEY,
    COVERAGE_KEY,
    LibraryFragments,
)
from constellation.massspec.library.io import load_library, save_library
from constellation.massspec.library.prediction.run import predict_onto
from constellation.massspec.quant.targets import targets_from_library, targets_from_table
from constellation.massspec.sweep import ChannelSettings, compile_queries
from constellation.massspec.sweep.engine import SweepSettings, sweep
from constellation.massspec.sweep.evidence import EvidenceGate
from constellation.massspec.sweep.queries import QueryPlan
from constellation.massspec.sweep.reduce import QuerySummaryReducer
from test_library_prediction import PROTEINS, FakePredictor, _search_space


@pytest.fixture
def predicted(tmp_path):
    lib = _search_space(tmp_path, PROTEINS, decoys=())
    out, _ = predict_onto(lib, {"fragments": FakePredictor("fragments")})
    return out


# ── (a) decoy sets ──────────────────────────────────────────────────


def test_each_decoy_set_rides_apart_into_targets_queries_and_the_summary(tmp_path):
    """Every decoy set is ``is_decoy``; which competes is FDR's choice, so
    the set itself must reach every table a decision could be made from."""
    from _massspec_fixtures import dia_scans, write_bundle, write_bundle_manifest

    lib = _search_space(tmp_path, PROTEINS, decoys=(DecoySet(), DecoySet("shuffle-interior", index=1)))
    targets = targets_from_library(lib)
    sets = targets.column("decoy_set").to_pylist()
    decoy = targets.column("is_decoy").to_pylist()
    assert {s for s, d in zip(sets, decoy) if d} == {0, 1}
    assert all(s is None for s, d in zip(sets, decoy) if not d)

    plan = compile_queries(targets, levels=[1])
    queries, _ = plan.to_tables()
    assert queries.column("decoy_set").to_pylist() == sets
    plan.write(tmp_path / "plan")
    assert QueryPlan.read(tmp_path / "plan").to_tables()[0].column("decoy_set").to_pylist() == sets

    bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans(cycles=2)))
    result = sweep(bundle, plan, SweepSettings(tolerances={1: (0.35, "Da")}), reducers=[QuerySummaryReducer()])
    summary = result.table("query_summary").sort_by("query")
    assert summary.column("decoy_set").to_pylist() == sets


# ── (b) coverage through a .dlib ────────────────────────────────────


def test_a_predicted_library_keeps_its_fragment_coverage_through_a_dlib(predicted, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_library(predicted, tmp_path / "lib.dlib", format="encyclopedia.dlib")
    back = load_library(tmp_path / "lib.dlib")
    assert back.metadata_extras[COVERAGE_KEY] == predicted.metadata_extras[COVERAGE_KEY]

    def abundance(lib):
        """Each channel's abundance by (peptide, charge, ion): the reader
        numbers precursors afresh, so channels are matched by what they are."""
        t = targets_from_library(lib)
        f = compile_queries(t, levels=[2], fragment_abundance=LibraryFragments.from_library(lib)).fragments
        seq, z, q = t.column("modified_sequence").to_pylist(), t.column("precursor_charge").to_pylist(), f.query_of()
        return {(seq[q[i]], z[q[i]], int(f.ion_type[i]), int(f.position[i]), int(f.fragment_charge[i])): f.abundance[i]
                for i in range(q.size)}  # fmt: skip

    before, after = abundance(predicted), abundance(back)
    assert before.keys() == after.keys()
    assert sum(v == 0 for v in before.values()) > 0
    for key, value in before.items():
        assert (np.isnan(after[key]), after[key] == 0) == (np.isnan(value), value == 0), key


# ── (c) an empirical library's annotation tolerance ────────────────


def _empirical(predicted, shift_ppm: float):
    """The predicted library as an empirical one (no coverage), its peaks
    stored ``shift_ppm`` off their ions, as an observed library's are."""
    f = predicted.fragments
    mz = f.column("mz_theoretical").to_numpy() * (1 + shift_ppm * 1e-6)
    f = f.set_column(f.schema.get_field_index("mz_theoretical"), "mz_theoretical", pa.array(mz))
    extras = {k: v for k, v in predicted.metadata_extras.items() if k != COVERAGE_KEY}
    return replace(predicted, fragments=f, metadata_extras=extras)


def test_an_empirical_library_is_held_to_the_tolerance_its_peaks_were_annotated_at(predicted):
    lib = _empirical(predicted, 15.0)
    targets = targets_from_library(lib)
    # Annotated at the readers' 20 ppm (the default when a library does not say).
    compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(lib))
    # A library that records a tighter tolerance is held to it.
    tight = replace(lib, metadata_extras={**lib.metadata_extras, ANNOTATION_TOLERANCE_KEY: {"value": 10, "unit": "ppm"}})
    with pytest.raises(ValueError, match="beyond the library's 10 ppm"):
        compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(tight))
    # ... and one annotated in Da by that many Da.
    da = replace(lib, metadata_extras={**lib.metadata_extras, ANNOTATION_TOLERANCE_KEY: {"value": 0.5, "unit": "Da"}})
    compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(da))


def test_reading_an_elib_records_the_tolerance_it_annotated_at(predicted, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_library(predicted, tmp_path / "lib.dlib", format="encyclopedia.dlib")
    back = load_library(tmp_path / "lib.dlib")
    assert back.metadata_extras[ANNOTATION_TOLERANCE_KEY] == {"value": 20.0, "unit": "ppm"}


# ── (d) I/L twins ───────────────────────────────────────────────────


def test_targets_differing_only_by_i_and_l_get_decoys_that_do_too(tmp_path):
    """One signal serves both twins, so their decoys must share one too."""
    lib = _search_space(tmp_path, {"A": "SAMPLEIVDEKAGR", "B": "SAMPLELVDEKAGR"}, charges=(2,))
    rows = lib.peptides.to_pylist()
    by_id = {r["peptide_id"]: r for r in rows}
    decoy_of = {by_id[r["decoy_of"]]["sequence"]: r["sequence"] for r in rows if r["is_decoy"]}
    assert set(decoy_of) == {"SAMPLEIVDEK", "SAMPLELVDEK"}
    a, b = decoy_of["SAMPLEIVDEK"], decoy_of["SAMPLELVDEK"]
    assert a != b and il_key(a) == il_key(b)
    # Both are the interior reversal, as a lone target's would be.
    assert {a, b} == {"SEDVIELPMAK", "SEDVLELPMAK"}


# ── (e) charge 0 ────────────────────────────────────────────────────


def test_a_charge_of_zero_is_no_charge():
    t = targets_from_table(pa.table({
        "target_id": pa.array([1, 2], pa.int64()),
        "modified_sequence": ["PEPTIDEK", "PEPTIDEK"],
        "precursor_charge": pa.array([0, 2], pa.int8()),
    }))  # fmt: skip
    with pytest.raises(ValueError, match="requires precursor_charge"):
        compile_queries(t, levels=[2])
    plan = compile_queries(t.slice(1), levels=[2])
    assert np.isfinite(plan.precursor_mz).all()
    with_ms1 = compile_queries(t.set_column(3, "precursor_mz", pa.array([500.0, None])), levels=[1, 2])
    queries, _ = with_ms1.to_tables()
    assert queries.column("precursor_charge").to_pylist() == [None, 2]
    assert np.isfinite(with_ms1.precursor_mz).all()


# ── (f) a gate no query can pass ────────────────────────────────────


def test_a_gate_asking_more_fragments_than_the_plan_keeps_is_refused(predicted, tmp_path):
    from _massspec_fixtures import dia_scans, write_bundle, write_bundle_manifest

    targets = targets_from_library(predicted)
    plan = compile_queries(targets, levels=[2], channels=ChannelSettings(top_fragments=5),
                           fragment_abundance=LibraryFragments.from_library(predicted))  # fmt: skip
    bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans(cycles=2)))
    gated = SweepSettings(tolerances={2: (0.35, "Da")}, gate=EvidenceGate(min_fragments=6))
    with pytest.raises(ValueError, match="no query could pass"):
        sweep(bundle, plan, gated)
    sweep(bundle, plan, replace(gated, gate=EvidenceGate(min_fragments=5)))


def test_the_command_line_refuses_top_fragments_below_the_gate(predicted, tmp_path, capsys):
    from constellation.cli.__main__ import main

    from _massspec_fixtures import dia_scans, write_bundle, write_bundle_manifest

    save_library(predicted, tmp_path / "lib", format="parquet_dir")
    bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans(cycles=2)))
    args = ["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(tmp_path / "lib"),
            "--level", "2", "--tolerance", "0.35", "--tolerance-unit", "Da", "--no-progress",
            "--fragment-abundance", "library", "--top-fragments", "5", "--gate", "-o", str(tmp_path / "out")]  # fmt: skip
    assert main(args) == 2
    assert "--min-fragments 6" in capsys.readouterr().err


# ── (g) a library with no coverage on the command line ─────────────


def test_the_command_line_refuses_a_library_with_no_coverage_unless_told_the_ions(predicted, tmp_path, capsys):
    from constellation.cli.__main__ import main

    from _massspec_fixtures import dia_scans, write_bundle, write_bundle_manifest

    empirical = replace(predicted, metadata_extras={k: v for k, v in predicted.metadata_extras.items()
                                                    if k != COVERAGE_KEY})  # fmt: skip
    save_library(empirical, tmp_path / "lib", format="parquet_dir")
    bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans(cycles=2)))
    args = ["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(tmp_path / "lib"),
            "--level", "2", "--tolerance", "0.35", "--tolerance-unit", "Da", "--no-progress",
            "--fragment-abundance", "library"]  # fmt: skip
    assert main([*args, "-o", str(tmp_path / "a")]) == 2
    assert "records no coverage" in capsys.readouterr().err
    assert main([*args, "-o", str(tmp_path / "b"), "--ion-types", "b,y", "--max-fragment-charge", "3"]) == 0
    assert pc.sum(pc.is_valid(load_library(tmp_path / "lib", format="parquet_dir").fragments.column("ion_type"))).as_py()


# ── (h) a precursor with no predicted fragment ─────────────────────


def test_a_precursor_with_no_positive_fragment_keeps_none_and_is_counted(predicted):
    targets = targets_from_library(predicted)
    silent = targets.column("precursor_id")[0].as_py()
    f = predicted.fragments
    zero = pc.equal(f.column("precursor_id"), silent)
    f = f.set_column(f.schema.get_field_index("intensity_predicted"), "intensity_predicted",
                     pc.if_else(zero, 0.0, f.column("intensity_predicted")))  # fmt: skip
    lib = replace(predicted, fragments=f)
    plan = compile_queries(targets, levels=[2], channels=ChannelSettings(top_fragments=2),
                           fragment_abundance=LibraryFragments.from_library(lib))  # fmt: skip
    counts = plan.fragments.counts()
    assert counts[0] == 0 and (counts[1:] == 2).all()
    report = plan.fragment_abundance["report"]
    assert report["no_positive_fragment"] == 1
    assert report["no_positive_fragment_examples"] == [targets.column("target_id")[0].as_py()]


# ── (i) compile in chunks ───────────────────────────────────────────


def _plans_equal(a: QueryPlan, b: QueryPlan) -> None:
    assert a.targets.equals(b.targets)
    for name in ("precursor_charge", "precursor_mz", "precursor_mz_derived", "rt_lo", "rt_hi", "assigned_scan"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    for kind in ("isotopes", "fragments"):
        x, y = getattr(a, kind), getattr(b, kind)
        assert (x is None) == (y is None)
        if x is None:
            continue
        assert x.loss_ids == y.loss_ids
        for name in ("offsets", "mz", "precursor_charge", "isotope", "ion_type", "position", "fragment_charge",
                     "loss", "abundance", "evidence_key"):  # fmt: skip
            got, want = getattr(x, name), getattr(y, name)
            assert got.dtype == want.dtype, (kind, name)
            np.testing.assert_array_equal(got, want, err_msg=f"{kind}.{name}")
    assert a.to_tables()[1].equals(b.to_tables()[1])
    assert a.fragment_abundance == b.fragment_abundance


@pytest.mark.parametrize("chunk", [1, 7, None])
def test_a_compile_in_chunks_is_the_compile_at_once_bit_for_bit(tmp_path, chunk):
    lib = _search_space(tmp_path, PROTEINS, decoys=(DecoySet(),))
    pred, _ = predict_onto(lib, {"fragments": FakePredictor("fragments")})
    targets = targets_from_library(pred)
    for settings in (ChannelSettings(top_fragments=2), ChannelSettings(neutral_losses=("H2O",))):
        whole = compile_queries(targets, levels=[1, 2], channels=settings,
                                fragment_abundance=LibraryFragments.from_library(pred), chunk_targets=None)  # fmt: skip
        parts = compile_queries(targets, levels=[1, 2], channels=settings,
                                fragment_abundance=LibraryFragments.from_library(pred),
                                chunk_targets=chunk or targets.num_rows)  # fmt: skip
        _plans_equal(parts, whole)


# ── (j) the fragment file, read by precursor ───────────────────────


def _fragment_file(predicted, path, *, sort: bool):
    import pyarrow.parquet as pq

    f = predicted.fragments
    f = f.sort_by("precursor_id") if sort else f.take(pa.array(np.random.default_rng(1).permutation(f.num_rows)))
    pq.write_table(f, path, row_group_size=4)
    return LibraryFragments(source=path, coverage=predicted.metadata_extras[COVERAGE_KEY],
                            predicted=predicted.precursors.column("precursor_id").to_numpy())  # fmt: skip


def test_a_fragment_file_sorted_by_precursor_is_read_a_few_row_groups_per_chunk(predicted, tmp_path):
    import pyarrow.parquet as pq

    targets = targets_from_library(predicted)
    whole = compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(predicted))
    source = _fragment_file(predicted, tmp_path / "sorted.parquet", sort=True)
    groups = pq.ParquetFile(tmp_path / "sorted.parquet").metadata.num_row_groups
    plan = compile_queries(targets, levels=[2], fragment_abundance=source, chunk_targets=3)
    np.testing.assert_array_equal(plan.fragments.abundance, whole.fragments.abundance)
    # Each row group is read by the chunk(s) whose precursors it holds: about once.
    assert source.row_groups_read <= groups + targets.num_rows // 3 + 1

    unsorted = _fragment_file(predicted, tmp_path / "unsorted.parquet", sort=False)
    plan = compile_queries(targets, levels=[2], fragment_abundance=unsorted, chunk_targets=3)
    np.testing.assert_array_equal(plan.fragments.abundance, whole.fragments.abundance)
    # An unsorted file is read once for the whole compile, not once per chunk.
    assert unsorted.row_groups_read == groups
    assert unsorted._cache is None


def test_the_library_join_is_exact_not_a_packed_key():
    """Under the old packing, position 2**20 at ion type t was the same
    integer as position 0 at ion type t + 1."""
    from constellation.massspec.library.fragment_abundance import _join

    rows = (np.array([5, 5]), np.array([1, 2]), np.array([1 << 20, 0]), np.array([1, 1]), np.array([0, 0]))
    cands = (np.array([5, 5]), np.array([2, 1]), np.array([0, 1 << 20]), np.array([1, 1]), np.array([0, 0]))
    i, j = _join(rows, np.ones(2, bool), cands)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 1), (1, 0)]
    big = (np.array([(1 << 23) + 7]), np.array([1]), np.array([3]), np.array([2]), np.array([0]))
    assert _join(big, np.ones(1, bool), big)[0].tolist() == [0]


# ── (m) the plan's loss ids ─────────────────────────────────────────


def test_a_plan_with_losses_reads_back_bit_for_bit(predicted, tmp_path):
    plan = compile_queries(targets_from_library(predicted), levels=[2],
                           channels=ChannelSettings(neutral_losses=("H2O", "NH3")))  # fmt: skip
    assert plan.fragments.loss_ids and (plan.fragments.loss > 0).any()
    plan.write(tmp_path / "plan")
    back = QueryPlan.read(tmp_path / "plan")
    np.testing.assert_array_equal(back.fragments.loss, plan.fragments.loss)
    assert back.fragments.loss_ids == plan.fragments.loss_ids
    assert back.to_tables()[1].equals(plan.to_tables()[1])


# ── (p) Counter's detection floor ───────────────────────────────────


def test_counters_censored_term_takes_a_floor_per_cell():
    import torch

    from constellation.massspec.counter.channels import censored_log_prob

    lam = torch.tensor([[1.0, 5.0], [2.0, 8.0]], requires_grad=True)
    floor = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    got = censored_log_prob(lam, floor)
    want = torch.stack([censored_log_prob(lam[:, c], float(floor[0, c])) for c in range(2)], dim=1)
    torch.testing.assert_close(got, want)
    got.sum().backward()
    assert lam.grad is not None and torch.isfinite(lam.grad).all()


def test_the_library_join_agrees_with_a_brute_force_join_and_its_fallback():
    from constellation.massspec.library.fragment_abundance import _arrow_join, _join

    rng = np.random.default_rng(3)
    for _ in range(5):
        n, m = int(rng.integers(0, 60)), int(rng.integers(0, 60))
        ids = rng.choice(np.array([3, 1 << 40, (1 << 62) + 5, 17]), 4, replace=False)
        left = (ids[rng.integers(0, 4, n)], *(rng.integers(0, 3, n) for _ in range(4)))
        right = (ids[rng.integers(0, 4, m)], *(rng.integers(0, 3, m) for _ in range(4)))
        usable = rng.random(n) < 0.8
        want = sorted((i, j) for i in range(n) if usable[i] for j in range(m)
                      if all(left[c][i] == right[c][j] for c in range(5)))  # fmt: skip
        i, j = _join(left, usable, right)
        assert list(zip(i.tolist(), j.tolist(), strict=True)) == want
        a, b = _arrow_join(left, np.flatnonzero(usable), right, np.arange(m))
        assert list(zip(a.tolist(), b.tolist(), strict=True)) == want
