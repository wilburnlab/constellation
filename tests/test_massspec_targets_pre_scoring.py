"""Targets and queries: defects found by the review of 2026-11-09
(``docs/reviews/pre-scoring-2026-11-09.md``, C1-C4), each held by a test
that fails on the code before: PSM targets that keep their decoys, their
precursor ids and their run; quantity targets of one run with their
windows; a target without fragment candidates beside a library; and a
library that names one ion twice."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.library.fragment_abundance import LibraryFragments
from constellation.massspec.library.prediction.run import predict_onto
from constellation.massspec.quant.schemas import PRECURSOR_QUANT
from constellation.massspec.quant.targets import (
    targets_from_library,
    targets_from_precursor_quant,
    targets_from_search,
)
from constellation.massspec.search.schemas import PSM_TABLE
from constellation.massspec.sweep import ChannelSettings, compile_queries
from test_library_prediction import PROTEINS, FakePredictor, _search_space


@pytest.fixture(scope="module")
def predicted(tmp_path_factory):
    lib = _search_space(tmp_path_factory.mktemp("lib"), PROTEINS, decoys=())
    lib, _ = predict_onto(lib, {"fragments": FakePredictor("fragments")})
    return lib


def _psms(rows, **columns) -> pa.Table:
    n = len(rows)
    base = {
        "psm_id": list(range(n)), "raw_file": ["r"] * n, "scan": pa.array(list(range(10, 10 + n)), pa.int32()),
        "sequence": [r["modified_sequence"] for r in rows], "modified_sequence": [r["modified_sequence"] for r in rows],
        "precursor_id": pa.array([r["precursor_id"] for r in rows], pa.int64()),
        "charge": pa.array([r["precursor_charge"] for r in rows], pa.int8()),
        "mz": [r["precursor_mz"] for r in rows], "retention_time_s": [100.0] * n,
        "is_decoy": [False] * n, "is_contaminant": [False] * n, "engine": ["scribetwo"] * n,
    }  # fmt: skip
    base.update(columns)
    return cast_to_schema(pa.table(base), PSM_TABLE)


# ── C1: targets from PSMs ───────────────────────────────────────────


def test_psm_targets_keep_their_decoys_and_precursors(predicted):
    rows = targets_from_library(predicted).to_pylist()[:4]
    psms = _psms(rows, is_decoy=[False, True, False, True])
    t = targets_from_search(SimpleNamespace(psms=psms))
    assert t.column("is_decoy").to_pylist() == [False, True, False, True]
    assert t.column("decoy_set").to_pylist() == [None, 0, None, 0]
    assert t.column("precursor_id").to_pylist() == [r["precursor_id"] for r in rows]


def test_psm_targets_get_their_librarys_abundances(predicted):
    rows = targets_from_library(predicted).to_pylist()
    t = targets_from_search(SimpleNamespace(psms=_psms(rows)))
    plan = compile_queries(t, levels=[2], channels=ChannelSettings(top_fragments=2),
                           fragment_abundance=LibraryFragments.from_library(predicted))  # fmt: skip
    # Without the precursor ids every query had NaN abundances, and the
    # top-2 selection left every one without fragment channels.
    assert (plan.fragments.counts() == 2).all()
    assert not np.isnan(plan.fragments.abundance).any()


def test_psm_targets_are_of_one_run(predicted):
    rows = targets_from_library(predicted).to_pylist()[:4]
    two = _psms(rows, acquisition_id=pa.array([3, 3, 4, 4], pa.int64()))
    with pytest.raises(ValueError, match="of 2 acquisitions"):
        targets_from_search(SimpleNamespace(psms=two))
    one = targets_from_search(SimpleNamespace(psms=two), acquisition_id=4)
    assert one.num_rows == 2 and one.column("scan").to_pylist() == [12, 13]
    with pytest.raises(ValueError, match="no rows of acquisition 5"):
        targets_from_search(SimpleNamespace(psms=two), acquisition_id=5)
    unlinked = _psms(rows, raw_file=["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="of 2 raw files"):
        targets_from_search(SimpleNamespace(psms=unlinked))
    # A bare table (no decoy flag, one run) still adapts.
    bare = pa.table({"psm_id": [1], "modified_sequence": ["PEPTIDEK"], "charge": pa.array([2], pa.int8()),
                     "mz": [465.0], "retention_time_s": [10.0], "scan": pa.array([7], pa.int32()),
                     "raw_file": ["r"], "sequence": ["PEPTIDEK"], "is_decoy": [False]})  # fmt: skip
    assert targets_from_search(SimpleNamespace(psms=bare)).num_rows == 1


# ── C3: targets from quantities ─────────────────────────────────────


def test_quantity_targets_are_of_one_run_with_their_windows(predicted):
    pid = predicted.precursors.column("precursor_id").to_pylist()[:2]
    q = cast_to_schema(pa.table({"precursor_id": pid * 2, "acquisition_id": [0, 0, 1, 1], "intensity": [1.0] * 4,
                                 "rt_observed": [100.0, 200.0, 900.0, 1000.0], "ccs_observed": [-1.0] * 4,
                                 "rt_start": [90.0, 190.0, 890.0, 990.0], "rt_end": [110.0, 210.0, 910.0, 1010.0]}),
                       PRECURSOR_QUANT)  # fmt: skip
    quant = SimpleNamespace(precursor_quant=q)
    with pytest.raises(ValueError, match="of 2 acquisitions"):
        targets_from_precursor_quant(quant, predicted)
    t = targets_from_precursor_quant(quant, predicted, acquisition_id=1)
    assert t.column("target_id").to_pylist() == pid  # one target per precursor, in the quantities' order
    assert t.column("rt_center").to_pylist() == [900.0, 1000.0]
    assert t.column("rt_start").to_pylist() == [890.0, 990.0] and t.column("rt_end").to_pylist() == [910.0, 1010.0]
    assert t.column("is_decoy").to_pylist() == [False, False]


# ── C2: a target without fragment candidates beside a library ───────


def test_a_target_without_a_charge_beside_a_library_compiles(predicted):
    t = targets_from_library(predicted)
    charges = t.column("precursor_charge").to_pylist()
    t = t.set_column(t.schema.get_field_index("precursor_charge"), "precursor_charge",
                     pa.array([None, *charges[1:]], pa.int8()))  # fmt: skip
    source = LibraryFragments.from_library(predicted)
    plan = compile_queries(t, levels=[1, 2], fragment_abundance=source)
    assert plan.fragments.counts()[0] == 0 and (plan.fragments.counts()[1:] > 0).all()
    assert plan.fragment_abundance["report"]["no_fragment_candidates"] == 1


# ── C4: one ion named twice ─────────────────────────────────────────


def test_one_ion_named_twice_takes_its_most_intense_row_in_any_order(predicted):
    t = targets_from_library(predicted).slice(0, 1)
    pid = t.column("precursor_id")[0].as_py()
    frags = predicted.fragments
    own = frags.filter(pa.compute.equal(frags.column("precursor_id"), pid))
    first = own.slice(0, 1).to_pylist()[0]
    louder = dict(first, intensity_predicted=float(first["intensity_predicted"]) * 3 + 1.0)
    shares = []
    for rows in ([*own.to_pylist(), louder], [louder, *own.to_pylist()]):
        lib = replace(predicted, fragments=pa.Table.from_pylist(rows, schema=frags.schema))
        source = LibraryFragments.from_library(lib)
        plan = compile_queries(t, levels=[2], fragment_abundance=source)
        shares.append(plan.fragments.abundance.copy())
        assert plan.fragment_abundance["report"]["duplicate_ions"] == 1
    np.testing.assert_array_equal(shares[0], shares[1])  # the rows' order changes nothing
    assert np.nansum(shares[0]) <= 1.0 + 1e-6
    total = sum(r["intensity_predicted"] for r in own.to_pylist() if r["intensity_predicted"] > 0) + louder["intensity_predicted"]
    assert np.isclose(np.nanmax(shares[0]), louder["intensity_predicted"] / total)
