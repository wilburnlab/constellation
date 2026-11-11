"""Targets and queries: defects found by the review of 2026-11-09
(``docs/reviews/pre-scoring-2026-11-09.md``, C1-C4), each held by a test
that fails on the code before: PSM targets that keep their decoys, their
precursor ids and their run; quantity targets of one run with their
windows; a target without fragment candidates beside a library; and a
library that names one ion twice. Then the second pass, of 2026-11-10
(``docs/reviews/pre-scoring-2026-11-10.md``, C5-C7, C9-C11, D11): PSM
decoys that find the library's decoys, a compile whose decoys lost their
library link, top-N chosen among acquirable ions, one run's
identifications held to its bundle, partly linked PSMs, a null library
intensity, the whole compile's report, and the channels a scorer sees."""

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


@pytest.fixture(scope="module")
def predicted_with_decoys(tmp_path_factory):
    """The same space with its decoys, every precursor predicted (b2 at half
    and y1 at full intensity)."""
    lib = _search_space(tmp_path_factory.mktemp("decoys"), PROTEINS)
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


# ── the second pass (review of 2026-11-10) ──────────────────────────


def test_psm_decoys_without_precursor_ids_find_the_librarys_decoys(predicted_with_decoys):
    """C5: ScribeTwo wrote decoy PSMs with no precursor id, so a decoy found
    no library rows: NaN abundances, and under top-N no fragment channel,
    while every target had N."""
    lib = predicted_with_decoys
    rows = targets_from_library(lib).to_pylist()
    decoy = [bool(r["is_decoy"]) for r in rows]
    assert any(decoy) and not all(decoy)
    psms = _psms(rows, precursor_id=pa.nulls(len(rows), pa.int64()), is_decoy=decoy)
    t = targets_from_search(SimpleNamespace(psms=psms))
    assert t.column("precursor_id").to_pylist() == [r["precursor_id"] for r in rows]
    plan = compile_queries(t, levels=[2], channels=ChannelSettings(top_fragments=2),
                           fragment_abundance=LibraryFragments.from_library(lib))  # fmt: skip
    assert (plan.fragments.counts() == 2).all()
    report = plan.fragment_abundance["report"]
    assert report["fragment_queries_decoys"] == sum(decoy) and report["no_positive_fragment_decoys"] == 0


def _with_precursor_ids(t: pa.Table, ids) -> pa.Table:
    return t.set_column(t.schema.get_field_index("precursor_id"), "precursor_id", pa.array(ids, pa.int64()))


@pytest.mark.parametrize("top", [None, 2])
def test_a_compile_whose_decoys_lost_their_library_link_is_refused(predicted_with_decoys, top):
    """C5's guard: decoys with no fragment of positive abundance far more
    often than targets is a lost link, not chemistry; scattered precursors
    without one, targets and decoys alike, compile as before."""
    lib = predicted_with_decoys
    t = targets_from_library(lib)
    decoy = np.asarray(t.column("is_decoy").to_pylist(), dtype=bool)
    ids = np.asarray(t.column("precursor_id").to_pylist(), dtype=np.int64)
    settings = ChannelSettings(top_fragments=top)
    lost = np.where(decoy, ids ^ 0x5A5A5A, ids)  # every decoy keyed to no library precursor
    with pytest.raises(ValueError, match="decoys have lost their library link"):
        compile_queries(_with_precursor_ids(t, lost), levels=[2], channels=settings,
                        fragment_abundance=LibraryFragments.from_library(lib))  # fmt: skip
    scattered = ids.copy()
    scattered[np.flatnonzero(decoy)[0]] ^= 0x5A5A5A
    scattered[np.flatnonzero(~decoy)[0]] ^= 0x5A5A5A
    plan = compile_queries(_with_precursor_ids(t, scattered), levels=[2], channels=settings,
                           fragment_abundance=LibraryFragments.from_library(lib))  # fmt: skip
    report = plan.fragment_abundance["report"]
    assert (report["no_positive_fragment_targets"], report["no_positive_fragment_decoys"]) == (1, 1)
    assert plan.n_queries == t.num_rows  # nothing dropped


def test_psms_linked_in_part_are_refused_unless_a_run_is_chosen(predicted):
    """C9: ids beside nulls from another raw file passed as one run."""
    rows = targets_from_library(predicted).to_pylist()[:4]
    part = _psms(rows, acquisition_id=pa.array([3, 3, None, None], pa.int64()), raw_file=["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="2 of the 4 PSMs are linked to no acquisition"):
        targets_from_search(SimpleNamespace(psms=part))
    assert targets_from_search(SimpleNamespace(psms=part), acquisition_id=3).column("scan").to_pylist() == [10, 11]


def test_psm_targets_say_which_run_they_are_of(predicted):
    from constellation.massspec.quant.targets import same_run, target_run

    rows = targets_from_library(predicted).to_pylist()[:3]
    t = targets_from_search(SimpleNamespace(psms=_psms(rows, raw_file=["run_a.dia"] * 3)))
    assert target_run(t) == "run_a.dia"
    assert target_run(targets_from_library(predicted)) is None
    assert same_run("run_a.dia", "/data/run_a.raw") and same_run("run_a", "C:\\data\\run_a.raw")
    assert not same_run("run_a.raw", "/data/run_ab.raw")
    # An unlinked table of one raw file is swept as the acquisition named.
    one = SimpleNamespace(psms=_psms(rows, raw_file=["run_a.dia"] * 3))
    assert targets_from_search(one, acquisition_id=5).num_rows == 3


def test_one_runs_identifications_are_refused_against_another_runs_bundle(predicted, tmp_path, capsys):
    """C7: a target source of one run was swept against any bundle, stamped
    acquisition 0, its scans pinned in another run."""
    from constellation.cli.__main__ import main
    from constellation.massspec.search.io import save_search
    from constellation.massspec.search.search import Search

    from _massspec_fixtures import dia_scans, write_bundle, write_bundle_manifest

    bundle = write_bundle_manifest(write_bundle(tmp_path, "run_b", dia_scans(cycles=3)))
    rows = targets_from_library(predicted).to_pylist()[:3]
    for name, raw in (("other", "run_a.raw"), ("own", "run_b.dia")):
        search = replace(Search.empty(), psms=_psms(rows, raw_file=[raw] * 3))
        save_search(search, tmp_path / name, format="parquet_dir")
    common = ["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--level", "1", "--no-progress"]
    assert main([*common, "--targets", str(tmp_path / "other"), "-o", str(tmp_path / "x")]) == 2
    err = capsys.readouterr().err
    assert "run_a.raw" in err and "/data/run_b.raw" in err and "--acquisition-id" in err
    assert main([*common, "--targets", str(tmp_path / "other"), "-o", str(tmp_path / "y"), "--acquisition-id", "0"]) == 0
    assert "warning" in capsys.readouterr().err
    assert main([*common, "--targets", str(tmp_path / "own"), "-o", str(tmp_path / "z")]) == 0


def test_a_library_row_without_an_intensity_is_not_modelled_not_absent(predicted):
    """C10: a null intensity became 0, "predicted absent"."""
    t = targets_from_library(predicted).slice(0, 1)
    pid = t.column("precursor_id")[0].as_py()
    rows = predicted.fragments.to_pylist()
    at = next(i for i, r in enumerate(rows) if r["precursor_id"] == pid and r["intensity_predicted"] > 0)
    lost = dict(rows[at], intensity_predicted=None)
    schema = predicted.fragments.schema
    nullable = schema.set(schema.get_field_index("intensity_predicted"), pa.field("intensity_predicted", pa.float64()))
    table = pa.Table.from_pylist([*rows[:at], lost, *rows[at + 1:]], schema=nullable)
    whole = LibraryFragments.from_library(predicted)
    source = LibraryFragments(source=table, coverage=whole.coverage, predicted=whole.predicted)
    plan = compile_queries(t, levels=[2], fragment_abundance=source)
    f = plan.fragments
    hit = (f.ion_type == lost["ion_type"]) & (f.position == lost["position"]) & (f.fragment_charge == lost["charge"])
    assert hit.sum() == 1 and np.isnan(f.abundance[hit][0])
    assert np.nansum(f.abundance) == pytest.approx(1.0)  # the rest are shares of what the library holds


def test_the_abundance_report_is_the_whole_compiles_not_its_last_chunks(predicted):
    """C11: the source kept the last compile chunk's report, which the
    command line wrote into its manifest."""
    t = targets_from_library(predicted)
    source = LibraryFragments.from_library(predicted)
    whole = compile_queries(t, levels=[2], fragment_abundance=source)
    assert source.report["rows"] == predicted.fragments.num_rows
    for _ in range(2):  # and each compile counts from nothing
        chunked = compile_queries(t, levels=[2], fragment_abundance=source, chunk_targets=3)
        for key in ("rows", "matched", "fragment_queries_targets", "no_positive_fragment"):
            assert source.report[key] == whole.fragment_abundance["report"][key], key
        assert source.describe()["report"] == chunked.fragment_abundance["report"]


def _ranged_meta() -> pa.Table:
    return pa.table({
        "level": pa.array([1, 2, 2, 2, 2, 2], pa.int8()),
        "isolation_lower": [None, 400.0, 400.0, 500.0, 500.0, 600.0],
        "isolation_upper": [None, 500.0, 500.0, 600.0, 600.0, 700.0],
        "scan_window_lower": [350.0, 250.0, 250.0, 150.0, 250.0, None],
        "scan_window_upper": [1250.0, 900.0, 900.0, 900.0, 800.0, None],
    })  # fmt: skip


def test_acquired_ranges_are_read_by_isolation_window():
    from constellation.massspec.acquisition_structure import acquired_ranges

    rows = acquired_ranges(_ranged_meta()).to_pylist()
    got = [(r["isolation_lower"], r["scan_window_lower"], r["scan_window_upper"], r["agree"], r["n_scans"]) for r in rows]
    # Agreeing scans give their range, disagreeing ones their envelope, and
    # a scan recording none leaves its window's range unknown.
    assert got == [(400.0, 250.0, 900.0, True, 2), (500.0, 150.0, 900.0, False, 2), (600.0, None, None, False, 1)]


def test_top_fragments_are_chosen_among_the_ions_the_scans_acquire(predicted):
    """C6: the top N were chosen among every ion, so a never-acquired y1
    (147 or 175 Th, each precursor's most abundant here) took a place."""
    from constellation.massspec.acquisition_structure import acquired_ranges

    t = targets_from_library(predicted)
    source = LibraryFragments.from_library(predicted)
    top1 = ChannelSettings(top_fragments=1)
    blind = compile_queries(t, levels=[2], channels=top1, fragment_abundance=source)
    assert (blind.fragments.mz < 200).all() and (blind.fragments.counts() == 1).all()
    seen = compile_queries(t, levels=[2], channels=top1, fragment_abundance=source, acquired_ranges=(200.0, 2000.0))
    assert (seen.fragments.mz >= 200).all() and seen.fragments.n > 0  # b2, where it is above 200
    report = seen.fragment_abundance["report"]
    assert report["not_made"]["not_acquired"] > 0
    assert report["no_fragment_made"] == int((seen.fragments.counts() == 0).sum()) > 0  # SAMPLER's b2 is 159 Th

    # By isolation window: the windows a query is matched in say its range.
    t4 = t.slice(0, 4)
    t4 = t4.set_column(t4.schema.get_field_index("precursor_mz"), "precursor_mz", pa.array([450.0, 550.0, 650.0, 800.0]))
    plain = compile_queries(t4, levels=[2])
    ranged = compile_queries(t4, levels=[2], acquired_ranges=acquired_ranges(_ranged_meta()))
    for q, (lo, hi) in enumerate([(250.0, 900.0), (150.0, 900.0), (-np.inf, np.inf), (-np.inf, np.inf)]):
        every = plain.fragments.mz[plain.fragments.offsets[q]:plain.fragments.offsets[q + 1]]
        kept = ranged.fragments.mz[ranged.fragments.offsets[q]:ranged.fragments.offsets[q + 1]]
        np.testing.assert_array_equal(kept, every[(every >= lo) & (every <= hi)])
    assert ranged.fragments.n < plain.fragments.n
    as_mapping = compile_queries(t4, levels=[2], acquired_ranges={(400.0, 500.0): (250.0, 900.0)})
    np.testing.assert_array_equal(as_mapping.fragments.counts()[1:], plain.fragments.counts()[1:])


def test_the_scorers_channels_leave_out_low_mz_and_the_precursor_before_the_top_n(predicted):
    """D11: no channel below 200 Th or within ± 10 Th of the precursor, the
    top N chosen among the rest, abundances shares of the whole ladder."""
    from constellation.massspec.peptide.channels import Uniform

    scoring = ChannelSettings.for_scoring()
    assert (scoring.fragment_charge_below_precursor, scoring.min_fragment_mz, scoring.precursor_exclusion_th,
            scoring.top_fragments, scoring.offsets()) == (1, 200.0, 10.0, 10, (-1, 0, 1, 2))  # fmt: skip
    assert ChannelSettings.from_dict(scoring.to_dict()) == scoring
    default = ChannelSettings()
    assert (default.min_fragment_mz, default.precursor_exclusion_th, default.fragment_charge_below_precursor) == (
        None, None, 0)  # fmt: skip
    old = {k: v for k, v in default.to_dict().items() if k not in ("min_fragment_mz", "precursor_exclusion_th")}
    assert ChannelSettings.from_dict(old) == default  # a plan written before reads as before
    with pytest.raises(ValueError, match="min_fragment_mz"):
        ChannelSettings(min_fragment_mz=-1.0)

    t = targets_from_library(predicted)
    plain = compile_queries(t, levels=[2], fragment_abundance=Uniform())
    cut = compile_queries(t, levels=[2], fragment_abundance=Uniform(),
                          channels=ChannelSettings(min_fragment_mz=200.0, precursor_exclusion_th=10.0))  # fmt: skip
    pmz = plain.precursor_mz[plain.fragments.query_of()]
    keep = (plain.fragments.mz >= 200.0) & (np.abs(plain.fragments.mz - pmz) > 10.0)
    assert 0 < keep.sum() < keep.size
    np.testing.assert_array_equal(cut.fragments.mz, plain.fragments.mz[keep])
    np.testing.assert_array_equal(cut.fragments.abundance, plain.fragments.abundance[keep])
    top = compile_queries(t, levels=[2], channels=ChannelSettings(top_fragments=1, min_fragment_mz=200.0),
                          fragment_abundance=LibraryFragments.from_library(predicted))  # fmt: skip
    assert (top.fragments.mz >= 200.0).all() and top.fragments.n > 0
    assert top.fragment_abundance["report"]["not_made"]["below_min_fragment_mz"] > 0


def test_the_command_line_chooses_fragments_within_the_bundles_ranges_and_takes_the_scorers_flags(predicted, tmp_path):
    """C6 on by default where the bundle is known, with a way off; D11's
    fields and the fragment-charge rule as flags."""
    import json

    from constellation.cli.__main__ import main
    from constellation.massspec.library.io import save_library

    from _massspec_fixtures import planted_run, write_bundle, write_bundle_manifest

    run = planted_run(cycles=6)
    for scan in run.scans:
        if scan["level"] == 2:
            scan["scan_window_lower"], scan["scan_window_upper"] = 200.0, 1500.0
    bundle = write_bundle_manifest(write_bundle(tmp_path, "run", run.scans))
    save_library(predicted, tmp_path / "lib", format="parquet_dir")
    common = ["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(tmp_path / "lib"),
              "--level", "2", "--tolerance", "0.35", "--tolerance-unit", "Da", "--no-progress",
              "--fragment-abundance", "library", "--top-fragments", "2"]  # fmt: skip

    def meta(name):
        manifest = json.loads((tmp_path / name / "manifest.json").read_text())
        return manifest.get("metadata", manifest)

    assert main([*common, "-o", str(tmp_path / "on")]) == 0
    on = meta("on")
    assert on["acquired_ranges"] == "bundle" and on["fragment_abundance"]["report"]["not_made"]["not_acquired"] > 0
    assert main([*common, "-o", str(tmp_path / "off"), "--no-acquired-ranges"]) == 0
    off = meta("off")
    assert off["acquired_ranges"] is None and "not_made" not in off["fragment_abundance"]["report"]
    assert main([*common, "-o", str(tmp_path / "scoring"), "--fragment-charge-below-precursor", "1",
                 "--min-fragment-mz", "250", "--precursor-exclusion", "10"]) == 0  # fmt: skip
    channels = meta("scoring")["channels"]
    assert (channels["fragment_charge_below_precursor"], channels["min_fragment_mz"],
            channels["precursor_exclusion_th"]) == (1, 250.0, 10.0)  # fmt: skip
