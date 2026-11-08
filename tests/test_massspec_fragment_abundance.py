"""A library's fragment intensities as query channel abundances
(`massspec.library.fragment_abundance`), top-N fragment channels, and the
command line that asks for them (2026-11-07)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.library.fragment_abundance import COVERAGE_KEY, LibraryFragments
from constellation.massspec.library.io import load_library, save_library
from constellation.massspec.library.prediction.run import predict_onto
from constellation.massspec.peptide.ions import IonType, fragment_ladder_arrays_batch
from constellation.massspec.quant.targets import load_targets, targets_from_library
from constellation.massspec.sweep import ChannelSettings, compile_queries
from test_library_prediction import PROTEINS, FakePredictor, _search_space


@pytest.fixture
def predicted(tmp_path):
    """A search space with fragments predicted by the fake predictor: b2 at
    half and y1 at full intensity, charge 1, scaled by precursor charge."""
    lib = _search_space(tmp_path, PROTEINS, decoys=())
    out, _ = predict_onto(lib, {"fragments": FakePredictor("fragments")})
    return out


def _abundances(plan, q):
    f = plan.fragments
    sl = slice(f.offsets[q], f.offsets[q + 1])
    return {(int(t), int(p), int(z), int(loss)): float(a) for t, p, z, loss, a in
            zip(f.ion_type[sl], f.position[sl], f.fragment_charge[sl], f.loss[sl], f.abundance[sl])}  # fmt: skip


def test_library_abundances_join_exactly_as_shares_of_each_spectrum(predicted):
    targets = targets_from_library(predicted)
    plan = compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(predicted))
    lengths = {r["peptide_id"]: len(r["sequence"]) for r in predicted.peptides.to_pylist()}
    for q, row in enumerate(targets.to_pylist()):
        got = _abundances(plan, q)
        n = lengths[row["peptide_id"]]
        b2, y1 = (int(IonType.B), 1, 1, 0), (int(IonType.Y), n - 2, 1, 0)
        assert got[b2] == pytest.approx(1 / 3) and got[y1] == pytest.approx(2 / 3)  # 0.5 and 1.0 of 1.5
        # Every other b/y ion the model covers was predicted below its floor: 0.
        assert all(v == 0.0 for k, v in got.items() if k not in (b2, y1))
    assert plan.fragment_abundance["source"] == "library"
    assert plan.fragment_abundance["report"]["matched"] == predicted.fragments.num_rows


def test_an_ion_outside_the_models_coverage_is_unknown_not_absent(predicted):
    """0 is "predicted absent", NaN "never predicted": a-ions and charges
    above the model's ceiling were never predicted."""
    plan = compile_queries(targets_from_library(predicted), levels=[2],
                           channels=ChannelSettings(ion_types=(IonType.A, IonType.B, IonType.Y), max_fragment_charge=3),
                           fragment_abundance=LibraryFragments.from_library(predicted))  # fmt: skip
    f = plan.fragments
    covered = np.isin(f.ion_type, [IonType.B, IonType.Y]) & (f.fragment_charge <= np.minimum(2, f.precursor_charge))
    assert np.isnan(f.abundance[~covered]).all() and not np.isnan(f.abundance[covered]).any()
    bare = replace(predicted, metadata_extras={k: v for k, v in predicted.metadata_extras.items() if k != COVERAGE_KEY})
    empirical = compile_queries(targets_from_library(bare), levels=[2], fragment_abundance=LibraryFragments.from_library(bare))
    a = empirical.fragments.abundance
    assert (a > 0).sum() == predicted.fragments.num_rows and np.isnan(a[~(a > 0)]).all()  # nothing said absent


def test_top_fragments_keeps_the_most_abundant_in_ladder_order(predicted):
    targets = targets_from_library(predicted)
    plan = compile_queries(targets, levels=[2], channels=ChannelSettings(top_fragments=1),
                           fragment_abundance=LibraryFragments.from_library(predicted))  # fmt: skip
    f = plan.fragments
    assert (f.counts() == 1).all()
    assert (f.ion_type == IonType.Y).all() and np.allclose(f.abundance, 2 / 3)  # y1, the more intense
    two = compile_queries(targets, levels=[2], channels=ChannelSettings(top_fragments=2),
                          fragment_abundance=LibraryFragments.from_library(predicted)).fragments  # fmt: skip
    for q in range(two.offsets.size - 1):
        sl = slice(two.offsets[q], two.offsets[q + 1])
        assert two.position[sl].tolist() == sorted(two.position[sl].tolist())  # ladder order kept
        assert sorted(two.ion_type[sl].tolist()) == [IonType.B, IonType.Y]


def test_a_loss_the_model_names_gets_a_channel_where_the_rules_would_not(predicted):
    """The model, not the loss rules, says where its losses are."""
    row = predicted.precursors.to_pylist()[0]
    peptide = next(p for p in predicted.peptides.to_pylist() if p["peptide_id"] == row["peptide_id"])
    pf = parse_proforma(peptide["modified_sequence"])
    free = fragment_ladder_arrays_batch([pf], ion_types=(IonType.B, IonType.Y), max_fragment_charge=1,
                                        neutral_losses=("NH3",), loss_rules=False)  # fmt: skip
    ruled = fragment_ladder_arrays_batch([pf], ion_types=(IonType.B, IonType.Y), max_fragment_charge=1,
                                         neutral_losses=("NH3",))  # fmt: skip
    ruled_keys = set(zip(ruled.ion_type.tolist(), ruled.position.tolist(), ruled.loss.tolist()))
    pick = next(i for i in range(free.mz.size) if free.loss[i] > 0
                and (int(free.ion_type[i]), int(free.position[i]), int(free.loss[i])) not in ruled_keys)
    extra = pa.table({
        "precursor_id": pa.array([row["precursor_id"]], pa.int64()), "ion_type": pa.array([free.ion_type[pick]], pa.int8()),
        "position": pa.array([free.position[pick]], pa.int32()), "charge": pa.array([1], pa.int32()),
        "loss_id": ["NH3"], "mz_theoretical": [float(free.mz[pick])], "intensity_predicted": [0.75],
    })  # fmt: skip
    fragments = pa.concat_tables([predicted.fragments, extra], promote_options="default")
    coverage = {**predicted.metadata_extras[COVERAGE_KEY], "losses": ["NH3"]}
    lib = replace(predicted, fragments=fragments, metadata_extras={**predicted.metadata_extras, COVERAGE_KEY: coverage})
    targets = targets_from_library(lib).slice(0, 1)
    plan = compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(lib))
    names = [None, *plan.fragments.loss_ids]
    got = {(int(t), int(p), int(z), names[int(k)]): float(a) for t, p, z, k, a in zip(
        plan.fragments.ion_type, plan.fragments.position, plan.fragments.fragment_charge, plan.fragments.loss,
        plan.fragments.abundance)}  # fmt: skip
    ion = (int(free.ion_type[pick]), int(free.position[pick]), 1, "NH3")
    assert got[ion] == pytest.approx(0.75 / (1.5 * row["charge"] / 10 + 0.75))
    # Without the model naming the loss, the rules place no such channel.
    plain = compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(predicted))
    assert plain.fragments.loss_ids == ()


def test_a_library_ion_that_is_no_channel_of_its_query_is_refused(predicted):
    """An intact ion inside the settings that the query's ladder lacks means
    the library and the query describe different molecules."""
    rows = predicted.fragments.to_pylist()
    bad = dict(rows[0], position=99)
    lib = replace(predicted, fragments=pa.Table.from_pylist(rows + [bad], schema=predicted.fragments.schema))
    with pytest.raises(ValueError, match="match no channel"):
        compile_queries(targets_from_library(lib), levels=[2], fragment_abundance=LibraryFragments.from_library(lib))
    off = dict(rows[0], mz_theoretical=rows[0]["mz_theoretical"] + 0.01)
    lib = replace(predicted, fragments=pa.Table.from_pylist([off] + rows[1:], schema=predicted.fragments.schema))
    with pytest.raises(ValueError, match="different molecules"):
        compile_queries(targets_from_library(lib), levels=[2], fragment_abundance=LibraryFragments.from_library(lib))


def test_library_rows_outside_the_settings_are_counted_not_refused(predicted):
    source = LibraryFragments.from_library(predicted)
    compile_queries(targets_from_library(predicted), levels=[2], channels=ChannelSettings(ion_types=(IonType.Y,)),
                    fragment_abundance=source)  # fmt: skip
    b_rows = int(pc.sum(pc.equal(predicted.fragments.column("ion_type"), int(IonType.B))).as_py())
    assert source.report["outside_settings"] == {"ion_type_not_asked": b_rows}


def test_fragments_are_read_from_a_directory_for_the_precursors_asked_only(predicted, tmp_path):
    save_library(predicted, tmp_path / "lib", format="parquet_dir")
    assert load_library(tmp_path / "lib", format="parquet_dir", fragments=False).fragments.num_rows == 0
    targets = load_targets(tmp_path / "lib").slice(0, 2)
    source = LibraryFragments.from_directory(tmp_path / "lib")
    plan = compile_queries(targets, levels=[2], fragment_abundance=source)
    wanted = set(targets.column("precursor_id").to_pylist())
    assert source.report["rows"] == sum(1 for r in predicted.fragments.to_pylist() if r["precursor_id"] in wanted)
    full = compile_queries(targets, levels=[2], fragment_abundance=LibraryFragments.from_library(predicted))
    np.testing.assert_array_equal(plan.fragments.abundance, full.fragments.abundance)


def test_the_command_line_asks_for_library_abundances_and_top_fragments(predicted, tmp_path, capsys):
    from constellation.cli.__main__ import main

    from _massspec_fixtures import dia_scans, write_bundle, write_bundle_manifest

    save_library(predicted, tmp_path / "lib", format="parquet_dir")
    bundle = write_bundle_manifest(write_bundle(tmp_path, "run", dia_scans(cycles=3)))
    common = ["massspec", "chromatogram", "extract", "--peaks-dir", str(bundle), "--targets", str(tmp_path / "lib"),
              "--level", "2", "--tolerance", "0.35", "--tolerance-unit", "Da", "--no-progress"]  # fmt: skip
    assert main([*common, "-o", str(tmp_path / "a"), "--top-fragments", "2"]) == 2
    assert "--fragment-abundance library" in capsys.readouterr().err
    assert main([*common, "-o", str(tmp_path / "b"), "--fragment-abundance", "library", "--top-fragments", "2"]) == 0
    manifest = json.loads((tmp_path / "b" / "manifest.json").read_text())
    meta = manifest.get("metadata", manifest)
    assert meta["channels"]["top_fragments"] == 2 and meta["fragment_abundance"]["source"] == "library"
    assert main([*common, "-o", str(tmp_path / "c"), "--fragment-abundance", "uniform", "--ion-weights", "b=0.5"]) == 0
