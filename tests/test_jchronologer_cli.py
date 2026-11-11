"""``massspec predict-library --backend jchronologer``.

The CLI runs on a fake bundle (``tests/_jchronologer_fixtures.py``: the
real preprocessing files, stand-in models), so its refusals, resume,
stage key, manifest and output are tested without the jar. One test runs
the real models where EncyclopeDIA is installed.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pyarrow.compute as pc
import pytest

from constellation.cli.__main__ import _build_parser
from constellation.massspec.library import load_library, save_library
from constellation.massspec.library.decoys import DecoySet
from constellation.massspec.library.search_space import Digestion, SearchSpaceSettings, build_search_space
from _jchronologer_fixtures import FakeBundle


def _run(*argv: str) -> int:
    args = _build_parser().parse_args(["massspec", "predict-library", *argv])
    return args.func(args)


@pytest.fixture
def fake(monkeypatch):
    bundle = FakeBundle()
    import constellation.massspec.library.jchronologer as package

    monkeypatch.setattr(package, "locate_bundle", lambda jar=None: bundle)
    return bundle


@pytest.fixture
def search_space(tmp_path):
    fasta = tmp_path / "proteins.fa"
    fasta.write_text(">P1\nMPEPTIDEKLVNELTEFAKSAMPLERHLVDEPQNLIK\n>P2\nAEFVEVTKYLYEIARQTALVELLKTMTPEPTIDER\n")
    settings = SearchSpaceSettings(
        digestion=Digestion(missed_cleavages=0, excise_initiator_met=False),
        charges=(2, 3), min_mz=0, max_mz=1e6, decoys=(DecoySet(),),
    )
    path = tmp_path / "search_space"
    save_library(build_search_space([fasta], settings), path, format="parquet_dir")
    return path


def _jc(search_space, out, *more):
    return _run("--backend", "jchronologer", "--from-library", str(search_space), "--output-dir", str(out), "--no-progress", *more)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--fasta", "x.fa"], "massspec search-space"),
        ([], "exactly one of --from-library"),
        (["--from-library", "a", "--peptides", "b"], "exactly one of --from-library"),
        (["--from-library", "a", "--ms2-model", "Prosit_2020_intensity_CID"], "--ms2-model"),
        (["--from-library", "a", "--ptm-oxidation", "off"], "--ptm-oxidation"),
        (["--from-library", "a", "--no-charge-states", "--min-charge-probability", "0.1"], "needs charge states"),
        (["--from-library", "a", "--collision-energy", "25,30"], "one collision energy"),
        (["--from-library", "a", "--output-library", "x"], "library_pqdir"),
    ],
)
def test_refusals(tmp_path, capsys, argv, message):
    assert _run("--backend", "jchronologer", "--output-dir", str(tmp_path / "out"), *argv) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("backend", ["koina", "encyclopedia"])
def test_this_backends_flags_are_refused_by_the_others(tmp_path, capsys, backend):
    assert _run("--backend", backend, "--output-dir", str(tmp_path), "--threads", "4") == 1
    assert "--threads" in capsys.readouterr().err


def test_a_search_space_is_predicted_and_written(fake, search_space, tmp_path):
    out = tmp_path / "out"
    assert _jc(search_space, out, "--threads", "2", "--collision-energy", "27") == 0
    library = load_library(out / "library_pqdir")
    source = load_library(search_space)
    assert library.precursors.num_rows == source.precursors.num_rows
    assert library.fragments.num_rows > 0
    assert set(library.fragments.column("precursor_id").to_pylist()) <= set(library.precursors.column("precursor_id").to_pylist())
    assert pc.min(library.precursors.column("rt_predicted")).as_py() > 0
    assert library.precursors.column("charge_probability").null_count == 0
    assert library.peptides.column("decoy_of").equals(source.peptides.column("decoy_of"))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["backend"] == "jchronologer"
    assert manifest["runtime"]["threads"] == 2 and manifest["runtime"]["batch_size"] == 2048
    prediction = manifest["prediction"]
    assert prediction["collision_energy"]["origin"] == "scaled_from_reference"
    assert prediction["collision_energy"]["nce"] == 27.0
    assert set(prediction["predictors"]) == {"fragments", "retention_time", "charge_state", "ccs"}
    assert (out / "_SUCCESS").read_text() == manifest["stage_key"]
    # Written sorted by precursor, so a compile reads a range of them from a few row groups.
    import pyarrow.parquet as pq

    from constellation.massspec.library.fragment_abundance import _sorted_by_precursor

    written = pq.read_table(out / "library_pqdir" / "fragments.parquet")
    ids = written.column("precursor_id").to_numpy()
    assert (ids[1:] >= ids[:-1]).all()
    assert _sorted_by_precursor(pq.ParquetFile(out / "library_pqdir" / "fragments.parquet"))


def test_a_search_space_older_code_made_is_predicted_with_a_warning(fake, search_space, tmp_path):
    """Until 2026-11-11 one with no layout (built before PR #110, with no
    decoy proteins) was predicted without a word (review of 2026-11-10, B8)."""
    import warnings

    source = load_library(search_space)
    record = {k: v for k, v in source.metadata["x.constellation.search_space"].items() if k != "layout"}
    old = tmp_path / "old_space"
    save_library(source.with_metadata({"x.constellation.search_space": record}), old, format="parquet_dir")
    with pytest.warns(UserWarning, match=r"search space records no layout.*rebuild it with `massspec search-space`"):
        assert _jc(old, tmp_path / "old_out") == 0
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert _jc(search_space, tmp_path / "out") == 0
    assert not [w for w in seen if "layout" in str(w.message)]


def test_resume_reuses_only_the_same_prediction(fake, search_space, tmp_path, capsys):
    out = tmp_path / "out"
    assert _jc(search_space, out) == 0
    key = (out / "_SUCCESS").read_text()
    assert _jc(search_space, out) == 1
    assert "--resume" in capsys.readouterr().err
    assert _jc(search_space, out, "--resume") == 0
    assert "already complete" in capsys.readouterr().out
    # Another energy is another prediction: refused without --resume, redone with it.
    assert _jc(search_space, out, "--collision-energy", "25") == 1
    assert "other inputs or settings" in capsys.readouterr().err
    assert _jc(search_space, out, "--collision-energy", "25", "--resume") == 0
    assert (out / "_SUCCESS").read_text() != key
    # Threads are recorded, not keyed.
    assert _jc(search_space, out, "--collision-energy", "25", "--threads", "3", "--resume") == 0
    assert "already complete" in capsys.readouterr().out


def test_resume_keeps_the_export_current(fake, search_space, tmp_path, capsys):
    out, dlib = tmp_path / "out", tmp_path / "lib.dlib"

    def exports() -> dict:
        return json.loads((out / "manifest.json").read_text())["exports"]

    def key() -> str:
        return (out / "_SUCCESS").read_text()

    assert _jc(search_space, out, "--output-dlib", str(dlib)) == 0
    assert exports() == {str(dlib): key()}
    # Other settings, resumed: the export is replaced with the prediction
    # (it used to stop at the end on FileExistsError, beside the new library).
    assert _jc(search_space, out, "--collision-energy", "25", "--resume", "--output-dlib", str(dlib)) == 0
    assert exports() == {str(dlib): key()}
    capsys.readouterr()
    # A finished prediction makes the export asked for now ...
    other = tmp_path / "other.dlib"
    assert _jc(search_space, out, "--collision-energy", "25", "--resume", "--output-dlib", str(other)) == 0
    assert "exported" in capsys.readouterr().out and other.is_file()
    assert exports() == {str(dlib): key(), str(other): key()}
    # ... and again when the file holds an earlier prediction, but not while it is current.
    assert _jc(search_space, out, "--resume") == 0
    assert exports()[str(dlib)] != key()
    capsys.readouterr()
    assert _jc(search_space, out, "--resume", "--output-dlib", str(dlib)) == 0
    assert "exported" in capsys.readouterr().out and exports()[str(dlib)] == key()
    assert _jc(search_space, out, "--resume", "--output-dlib", str(dlib)) == 0
    assert "exported" not in capsys.readouterr().out
    # A file it did not write is refused before anything runs, and left alone.
    foreign = tmp_path / "foreign.dlib"
    foreign.write_bytes(b"not ours")
    before = key()
    assert _jc(search_space, out, "--collision-energy", "27", "--resume", "--output-dlib", str(foreign)) == 1
    assert "was not exported from" in capsys.readouterr().err
    assert foreign.read_bytes() == b"not ours" and key() == before
    assert _jc(search_space, out, "--resume", "--output-dlib", str(foreign)) == 1


def test_a_property_alone_keeps_the_librarys_own_fragments(fake, search_space, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert _jc(search_space, first) == 0
    assert _jc(first / "library_pqdir", second, "--ms2-model", "none", "--rt-model", "none", "--no-charge-states") == 0
    before, after = load_library(first / "library_pqdir"), load_library(second / "library_pqdir")
    assert after.fragments.num_rows == before.fragments.num_rows > 0
    assert after.fragments.equals(before.fragments)
    made_by = after.metadata_extras["x.massspec.prediction.predictors"]
    assert made_by["fragments"]["model"] == "cartographer" and made_by["ccs"]["model"] == "sculptor"
    assert after.metadata_extras["x.massspec.prediction.collision_energy"]["origin"] == "scaled_from_reference"


def test_an_energy_table_must_cover_every_charge(fake, search_space, tmp_path, capsys):
    assert _jc(search_space, tmp_path / "a", "--collision-energy-by-charge", "2=25") == 1
    assert "no energy for charge 3" in capsys.readouterr().err
    assert _jc(search_space, tmp_path / "b", "--collision-energy-by-charge", "2=25,3=28") == 0
    manifest = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert manifest["prediction"]["collision_energy"]["table"] == {"2": 25.0, "3": 28.0}


def test_properties_can_be_left_out_and_charges_filtered(fake, search_space, tmp_path):
    assert _jc(search_space, tmp_path / "a", "--no-ccs", "--no-charge-states") == 0
    library = load_library(tmp_path / "a" / "library_pqdir")
    assert set(library.precursors.column("ccs_predicted").to_pylist()) == {-1.0}
    assert library.precursors.column("charge_probability").null_count == library.precursors.num_rows

    assert _jc(search_space, tmp_path / "b", "--min-charge-probability", "0.3") == 0
    filtered = load_library(tmp_path / "b" / "library_pqdir")
    report = json.loads((tmp_path / "b" / "manifest.json").read_text())["prediction"]
    kept = filtered.precursors.join(filtered.peptides.select(["peptide_id", "is_decoy"]), keys="peptide_id")
    # Targets and decoys alike are judged on their own predictions (until
    # 2026-11-07 decoys were never judged and followed their targets).
    for decoy in (False, True):
        part = kept.filter(pc.equal(kept.column("is_decoy"), decoy))
        assert part.num_rows and pc.min(part.column("charge_probability")).as_py() >= 0.3
    assert report["refused"]["charge_state"]["below_min_charge_probability"] > 0
    assert report["dropped_targets"] > 0 and report["dropped_decoys"] > 0
    assert "dropped_with_partner" not in report


def test_a_peptide_list(fake, tmp_path):
    peptides = tmp_path / "peptides.tsv"
    peptides.write_text("modified_sequence\tcharge\nLVNELTEFAK\t2\nYLYEIAR\t\n")
    out = tmp_path / "out"
    assert _run("--backend", "jchronologer", "--peptides", str(peptides), "--min-charge", "2", "--max-charge", "3",
                "--output-dir", str(out), "--no-progress") == 0
    library = load_library(out / "library_pqdir")
    assert sorted(library.precursors.column("charge").to_pylist()) == [2, 2, 3]


def test_the_real_models_on_a_search_space(search_space, tmp_path):
    from constellation.massspec.library.jchronologer import locate_bundle
    from constellation.massspec.tokenize.jchronologer import UnknownModelError
    from constellation.thirdparty.registry import ToolNotFoundError

    try:
        locate_bundle()
    except (ToolNotFoundError, FileNotFoundError, UnknownModelError) as e:
        pytest.skip(f"JChronologer is not installed here: {e}")
    out = tmp_path / "out"
    assert _jc(search_space, out, "--threads", "2", "--output-dlib", str(tmp_path / "lib.dlib")) == 0
    library = load_library(out / "library_pqdir")
    assert library.fragments.num_rows > 10 * library.precursors.num_rows
    assert (tmp_path / "lib.dlib").stat().st_size > 0
    assert replace(library, fragments=library.fragments).n_precursors == library.precursors.num_rows


def test_doctor_says_whether_the_models_can_be_read(monkeypatch, tmp_path):
    import constellation.cli.__main__ as cli
    import constellation.massspec.library.jchronologer as package
    from constellation.massspec.tokenize.jchronologer import UnknownModelError
    from constellation.thirdparty.registry import ToolHandle, registered

    spec = next(s for s in registered() if s.name == "encyclopedia")

    def install(version: str) -> ToolHandle:
        return ToolHandle(spec, tmp_path / f"encyclopedia-{version}.jar", "env", version)

    monkeypatch.setattr(cli, "try_find", lambda name: None)
    assert cli._doctor_jchronologer_rows() == []
    # The row reads the install the encyclopedia row found, never a second
    # lookup (which, with no EncyclopeDIA on the machine, raised).
    asked: list = []

    def unknown(jar=None, *, encyclopedia=None):
        asked.append(encyclopedia)
        raise UnknownModelError("JChronologer 9.9 is not one Constellation was checked against")

    monkeypatch.setattr(cli, "try_find", lambda name: install("6.9.23"))
    monkeypatch.setattr(package, "locate_bundle", unknown)
    assert cli._doctor_jchronologer_rows()[0][:2] == ("jchronologer", "unverified")
    assert asked[0].path == tmp_path / "encyclopedia-6.9.23.jar"

    def missing(jar=None, *, encyclopedia=None):
        raise FileNotFoundError("names 0 JChronologer jars")

    monkeypatch.setattr(package, "locate_bundle", missing)
    assert cli._doctor_jchronologer_rows()[0][:2] == ("jchronologer", "missing")

    monkeypatch.setattr(cli, "try_find", lambda name: install("6.5.15"))
    row = cli._doctor_jchronologer_rows()[0]
    assert row[:2] == ("jchronologer", "missing") and "older than 6.9.23" in row[3]
