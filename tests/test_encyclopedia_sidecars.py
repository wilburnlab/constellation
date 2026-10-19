"""The text files EncyclopeDIA writes beside a result (``io.encyclopedia._sidecar``).

Each file here is written in the layout read off real 6.9.23 output (the
headers are copied from the lab's searches, 2026-10-18), small enough to
check every conversion by hand: minutes to seconds, the unit a mass
error is in, ``?`` and empty fields as null, the results file's
``pi_0`` trailer, ids that spell a decoy ``decoy<modseq>``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from constellation.massspec.acquisitions import ACQUISITION_TABLE, Acquisitions
from constellation.massspec.io.encyclopedia import read_encyclopedia
from constellation.massspec.io.encyclopedia._sidecar import (
    SidecarError,
    peptide_centric_sidecars,
    read_features,
    read_mass_fits,
    read_rt_fits,
    read_scored_precursors,
    read_scribe2_fits,
    read_tsv,
)
from constellation.massspec.library.ids import peptide_id, precursor_id

from _encyclopedia_fixtures import Entry, build_result_library

RUN = "run.raw"
SCORES_HEADER = "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tproteinIds"


def _acquisitions(*names: str) -> Acquisitions:
    return Acquisitions(
        pa.Table.from_pylist(
            [{"acquisition_id": i, "source_file": n, "source_kind": "test", "acquisition_datetime": None}
             for i, n in enumerate(names)],
            schema=ACQUISITION_TABLE,
        )
    )  # fmt: skip


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_sidecar_table_reads_nulls_booleans_and_skips_the_trailer(tmp_path) -> None:
    path = _write(tmp_path / "x.txt", [
        "a\tb\tc", "1.5\ttrue\t?", "2.5\tfalse\t", "pi_0=0.456647",
    ])  # fmt: skip
    t = read_tsv(path)
    assert t.column("a").to_pylist() == [1.5, 2.5]
    assert t.column("b").to_pylist() == [True, False]
    assert t.column("c").null_count == 2
    _write(tmp_path / "bad.txt", ["a\tb", "1\t2\t3"])
    with pytest.raises(SidecarError, match="1 rows do not have the header's columns"):
        read_tsv(tmp_path / "bad.txt")


def test_scored_precursors_every_one_with_its_run_and_decoys_apart(tmp_path) -> None:
    targets = _write(tmp_path / "run.dia.encyclopedia.txt", [
        SCORES_HEADER,
        f"{RUN}:100.5:PEPTIDEK+2\t1.2\t0.001\t0.0001\t-.PEPTIDEK.-\tP1",
        f"{RUN}:200.0:AAC[+57.0214635]K+3\t0.4\t0.05\t0.2\t-.AACK.-\tP2",
        # The same peptide as the next, spelled the other way the jar spells
        # an N-terminal modification: one precursor, its better row kept.
        f"{RUN}:300.0:[+42.010565]MDFQK+2\t0.9\t0.004\t0.001\t-.MDFQK.-\tP3",
        f"{RUN}:301.0:M[+42.010565]DFQK+2\t0.1\t0.2\t0.5\t-.MDFQK.-\tP3",
        "pi_0=0.45",
    ])  # fmt: skip
    decoys = _write(tmp_path / "run.dia.encyclopedia.decoy.txt", [
        SCORES_HEADER, f"{RUN}:150.0:decoyKEDITPEP+2\t-0.5\t0.3\t0.9\t-.KEDITPEP.-\tDECOY_P1",
    ])  # fmt: skip
    t = read_scored_precursors([targets, decoys], acquisitions=_acquisitions("other.raw", RUN), engine="encyclopedia")
    rows = {(r["modified_sequence"], r["charge"], r["is_decoy"]): r for r in t.to_pylist()}
    assert set(rows) == {
        ("PEPTIDEK", 2, False), ("AAC[UNIMOD:4]K", 3, False), ("[UNIMOD:1]-MDFQK", 2, False),
        ("KEDITPEP", 2, True),
    }  # fmt: skip
    assert rows[("[UNIMOD:1]-MDFQK", 2, False)]["qvalue"] == 0.004
    assert {r["acquisition_id"] for r in t.to_pylist()} == {1}
    assert rows[("PEPTIDEK", 2, False)]["score"] == 1.2 and rows[("PEPTIDEK", 2, False)]["pep"] == 0.0001
    assert t.column("precursor_id").null_count == t.num_rows  # the caller's to fill


def test_retention_time_and_mass_fits(tmp_path) -> None:
    stem = tmp_path / "run.dia"
    _write(Path(f"{stem}.encyclopedia.txt.final.rt_fit.txt"), [
        "library\tactual\twarpToactual\tdelta\twithin95p\tisDecoy\tsequence",
        "46.0\t48.0\t49.0\t-1.0\ttrue\tfalse\tLAQLTRV",
        "38.0\t40.0\t40.5\t-0.5\tfalse\tfalse\tAAC[+57.0214635]DLVR",
    ])  # fmt: skip
    _write(Path(f"{stem}.encyclopedia.txt.first.ms1_fit.txt"), [
        "Retention Time (Min)\tMass Error (AMU)\tpredictedMassError\tcorrectedMassError\twithin95p\tisDecoy\tsequence",
        "48.0\t-0.04\t0.01\t-0.05\ttrue\t?\t?",
    ])  # fmt: skip
    _write(Path(f"{stem}.encyclopedia.txt.final.ms2_fit.txt"), [
        "Retention Time (Min)\tMass Error (PPM)\tpredictedMassError\tcorrectedMassError\twithin95p\tisDecoy\tsequence",
        "40.0\t3.0\t1.0\t2.0\tfalse\t?\t?",
    ])  # fmt: skip
    files = peptide_centric_sidecars(f"{stem}.elib")
    assert set(files) == {"rt_fit:final", "ms1_fit:first", "ms2_fit:final"}

    rt = read_rt_fits(files, acquisition_id=7, engine="encyclopedia")
    first = rt.to_pylist()[0]
    assert first["pass"] == "final" and first["acquisition_id"] == 7
    assert (first["library_rt"], first["observed_rt"], first["fitted_rt"], first["residual"]) == (2760.0, 2880.0, 2940.0, -60.0)
    assert first["peptide_id"] == peptide_id("LAQLTRV") and first["charge"] is None
    assert rt.column("modified_sequence").to_pylist()[1] == "AAC[UNIMOD:4]DLVR"
    assert rt.column("within_95").to_pylist() == [True, False]

    mass = read_mass_fits(files, acquisition_id=7, engine="encyclopedia")
    by_level = {r["ms_level"]: r for r in mass.to_pylist()}
    assert by_level[1]["unit"] == "Da" and by_level[1]["pass"] == "first" and by_level[1]["rt"] == 2880.0
    assert by_level[1]["mass_error"] == -0.04 and by_level[1]["modified_sequence"] is None
    assert by_level[2]["unit"] == "ppm" and by_level[2]["corrected_mass_error"] == 2.0


def test_scribetwo_fits_from_its_reports(tmp_path) -> None:
    reports = tmp_path / "run.dia.scribe2"
    reports.mkdir()
    header = ("pass\tsequence\tcharge\trtLibrary\trtActual\trtPredicted\trtDelta\trtWithin95p\tms1RT\t"
              "ms1MassError\tms1PredictedMassError\tms1Delta\tms1Within95p\tms2RT\tms2MassError\t"
              "ms2PredictedMassError\tms2Delta\tms2Within95p")  # fmt: skip
    _write(reports / "run.dia.scribe2.main.txt", [
        header,
        "main\tPEPTIDEK\t2\t10.0\t50.0\t51.0\t-1.0\ttrue\t50.0\t0.09\t-0.003\t0.093\ttrue\t50.0\t-0.03\t-0.01\t-0.02\ttrue",
        # No MS1 measurement for this one: no MS1 row.
        "main\tAAC[+57.0214635]K\t3\t9.0\t45.0\t46.0\t-1.0\tfalse\t\t\t\t\t\t45.0\t0.001\t-0.008\t0.009\ttrue",
    ])  # fmt: skip
    rt, mass = read_scribe2_fits(reports, acquisition_id=0, units=("Da", "Da"))
    assert rt.num_rows == 2 and set(rt.column("engine").to_pylist()) == {"scribetwo"}
    assert rt.column("charge").to_pylist() == [2, 3] and rt.column("observed_rt").to_pylist()[0] == 3000.0
    assert pc.value_counts(mass.column("ms_level")).to_pylist() == [
        {"values": 1, "counts": 1}, {"values": 2, "counts": 2},
    ]  # fmt: skip
    assert mass.column("peptide_id").to_pylist()[0] == peptide_id("PEPTIDEK")


def test_the_feature_matrix_keeps_every_feature(tmp_path) -> None:
    path = _write(tmp_path / "run.dia.features.txt", [
        "id\tLabel\tScanNr\tprimary\tevalue\tdeltaRT\tsequence\tProteins",
        f"{RUN}:100.5:PEPTIDEK+2\t1\t2057\t5.4\t0.7\t49.7\t-.PEPTIDEK.-\tP1",
        f"{RUN}:150.0:decoyKEDITPEP+2\t-1\t11\t1.0\t0.1\t3.0\t-.KEDITPEP.-\tDECOY_P1",
    ])  # fmt: skip
    t = read_features(path, acquisitions=_acquisitions(RUN), engine="encyclopedia")
    assert t.column_names == [
        "acquisition_id", "engine", "modified_sequence", "charge", "rt", "is_decoy", "precursor_id",
        "scan_nr", "primary", "evalue", "deltaRT", "proteins",
    ]  # fmt: skip
    assert t.column("is_decoy").to_pylist() == [False, True]
    assert t.column("precursor_id").to_pylist() == [precursor_id("PEPTIDEK", 2), None]
    assert t.column("primary").type == pa.float64() and t.column("scan_nr").to_pylist() == [2057, 11]


def _entry(modseq: str, sequence: str, charge: int) -> Entry:
    return Entry(modseq=modseq, sequence=sequence, charge=charge, precursor_mz=500.0,
                 mz=[200.0, 300.0], intensity=[1.0, 2.0], source=RUN)  # fmt: skip


def test_read_encyclopedia_completes_the_scores_from_the_sidecars(tmp_path) -> None:
    """The result library keeps the precursors within the threshold;
    Percolator's output beside it has every one scored."""
    library = build_result_library(
        tmp_path / "run.dia.elib",
        [_entry("PEPTIDEK", "PEPTIDEK", 2)],
        decoy_peptides=[("KEDITPEP", "DECOY_1")],
        source=RUN,
        metadata={"sourcefile": RUN, f"{RUN} search type": "EncyclopeDIA"},
    )
    _write(tmp_path / "run.dia.encyclopedia.txt", [
        SCORES_HEADER,
        f"{RUN}:600.0:PEPTIDEK+2\t1.5\t0.001\t0.001\t-.PEPTIDEK.-\tP1",
        f"{RUN}:700.0:ELVISK+2\t-0.2\t0.4\t0.8\t-.ELVISK.-\tP2",
        "pi_0=0.5",
    ])  # fmt: skip
    _write(tmp_path / "run.dia.encyclopedia.decoy.txt", [
        SCORES_HEADER, f"{RUN}:650.0:decoyKEDITPEP+2\t-1.0\t0.5\t0.5\t-.KEDITPEP.-\tDECOY_1",
    ])  # fmt: skip
    _write(tmp_path / "run.dia.encyclopedia.txt.final.rt_fit.txt", [
        "library\tactual\twarpToactual\tdelta\twithin95p\tisDecoy\tsequence",
        "10.0\t10.0\t10.0\t0.0\ttrue\tfalse\tPEPTIDEK",
    ])  # fmt: skip
    read = read_encyclopedia(library, annotate_fragments=False)
    scores = {(r["modified_sequence"], r["is_decoy"]): r for r in read.search.precursor_scores.to_pylist()}
    assert set(scores) == {("PEPTIDEK", False), ("ELVISK", False), ("KEDITPEP", True)}
    assert scores[("PEPTIDEK", False)]["precursor_id"] is not None  # the library's precursor
    assert scores[("ELVISK", False)]["precursor_id"] is None  # in no table of the library
    assert scores[("ELVISK", False)]["qvalue"] == 0.4 and scores[("PEPTIDEK", False)]["score"] == 1.5
    assert set(read.sidecars) == {"rt_alignment"} and read.sidecars["rt_alignment"].num_rows == 1
    assert read.sidecars["rt_alignment"].column("acquisition_id").to_pylist() == [0]

    plain = read_encyclopedia(library, annotate_fragments=False, sidecars=False)
    assert plain.sidecars == {} and plain.search.precursor_scores.num_rows == 2  # target + decoy
