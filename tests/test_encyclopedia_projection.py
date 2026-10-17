"""What ``read_encyclopedia`` makes of a 6.9.23 library file.

The rules are in ``massspec.io.encyclopedia._project``. These tests pin
them on files built from the captured 6.9.23 table definitions (no jar,
no data): every stored peak becomes exactly one fragment row; a
precursor's quantity is the jar's own; every score row is kept, decoys
flagged; and the file's own values come through field for field.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow.compute as pc
import pytest

from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.io.encyclopedia import read_encyclopedia
from constellation.massspec.io.encyclopedia._modseq import parse_encyclopedia_modseq
from constellation.massspec.peptide.ions import fragment_ladder
from constellation.massspec.peptide.mz import precursor_mz

from _encyclopedia_fixtures import Entry, build_result_library


def _entry(modseq: str, sequence: str, charge: int, *, extra: tuple[float, ...] = (), **kwargs) -> Entry:
    """An entry whose stored spectrum is its singly-charged b/y ladder
    plus peaks at ``extra`` offsets from the precursor m/z (0.0 is the
    intact precursor, which real results store too)."""
    peptidoform = parse_encyclopedia_modseq(modseq)
    table, _ = fragment_ladder(peptidoform, max_fragment_charge=1, return_tensor=False)
    mz = sorted(table.column("mz_theoretical").to_pylist())
    pmz = float(precursor_mz(peptidoform, charge=charge))
    mz += [pmz + offset for offset in extra]
    return Entry(
        modseq=modseq,
        sequence=sequence,
        charge=charge,
        precursor_mz=pmz,
        mz=mz,
        intensity=[100.0 + i for i in range(len(mz))],
        **kwargs,
    )


@pytest.fixture
def result(tmp_path: Path) -> Path:
    entries = [
        # the intact precursor, and a peak that is nothing we can name
        _entry("PEPTIDEK", "PEPTIDEK", 2, proteins=("P1",), extra=(0.0, 3.3), total_intensity=5.0e5),
        _entry("ELVISLIVESK", "ELVISLIVESK", 2, proteins=("P1", "P2"), total_intensity=2.0e6),
        _entry("ELVISLIVESK", "ELVISLIVESK", 3, proteins=("P1", "P2"), total_intensity=0.0),
        _entry("C[+57.0214635]PEPTIDER", "CPEPTIDER", 2, proteins=("P3",), rt=900.0, score=0.004),
    ]
    return build_result_library(
        tmp_path / "run.elib", entries, decoy_peptides=[("KEDITPEP", "DECOY_P1")]
    )


# ── fragments ──────────────────────────────────────────────────────────


def test_every_stored_peak_becomes_exactly_one_fragment_row(result: Path) -> None:
    read = read_encyclopedia(result)
    fragments = read.library.fragments
    stored = (14 + 2) + 20 + 20 + 16  # b/y ladders of 14, 20, 20, 16 ions, two extra peaks
    assert fragments.num_rows == stored
    from collections import Counter

    assert Counter(fragments.column("precursor_id").to_pylist()) == {0: 16, 1: 20, 2: 20, 3: 16}
    assert read.caveats == ()


def test_b_and_y_ions_are_annotated_and_the_rest_is_kept(result: Path) -> None:
    fragments = read_encyclopedia(result).library.fragments
    first = fragments.filter(pc.equal(fragments.column("precursor_id"), 0)).to_pylist()
    ions = [r for r in first if r["ion_type"] is not None]
    assert len(ions) == 14 and all(r["charge"] == 1 and r["annotation"] is None for r in ions)

    (precursor,) = [r for r in first if r["annotation"] is not None]
    assert precursor["annotation"] == "p^2"  # the intact 2+ precursor, in mzPAF
    assert precursor["ion_type"] is None and precursor["position"] is None
    pmz = float(precursor_mz(parse_encyclopedia_modseq("PEPTIDEK"), charge=2))
    assert precursor["mz_theoretical"] == pytest.approx(pmz, abs=1e-6)

    (unknown,) = [r for r in first if r["ion_type"] is None and r["annotation"] is None]
    assert unknown["mz_theoretical"] == pytest.approx(pmz + 3.3)  # kept, as stored
    assert unknown["intensity_predicted"] > 0


def test_a_peak_that_fits_two_ions_is_listed_once_under_the_lower_charge(tmp_path: Path) -> None:
    """A 3+ precursor may have a singly and a doubly (or triply) charged
    fragment at one m/z. The old reader wrote a row for each, repeating
    the intensity."""
    path = build_result_library(tmp_path / "run.elib", [_entry("ELVISLIVESK", "ELVISLIVESK", 3)])
    fragments = read_encyclopedia(path).library.fragments
    assert fragments.num_rows == 20
    assert set(fragments.column("charge").to_pylist()) == {1}


def test_fragments_can_be_skipped(result: Path) -> None:
    read = read_encyclopedia(result, annotate_fragments=False)
    assert read.library.fragments.num_rows == 0
    assert read.library.precursors.num_rows == 4 and read.quant.precursor_quant.num_rows == 4


# ── quantities ─────────────────────────────────────────────────────────


def test_a_precursors_quantity_is_the_jars_own(result: Path) -> None:
    read = read_encyclopedia(result)
    rows = {r["precursor_id"]: r for r in read.quant.precursor_quant.to_pylist()}
    # TotalIntensity, not the sum of the stored spectrum.
    assert [rows[i]["intensity"] for i in range(4)] == [5.0e5, 2.0e6, 0.0, 1.0e6]
    # Identified but not quantified is 0 with 0 quant ions; it is not "unknown".
    assert [rows[i]["n_quant_ions"] for i in range(4)] == [3, 3, 0, 3]
    assert (rows[3]["rt_observed"], rows[3]["rt_start"], rows[3]["rt_end"]) == (900.0, 890.0, 910.0)
    assert not any("precursor_quant.intensity" in c for c in read.caveats)


def test_a_peptides_quantity_sums_its_charge_states(result: Path) -> None:
    read = read_encyclopedia(result)
    peptide = dict(
        zip(
            read.library.peptides.column("sequence").to_pylist(),
            read.library.peptides.column("peptide_id").to_pylist(),
            strict=True,
        )
    )
    abundance = dict(
        zip(
            read.quant.peptide_quant.column("peptide_id").to_pylist(),
            read.quant.peptide_quant.column("abundance").to_pylist(),
            strict=True,
        )
    )
    assert abundance[peptide["PEPTIDEK"]] == 5.0e5
    assert abundance[peptide["ELVISLIVESK"]] == 2.0e6  # 2+ (2e6) and 3+ (0)


def test_quantities_join_across_the_two_n_terminal_spellings(tmp_path: Path) -> None:
    """``entries`` says ``[-17.026549]QLSALHR``, ``peptidequants`` says
    ``Q[-17.026549]LSALHR``: one peptide, and it keeps its quantity."""
    entries = [
        _entry("[-17.026549]QLSALHR", "QLSALHR", 2,
               quant_modseq="Q[-17.026549]LSALHR", total_intensity=3.0e6),
        _entry("[+42.010565]C[+57.0214635]DFTEDK", "CDFTEDK", 2,
               quant_modseq="C[+42.010565][+57.0214635]DFTEDK", total_intensity=4.0e6),
    ]
    read = read_encyclopedia(build_result_library(tmp_path / "run.elib", entries))
    assert read.library.peptides.column("modified_sequence").to_pylist() == [
        "[UNIMOD:28]-QLSALHR",
        "[UNIMOD:1]-C[UNIMOD:4]DFTEDK",
    ]
    assert read.quant.precursor_quant.column("intensity").to_pylist() == [3.0e6, 4.0e6]
    assert read.caveats == ()


def test_a_quantity_row_that_fits_two_precursors_is_not_guessed(tmp_path: Path) -> None:
    """N-terminal acetyl and lysine acetyl on a leading K are two
    precursors in ``entries`` and one spelling in ``peptidequants``."""
    entries = [
        _entry("[+42.010565]KVERPDK", "KVERPDK", 2, quant_modseq="K[+42.010565]VERPDK"),
        _entry("K[+42.010565]VERPDK", "KVERPDK", 2),
    ]
    read = read_encyclopedia(build_result_library(tmp_path / "run.elib", entries))
    assert read.library.peptides.num_rows == 2
    assert read.quant.precursor_quant.column("intensity").to_pylist() == [0.0, 0.0]
    assert read.quant.precursor_quant.column("n_quant_ions").to_pylist() == [None, None]
    (ambiguous,) = [c for c in read.caveats if "fit more than one precursor" in c]
    assert ambiguous.startswith("2 peptidequants rows")
    (missing,) = [c for c in read.caveats if "have no peptidequants row" in c]
    assert missing.startswith("2 precursors")


def test_quantity_rows_that_match_nothing_are_counted(tmp_path: Path) -> None:
    entries = [_entry("PEPTIDEK", "PEPTIDEK", 2, extra_quant_rows=[("PEPT[+79.966331]IDEK", 5.0e5)])]
    read = read_encyclopedia(build_result_library(tmp_path / "run.elib", entries))
    (note,) = [c for c in read.caveats if "matched no precursor" in c]
    assert note.startswith("1 peptidequants rows")
    assert read.quant.precursor_quant.column("intensity").to_pylist() == [1.0e6]


def test_a_library_with_no_quantities_has_no_quant(tmp_path: Path) -> None:
    path = build_result_library(
        tmp_path / "lib.dlib", [_entry("PEPTIDEK", "PEPTIDEK", 2, total_intensity=None)]
    )
    read = read_encyclopedia(path)
    assert read.quant is None and read.caveats == ()
    assert read.library.fragments.num_rows == 14


# ── scores ─────────────────────────────────────────────────────────────


def test_every_score_row_is_kept_and_decoys_are_flagged(result: Path) -> None:
    read = read_encyclopedia(result)
    scores = read.search.precursor_scores.to_pylist()
    assert len(scores) == 5  # four targets and the decoy
    (decoy,) = [s for s in scores if s["is_decoy"]]
    assert decoy["precursor_id"] is None and decoy["modified_sequence"] == "KEDITPEP"
    targets = {s["precursor_id"]: s for s in scores if not s["is_decoy"]}
    assert sorted(targets) == [0, 1, 2, 3]
    assert targets[3]["modified_sequence"] == "C[UNIMOD:4]PEPTIDER" and targets[3]["charge"] == 2
    assert (targets[3]["qvalue"], targets[3]["pep"]) == (0.004, 0.004)
    assert {s["engine"] for s in scores} == {"encyclopedia"}
    # The peptide-level table is the roll-up over targets.
    assert read.search.peptide_scores.num_rows == 3
    assert read.search.protein_scores.num_rows == 3


# ── the file's own values, field for field ─────────────────────────────


def test_counts_and_fields_agree_with_the_file(result: Path) -> None:
    """Not only how many rows: for every precursor, the quantity,
    retention time and scores are the file's own numbers for it."""
    read = read_encyclopedia(result)
    con = sqlite3.connect(result)
    n_targets = con.execute("SELECT COUNT(*) FROM peptidescores WHERE IsDecoy = 0").fetchone()[0]
    assert read.library.precursors.num_rows == con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert read.search.precursor_scores.num_rows == con.execute("SELECT COUNT(*) FROM peptidescores").fetchone()[0]
    assert read.quant.precursor_quant.num_rows == n_targets

    modseq = dict(
        zip(
            read.library.peptides.column("peptide_id").to_pylist(),
            read.library.peptides.column("modified_sequence").to_pylist(),
            strict=True,
        )
    )
    identity = {
        p["precursor_id"]: (modseq[p["peptide_id"]], p["charge"])
        for p in read.library.precursors.to_pylist()
    }
    quant = {identity[r["precursor_id"]]: r for r in read.quant.precursor_quant.to_pylist()}
    score = {
        (s["modified_sequence"], s["charge"]): s
        for s in read.search.precursor_scores.to_pylist()
        if not s["is_decoy"]
    }
    from constellation.core.sequence.proforma import format_proforma

    checked = 0
    for raw, charge, rt, total, start, stop, ions, qvalue, pep in con.execute(
        """SELECT e.PeptideModSeq, e.PrecursorCharge, e.RTInSeconds, q.TotalIntensity,
                  q.RTInSecondsStart, q.RTInSecondsStop, q.NumberOfQuantIons,
                  s.QValue, s.PosteriorErrorProbability
           FROM entries e
           JOIN peptidequants q ON q.PeptideModSeq = e.PeptideModSeq AND q.PrecursorCharge = e.PrecursorCharge
           JOIN peptidescores s ON s.PeptideModSeq = e.PeptideModSeq AND s.PrecursorCharge = e.PrecursorCharge"""
    ):
        key = (format_proforma(parse_encyclopedia_modseq(raw)), charge)
        assert quant[key]["intensity"] == total
        assert quant[key]["rt_observed"] == rt
        assert (quant[key]["rt_start"], quant[key]["rt_end"]) == (start, stop)
        assert quant[key]["n_quant_ions"] == ions
        assert (score[key]["qvalue"], score[key]["pep"]) == (qvalue, pep)
        checked += 1
    assert checked == 4


def test_protein_peptide_links_are_by_sequence_and_exclude_decoys(result: Path) -> None:
    read = read_encyclopedia(result)
    accession = dict(
        zip(
            read.library.proteins.column("protein_id").to_pylist(),
            read.library.proteins.column("accession").to_pylist(),
            strict=True,
        )
    )
    sequence = dict(
        zip(
            read.library.peptides.column("peptide_id").to_pylist(),
            read.library.peptides.column("sequence").to_pylist(),
            strict=True,
        )
    )
    links = {
        (accession[p], sequence[k])
        for p, k in zip(
            read.library.protein_peptide.column("protein_id").to_pylist(),
            read.library.protein_peptide.column("peptide_id").to_pylist(),
            strict=True,
        )
    }
    assert links == {
        ("P1", "PEPTIDEK"),
        ("P1", "ELVISLIVESK"),
        ("P2", "ELVISLIVESK"),
        ("P3", "CPEPTIDER"),
    }
    assert "DECOY_P1" not in accession.values()


# ── binding to an experiment's acquisitions ────────────────────────────


def test_runs_are_bound_to_given_acquisitions_by_name(result: Path) -> None:
    """Inside an experiment the file's run is one of the experiment's
    acquisitions, found by name; it is not numbered afresh."""
    record = lambda name, sha: {"source_file": name, "source_kind": "thermo_acquisition", "source_sha256": sha}  # noqa: E731
    experiment, _ = Acquisitions.empty().register(
        [record("/data/other.raw", "aa"), record("/data/run.raw", "bb")]
    )
    read = read_encyclopedia(result, acquisitions=experiment)
    assert set(read.quant.precursor_quant.column("acquisition_id").to_pylist()) == {1}
    assert set(read.search.precursor_scores.column("acquisition_id").to_pylist()) == {1}
    assert read.quant.acquisitions.ids == [0, 1]

    strangers, _ = Acquisitions.empty().register([record("/data/other.raw", "aa")])
    with pytest.raises(KeyError, match="no acquisition is named 'run.raw'"):
        read_encyclopedia(result, acquisitions=strangers)


# ── the commands show what the reader says ─────────────────────────────


def test_commands_print_and_record_the_caveats(tmp_path: Path, capsys) -> None:
    from constellation.massspec.cli import _report_ingest_caveats

    entries = [_entry("PEPTIDEK", "PEPTIDEK", 2, extra_quant_rows=[("PEPT[+79.966331]IDEK", 5.0e5)])]
    read = read_encyclopedia(build_result_library(tmp_path / "run.elib", entries))
    info: dict[str, object] = {"skipped": False}
    _report_ingest_caveats(read, info)
    assert info["caveats"] == list(read.caveats) and len(read.caveats) == 1
    assert capsys.readouterr().err.count("note: ingest: ") == 1

    quiet: dict[str, object] = {}
    clean = read_encyclopedia(build_result_library(tmp_path / "ok.elib", [_entry("PEPTIDEK", "PEPTIDEK", 2)]))
    _report_ingest_caveats(clean, quiet)
    assert quiet == {} and capsys.readouterr().err == ""
