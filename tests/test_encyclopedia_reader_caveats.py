"""What ``read_encyclopedia`` says about a 6.9.23 search result it cannot fully carry.

The Parquet projection of a result library is a partial copy: only
annotatable fragment peaks are kept, and its precursor-level intensity
is not the jar's quantity. These tests pin that the reader reports both
with counts (rather than silently), that the protein-peptide links are
what the brute-force rule gives, and that the commands record the notes.
All on files built from the captured 6.9.23 table definitions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.io.encyclopedia import read_encyclopedia
from constellation.massspec.io.encyclopedia._modseq import parse_encyclopedia_modseq
from constellation.massspec.peptide.ions import fragment_ladder
from constellation.massspec.peptide.mz import precursor_mz

from _encyclopedia_fixtures import Entry, build_result_library


def _entry(modseq: str, sequence: str, charge: int, *, extra_peaks: int = 0, **kwargs) -> Entry:
    """An entry whose stored spectrum is its singly-charged b/y ladder,
    plus ``extra_peaks`` peaks that are no sequence fragment (the intact
    precursor and neighbours), as real results carry."""
    peptidoform = parse_encyclopedia_modseq(modseq)
    table, _ = fragment_ladder(peptidoform, max_fragment_charge=1, return_tensor=False)
    mz = sorted(table.column("mz_theoretical").to_pylist())
    pmz = float(precursor_mz(peptidoform, charge=charge))
    mz += [pmz + 0.5 * i for i in range(extra_peaks)]
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
        _entry("PEPTIDEK", "PEPTIDEK", 2, proteins=("P1",), extra_peaks=2),
        _entry("ELVISLIVESK", "ELVISLIVESK", 2, proteins=("P1", "P2"), extra_peaks=1),
        _entry("ELVISLIVESK", "ELVISLIVESK", 3, proteins=("P1", "P2")),
        _entry("C[+57.0214635]PEPTIDER", "CPEPTIDER", 2, proteins=("P3",)),
    ]
    return build_result_library(
        tmp_path / "run.elib", entries, decoy_peptides=[("KEDITPEP", "DECOY_P1")]
    )


def test_dropped_fragment_peaks_are_counted(result: Path) -> None:
    read = read_encyclopedia(result)
    # b/y ladders of 14, 20, 20 and 16 ions, plus the three extra peaks.
    stored = 14 + 2 + 20 + 1 + 20 + 16
    (peaks,) = [c for c in read.caveats if "stored fragment peaks" in c]
    assert peaks.startswith(f"3 of {stored} stored fragment peaks (4.1%)")
    assert "20 ppm" in peaks and "b or y ion" in peaks


def test_a_peak_listed_under_two_ions_is_counted_once(result: Path) -> None:
    """ELVISLIVESK has a singly and a doubly charged fragment of the same
    m/z, so one stored peak gives two rows. "Dropped" is counted in
    peaks, not rows, and the repeats are reported."""
    read = read_encyclopedia(result)
    stored, dropped = 73, 3
    rows = read.library.fragments.num_rows
    assert rows > stored - dropped
    (twice,) = [c for c in read.caveats if "repeat a stored peak" in c]
    assert twice.startswith(f"{rows - (stored - dropped)} fragment rows repeat")


def test_no_peak_caveat_when_every_peak_is_carried(tmp_path: Path) -> None:
    path = build_result_library(tmp_path / "run.elib", [_entry("PEPTIDEK", "PEPTIDEK", 2)])
    read = read_encyclopedia(path)
    assert not any("stored fragment peaks" in c for c in read.caveats)
    # ...and none at all about peaks when fragments were not asked for.
    assert not any(
        "stored fragment peaks" in c
        for c in read_encyclopedia(path, annotate_fragments=False).caveats
    )


def test_precursor_intensity_caveat_states_what_the_number_is(result: Path) -> None:
    read = read_encyclopedia(result)
    (note,) = [c for c in read.caveats if c.startswith("precursor_quant.intensity")]
    assert "not the quantity EncyclopeDIA computed" in note
    assert "peptide_quant.abundance" in note
    # It is what the note says: the stored spectrum's sum, not TotalIntensity (1e6).
    intensities = read.quant.precursor_quant.column("intensity").to_pylist()
    assert all(i != 1.0e6 for i in intensities)
    # The jar's number is at the peptide level, summed across charge states.
    by_peptide = dict(
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
    assert abundance[by_peptide["PEPTIDEK"]] == 1.0e6
    assert abundance[by_peptide["ELVISLIVESK"]] == 2.0e6  # 2+ and 3+


def test_unmatched_quantity_rows_are_counted(tmp_path: Path) -> None:
    """A ``peptidequants`` row whose modified sequence is no peptide of
    ``entries`` used to vanish; now it is counted."""
    entries = [
        _entry("PEPTIDEK", "PEPTIDEK", 2, extra_quant_rows=[("PEPT[+79.966331]IDEK", 5.0e5)]),
    ]
    read = read_encyclopedia(build_result_library(tmp_path / "run.elib", entries))
    (note,) = [c for c in read.caveats if "peptidequants rows matched no peptide" in c]
    assert note.startswith("1 peptidequants rows")
    assert read.quant.peptide_quant.num_rows == 1


def test_a_library_without_quantities_has_no_caveats(tmp_path: Path) -> None:
    path = build_result_library(
        tmp_path / "lib.dlib", [_entry("PEPTIDEK", "PEPTIDEK", 2, total_intensity=None)]
    )
    assert read_encyclopedia(path).caveats == ()


def test_protein_peptide_links_match_the_brute_force_rule(result: Path) -> None:
    """Every target (accession, sequence) pair links the protein to each
    peptide with that sequence; decoys are left out by default."""
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


def test_commands_print_and_record_the_caveats(result: Path, capsys) -> None:
    from constellation.massspec.cli import _report_ingest_caveats

    read = read_encyclopedia(result)
    info: dict[str, object] = {"skipped": False}
    _report_ingest_caveats(read, info)
    assert info["caveats"] == list(read.caveats) and len(read.caveats) == 3
    err = capsys.readouterr().err
    assert err.count("note: ingest: ") == 3

    quiet: dict[str, object] = {}
    _report_ingest_caveats(read_encyclopedia(result).__class__(read.library, None, None), quiet)
    assert quiet == {} and capsys.readouterr().err == ""
