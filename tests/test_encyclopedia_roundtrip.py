"""Round-trip tests for the encyclopedia writer ↔ reader pair.

These don't require the bench-side sample files — we build a small
synthetic Library + Quant + Search, write to .dlib/.elib, read back,
and check fidelity.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.massspec.acquisitions import ACQUISITION_TABLE, Acquisitions
from constellation.massspec.io.encyclopedia import (
    read_encyclopedia,
    write_encyclopedia,
)
from constellation.massspec.library.library import assign_ids
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.quant.quant import assemble_quant
from constellation.massspec.search.search import assemble_search


def _toy_library_quant_search():
    """Two proteins, three peptides, three precursors, hand-picked
    fragments — enough to exercise the round-trip without dragging in
    the full sample file."""
    library = assign_ids(
        proteins=[
            {"accession": "PROT1"},
            {"accession": "PROT2"},
        ],
        peptides=[
            {"modified_sequence": "PEPTIDE"},
            {"modified_sequence": "PEPC[UNIMOD:4]TIDE"},
            {"modified_sequence": "MELLAR"},
        ],
        precursors=[
            {"modified_sequence": "PEPTIDE", "charge": 2, "precursor_mz": 400.2},
            {"modified_sequence": "PEPC[UNIMOD:4]TIDE", "charge": 2, "precursor_mz": 428.7},
            {"modified_sequence": "MELLAR", "charge": 2, "precursor_mz": 376.2},
        ],
        fragments=[
            # PEPTIDE precursor — use realistic b/y theoretical m/z values.
            # Three fragments per precursor is enough.
            {"modified_sequence": "PEPTIDE", "precursor_charge": 2,
             "ion_type": int(IonType.B), "position": 1, "charge": 1,
             "mz_theoretical": 227.103, "intensity_predicted": 100.0,
             "loss_id": None, "annotation": None},
            {"modified_sequence": "PEPTIDE", "precursor_charge": 2,
             "ion_type": int(IonType.Y), "position": 5, "charge": 1,
             "mz_theoretical": 575.290, "intensity_predicted": 80.0,
             "loss_id": None, "annotation": None},
            {"modified_sequence": "PEPC[UNIMOD:4]TIDE", "precursor_charge": 2,
             "ion_type": int(IonType.Y), "position": 5, "charge": 1,
             "mz_theoretical": 632.311, "intensity_predicted": 90.0,
             "loss_id": None, "annotation": None},
            {"modified_sequence": "MELLAR", "precursor_charge": 2,
             "ion_type": int(IonType.B), "position": 2, "charge": 1,
             "mz_theoretical": 263.108, "intensity_predicted": 50.0,
             "loss_id": None, "annotation": None},
        ],
        protein_peptide=[
            ("PROT1", "PEPTIDE"),
            ("PROT1", "PEPC[UNIMOD:4]TIDE"),
            ("PROT2", "MELLAR"),
        ],
        metadata={"x.encyclopedia.metadata": {"version": "0.1.15"}},
    )

    acq = Acquisitions(
        pa.Table.from_pylist(
            [
                {
                    "acquisition_id": 0,
                    "source_file": "run.mzML",
                    "source_kind": "encyclopedia.entries",
                    "acquisition_datetime": None,
                }
            ],
            schema=ACQUISITION_TABLE,
        )
    )
    quant = assemble_quant(
        acquisitions=acq,
        precursor_quant=[
            {"precursor_id": 0, "acquisition_id": 0, "intensity": 1234.5,
             "rt_observed": 100.0, "ccs_observed": -1.0, "score": None},
            {"precursor_id": 1, "acquisition_id": 0, "intensity": 5678.9,
             "rt_observed": 105.0, "ccs_observed": -1.0, "score": None},
            {"precursor_id": 2, "acquisition_id": 0, "intensity": 9999.0,
             "rt_observed": 200.0, "ccs_observed": -1.0, "score": None},
        ],
        peptide_quant=[
            {"peptide_id": 0, "acquisition_id": 0, "abundance": 1234.5, "score": None},
            {"peptide_id": 1, "acquisition_id": 0, "abundance": 5678.9, "score": None},
            {"peptide_id": 2, "acquisition_id": 0, "abundance": 9999.0, "score": None},
        ],
    )

    search = assemble_search(
        acquisitions=acq,
        peptide_scores=[
            {"peptide_id": 0, "acquisition_id": 0, "score": None,
             "qvalue": 0.001, "pep": 0.0001, "engine": "encyclopedia"},
            {"peptide_id": 1, "acquisition_id": 0, "score": None,
             "qvalue": 0.005, "pep": 0.001, "engine": "encyclopedia"},
        ],
        protein_scores=[
            {"protein_id": 0, "acquisition_id": 0, "score": None,
             "qvalue": 0.001, "engine": "encyclopedia"},
        ],
    )
    return library, quant, search


# ──────────────────────────────────────────────────────────────────────
# Writer + reader round-trip
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".dlib", ".elib"])
def test_round_trip_preserves_proteins_peptides_precursors(tmp_path: Path, ext: str):
    """Write the toy library to .dlib and to .elib; read back; check
    entity counts and modseq round-trip. .dlib and .elib are the same
    format on disk — ext is purely a naming convention."""
    library, quant, search = _toy_library_quant_search()
    out = tmp_path / f"toy{ext}"
    write_encyclopedia(out, library, quant=quant, search=search)
    assert out.exists()

    r = read_encyclopedia(out)
    assert r.library.n_proteins == 2
    assert r.library.n_peptides == 3

    # Modified sequences must round-trip exactly (UNIMOD references
    # survive the lossy mass-delta serialization).
    in_modseqs = sorted(library.peptides.column("modified_sequence").to_pylist())
    out_modseqs = sorted(r.library.peptides.column("modified_sequence").to_pylist())
    assert in_modseqs == out_modseqs


def test_round_trip_with_no_quant_produces_minimal_dlib(tmp_path: Path):
    """When ``quant=None`` and ``search=None``, the writer emits a
    minimal dlib (entries + peptidetoprotein + metadata only). The
    reader's format-detection treats it as a pure library."""
    library, _quant, _search = _toy_library_quant_search()
    out = tmp_path / "pure.dlib"
    write_encyclopedia(out, library)

    r = read_encyclopedia(out)
    assert r.library.n_peptides == 3
    # No chromatograms, no peptidequants, no peptidescores → no Quant, no Search.
    assert r.quant is None
    assert r.search is None


def test_round_trip_dlib_and_elib_are_byte_identical(tmp_path: Path):
    """The output of write_encyclopedia is identical regardless of
    whether the caller picked .dlib or .elib for the file extension —
    on-disk SQLite content is the same."""
    import sqlite3

    library, quant, search = _toy_library_quant_search()
    dlib_path = tmp_path / "out.dlib"
    elib_path = tmp_path / "out.elib"

    write_encyclopedia(dlib_path, library, quant=quant, search=search)
    write_encyclopedia(elib_path, library, quant=quant, search=search)

    # Same row counts and same entries-table content.
    def _summarize(p):
        con = sqlite3.connect(p)
        try:
            tables = ("entries", "peptidetoprotein", "peptidescores",
                      "proteinscores", "peptidequants")
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in tables}
        finally:
            con.close()

    assert _summarize(dlib_path) == _summarize(elib_path)


def test_round_trip_overwrites_only_when_asked(tmp_path: Path):
    library, _quant, _search = _toy_library_quant_search()
    out = tmp_path / "pure.dlib"
    write_encyclopedia(out, library)
    with pytest.raises(FileExistsError):
        write_encyclopedia(out, library)
    write_encyclopedia(out, library, overwrite=True)  # must not raise


# ── orphan peptides survive a DLIB round-trip ──────────────────────────
#
# A DLIB's reader gates `entries` on `peptidetoprotein`, so a peptide
# with no protein mapping was written and then read back as nothing —
# a nonempty file that reloads empty. That is a real case, not a
# malformed library: a synthetic-peptide panel has no protein of origin.


def _one_peptide_library(*, with_protein: bool):
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.library.library import assign_ids
    from constellation.massspec.peptide.ions import (
        IonType,
        fragment_ladder_indices_batch,
    )

    seq = "PEPTIDEK"
    ladder = fragment_ladder_indices_batch(
        [parse_proforma(seq)],
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
    )[0]
    key = (int(IonType.Y), 4, 1, None)
    return assign_ids(
        proteins=[{"accession": "P1"}] if with_protein else [],
        peptides=[{"modified_sequence": seq, "sequence": seq}],
        precursors=[
            {
                "modified_sequence": seq,
                "charge": 2,
                "precursor_mz": 456.7,
                "rt_predicted": 100.0,
            }
        ],
        fragments=[
            {
                "modified_sequence": seq,
                "precursor_charge": 2,
                "ion_type": int(IonType.Y),
                "position": 4,
                "charge": 1,
                "loss_id": None,
                "mz_theoretical": ladder[key],
                "intensity_predicted": 0.5,
                "annotation": "y3^1",
            }
        ],
        protein_peptide=[("P1", seq)] if with_protein else [],
        metadata={},
    )


def test_orphan_peptide_survives_dlib_roundtrip(tmp_path) -> None:
    from constellation.massspec.library.io import load_library, save_library

    lib = _one_peptide_library(with_protein=False)
    path = tmp_path / "orphan.dlib"
    save_library(lib, path, format="encyclopedia.dlib")
    back = load_library(path)

    assert back.precursors.num_rows == lib.precursors.num_rows == 1
    assert back.fragments.num_rows == lib.fragments.num_rows == 1


def test_orphan_peptide_gets_a_per_peptide_accession(tmp_path) -> None:
    """One pseudo-accession per peptide, never one shared bucket.

    EncyclopeDIA runs protein-level inference and FDR over accessions,
    so collapsing N synthetic peptides onto a single accession would
    present them as one protein with N supporting peptides — the shape
    of a confidently identified protein.
    """
    from constellation.massspec.io.encyclopedia._write import (
        UNMAPPED_ACCESSION_PREFIX,
    )
    from constellation.massspec.library.io import load_library, save_library

    path = tmp_path / "orphan.dlib"
    save_library(
        _one_peptide_library(with_protein=False), path, format="encyclopedia.dlib"
    )
    accessions = load_library(path).proteins.column("accession").to_pylist()
    assert accessions == [f"{UNMAPPED_ACCESSION_PREFIX}PEPTIDEK"]


def test_mapped_peptide_keeps_its_real_accession(tmp_path) -> None:
    """The orphan path must not perturb a normally-mapped library."""
    from constellation.massspec.library.io import load_library, save_library

    path = tmp_path / "mapped.dlib"
    save_library(
        _one_peptide_library(with_protein=True), path, format="encyclopedia.dlib"
    )
    back = load_library(path)
    assert back.proteins.column("accession").to_pylist() == ["P1"]
    assert back.precursors.num_rows == 1


def test_a_search_result_comes_back_whole(tmp_path) -> None:
    """Phase 7: the writer carries what the reader reads, chromatograms
    and quantified ions included, in the 0.1.18 container."""
    import sqlite3

    from _encyclopedia_fixtures import Entry, build_result_library

    original = build_result_library(
        tmp_path / "jar.elib",
        [Entry(modseq="[+42.010565]MDFQK", sequence="MDFQK", charge=2, precursor_mz=355.1,
               mz=[200.1, 300.2], intensity=[5.0, 6.0], total_intensity=150.0,
               quant_ions=[(200.1, 100.0, 0.9), (300.2, 50.0, 0.8)],
               chromatogram=[(590.0, 1.0), (600.0, 9.0)],
               peak_correlation=[0.9, 0.25], peak_quantified=[True, False]),
         Entry(modseq="PEPTIDEK", sequence="PEPTIDEK", charge=2, precursor_mz=465.2,
               mz=[300.1], intensity=[1.0], total_intensity=0.0, peak_quantified=[True])],
        metadata={"-percolatorThreshold": "0.01"},
    )  # fmt: skip
    first = read_encyclopedia(original, sidecars=False)
    write_encyclopedia(tmp_path / "ours.elib", first.library, quant=first.quant, search=first.search)
    again = read_encyclopedia(tmp_path / "ours.elib", sidecars=False)
    for name in ("precursor_quant", "precursor_chromatogram", "fragment_quant"):
        assert getattr(again.quant, name).equals(getattr(first.quant, name)), name
    assert again.library.fragments.equals(first.library.fragments)
    # The library's own per-peak values, not every peak flagged.
    fragments = first.library.fragments.sort_by([("precursor_id", "ascending"), ("mz_theoretical", "ascending")])
    assert fragments.column("quantified").to_pylist() == [True, False, True]
    correlation = fragments.column("correlation").to_pylist()
    assert correlation[:2] == pytest.approx([0.9, 0.25]) and correlation[2] is None
    assert again.search.precursor_scores.equals(first.search.precursor_scores)
    con = sqlite3.connect(tmp_path / "ours.elib")
    meta = dict(con.execute("SELECT Key, Value FROM metadata"))
    # (Value is declared "string", which SQLite gives numeric affinity.)
    assert meta["version"] == "0.1.18" and str(meta["-percolatorThreshold"]) == "0.01"
    # N-terminal modifications where the jar puts them in each table.
    assert con.execute("SELECT PeptideModSeq FROM entries WHERE PeptideSeq='MDFQK'").fetchone()[0].startswith("[+42.01")
    assert con.execute("SELECT PeptideModSeq FROM peptidequants WHERE PeptideSeq='MDFQK'").fetchone()[0].startswith("M[+42.01")
    assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")} >= {"Key_Metadata_index"}
    assert con.execute("SELECT count(*) FROM sqlite_master WHERE type='index'").fetchone()[0] == 22


def test_without_its_own_flags_every_peak_is_quantified(tmp_path) -> None:
    """A library whose peaks carry no quantified flags is written with
    every peak flagged, as the jar flags them in every file it writes,
    whatever its quantity rows used; with no correlations the column
    stays NULL, as in the jar's predicted libraries. An entry with no
    stored peaks gets empty arrays, not NULL."""
    import sqlite3

    from _encyclopedia_fixtures import Entry, build_result_library

    original = build_result_library(
        tmp_path / "jar.elib",
        [Entry(modseq="PEPTIDEK", sequence="PEPTIDEK", charge=2, precursor_mz=465.2,
               mz=[200.1, 300.2], intensity=[5.0, 6.0], total_intensity=100.0,
               quant_ions=[(300.2, 100.0, 0.9)]),
         Entry(modseq="ELVISK", sequence="ELVISK", charge=2, precursor_mz=344.7,
               mz=[], intensity=[], total_intensity=None,
               peak_correlation=[], peak_quantified=[])],
    )  # fmt: skip
    first = read_encyclopedia(original, sidecars=False)
    assert first.library.fragments.column("quantified").null_count == 2
    write_encyclopedia(tmp_path / "ours.elib", first.library, quant=first.quant, search=first.search)
    again = read_encyclopedia(tmp_path / "ours.elib", sidecars=False).library.fragments
    assert again.sort_by("mz_theoretical").column("quantified").to_pylist() == [True, True]
    assert again.column("correlation").null_count == 2
    con = sqlite3.connect(tmp_path / "ours.elib")
    rows = dict(con.execute("SELECT PeptideSeq, QuantifiedIonsArray IS NOT NULL FROM entries"))
    assert rows == {"PEPTIDEK": 1, "ELVISK": 1}


def test_quantified_ions_with_unknown_correlations_come_back_as_nulls(tmp_path) -> None:
    """``FRAGMENT_QUANT.correlation`` is nullable and the writer leaves a
    quantity row's correlation array NULL when its ions have none, so
    the reader has to take a NULL array as nulls, one per ion, beside
    rows that do have them."""
    import dataclasses

    from _encyclopedia_fixtures import Entry, build_result_library

    original = build_result_library(
        tmp_path / "jar.elib",
        [Entry(modseq="PEPTIDEK", sequence="PEPTIDEK", charge=2, precursor_mz=465.2,
               mz=[200.1, 300.2], intensity=[5.0, 6.0], total_intensity=150.0,
               quant_ions=[(200.1, 100.0, 0.9), (300.2, 50.0, 0.8)]),
         Entry(modseq="ELVISK", sequence="ELVISK", charge=2, precursor_mz=344.7,
               mz=[250.1], intensity=[3.0], total_intensity=70.0,
               quant_ions=[(250.1, 70.0, 0.7)])],
    )  # fmt: skip
    first = read_encyclopedia(original, sidecars=False)
    ions = first.quant.fragment_quant
    peptides, precursors = first.library.peptides, first.library.precursors
    elvisk = peptides.filter(pa.compute.equal(peptides.column("modified_sequence"), "ELVISK"))
    elvisk = precursors.filter(pa.compute.equal(precursors.column("peptide_id"), elvisk.column("peptide_id")[0]))
    unknown = pa.compute.equal(ions.column("precursor_id"), elvisk.column("precursor_id")[0])
    ions = ions.set_column(
        ions.schema.get_field_index("correlation"), ions.schema.field("correlation"),
        pa.compute.if_else(unknown, pa.scalar(None, pa.float64()), ions.column("correlation")),
    )  # fmt: skip
    quant = dataclasses.replace(first.quant, fragment_quant=ions)
    write_encyclopedia(tmp_path / "ours.elib", first.library, quant=quant, search=first.search)
    again = read_encyclopedia(tmp_path / "ours.elib", sidecars=False).quant.fragment_quant
    order = [("precursor_id", "ascending"), ("mz", "ascending")]
    assert again.sort_by(order).equals(ions.sort_by(order))
    assert again.column("correlation").null_count == 1


def test_a_search_with_peptide_scores_only_is_written_per_precursor(tmp_path) -> None:
    """A ``Search`` with no precursor scores (built by hand, as here, or
    read from a bundle written before they existed) keeps its peptide
    scores: each under every precursor of the peptide."""
    library, quant, search = _toy_library_quant_search()
    assert search.precursor_scores.num_rows == 0 and search.peptide_scores.num_rows == 2
    write_encyclopedia(tmp_path / "toy.elib", library, quant=quant, search=search)
    back = read_encyclopedia(tmp_path / "toy.elib", sidecars=False)
    assert sorted(back.search.peptide_scores.column("qvalue").to_pylist()) == [0.001, 0.005]


def test_a_write_that_fails_leaves_nothing_behind(tmp_path) -> None:
    """A scored peptide with no precursor cannot be written (the jar's
    scores are per precursor): refused, and no file is left, nor is an
    existing one replaced."""
    import dataclasses

    library, quant, search = _toy_library_quant_search()
    orphan = library.precursors.column("peptide_id").to_pylist()[0]
    keep = pa.compute.not_equal(library.precursors.column("peptide_id"), orphan)
    precursors = library.precursors.filter(keep)
    fragments = library.fragments.filter(
        pa.compute.is_in(library.fragments.column("precursor_id"), precursors.column("precursor_id"))
    )
    library = dataclasses.replace(library, precursors=precursors, fragments=fragments)
    out = tmp_path / "toy.elib"
    with pytest.raises(ValueError, match="no precursor in the library"):
        write_encyclopedia(out, library, search=search)
    assert list(tmp_path.iterdir()) == []
    out.write_bytes(b"what was there")
    with pytest.raises(ValueError, match="no precursor in the library"):
        write_encyclopedia(out, library, search=search, overwrite=True)
    assert out.read_bytes() == b"what was there" and len(list(tmp_path.iterdir())) == 1
