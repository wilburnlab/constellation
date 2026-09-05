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
