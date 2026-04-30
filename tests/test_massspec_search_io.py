"""Tests for :mod:`massspec.search` — Search container + ParquetDir round-trip."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.massspec.acquisitions import (
    ACQUISITION_TABLE,
    Acquisitions,
)
from constellation.massspec.library.library import assign_ids
from constellation.massspec.search import Search, assemble_search
from constellation.massspec.search.io import (
    SEARCH_READERS,
    SEARCH_WRITERS,
    SearchReader,
    SearchWriter,
    load_search,
    save_search,
)
from constellation.massspec.search.schemas import (
    PEPTIDE_SCORE_TABLE,
    PROTEIN_SCORE_TABLE,
)


def _two_acq() -> Acquisitions:
    table = pa.Table.from_pylist(
        [
            {"acquisition_id": 0, "source_file": "a.mzML", "source_kind": "mzml", "acquisition_datetime": None},
            {"acquisition_id": 1, "source_file": "b.mzML", "source_kind": "mzml", "acquisition_datetime": None},
        ],
        schema=ACQUISITION_TABLE,
    )
    return Acquisitions(table)


# ──────────────────────────────────────────────────────────────────────
# Schema registration
# ──────────────────────────────────────────────────────────────────────


def test_schemas_registered():
    from constellation.core.io.schemas import get_schema

    assert get_schema("PeptideScoreTable") is PEPTIDE_SCORE_TABLE
    assert get_schema("ProteinScoreTable") is PROTEIN_SCORE_TABLE


# ──────────────────────────────────────────────────────────────────────
# Container — empty + assembled
# ──────────────────────────────────────────────────────────────────────


def test_empty_search_validates():
    Search.empty()  # no acquisitions → empty scores → validates


def test_assemble_search_with_records():
    acq = _two_acq()
    s = assemble_search(
        acquisitions=acq,
        peptide_scores=[
            {
                "peptide_id": 0,
                "acquisition_id": 0,
                "score": 5.0,
                "qvalue": 0.001,
                "pep": 0.0001,
                "engine": "encyclopedia",
            },
            {
                "peptide_id": 1,
                "acquisition_id": 1,
                "score": 3.0,
                "qvalue": 0.01,
                "pep": 0.001,
                "engine": "encyclopedia",
            },
        ],
        protein_scores=[
            {
                "protein_id": 0,
                "acquisition_id": 0,
                "score": 12.0,
                "qvalue": 0.001,
                "engine": "encyclopedia",
            },
        ],
    )
    assert s.peptide_scores.num_rows == 2
    assert s.protein_scores.num_rows == 1


def test_run_agnostic_score_with_null_acquisition_id():
    acq = _two_acq()
    s = assemble_search(
        acquisitions=acq,
        peptide_scores=[
            {
                "peptide_id": 0,
                "acquisition_id": None,  # run-agnostic
                "score": 5.0,
                "qvalue": None,
                "pep": None,
                "engine": "encyclopedia",
            },
        ],
    )
    assert s.peptide_scores.column("acquisition_id").to_pylist() == [None]


def test_duplicate_pk_rejected():
    acq = _two_acq()
    with pytest.raises(ValueError, match="duplicate"):
        assemble_search(
            acquisitions=acq,
            peptide_scores=[
                {
                    "peptide_id": 0,
                    "acquisition_id": 0,
                    "score": 1.0,
                    "qvalue": 0.0,
                    "pep": 0.0,
                    "engine": "encyclopedia",
                },
                {
                    "peptide_id": 0,
                    "acquisition_id": 0,
                    "score": 2.0,
                    "qvalue": 0.0,
                    "pep": 0.0,
                    "engine": "encyclopedia",
                },
            ],
        )


def test_unknown_acquisition_id_rejected():
    acq = _two_acq()
    with pytest.raises(ValueError, match="acquisition_id"):
        assemble_search(
            acquisitions=acq,
            peptide_scores=[
                {
                    "peptide_id": 0,
                    "acquisition_id": 99,
                    "score": 1.0,
                    "qvalue": 0.0,
                    "pep": 0.0,
                    "engine": "encyclopedia",
                }
            ],
        )


def test_validate_against_library_fk_closure():
    acq = _two_acq()
    library = assign_ids(
        proteins=[{"accession": "P1"}],
        peptides=[{"modified_sequence": "PEPK"}],
        precursors=[{"modified_sequence": "PEPK", "charge": 2, "precursor_mz": 471.0}],
        protein_peptide=[("P1", "PEPK")],
    )
    s = assemble_search(
        acquisitions=acq,
        peptide_scores=[
            {
                "peptide_id": 0,
                "acquisition_id": 0,
                "score": 1.0,
                "qvalue": 0.0,
                "pep": 0.0,
                "engine": "encyclopedia",
            }
        ],
    )
    s.validate_against(library)  # no raise

    bad = assemble_search(
        acquisitions=acq,
        peptide_scores=[
            {
                "peptide_id": 999,
                "acquisition_id": 0,
                "score": 1.0,
                "qvalue": 0.0,
                "pep": 0.0,
                "engine": "encyclopedia",
            }
        ],
    )
    with pytest.raises(ValueError, match="peptide_id"):
        bad.validate_against(library)


# ──────────────────────────────────────────────────────────────────────
# Protocol conformance + registry
# ──────────────────────────────────────────────────────────────────────


def test_parquet_writer_satisfies_protocol():
    w = SEARCH_WRITERS["parquet_dir"]
    assert isinstance(w, SearchWriter)


def test_parquet_reader_satisfies_protocol():
    r = SEARCH_READERS["parquet_dir"]
    assert isinstance(r, SearchReader)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir round-trip
# ──────────────────────────────────────────────────────────────────────


def test_parquet_round_trip_preserves_scores(tmp_path):
    acq = _two_acq()
    s = assemble_search(
        acquisitions=acq,
        peptide_scores=[
            {
                "peptide_id": 0,
                "acquisition_id": 0,
                "score": 5.0,
                "qvalue": 0.001,
                "pep": 0.0001,
                "engine": "encyclopedia",
            }
        ],
        protein_scores=[
            {
                "protein_id": 0,
                "acquisition_id": 0,
                "score": 12.0,
                "qvalue": 0.001,
                "engine": "encyclopedia",
            }
        ],
        metadata={"library_id": "test-lib"},
    )
    out = tmp_path / "search"
    save_search(s, out, format="parquet_dir")

    s2 = load_search(out, format="parquet_dir")
    assert s2.peptide_scores.num_rows == 1
    assert s2.protein_scores.num_rows == 1
    assert s2.acquisitions.table.num_rows == 2
    assert s2.metadata_extras["library_id"] == "test-lib"
