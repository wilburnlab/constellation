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
    PSM_TABLE,
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


def _psm(**over) -> dict:
    """A complete PSM_TABLE row (all non-nullable fields set); override as needed."""
    row = {
        "psm_id": 0,
        "raw_file": "a",
        "acquisition_id": 0,
        "scan": 100,
        "precursor_scan": 99,
        "sequence": "PEPTIDE",
        "modified_sequence": "PEPTIDE",
        "peptide_id": 0,
        "mod_peptide_id": 0,
        "evidence_id": 0,
        "proteins": "P1",
        "charge": 2,
        "mz": 400.0,
        "mass": 800.0,
        "mass_error_ppm": 1.0,
        "retention_time_s": 600.0,
        "fragmentation": "HCD",
        "mass_analyzer": "FTMS",
        "psm_type": "MULTI-MSMS",
        "score": 50.0,
        "delta_score": 40.0,
        "pep": 0.001,
        "is_decoy": False,
        "is_contaminant": False,
        "engine": "maxquant",
    }
    row.update(over)
    return row


# ──────────────────────────────────────────────────────────────────────
# Schema registration
# ──────────────────────────────────────────────────────────────────────


def test_schemas_registered():
    from constellation.core.io.schemas import get_schema

    assert get_schema("PeptideScoreTable") is PEPTIDE_SCORE_TABLE
    assert get_schema("ProteinScoreTable") is PROTEIN_SCORE_TABLE
    assert get_schema("PsmTable") is PSM_TABLE


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


# ──────────────────────────────────────────────────────────────────────
# PSM table — Search.psms
# ──────────────────────────────────────────────────────────────────────


def test_empty_search_has_empty_psms():
    s = Search.empty()
    assert s.psms.num_rows == 0
    assert s.psms.schema == PSM_TABLE


def test_assemble_search_with_psms():
    acq = _two_acq()
    s = assemble_search(
        acquisitions=acq,
        psms=[_psm(psm_id=0, raw_file="a"), _psm(psm_id=1, raw_file="b", acquisition_id=1)],
    )
    assert s.psms.num_rows == 2
    assert s.peptide_scores.num_rows == 0
    assert s.protein_scores.num_rows == 0


def test_psm_multi_secpep_duplicate_scan_accepted():
    # Two matches on the SAME (raw_file, scan) but distinct psm_id — the
    # MULTI-SECPEP case. (raw_file, scan) is not a PK; (raw_file, psm_id) is.
    acq = _two_acq()
    s = assemble_search(
        acquisitions=acq,
        psms=[
            _psm(psm_id=0, raw_file="a", scan=9912, psm_type="MULTI-MSMS"),
            _psm(psm_id=1, raw_file="a", scan=9912, psm_type="MULTI-SECPEP"),
        ],
    )
    assert s.psms.num_rows == 2


def test_psm_duplicate_pk_rejected():
    acq = _two_acq()
    with pytest.raises(ValueError, match=r"\(raw_file, psm_id\)"):
        assemble_search(
            acquisitions=acq,
            psms=[
                _psm(psm_id=0, raw_file="a", scan=100),
                _psm(psm_id=0, raw_file="a", scan=200),
            ],
        )


def test_psm_unknown_acquisition_id_rejected():
    acq = _two_acq()
    with pytest.raises(ValueError, match="acquisition_id"):
        assemble_search(
            acquisitions=acq,
            psms=[_psm(psm_id=0, raw_file="a", acquisition_id=99)],
        )


def test_psm_null_acquisition_id_allowed():
    # Pre-link PSMs (acquisition_id null) are permitted.
    acq = _two_acq()
    s = assemble_search(
        acquisitions=acq,
        psms=[_psm(psm_id=0, raw_file="a", acquisition_id=None)],
    )
    assert s.psms.column("acquisition_id").to_pylist() == [None]


def test_parquet_round_trip_preserves_psms(tmp_path):
    acq = _two_acq()
    s = assemble_search(
        acquisitions=acq,
        psms=[
            _psm(psm_id=0, raw_file="a", scan=100, is_decoy=False),
            _psm(psm_id=1, raw_file="a", scan=101, is_decoy=True, proteins=None),
        ],
    )
    out = tmp_path / "search"
    save_search(s, out, format="parquet_dir")
    assert (out / "psms.parquet").exists()

    s2 = load_search(out, format="parquet_dir")
    assert s2.psms.num_rows == 2
    assert s2.psms.schema == PSM_TABLE
    assert s2.psms.column("is_decoy").to_pylist() == [False, True]


def test_parquet_back_compat_missing_psms(tmp_path):
    # A bundle written before the PSM table existed has no psms.parquet —
    # the reader must return an empty PSM table, not raise.
    acq = _two_acq()
    s = assemble_search(acquisitions=acq, psms=[_psm(psm_id=0, raw_file="a")])
    out = tmp_path / "search"
    save_search(s, out, format="parquet_dir")
    (out / "psms.parquet").unlink()  # simulate a pre-PSM bundle

    s2 = load_search(out, format="parquet_dir")
    assert s2.psms.num_rows == 0
    assert s2.psms.schema == PSM_TABLE
