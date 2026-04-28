"""Schema-level tests for ``constellation.massspec.quant``."""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import get_schema
from constellation.massspec.library.schemas import PROTEIN_TABLE
from constellation.massspec.quant.schemas import (
    PEPTIDE_QUANT,
    PRECURSOR_QUANT,
    PROTEIN_QUANT,
    TRANSMISSION_PEPTIDE_PRECURSOR,
    TRANSMISSION_PROTEIN_PEPTIDE,
)


def test_all_schemas_registered() -> None:
    assert get_schema("ProteinQuant") is PROTEIN_QUANT
    assert get_schema("PeptideQuant") is PEPTIDE_QUANT
    assert get_schema("PrecursorQuant") is PRECURSOR_QUANT
    assert get_schema("TransmissionProteinPeptide") is TRANSMISSION_PROTEIN_PEPTIDE
    assert get_schema("TransmissionPeptidePrecursor") is TRANSMISSION_PEPTIDE_PRECURSOR


def test_quant_tables_use_acquisition_id_fk() -> None:
    """All quant tables use ``acquisition_id`` (not ``run_id``)."""
    for schema in (PROTEIN_QUANT, PEPTIDE_QUANT, PRECURSOR_QUANT):
        assert "acquisition_id" in schema.names
        assert "run_id" not in schema.names


def test_quant_acquisition_id_not_null() -> None:
    """Per-entity quants always reference a specific acquisition."""
    for schema in (PROTEIN_QUANT, PEPTIDE_QUANT, PRECURSOR_QUANT):
        assert not schema.field("acquisition_id").nullable


def test_transmission_acquisition_id_nullable() -> None:
    """Run-agnostic calibrations are represented with null acquisition_id."""
    assert TRANSMISSION_PROTEIN_PEPTIDE.field("acquisition_id").nullable
    assert TRANSMISSION_PEPTIDE_PRECURSOR.field("acquisition_id").nullable


def test_efficiency_column_types() -> None:
    assert TRANSMISSION_PROTEIN_PEPTIDE.field("efficiency").type == pa.float64()
    assert TRANSMISSION_PEPTIDE_PRECURSOR.field("efficiency").type == pa.float64()


def test_fk_column_names_align_with_library() -> None:
    """Quant FKs are named the same as their target Library PKs so
    cross-validation can use a single column name."""
    pq_protein_id = PROTEIN_QUANT.field("protein_id")
    lib_protein_id = PROTEIN_TABLE.field("protein_id")
    assert pq_protein_id.type == lib_protein_id.type


def test_score_column_nullable() -> None:
    for schema in (PROTEIN_QUANT, PEPTIDE_QUANT, PRECURSOR_QUANT):
        assert schema.field("score").nullable
