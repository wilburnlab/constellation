"""Schema-level tests for ``constellation.massspec.library``."""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import get_schema
from constellation.massspec.library.schemas import (
    LIBRARY_FRAGMENT_TABLE,
    PEPTIDE_TABLE,
    PRECURSOR_TABLE,
    PROTEIN_PEPTIDE_EDGE,
    PROTEIN_TABLE,
)
from constellation.massspec.schemas import FRAGMENT_ION_TABLE


def test_all_schemas_registered() -> None:
    assert get_schema("ProteinTable") is PROTEIN_TABLE
    assert get_schema("PeptideTable") is PEPTIDE_TABLE
    assert get_schema("PrecursorTable") is PRECURSOR_TABLE
    assert get_schema("LibraryFragmentTable") is LIBRARY_FRAGMENT_TABLE
    assert get_schema("ProteinPeptideEdge") is PROTEIN_PEPTIDE_EDGE


def test_protein_table_columns() -> None:
    assert PROTEIN_TABLE.field("protein_id").type == pa.int64()
    assert not PROTEIN_TABLE.field("protein_id").nullable
    assert not PROTEIN_TABLE.field("accession").nullable
    assert PROTEIN_TABLE.field("description").nullable


def test_peptide_table_has_both_sequence_and_modified_sequence() -> None:
    cols = set(PEPTIDE_TABLE.names)
    assert {"peptide_id", "sequence", "modified_sequence"} <= cols
    assert not PEPTIDE_TABLE.field("modified_sequence").nullable


def test_precursor_table_fk_and_predicted_columns() -> None:
    assert PRECURSOR_TABLE.field("peptide_id").type == pa.int64()
    assert PRECURSOR_TABLE.field("rt_predicted").type == pa.float64()
    assert PRECURSOR_TABLE.field("ccs_predicted").type == pa.float64()


def test_fragment_table_columns_align_with_fragment_ion_table() -> None:
    """Shared columns between LibraryFragmentTable and FragmentIonTable
    have identical names + types so a library fragment row projects
    cleanly into a FragmentIonTable view via cast_to_schema."""
    shared = ["ion_type", "position", "charge", "loss_id", "mz_theoretical"]
    for name in shared:
        lib_field = LIBRARY_FRAGMENT_TABLE.field(name)
        ion_field = FRAGMENT_ION_TABLE.field(name)
        assert lib_field.type == ion_field.type, name
        assert lib_field.nullable == ion_field.nullable, name


def test_library_fragment_table_intensity_and_annotation() -> None:
    assert LIBRARY_FRAGMENT_TABLE.field("intensity_predicted").type == pa.float32()
    assert not LIBRARY_FRAGMENT_TABLE.field("intensity_predicted").nullable
    assert LIBRARY_FRAGMENT_TABLE.field("annotation").nullable


def test_protein_peptide_edge_has_no_efficiency_column() -> None:
    """Calibrated efficiencies live in massspec.quant — not here."""
    assert set(PROTEIN_PEPTIDE_EDGE.names) == {"protein_id", "peptide_id"}


def test_schema_metadata_carries_schema_name() -> None:
    for schema, name in [
        (PROTEIN_TABLE, b"ProteinTable"),
        (PEPTIDE_TABLE, b"PeptideTable"),
        (PRECURSOR_TABLE, b"PrecursorTable"),
        (LIBRARY_FRAGMENT_TABLE, b"LibraryFragmentTable"),
        (PROTEIN_PEPTIDE_EDGE, b"ProteinPeptideEdge"),
    ]:
        assert schema.metadata[b"schema_name"] == name
