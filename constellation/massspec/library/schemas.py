"""Theoretical / sample-agnostic spectral-library schemas.

A spectral library models the protein → peptide → precursor → fragment
DAG. These five schemas hold *theoretical* structure plus *predicted*
entity-level properties (RT, CCS, fragment intensities) that are tied
to entity identity rather than any single acquisition. Empirical /
sample-specific data — abundances, calibrated transmission efficiencies,
search scores — lives in ``massspec.quant`` and ``massspec.search``.

    PROTEIN_TABLE              one row per protein (sequence-level)
    PEPTIDE_TABLE              one row per (sequence, modification) pair
    PRECURSOR_TABLE            one row per (peptide, charge); has
                               peptide_id FK + predicted RT/CCS
    LIBRARY_FRAGMENT_TABLE     one row per (precursor, ion_type,
                               position, charge, loss); has predicted
                               intensity. Column names align with
                               ``massspec.schemas.FRAGMENT_ION_TABLE``
                               so a library fragment row projects
                               cleanly via ``cast_to_schema``.
    PROTEIN_PEPTIDE_EDGE       pure adjacency for the only true M:N
                               relationship (peptides shared across
                               proteoforms). No efficiency column —
                               calibrated efficiencies live in
                               ``massspec.quant``.

Sentinel ``-1.0`` for unset float fields (rt_predicted, ccs_predicted,
intensity_predicted) — same convention as the rest of the package.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema

# ──────────────────────────────────────────────────────────────────────
# Tier tables
# ──────────────────────────────────────────────────────────────────────


PROTEIN_TABLE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("accession", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"ProteinTable"},
)


PEPTIDE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("peptide_id", pa.int64(), nullable=False),
        # Canonical-stripped (e.g. "PEPTIDE")
        pa.field("sequence", pa.string(), nullable=False),
        # Bracket form, parse_modified_sequence-compatible
        # (e.g. "PEP[UNIMOD:4]TIDE"); equals sequence when unmodified.
        pa.field("modified_sequence", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"PeptideTable"},
)


PRECURSOR_TABLE: pa.Schema = pa.schema(
    [
        pa.field("precursor_id", pa.int64(), nullable=False),
        pa.field("peptide_id", pa.int64(), nullable=False),
        pa.field("charge", pa.int32(), nullable=False),
        pa.field("precursor_mz", pa.float64(), nullable=False),
        # -1.0 sentinel = no prediction available
        pa.field("rt_predicted", pa.float64(), nullable=False),
        pa.field("ccs_predicted", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"PrecursorTable"},
)


# Column names that overlap ``FRAGMENT_ION_TABLE`` (ion_type, position,
# charge, loss_id, mz_theoretical) are deliberately identical — a row
# from this table projects cleanly into a FragmentIonTable view.
LIBRARY_FRAGMENT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("precursor_id", pa.int64(), nullable=False),
        pa.field("ion_type", pa.int8(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("charge", pa.int32(), nullable=False),
        pa.field("loss_id", pa.string(), nullable=True),
        pa.field("mz_theoretical", pa.float64(), nullable=False),
        # -1.0 sentinel = no prediction; otherwise normalised to [0, 1]
        pa.field("intensity_predicted", pa.float32(), nullable=False),
        pa.field("annotation", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"LibraryFragmentTable"},
)


PROTEIN_PEPTIDE_EDGE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("peptide_id", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"ProteinPeptideEdge"},
)


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────


register_schema("ProteinTable", PROTEIN_TABLE)
register_schema("PeptideTable", PEPTIDE_TABLE)
register_schema("PrecursorTable", PRECURSOR_TABLE)
register_schema("LibraryFragmentTable", LIBRARY_FRAGMENT_TABLE)
register_schema("ProteinPeptideEdge", PROTEIN_PEPTIDE_EDGE)


__all__ = [
    "PROTEIN_TABLE",
    "PEPTIDE_TABLE",
    "PRECURSOR_TABLE",
    "LIBRARY_FRAGMENT_TABLE",
    "PROTEIN_PEPTIDE_EDGE",
]
