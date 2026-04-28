"""Empirical / sample-specific quantification schemas.

Five Arrow schemas paired with the theoretical structure in
``massspec.library``: per-tier abundance observations and calibrated
transmission efficiencies for the two conversion edges in the DAG.

    PROTEIN_QUANT                  per-acquisition protein abundances
    PEPTIDE_QUANT                  per-acquisition peptide abundances
    PRECURSOR_QUANT                per-acquisition precursor intensity
                                   + observed RT/CCS
    TRANSMISSION_PROTEIN_PEPTIDE   calibrated efficiency for the
                                   protein → peptide conversion;
                                   acquisition_id nullable to support
                                   run-agnostic calibration
    TRANSMISSION_PEPTIDE_PRECURSOR calibrated efficiency for the
                                   peptide → precursor conversion;
                                   same nullable convention

``efficiency`` is bounded ``(0, 1]`` on populated rows; ``-1.0`` is the
"uncalibrated" sentinel and is excluded from validation.

Sentinel ``-1.0`` for unset float fields (abundance, intensity,
rt_observed, ccs_observed, efficiency).
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema

# ──────────────────────────────────────────────────────────────────────
# Per-entity quantification
# ──────────────────────────────────────────────────────────────────────


PROTEIN_QUANT: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("abundance", pa.float64(), nullable=False),
        pa.field("score", pa.float64(), nullable=True),
    ],
    metadata={b"schema_name": b"ProteinQuant"},
)


PEPTIDE_QUANT: pa.Schema = pa.schema(
    [
        pa.field("peptide_id", pa.int64(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("abundance", pa.float64(), nullable=False),
        pa.field("score", pa.float64(), nullable=True),
    ],
    metadata={b"schema_name": b"PeptideQuant"},
)


PRECURSOR_QUANT: pa.Schema = pa.schema(
    [
        pa.field("precursor_id", pa.int64(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        pa.field("rt_observed", pa.float64(), nullable=False),
        pa.field("ccs_observed", pa.float64(), nullable=False),
        pa.field("score", pa.float64(), nullable=True),
    ],
    metadata={b"schema_name": b"PrecursorQuant"},
)


# ──────────────────────────────────────────────────────────────────────
# Transmission efficiencies (per-edge calibrated parameters)
# ──────────────────────────────────────────────────────────────────────


TRANSMISSION_PROTEIN_PEPTIDE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("peptide_id", pa.int64(), nullable=False),
        # Nullable: a missing acquisition_id means the calibration is
        # run-agnostic (single set of efficiencies for the whole study).
        pa.field("acquisition_id", pa.int64(), nullable=True),
        pa.field("efficiency", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"TransmissionProteinPeptide"},
)


TRANSMISSION_PEPTIDE_PRECURSOR: pa.Schema = pa.schema(
    [
        pa.field("peptide_id", pa.int64(), nullable=False),
        pa.field("precursor_id", pa.int64(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=True),
        pa.field("efficiency", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"TransmissionPeptidePrecursor"},
)


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────


register_schema("ProteinQuant", PROTEIN_QUANT)
register_schema("PeptideQuant", PEPTIDE_QUANT)
register_schema("PrecursorQuant", PRECURSOR_QUANT)
register_schema("TransmissionProteinPeptide", TRANSMISSION_PROTEIN_PEPTIDE)
register_schema("TransmissionPeptidePrecursor", TRANSMISSION_PEPTIDE_PRECURSOR)


__all__ = [
    "PROTEIN_QUANT",
    "PEPTIDE_QUANT",
    "PRECURSOR_QUANT",
    "TRANSMISSION_PROTEIN_PEPTIDE",
    "TRANSMISSION_PEPTIDE_PRECURSOR",
]
