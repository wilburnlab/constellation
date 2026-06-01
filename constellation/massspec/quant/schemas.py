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
# Chromatogram (XIC) extraction — target input + trace output
# ──────────────────────────────────────────────────────────────────────


XIC_TARGET_SCHEMA_VERSION: int = 1


# The normalized extraction-target contract: the single thing the
# chromatogram extractor consumes. Every real source (theoretical
# library `PRECURSOR_TABLE`, search `PSM_TABLE`, `PRECURSOR_QUANT`, or a
# bare hand-authored list) is adapted into this shape (see
# `massspec.quant.targets`), so "general m/z+RT target" and
# "search-results" are *sources*, not separate operations.
#
# `precursor_mz` is nullable because it is derivable from
# `modified_sequence` + `precursor_charge`; a valid row carries either an
# explicit `precursor_mz` (the no-peptide m/z case — MS1 only) OR the
# modseq+charge pair (required for MS2 fragment ladders). The extractor
# enforces that, not the schema. RT gating: explicit `[rt_start, rt_end]`
# overrides `rt_center ± rt_window`; all-null → all-RT (untargeted).
XIC_TARGET_TABLE: pa.Schema = pa.schema(
    [
        # Caller's id for output join-back (precursor_id / psm_id / row index).
        pa.field("target_id", pa.int64(), nullable=False),
        # ProForma 2.0; required for MS2, optional for MS1.
        pa.field("modified_sequence", pa.string(), nullable=True),
        pa.field("precursor_charge", pa.int8(), nullable=True),
        # The anchor; derivable from modseq+charge when null.
        pa.field("precursor_mz", pa.float64(), nullable=True),
        # RT gating. Explicit [start, end] overrides center ± window.
        pa.field("rt_center", pa.float64(), nullable=True),
        pa.field("rt_start", pa.float64(), nullable=True),
        pa.field("rt_end", pa.float64(), nullable=True),
        # Assigned-scan restriction (populated only from PSM sources).
        pa.field("scan", pa.int32(), nullable=True),
        # FK passthrough into library tiers.
        pa.field("precursor_id", pa.int64(), nullable=True),
        pa.field("peptide_id", pa.int64(), nullable=True),
    ],
    metadata={
        b"schema_name": b"XICTarget",
        b"schema_version": str(XIC_TARGET_SCHEMA_VERSION).encode("utf-8"),
    },
)


XIC_TRACE_SCHEMA_VERSION: int = 1


# Long-form XIC extraction output — one row per matched (target, scan,
# ion-coordinate). A single `level`-discriminated table with nullable
# MS1-only / MS2-only coordinate columns (the `MS_PEAK_TABLE` /
# `SCAN_METADATA_TABLE` one-table-nullable-subset precedent).
#
# Schema dtype is float64 for every m/z-valued / m/z-derived column,
# aligned with `MS_PEAK_TABLE.mz`. NOTE this is the column dtype, not a
# precision guarantee: extracted against an f32-m/z peak index, the
# *realized* precision of `mz_observed` / `mz_error_*` is f32 unless the
# extractor's `exact_error` path re-derives them from the canonical
# table. `mz_error_*` are SIGNED, conventional (observed - theoretical) —
# derived from the matching kernel's `signed_delta` — so downstream
# probabilistic scoring stays re-extraction-free. This is the terminal artifact;
# trace → PRECURSOR_QUANT reduction is a separate downstream step.
XIC_TRACE_TABLE: pa.Schema = pa.schema(
    [
        # FK → ACQUISITION_TABLE (supplied by caller; not in MS_PEAK_TABLE).
        pa.field("acquisition_id", pa.int64(), nullable=False),
        # FK → XIC_TARGET_TABLE.target_id (the join-back key).
        pa.field("target_id", pa.int64(), nullable=False),
        # ProForma, carried from the target for readability.
        pa.field("modified_sequence", pa.string(), nullable=True),
        pa.field("precursor_charge", pa.int8(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        # Join keys back to MS_PEAK_TABLE / SCAN_METADATA_TABLE.
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),
        # MS2 isolation-window provenance (null for MS1).
        pa.field("isolation_lower", pa.float64(), nullable=True),
        pa.field("isolation_upper", pa.float64(), nullable=True),
        # MS1 coordinate (null on MS2 rows).
        pa.field("isotope", pa.int8(), nullable=True),
        # MS2 coordinates (reuse FRAGMENT_ION_TABLE conventions; null on MS1).
        pa.field("ion_type", pa.int8(), nullable=True),  # IonType enum value
        pa.field("position", pa.int32(), nullable=True),
        pa.field("fragment_charge", pa.int32(), nullable=True),
        pa.field("loss_id", pa.string(), nullable=True),
        pa.field("mz_theoretical", pa.float64(), nullable=False),
        pa.field("mz_observed", pa.float64(), nullable=True),
        # 0.0 for kept-unmatched rows (drop_unmatched=False).
        pa.field("intensity", pa.float64(), nullable=False),
        # Signed (observed - theoretical); null on unmatched rows.
        pa.field("mz_error_da", pa.float64(), nullable=True),
        pa.field("mz_error_ppm", pa.float64(), nullable=True),
    ],
    metadata={
        b"schema_name": b"XICTrace",
        b"schema_version": str(XIC_TRACE_SCHEMA_VERSION).encode("utf-8"),
    },
)


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────


register_schema("ProteinQuant", PROTEIN_QUANT)
register_schema("PeptideQuant", PEPTIDE_QUANT)
register_schema("PrecursorQuant", PRECURSOR_QUANT)
register_schema("TransmissionProteinPeptide", TRANSMISSION_PROTEIN_PEPTIDE)
register_schema("TransmissionPeptidePrecursor", TRANSMISSION_PEPTIDE_PRECURSOR)
register_schema("XICTarget", XIC_TARGET_TABLE)
register_schema("XICTrace", XIC_TRACE_TABLE)


__all__ = [
    "PROTEIN_QUANT",
    "PEPTIDE_QUANT",
    "PRECURSOR_QUANT",
    "TRANSMISSION_PROTEIN_PEPTIDE",
    "TRANSMISSION_PEPTIDE_PRECURSOR",
    "XIC_TARGET_TABLE",
    "XIC_TRACE_TABLE",
]
