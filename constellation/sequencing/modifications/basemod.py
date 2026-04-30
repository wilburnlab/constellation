"""Modified-base calling from BAM MM/ML tags.

Dorado emits SAM/BAM MM tags ('modification calls', positions of
modified bases) and ML tags ('modification likelihoods', per-call
probabilities 0-255 mapped to 0-1). This module parses them into a
long-format per-call Arrow table.

Common modifications:
    DNA:  5mC (5-methylcytosine), 5hmC (5-hydroxymethylcytosine),
          6mA (N6-methyladenine)
    RNA:  m6A (N6-methyladenosine), m5C, pseU (pseudouridine),
          m6A_DRACH (m6A in DRACH motif context only)

Status: STUB. Pending a downstream phase (tied to BAM tag parsing in
Phase 2 sam_bam reader).
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


BASEMOD_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        # 0-based half-open position on the read sense strand
        pa.field("read_position", pa.int32(), nullable=False),
        # If the read is aligned: 0-based half-open position on the
        # reference; else null
        pa.field("ref_name", pa.string(), nullable=True),
        pa.field("ref_position", pa.int64(), nullable=True),
        # Canonical base at this position ('A', 'C', 'G', 'T', 'U')
        pa.field("canonical_base", pa.string(), nullable=False),
        # Modification token: '5mC', '5hmC', '6mA', 'm6A', 'pseU', ...
        pa.field("mod_code", pa.string(), nullable=False),
        # Probability the base is modified [0, 1]; from ML tag scaled
        # to [0, 1]
        pa.field("probability", pa.float32(), nullable=False),
    ],
    metadata={b"schema_name": b"BasemodTable"},
)


register_schema("BasemodTable", BASEMOD_TABLE)


# ──────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────


_PHASE = "Phase 2+ (sam_bam reader populates MM/ML tags)"


def extract_basemods(
    alignment_tags: pa.Table,        # ALIGNMENT_TAG_TABLE rows for MM, ML
    read_table: pa.Table,            # READ_TABLE for canonical sequences
    *,
    min_probability: float = 0.5,
) -> pa.Table:                       # BASEMOD_TABLE
    """Parse MM/ML tag rows into per-position modified-base calls.

    Filters by ``min_probability`` (Dorado's default cutoff is 0.5;
    raise to 0.95 for stringent calls). One row per modified
    position per read; multiple modifications at the same position
    (5mC + 5hmC, both calls present) emit one row each.
    """
    raise NotImplementedError(f"extract_basemods pending {_PHASE}")


__all__ = [
    "BASEMOD_TABLE",
    "extract_basemods",
]
