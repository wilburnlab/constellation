"""Mass-spectrometry-domain Arrow schemas.

Per the generic-vs-modality split in `core.io.schemas`: a schema lands in
`core.io` only if it's universal across ≥2 modalities. Anything narrower —
like fragment-ion ladders, MS1/MS2 chromatogram tables, or PSMs — lives
in this module. Schemas here register with `core.io.schemas.register_schema`
at import so cross-modality consumers can ask the registry "is this an
FragmentIonTable?" without importing massspec.

    FRAGMENT_ION_TABLE   one row per (peptide, position, ion_type, charge, loss)
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema

# ──────────────────────────────────────────────────────────────────────
# FragmentIonTable
# ──────────────────────────────────────────────────────────────────────


# Either-or columns: callers populate `peptide_idx` (when ions belong to
# rows of an external peptide table) or `peptide_seq` (standalone use).
# Both nullable so each style is allowed.
#
# `ion_type` is stored as int8 — the IonType enum value (A=0, B=1, ...).
# `loss_id` is the string id from `LOSS_REGISTRY` (e.g., "H2O") or null
# for the no-loss baseline.
FRAGMENT_ION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("peptide_idx", pa.int32(), nullable=True),
        pa.field("peptide_seq", pa.string(), nullable=True),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("ion_type", pa.int8(), nullable=False),
        pa.field("charge", pa.int32(), nullable=False),
        pa.field("loss_id", pa.string(), nullable=True),
        pa.field("mz_theoretical", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"FragmentIonTable"},
)


register_schema("FragmentIonTable", FRAGMENT_ION_TABLE)


__all__ = ["FRAGMENT_ION_TABLE"]
