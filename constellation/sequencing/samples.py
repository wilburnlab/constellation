"""Samples — first-class M:N resolver between acquisitions and barcodes.

A *sample* is the unit of biological identity downstream code keys on
(per-feature counts, per-feature TPM, differential abundance). Samples
do not always correspond 1:1 with acquisitions:

    1 sample × N flowcells   genomic ultra-HMW (>60× coverage for
                             eukaryote genomes; the lab routinely runs
                             multi-flowcell aggregations); barcode null

    M samples × 1 flowcell   multiplexed cDNA / direct-RNA — M barcodes
                             discriminate samples on a single flow cell

    M:N mixed                multi-flowcell + multiplexed in any
                             combination

The ``SAMPLE_ACQUISITION_EDGE`` table encodes the resolver as a tiny
Arrow table; ``Samples`` wraps both with PK uniqueness + FK closure.
``barcode_id`` may be null (= "use all reads from this acquisition")
to handle the genomic case without bifurcating the schema.

Status: STUB. Schemas final; container methods raise
``NotImplementedError`` pending Phase 1 (Foundation).
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# ──────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────


SAMPLE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("sample_name", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"SampleTable"},
)


SAMPLE_ACQUISITION_EDGE: pa.Schema = pa.schema(
    [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        # null = "all reads from this acquisition" (no barcoding); int64
        # otherwise FKs into a barcode registry resolved at demux time.
        pa.field("barcode_id", pa.int64(), nullable=True),
    ],
    metadata={b"schema_name": b"SampleAcquisitionEdge"},
)


register_schema("SampleTable", SAMPLE_TABLE)
register_schema("SampleAcquisitionEdge", SAMPLE_ACQUISITION_EDGE)


# ──────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────


_PHASE = "Phase 1 (Foundation)"


@dataclass(frozen=True, slots=True)
class Samples:
    """Bundles a ``SAMPLE_TABLE`` with its ``SAMPLE_ACQUISITION_EDGE``.

    Validation on construction: ``sample_id`` PK uniqueness, edge-table
    cast, and (when an ``Acquisitions`` is supplied) FK closure of
    ``acquisition_id`` references via :func:`validate_acquisitions`.
    """

    samples: pa.Table
    edges: pa.Table

    def __post_init__(self) -> None:
        raise NotImplementedError(
            f"Samples PK uniqueness + edge cast pending {_PHASE}"
        )

    @classmethod
    def empty(cls) -> "Samples":
        raise NotImplementedError(f"Samples.empty pending {_PHASE}")

    @property
    def ids(self) -> list[int]:
        raise NotImplementedError(f"Samples.ids pending {_PHASE}")

    def acquisitions_for(self, sample_id: int) -> list[int]:
        """All acquisition_ids that contribute to ``sample_id``."""
        raise NotImplementedError(f"Samples.acquisitions_for pending {_PHASE}")

    def samples_for(
        self,
        acquisition_id: int,
        barcode_id: int | None = None,
    ) -> list[int]:
        """All sample_ids that draw from ``(acquisition_id, barcode_id)``.

        If ``barcode_id`` is None, returns the sample(s) for un-barcoded
        rows (genomic). If barcode_id is supplied, returns the sample
        for that specific barcode within the acquisition.
        """
        raise NotImplementedError(f"Samples.samples_for pending {_PHASE}")

    def __len__(self) -> int:
        raise NotImplementedError(f"Samples.__len__ pending {_PHASE}")


__all__ = [
    "SAMPLE_TABLE",
    "SAMPLE_ACQUISITION_EDGE",
    "Samples",
]
