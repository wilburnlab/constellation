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
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import (
    cast_to_schema,
    register_schema,
)


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


@dataclass(frozen=True, slots=True)
class Samples:
    """Bundles a ``SAMPLE_TABLE`` with its ``SAMPLE_ACQUISITION_EDGE``.

    Validation on construction: ``sample_id`` PK uniqueness, edge-table
    cast, and edge-side FK closure (every ``edges.sample_id`` exists in
    ``samples.sample_id``). ``acquisition_id`` FK closure across an
    external :class:`Acquisitions` is left to ``validate_acquisitions``
    at the call site so the container itself doesn't need an
    ``Acquisitions`` reference.
    """

    samples: pa.Table
    edges: pa.Table

    def __post_init__(self) -> None:
        samples_cast = cast_to_schema(self.samples, SAMPLE_TABLE)
        edges_cast = cast_to_schema(self.edges, SAMPLE_ACQUISITION_EDGE)
        object.__setattr__(self, "samples", samples_cast)
        object.__setattr__(self, "edges", edges_cast)

        ids = samples_cast.column("sample_id").to_pylist()
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"sample_id contains duplicates: {duplicates[:5]}"
                f"{'...' if len(duplicates) > 5 else ''}"
            )

        known = set(ids)
        edge_sample_ids = edges_cast.column("sample_id").to_pylist()
        unknown = {sid for sid in edge_sample_ids if sid not in known}
        if unknown:
            sample = sorted(unknown)[:5]
            raise ValueError(
                f"edges reference unknown sample_ids: "
                f"{sample}{'...' if len(unknown) > 5 else ''}"
            )

    @classmethod
    def empty(cls) -> Samples:
        return cls(SAMPLE_TABLE.empty_table(), SAMPLE_ACQUISITION_EDGE.empty_table())

    @classmethod
    def from_records(
        cls,
        samples: Iterable[dict[str, object]],
        edges: Iterable[dict[str, object]],
    ) -> Samples:
        sample_rows = list(samples)
        edge_rows = list(edges)
        s_table = (
            pa.Table.from_pylist(sample_rows, schema=SAMPLE_TABLE)
            if sample_rows
            else SAMPLE_TABLE.empty_table()
        )
        e_table = (
            pa.Table.from_pylist(edge_rows, schema=SAMPLE_ACQUISITION_EDGE)
            if edge_rows
            else SAMPLE_ACQUISITION_EDGE.empty_table()
        )
        return cls(s_table, e_table)

    @property
    def ids(self) -> list[int]:
        return self.samples.column("sample_id").to_pylist()

    def acquisitions_for(self, sample_id: int) -> list[int]:
        """All acquisition_ids that contribute to ``sample_id``."""
        mask = pc.equal(self.edges.column("sample_id"), sample_id)
        filtered = self.edges.filter(mask)
        return filtered.column("acquisition_id").to_pylist()

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
        acq_match = pc.equal(self.edges.column("acquisition_id"), acquisition_id)
        if barcode_id is None:
            bc_match = pc.is_null(self.edges.column("barcode_id"))
        else:
            bc_match = pc.equal(self.edges.column("barcode_id"), barcode_id)
        mask = pc.and_(acq_match, bc_match)
        return self.edges.filter(mask).column("sample_id").to_pylist()

    def __len__(self) -> int:
        return self.samples.num_rows


__all__ = [
    "SAMPLE_TABLE",
    "SAMPLE_ACQUISITION_EDGE",
    "Samples",
]
