"""Per-acquisition provenance for sequencing data.

An *acquisition* is one nanopore run — a flow cell loaded with a single
library and basecalled to BAM (or just collected as raw POD5). Sample
identity is *derived* from acquisitions via :mod:`sequencing.samples`
(an M:N edge table) — a single sample may pool reads from multiple
flowcells (genomic ultra-HMW), and one flowcell may carry many
multiplexed samples (cDNA barcoded).

Mirrors the shape of :mod:`massspec.acquisitions` so cross-modality
work (transcriptome → proteome bridging) sees one provenance idiom.
The columns differ — sequencing carries flow-cell + chemistry kit +
basecaller-model fields that don't exist in MS — but the container
contract (PK uniqueness, ``validate_acquisitions`` cross-check) is
identical.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa

from constellation.core.io.schemas import (
    cast_to_schema,
    pack_metadata,
    register_schema,
    unpack_metadata,
)


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


SEQUENCING_ACQUISITION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        # 'pod5_dir' | 'pod5_file' | 'bam' | 'fastq' | 'fasta' | 'sam'
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("acquisition_datetime", pa.string(), nullable=True),
        # PromethION / MinION / GridION host + position id
        pa.field("instrument_id", pa.string(), nullable=True),
        # ONT flow-cell serial — e.g. "FAQ12345"
        pa.field("flow_cell_id", pa.string(), nullable=True),
        # Flow-cell chemistry — e.g. "FLO-PRO114M" (R10.4.1)
        pa.field("flow_cell_type", pa.string(), nullable=True),
        # Library prep kit — e.g. "SQK-LSK114", "SQK-PCS111"
        pa.field("sample_kit", pa.string(), nullable=True),
        # Dorado model used to basecall — e.g. "dna_r10.4.1_e8.2_sup@v5.0.0"
        pa.field("basecaller_model", pa.string(), nullable=True),
        # 'genomic_dna' | 'cdna' | 'drna'
        pa.field("experiment_type", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"SequencingAcquisitionTable"},
)


register_schema("SequencingAcquisitionTable", SEQUENCING_ACQUISITION_TABLE)


# ──────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Acquisitions:
    """Container around a ``SEQUENCING_ACQUISITION_TABLE``-shaped Arrow
    table.

    Cast on construction (so callers can hand in a slightly wider table
    and have extras dropped); duplicate ``acquisition_id`` values raise.
    Mirrors :class:`massspec.acquisitions.Acquisitions`.
    """

    table: pa.Table

    def __post_init__(self) -> None:
        cast = cast_to_schema(self.table, SEQUENCING_ACQUISITION_TABLE)
        object.__setattr__(self, "table", cast)
        ids = cast.column("acquisition_id").to_pylist()
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"acquisition_id contains duplicates: {duplicates[:5]}"
                f"{'...' if len(duplicates) > 5 else ''}"
            )

    @classmethod
    def empty(cls) -> Acquisitions:
        return cls(SEQUENCING_ACQUISITION_TABLE.empty_table())

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, object]],
    ) -> Acquisitions:
        rows = list(records)
        if not rows:
            return cls.empty()
        table = pa.Table.from_pylist(rows, schema=SEQUENCING_ACQUISITION_TABLE)
        return cls(table)

    @property
    def ids(self) -> list[int]:
        return self.table.column("acquisition_id").to_pylist()

    @property
    def metadata(self) -> dict[str, object]:
        return unpack_metadata(self.table.schema.metadata)

    def with_metadata(self, extras: dict[str, object]) -> Acquisitions:
        existing = unpack_metadata(self.table.schema.metadata)
        existing.update(extras)
        new_table = self.table.replace_schema_metadata(pack_metadata(existing))
        return Acquisitions(new_table)

    def __len__(self) -> int:
        return self.table.num_rows


# ──────────────────────────────────────────────────────────────────────
# Cross-check helper
# ──────────────────────────────────────────────────────────────────────


def validate_acquisitions(
    table: pa.Table,
    acquisitions: Acquisitions,
    *,
    column: str = "acquisition_id",
    nullable: bool = False,
) -> None:
    """Raise if ``table[column]`` references an unknown ``acquisition_id``.

    Sequencing equivalent of
    :func:`constellation.massspec.acquisitions.validate_acquisitions`;
    sibling helper that downstream containers (`Reads`, `Alignments`,
    `Quant`) call from their own ``validate()`` paths. ``nullable=True``
    permits a column where missing values mean "applies to all
    acquisitions" (e.g. run-agnostic reference annotations).
    """
    if column not in table.column_names:
        raise ValueError(f"table missing FK column {column!r}")
    col = table.column(column)
    known = set(acquisitions.ids)
    values = col.to_pylist()
    unknown: set[int] = set()
    for v in values:
        if v is None:
            if not nullable:
                raise ValueError(
                    f"column {column!r} contains nulls but nullable=False"
                )
            continue
        if v not in known:
            unknown.add(v)
    if unknown:
        sample = sorted(unknown)[:5]
        raise ValueError(
            f"{column} references unknown acquisition_ids: "
            f"{sample}{'...' if len(unknown) > 5 else ''}"
        )


__all__ = [
    "SEQUENCING_ACQUISITION_TABLE",
    "Acquisitions",
    "validate_acquisitions",
]
