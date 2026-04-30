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

Status: STUB. Schema definition is final; container methods raise
``NotImplementedError`` pending Phase 1 (Foundation). See plan file
in-our-development-of-fuzzy-quilt.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


SEQUENCING_ACQUISITION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        # 'pod5_dir' | 'pod5_file' | 'bam' | 'fastq' | 'fasta'
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


_PHASE = "Phase 1 (Foundation)"


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
        raise NotImplementedError(
            f"Acquisitions PK uniqueness + cast pending {_PHASE}"
        )

    @classmethod
    def empty(cls) -> "Acquisitions":
        raise NotImplementedError(f"Acquisitions.empty pending {_PHASE}")

    @classmethod
    def from_records(cls, records) -> "Acquisitions":
        raise NotImplementedError(f"Acquisitions.from_records pending {_PHASE}")

    @property
    def ids(self) -> list[int]:
        raise NotImplementedError(f"Acquisitions.ids pending {_PHASE}")

    def __len__(self) -> int:
        raise NotImplementedError(f"Acquisitions.__len__ pending {_PHASE}")


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
    raise NotImplementedError(f"validate_acquisitions pending {_PHASE}")


__all__ = [
    "SEQUENCING_ACQUISITION_TABLE",
    "Acquisitions",
    "validate_acquisitions",
]
