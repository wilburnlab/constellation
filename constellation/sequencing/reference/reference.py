"""``GenomeReference`` container — contigs + sequences as a unit.

Validation on construction: cast both tables to schema; PK uniqueness on
``contig_id``; FK closure (``SEQUENCE_TABLE.contig_id`` ⊆
``CONTIG_TABLE.contig_id``).

``GenomeReference`` is sample-agnostic — no per-acquisition columns. Per-run
observations against this reference live in
:class:`sequencing.alignments.Alignments`, per-feature counts in
:mod:`sequencing.quant`, and annotation features in
:class:`sequencing.annotation.Annotation` (which references this
container's contigs by id).

Two origins flow into this container:

    External reference     FASTA → GenomeReference (paired with an
                           Annotation built from companion GFF3)
    De novo assembly       Assembly.to_genome_reference() lifts a
                           finished assembly into the same shape so
                           downstream alignment / quant code is uniform

Mirrors the ``massspec.library.Library`` template: cast on construction,
PK / FK validation, ParquetDir round-trip via :mod:`sequencing.reference.io`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa

from constellation.core.io.schemas import cast_to_schema
from constellation.sequencing.schemas.reference import (
    CONTIG_TABLE,
    SEQUENCE_TABLE,
)


@dataclass(frozen=True, slots=True)
class GenomeReference:
    """Bundles ``CONTIG_TABLE`` + ``SEQUENCE_TABLE``."""

    contigs: pa.Table
    sequences: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contigs", cast_to_schema(self.contigs, CONTIG_TABLE))
        object.__setattr__(
            self, "sequences", cast_to_schema(self.sequences, SEQUENCE_TABLE)
        )
        self.validate()

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        """PK uniqueness on ``contig_id`` plus FK closure
        (``sequences.contig_id`` ⊆ ``contigs.contig_id``)."""
        _check_unique(self.contigs, "contig_id")
        contig_ids = set(self.contigs.column("contig_id").to_pylist())
        _check_fk(self.sequences, "contig_id", contig_ids, "CONTIG_TABLE")
        # One sequence per contig (uniqueness of sequence rows on
        # contig_id) — multiple rows would imply multi-allele or
        # ambiguous sequence, neither modelled here.
        seq_contigs = self.sequences.column("contig_id").to_pylist()
        if len(set(seq_contigs)) != len(seq_contigs):
            raise ValueError(
                "SEQUENCE_TABLE has multiple rows for the same contig_id; "
                "GenomeReference allows one sequence per contig"
            )

    # ── views ───────────────────────────────────────────────────────
    @property
    def n_contigs(self) -> int:
        return self.contigs.num_rows

    @property
    def total_length(self) -> int:
        """Sum of contig lengths in bp."""
        col = self.contigs.column("length").to_pylist()
        return int(sum(col))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def sequence_of(self, contig_id: int) -> str:
        """Return the literal nucleotide string for a contig.

        Raises ``KeyError`` if ``contig_id`` is not present.
        """
        sequences = self.sequences
        ids = sequences.column("contig_id").to_pylist()
        try:
            idx = ids.index(int(contig_id))
        except ValueError as exc:
            raise KeyError(f"contig_id {contig_id!r} not in GenomeReference") from exc
        return sequences.column("sequence")[idx].as_py()

    def with_metadata(self, extras: dict[str, Any]) -> "GenomeReference":
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)


# ──────────────────────────────────────────────────────────────────────
# Validation helpers (sibling to massspec.library — kept local to avoid
# a cross-domain import dependency).
# ──────────────────────────────────────────────────────────────────────


def _check_unique(table: pa.Table, column: str) -> None:
    values = table.column(column).to_pylist()
    if len(set(values)) != len(values):
        seen: set[Any] = set()
        dups: list[Any] = []
        for v in values:
            if v in seen:
                dups.append(v)
            else:
                seen.add(v)
        raise ValueError(
            f"{column} contains duplicate values: "
            f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
        )


def _check_fk(
    table: pa.Table,
    column: str,
    valid_ids: set[Any],
    target_name: str,
) -> None:
    values = table.column(column).to_pylist()
    missing = {v for v in values if v is not None and v not in valid_ids}
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{column} references ids not present in {target_name}: "
            f"{sample}{'...' if len(missing) > 5 else ''}"
        )


__all__ = ["GenomeReference"]
