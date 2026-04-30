"""``Reference`` container — contigs + sequences + features as a unit.

Validation on construction: cast all three tables to schema; PK
uniqueness on ``contig_id`` + ``feature_id``; FK closure
(SEQUENCE_TABLE.contig_id ⊆ CONTIG_TABLE.contig_id;
FEATURE_TABLE.contig_id ⊆ CONTIG_TABLE.contig_id;
FEATURE_TABLE.parent_id ⊆ FEATURE_TABLE.feature_id ∪ {null}).

``Reference`` is sample-agnostic — no per-acquisition columns. Per-run
observations against this reference live in
:class:`sequencing.alignments.Alignments` and per-feature counts in
:class:`sequencing.quant`.

Status: STUB. Container methods raise ``NotImplementedError`` pending
Phase 1 (Foundation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa


_PHASE = "Phase 1 (Foundation)"


@dataclass(frozen=True, slots=True)
class Reference:
    """Bundles ``CONTIG_TABLE`` + ``SEQUENCE_TABLE`` + ``FEATURE_TABLE``.

    Mirrors the ``massspec.library.Library`` template: cast on
    construction, PK / FK validation, ParquetDir round-trip via
    :mod:`sequencing.reference.io`.
    """

    contigs: pa.Table
    sequences: pa.Table
    features: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raise NotImplementedError(f"Reference cast + validate pending {_PHASE}")

    def validate(self) -> None:
        """Re-run PK uniqueness + FK closure checks. Called from
        ``__post_init__`` and after any mutation that produces a new
        Reference."""
        raise NotImplementedError(f"Reference.validate pending {_PHASE}")

    @property
    def n_contigs(self) -> int:
        raise NotImplementedError(f"Reference.n_contigs pending {_PHASE}")

    @property
    def n_features(self) -> int:
        raise NotImplementedError(f"Reference.n_features pending {_PHASE}")

    @property
    def total_length(self) -> int:
        """Sum of contig lengths in bp."""
        raise NotImplementedError(f"Reference.total_length pending {_PHASE}")

    def features_on(self, contig_id: int) -> pa.Table:
        """Return all features on a given contig."""
        raise NotImplementedError(f"Reference.features_on pending {_PHASE}")

    def sequence_of(self, contig_id: int) -> str:
        """Return the literal nucleotide string for a contig."""
        raise NotImplementedError(f"Reference.sequence_of pending {_PHASE}")

    def with_metadata(self, extras: dict[str, Any]) -> "Reference":
        raise NotImplementedError(f"Reference.with_metadata pending {_PHASE}")


__all__ = ["Reference"]
