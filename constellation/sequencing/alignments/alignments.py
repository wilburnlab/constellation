"""``Alignments`` container — ALIGNMENT_TABLE + ALIGNMENT_TAG_TABLE.

Per-acquisition mapped-read records. Validation on construction:
``alignment_id`` PK uniqueness; ``acquisition_id`` FK closure into a
supplied ``Acquisitions``; tag-table ``alignment_id`` references must
exist in the parent table.

``validate_against(reference)`` cross-checks that every ``ref_name``
appears in the supplied ``GenomeReference.contigs.name`` set.

Status: STUB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from constellation.sequencing.acquisitions import Acquisitions
from constellation.sequencing.reference.reference import GenomeReference


_PHASE = "Phase 2 (Reader/Writer Protocols + readers/sam_bam)"


@dataclass(frozen=True, slots=True)
class Alignments:
    """Bundles ``ALIGNMENT_TABLE`` with its ``ALIGNMENT_TAG_TABLE``.

    Tag-table reference: alignment_id ⊆ alignments.alignment_id. Tag
    table may be empty (most alignments only carry promoted columns).
    """

    alignments: pa.Table
    tags: pa.Table
    acquisitions: Acquisitions | None = None
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raise NotImplementedError(f"Alignments cast + validate pending {_PHASE}")

    def validate(self) -> None:
        """PK uniqueness on alignment_id + FK closure (tags →
        alignments, alignments → acquisitions if set)."""
        raise NotImplementedError(f"Alignments.validate pending {_PHASE}")

    def validate_against(self, reference: GenomeReference) -> None:
        """Check that every ``ref_name`` appears in the reference's
        contig name list."""
        raise NotImplementedError(
            f"Alignments.validate_against pending {_PHASE}"
        )

    @property
    def n_alignments(self) -> int:
        raise NotImplementedError(f"Alignments.n_alignments pending {_PHASE}")

    def primary(self) -> "Alignments":
        """Filter to primary (non-secondary, non-supplementary) alignments."""
        raise NotImplementedError(f"Alignments.primary pending {_PHASE}")


__all__ = ["Alignments"]
