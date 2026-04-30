"""BAM/SAM cross-tier adapter.

A BAM file is conceptually two tables glued together: a Reads table
(sequence + quality + read_id) and an Alignments table (ref position +
CIGAR + tags). The :class:`Pod5Reader`-style ``RawReader`` lives at
:mod:`sequencing.readers.sam_bam` and produces the raw ``ReadResult``
companions; this module produces ``(Reads, Alignments)`` containers
ready for downstream code.

Mirrors :mod:`massspec.io.encyclopedia.adapters` — one shared decode
core, slot-specific adapters projecting it into each container's
Reader registry. Self-registers with ``ALIGNMENTS_READERS`` on import.

Status: STUB. Pending Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from constellation.sequencing.alignments.alignments import Alignments
from constellation.sequencing.alignments.io import (
    register_reader as register_alignments_reader,
)


_PHASE = "Phase 2 (readers/sam_bam + io/sam_bam adapter)"


def read_bam(path: Path, **opts: Any) -> tuple[Any, Alignments]:
    """Decode a BAM file into a ``(Reads, Alignments)`` tuple.

    Single shared core that callers project into per-container readers
    via the adapter classes below. Caller supplies an
    ``acquisition_id`` (typically via the ``Project`` layer that knows
    which acquisition the BAM belongs to).

    Returns:
        ``Reads`` container (TBD; not yet defined as a separate type
        — Phase 2 will decide whether to introduce ``Reads`` as a thin
        container around READ_TABLE or just pass the Arrow table)
        and ``Alignments``.
    """
    raise NotImplementedError(f"read_bam pending {_PHASE}")


def write_bam(path: Path, alignments: Alignments, **opts: Any) -> None:
    """Encode an Alignments container back to BAM."""
    raise NotImplementedError(f"write_bam pending {_PHASE}")


# ──────────────────────────────────────────────────────────────────────
# Container-Reader adapters (slot the shared decode into each registry)
# ──────────────────────────────────────────────────────────────────────


class _BamAlignmentsReader:
    """Slots BAM into the ALIGNMENTS_READERS registry."""

    extension: ClassVar[str] = ".bam"
    format_name: ClassVar[str] = "bam"

    def read(self, path: Path, **opts: Any) -> Alignments:
        _, alignments = read_bam(path, **opts)
        return alignments


# Register on import so consumers see ``.bam`` resolution by default.
register_alignments_reader(_BamAlignmentsReader())


__all__ = ["read_bam", "write_bam"]
