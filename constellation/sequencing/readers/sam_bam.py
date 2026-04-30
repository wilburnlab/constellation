"""SAM / BAM reader — produces (READ_TABLE, ALIGNMENT_TABLE, ALIGNMENT_TAG_TABLE).

BAM is a cross-tier format: each record carries both basecalled-read
data (sequence, quality, read_id) AND alignment data (ref position,
CIGAR, tags). The reader populates both tiers and the cross-tier
adapter at :mod:`sequencing.io.sam_bam` projects the result into the
relevant containers (``Reads``, ``Alignments``).

Uses ``pysam`` for the binary decode — declared in the ``[sequencing]``
extra (lands in Phase 2). Imports are deferred to inside ``read`` so
the module imports without pysam installed.

Status: STUB. Pending Phase 2.
"""

from __future__ import annotations

from typing import ClassVar

from constellation.core.io.readers import RawReader, ReadResult, register_reader


_PHASE = "Phase 2 (readers/sam_bam)"


@register_reader
class BamReader(RawReader):
    """Decodes ``.bam`` into READ_TABLE + ALIGNMENT_TABLE + ALIGNMENT_TAG_TABLE.

    Returns a ``ReadResult`` with ``primary=READ_TABLE`` and companions
    keyed ``"alignments"`` and ``"alignment_tags"``. The cross-tier
    adapter at :mod:`sequencing.io.sam_bam` knows how to split this
    into ``(Reads, Alignments)``.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".bam",)
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        # import pysam  # noqa: ERA001
        raise NotImplementedError(f"BamReader.read pending {_PHASE}")


@register_reader
class SamReader(RawReader):
    """Decodes ``.sam`` (text-format alignments). Same output shape
    as BamReader."""

    suffixes: ClassVar[tuple[str, ...]] = (".sam",)
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"SamReader.read pending {_PHASE}")


__all__ = ["BamReader", "SamReader"]
