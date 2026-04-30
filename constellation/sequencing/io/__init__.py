"""Cross-tier file-format adapters.

Houses readers/writers whose on-disk format crosses tier-of-abstraction
boundaries. SAM/BAM is the prototypical example: each record carries
both basecalled-read data (sequence, quality, read_id) and alignment
data (ref_name, position, CIGAR), so a single BAM file populates both
``Reads`` and ``Alignments`` containers.

Mirrors the role of :mod:`massspec.io` (where ``encyclopedia.dlib``
spans Library + Quant + Search). Each adapter self-registers with the
relevant Reader/Writer Protocol registries on import.

Modules:

    sam_bam     BAM/SAM ↔ (Reads, Alignments) — uses pysam
"""

from __future__ import annotations

from constellation.sequencing.io import sam_bam  # noqa: F401  registers adapters

__all__ = ["sam_bam"]
