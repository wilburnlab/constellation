"""PAF (minimap2 tab-separated alignment) reader → ALIGNMENT_TABLE.

PAF columns map to a strict subset of BAM, so PAF rows land in
``ALIGNMENT_TABLE`` directly with the BAM-only fields (cigar tags
beyond the basic CIGAR string, secondary/supplementary flags) filled
with sensible defaults. No separate ``PAF_TABLE`` schema.

Pure-Python decode (no pysam dependency for PAF — it's tab-separated
text). Implementation deferred to Phase 2.
"""

from __future__ import annotations

from typing import ClassVar

from constellation.core.io.readers import RawReader, ReadResult, register_reader


_PHASE = "Phase 2 (readers/paf)"


@register_reader
class PafReader(RawReader):
    """Decodes ``.paf`` → ALIGNMENT_TABLE."""

    suffixes: ClassVar[tuple[str, ...]] = (".paf",)
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"PafReader.read pending {_PHASE}")


__all__ = ["PafReader"]
