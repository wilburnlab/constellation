"""GFF3 / GTF feature reader → ``FEATURE_TABLE``.

GFF3 is the canonical genome-annotation interchange format — gene /
mRNA / exon / CDS hierarchies linked via ``Parent=`` attributes, free-
form attribute column. Reader resolves the parent links into
``feature_id`` FKs and converts 1-based-inclusive GFF coordinates to
0-based-half-open BAM-internal coordinates at the boundary.

GTF is a near-subset (Ensembl-flavored); same reader handles both.

Pure-Python decode. Implementation deferred to Phase 2.
"""

from __future__ import annotations

from typing import ClassVar

from constellation.core.io.readers import RawReader, ReadResult, register_reader


_PHASE = "Phase 2 (readers/gff)"


@register_reader
class GffReader(RawReader):
    """Decodes ``.gff`` / ``.gff3`` / ``.gtf`` → FEATURE_TABLE."""

    suffixes: ClassVar[tuple[str, ...]] = (".gff", ".gff3", ".gtf")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"GffReader.read pending {_PHASE}")


@register_reader
class GffGzReader(RawReader):
    """Gzipped GFF / GTF — common in genome assembly pipelines."""

    suffixes: ClassVar[tuple[str, ...]] = (".gff.gz", ".gff3.gz", ".gtf.gz")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"GffGzReader.read pending {_PHASE}")


__all__ = ["GffReader", "GffGzReader"]
