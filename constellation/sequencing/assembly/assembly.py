"""``Assembly`` container — de novo contigs, scaffolds, summary stats.

Distinct from :class:`GenomeReference` because an assembly carries
provenance Reference doesn't need: read-coverage per contig,
polishing-round counts, haplotype tags, scaffolding edges. Once an
assembly is finalized, ``Assembly.to_genome_reference()`` lifts it into
a ``GenomeReference`` so downstream alignment / quant code is uniform
across external-ref and de-novo-ref workflows. Annotation features
(genes, repeats, telomeres) are produced separately by downstream
BUSCO / RepeatMasker / telomere passes and live in their own
:class:`sequencing.annotation.Annotation` container that pairs with the
GenomeReference.

Tables:

    contigs       ASSEMBLY_CONTIG_TABLE — per-contig provenance
    sequences     SEQUENCE_TABLE        — same shape as GenomeReference
    scaffolds     SCAFFOLD_TABLE        — optional; null-table when
                                          no scaffolding has run
    stats         ASSEMBLY_STATS        — one-row summary (N50, BUSCO)

Status: STUB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from constellation.sequencing.reference.reference import GenomeReference


_PHASE = "Phase 7 (assembly/{hifiasm, stats}) for HiFiAsmRunner output"


@dataclass(frozen=True, slots=True)
class Assembly:
    """De novo assembly product, pre-finalization.

    ``to_genome_reference()`` projects the contigs + sequences into a
    ``GenomeReference`` so the rest of the pipeline can treat
    external-ref and de-novo-ref workflows identically. Annotation
    features (BUSCO genes, RepeatMasker hits, telomere detections) are
    built separately and ride in a paired ``Annotation`` container.
    """

    contigs: pa.Table
    sequences: pa.Table
    scaffolds: pa.Table | None
    stats: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raise NotImplementedError(f"Assembly cast + validate pending {_PHASE}")

    def validate(self) -> None:
        raise NotImplementedError(f"Assembly.validate pending {_PHASE}")

    def to_genome_reference(self) -> GenomeReference:
        """Project this Assembly into a ``GenomeReference``.

        Annotation features are NOT carried — those build downstream
        (BUSCO genes, RepeatMasker hits, telomere detections) into a
        sibling ``Annotation`` container that pairs with the resulting
        ``GenomeReference``.
        """
        raise NotImplementedError(f"Assembly.to_genome_reference pending {_PHASE}")

    @property
    def n_contigs(self) -> int:
        raise NotImplementedError(f"Assembly.n_contigs pending {_PHASE}")

    @property
    def total_length(self) -> int:
        raise NotImplementedError(f"Assembly.total_length pending {_PHASE}")


__all__ = ["Assembly"]
