"""``Assembly`` container — de novo contigs, scaffolds, summary stats.

Distinct from :class:`Reference` because an assembly carries
provenance Reference doesn't need: read-coverage per contig,
polishing-round counts, haplotype tags, scaffolding edges. Once an
assembly is finalized, ``Assembly.to_reference()`` lifts it into a
``Reference`` so downstream alignment / quant / annotation code is
uniform across external-ref and de-novo-ref workflows.

Tables:

    contigs       ASSEMBLY_CONTIG_TABLE — per-contig provenance
    sequences     SEQUENCE_TABLE        — same shape as Reference
    scaffolds     SCAFFOLD_TABLE        — optional; null-table when
                                          no scaffolding has run
    stats         ASSEMBLY_STATS        — one-row summary (N50, BUSCO)

Status: STUB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from constellation.sequencing.reference.reference import Reference


_PHASE = "Phase 7 (assembly/{hifiasm, stats}) for HiFiAsmRunner output"


@dataclass(frozen=True, slots=True)
class Assembly:
    """De novo assembly product, pre-finalization.

    ``to_reference()`` projects the contigs + sequences (and any
    annotated features added by downstream BUSCO / repeat-finding
    passes) into a ``Reference`` so the rest of the pipeline can
    treat external-ref and de-novo-ref workflows identically.
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

    def to_reference(self, *, features: pa.Table | None = None) -> Reference:
        """Project this Assembly into a ``Reference``.

        ``features`` is supplied separately because Assembly itself
        does not carry annotation features — those are added downstream
        (BUSCO genes, RepeatMasker hits, telomere detections). If
        omitted, an empty ``FEATURE_TABLE`` is used.
        """
        raise NotImplementedError(f"Assembly.to_reference pending {_PHASE}")

    @property
    def n_contigs(self) -> int:
        raise NotImplementedError(f"Assembly.n_contigs pending {_PHASE}")

    @property
    def total_length(self) -> int:
        raise NotImplementedError(f"Assembly.total_length pending {_PHASE}")


__all__ = ["Assembly"]
