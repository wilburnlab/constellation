"""Map cDNA transcripts to a genome reference → gene/exon features.

Splice-aware mapping with minimap2 (``-ax splice``) projects
``READ_DEMUX_TABLE`` consensus transcripts (or external transcript
FASTAs) onto a ``Reference`` and emits ``FEATURE_TABLE`` rows for
genes / mRNAs / exons.

This is the bridge between de novo transcriptome assembly and a
genomic Reference — a key step for the lab's matched genome +
transcriptome workflow.

Status: STUB. Pending Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from constellation.sequencing.reference.reference import Reference


_PHASE = "Phase 9 (annotation/transcripts)"


@dataclass(frozen=True)
class TranscriptMapper:
    """Splice-aware mapping of transcript sequences against a genome
    Reference. Uses minimap2 ``-ax splice`` under the hood; emits
    FEATURE_TABLE rows for the inferred gene / mRNA / exon hierarchy.
    """

    threads: int = 8
    extra_args: tuple[str, ...] = ()

    def run(
        self,
        reference: Reference,
        transcripts_fasta: Path,
        output_dir: Path,
    ) -> pa.Table:
        raise NotImplementedError(f"TranscriptMapper.run pending {_PHASE}")


def map_transcripts(
    reference: Reference,
    transcripts_fasta: Path,
    output_dir: Path,
    *,
    threads: int = 8,
) -> pa.Table:
    """Convenience wrapper — same as ``TranscriptMapper(threads=...)
    .run(...)`` for one-off use."""
    raise NotImplementedError(f"map_transcripts pending {_PHASE}")


__all__ = [
    "TranscriptMapper",
    "map_transcripts",
]
