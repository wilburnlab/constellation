"""FASTA / FASTQ reader → ``READ_TABLE``.

Unified FASTA + FASTQ reader (single-record or multi-record), gzip-
aware (``.fa.gz`` / ``.fastq.gz``). FASTA → READ_TABLE with
``quality=null``; FASTQ → fully-populated READ_TABLE with the offset-33
ASCII quality string preserved verbatim. Per-base Q-score decoding via
:func:`sequencing.quality.phred.decode_phred` is on demand.

The reader does NOT use pyteomics or biopython — these are simple
formats and the decoder is a few dozen lines of Python. Implementation
deferred to Phase 2.
"""

from __future__ import annotations

from typing import ClassVar

from constellation.core.io.readers import RawReader, ReadResult, register_reader


_PHASE = "Phase 2 (readers/fastx)"


@register_reader
class FastaReader(RawReader):
    suffixes: ClassVar[tuple[str, ...]] = (".fa", ".fasta", ".fna", ".faa")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"FastaReader.read pending {_PHASE}")


@register_reader
class FastaGzReader(RawReader):
    """Gzipped FASTA. Suffix-list covers the common combinations."""

    suffixes: ClassVar[tuple[str, ...]] = (".fa.gz", ".fasta.gz", ".fna.gz")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"FastaGzReader.read pending {_PHASE}")


@register_reader
class FastqReader(RawReader):
    suffixes: ClassVar[tuple[str, ...]] = (".fq", ".fastq")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"FastqReader.read pending {_PHASE}")


@register_reader
class FastqGzReader(RawReader):
    suffixes: ClassVar[tuple[str, ...]] = (".fq.gz", ".fastq.gz")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"FastqGzReader.read pending {_PHASE}")


__all__ = [
    "FastaReader",
    "FastaGzReader",
    "FastqReader",
    "FastqGzReader",
]
