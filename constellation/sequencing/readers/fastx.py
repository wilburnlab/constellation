"""FASTA / FASTQ readers.

Two classes of consumer share this module:

1. **Reads-as-input** — basecalled reads in FASTA/FASTQ form flowing into
   ``READ_TABLE``. ``FastaReader`` / ``FastqReader`` (RawReader
   subclasses) target this path; full implementation pending Phase 2.

2. **Reference-as-input** — FASTA files holding a genome, transcriptome,
   or genetic-tools panel that flow into ``GenomeReference`` /
   ``TranscriptReference`` / ``GeneticTools``. The ``read_fasta_*``
   functions below cover this path.

Both groups share a common streaming parser :func:`_iter_fasta` that
yields ``(record_id, description, sequence)`` per record. Gzip-aware
via stdlib ``gzip.open``. No external dependencies — pysam.FastaFile
would only earn its keep on huge (human-scale) genomes that need
``.fai``-indexed random contig access; we do full one-pass reads, so a
streaming parser is simpler and consistent with the GFF3/BED hand-rolled
philosophy.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.sequencing.genetic_tools import GeneticTools
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.reference import (
    CONTIG_TABLE,
    GENETIC_TOOL_CATEGORIES,
    GENETIC_TOOL_SEQUENCE_TYPES,
    GENETIC_TOOL_TABLE,
    SEQUENCE_TABLE,
    TRANSCRIPT_TABLE,
)
from constellation.sequencing.transcripts.transcripts import TranscriptReference


_PHASE = "Phase 2 (readers/fastx)"


# ──────────────────────────────────────────────────────────────────────
# Streaming FASTA parser (shared)
# ──────────────────────────────────────────────────────────────────────


def _open_text(path: Path) -> Any:
    """Open a file for text reading, transparently handling gzip."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield ``(record_id, description, sequence)`` per record.

    Header is parsed by splitting on first whitespace. Sequence lines
    are concatenated with whitespace stripped (handles tab-padded or
    line-wrapped sequences). Blank lines are skipped. Lines beginning
    with ``;`` are treated as comments per the legacy FASTA convention.
    """
    header: str | None = None
    seq_chunks: list[str] = []
    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if header is not None:
                    rec_id, desc = _split_header(header)
                    yield rec_id, desc, "".join(seq_chunks).replace(" ", "").replace(
                        "\t", ""
                    )
                header = line[1:].rstrip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
    if header is not None:
        rec_id, desc = _split_header(header)
        yield rec_id, desc, "".join(seq_chunks).replace(" ", "").replace("\t", "")


def _split_header(header: str) -> tuple[str, str]:
    parts = header.split(None, 1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


# ──────────────────────────────────────────────────────────────────────
# Reference-side public API
# ──────────────────────────────────────────────────────────────────────


def read_fasta_genome(path: str | Path) -> GenomeReference:
    """Parse a multi-record FASTA into a ``GenomeReference``.

    Records become contigs in input order with sequential ``contig_id``
    assigned starting at 0. ``name`` is the FASTA record id (everything
    before the first whitespace in the header line). ``length`` is
    ``len(sequence)``. ``topology`` and ``circular`` are left null —
    callers can stamp them via ``GenomeReference.with_metadata`` if the
    information is available.

    Header descriptions (text after the first whitespace) ride into
    ``metadata_extras['contig_descriptions']`` keyed by record name —
    useful when downstream code wants the Ensembl-style header bits
    (``chromosome:GRCh38:1:1:248956422:1 REF``) without being forced to
    parse them.
    """
    p = Path(path)
    contig_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    descriptions: dict[str, str] = {}
    for idx, (rec_id, desc, seq) in enumerate(_iter_fasta(p)):
        contig_rows.append(
            {
                "contig_id": idx,
                "name": rec_id,
                "length": len(seq),
                "topology": None,
                "circular": None,
            }
        )
        sequence_rows.append({"contig_id": idx, "sequence": seq})
        if desc:
            descriptions[rec_id] = desc

    contigs = (
        pa.Table.from_pylist(contig_rows, schema=CONTIG_TABLE)
        if contig_rows
        else CONTIG_TABLE.empty_table()
    )
    sequences = (
        pa.Table.from_pylist(sequence_rows, schema=SEQUENCE_TABLE)
        if sequence_rows
        else SEQUENCE_TABLE.empty_table()
    )
    metadata: dict[str, Any] = {"source_path": str(p)}
    if descriptions:
        metadata["contig_descriptions"] = descriptions
    return GenomeReference(
        contigs=contigs, sequences=sequences, metadata_extras=metadata
    )


def read_fasta_transcriptome(path: str | Path) -> TranscriptReference:
    """Parse a multi-record FASTA into a ``TranscriptReference``.

    Records become transcript rows in input order with sequential
    ``transcript_id`` starting at 0. ``name`` is the FASTA id;
    ``gene_id`` is left null (no Annotation linkage at parse time —
    callers can resolve it later by walking against an Annotation when
    one is available). ``source`` is set to ``'fasta_import'``.

    Header descriptions are stashed into
    ``metadata_extras['transcript_descriptions']``.
    """
    p = Path(path)
    rows: list[dict[str, Any]] = []
    descriptions: dict[str, str] = {}
    for idx, (rec_id, desc, seq) in enumerate(_iter_fasta(p)):
        rows.append(
            {
                "transcript_id": idx,
                "name": rec_id,
                "gene_id": None,
                "sequence": seq,
                "length": len(seq),
                "source": "fasta_import",
            }
        )
        if desc:
            descriptions[rec_id] = desc

    table = (
        pa.Table.from_pylist(rows, schema=TRANSCRIPT_TABLE)
        if rows
        else TRANSCRIPT_TABLE.empty_table()
    )
    metadata: dict[str, Any] = {"source_path": str(p)}
    if descriptions:
        metadata["transcript_descriptions"] = descriptions
    return TranscriptReference(transcripts=table, metadata_extras=metadata)


def read_fasta_genetic_tools(
    path: str | Path,
    *,
    category: str,
    sequence_type: str,
    source: str = "fasta_import",
) -> GeneticTools:
    """Parse a multi-record FASTA into a ``GeneticTools`` container.

    Caller supplies ``category`` and ``sequence_type`` because they are
    not derivable from the FASTA itself. For a panel that crosses
    multiple categories or sequence types, build separate FASTAs per
    category and merge the resulting GeneticTools containers via Arrow
    concat at the call site.
    """
    if category not in GENETIC_TOOL_CATEGORIES:
        raise ValueError(
            f"unknown category {category!r}; "
            f"allowed: {sorted(GENETIC_TOOL_CATEGORIES)}"
        )
    if sequence_type not in GENETIC_TOOL_SEQUENCE_TYPES:
        raise ValueError(
            f"unknown sequence_type {sequence_type!r}; "
            f"allowed: {sorted(GENETIC_TOOL_SEQUENCE_TYPES)}"
        )

    p = Path(path)
    rows: list[dict[str, Any]] = []
    for idx, (rec_id, _desc, seq) in enumerate(_iter_fasta(p)):
        rows.append(
            {
                "tool_id": idx,
                "name": rec_id,
                "category": category,
                "sequence_type": sequence_type,
                "sequence": seq,
                "source": source,
                "source_url": None,
                "references_json": None,
            }
        )

    table = (
        pa.Table.from_pylist(rows, schema=GENETIC_TOOL_TABLE)
        if rows
        else GENETIC_TOOL_TABLE.empty_table()
    )
    return GeneticTools(
        tools=table,
        metadata_extras={
            "source_path": str(p),
            "imported_category": category,
            "imported_sequence_type": sequence_type,
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Reads-as-input RawReader subclasses (still stubbed pending Phase 2)
# ──────────────────────────────────────────────────────────────────────


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
    "read_fasta_genome",
    "read_fasta_transcriptome",
    "read_fasta_genetic_tools",
]
