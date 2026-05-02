"""SAM / BAM reader — produces READ_TABLE (and ALIGNMENT_TABLE when relevant).

For the S1 transcriptomics pipeline, Dorado emits *unaligned* SAMs (FLAG=4,
RNAME=*, no CIGAR) and aligned BAMs (post-minimap2), so the natural primary
output is just READ_TABLE. The cross-tier `sequencing.io.sam_bam` adapter
(which would split aligned BAMs into Reads + Alignments containers) stays
stubbed until aligned-BAM workflows actually need the split.

Two read paths, identical API on both readers:

    {Sam,Bam}Reader.read(source, *, acquisition_id) -> ReadResult
        Eager full-file read into a single READ_TABLE.

    {Sam,Bam}Reader.iter_batches(path, batch_size, *, acquisition_id)
                                                -> Iterator[pa.Table]
        Streaming batch read. Yields READ_TABLE chunks of up to
        ``batch_size`` records via a single pass.

``SamReader`` is pure-Python text scan (no pysam dep). ``BamReader`` is
pysam-backed (``[sequencing]`` extra) and skips secondary (FLAG 0x100)
and supplementary (FLAG 0x800) alignments — primary records only, so
downstream READ_TABLE consumers see one row per read regardless of any
alignment-stage chimera split.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from constellation.core.io.bundle import Bundle
from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.sequencing.schemas.reads import READ_TABLE


# Canonical SAM fixed-field order (positions 1-11). Aligned to BAM/SAM
# spec; unaligned reads still occupy these slots with sentinel values.
_SAM_FIXED_FIELDS = (
    "QNAME",
    "FLAG",
    "RNAME",
    "POS",
    "MAPQ",
    "CIGAR",
    "RNEXT",
    "PNEXT",
    "TLEN",
    "SEQ",
    "QUAL",
)


def _mean_phred_offset33(quality: str) -> float | None:
    """Mean Phred score across a quality string (offset-33 ASCII).

    NanoporeAnalysis-equivalent: ``np.mean([ord(c) for c in QUAL]) - 33``.
    Computed in Python because the test SAM is ~7 MB; numpy round-trip
    isn't worth the overhead. Returns None for empty quality strings.
    """
    if not quality or quality == "*":
        return None
    total = 0
    for c in quality:
        total += ord(c)
    return (total / len(quality)) - 33.0


def _parse_st_z_to_unix_seconds(value: str) -> float | None:
    """Parse Dorado's ISO-8601 timestamp tag (``st:Z:2023-12-21T...``)
    to unix seconds. Returns None on parse failure."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


def _parse_record(line: str, acquisition_id: int) -> dict[str, Any] | None:
    """Convert one SAM body line to a READ_TABLE-shaped row dict.

    Returns None for empty / malformed lines (caller skips). Optional
    BAM-style tags ``ch:i:N``, ``st:Z:...``, ``du:f:F`` populate
    channel / start_time_s / duration_s; absent tags leave those nullable
    columns as None.
    """
    line = line.rstrip("\n").rstrip("\r")
    if not line:
        return None
    fields = line.split("\t")
    if len(fields) < 11:
        return None  # malformed; skip silently — header lines are
        # already gone, anything else short is corrupt
    fixed = dict(zip(_SAM_FIXED_FIELDS, fields[:11]))
    seq = fixed["SEQ"]
    qual = fixed["QUAL"]

    channel: int | None = None
    start_time_s: float | None = None
    duration_s: float | None = None
    for tag in fields[11:]:
        # Tag format: TT:Y:VALUE where TT is 2-char tag name and Y is
        # type-spec ('i'|'f'|'Z'|...). We only consume the three Dorado
        # tags S1's READ_TABLE projects.
        if len(tag) < 5 or tag[2] != ":" or tag[4] != ":":
            continue
        name = tag[:2]
        value = tag[5:]
        if name == "ch":
            try:
                channel = int(value)
            except ValueError:
                pass
        elif name == "st":
            start_time_s = _parse_st_z_to_unix_seconds(value)
        elif name == "du":
            try:
                duration_s = float(value)
            except ValueError:
                pass

    return {
        "read_id": fixed["QNAME"],
        "acquisition_id": acquisition_id,
        "sequence": seq,
        "quality": qual if qual != "*" else None,
        "length": len(seq),
        "mean_quality": _mean_phred_offset33(qual),
        "channel": channel,
        "start_time_s": start_time_s,
        "duration_s": duration_s,
    }


def _records_to_table(records: list[dict[str, Any]]) -> pa.Table:
    """Materialize a list of row dicts into a READ_TABLE-shaped Arrow
    table. Empty lists return an empty table conforming to schema."""
    if not records:
        return READ_TABLE.empty_table()
    return pa.Table.from_pylist(records, schema=READ_TABLE)


def _iter_record_lines(path: Path) -> Iterable[str]:
    """Open a SAM file and yield body lines (skipping ``@``-prefixed
    header lines and blanks)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith("@"):
                continue
            yield line


@register_reader
class SamReader(RawReader):
    """Decodes ``.sam`` (text-format) into READ_TABLE.

    Aligned-BAM tier (READ_TABLE + ALIGNMENT_TABLE companion split) is
    deferred to the cross-tier adapter at :mod:`sequencing.io.sam_bam`;
    today's transcriptomics workflow only consumes unaligned Dorado
    SAMs, so this reader emits READ_TABLE alone.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".sam",)
    modality: ClassVar[str | None] = "nanopore"

    def read(
        self,
        source: Path | Bundle | str,
        *,
        acquisition_id: int = 0,
    ) -> ReadResult:
        """Eager full-file read; returns a ReadResult with primary
        READ_TABLE."""
        path = self._resolve_path(source)
        records: list[dict[str, Any]] = []
        for line in _iter_record_lines(path):
            row = _parse_record(line, acquisition_id)
            if row is not None:
                records.append(row)
        table = _records_to_table(records)
        return ReadResult(
            primary=table,
            run_metadata={
                "source_path": str(path),
                "source_kind": "sam",
            },
        )

    def iter_batches(
        self,
        source: Path | Bundle | str,
        batch_size: int,
        *,
        acquisition_id: int = 0,
    ) -> Iterator[pa.Table]:
        """Yield READ_TABLE chunks of up to ``batch_size`` records each.

        Single streaming pass — the full file never sits in memory.
        Final partial batch (if any) is yielded as well. An empty file
        yields no batches at all.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        path = self._resolve_path(source)
        buffer: list[dict[str, Any]] = []
        for line in _iter_record_lines(path):
            row = _parse_record(line, acquisition_id)
            if row is None:
                continue
            buffer.append(row)
            if len(buffer) >= batch_size:
                yield _records_to_table(buffer)
                buffer = []
        if buffer:
            yield _records_to_table(buffer)

    @staticmethod
    def _resolve_path(source: Path | Bundle | str) -> Path:
        if isinstance(source, Bundle):
            return Path(source.path)
        return Path(source)


def _bam_record_to_row(rec: Any, acquisition_id: int) -> dict[str, Any]:
    """Convert a pysam ``AlignedSegment`` to a READ_TABLE-shaped row.

    Mirrors :func:`_parse_record`'s output exactly so SAM and BAM
    ingest produce byte-identical READ_TABLE rows for the same
    underlying record. Quality is rebuilt as offset-33 ASCII so the
    on-disk parquet representation matches the SAM path.
    """
    seq = rec.query_sequence or ""
    quals = rec.query_qualities  # numpy-like int array or None
    if quals is not None and len(quals) > 0:
        quality: str | None = "".join(chr(int(q) + 33) for q in quals)
        mean_q: float | None = float(sum(int(q) for q in quals) / len(quals))
    else:
        quality = None
        mean_q = None

    channel: int | None = None
    start_time_s: float | None = None
    duration_s: float | None = None
    try:
        channel = int(rec.get_tag("ch"))
    except KeyError:
        pass
    try:
        st_value = rec.get_tag("st")
        start_time_s = _parse_st_z_to_unix_seconds(str(st_value))
    except KeyError:
        pass
    try:
        duration_s = float(rec.get_tag("du"))
    except KeyError:
        pass

    return {
        "read_id": rec.query_name,
        "acquisition_id": acquisition_id,
        "sequence": seq,
        "quality": quality,
        "length": len(seq),
        "mean_quality": mean_q,
        "channel": channel,
        "start_time_s": start_time_s,
        "duration_s": duration_s,
    }


@register_reader
class BamReader(RawReader):
    """Decodes ``.bam`` (binary) into READ_TABLE.

    Mirrors :class:`SamReader`'s API exactly. Skips secondary
    (FLAG 0x100) and supplementary (FLAG 0x800) alignments — only
    primary records contribute to READ_TABLE so cDNA pipelines see
    one row per read regardless of any alignment-stage chimera split.

    pysam is imported lazily (sequencing extra dep) so environments
    without ``[sequencing]`` installed can still import the rest of
    the package.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".bam",)
    modality: ClassVar[str | None] = "nanopore"

    def read(
        self,
        source: Path | Bundle | str,
        *,
        acquisition_id: int = 0,
    ) -> ReadResult:
        path = self._resolve_path(source)
        records = list(self._iter_records(path, acquisition_id))
        table = _records_to_table(records)
        return ReadResult(
            primary=table,
            run_metadata={
                "source_path": str(path),
                "source_kind": "bam",
            },
        )

    def iter_batches(
        self,
        source: Path | Bundle | str,
        batch_size: int,
        *,
        acquisition_id: int = 0,
    ) -> Iterator[pa.Table]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        path = self._resolve_path(source)
        buffer: list[dict[str, Any]] = []
        for row in self._iter_records(path, acquisition_id):
            buffer.append(row)
            if len(buffer) >= batch_size:
                yield _records_to_table(buffer)
                buffer = []
        if buffer:
            yield _records_to_table(buffer)

    @staticmethod
    def _resolve_path(source: Path | Bundle | str) -> Path:
        if isinstance(source, Bundle):
            return Path(source.path)
        return Path(source)

    @staticmethod
    def _iter_records(path: Path, acquisition_id: int) -> Iterator[dict[str, Any]]:
        # Local import keeps the rest of the package usable in
        # environments that didn't install the [sequencing] extra.
        import pysam  # type: ignore[import-not-found]

        # check_sq=False: unaligned BAMs (Dorado direct output) have
        # no @SQ headers; without the flag pysam errors on open.
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as fh:
            for rec in fh.fetch(until_eof=True):
                if rec.is_secondary or rec.is_supplementary:
                    continue
                yield _bam_record_to_row(rec, acquisition_id)


__all__ = ["BamReader", "SamReader"]
