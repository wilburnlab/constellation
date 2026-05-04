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
    dorado_quality: float | None = None
    read_group: str | None = None
    duplex_class: int | None = None
    for tag in fields[11:]:
        # Tag format: TT:Y:VALUE where TT is 2-char tag name and Y is
        # type-spec ('i'|'f'|'Z'|...).
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
        elif name == "qs":
            try:
                dorado_quality = float(value)
            except ValueError:
                pass
        elif name == "RG":
            read_group = value
        elif name == "dx":
            try:
                duplex_class = int(value)
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
        "dorado_quality": dorado_quality,
        "read_group": read_group,
        "duplex_class": duplex_class,
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
        # Vectorized: pysam returns a uint8-typed array (or array.array
        # we can wrap zero-copy); shift by 33 in C and decode the byte
        # buffer once instead of doing chr() per base. ~5-10x faster
        # than the per-element Python loop on long nanopore reads.
        import numpy as np

        arr = np.asarray(quals, dtype=np.uint8)
        quality: str | None = (arr + np.uint8(33)).tobytes().decode("ascii")
        mean_q: float | None = float(arr.mean())
    else:
        quality = None
        mean_q = None

    channel: int | None = None
    start_time_s: float | None = None
    duration_s: float | None = None
    dorado_quality: float | None = None
    read_group: str | None = None
    duplex_class: int | None = None
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
    try:
        dorado_quality = float(rec.get_tag("qs"))
    except KeyError:
        pass
    try:
        read_group = str(rec.get_tag("RG"))
    except KeyError:
        pass
    try:
        duplex_class = int(rec.get_tag("dx"))
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
        "dorado_quality": dorado_quality,
        "read_group": read_group,
        "duplex_class": duplex_class,
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
        threads: int = 1,
    ) -> ReadResult:
        path = self._resolve_path(source)
        records = list(self._iter_records(path, acquisition_id, threads=threads))
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
        threads: int = 1,
    ) -> Iterator[pa.Table]:
        """Streaming batch read.

        ``threads`` controls htslib's bgzip-decompression thread pool
        (passed straight through to ``pysam.AlignmentFile(threads=N)``).
        Higher values parallelise BGZF block decompression in the
        background while the main thread parses records — ~2-3x
        speedup on large unaligned Dorado BAMs at ``threads=4-8``.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        path = self._resolve_path(source)
        buffer: list[dict[str, Any]] = []
        for row in self._iter_records(path, acquisition_id, threads=threads):
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
    def _iter_records(
        path: Path, acquisition_id: int, *, threads: int = 1
    ) -> Iterator[dict[str, Any]]:
        # Local import keeps the rest of the package usable in
        # environments that didn't install the [sequencing] extra.
        import pysam  # type: ignore[import-not-found]

        # check_sq=False: unaligned BAMs (Dorado direct output) have
        # no @SQ headers; without the flag pysam errors on open.
        # threads=N: htslib's BGZF thread pool decompresses blocks in
        # parallel with record parsing — biggest single win for the
        # ingest phase on >100 GB Dorado BAMs.
        with pysam.AlignmentFile(
            str(path), "rb", check_sq=False, threads=threads
        ) as fh:
            for rec in fh.fetch(until_eof=True):
                if rec.is_secondary or rec.is_supplementary:
                    continue
                yield _bam_record_to_row(rec, acquisition_id)


# ──────────────────────────────────────────────────────────────────────
# Parallel-ingest helpers
# ──────────────────────────────────────────────────────────────────────
#
# The high-level pipeline parallelises ingest by handing each worker a
# slice of the source file. The "what's a slice?" question has different
# answers per format:
#
#   BAM: a (virtual_offset_start, n_primary_records) pair. We pre-scan
#        the file once with bare pysam iteration (no field extraction)
#        to record record-boundary virtual offsets every N records, then
#        each worker seeks to its assigned vo_start and reads exactly
#        n_primary_records primary records. virtual offsets pulled from
#        ``pysam.AlignmentFile.tell()`` after a record read are always
#        record boundaries by construction, sidestepping the BAM-record-
#        spans-BGZF-block problem.
#   SAM: a (byte_start, byte_end) byte range. SAM is line-oriented
#        text, so workers seek to byte_start, skip a partial first line
#        if the offset is mid-record, and read complete lines until
#        byte_end. Each line goes through one worker exactly.


def _bam_record_chunks(
    path: Path,
    *,
    chunk_size: int = 100_000,
    threads: int = 1,
) -> list[tuple[int, int]]:
    """Scan a BAM and return record-boundary chunks for parallel demux.

    Each chunk is ``(virtual_offset_start, n_primary_records)``. Workers
    take a chunk, seek to ``virtual_offset_start``, and read exactly
    ``n_primary_records`` primary records (skipping any secondary /
    supplementary records that fall inside the range). Secondary /
    supplementary records are NOT counted toward chunk size — chunks
    are sized to the records that actually become READ_TABLE rows.

    The scan is intentionally minimal: bare ``for rec in fh.fetch()``
    iteration with no field extraction. ``threads`` is the htslib BGZF
    decompression thread pool — push higher for large BAMs.
    """
    import pysam  # type: ignore[import-not-found]

    chunks: list[tuple[int, int]] = []
    n_in_chunk = 0
    chunk_start_vo: int | None = None

    with pysam.AlignmentFile(
        str(path), "rb", check_sq=False, threads=threads
    ) as fh:
        last_vo = fh.tell()
        for rec in fh.fetch(until_eof=True):
            rec_vo = last_vo
            last_vo = fh.tell()
            if rec.is_secondary or rec.is_supplementary:
                continue
            if chunk_start_vo is None:
                chunk_start_vo = rec_vo
            n_in_chunk += 1
            if n_in_chunk >= chunk_size:
                chunks.append((chunk_start_vo, n_in_chunk))
                chunk_start_vo = None
                n_in_chunk = 0
        if n_in_chunk > 0 and chunk_start_vo is not None:
            chunks.append((chunk_start_vo, n_in_chunk))
    return chunks


def read_bam_chunk(
    path: Path,
    *,
    vo_start: int,
    n_primary: int,
    acquisition_id: int = 0,
) -> list[dict[str, Any]]:
    """Read exactly ``n_primary`` primary records starting at the
    given virtual offset; return a list of READ_TABLE-shaped dicts.

    Public so :func:`run_demux_pipeline`'s fused worker can call it
    directly. Worker-side: opens its own pysam handle (no shared state
    with the parent), seeks, reads, returns rows.
    """
    import pysam  # type: ignore[import-not-found]

    rows: list[dict[str, Any]] = []
    with pysam.AlignmentFile(
        str(path), "rb", check_sq=False, threads=1
    ) as fh:
        if vo_start > 0:
            fh.seek(vo_start)
        n_emitted = 0
        for rec in fh.fetch(until_eof=True):
            if rec.is_secondary or rec.is_supplementary:
                continue
            rows.append(_bam_record_to_row(rec, acquisition_id))
            n_emitted += 1
            if n_emitted >= n_primary:
                break
    return rows


def _sam_byte_chunks(
    path: Path,
    *,
    n_chunks: int,
) -> list[tuple[int, int]]:
    """Partition a SAM file into N byte-range chunks aligned at newlines.

    Each chunk is ``(byte_start, byte_end)``, byte_end exclusive. The
    parent doesn't pre-scan — workers handle line boundaries themselves
    via the protocol below. ``n_chunks`` is clamped to avoid microscopic
    chunks on tiny inputs.

    Boundary protocol (implemented in :func:`read_sam_chunk`):

      - Each worker seeks to ``byte_start``.
      - If ``byte_start > 0``, the worker skips the partial first line
        — that line was consumed by the previous worker, whose
        ``readline`` continued past its own ``byte_end``.
      - The worker then reads complete lines until ``fh.tell() >= byte_end``.
      - This guarantees every line is owned by exactly one worker.
    """
    file_size = path.stat().st_size
    if file_size == 0 or n_chunks < 1:
        return []
    n_chunks = max(1, min(n_chunks, file_size // 1024 + 1))
    chunks: list[tuple[int, int]] = []
    for i in range(n_chunks):
        byte_start = (file_size * i) // n_chunks
        byte_end = (
            (file_size * (i + 1)) // n_chunks if i < n_chunks - 1 else file_size
        )
        chunks.append((byte_start, byte_end))
    return chunks


def read_sam_chunk(
    path: Path,
    *,
    byte_start: int,
    byte_end: int,
    acquisition_id: int = 0,
) -> list[dict[str, Any]]:
    """Read a SAM byte range; return READ_TABLE-shaped row dicts.

    Skips ``@``-prefixed header lines (only encountered by chunk 0).
    Skips a partial first line if ``byte_start > 0`` per the boundary
    protocol in :func:`_sam_byte_chunks`.
    """
    rows: list[dict[str, Any]] = []
    with open(path, "rb") as fh:
        fh.seek(byte_start)
        if byte_start > 0:
            fh.readline()
        while True:
            pos = fh.tell()
            if pos >= byte_end:
                break
            line = fh.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text or text.startswith("@"):
                continue
            row = _parse_record(text, acquisition_id)
            if row is not None:
                rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Alignment-mode decode — emits ALIGNMENT_TABLE + ALIGNMENT_TAG_TABLE.
#
# Distinct from the READ_TABLE iterators above because:
#   - Cardinality differs: one BAM record per ALIGNMENT_TABLE row, but
#     READ_TABLE collapses secondary / supplementary records onto the
#     primary. Skipping secondaries here would miss real alignment data.
#   - Tag handling differs: ALIGNMENT_TAG_TABLE is long-format
#     (alignment_id, tag, type, value); the READ_TABLE path only pulls
#     the handful of Dorado tags it cares about.
#   - Unmapped records (FLAG 0x4) are skipped — ALIGNMENT_TABLE requires
#     ref_name / ref_start / ref_end to be non-null.
# ──────────────────────────────────────────────────────────────────────


def _serialize_tag_value(value: Any) -> str:
    """ALIGNMENT_TAG_TABLE.value is string-typed; cast on read.

    pysam returns bytes for Z-type tags on some Python versions and
    array.array for B-type arrays — handle both.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _bam_record_to_alignment_row(
    rec: Any,
    *,
    alignment_id: int,
    acquisition_id: int,
    tags_to_keep: tuple[str, ...] = (),
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Convert a pysam ``AlignedSegment`` to ALIGNMENT_TABLE + tag rows.

    Returns ``(None, [])`` for unmapped records (caller skips). Mapped
    records produce one ALIGNMENT_TABLE row plus zero or more
    ALIGNMENT_TAG_TABLE rows (one per tag listed in ``tags_to_keep``).

    Promoted-column extraction: ``nm_tag`` from NM, ``as_tag`` from AS,
    ``read_group`` from RG. ``tags_to_keep`` is the long-tail allowlist
    for entries that go into the long-format tag table — promoted tags
    don't need to be listed there but harmlessly may be.
    """
    if rec.is_unmapped:
        return None, []

    nm_tag: int | None = None
    as_tag: float | None = None
    read_group: str | None = None
    try:
        nm_tag = int(rec.get_tag("NM"))
    except KeyError:
        pass
    try:
        as_tag = float(rec.get_tag("AS"))
    except KeyError:
        pass
    try:
        read_group = str(rec.get_tag("RG"))
    except KeyError:
        pass

    tag_rows: list[dict[str, Any]] = []
    if tags_to_keep:
        keep_set = frozenset(tags_to_keep)
        for tag_name, value, type_code in rec.get_tags(with_value_type=True):
            if tag_name not in keep_set:
                continue
            tag_rows.append(
                {
                    "alignment_id": alignment_id,
                    "tag": tag_name,
                    "type": type_code,
                    "value": _serialize_tag_value(value),
                }
            )

    return (
        {
            "alignment_id": alignment_id,
            "read_id": rec.query_name,
            "acquisition_id": acquisition_id,
            "ref_name": rec.reference_name,
            "ref_start": int(rec.reference_start),
            "ref_end": int(rec.reference_end),
            "strand": "-" if rec.is_reverse else "+",
            "mapq": int(rec.mapping_quality),
            "flag": int(rec.flag),
            "cigar_string": rec.cigarstring or "",
            "nm_tag": nm_tag,
            "as_tag": as_tag,
            "read_group": read_group,
            "is_secondary": bool(rec.is_secondary),
            "is_supplementary": bool(rec.is_supplementary),
        },
        tag_rows,
    )


def _alignment_records_to_tables(
    alignment_rows: list[dict[str, Any]],
    tag_rows: list[dict[str, Any]],
) -> tuple[pa.Table, pa.Table]:
    """Materialise lists of dicts into (ALIGNMENT_TABLE, ALIGNMENT_TAG_TABLE)."""
    from constellation.sequencing.schemas.alignment import (
        ALIGNMENT_TABLE,
        ALIGNMENT_TAG_TABLE,
    )

    if alignment_rows:
        alignments = pa.Table.from_pylist(alignment_rows, schema=ALIGNMENT_TABLE)
    else:
        alignments = ALIGNMENT_TABLE.empty_table()
    if tag_rows:
        tags = pa.Table.from_pylist(tag_rows, schema=ALIGNMENT_TAG_TABLE)
    else:
        tags = ALIGNMENT_TAG_TABLE.empty_table()
    return alignments, tags


def read_bam_alignments(
    path: Path,
    *,
    acquisition_id: int = 0,
    tags_to_keep: tuple[str, ...] = (),
    threads: int = 1,
) -> tuple[pa.Table, pa.Table]:
    """Decode all mapped alignment records (incl. secondary /
    supplementary) into ``(ALIGNMENT_TABLE, ALIGNMENT_TAG_TABLE)``.

    For pipeline-scale parallel decode, use the chunked variant
    :func:`read_bam_alignments_chunk` together with
    :func:`_bam_alignment_chunks` and ``run_batched``. This entry point
    materialises the full table into memory — fine for tests and small
    Jupyter use, but at 30–200M alignments it OOMs.

    ``tags_to_keep`` controls which long-tail BAM tags land in the
    tag table; promoted columns (``nm_tag``, ``as_tag``, ``read_group``)
    always populate. Unmapped records are skipped — ALIGNMENT_TABLE
    requires ref_name / ref_start / ref_end to be non-null.
    """
    import pysam  # type: ignore[import-not-found]

    alignment_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    next_id = 0
    with pysam.AlignmentFile(
        str(path), "rb", check_sq=False, threads=threads
    ) as fh:
        for rec in fh.fetch(until_eof=True):
            row, tags = _bam_record_to_alignment_row(
                rec,
                alignment_id=next_id,
                acquisition_id=acquisition_id,
                tags_to_keep=tags_to_keep,
            )
            if row is None:
                continue
            alignment_rows.append(row)
            tag_rows.extend(tags)
            next_id += 1
    return _alignment_records_to_tables(alignment_rows, tag_rows)


def _bam_alignment_chunks(
    path: Path,
    *,
    chunk_size: int = 100_000,
    threads: int = 1,
) -> list[tuple[int, int]]:
    """Like :func:`_bam_record_chunks` but counts ALL records (incl.
    secondary / supplementary / unmapped) toward the chunk budget.

    Returns ``(virtual_offset_start, n_records)`` pairs. Workers seek
    to ``virtual_offset_start`` and read exactly ``n_records`` records
    of any kind. Unmapped records are still emitted by the iterator;
    the alignment-row converter drops them at row-build time. Counting
    them in the chunk budget keeps the chunker itself decision-free.
    """
    import pysam  # type: ignore[import-not-found]

    chunks: list[tuple[int, int]] = []
    n_in_chunk = 0
    chunk_start_vo: int | None = None

    with pysam.AlignmentFile(
        str(path), "rb", check_sq=False, threads=threads
    ) as fh:
        last_vo = fh.tell()
        for _rec in fh.fetch(until_eof=True):
            rec_vo = last_vo
            last_vo = fh.tell()
            if chunk_start_vo is None:
                chunk_start_vo = rec_vo
            n_in_chunk += 1
            if n_in_chunk >= chunk_size:
                chunks.append((chunk_start_vo, n_in_chunk))
                chunk_start_vo = None
                n_in_chunk = 0
        if n_in_chunk > 0 and chunk_start_vo is not None:
            chunks.append((chunk_start_vo, n_in_chunk))
    return chunks


def read_bam_alignments_chunk(
    path: Path,
    *,
    vo_start: int,
    n_records: int,
    worker_idx: int,
    acquisition_id: int = 0,
    tags_to_keep: tuple[str, ...] = (),
) -> tuple[pa.Table, pa.Table]:
    """Read exactly ``n_records`` records starting at the given virtual
    offset; return (ALIGNMENT_TABLE, ALIGNMENT_TAG_TABLE) for the chunk.

    ``alignment_id`` is allocated as ``(worker_idx << 32) | local_idx``
    so the partitioned shards can be concatenated without renumbering.
    Caller is responsible for ensuring ``worker_idx`` is unique across
    workers within a single ``run_batched`` invocation.
    """
    import pysam  # type: ignore[import-not-found]

    base = (int(worker_idx) & 0xFFFFFFFF) << 32
    alignment_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    local_idx = 0
    n_seen = 0
    with pysam.AlignmentFile(
        str(path), "rb", check_sq=False, threads=1
    ) as fh:
        if vo_start > 0:
            fh.seek(vo_start)
        for rec in fh.fetch(until_eof=True):
            n_seen += 1
            row, tags = _bam_record_to_alignment_row(
                rec,
                alignment_id=base | local_idx,
                acquisition_id=acquisition_id,
                tags_to_keep=tags_to_keep,
            )
            if row is not None:
                alignment_rows.append(row)
                tag_rows.extend(tags)
                local_idx += 1
            if n_seen >= n_records:
                break
    return _alignment_records_to_tables(alignment_rows, tag_rows)


__all__ = [
    "BamReader",
    "SamReader",
    "read_bam_chunk",
    "read_sam_chunk",
    "read_bam_alignments",
    "read_bam_alignments_chunk",
]
