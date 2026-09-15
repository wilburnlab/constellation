"""PAF (minimap2 tab-separated alignment) reader.

Two layers, because the two use cases have opposite shapes:

* :func:`iter_paf_batches` — the streaming primitive. Accepts a path **or a
  live pipe** (minimap2's stdout) and yields ``PAF_RECORD_SCHEMA`` record
  batches. This is what the ORF-anchored E-step consumes: with
  ``--secondary=yes -N 50`` every read can emit 51 lines, so ~51M lines
  (~6 GB of text) at 1M reads and ~510M at 10M. Nothing is written to disk
  and nothing accumulates.
* :class:`PafReader` — the registered ``RawReader`` for a ``.paf`` file,
  projecting into ``ALIGNMENT_TABLE`` + ``ALIGNMENT_TAG_TABLE``. PAF is a
  strict subset of BAM in column space, so there is no separate PAF schema
  on the output side; the BAM-only fields are filled with defaults and the
  query clipping is reconstructed into ``cigar_string`` so query coordinates
  round-trip through ``align/cigar.py``.

Parsing is **vectorized numpy buffer surgery**, not a per-line Python loop:
newline and tab offsets come from ``np.flatnonzero`` over the raw bytes, the
twelve fixed columns are built as zero-copy Arrow string arrays from
``(offsets, buffer)`` and cast in one shot, and the variable tag tail is
regex-extracted with ``pyarrow.compute``. Same technique as
``dereplicate._hash_sequences`` and ``map.transcript_window_buffers``; a
``for line in fh: line.split('\\t')`` loop is the exact pattern the
resolve-stage rules forbid.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, ClassVar, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.readers import RawReader, ReadResult, register_reader


# The 12 mandatory PAF columns plus the four tags anything downstream wants.
PAF_RECORD_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("q_name", pa.large_string(), nullable=False),
        pa.field("q_len", pa.int32(), nullable=False),
        pa.field("q_start", pa.int32(), nullable=False),
        pa.field("q_end", pa.int32(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),  # '+' | '-'
        pa.field("t_name", pa.large_string(), nullable=False),
        pa.field("t_len", pa.int32(), nullable=False),
        pa.field("t_start", pa.int32(), nullable=False),
        pa.field("t_end", pa.int32(), nullable=False),
        pa.field("n_match", pa.int32(), nullable=False),
        pa.field("aln_len", pa.int32(), nullable=False),
        pa.field("mapq", pa.int32(), nullable=False),
        # Tags — nullable, absent on some minimap2 modes.
        pa.field("as_score", pa.int32(), nullable=True),  # AS:i
        pa.field("tp", pa.string(), nullable=True),  # tp:A — P/S/I
        pa.field("nm", pa.int32(), nullable=True),  # NM:i
        pa.field("cigar", pa.large_string(), nullable=True),  # cg:Z
    ],
    metadata={b"schema_name": b"PafRecord"},
)

_N_FIXED = 12
_TAB = ord("\t")
_NL = ord("\n")


def _split_fixed_columns(buf: np.ndarray) -> tuple[list[pa.Array], pa.Array]:
    """Slice a newline-terminated PAF byte block into its columns.

    Returns the 12 mandatory columns as Arrow string arrays plus one array
    holding each row's remaining tag tail.
    """
    nl = np.flatnonzero(buf == _NL)
    if nl.size == 0:
        raise ValueError("PAF block contains no complete record")
    row_start = np.empty(nl.size, dtype=np.int64)
    row_start[0] = 0
    row_start[1:] = nl[:-1] + 1
    row_end = nl  # exclusive of the newline itself

    tab = np.flatnonzero(buf == _TAB)
    # Which row each tab belongs to, and where each row's tabs begin.
    tab_row = np.searchsorted(row_end, tab, side="left")
    first_tab = np.searchsorted(tab_row, np.arange(nl.size), side="left")
    n_tabs_per_row = np.diff(np.append(first_tab, tab.size))
    if np.any(n_tabs_per_row < _N_FIXED - 1):
        bad = int(np.flatnonzero(n_tabs_per_row < _N_FIXED - 1)[0])
        raise ValueError(
            f"malformed PAF record at row {bad}: expected at least "
            f"{_N_FIXED} fields, found {int(n_tabs_per_row[bad]) + 1}"
        )

    raw = pa.py_buffer(buf.tobytes())

    def _column(k: int) -> pa.Array:
        """Field ``k`` of every row, as a zero-copy Arrow string array."""
        starts = row_start if k == 0 else tab[first_tab + (k - 1)] + 1
        if k == _N_FIXED - 1:
            # Last fixed field ends at its own trailing tab (tags follow) or
            # at end-of-row when the record carries no tags.
            has_more = n_tabs_per_row >= _N_FIXED
            ends = np.where(
                has_more,
                tab[np.clip(first_tab + (_N_FIXED - 1), 0, tab.size - 1)],
                row_end,
            )
        else:
            ends = tab[first_tab + k]
        offsets = np.empty(nl.size + 1, dtype=np.int32)
        offsets[0] = 0
        widths = (ends - starts).astype(np.int32)
        np.cumsum(widths, out=offsets[1:])
        # Gather the field bytes into one contiguous buffer.
        idx = np.repeat(starts - offsets[:-1].astype(np.int64), widths) + np.arange(
            int(widths.sum()), dtype=np.int64
        )
        return pa.StringArray.from_buffers(
            nl.size,
            pa.py_buffer(offsets.tobytes()),
            pa.py_buffer(buf[idx].tobytes()),
        )

    cols = [_column(k) for k in range(_N_FIXED)]

    # Tag tail: everything after the 12th field's terminating tab.
    has_tags = n_tabs_per_row >= _N_FIXED
    tstart = np.where(
        has_tags, tab[np.clip(first_tab + (_N_FIXED - 1), 0, tab.size - 1)] + 1, row_end
    )
    widths = (row_end - tstart).astype(np.int32)
    offsets = np.empty(nl.size + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(widths, out=offsets[1:])
    idx = np.repeat(tstart - offsets[:-1].astype(np.int64), widths) + np.arange(
        int(widths.sum()), dtype=np.int64
    )
    tags = pa.StringArray.from_buffers(
        nl.size, pa.py_buffer(offsets.tobytes()), pa.py_buffer(buf[idx].tobytes())
    )
    del raw
    return cols, tags


def _extract_tag(tags: pa.Array, pattern: str, cast: pa.DataType) -> pa.Array:
    """Pull one ``XX:t:value`` tag out of the tail column."""
    m = pc.extract_regex(tags, pattern=pattern)
    vals = pc.struct_field(m, 0)
    return vals.cast(cast, safe=False)


def _block_to_batch(buf: np.ndarray, *, want_cigar: bool) -> pa.RecordBatch:
    cols, tags = _split_fixed_columns(buf)
    i32 = pa.int32()
    arrays = [
        cols[0].cast(pa.large_string()),
        cols[1].cast(i32),
        cols[2].cast(i32),
        cols[3].cast(i32),
        cols[4],
        cols[5].cast(pa.large_string()),
        cols[6].cast(i32),
        cols[7].cast(i32),
        cols[8].cast(i32),
        cols[9].cast(i32),
        cols[10].cast(i32),
        cols[11].cast(i32),
        _extract_tag(tags, r"AS:i:(?P<v>-?\d+)", i32),
        _extract_tag(tags, r"tp:A:(?P<v>[A-Za-z])", pa.string()),
        _extract_tag(tags, r"NM:i:(?P<v>\d+)", i32),
        (
            _extract_tag(tags, r"cg:Z:(?P<v>[0-9=XIDMSHNP]+)", pa.large_string())
            if want_cigar
            else pa.nulls(len(tags), pa.large_string())
        ),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=PAF_RECORD_SCHEMA)


def iter_paf_batches(
    source: "Path | str | BinaryIO | Iterable[bytes]",
    *,
    want_cigar: bool = True,
    chunk_bytes: int = 64 << 20,
) -> Iterator[pa.RecordBatch]:
    """Stream PAF into ``PAF_RECORD_SCHEMA`` record batches.

    ``source`` may be a path, an open binary file object, or an **iterable of
    byte chunks** — the last is what ``minimap2_stream`` yields off a live
    pipe, and it has no ``.read``. Reads ``chunk_bytes`` at a time and splits
    at the last newline, carrying the partial tail into the next block, so
    batch boundaries never split a record and the result is independent of
    ``chunk_bytes`` and of how the producer happened to chunk its output.
    """

    def _blocks():
        if isinstance(source, (str, Path)):
            with open(source, "rb") as fh:
                while True:
                    b = fh.read(chunk_bytes)
                    if not b:
                        return
                    yield b
        elif hasattr(source, "read"):
            while True:
                b = source.read(chunk_bytes)
                if not b:
                    return
                yield b
        else:
            yield from source

    tail = b""
    for raw_block in _blocks():
        block = tail + raw_block
        cut = block.rfind(b"\n")
        if cut < 0:
            tail = block
            continue
        tail = block[cut + 1 :]
        usable = block[: cut + 1]
        if usable:
            yield _block_to_batch(
                np.frombuffer(usable, dtype=np.uint8), want_cigar=want_cigar
            )
    leftover = tail.strip()
    if leftover:
        yield _block_to_batch(
            np.frombuffer(leftover + b"\n", dtype=np.uint8), want_cigar=want_cigar
        )


def read_paf(source: "Path | str", *, want_cigar: bool = True) -> pa.Table:
    """Whole-file convenience wrapper — only for small / diagnostic PAFs."""
    batches = list(iter_paf_batches(source, want_cigar=want_cigar))
    if not batches:
        return PAF_RECORD_SCHEMA.empty_table()
    return pa.Table.from_batches(batches, schema=PAF_RECORD_SCHEMA)


def paf_to_alignment_table(paf: pa.Table, *, acquisition_id: int = 0) -> pa.Table:
    """Project PAF records into ``ALIGNMENT_TABLE``.

    Query clipping is folded back into ``cigar_string`` as leading/trailing
    ``S`` ops, so ``align/cigar.py::query_start_from_cigar`` recovers
    ``q_start`` exactly the way it does for a BAM record.
    """
    from constellation.sequencing.schemas.alignment import ALIGNMENT_TABLE

    n = paf.num_rows
    if n == 0:
        return ALIGNMENT_TABLE.empty_table()
    q_start = paf.column("q_start").to_numpy(zero_copy_only=False)
    q_end = paf.column("q_end").to_numpy(zero_copy_only=False)
    q_len = paf.column("q_len").to_numpy(zero_copy_only=False)
    cig = paf.column("cigar").to_pylist()
    # PAF reports q_start/q_end on the FORWARD query regardless of strand,
    # but a SAM CIGAR is written in alignment orientation — so on a '-' hit the
    # clips swap. Reconstructing them forward-first would describe a different
    # molecule than the CIGAR does.
    fwd_lead, fwd_trail = q_start.astype(int), (q_len - q_end).astype(int)
    is_rev = np.array(
        [x == "-" for x in paf.column("strand").to_pylist()], dtype=bool
    )
    lead = np.where(is_rev, fwd_trail, fwd_lead)
    trail = np.where(is_rev, fwd_lead, fwd_trail)
    clipped = []
    for i in range(n):
        if not cig[i]:
            clipped.append("")
            continue
        head = f"{lead[i]}S" if lead[i] > 0 else ""
        tail = f"{trail[i]}S" if trail[i] > 0 else ""
        clipped.append(head + cig[i] + tail)
    strand = paf.column("strand").to_pylist()
    tp = paf.column("tp").to_pylist()
    return pa.table(
        {
            "alignment_id": pa.array(np.arange(n, dtype=np.int64)),
            "read_id": paf.column("q_name").cast(pa.string()),
            "acquisition_id": pa.array(np.full(n, acquisition_id, dtype=np.int64)),
            "ref_name": paf.column("t_name").cast(pa.string()),
            "ref_start": paf.column("t_start").cast(pa.int64()),
            "ref_end": paf.column("t_end").cast(pa.int64()),
            "strand": paf.column("strand"),
            "mapq": paf.column("mapq"),
            "flag": pa.array(
                [
                    (0x10 if s == "-" else 0) | (0x100 if t == "S" else 0)
                    for s, t in zip(strand, tp)
                ],
                type=pa.int32(),
            ),
            "cigar_string": pa.array(clipped, type=pa.string()),
            "nm_tag": paf.column("nm"),
            "as_tag": paf.column("as_score").cast(pa.float32()),
            "read_group": pa.nulls(n, pa.string()),
            "is_secondary": pa.array([t == "S" for t in tp], type=pa.bool_()),
            "is_supplementary": pa.array([t == "I" for t in tp], type=pa.bool_()),
        },
        schema=ALIGNMENT_TABLE,
    )


@register_reader
class PafReader(RawReader):
    """Decodes ``.paf`` → ALIGNMENT_TABLE (raw PAF view as a companion)."""

    suffixes: ClassVar[tuple[str, ...]] = (".paf",)
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        paf = read_paf(Path(source))
        return ReadResult(
            primary=paf_to_alignment_table(paf),
            companions={"paf": paf},
            run_metadata={"source": str(source), "format": "paf"},
        )


__all__ = [
    "PAF_RECORD_SCHEMA",
    "PafReader",
    "iter_paf_batches",
    "paf_to_alignment_table",
    "read_paf",
]
