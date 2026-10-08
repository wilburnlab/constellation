"""The read corpus: written once, mmapped by every round and every worker.

The EM loop reads the same ~9.4M transcript windows in every round, from
three places at once — minimap2 (as a FASTA), the seeder (as a table), and
the M-step (as per-member sequences for the PWM). The prototype held them as
a ``dict[read_id, str]`` and forked an M-step pool expecting copy-on-write to
share it. Measured, that does not work: a child touching 14% of a 1.5M-entry
dict privately dirties **845 MB**, because CPython writes the ``ob_refcnt``
word of every object it touches and those headers are scattered across
essentially every page. The same access against a memory-mapped Arrow buffer
dirties **8 MB** — the pages are file-backed and there are no per-element
Python objects to refcount.

So the corpus is an Arrow IPC file plus a FASTA, both written once:

* ``reads.arrow`` — ``read_id, sequence, sample_id, dorado_quality``, opened
  with :func:`pyarrow.memory_map` so the sequence bytes are never copied into
  any process's heap. Workers open it themselves rather than inheriting it,
  which keeps the design correct under ``spawn`` as well as ``fork``.
* ``reads.fa`` — the minimap2 query. **Each read is named by its corpus row
  index**, not its read_id. That is the keystone of the E-step rewrite: PAF's
  ``q_name`` becomes an integer parsable straight out of the byte buffer, so
  read grouping is ``np.diff`` on int64 rather than string comparison, the
  row index needed to look a sequence back up rides along for free, and the
  read_id string is materialised exactly once, on the winners.

Rows are dense and stable: row ``i`` of the IPC file is the read named
``>i`` in the FASTA, for the life of the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.align.map import _iter_demux_read_batches
from constellation.sequencing.transcriptome.cluster.denovo._io import (
    _READS_SCHEMA,
    _trim_batch,
)


CORPUS_ARROW = "reads.arrow"
CORPUS_FASTA = "reads.fa"
CORPUS_SUCCESS = "_SUCCESS"

# Reads per IPC record batch. Large enough that a 9.4M-read corpus is ~10
# batches (so `pc.take` has few chunks to hop), small enough that the writer's
# peak stays bounded.
_BATCH_ROWS = 1_000_000

# Rows per FASTA write chunk. Bounds the transient Python-string cost of the
# emitter to ~50k sequences regardless of how big a batch the join hands over.
_FASTA_CHUNK_ROWS = 50_000


@dataclass(frozen=True, slots=True)
class Corpus:
    """Where the corpus lives, plus what the length filter did to it."""

    directory: Path
    n_reads: int
    stats: dict[str, int]

    @property
    def arrow_path(self) -> Path:
        return self.directory / CORPUS_ARROW

    @property
    def fasta_path(self) -> Path:
        return self.directory / CORPUS_FASTA


@dataclass(frozen=True, slots=True)
class ReadStore:
    """Row-indexed, zero-copy access to a written corpus.

    Holds the mmapped table. ``sequence`` is a ChunkedArray over file-backed
    buffers, so ``pc.take(store.sequence, rows)`` materialises only the rows
    asked for — which is how the M-step gets one template's members without
    anything read-cardinality ever existing in Python.
    """

    table: pa.Table
    _mm: pa.MemoryMappedFile
    #: Row index at which each chunk starts, ``(n_chunks + 1,)``. Computed on
    #: open from chunk lengths alone — no data is touched.
    chunk_starts: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))

    @classmethod
    def open(cls, path: Path | str) -> ReadStore:
        mm = pa.memory_map(str(path), "r")
        with pa.ipc.open_file(mm) as reader:
            table = reader.read_all()
        col = table.column("sequence")
        lengths = (
            [len(c) for c in col.chunks] if isinstance(col, pa.ChunkedArray) else [len(col)]
        )
        starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        return cls(table=table, _mm=mm, chunk_starts=starts)

    @property
    def n_reads(self) -> int:
        return self.table.num_rows

    @property
    def read_id(self) -> pa.ChunkedArray:
        return self.table.column("read_id")

    @property
    def sequence(self) -> pa.ChunkedArray:
        return self.table.column("sequence")

    @property
    def sample_id(self) -> np.ndarray:
        return self.table.column("sample_id").to_numpy(zero_copy_only=False)

    @property
    def dorado_quality(self) -> np.ndarray:
        return self.table.column("dorado_quality").to_numpy(zero_copy_only=False)

    def lengths(self) -> np.ndarray:
        """Window length per row, without decoding any sequence."""
        return pc.utf8_length(self.sequence).to_numpy(zero_copy_only=False)

    def take_sequences(self, rows: np.ndarray) -> list[str]:
        """The sequences at ``rows``, as Python strings.

        The only sanctioned way to get ``str`` out of the corpus, and bounded
        in **residency** as well as in output: callers pass one template's
        member rows, never the whole corpus.

        That second guarantee is the whole point of :func:`chunked_take`. The
        obvious ``pc.take(self.sequence, rows)`` is not bounded — ``pc.take``
        concatenates every chunk of a ChunkedArray before indexing, so
        fetching ONE row off a 232 MB / 48-chunk mmapped corpus cost +219 MB
        RSS (measured). At 9.4M reads the corpus is ~25 GB, so that is ~47 GB
        of anonymous memory per worker, on first access, in every worker — a
        lazy file-backed corpus turned fully resident.
        """
        return chunked_take(self.sequence, rows, self.chunk_starts).to_pylist()

    def take_read_ids(self, rows: np.ndarray) -> pa.Array:
        """The read ids at ``rows``, without materialising the whole column."""
        return chunked_take(self.read_id, rows, self.chunk_starts).cast(pa.string())

    def close(self) -> None:
        self._mm.close()


def chunked_take(
    column: pa.ChunkedArray | pa.Array,
    rows: np.ndarray,
    chunk_starts: np.ndarray,
) -> pa.Array:
    """``column.take(rows)`` that touches only the chunks ``rows`` fall in.

    ``pyarrow.compute.take`` on a ChunkedArray concatenates the whole array
    first, so its cost is the column's size no matter how few rows are asked
    for. Resolving each row to ``(chunk, offset)`` and taking chunk-locally
    costs the chunks actually hit — measured +219 MB against +41 MB for one
    row of a 232 MB corpus, and the gap widens with the corpus.

    Input order is preserved.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return (
            column.chunk(0).slice(0, 0)
            if isinstance(column, pa.ChunkedArray) and column.num_chunks
            else column.slice(0, 0)
        )
    if not isinstance(column, pa.ChunkedArray) or column.num_chunks == 1:
        flat = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
        return flat.take(pa.array(rows))

    which = np.searchsorted(chunk_starts, rows, side="right") - 1
    np.clip(which, 0, column.num_chunks - 1, out=which)
    local = rows - chunk_starts[which]

    # Visit each chunk once, in chunk order, then restore the caller's order.
    order = np.argsort(which, kind="stable")
    parts, sizes = [], []
    lo = 0
    ordered_which = which[order]
    while lo < order.size:
        hi = int(np.searchsorted(ordered_which, ordered_which[lo], side="right"))
        sel = order[lo:hi]
        parts.append(column.chunk(int(ordered_which[lo])).take(pa.array(local[sel])))
        sizes.append(sel.size)
        lo = hi
    gathered = parts[0] if len(parts) == 1 else pa.concat_arrays(parts)
    inverse = np.empty(order.size, dtype=np.int64)
    inverse[order] = np.arange(order.size)
    return gathered.take(pa.array(inverse))


#: The corpus's row order: these columns, ascending. ``read_id`` alone is
#: unique for every demux written today; the other two only break ties
#: between rows that share one (a read whose demux record has several
#: windows), and rows equal in all three are identical, so their order
#: cannot matter.
CORPUS_ORDER: tuple[str, ...] = ("read_id", "sequence", "sample_id")


def corpus_digest(arrow_path: Path | str) -> str:
    """xxh3-128 of the corpus as ROWS: every read's id and sequence, in order.

    What anything keyed on a corpus ROW depends on. A row count is not an
    identity — the same reads in another order have the same count, which
    is exactly what a corpus rebuilt under :data:`CORPUS_ORDER` is to the
    one it replaced — and nor is the path, which is where the rebuilt one
    is written.

    A digest of the rows, not of how the file holds them: record-batch
    boundaries do not enter. One pass over the memory-mapped file.
    """
    import xxhash

    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        _string_chunks,
    )

    streams = {
        name: (xxhash.xxh3_128(), xxhash.xxh3_128())
        for name in ("read_id", "sequence")
    }
    n = 0
    with pa.memory_map(str(arrow_path), "r") as mm:
        reader = pa.ipc.open_file(mm)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            n += batch.num_rows
            for name, (of_lengths, of_bytes) in streams.items():
                column = batch.column(batch.schema.get_field_index(name))
                for lengths, held in _string_chunks(column):
                    # Lengths as well as bytes: the same bytes cut at other
                    # places are other reads.
                    of_lengths.update(np.ascontiguousarray(lengths, dtype="<i8"))
                    of_bytes.update(np.ascontiguousarray(held))
    state = xxhash.xxh3_128()
    state.update(b"constellation.em.corpus.rows/1")
    state.update(np.array([n], dtype="<i8"))
    for of_lengths, of_bytes in streams.values():
        state.update(of_lengths.digest())
        state.update(of_bytes.digest())
    return state.hexdigest()


def _in_corpus_order(table: pa.Table) -> pa.Table:
    """``table`` sorted by :data:`CORPUS_ORDER`, in memory."""
    if table.num_rows < 2:
        return table
    return table.take(
        pc.sort_indices(
            table.select(list(CORPUS_ORDER)),
            sort_keys=[(k, "ascending") for k in CORPUS_ORDER],
        )
    )


def _sort_into(unsorted_path: Path, arrow_path: Path, fasta_path: Path) -> int:
    """Rewrite ``unsorted_path`` to ``arrow_path`` + ``fasta_path`` in
    :data:`CORPUS_ORDER`. Returns the row count.

    Row order IS the corpus's identity downstream: FASTA names are row
    indices, ``uniq_id`` is first-occurrence order, the anchor-star breaks
    abundance ties on it, and the seed templates follow (ledger #56). The
    demux join hands reads over in whatever order its thread pool finishes
    them, so no streaming order is safe to rely on — not the default plan,
    and not a sequenced scan on a serial executor either, which was stable
    on 500k reads on a 12-core workstation and gave seven different orders
    of the same 9,445,987 reads on 96-core nodes (ledger #59, #61).

    Sorting by ``read_id`` takes the order out of the reader's hands
    entirely. It is a property of the read set alone, so it also survives
    a demux re-run that sharded its output differently, which the order of
    the ``reads/`` dataset would not.

    ``unsorted_path`` holds **sorted runs** — each record batch already in
    :data:`CORPUS_ORDER`, sorted in memory as it was written — so this is a
    merge: the global order takes a prefix of every run, then the next
    stretch of every run, and each :data:`_BATCH_ROWS` block gathers one
    contiguous, ascending range from each run. Every run is read once, front
    to back. Measured at 1.5M reads (2.1 GB) from a cold page cache, the
    gather takes 24 s from sorted runs against 47-55 s from an unsorted file
    of the same rows — the difference is random 4 kB faults into the memory
    map, which on a network filesystem and a 25 GB corpus cost more still.
    The sort itself is under a second; what remains is reading and writing
    the corpus once.

    Bounded: only the sort keys are compared (``sequence`` only on a tied
    ``read_id``), one block is gathered at a time, and the order is verified
    on what is written, batch by batch. Costs a second copy of the corpus on
    disk until the unsorted one is removed.
    """
    with pa.memory_map(str(unsorted_path), "r") as mm:
        table = pa.ipc.open_file(mm).read_all()
        n = table.num_rows
        lengths = [len(c) for c in table.column("read_id").chunks]
        chunk_starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        order = (
            pc.sort_indices(
                table.select(list(CORPUS_ORDER)),
                sort_keys=[(k, "ascending") for k in CORPUS_ORDER],
            )
            .to_numpy()
            .astype(np.int64)
            if n
            else np.empty(0, dtype=np.int64)
        )
        last_id: str | None = None
        with pa.OSFile(str(arrow_path), "wb") as sink, pa.ipc.new_file(
            sink, _READS_SCHEMA
        ) as writer, fasta_path.open("w", encoding="utf-8") as fh:
            row = 0
            for lo in range(0, n, _BATCH_ROWS):
                rows = order[lo : lo + _BATCH_ROWS]
                block = pa.table(
                    {
                        f.name: chunked_take(table.column(f.name), rows, chunk_starts)
                        for f in _READS_SCHEMA
                    },
                    schema=_READS_SCHEMA,
                )
                last_id = _check_sorted(block.column("read_id"), last_id)
                writer.write_table(block)
                row = _write_fasta_rows(fh, block, start_row=row)
    return n


def _check_sorted(ids: pa.ChunkedArray | pa.Array, before: str | None) -> str | None:
    """Raise unless ``ids`` is non-decreasing and starts at or after
    ``before``; return its last value. One pass over one written batch, so
    it is as cheap at 9.4M reads as at nine — the check it replaces loaded
    the whole ``reads/`` dataset (154M ids) into one string column and
    overflowed its offsets."""
    if isinstance(ids, pa.ChunkedArray):
        ids = ids.combine_chunks()
    if len(ids) == 0:
        return before
    if before is not None and ids[0].as_py() < before:
        raise RuntimeError(
            "the corpus is not in read_id order across a batch boundary; "
            "this is a defect in the corpus writer"
        )
    if len(ids) > 1 and pc.any(pc.less(ids.slice(1), ids.slice(0, len(ids) - 1))).as_py():
        raise RuntimeError(
            "the corpus is not in read_id order; this is a defect in the "
            "corpus writer"
        )
    return ids[len(ids) - 1].as_py()


def write_corpus(
    demux_dir: Path,
    output_dir: Path,
    *,
    max_window_length: int | None = 15_000,
    resume: bool = False,
) -> Corpus:
    """Stream the demux windows into ``output_dir/{reads.arrow,reads.fa}``.

    Replaces :func:`_io.load_demux_windows`'s ``pa.concat_tables`` of the whole
    corpus (~10 GB resident at 9.4M reads) with a streaming write: batches go
    straight to disk and the peak is one batch.

    ``max_window_length`` drops oversized windows here, at the corpus boundary,
    for the reason the original loader documents — every consumer reads this
    corpus, so filtering later removes a read from only one of them while it
    still wins a banded hit, still lands on a template, and still contributes
    its unaligned flank as a terminal extension event. Oversized windows are
    **dropped, never trimmed**: a 361,908 nt "cDNA" is a concatemer, and
    truncating it manufactures a read that was never sequenced.

    Empty windows are dropped too, so corpus rows are dense and every row has
    a FASTA record.

    Rows are in :data:`CORPUS_ORDER` — by ``read_id`` — whatever order the
    demux join produced them in (:func:`_sort_into` says why). Two corpora
    from one demux dir are byte-identical, and so is one from a demux re-run
    that sharded the same reads differently.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = output_dir / CORPUS_ARROW
    fasta_path = output_dir / CORPUS_FASTA
    success = output_dir / CORPUS_SUCCESS

    stats_path = output_dir / "stats.json"
    settings = {
        "demux_dir": str(Path(demux_dir).resolve()),
        "max_window_length": (
            int(max_window_length) if max_window_length is not None else None
        ),
    }
    settings_path = output_dir / "settings.json"
    if resume and success.exists() and arrow_path.exists() and stats_path.exists():
        _check_resume_settings(settings_path, settings)
        stats = {k: int(v) for k, v in json.loads(stats_path.read_text()).items()}
        return Corpus(directory=output_dir, n_reads=stats["n_reads"], stats=stats)

    n_input = 0
    n_dropped_long = 0
    n_dropped_empty = 0
    max_seen = 0

    tmp_arrow = arrow_path.with_suffix(".arrow.tmp")
    tmp_fasta = fasta_path.with_suffix(".fa.tmp")
    unsorted = output_dir / "reads.unsorted.arrow.tmp"
    writer = pa.ipc.new_file(pa.OSFile(str(unsorted), "wb"), _READS_SCHEMA)
    pending: list[pa.Table] = []
    pending_rows = 0
    try:
        # In whatever order the join hands them over: the order is imposed
        # by sorting each batch written and then merging the batches
        # (`_sort_into`), and nothing here may depend on it.
        for batch in _iter_demux_read_batches(demux_dir, only_complete=True):
            if batch.num_rows == 0:
                continue
            trimmed = _trim_batch(batch)
            n_input += trimmed.num_rows

            lengths = pc.utf8_length(trimmed.column("sequence"))
            if trimmed.num_rows:
                max_seen = max(max_seen, int(pc.max(lengths).as_py() or 0))

            keep = pc.greater(lengths, 0)
            n_dropped_empty += trimmed.num_rows - int(pc.sum(keep).as_py() or 0)
            if max_window_length is not None and max_window_length > 0:
                short_enough = pc.less_equal(lengths, max_window_length)
                n_dropped_long += trimmed.num_rows - int(
                    pc.sum(short_enough).as_py() or 0
                )
                keep = pc.and_(keep, short_enough)
            trimmed = trimmed.filter(keep)
            if trimmed.num_rows == 0:
                continue

            pending.append(trimmed)
            pending_rows += trimmed.num_rows
            if pending_rows >= _BATCH_ROWS:
                # One sorted run per batch: what makes `_sort_into` a merge
                # that reads each run once, in order, rather than a gather.
                writer.write_table(_in_corpus_order(pa.concat_tables(pending)))
                pending, pending_rows = [], 0
        if pending:
            writer.write_table(_in_corpus_order(pa.concat_tables(pending)))
    finally:
        writer.close()

    try:
        # FASTA names are corpus row indices, so the FASTA is written here,
        # in the final order, and nowhere else.
        n_written = _sort_into(unsorted, tmp_arrow, tmp_fasta)
    finally:
        unsorted.unlink(missing_ok=True)
    tmp_arrow.replace(arrow_path)
    tmp_fasta.replace(fasta_path)

    stats = {
        "n_reads": n_written,
        "n_input": n_input,
        "n_dropped_long": n_dropped_long,
        "n_dropped_empty": n_dropped_empty,
        "max_input_length": max_seen,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    # `row_order` is recorded, not compared: a corpus written before it was
    # sorted resumes as what it is, consistent with itself and with no other
    # run. Refusing it would strand every run directory written before.
    settings_path.write_text(
        json.dumps({**settings, "row_order": list(CORPUS_ORDER)}, indent=2)
    )
    success.write_bytes(b"")
    return Corpus(directory=output_dir, n_reads=n_written, stats=stats)


def _check_resume_settings(settings_path: Path, settings: dict) -> None:
    """Refuse a resume whose corpus was built from different inputs.

    The cache was keyed on nothing but its own existence, so pointing a
    resumed run at a different ``--demux-dir``, or asking for a different
    ``--max-window-length``, silently reused the corpus already there — and
    the manifest then recorded the parameters that had been ASKED for rather
    than the ones the reads on disk came from. Every number downstream is
    attributed to the wrong run.

    Rejecting rather than rebuilding, because the corpus is the run's whole
    substrate and a mistyped path is the more likely cause than a deliberate
    change. A corpus written before settings were recorded is accepted, since
    there is nothing to disagree with.
    """
    if not settings_path.exists():
        return
    try:
        stored = json.loads(settings_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    differing = {
        k: (stored.get(k), v) for k, v in settings.items() if stored.get(k) != v
    }
    if not differing:
        return
    detail = "; ".join(
        f"{k}: corpus was built with {was!r}, this run asks for {now!r}"
        for k, (was, now) in sorted(differing.items())
    )
    raise ValueError(
        f"cannot resume: the corpus at {settings_path.parent} does not match "
        f"this run's settings ({detail}). Use a different --output-dir, or "
        f"delete {settings_path.parent} to rebuild it."
    )


def _write_fasta_rows(fh, table: pa.Table, *, start_row: int) -> int:
    """Append ``table``'s windows as ``>{row}`` records. Returns the next row.

    Deliberately a Python loop over ``to_pylist()`` slices rather than numpy
    buffer surgery: a fully vectorised emitter measures **10x slower** here
    (7.18 s vs 0.70 s for 200k x 1.5 kb), because the ``np.repeat`` int64
    intermediates over the concatenated sequence dominate. The slicing is what
    bounds the transient — one chunk of strings at a time, not a batch.
    """
    row = start_row
    seq_col = table.column("sequence")
    for lo in range(0, table.num_rows, _FASTA_CHUNK_ROWS):
        for seq in seq_col.slice(lo, _FASTA_CHUNK_ROWS).to_pylist():
            fh.write(f">{row}\n{seq}\n")
            row += 1
    return row


__all__ = [
    "CORPUS_ARROW",
    "CORPUS_ORDER",
    "chunked_take",
    "CORPUS_FASTA",
    "Corpus",
    "ReadStore",
    "write_corpus",
]
