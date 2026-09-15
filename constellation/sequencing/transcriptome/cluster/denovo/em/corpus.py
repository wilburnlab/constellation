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
from dataclasses import dataclass
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

    @classmethod
    def open(cls, path: Path | str) -> ReadStore:
        mm = pa.memory_map(str(path), "r")
        with pa.ipc.open_file(mm) as reader:
            table = reader.read_all()
        return cls(table=table, _mm=mm)

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

        The only sanctioned way to get ``str`` out of the corpus, and it is
        bounded by design: callers pass one template's member rows (capped at
        ``max_members_per_template``), never the whole corpus.
        """
        if len(rows) == 0:
            return []
        idx = pa.array(np.asarray(rows, dtype=np.int64))
        return pc.take(self.sequence, idx).to_pylist()

    def close(self) -> None:
        self._mm.close()


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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = output_dir / CORPUS_ARROW
    fasta_path = output_dir / CORPUS_FASTA
    success = output_dir / CORPUS_SUCCESS

    stats_path = output_dir / "stats.json"
    if resume and success.exists() and arrow_path.exists() and stats_path.exists():
        stats = {k: int(v) for k, v in json.loads(stats_path.read_text()).items()}
        return Corpus(directory=output_dir, n_reads=stats["n_reads"], stats=stats)

    n_input = 0
    n_dropped_long = 0
    n_dropped_empty = 0
    max_seen = 0
    n_written = 0

    tmp_arrow = arrow_path.with_suffix(".arrow.tmp")
    tmp_fasta = fasta_path.with_suffix(".fa.tmp")
    writer = pa.ipc.new_file(pa.OSFile(str(tmp_arrow), "wb"), _READS_SCHEMA)
    pending: list[pa.Table] = []
    pending_rows = 0
    try:
        with tmp_fasta.open("w", encoding="utf-8") as fh:
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

                # FASTA names are corpus row indices, assigned in write order.
                n_written = _write_fasta_rows(fh, trimmed, start_row=n_written)

                pending.append(trimmed)
                pending_rows += trimmed.num_rows
                if pending_rows >= _BATCH_ROWS:
                    writer.write_table(pa.concat_tables(pending))
                    pending, pending_rows = [], 0
            if pending:
                writer.write_table(pa.concat_tables(pending))
    finally:
        writer.close()

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
    success.write_bytes(b"")
    return Corpus(directory=output_dir, n_reads=n_written, stats=stats)


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
    "CORPUS_FASTA",
    "Corpus",
    "ReadStore",
    "write_corpus",
]
