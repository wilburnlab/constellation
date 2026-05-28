"""Per-sample FASTQ emission from an S1 demux output directory.

A thin sink that consumes the join + window-slice stream from
:func:`constellation.sequencing.align.map._iter_demux_read_batches`
(acero streaming hashjoin — one hash build over the filtered demux
table, ``reads/`` streamed through it) and routes records to one
gzipped FASTQ per ``sample_name``.  Aggregation across acquisitions /
barcodes that share a ``sample_id`` is automatic because ``sample_id``
is the routing key.

Output layout::

    <demux_dir>/fastq/
        _SUCCESS                       (stage-resume marker)
        <sample_name_1>.fq.gz
        <sample_name_2>.fq.gz
        ...

Filter policy (matches ``_iter_demux_read_batches(only_complete=True)``):
``status == 'Complete'`` + ``sample_id`` not null + ``is_fragment == False``
+ ``is_chimera == False`` + valid transcript window.  Reads with
``sample_id IS NULL`` are skipped silently.

Performance shape — three stacked optimisations vs the original
implementation:

* **Acero streaming join.**  The hash table over the filtered demux
  index is built once and reads stream through it (was rebuilt
  per-batch, catastrophic at 200M-read scale).
* **Byte-direct FASTQ assembly.**  Each batch's StringArrays are read
  via their value_offsets buffer; per-row work is bytearray slicing +
  ``.extend``, not Python str slicing + f-strings.  Skips ~200M Python
  string allocations + ASCII encodes at PromethION scale.
* **Threaded per-sample writers.**  One worker thread per open sample
  handle owns its gzip writer.  zlib releases the GIL during
  ``deflate`` so N samples compress concurrently, bounded by the
  number of available cores.
"""

from __future__ import annotations

import gzip
import os
import re
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pyarrow as pa

from constellation.sequencing.align.map import (
    _iter_demux_read_batches,
    _string_buf_and_offsets,
)
from constellation.core.progress import (
    ProgressCallback,
    emit_done,
    emit_progress,
    emit_start,
)
from constellation.sequencing.samples import Samples


_SANITIZE_RE = re.compile(r"[^\w.\-]")
# Hard-coded write-once compression level — trades ~5% size for ~2×
# throughput vs default 9.  Users who need maximum compression can
# re-gzip out-of-band.
_GZIP_COMPRESSLEVEL = 4
# Per-sample writer queue depth.  Small enough to bound memory at
# ~depth × largest_batch_chunk per sample, large enough to keep the
# zlib worker fed while the producer assembles the next batch.
_WRITER_QUEUE_DEPTH = 4


def _sanitize_name(name: str) -> str:
    """Map a sample_name to a filesystem-safe basename.

    Replaces any character outside ``[A-Za-z0-9_.-]`` with ``_``.
    """
    return _SANITIZE_RE.sub("_", name)


def _build_sample_filename_map(samples: Samples) -> dict[int, str]:
    """Resolve ``sample_id -> sanitised filename stem``.

    Raises ``ValueError`` if two distinct ``sample_name`` values
    sanitise to the same string — better to fail loudly than silently
    overwrite one sample's reads with another's.
    """
    ids = samples.samples.column("sample_id").to_pylist()
    names = samples.samples.column("sample_name").to_pylist()
    sanitised = [_sanitize_name(n) for n in names]

    collisions: dict[str, list[str]] = defaultdict(list)
    for original, clean in zip(names, sanitised):
        collisions[clean].append(original)
    duplicates = {clean: origs for clean, origs in collisions.items() if len(origs) > 1}
    if duplicates:
        raise ValueError(
            "sample_name values sanitise to colliding filenames: "
            + "; ".join(f"{clean!r} <- {origs}" for clean, origs in duplicates.items())
        )

    return dict(zip(ids, sanitised))


class _SampleWriter:
    """One worker thread per sample, fed by a bounded queue of byte chunks.

    The producer thread enqueues `bytes`; the worker pulls and calls
    ``handle.write(chunk)`` — zlib releases the GIL during deflate, so
    N writers run concurrently.  Bounded queue depth provides
    backpressure so a slow worker can't unbounded-buffer memory.
    """

    __slots__ = ("handle", "queue", "thread", "exc")

    def __init__(self, handle: gzip.GzipFile) -> None:
        self.handle = handle
        self.queue: Queue[bytes | None] = Queue(maxsize=_WRITER_QUEUE_DEPTH)
        self.exc: BaseException | None = None
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, chunk: bytes) -> None:
        if self.exc is not None:
            raise self.exc
        self.queue.put(chunk)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        try:
            self.handle.close()
        except OSError:
            pass
        if self.exc is not None:
            raise self.exc

    def _run(self) -> None:
        try:
            while True:
                chunk = self.queue.get()
                if chunk is None:
                    return
                self.handle.write(chunk)
        except BaseException as e:  # noqa: BLE001 — surfaced via .exc
            self.exc = e


def _format_per_sample_chunks(batch: pa.RecordBatch) -> dict[int, bytes]:
    """Group a joined batch by sample_id and assemble FASTQ bytes per sample.

    Operates on the raw StringArray byte buffers (no per-row str
    decode + encode round-trip).  Uses numpy.argsort for stable
    grouping by ``sample_id`` so the inner write loop touches each
    sample's contiguous run of rows once.
    """
    rid_data, rid_off = _string_buf_and_offsets(batch.column("read_id"))
    seq_data, seq_off = _string_buf_and_offsets(batch.column("sequence"))
    has_quality = "quality" in batch.schema.names
    if has_quality:
        qual_data, qual_off = _string_buf_and_offsets(batch.column("quality"))
    else:
        qual_data, qual_off = b"", None

    sample_ids = batch.column("sample_id").to_numpy()
    ts = batch.column("transcript_start").to_numpy()
    te = batch.column("transcript_end").to_numpy()
    n_rows = batch.num_rows
    if n_rows == 0:
        return {}

    # Stable sort indices by sample_id; np.split on diff boundaries
    # yields per-sample row-index runs in one pass.
    order = np.argsort(sample_ids, kind="stable")
    sorted_sids = sample_ids[order]
    if n_rows == 1:
        boundaries: np.ndarray = np.empty(0, dtype=np.int64)
    else:
        boundaries = np.where(np.diff(sorted_sids) != 0)[0] + 1
    runs = np.split(order, boundaries)

    out: dict[int, bytes] = {}
    for run in runs:
        if run.size == 0:
            continue
        sid = int(sample_ids[run[0]])
        buf = bytearray()
        for i in run:
            ts_i = int(ts[i])
            te_i = int(te[i])
            window_len = te_i - ts_i
            if window_len <= 0:
                continue
            row_data_end = int(seq_off[i + 1])
            seq_start = int(seq_off[i]) + ts_i
            seq_end = int(seq_off[i]) + te_i
            if seq_end > row_data_end:
                continue
            buf.append(0x40)  # '@'
            buf += rid_data[rid_off[i]:rid_off[i + 1]]
            buf.append(0x0A)  # '\n'
            buf += seq_data[seq_start:seq_end]
            buf += b"\n+\n"
            if qual_off is not None:
                q_start = int(qual_off[i]) + ts_i
                q_end = int(qual_off[i]) + te_i
                if q_end - q_start == window_len:
                    buf += qual_data[q_start:q_end]
                else:
                    # Null / wrong-length quality → synthetic Q40.
                    buf += b"I" * window_len
            else:
                buf += b"I" * window_len
            buf.append(0x0A)  # '\n'
        if buf:
            out[sid] = bytes(buf)
    return out


def emit_per_sample_fastq(
    demux_dir: Path,
    *,
    samples: Samples,
    batch_size: int = 100_000,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> Path:
    """Emit one ``.fq.gz`` per sample from an S1 demux output directory.

    ``demux_dir`` must contain partitioned ``reads/`` and ``read_demux/``
    datasets (the standard output of ``run_demux_pipeline``).  Writes
    each sample's reads to ``demux_dir/fastq/<sample_name>.fq.gz`` using
    transcript-window slicing + the strict Complete-only filter; reads
    with ``sample_id IS NULL`` are dropped silently.

    Resume semantics: if ``resume=True`` and
    ``demux_dir/fastq/_SUCCESS`` already exists, returns immediately
    without re-scanning any data.

    Atomicity: each sample writes to ``<name>.fq.gz.tmp`` and is renamed
    via ``os.replace`` after its last write.  ``_SUCCESS`` is touched
    only after every handle closes cleanly — a crashed run leaves
    ``.tmp`` files that a resumed run will overwrite.

    ``batch_size`` is accepted for source-compatibility but unused —
    acero's streaming join chooses its own internal batch size.

    Returns the ``fastq/`` directory path.
    """
    del batch_size  # acero handles its own batching
    demux_dir = Path(demux_dir)
    fastq_dir = demux_dir / "fastq"
    success_marker = fastq_dir / "_SUCCESS"

    if resume and success_marker.is_file():
        emit_done(
            progress_cb,
            "emit_fastq",
            message=f"resumed (existing fastq/ at {fastq_dir})",
        )
        return fastq_dir

    fastq_dir.mkdir(parents=True, exist_ok=True)
    if success_marker.is_file():
        success_marker.unlink()

    name_for_id = _build_sample_filename_map(samples)
    n_samples_known = len(name_for_id)

    emit_start(
        progress_cb,
        "emit_fastq",
        message=f"emitting per-sample FASTQ for up to {n_samples_known} samples",
    )

    writers: dict[int, _SampleWriter] = {}
    tmp_paths: dict[int, Path] = {}
    final_paths: dict[int, Path] = {}
    reads_written = 0

    try:
        for batch in _iter_demux_read_batches(demux_dir, only_complete=True):
            per_sample_bytes = _format_per_sample_chunks(batch)
            if not per_sample_bytes:
                continue

            for sid, chunk in per_sample_bytes.items():
                writer = writers.get(sid)
                if writer is None:
                    stem = name_for_id.get(sid)
                    if stem is None:
                        raise KeyError(
                            f"read_demux references sample_id={sid} not present "
                            f"in Samples; cannot resolve a filename"
                        )
                    final = fastq_dir / f"{stem}.fq.gz"
                    tmp = fastq_dir / f"{stem}.fq.gz.tmp"
                    handle = gzip.open(
                        tmp, "wb", compresslevel=_GZIP_COMPRESSLEVEL
                    )
                    writer = _SampleWriter(handle)
                    writers[sid] = writer
                    tmp_paths[sid] = tmp
                    final_paths[sid] = final
                writer.write(chunk)
                # Per-record count = (newlines in chunk) // 4, since
                # each FASTQ record is exactly 4 lines.
                reads_written += chunk.count(b"\n") // 4

            emit_progress(
                progress_cb,
                "emit_fastq",
                completed=reads_written,
                message=f"{len(writers)} samples open",
            )
    finally:
        # Drain + close every writer; a crashed run leaves .tmp files
        # for a future resumed run to overwrite.
        first_exc: BaseException | None = None
        for writer in writers.values():
            try:
                writer.close()
            except BaseException as e:  # noqa: BLE001
                if first_exc is None:
                    first_exc = e
        if first_exc is not None:
            raise first_exc

    for sid, tmp in tmp_paths.items():
        os.replace(tmp, final_paths[sid])

    success_marker.touch()

    emit_done(
        progress_cb,
        "emit_fastq",
        completed=reads_written,
        message=f"{len(final_paths)} files in {fastq_dir}",
    )
    return fastq_dir


__all__ = ["emit_per_sample_fastq"]
