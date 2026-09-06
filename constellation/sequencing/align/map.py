"""Mapping verbs — use-case orchestrators on top of ``minimap2_run``.

Each verb is a thin composition: materialise/locate the target,
construct the use-case-specific minimap2 arg list, stream input through
the minimap2 → samtools-sort pipeline, index. The minimap2 subprocess
wrapper itself stays generic (see :mod:`.minimap2`).

Shipped:

    map_to_genome      splice-aware full-length cDNA → genome alignment.
                       Input is an S1 demux output dir; we stream the
                       trimmed transcript window of each successfully-
                       demuxed read directly into minimap2's stdin —
                       NOT the raw Dorado BAMs (those still carry the
                       library scaffold S1 already located).

Pending:

    map_assembly       assembly-vs-assembly (asm5/asm10/asm20). Stub
                       carries a TODO flagging the same use-case
                       orchestrator pattern as ``map_to_genome``.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.acero as pa_ac
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds

from constellation.sequencing.align.minimap2 import minimap2_build_index
from constellation.core.progress import ProgressCallback, ProgressEvent
from constellation.sequencing.reference.materialize import materialise_genome_fasta
from constellation.sequencing.reference.reference import GenomeReference
from constellation.thirdparty.registry import ToolNotFoundError, find


# Flag set for forward-stranded full-length cDNA → genome (cdna_wilburn_v1).
# Lives at the orchestrator layer, not in the minimap2 runner.
_GENOME_MODE_ARGS: tuple[str, ...] = (
    "-ax",
    "splice",
    "-uf",
    "--cs=long",
    "--secondary=no",
)


def _resolve_minimap2() -> Path:
    try:
        return find("minimap2").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(
            "minimap2 not on $PATH; install via bioconda: "
            "`conda install -c bioconda minimap2 samtools`"
        ) from exc


def _resolve_samtools() -> Path:
    try:
        return find("samtools").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(
            "samtools not on $PATH; install via bioconda: "
            "`conda install -c bioconda samtools`"
        ) from exc


# Promoted to a shared helper so align / scaffold / polish share one
# implementation; kept as a private alias for the call site below.
_materialise_genome_fasta = materialise_genome_fasta


# ──────────────────────────────────────────────────────────────────────
# Transcript-window extraction (orientation-aware)
# ──────────────────────────────────────────────────────────────────────

#: A→T, C→G, G→C, T→A, case-preserved; everything else maps to itself so
#: IUPAC ambiguity codes and N pass through rather than corrupting.
_COMPLEMENT_LUT = np.arange(256, dtype=np.uint8)
for _a, _b in zip(b"ACGTacgt", b"TGCAtgca", strict=True):
    _COMPLEMENT_LUT[_a] = _b


def transcript_window_buffers(
    batch: "pa.RecordBatch",
    *,
    column: str = "sequence",
    complement: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract each read's transcript window as ``(bytes, offsets)``.

    ``transcript_start`` / ``transcript_end`` index the frame named by
    ``orientation``: for ``'-'`` reads that is the reverse-complement of
    the stored sequence, not the stored sequence itself. Slicing the
    stored bytes directly therefore takes the wrong strand over the
    wrong interval — a silent mis-extraction that makes reverse-oriented
    reads cluster apart from the forward reads of the same transcript.

    In the raw frame the chosen-frame window ``[ts, te)`` of a length-L
    read is ``[L - te, L - ts)``, read backwards and complemented. Both
    orientations are handled in one vectorized gather; no per-row Python.

    ``complement=False`` reverses without complementing — for quality
    strings, which are reversed alongside their bases but not
    complemented.
    """
    data, off = _string_buf_and_offsets(batch.column(column))
    off = off.astype(np.int64)
    n = batch.num_rows
    ts = batch.column("transcript_start").to_numpy(zero_copy_only=False).astype(np.int64)
    te = batch.column("transcript_end").to_numpy(zero_copy_only=False).astype(np.int64)
    row_start, row_end = off[:-1], off[1:]
    lengths = row_end - row_start

    rev = pc.equal(batch.column("orientation"), "-").to_numpy(zero_copy_only=False)
    rev = np.asarray(rev, dtype=bool)

    ts_c = np.clip(ts, 0, lengths)
    te_c = np.clip(te, ts_c, lengths)
    raw_start = np.where(rev, row_start + (lengths - te_c), row_start + ts_c)
    raw_end = np.where(rev, row_start + (lengths - ts_c), row_start + te_c)

    lens = raw_end - raw_start
    new_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=new_off[1:])
    total = int(new_off[-1])
    if total == 0:
        return np.empty(0, dtype=np.uint8), new_off

    seg = np.repeat(np.arange(n, dtype=np.int64), lens)
    within = np.arange(total, dtype=np.int64) - new_off[seg]
    rev_rows = rev[seg]
    src = np.where(rev_rows, raw_end[seg] - 1 - within, raw_start[seg] + within)
    buf = np.frombuffer(data, dtype=np.uint8)[src]
    if complement and rev_rows.any():
        buf = buf.copy()
        buf[rev_rows] = _COMPLEMENT_LUT[buf[rev_rows]]
    return buf, new_off


def _iter_demux_read_batches(
    demux_dir: Path,
    *,
    only_complete: bool = True,
    batch_size: int = 100_000,
) -> Iterator[pa.RecordBatch]:
    """Stream joined reads ⨝ demux record batches.

    Builds a single acero ExecPlan: scan ``reads/`` as a dataset on the
    probe side, hash-build over the filtered ``read_demux/`` table on
    the build side, inner-join on ``read_id``.  The hash table is
    constructed **once** and the reads stream through it — the
    previous implementation rebuilt the join plan per batch, which is
    catastrophic at 200M-read scale.

    Filter when ``only_complete=True`` (default for genome-guided
    quantification — see the chimera-handling discussion in the S2 plan):

        * ``status == 'Complete'``
        * ``sample_id`` not null
        * ``is_fragment == False``
        * ``is_chimera == False``  (forward-compat; S1 always emits False
          today, lights up when S3 enables multi-segment splitting)
        * ``transcript_start >= 0`` AND ``transcript_end > transcript_start``

    Yields ``pa.RecordBatch`` instances with columns ``read_id``,
    ``sequence``, ``quality`` (if present on the reads dataset),
    ``sample_id``, ``orientation``, ``transcript_start``,
    ``transcript_end``.  Window slicing is left to the consumer — the
    byte buffers + value_offsets are addressable directly without
    per-row Python string allocation.

    ``orientation`` is load-bearing, not informational: when it is
    ``'-'`` the window offsets index the reverse-complement of
    ``sequence``, so a consumer that slices ``sequence`` directly
    extracts the wrong strand over the wrong interval. Use
    :func:`reoriented_windows` rather than slicing by hand.

    ``batch_size`` is unused (preserved for source compatibility with
    the previous signature); acero chooses its own internal batch size.

    Memory at 200M-read scale: filtered demux index ~10 GB (50 B/row),
    resident as the hash side; reads streamed via the scan node.
    """
    del batch_size  # acero handles its own batching
    demux_dir = Path(demux_dir)
    reads_dir = demux_dir / "reads"
    demux_part_dir = demux_dir / "read_demux"
    if not reads_dir.is_dir():
        raise FileNotFoundError(
            f"demux output missing reads/: {demux_dir}"
        )
    if not demux_part_dir.is_dir():
        raise FileNotFoundError(
            f"demux output missing read_demux/: {demux_dir}"
        )

    if only_complete:
        filt = (
            (pc.field("status") == "Complete")
            & pc.is_valid(pc.field("sample_id"))
            & pc.invert(pc.field("is_fragment"))
            & pc.invert(pc.field("is_chimera"))
            & (pc.field("transcript_start") >= 0)
            & (pc.field("transcript_end") > pc.field("transcript_start"))
        )
    else:
        filt = (pc.field("transcript_start") >= 0) & (
            pc.field("transcript_end") > pc.field("transcript_start")
        )

    demux_ds = pa_ds.dataset(demux_part_dir)
    if "orientation" not in set(demux_ds.schema.names):
        # Pre-orientation demux output. Slicing without it silently
        # extracts the wrong strand over the wrong interval for every
        # reverse-oriented read, so refuse rather than quietly mis-slice.
        raise ValueError(
            f"{demux_part_dir} predates the `orientation` column in "
            f"READ_DEMUX_TABLE, so transcript windows on reverse-oriented "
            f"reads cannot be sliced correctly. Re-run `constellation "
            f"transcriptome demultiplex` to regenerate this directory."
        )
    demux_table = demux_ds.to_table(
        columns=[
            "read_id",
            "sample_id",
            "orientation",
            "transcript_start",
            "transcript_end",
        ],
        filter=filt,
    )
    if demux_table.num_rows == 0:
        return

    reads_ds = pa_ds.dataset(reads_dir)
    has_quality = "quality" in set(reads_ds.schema.names)
    reads_columns = ["read_id", "sequence"] + (["quality"] if has_quality else [])

    # Build the streaming plan: dataset scan -> hashjoin with the
    # in-memory demux table on the right (build) side.
    scan_node = pa_ac.Declaration(
        "scan",
        pa_ac.ScanNodeOptions(reads_ds, columns=reads_columns),
    )
    # ``scan`` emits internal __fragment_index / __batch_index /
    # __last_in_fragment columns; project them away before joining so
    # they don't end up in the output schema.
    projection_fields = [pc.field(col) for col in reads_columns]
    project_node = pa_ac.Declaration(
        "project",
        pa_ac.ProjectNodeOptions(projection_fields, reads_columns),
    )
    demux_node = pa_ac.Declaration(
        "table_source",
        pa_ac.TableSourceNodeOptions(demux_table),
    )
    right_output = [
        "sample_id",
        "orientation",
        "transcript_start",
        "transcript_end",
    ]
    join_node = pa_ac.Declaration(
        "hashjoin",
        pa_ac.HashJoinNodeOptions(
            "inner",
            left_keys=["read_id"],
            right_keys=["read_id"],
            left_output=reads_columns,
            right_output=right_output,
        ),
        inputs=[
            pa_ac.Declaration.from_sequence([scan_node, project_node]),
            demux_node,
        ],
    )

    reader = join_node.to_reader(use_threads=True)
    try:
        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield batch
    finally:
        reader.close()


def _string_buf_and_offsets(
    arr: pa.Array | pa.ChunkedArray,
) -> tuple[bytes, np.ndarray]:
    """Extract the raw byte buffer + value_offsets from a StringArray.

    Acero's join output may be a ChunkedArray with a single chunk
    (combine before slicing); the returned offsets are int32 with
    ``len(arr) + 1`` entries, and ``arr[i] == data[offsets[i]:offsets[i+1]]``.
    """
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if arr.offset != 0:
        # An Arrow slice keeps a non-zero offset; combine_chunks resets it
        # for ChunkedArray, but for a single Array we'd need to copy.
        # Easiest: round-trip through combine_chunks via a 1-chunk Chunked.
        arr = pa.chunked_array([arr]).combine_chunks()
    buffers = arr.buffers()
    offsets = np.frombuffer(buffers[1], dtype=np.int32, count=len(arr) + 1)
    data = bytes(buffers[2]) if buffers[2] is not None else b""
    return data, offsets


def _format_fastq_bytes(batch: pa.RecordBatch) -> tuple[bytes, int]:
    """Encode a joined batch to FASTQ ASCII bytes (sample identity dropped).

    Output: ``@read_id\ntranscript_window\n+\nquality_window\n`` per
    row. Suitable for piping into a single alignment process (e.g.
    minimap2 stdin).

    The window is taken via :func:`transcript_window_buffers`, so a
    ``'-'`` read is emitted reverse-complemented into the sense strand
    with its quality reversed to match. Slicing the stored bytes
    directly emitted the wrong strand over the wrong interval for every
    reverse-oriented read.
    """
    rid_data, rid_off = _string_buf_and_offsets(batch.column("read_id"))
    seq_buf, seq_off = transcript_window_buffers(batch)
    has_quality = "quality" in batch.schema.names
    if has_quality:
        qual_buf, qual_off = transcript_window_buffers(
            batch, column="quality", complement=False
        )
    else:
        qual_buf, qual_off = None, None

    out = bytearray()
    n = 0
    for i in range(batch.num_rows):
        start, end = int(seq_off[i]), int(seq_off[i + 1])
        width = end - start
        if width <= 0:
            continue
        out.append(0x40)  # '@'
        out += rid_data[rid_off[i]:rid_off[i + 1]]
        out.append(0x0A)  # '\n'
        out += seq_buf[start:end].tobytes()
        out += b"\n+\n"
        if qual_off is not None and int(qual_off[i + 1]) - int(qual_off[i]) == width:
            out += qual_buf[int(qual_off[i]):int(qual_off[i + 1])].tobytes()
        else:
            # No quality column, or this row's value was null / the wrong
            # length — synthesise Q40 across the window.
            out += b"I" * width
        out.append(0x0A)  # '\n'
        n += 1
    return bytes(out), n


def map_to_genome(
    demux_dir: Path,
    genome: GenomeReference,
    *,
    output_dir: Path,
    threads: int = 8,
    only_complete: bool = True,
    minimap2_args: tuple[str, ...] | None = None,
    extra_minimap2_args: tuple[str, ...] = (),
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Splice-aware full-length cDNA → genome alignment via minimap2.

    Streams trimmed transcript-window FASTQ from the S1 demux output
    directory directly into minimap2's stdin (no intermediate FASTQ
    file, no raw-BAM input) — the S1 demux already located the library
    scaffold (5'/3' adapter, polyA, barcode), so we map only the
    transcript window of each Complete-status read. See
    :func:`_iter_demux_read_batches` for the filter set.

    Pipeline shape:

        [Python writer thread]    [minimap2 -t N]    [samtools sort]
          pa.dataset batches ──FASTQ─> reads stdin ──SAM─> sorts ──BAM─> aligned.bam

    minimap2 sees one continuous FASTQ stream — batch boundaries are a
    Python-side memory knob, not a parallelism boundary. ``samtools
    sort`` consumes minimap2's SAM stdout and writes a coordinate-
    sorted BAM in one pass; ``samtools index`` follows.

    Materialises ``genome`` to a cached FASTA at ``output_dir/genome.fa``
    and a ``.mmi`` index at ``output_dir/genome.mmi``; both reuse on
    subsequent calls (cache-busted by contig count).

    Parameters
    ----------
    minimap2_args
        Full minimap2 argument tuple. When provided, this is used
        verbatim (and ``extra_minimap2_args`` is ignored). Typically
        constructed by
        :func:`constellation.sequencing.align.presets.resolve_minimap2_args`
        at the CLI layer so the splice-mode base flags + preset +
        explicit overrides + escape-hatch extra args compose
        consistently. When ``None``, the function falls back to
        ``_GENOME_MODE_ARGS + extra_minimap2_args`` for backward
        compatibility with callers that don't use the resolver.
    extra_minimap2_args
        Back-compat path. Only consulted when ``minimap2_args`` is
        ``None``. New callers should construct the full arg list via
        the resolver and pass it through ``minimap2_args`` instead.

    Returns the final sorted, indexed BAM path.
    """
    demux_dir = Path(demux_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bam_dir = output_dir / "bam"
    bam_dir.mkdir(parents=True, exist_ok=True)

    fasta = _materialise_genome_fasta(
        genome,
        output_dir / "genome.fa",
        output_dir / ".genome_meta.json",
    )
    # Build the index with the same preset the aligner uses below
    # (_GENOME_MODE_ARGS -> -ax splice) so minimap2's -k/-w match and it
    # doesn't fall back to a coarser default index at alignment time.
    mmi = minimap2_build_index(
        fasta, output_dir / "genome.mmi", preset="splice", threads=threads
    )

    minimap2_bin = _resolve_minimap2()
    samtools_bin = _resolve_samtools()
    if minimap2_args is None:
        args = _GENOME_MODE_ARGS + tuple(extra_minimap2_args)
    else:
        args = tuple(minimap2_args)
    final_bam = bam_dir / "aligned.bam"

    cmd_mm2 = [
        str(minimap2_bin),
        *args,
        "-t",
        str(int(threads)),
        str(mmi),
        "-",
    ]
    cmd_sort = [
        str(samtools_bin),
        "sort",
        "-@",
        str(int(threads)),
        "-o",
        str(final_bam),
    ]

    if progress_cb is not None:
        progress_cb(
            ProgressEvent(
                kind="stage_start",
                stage="align",
                message=(
                    f"streaming demux FASTQ → minimap2 → samtools sort → {final_bam}"
                ),
            )
        )

    writer_exc: list[BaseException] = []
    reads_emitted = [0]

    procs: list[tuple[subprocess.Popen, list[str]]] = []
    try:
        mm2 = subprocess.Popen(
            cmd_mm2, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        procs.append((mm2, cmd_mm2))
        sort = subprocess.Popen(cmd_sort, stdin=mm2.stdout)
        procs.append((sort, cmd_sort))
        # Let mm2 receive SIGPIPE if sort exits early
        if mm2.stdout is not None:
            mm2.stdout.close()

        def _writer() -> None:
            try:
                assert mm2.stdin is not None
                for batch in _iter_demux_read_batches(
                    demux_dir,
                    only_complete=only_complete,
                ):
                    chunk_bytes, n_reads = _format_fastq_bytes(batch)
                    if n_reads == 0:
                        continue
                    mm2.stdin.write(chunk_bytes)
                    reads_emitted[0] += n_reads
                    if progress_cb is not None:
                        progress_cb(
                            ProgressEvent(
                                kind="stage_progress",
                                stage="align",
                                completed=reads_emitted[0],
                                message=f"{reads_emitted[0]:,} reads streamed",
                            )
                        )
            except BrokenPipeError:
                # Downstream exited early — let the wait()s surface its rc
                pass
            except BaseException as e:
                writer_exc.append(e)
            finally:
                try:
                    if mm2.stdin is not None:
                        mm2.stdin.close()
                except OSError:
                    pass

        writer_thread = threading.Thread(target=_writer, daemon=True)
        writer_thread.start()

        # Wait downstream-first so SIGPIPE propagates upward cleanly
        sort_rc = sort.wait()
        mm2_rc = mm2.wait()
        writer_thread.join()
    finally:
        for proc, _ in procs:
            if proc.poll() is None:
                proc.kill()

    if writer_exc:
        raise writer_exc[0]
    if mm2_rc:
        raise subprocess.CalledProcessError(mm2_rc, cmd_mm2)
    if sort_rc:
        raise subprocess.CalledProcessError(sort_rc, cmd_sort)

    subprocess.run(
        [str(samtools_bin), "index", str(final_bam)],
        check=True,
    )

    if progress_cb is not None:
        progress_cb(
            ProgressEvent(
                kind="stage_done",
                stage="align",
                completed=reads_emitted[0],
                message=str(final_bam),
            )
        )
    return final_bam


def map_assembly(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Map one assembly against another (asm5 / asm10 / asm20).

    TODO: ship as a thin orchestrator over :func:`minimap2_run` mirroring
    :func:`map_to_genome`. Same use-case-specific-flags pattern.
    """
    raise NotImplementedError(
        "map_assembly: pending — orchestrator on top of minimap2_run "
        "with asm5/asm10/asm20 preset"
    )


__all__ = ["map_to_genome", "map_assembly"]
