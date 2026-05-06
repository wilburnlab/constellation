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

import json
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds

from constellation.sequencing.align.minimap2 import minimap2_build_index
from constellation.sequencing.progress import ProgressCallback, ProgressEvent
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


def _materialise_genome_fasta(
    genome: GenomeReference,
    fasta_path: Path,
    meta_path: Path,
) -> Path:
    """Write a GenomeReference's contigs to FASTA, caching by contig count.

    Skips rewrite if ``fasta_path`` exists and ``meta_path`` records a
    matching contig count. Coarse cache key — adequate for the lab's
    "import once, align many" workflow; if a future caller mutates a
    GenomeReference between align invocations, drop the meta file.
    """
    fasta_path = Path(fasta_path)
    meta_path = Path(meta_path)
    expected_n = int(genome.contigs.num_rows)
    if fasta_path.exists() and meta_path.exists():
        try:
            stamp = json.loads(meta_path.read_text())
            if int(stamp.get("n_contigs", -1)) == expected_n:
                return fasta_path
        except (OSError, json.JSONDecodeError):
            pass

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    contig_ids = genome.contigs.column("contig_id").to_pylist()
    names = genome.contigs.column("name").to_pylist()
    with fasta_path.open("w", encoding="utf-8") as fh:
        for contig_id, name in zip(contig_ids, names):
            fh.write(f">{name}\n")
            fh.write(genome.sequence_of(int(contig_id)))
            fh.write("\n")
    meta_path.write_text(
        json.dumps({"n_contigs": expected_n, "fasta": fasta_path.name}, indent=2)
    )
    return fasta_path


def _stream_demux_fastq(
    demux_dir: Path,
    *,
    only_complete: bool = True,
    batch_size: int = 100_000,
) -> Iterator[tuple[bytes, int]]:
    """Stream FASTQ bytes from an S1 demux output directory.

    Joins the ``reads/`` partition (read_id, sequence, quality) against
    the filtered ``read_demux/`` partition (read_id, transcript_start,
    transcript_end), slices each sequence to its transcript window, and
    yields per-batch FASTQ-encoded bytes.

    Filter when ``only_complete=True`` (default for genome-guided
    quantification — see the chimera-handling discussion in the S2 plan):

        * ``status == 'Complete'``
        * ``sample_id`` not null
        * ``is_fragment == False``
        * ``is_chimera == False``  (forward-compat; S1 always emits False
          today, lights up when S3 enables multi-segment splitting)
        * ``transcript_start >= 0`` AND ``transcript_end > transcript_start``

    Yields ``(fastq_bytes, n_reads)`` per batch so callers can update
    progress without re-decoding the bytes.

    Memory at 200M-read scale: filtered demux index ~10 GB (50 B/row);
    each yielded FASTQ batch ~200 MB (100k rows × ~2 KB seq+qual). The
    demux index sits resident as the join's hash-build side; the reads
    side streams via ``pa.dataset.to_batches``.
    """
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

    demux_table = pa_ds.dataset(demux_part_dir).to_table(
        columns=["read_id", "transcript_start", "transcript_end"],
        filter=filt,
    )
    if demux_table.num_rows == 0:
        return

    reads_ds = pa_ds.dataset(reads_dir)
    has_quality = "quality" in set(reads_ds.schema.names)
    columns = ["read_id", "sequence"] + (["quality"] if has_quality else [])

    for batch in reads_ds.to_batches(columns=columns, batch_size=batch_size):
        if batch.num_rows == 0:
            continue
        reads_table = pa.Table.from_batches([batch])
        joined = reads_table.join(
            demux_table, keys="read_id", join_type="inner"
        )
        if joined.num_rows == 0:
            continue

        read_ids = joined.column("read_id").to_pylist()
        sequences = joined.column("sequence").to_pylist()
        if has_quality and "quality" in joined.column_names:
            qualities = joined.column("quality").to_pylist()
        else:
            qualities = [None] * joined.num_rows
        ts_list = joined.column("transcript_start").to_pylist()
        te_list = joined.column("transcript_end").to_pylist()

        parts: list[str] = []
        for read_id, seq, qual, ts, te in zip(
            read_ids, sequences, qualities, ts_list, te_list
        ):
            s = seq[ts:te]
            if not s:
                continue
            if qual is not None and len(qual) == len(seq):
                q = qual[ts:te]
            else:
                # No quality (FASTA-shaped reads) → synthetic Q40 = 'I'
                q = "I" * len(s)
            parts.append(f"@{read_id}\n{s}\n+\n{q}\n")
        if parts:
            yield "".join(parts).encode("ascii"), len(parts)


def map_to_genome(
    demux_dir: Path,
    genome: GenomeReference,
    *,
    output_dir: Path,
    threads: int = 8,
    only_complete: bool = True,
    batch_size: int = 100_000,
    extra_minimap2_args: tuple[str, ...] = (),
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Splice-aware full-length cDNA → genome alignment via minimap2.

    Streams trimmed transcript-window FASTQ from the S1 demux output
    directory directly into minimap2's stdin (no intermediate FASTQ
    file, no raw-BAM input) — the S1 demux already located the library
    scaffold (5'/3' adapter, polyA, barcode), so we map only the
    transcript window of each Complete-status read. See
    :func:`_stream_demux_fastq` for the filter set.

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
    mmi = minimap2_build_index(
        fasta, output_dir / "genome.mmi", threads=threads
    )

    minimap2_bin = _resolve_minimap2()
    samtools_bin = _resolve_samtools()
    args = _GENOME_MODE_ARGS + tuple(extra_minimap2_args)
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
                for chunk_bytes, n_reads in _stream_demux_fastq(
                    demux_dir,
                    only_complete=only_complete,
                    batch_size=batch_size,
                ):
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
