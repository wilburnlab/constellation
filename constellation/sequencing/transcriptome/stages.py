"""Stage orchestration for the transcriptomics demux pipeline.

The pipeline DAG:

    plan_chunks       Compute per-worker file ranges (BAM record-boundary
                      virtual offsets via a fast no-decode scan; SAM
                      newline-aligned byte ranges via simple division)
    fused_demux       per-chunk worker:
                        ingest its assigned slice → READ_TABLE shard
                        + locate_segments → READ_SEGMENT_TABLE shard
                        + READ_DEMUX_TABLE shard (sample_id unresolved)
                        + predict_orfs → ORF_TABLE shard
    resolve           sequential aggregation:
                        concat shards
                        resolve_demux → READ_DEMUX_TABLE w/ sample_id
                        build_protein_count_matrix → quant + FASTA + TSV

Stages 1 and 3 run in the parent process (the chunk planner and the
small / sequential aggregation). Stage 2 fans out via
:func:`constellation.sequencing.parallel.run_batched`, with each worker
opening its own pysam handle on the source file (no parent-serial
ingest bottleneck) and writing its own ``part-NNNNN.parquet`` shards
under ``<output>/<key>/``.

This is the "NanoporeAnalysis-style" parallelism — every worker does
ingest + demux + write end-to-end, no shared parent reader. With 96
workers on a single node, the source file is read 96 ways in parallel
at the htslib layer (each worker gets its own BGZF decompression) AND
the demux work parallelises 96-fold. The architecture eliminates the
single-threaded ingest bottleneck that dominated the previous design.

Resume semantics. Each stage's directory carries a ``_SUCCESS`` marker
when it completes. With ``resume=True``, an already-complete stage is
skipped and its shards are reused; a partially-written stage (shards
present, no marker) re-runs only the missing shard indices. The
parent ``resolve`` stage writes its own marker on completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.parallel import (
    StageOutput,
    read_dataset,
    run_batched,
)
from constellation.sequencing.progress import (
    NullProgress,
    ProgressCallback,
    emit_done,
    emit_start,
)
from constellation.sequencing.samples import Samples
from constellation.sequencing.schemas.reads import READ_TABLE
from constellation.sequencing.transcriptome.demux import (
    locate_segments,
    resolve_demux,
)
from constellation.sequencing.transcriptome.designs import load_design
from constellation.sequencing.transcriptome.orf import ORF_TABLE, predict_orfs
from constellation.sequencing.transcriptome.quant import (
    PROTEIN_COUNT_TABLE,
    FastaRecord,
    build_protein_count_matrix,
    fasta_records_to_text,
)


# Output-key constants — used for stage subdirectory names and
# ``run_batched`` output keys. Centralised so callers don't carry
# magic strings.
KEY_READS = "reads"
KEY_READ_SEGMENTS = "read_segments"
KEY_READ_DEMUX = "read_demux"
KEY_ORFS = "orfs"
KEY_FEATURE_QUANT = "feature_quant"
KEY_PROTEINS_FASTA = "proteins.fasta"
KEY_PROTEIN_COUNTS_TSV = "protein_counts.tsv"


# ──────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────


def _detect_format(input_path: Path) -> str:
    """Return ``"bam"`` or ``"sam"`` from suffix; raise otherwise."""
    suffix = input_path.suffix.lower()
    if suffix == ".bam":
        return "bam"
    if suffix == ".sam":
        return "sam"
    raise ValueError(
        f"unsupported input suffix {suffix!r} (expected .sam or .bam) "
        f"for {input_path}"
    )


# ──────────────────────────────────────────────────────────────────────
# Fused per-chunk worker — ingest + demux + ORF in one process
# ──────────────────────────────────────────────────────────────────────


def _fused_chunk_worker(
    chunk_spec: tuple,
    *,
    input_path: str,
    fmt: str,
    library_design: str,
    acquisition_id: int,
    min_aa_length: int,
) -> dict[str, pa.Table]:
    """Process one file chunk end-to-end: ingest → demux → ORF.

    ``chunk_spec`` shape depends on ``fmt``:

      bam: ``(virtual_offset_start, n_primary_records)`` — the worker
           seeks to ``virtual_offset_start`` and reads exactly
           ``n_primary_records`` primary records.
      sam: ``(byte_start, byte_end)`` — the worker reads complete
           lines from ``byte_start`` to ``byte_end``.

    Each worker opens its own pysam / file handle, so ingest is fully
    parallel across workers. Output: 4 tables (reads + segments +
    demux + orfs); each is written as one shard by ``run_batched``.

    Top-level function (not a closure) so ``ProcessPoolExecutor`` can
    pickle it. The design is loaded by NAME inside the worker; the
    cdna_wilburn_v1 JSON load is cheap and module-level cached.
    """
    from constellation.sequencing.readers.sam_bam import (
        read_bam_chunk,
        read_sam_chunk,
    )

    path = Path(input_path)
    if fmt == "bam":
        vo_start, n_primary = chunk_spec
        rows = read_bam_chunk(
            path,
            vo_start=vo_start,
            n_primary=n_primary,
            acquisition_id=acquisition_id,
        )
    elif fmt == "sam":
        byte_start, byte_end = chunk_spec
        rows = read_sam_chunk(
            path,
            byte_start=byte_start,
            byte_end=byte_end,
            acquisition_id=acquisition_id,
        )
    else:
        raise ValueError(f"unknown format {fmt!r}")

    reads_table = (
        pa.Table.from_pylist(rows, schema=READ_TABLE)
        if rows
        else READ_TABLE.empty_table()
    )

    design = load_design(library_design)
    segments, demux, results = locate_segments(reads_table, design)
    orfs = predict_orfs(results, min_aa_length=min_aa_length)

    return {
        KEY_READS: reads_table,
        KEY_READ_SEGMENTS: segments,
        KEY_READ_DEMUX: demux,
        KEY_ORFS: orfs,
    }


# ──────────────────────────────────────────────────────────────────────
# Chunk planner — runs in the parent process before fan-out
# ──────────────────────────────────────────────────────────────────────


def _plan_chunks(
    input_path: Path,
    fmt: str,
    *,
    n_workers: int,
    chunk_size: int,
    progress_cb: ProgressCallback | None = None,
) -> list[tuple]:
    """Compute per-worker chunk specs for the source file.

    For BAM, walks the file once via :func:`_bam_record_chunks` to record
    record-boundary virtual offsets every ``chunk_size`` primary
    records. The scan is bare iteration (no field decoding) and runs
    with htslib's BGZF thread pool at ``threads=n_workers``, so it's
    fast — ~200k records/sec on typical hardware.

    For SAM, returns ``n_workers`` evenly-sized byte ranges aligned to
    newlines via the worker-side handshake protocol. No pre-scan.

    The two return shapes are consumed uniformly by
    :func:`_fused_chunk_worker` (which dispatches on ``fmt``).
    """
    from constellation.sequencing.readers.sam_bam import (
        _bam_record_chunks,
        _sam_byte_chunks,
    )

    emit_start(
        progress_cb,
        "plan",
        message=f"planning chunks for {input_path.name} (fmt={fmt})",
    )
    if fmt == "bam":
        chunks = _bam_record_chunks(
            input_path,
            chunk_size=chunk_size,
            threads=max(1, n_workers),
        )
    elif fmt == "sam":
        # For SAM we shoot for ~chunk_size records per worker; with no
        # record-count knowledge ahead of time we just split into
        # max(n_workers, file_size / target_chunk_bytes) byte ranges.
        # A SAM line averages ~5KB for nanopore; target chunk size in
        # bytes ≈ chunk_size * 5KB.
        target_bytes = chunk_size * 5 * 1024
        file_size = input_path.stat().st_size
        n = max(n_workers, max(1, file_size // target_bytes))
        chunks = _sam_byte_chunks(input_path, n_chunks=n)
    else:
        raise ValueError(f"unknown format {fmt!r}")

    emit_done(
        progress_cb,
        "plan",
        completed=len(chunks),
        total=len(chunks),
        message=f"{len(chunks)} chunk(s)",
    )
    return chunks


# ──────────────────────────────────────────────────────────────────────
# Fused stage — N workers, each does ingest + demux + ORF for one chunk
# ──────────────────────────────────────────────────────────────────────


def run_fused_demux_stage(
    input_path: Path,
    *,
    library_design: str,
    acquisition_id: int,
    output_dir: Path,
    n_workers: int = 1,
    chunk_size: int = 100_000,
    min_aa_length: int = 60,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> dict[str, StageOutput]:
    """Plan chunks + fan out per-chunk workers.

    Each worker writes 4 output shards (``reads``, ``read_segments``,
    ``read_demux``, ``orfs``) under their respective ``<output>/<key>/``
    directories. The reads dataset is partitioned by chunk, not a
    single parquet — query it via ``pa.dataset.dataset(reads/)``.
    """
    fmt = _detect_format(input_path)
    chunks = _plan_chunks(
        input_path,
        fmt,
        n_workers=n_workers,
        chunk_size=chunk_size,
        progress_cb=progress_cb,
    )

    return run_batched(
        worker_fn=_fused_chunk_worker,
        batches=chunks,
        output_dir=output_dir,
        output_keys=(KEY_READS, KEY_READ_SEGMENTS, KEY_READ_DEMUX, KEY_ORFS),
        n_workers=n_workers,
        worker_kwargs={
            "input_path": str(input_path),
            "fmt": fmt,
            "library_design": library_design,
            "acquisition_id": acquisition_id,
            "min_aa_length": min_aa_length,
        },
        progress_cb=progress_cb,
        resume=resume,
        stage_label="fused_demux",
        total=len(chunks),
    )


# ──────────────────────────────────────────────────────────────────────
# Resolve — sample assignment + count matrix (sequential)
# ──────────────────────────────────────────────────────────────────────


def resolve_and_quantify(
    demux_outputs: dict[str, StageOutput],
    *,
    samples: Samples,
    acquisition_id: int,
    output_dir: Path,
    min_protein_count: int,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> tuple[pa.Table, pa.Table, list[FastaRecord], str]:
    """Concatenate stage-2 shards, resolve sample assignments, and
    build the count matrix.

    Returns ``(demux_table, quant_table, fasta_records, tsv_text)`` —
    the writer below persists each. Sequential by design: the
    aggregation work is small (one row per read at S1) and benefits
    from a single in-memory pass.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success = output_dir / "_quant_success__"

    if resume and success.is_file() and (output_dir / "feature_quant.parquet").is_file():
        emit_start(
            progress_cb, "resolve", message="resume: reusing existing outputs"
        )
        demux_table = pq.read_table(output_dir / "read_demux.parquet")
        quant = pq.read_table(output_dir / "feature_quant.parquet")
        # FASTA / TSV are emitted-only; we don't need to round-trip
        # them on resume. Return empty stubs and let the caller
        # acknowledge resume.
        emit_done(progress_cb, "resolve", message="resumed")
        return demux_table, quant, [], ""

    emit_start(progress_cb, "resolve", message="aggregating shards")

    # Concatenate per-batch tables. order-stable because parallel.py
    # sorted shards by batch index.
    segments_table = read_dataset(demux_outputs[KEY_READ_SEGMENTS])
    demux_table = read_dataset(demux_outputs[KEY_READ_DEMUX])
    orf_table = read_dataset(demux_outputs[KEY_ORFS])

    demux_table = resolve_demux(
        demux_table,
        segments_table,
        samples=samples,
        acquisition_id=acquisition_id,
    )
    pq.write_table(demux_table, output_dir / "read_demux.parquet")

    quant, fasta_records, tsv_text = build_protein_count_matrix(
        demux_table, orf_table, samples, min_protein_count=min_protein_count
    )
    pq.write_table(quant, output_dir / "feature_quant.parquet")
    (output_dir / "proteins.fasta").write_text(
        fasta_records_to_text(fasta_records)
    )
    (output_dir / "protein_counts.tsv").write_text(tsv_text)

    success.touch()
    emit_done(
        progress_cb,
        "resolve",
        completed=quant.num_rows,
        total=quant.num_rows,
        message=f"{len(fasta_records)} proteins",
    )
    return demux_table, quant, fasta_records, tsv_text


# ──────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────────


def run_demux_pipeline(
    input_path: Path,
    *,
    library_design: str,
    samples: Samples,
    acquisition_id: int,
    output_dir: Path,
    chunk_size: int = 100_000,
    n_workers: int = 1,
    min_aa_length: int = 60,
    min_protein_count: int = 2,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
    # Backwards-compat alias for the old ``batch_size`` kwarg — same
    # semantic role (records-per-worker-chunk). Older callers pass it;
    # accept and forward to ``chunk_size`` if the new kwarg is left
    # at default.
    batch_size: int | None = None,
) -> dict[str, object]:
    """End-to-end demux pipeline: SAM/BAM → counts.

    Per-worker chunked: each of ``n_workers`` workers opens its own
    pysam handle on a slice of the source file, ingests its slice
    directly into READ_TABLE rows, runs demux + ORF prediction, and
    writes its own shards. There is no parent-serial ingest stage.

    With ``n_workers`` matching the available cores on a node (e.g. 96
    on a cluster node), ingest scales N-fold over the previous
    streaming-but-single-threaded design — the BGZF decompression
    happens N times in parallel inside N processes.

    Returns a dict of artefact tables / records / paths that the CLI
    handler renders into stdout / disk locations. ``n_reads`` is the
    primary-record count summed across worker shards (== demux table
    row count in S1, where every read produces exactly one demux row).
    """
    cb = progress_cb or NullProgress()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backwards-compat: callers passing the old ``batch_size`` get it
    # wired to the new kwarg name.
    if batch_size is not None and chunk_size == 100_000:
        chunk_size = batch_size

    demux_outputs = run_fused_demux_stage(
        input_path,
        library_design=library_design,
        acquisition_id=acquisition_id,
        output_dir=output_dir,
        n_workers=n_workers,
        chunk_size=chunk_size,
        min_aa_length=min_aa_length,
        progress_cb=cb,
        resume=resume,
    )

    demux_table, quant, fasta_records, tsv_text = resolve_and_quantify(
        demux_outputs,
        samples=samples,
        acquisition_id=acquisition_id,
        output_dir=output_dir,
        min_protein_count=min_protein_count,
        progress_cb=cb,
        resume=resume,
    )

    n_reads = demux_table.num_rows  # one row per primary read in S1

    return {
        "n_reads": n_reads,
        "demux_outputs": demux_outputs,
        "demux_table": demux_table,
        "quant_table": quant,
        "fasta_records": fasta_records,
        "tsv_text": tsv_text,
    }


__all__ = [
    "KEY_FEATURE_QUANT",
    "KEY_ORFS",
    "KEY_PROTEIN_COUNTS_TSV",
    "KEY_PROTEINS_FASTA",
    "KEY_READS",
    "KEY_READ_DEMUX",
    "KEY_READ_SEGMENTS",
    "resolve_and_quantify",
    "run_demux_pipeline",
    "run_fused_demux_stage",
]
