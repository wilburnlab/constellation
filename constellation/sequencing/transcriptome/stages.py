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
    PARTIAL_QUANT_TABLE,
    PROTEIN_COUNT_TABLE,
    FastaRecord,
    aggregate_partial_quants,
    build_partial_quant,
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
    library_design: str,
    samples: Samples,
    min_aa_length: int,
) -> dict[str, pa.Table]:
    """Process one file chunk end-to-end: ingest → demux → ORF →
    sample-resolve → partial quant.

    ``chunk_spec`` carries everything needed to locate the worker's
    slice — the input file path, format, acquisition_id, and the
    range data:

      ``(file_path, fmt, acquisition_id, range_data)``

    where ``range_data`` is ``(vo_start, n_primary)`` for BAM or
    ``(byte_start, byte_end)`` for SAM. The acquisition_id rides
    inside the chunk_spec (rather than worker_kwargs) because
    multi-file inputs may have different acquisitions per chunk.

    Each worker opens its own pysam / file handle (no shared state
    with other workers or the parent). Output: 5 tables that become
    5 shards via :func:`run_batched`:

      - reads          (READ_TABLE — full per-read fields)
      - read_segments  (READ_SEGMENT_TABLE — one row per located segment)
      - read_demux     (READ_DEMUX_TABLE — sample_id resolved here)
      - orfs           (ORF_TABLE — predicted proteins per read)
      - feature_quant  (PARTIAL_QUANT_TABLE — pre-aggregated counts)

    Sample resolution + partial quant aggregation happen INSIDE the
    worker on its own small tables, so the resolve stage only has to
    sum tiny per-worker count contributions — no read-level cross-
    shard joins.

    Top-level function (not a closure) so ``ProcessPoolExecutor`` can
    pickle it. The design is loaded by NAME inside the worker; the
    cdna_wilburn_v1 JSON load is cheap and module-level cached.
    """
    from constellation.sequencing.readers.sam_bam import (
        read_bam_chunk,
        read_sam_chunk,
    )

    file_path, fmt, acquisition_id, range_data = chunk_spec
    path = Path(file_path)
    if fmt == "bam":
        vo_start, n_primary = range_data
        rows = read_bam_chunk(
            path,
            vo_start=vo_start,
            n_primary=n_primary,
            acquisition_id=acquisition_id,
        )
    elif fmt == "sam":
        byte_start, byte_end = range_data
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
    # Resolve sample_id locally — the worker's segments table tells it
    # each read's barcode_id, and the (small, picklable) Samples
    # container resolves (acquisition, barcode) → sample_id without
    # crossing process boundaries again.
    demux = resolve_demux(
        demux,
        segments,
        samples=samples,
        acquisition_id=acquisition_id,
    )
    # Pre-aggregate (protein_sequence, sample_id) counts for THIS
    # worker's reads. Bounded by unique-pairs in the chunk — usually
    # much smaller than the read count.
    partial_quant = build_partial_quant(demux, orfs)

    return {
        KEY_READS: reads_table,
        KEY_READ_SEGMENTS: segments,
        KEY_READ_DEMUX: demux,
        KEY_ORFS: orfs,
        KEY_FEATURE_QUANT: partial_quant,
    }


# ──────────────────────────────────────────────────────────────────────
# Chunk planner — runs in the parent process before fan-out
# ──────────────────────────────────────────────────────────────────────


def _plan_chunks_for_file(
    input_path: Path,
    fmt: str,
    *,
    acquisition_id: int,
    chunk_size: int,
    threads: int,
) -> list[tuple]:
    """Compute per-worker chunk specs for ONE source file.

    For BAM, walks the file once via :func:`_bam_record_chunks` to record
    record-boundary virtual offsets every ``chunk_size`` primary
    records. The scan is bare iteration (no field decoding) and runs
    with htslib's BGZF thread pool at ``threads``, so it's
    fast — ~200k records/sec on typical hardware.

    For SAM, returns evenly-sized byte ranges aligned to newlines via
    the worker-side handshake protocol — no pre-scan needed.

    Returns chunk specs of shape
    ``(str(file_path), fmt, acquisition_id, range_data)`` where
    ``range_data`` is ``(vo_start, n_primary)`` for BAM or
    ``(byte_start, byte_end)`` for SAM. The acquisition_id is baked
    into each spec so multi-file inputs can pool their chunks into a
    single ``run_batched`` invocation while keeping per-file
    provenance straight.
    """
    from constellation.sequencing.readers.sam_bam import (
        _bam_record_chunks,
        _sam_byte_chunks,
    )

    if fmt == "bam":
        ranges = _bam_record_chunks(
            input_path,
            chunk_size=chunk_size,
            threads=max(1, threads),
        )
    elif fmt == "sam":
        # For SAM we shoot for ~chunk_size records per worker; with no
        # record-count knowledge ahead of time we just split into
        # max(threads, file_size / target_chunk_bytes) byte ranges.
        # A SAM line averages ~5KB for nanopore; target chunk size in
        # bytes ≈ chunk_size * 5KB.
        target_bytes = chunk_size * 5 * 1024
        file_size = input_path.stat().st_size
        n = max(max(1, threads), max(1, file_size // target_bytes))
        ranges = _sam_byte_chunks(input_path, n_chunks=n)
    else:
        raise ValueError(f"unknown format {fmt!r}")

    return [
        (str(input_path), fmt, acquisition_id, range_data)
        for range_data in ranges
    ]


def _plan_chunks(
    inputs: list[tuple[Path, int]],
    *,
    n_workers: int,
    chunk_size: int,
    progress_cb: ProgressCallback | None = None,
) -> list[tuple]:
    """Plan chunk specs across one or more input files.

    ``inputs`` is a list of ``(input_path, acquisition_id)`` pairs. The
    planner runs per-file (each file's chunk planner is independent)
    and concatenates the results. Workers receive the pooled list and
    are dispatched in chunk-spec order across the worker pool — so
    the bigger files get more chunks but the work itself fans out
    evenly across all available workers regardless of which file the
    chunk belongs to.
    """
    emit_start(
        progress_cb,
        "plan",
        message=f"planning chunks for {len(inputs)} input file(s)",
    )
    all_chunks: list[tuple] = []
    for path, acq_id in inputs:
        fmt = _detect_format(path)
        # Per-file thread budget: at most n_workers BGZF threads while
        # planning — same as the demux stage will use.
        per_file_chunks = _plan_chunks_for_file(
            path,
            fmt,
            acquisition_id=acq_id,
            chunk_size=chunk_size,
            threads=n_workers,
        )
        all_chunks.extend(per_file_chunks)

    emit_done(
        progress_cb,
        "plan",
        completed=len(all_chunks),
        total=len(all_chunks),
        message=f"{len(all_chunks)} chunk(s) across {len(inputs)} file(s)",
    )
    return all_chunks


# ──────────────────────────────────────────────────────────────────────
# Fused stage — N workers, each does ingest + demux + ORF for one chunk
# ──────────────────────────────────────────────────────────────────────


def run_fused_demux_stage(
    inputs: list[tuple[Path, int]],
    *,
    library_design: str,
    samples: Samples,
    output_dir: Path,
    n_workers: int = 1,
    chunk_size: int = 100_000,
    min_aa_length: int = 60,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> dict[str, StageOutput]:
    """Plan chunks + fan out per-chunk workers across one or more files.

    Each worker writes 5 shards under ``<output>/<key>/``:
    ``reads``, ``read_segments``, ``read_demux``, ``orfs``,
    ``feature_quant`` (partial — pre-aggregated per-worker counts).
    All five datasets are partitioned by chunk; downstream consumers
    open them as ``pa.dataset.dataset(...)`` directly — no single-file
    aggregation between stages.
    """
    chunks = _plan_chunks(
        inputs,
        n_workers=n_workers,
        chunk_size=chunk_size,
        progress_cb=progress_cb,
    )

    return run_batched(
        worker_fn=_fused_chunk_worker,
        batches=chunks,
        output_dir=output_dir,
        output_keys=(
            KEY_READS,
            KEY_READ_SEGMENTS,
            KEY_READ_DEMUX,
            KEY_ORFS,
            KEY_FEATURE_QUANT,
        ),
        n_workers=n_workers,
        worker_kwargs={
            "library_design": library_design,
            "samples": samples,
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
    output_dir: Path,
    *,
    samples: Samples,
    min_protein_count: int,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> tuple[pa.Table, list[FastaRecord], str]:
    """Aggregate per-worker partial counts into the final count matrix.

    No single-file `read_demux.parquet` write — `read_demux/` stays a
    partitioned dataset (consumers open it as
    ``pa.dataset.dataset(read_demux/)`` and stream-iterate / filter
    via Arrow's compute kernels). The only single-file outputs are
    the human-facing terminal artifacts:

      - ``feature_quant.parquet`` (small — bounded by unique proteins
        × samples)
      - ``proteins.fasta``
      - ``protein_counts.tsv``

    The actual aggregation is a single Arrow ``group_by`` + ``sum``
    over the ``feature_quant/`` partitioned dataset, whose row count
    is bounded by per-worker unique-pair counts (small at any scale).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success = output_dir / "_quant_success__"

    if resume and success.is_file() and (output_dir / "feature_quant.parquet").is_file():
        emit_start(
            progress_cb, "resolve", message="resume: reusing existing outputs"
        )
        quant = pq.read_table(output_dir / "feature_quant.parquet")
        # FASTA / TSV are emitted-only; we don't need to round-trip
        # them on resume. Return empty stubs and let the caller
        # acknowledge resume.
        emit_done(progress_cb, "resolve", message="resumed")
        return quant, [], ""

    emit_start(
        progress_cb,
        "resolve",
        message="aggregating partial-quant shards",
    )

    # Stream over the partitioned feature_quant dataset; the
    # aggregator handles group_by + sum + ordering + labelling
    # internally.
    quant, fasta_records, tsv_text = aggregate_partial_quants(
        output_dir / KEY_FEATURE_QUANT,
        samples,
        min_protein_count=min_protein_count,
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
    return quant, fasta_records, tsv_text


# ──────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────────


def run_demux_pipeline(
    input_path: Path | list[tuple[Path, int]],
    *,
    library_design: str,
    samples: Samples,
    output_dir: Path,
    acquisition_id: int = 1,
    chunk_size: int = 100_000,
    n_workers: int = 1,
    min_aa_length: int = 60,
    min_protein_count: int = 2,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
    # Backwards-compat alias for the old ``batch_size`` kwarg.
    batch_size: int | None = None,
) -> dict[str, object]:
    """End-to-end demux pipeline: SAM/BAM(s) → counts.

    ``input_path`` can be a single ``Path`` (single-acquisition mode —
    the legacy shape; uses the ``acquisition_id`` kwarg for the
    stamped acquisition_id) or a list of ``(path, acquisition_id)``
    pairs (multi-acquisition mode — used when running multiple flow
    cells / re-runs through one pipeline invocation).

    Per-worker chunked: each chunk worker opens its own pysam handle
    on a slice of one source file, ingests its slice directly into
    READ_TABLE rows, runs demux + ORF prediction, resolves sample_id
    locally against the (small, picklable) Samples object, and writes
    its 5 shards (reads, read_segments, read_demux, orfs,
    feature_quant). There is no parent-serial ingest stage and no
    cross-shard read-level join in the resolver.

    Returns a dict with ``n_reads`` (sum of records ingested across
    all input files), ``demux_outputs`` (per-key StageOutput), the
    final ``quant_table`` + ``fasta_records`` + ``tsv_text`` for the
    CLI manifest. ``read_demux/`` is a partitioned dataset on disk —
    consumers open it via ``pa.dataset.dataset(...)`` directly.
    """
    cb = progress_cb or NullProgress()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backwards-compat: callers passing the old ``batch_size`` get it
    # wired to the new kwarg name.
    if batch_size is not None and chunk_size == 100_000:
        chunk_size = batch_size

    # Normalise input: single Path → list of one (path, acquisition_id).
    if isinstance(input_path, (str, Path)):
        inputs = [(Path(input_path), acquisition_id)]
    else:
        inputs = [(Path(p), aid) for p, aid in input_path]

    demux_outputs = run_fused_demux_stage(
        inputs,
        library_design=library_design,
        samples=samples,
        output_dir=output_dir,
        n_workers=n_workers,
        chunk_size=chunk_size,
        min_aa_length=min_aa_length,
        progress_cb=cb,
        resume=resume,
    )

    quant, fasta_records, tsv_text = resolve_and_quantify(
        output_dir,
        samples=samples,
        min_protein_count=min_protein_count,
        progress_cb=cb,
        resume=resume,
    )

    # n_reads from the partitioned read_demux dataset metadata — each
    # shard contributes its row count, summed without materialising.
    import pyarrow.dataset as ds

    demux_dir = output_dir / KEY_READ_DEMUX
    n_reads = (
        ds.dataset(str(demux_dir), format="parquet").count_rows()
        if demux_dir.is_dir()
        else 0
    )

    return {
        "n_reads": n_reads,
        "demux_outputs": demux_outputs,
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
