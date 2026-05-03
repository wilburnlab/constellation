"""Stage orchestration for the transcriptomics demux pipeline.

The pipeline DAG:

    ingest_reads      SAM/BAM → READ_TABLE   (single table, written once)
    demux             per-batch worker:
                        locate_segments → READ_SEGMENT_TABLE
                        + READ_DEMUX_TABLE (sample_id unresolved)
                        + predict_orfs → ORF_TABLE
    resolve           sequential aggregation:
                        concat shards
                        resolve_demux → READ_DEMUX_TABLE w/ sample_id
                        build_protein_count_matrix → quant + FASTA + TSV

Stages 1 and 3 run in the parent process (single-pass disk I/O for
``ingest_reads``; small / sequential aggregation for ``resolve``).
Stage 2 fans out via :func:`constellation.sequencing.parallel.run_batched`,
so per-batch work scales with ``n_workers`` while writing
deterministic ``part-NNNN.parquet`` shards under ``<output>/<key>/``.

Resume semantics. Each stage's directory carries a ``_SUCCESS`` marker
when it completes. With ``resume=True``, an already-complete stage is
skipped and its shards are reused; a partially-written stage (shards
present, no marker) re-runs only the missing shard indices. The
parent stages (``sam_ingest``, ``resolve``) write their own markers
on completion.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Iterable

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
    emit_progress,
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
# Stage 2 worker — runs in subprocesses
# ──────────────────────────────────────────────────────────────────────


def _demux_worker(
    batch: pa.Table,
    *,
    library_design: str,
    min_aa_length: int,
) -> dict[str, pa.Table]:
    """Run demux + ORF prediction on one batch of reads.

    Top-level function (not a closure) so ``ProcessPoolExecutor`` can
    pickle it. Loads the design by name inside the worker so we
    don't pay for cross-process pickling of the design object — the
    JSON load is cheap and cached at the module level after first
    call.

    The output ``read_demux`` table has ``sample_id`` left null;
    sample assignment runs in the parent's ``resolve`` stage where
    the :class:`Samples` container is available.
    """
    design = load_design(library_design)
    segments, demux, results = locate_segments(batch, design)
    orfs = predict_orfs(results, min_aa_length=min_aa_length)
    return {
        KEY_READ_SEGMENTS: segments,
        KEY_READ_DEMUX: demux,
        KEY_ORFS: orfs,
    }


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — SAM ingest
# ──────────────────────────────────────────────────────────────────────


def _iter_reads_batches(
    reads_path: Path, batch_size: int
) -> Iterator[pa.Table]:
    """Lazily iterate over a reads.parquet in row-batch slices.

    Each yielded chunk is a :class:`pa.Table` of at most ``batch_size``
    rows. RAM peak per iteration step = one batch (rows held by the
    parquet reader's buffer), regardless of file size. ``ParquetFile``'s
    streaming reader allocates per-row-group buffers lazily.

    NOTE: ``batch_size`` controls the *yielded* chunk size; row groups
    on disk may be larger or smaller. ``ParquetFile.iter_batches``
    re-batches across row-group boundaries.
    """
    pf = pq.ParquetFile(reads_path)
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    for rb in pf.iter_batches(batch_size=batch_size):
        pending.append(rb)
        pending_rows += rb.num_rows
        if pending_rows >= batch_size:
            yield pa.Table.from_batches(pending)
            pending = []
            pending_rows = 0
    if pending:
        yield pa.Table.from_batches(pending)


def ingest_reads(
    input_path: Path,
    *,
    acquisition_id: int,
    output_dir: Path,
    batch_size: int = 100_000,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> tuple[Path, int]:
    """Stream-ingest a SAM/BAM into ``output_dir/reads.parquet``.

    Reader selection is suffix-dispatched via
    :func:`core.io.readers.find_reader` with ``modality="nanopore"`` —
    so ``.sam`` resolves to :class:`SamReader` (pure-Python text scan)
    and ``.bam`` resolves to :class:`BamReader` (pysam-backed). Both
    yield the same :class:`READ_TABLE` shape; downstream stages don't
    care which file format the bytes came from.

    The reader's :meth:`iter_batches` yields ``batch_size``-row chunks
    one at a time; we feed each chunk into a :class:`pa.parquet.ParquetWriter`
    so peak RAM = one batch (~hundreds of MB at batch_size=100k for
    nanopore reads), independent of total file size. Each ``iter_batches``
    chunk becomes one parquet row group, which makes downstream
    streaming consumers (``ParquetFile.iter_batches``) see the same
    natural batch shape on the disk side too.

    Resume: if ``reads.parquet`` exists *and* a sentinel
    ``reads.parquet.__success__`` marker sits next to it, the existing
    parquet is reused (file is not re-read).

    Returns ``(reads_path, n_reads)``. The full reads table is *not*
    returned — downstream stages stream from the parquet path.
    """
    from constellation.core.io.readers import find_reader

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{KEY_READS}.parquet"
    success = target.with_suffix(target.suffix + ".__success__")

    if resume and target.is_file() and success.is_file():
        emit_start(
            progress_cb, KEY_READS, message=f"resume: reusing {target}"
        )
        n_reads = pq.ParquetFile(target).metadata.num_rows
        emit_done(
            progress_cb,
            KEY_READS,
            completed=n_reads,
            total=n_reads,
            message=f"resumed ({n_reads} reads)",
        )
        return target, n_reads

    emit_start(progress_cb, KEY_READS, message=f"streaming {input_path}")
    reader = find_reader(input_path, modality="nanopore")

    n_reads = 0
    writer: pq.ParquetWriter | None = None
    try:
        for batch in reader.iter_batches(
            input_path, batch_size, acquisition_id=acquisition_id
        ):
            if writer is None:
                # Open the writer with the first batch's schema (which
                # equals READ_TABLE for both SAM and BAM readers).
                writer = pq.ParquetWriter(target, batch.schema)
            writer.write_table(batch)
            n_reads += batch.num_rows
            emit_progress(
                progress_cb,
                KEY_READS,
                completed=n_reads,
                total=None,
                message=f"{n_reads} reads ingested",
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # Empty input — write an empty parquet so downstream stages
        # have a real file to open.
        from constellation.sequencing.schemas.reads import READ_TABLE

        pq.write_table(READ_TABLE.empty_table(), target)

    success.touch()
    emit_done(
        progress_cb,
        KEY_READS,
        completed=n_reads,
        total=n_reads,
        message=f"{n_reads} reads",
    )
    return target, n_reads


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — demux + ORF (parallelizable)
# ──────────────────────────────────────────────────────────────────────


def run_demux_stage(
    reads_path: Path,
    *,
    library_design: str,
    output_dir: Path,
    batch_size: int,
    n_workers: int = 1,
    min_aa_length: int = 60,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> dict[str, StageOutput]:
    """Fan out demux + ORF over batches of reads, streaming.

    Reads are pulled from ``reads_path`` in row-batches of ``batch_size``
    rows via :func:`_iter_reads_batches`; each batch is dispatched to
    a worker as it's yielded. Peak parent RAM = one batch (plus the
    bounded in-flight queue inside ``run_batched``), regardless of
    total read count. Each batch becomes one ``part-NNNN.parquet``
    shard per output key.

    The expected total batch count is computed from parquet metadata
    (``num_rows / batch_size``, ceiling) for progress reporting.
    """
    pf = pq.ParquetFile(reads_path)
    n_rows = pf.metadata.num_rows
    total_batches = max(1, (n_rows + batch_size - 1) // batch_size) if n_rows else 0

    return run_batched(
        worker_fn=_demux_worker,
        batches=_iter_reads_batches(reads_path, batch_size),
        output_dir=output_dir,
        output_keys=(KEY_READ_SEGMENTS, KEY_READ_DEMUX, KEY_ORFS),
        n_workers=n_workers,
        worker_kwargs={
            "library_design": library_design,
            "min_aa_length": min_aa_length,
        },
        progress_cb=progress_cb,
        resume=resume,
        stage_label="demux",
        total=total_batches,
    )


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — sample resolution + count matrix (sequential)
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
    aggregation work is small (340 rows on test_simplex.sam) and
    benefits from a single in-memory pass.
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
    batch_size: int = 100_000,
    n_workers: int = 1,
    min_aa_length: int = 60,
    min_protein_count: int = 2,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """End-to-end demux pipeline: SAM/BAM → counts.

    Streaming end-to-end: ingest → demux → resolve. Peak parent RAM is
    bounded by ``batch_size`` rows (plus the resolve stage's
    aggregation, which is small in row count).

    Returns a dict of artefact tables / records / paths that the CLI
    handler renders into stdout / disk locations. The ``reads`` slot
    holds the ingested parquet path + row count rather than the full
    Arrow table — callers that need the table should
    ``pq.read_table(art["reads_path"])`` themselves.
    """
    cb = progress_cb or NullProgress()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reads_path, n_reads = ingest_reads(
        input_path,
        acquisition_id=acquisition_id,
        output_dir=output_dir,
        batch_size=batch_size,
        progress_cb=cb,
        resume=resume,
    )

    demux_outputs = run_demux_stage(
        reads_path,
        library_design=library_design,
        output_dir=output_dir,
        batch_size=batch_size,
        n_workers=n_workers,
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

    return {
        "reads_path": reads_path,
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
    "ingest_reads",
    "resolve_and_quantify",
    "run_demux_pipeline",
    "run_demux_stage",
]
