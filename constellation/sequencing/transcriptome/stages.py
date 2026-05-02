"""Stage orchestration for the transcriptomics demux pipeline.

The pipeline DAG:

    sam_ingest        SAM → READ_TABLE   (single table, written once)
    demux             per-batch worker:
                        locate_segments → READ_SEGMENT_TABLE
                        + READ_DEMUX_TABLE (sample_id unresolved)
                        + predict_orfs → ORF_TABLE
    resolve           sequential aggregation:
                        concat shards
                        resolve_demux → READ_DEMUX_TABLE w/ sample_id
                        build_protein_count_matrix → quant + FASTA + TSV

Stages 1 and 3 run in the parent process (single-pass disk I/O for
``sam_ingest``; small / sequential aggregation for ``resolve``).
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
    emit_start,
)
from constellation.sequencing.samples import Samples
from constellation.sequencing.schemas.reads import READ_TABLE
from constellation.sequencing.transcriptome.demux import (
    locate_segments,
    resolve_demux,
)
from constellation.sequencing.transcriptome.orf import ORF_TABLE, predict_orfs
from constellation.sequencing.transcriptome.panels import load_panel
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
    construct_name: str,
) -> dict[str, pa.Table]:
    """Run demux + ORF prediction on one batch of reads.

    Top-level function (not a closure) so ``ProcessPoolExecutor`` can
    pickle it. Loads the construct by name inside the worker so we
    don't pay for cross-process pickling of the panel object — the
    JSON load is cheap and cached at the module level after first
    call.

    The output ``read_demux`` table has ``sample_id`` left null;
    sample assignment runs in the parent's ``resolve`` stage where
    the :class:`Samples` container is available.
    """
    construct = load_panel(construct_name)
    segments, demux, results = locate_segments(batch, construct)
    orfs = predict_orfs(results)
    return {
        KEY_READ_SEGMENTS: segments,
        KEY_READ_DEMUX: demux,
        KEY_ORFS: orfs,
    }


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — SAM ingest
# ──────────────────────────────────────────────────────────────────────


def _read_table_to_batches(
    table: pa.Table, batch_size: int
) -> Iterator[pa.Table]:
    """Slice an Arrow table into row-batches of at most ``batch_size``
    rows each."""
    n = table.num_rows
    for start in range(0, n, batch_size):
        yield table.slice(start, batch_size)


def ingest_sam(
    sam_path: Path,
    *,
    acquisition_id: int,
    output_dir: Path,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> pa.Table:
    """Ingest a SAM file into ``output_dir/reads.parquet`` and return
    the resulting :class:`READ_TABLE`-shaped :class:`pa.Table`.

    Single file, single writer — no partitioned dataset (``READ_TABLE``
    is small enough to keep as one parquet). Resume: if the output
    file exists *and* a sentinel ``reads.parquet._SUCCESS`` marker
    sits next to it, the existing parquet is reused.
    """
    from constellation.sequencing.readers.sam_bam import SamReader

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{KEY_READS}.parquet"
    success = target.with_suffix(target.suffix + ".__success__")

    if resume and target.is_file() and success.is_file():
        emit_start(
            progress_cb, KEY_READS, message=f"resume: reusing {target}"
        )
        table = pq.read_table(target)
        emit_done(
            progress_cb,
            KEY_READS,
            completed=table.num_rows,
            total=table.num_rows,
            message="resumed",
        )
        return table

    emit_start(progress_cb, KEY_READS, message=f"reading {sam_path}")
    table = SamReader().read(sam_path, acquisition_id=acquisition_id).primary
    pq.write_table(table, target)
    success.touch()
    emit_done(
        progress_cb,
        KEY_READS,
        completed=table.num_rows,
        total=table.num_rows,
        message=f"{table.num_rows} reads",
    )
    return table


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — demux + ORF (parallelizable)
# ──────────────────────────────────────────────────────────────────────


def run_demux_stage(
    reads: pa.Table,
    *,
    construct_name: str,
    output_dir: Path,
    batch_size: int,
    n_workers: int = 1,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> dict[str, StageOutput]:
    """Fan out demux + ORF over batches of reads.

    ``reads`` is split into row-batches of ``batch_size`` each (the
    final batch may be shorter). Each batch becomes one
    ``part-NNNN.parquet`` shard per output key under ``output_dir``.
    """
    if reads.schema != READ_TABLE:
        # Cast or align; tolerant of metadata-stamped variants.
        reads = reads.cast(READ_TABLE)

    batches = list(_read_table_to_batches(reads, batch_size))
    return run_batched(
        worker_fn=_demux_worker,
        batches=batches,
        output_dir=output_dir,
        output_keys=(KEY_READ_SEGMENTS, KEY_READ_DEMUX, KEY_ORFS),
        n_workers=n_workers,
        worker_kwargs={"construct_name": construct_name},
        progress_cb=progress_cb,
        resume=resume,
        stage_label="demux",
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
    sam_path: Path,
    *,
    construct_name: str,
    samples: Samples,
    acquisition_id: int,
    output_dir: Path,
    batch_size: int = 100_000,
    n_workers: int = 1,
    min_protein_count: int = 2,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """End-to-end demux pipeline: SAM → counts.

    Returns a dict of artefact tables / records / paths that the CLI
    handler renders into stdout / disk locations.
    """
    cb = progress_cb or NullProgress()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reads = ingest_sam(
        sam_path,
        acquisition_id=acquisition_id,
        output_dir=output_dir,
        progress_cb=cb,
        resume=resume,
    )

    demux_outputs = run_demux_stage(
        reads,
        construct_name=construct_name,
        output_dir=output_dir,
        batch_size=batch_size,
        n_workers=n_workers,
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
        "reads": reads,
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
    "ingest_sam",
    "resolve_and_quantify",
    "run_demux_pipeline",
    "run_demux_stage",
]
