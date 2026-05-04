"""Gene-level count function + fused per-chunk worker for the pipeline.

Two surfaces:

    count_reads_per_gene(gene_assignments, read_demux, samples)
        Hash-join + group_by aggregator. ``gene_assignments`` and
        ``read_demux`` are pa.Tables — the caller materialises from
        partitioned datasets (via ``.to_table()``) before invocation.
        Used both by tests/Jupyter and by the CLI handler's resolve
        stage.

    fused_decode_filter_overlap_worker(chunk_spec, ...)
        Per-chunk pickleable worker function. Decodes its BAM chunk
        via ``read_bam_alignments_chunk``, applies the filter
        predicates + gene-overlap join, returns shards for the three
        partitioned-dataset outputs (alignments, alignment_tags,
        gene_assignments) plus a stats dict. Mirrors S1's
        ``transcriptome.stages._fused_chunk_worker`` placement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.quant._kernels import (
    apply_filter_predicates,
    compute_gene_overlap,
)
from constellation.sequencing.readers.sam_bam import read_bam_alignments_chunk
from constellation.sequencing.samples import Samples
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_TABLE,
    ALIGNMENT_TAG_TABLE,
)
from constellation.sequencing.schemas.quant import FEATURE_QUANT


def count_reads_per_gene(
    gene_assignments: pa.Table,
    read_demux: pa.Table,
    samples: Samples,
    *,
    engine: str = "constellation_overlap",
) -> tuple[pa.Table, dict[str, int]]:
    """Hash-join ``gene_assignments`` × ``read_demux`` on ``read_id``,
    group_by-sum to FEATURE_QUANT.

    Both inputs are pa.Tables — caller materialises partitioned
    datasets via ``.to_table()`` before invocation. Memory budget at
    200M-read scale: ~3 GB for ``gene_assignments`` + ~10 GB for
    ``read_demux``; fits comfortably on workstation RAM, comfortable
    on cluster nodes.

    ``samples`` is consumed for FK-validation only — the FEATURE_QUANT
    output is keyed on ``sample_id`` integers.
    """
    stats: dict[str, int] = {
        "gene_assignments_in": int(gene_assignments.num_rows),
        "reads_with_sample": 0,
        "reads_without_sample": 0,
        "unique_(gene,sample)_pairs": 0,
        "total_count": 0,
    }
    if gene_assignments.num_rows == 0 or read_demux.num_rows == 0:
        return FEATURE_QUANT.empty_table(), stats

    # Collapse read_demux to (read_id, sample_id) — drop nulls + duplicates
    rd = read_demux.select(["read_id", "sample_id"])
    valid_mask = pc.is_valid(rd.column("sample_id"))
    rd_with = rd.filter(valid_mask)
    # If a read appears in multiple demux rows (multi-segment reads in
    # transcriptome demux output), all rows must agree on sample_id;
    # we'd rather count a read once per gene assignment, so deduplicate
    # on read_id before the join.
    unique_reads = rd_with.group_by("read_id").aggregate(
        [("sample_id", "min"), ("sample_id", "max")]
    )
    consistent = pc.equal(
        unique_reads.column("sample_id_min"),
        unique_reads.column("sample_id_max"),
    )
    consistent_reads = unique_reads.filter(consistent).select(
        ["read_id", "sample_id_min"]
    )
    consistent_reads = consistent_reads.rename_columns(["read_id", "sample_id"])

    # Hash-join on read_id (Arrow's default join strategy).
    joined = gene_assignments.join(
        consistent_reads, keys="read_id", join_type="inner"
    )
    stats["reads_with_sample"] = int(joined.num_rows)
    stats["reads_without_sample"] = int(
        gene_assignments.num_rows - joined.num_rows
    )

    if joined.num_rows == 0:
        return FEATURE_QUANT.empty_table(), stats

    # FK-validate sample_ids against Samples (defensive)
    known_samples = set(samples.ids)
    sids = joined.column("sample_id").to_pylist()
    unknown = {sid for sid in sids if sid not in known_samples}
    if unknown:
        sample = sorted(unknown)[:5]
        raise ValueError(
            f"gene_assignments × read_demux join produced sample_ids "
            f"absent from Samples: {sample}"
            f"{'...' if len(unknown) > 5 else ''}"
        )

    counted = joined.group_by(["gene_id", "sample_id"]).aggregate(
        [("read_id", "count")]
    )
    stats["unique_(gene,sample)_pairs"] = int(counted.num_rows)
    counts = counted.column("read_id_count").to_pylist()
    stats["total_count"] = int(sum(counts))

    gene_ids = counted.column("gene_id").to_pylist()
    sample_ids_out = counted.column("sample_id").to_pylist()
    rows = [
        {
            "feature_id": int(gid),
            "sample_id": int(sid),
            "engine": engine,
            "feature_origin": "gene_id",
            "count": float(c),
            "tpm": None,
            "cpm": None,
            "coverage_mean": None,
            "coverage_median": None,
            "coverage_fraction": None,
            "multimap_fraction": None,
        }
        for gid, sid, c in zip(gene_ids, sample_ids_out, counts, strict=True)
    ]
    return pa.Table.from_pylist(rows, schema=FEATURE_QUANT), stats


# ──────────────────────────────────────────────────────────────────────
# Fused per-chunk worker
# ──────────────────────────────────────────────────────────────────────


def fused_decode_filter_overlap_worker(
    chunk_spec: tuple[str, int, int, int],
    *,
    gene_set_bytes: bytes,
    filter_kwargs: dict[str, Any],
    overlap_kwargs: dict[str, Any],
    acquisition_id: int = 0,
    tags_to_keep: tuple[str, ...] = (),
) -> dict[str, pa.Table]:
    """Per-chunk pipeline worker.

    ``chunk_spec`` is ``(bam_path_str, vo_start, n_records, worker_idx)``
    — ``run_batched`` consumers pickle this tuple per work item. The
    worker decodes its chunk, applies the filter predicates, runs the
    gene-overlap join against the broadcast gene set, and returns shards
    keyed by ``alignments`` (filtered), ``alignment_tags`` (filtered),
    ``gene_assignments``, and ``stats``.

    ``gene_set_bytes`` is the serialised Arrow IPC form of the gene
    set produced by :func:`constellation.sequencing.quant._kernels.gene_set_from_annotation`.
    Workers deserialise locally; avoids per-chunk pickling of the full
    Annotation. Bounded-small (~5–50k rows) so the IPC blob is a few MB.
    """
    bam_path_str, vo_start, n_records, worker_idx = chunk_spec
    bam_path = Path(bam_path_str)

    chunk_alignments, chunk_tags = read_bam_alignments_chunk(
        bam_path,
        vo_start=int(vo_start),
        n_records=int(n_records),
        worker_idx=int(worker_idx),
        acquisition_id=int(acquisition_id),
        tags_to_keep=tags_to_keep,
    )
    decoded = chunk_alignments.num_rows

    filtered, _audit = apply_filter_predicates(chunk_alignments, **filter_kwargs)
    after_filter = filtered.num_rows

    # Subset tags to the surviving alignment_ids.
    if filtered.num_rows < chunk_alignments.num_rows and chunk_tags.num_rows > 0:
        kept_ids = filtered.column("alignment_id")
        tag_mask = pc.is_in(chunk_tags.column("alignment_id"), value_set=kept_ids)
        chunk_tags = chunk_tags.filter(tag_mask)

    with pa.ipc.open_stream(pa.py_buffer(gene_set_bytes)) as reader:
        gene_set = reader.read_all()

    assignments = compute_gene_overlap(filtered, gene_set, **overlap_kwargs)
    after_overlap = assignments.num_rows

    stats_table = pa.Table.from_pylist(
        [
            {
                "worker_idx": int(worker_idx),
                "decoded": int(decoded),
                "after_filter": int(after_filter),
                "after_overlap": int(after_overlap),
            }
        ]
    )

    # Cast filtered shards to schema for partitioned-parquet uniformity.
    if filtered.num_rows == 0:
        filtered = ALIGNMENT_TABLE.empty_table()
    if chunk_tags.num_rows == 0:
        chunk_tags = ALIGNMENT_TAG_TABLE.empty_table()

    return {
        "alignments": filtered,
        "alignment_tags": chunk_tags,
        "gene_assignments": assignments,
        "stats": stats_table,
    }


def serialise_gene_set(gene_set: pa.Table) -> bytes:
    """Serialise a gene_set table for cross-process broadcast.

    Helper used by the CLI handler before fanning out workers.
    """
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, gene_set.schema) as writer:
        writer.write_table(gene_set)
    return bytes(sink.getvalue().to_pybytes())


__all__ = [
    "count_reads_per_gene",
    "fused_decode_filter_overlap_worker",
    "serialise_gene_set",
]
