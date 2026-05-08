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

        With ``emit_blocks=True`` the worker also captures the cs:Z
        tag, parses each surviving alignment into its CIGAR-derived
        blocks (preferring cs:long for match/mismatch attribution,
        falling back to CIGAR), and returns a fifth shard
        ``alignment_blocks``. Phase 1 power-user toggle — off by
        default leaves the existing fast path untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.align.cigar import (
    parse_cigar_blocks,
    parse_cs_long_blocks,
    query_start_from_cigar,
)
from constellation.sequencing.quant._kernels import (
    apply_filter_predicates,
    compute_gene_overlap,
)
from constellation.sequencing.readers.sam_bam import read_bam_alignments_chunk
from constellation.sequencing.samples import Samples
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_CS_TABLE,
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
    group_by-sum to FEATURE_QUANT, and populate per-sample TPM.

    Both inputs are pa.Tables — caller materialises partitioned
    datasets via ``.to_table()`` before invocation. Memory budget at
    200M-read scale: ~3 GB for ``gene_assignments`` + ~10 GB for
    ``read_demux``; fits comfortably on workstation RAM, comfortable
    on cluster nodes.

    ``samples`` is consumed for FK-validation only — the FEATURE_QUANT
    output is keyed on ``sample_id`` integers.

    **Long-read TPM convention.** Each emitted row has
    ``tpm = count × 1e6 / sum_in_sample(count)`` — depth-normalised, no
    length division. In long-read full-length cDNA, one read ≈ one
    transcript, so ``count`` is already proportional to transcript
    abundance with no length dependence; the short-read
    ``count/length × 1e6 / sum(count/length)`` formula would actively
    introduce error here. Mathematically this is identical to CPM, but
    semantically it's TPM in the long-read sense (transcripts per
    million transcripts in the sample). ``cpm`` is left null because
    populating it with the same value would be redundant noise.

    TODO(short_read): when short-read RNA-seq workflows arrive, add a
    ``mode: Literal['long_read', 'short_read']`` parameter that routes
    to the length-aware formula and pulls gene length from an
    ``Annotation`` argument.
    """
    stats: dict[str, int] = {
        "gene_assignments_in": int(gene_assignments.num_rows),
        "reads_with_sample": 0,
        "reads_without_sample": 0,
        "unique_(gene,sample)_pairs": 0,
        "total_count": 0,
        "samples_normalised": 0,
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

    # Per-sample total-count for depth normalisation. Long-read TPM:
    # count × 1e6 / sum_in_sample(count). See function docstring.
    sample_totals = counted.group_by("sample_id").aggregate(
        [("read_id_count", "sum")]
    )
    total_by_sid: dict[int, int] = {
        int(sid): int(tot)
        for sid, tot in zip(
            sample_totals.column("sample_id").to_pylist(),
            sample_totals.column("read_id_count_sum").to_pylist(),
            strict=True,
        )
    }
    stats["samples_normalised"] = len(total_by_sid)

    gene_ids = counted.column("gene_id").to_pylist()
    sample_ids_out = counted.column("sample_id").to_pylist()
    rows = [
        {
            "feature_id": int(gid),
            "sample_id": int(sid),
            "engine": engine,
            "feature_origin": "gene_id",
            "count": float(c),
            "tpm": (
                float(c) * 1e6 / total_by_sid[int(sid)]
                if total_by_sid.get(int(sid))
                else 0.0
            ),
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
# Block extraction (Phase 1 power-user toggle)
# ──────────────────────────────────────────────────────────────────────


def extract_alignment_blocks(
    alignments: pa.Table,
    tags: pa.Table,
    *,
    intron_min_bp: int = 25,
) -> pa.Table:
    """Lift CIGAR + cs:long into one ``ALIGNMENT_BLOCK_TABLE`` row per
    aligned exon-block.

    Per surviving alignment, prefer cs:long (populates n_match /
    n_mismatch) and fall back to CIGAR (leaves them null). ``query_start``
    accounts for leading soft clips; hard clips are invisible to SEQ
    and don't affect query coords. Block-break rule: any ``N``, plus
    any ``D >= intron_min_bp``. Long ``I`` never breaks a block.

    Returns an empty ALIGNMENT_BLOCK_TABLE-shaped table when alignments
    is empty.
    """
    if alignments.num_rows == 0:
        return ALIGNMENT_BLOCK_TABLE.empty_table()

    # Build alignment_id → cs lookup once. Worker layer feeds us only
    # the tag rows whose ``tag == 'cs'`` if it filtered upstream;
    # otherwise we filter here.
    cs_by_aid: dict[int, str] = {}
    if tags.num_rows > 0:
        if "tag" in tags.schema.names:
            cs_mask = pc.equal(tags.column("tag"), "cs")
            cs_rows = tags.filter(cs_mask)
        else:
            cs_rows = tags
        if cs_rows.num_rows > 0:
            cs_by_aid = {
                int(aid): str(val)
                for aid, val in zip(
                    cs_rows.column("alignment_id").to_pylist(),
                    cs_rows.column("value").to_pylist(),
                    strict=True,
                )
            }

    aids = alignments.column("alignment_id").to_pylist()
    ref_starts = alignments.column("ref_start").to_pylist()
    cigars = alignments.column("cigar_string").to_pylist()

    rows: list[dict[str, Any]] = []
    for aid, ref_start, cigar in zip(aids, ref_starts, cigars, strict=True):
        if not cigar or cigar == "*":
            continue
        cs = cs_by_aid.get(int(aid))
        if cs:
            # cs:long does not encode soft clips — the leading soft-clip
            # count from CIGAR is the cs-side query offset.
            blocks = parse_cs_long_blocks(
                cs,
                ref_start=int(ref_start),
                query_start=query_start_from_cigar(cigar),
            )
        else:
            # parse_cigar_blocks handles soft clips internally; do NOT
            # pre-add the leading-S count here or it gets double-counted.
            blocks = parse_cigar_blocks(
                cigar,
                ref_start=int(ref_start),
                intron_min_bp=intron_min_bp,
            )
        for blk in blocks:
            rows.append(
                {
                    "alignment_id": int(aid),
                    "block_index": int(blk.block_index),
                    "ref_start": int(blk.ref_start),
                    "ref_end": int(blk.ref_end),
                    "query_start": int(blk.query_start),
                    "query_end": int(blk.query_end),
                    "n_match": (
                        None if blk.n_match is None else int(blk.n_match)
                    ),
                    "n_mismatch": (
                        None if blk.n_mismatch is None else int(blk.n_mismatch)
                    ),
                    "n_insert": int(blk.n_insert),
                    "n_delete": int(blk.n_delete),
                }
            )

    if not rows:
        return ALIGNMENT_BLOCK_TABLE.empty_table()
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_BLOCK_TABLE)


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
    emit_blocks: bool = False,
    emit_cs: bool = False,
    intron_min_bp: int = 25,
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

    With ``emit_blocks=True`` the worker also captures the cs tag (added
    transparently to ``tags_to_keep`` if not already present), parses
    each surviving alignment into ``ALIGNMENT_BLOCK_TABLE`` rows via
    :func:`extract_alignment_blocks`, and returns a fifth shard
    ``alignment_blocks``. The transparently-added cs tag is stripped
    from the returned ``alignment_tags`` shard so the on-disk tag
    table stays the size the user asked for.

    With ``emit_cs=True`` the worker additionally emits a sixth shard
    ``alignment_cs`` (``ALIGNMENT_CS_TABLE``) preserving the per-alignment
    cs:long string. Required by Phase 2 ``--build-consensus`` for
    base-resolution PWM accumulation. Implies the cs tag will be
    captured even when ``emit_blocks=False``; the user-visible
    ``alignment_tags`` shard is still stripped of the worker-internal
    cs row when the user did not explicitly request it via ``tags_to_keep``.
    """
    bam_path_str, vo_start, n_records, worker_idx = chunk_spec
    bam_path = Path(bam_path_str)

    user_tags = frozenset(tags_to_keep)
    cs_added = (emit_blocks or emit_cs) and "cs" not in user_tags
    effective_tags = (
        tags_to_keep + ("cs",) if cs_added else tags_to_keep
    )

    chunk_alignments, chunk_tags = read_bam_alignments_chunk(
        bam_path,
        vo_start=int(vo_start),
        n_records=int(n_records),
        worker_idx=int(worker_idx),
        acquisition_id=int(acquisition_id),
        tags_to_keep=effective_tags,
    )
    decoded = chunk_alignments.num_rows

    filtered, _audit = apply_filter_predicates(chunk_alignments, **filter_kwargs)
    after_filter = filtered.num_rows

    # Subset tags to the surviving alignment_ids.
    if filtered.num_rows < chunk_alignments.num_rows and chunk_tags.num_rows > 0:
        kept_ids = filtered.column("alignment_id")
        tag_mask = pc.is_in(chunk_tags.column("alignment_id"), value_set=kept_ids)
        chunk_tags = chunk_tags.filter(tag_mask)

    # Block extraction reads cs out of the (possibly augmented) tag set
    # before we drop the worker-internal cs rows. Same source for the
    # cs sidecar shard.
    blocks_table: pa.Table | None = None
    if emit_blocks:
        blocks_table = extract_alignment_blocks(
            filtered, chunk_tags, intron_min_bp=intron_min_bp
        )

    cs_table: pa.Table | None = None
    if emit_cs:
        cs_table = extract_alignment_cs(filtered, chunk_tags)

    if cs_added and chunk_tags.num_rows > 0:
        # Strip the worker-internal cs rows from the user-visible shard.
        not_cs = pc.not_equal(chunk_tags.column("tag"), "cs")
        chunk_tags = chunk_tags.filter(not_cs)

    with pa.ipc.open_stream(pa.py_buffer(gene_set_bytes)) as reader:
        gene_set = reader.read_all()

    assignments = compute_gene_overlap(filtered, gene_set, **overlap_kwargs)
    after_overlap = assignments.num_rows

    stats_row: dict[str, Any] = {
        "worker_idx": int(worker_idx),
        "decoded": int(decoded),
        "after_filter": int(after_filter),
        "after_overlap": int(after_overlap),
    }
    if blocks_table is not None:
        stats_row["n_blocks"] = int(blocks_table.num_rows)
    if cs_table is not None:
        stats_row["n_cs"] = int(cs_table.num_rows)
    stats_table = pa.Table.from_pylist([stats_row])

    # Cast filtered shards to schema for partitioned-parquet uniformity.
    if filtered.num_rows == 0:
        filtered = ALIGNMENT_TABLE.empty_table()
    if chunk_tags.num_rows == 0:
        chunk_tags = ALIGNMENT_TAG_TABLE.empty_table()

    out: dict[str, pa.Table] = {
        "alignments": filtered,
        "alignment_tags": chunk_tags,
        "gene_assignments": assignments,
        "stats": stats_table,
    }
    if blocks_table is not None:
        out["alignment_blocks"] = blocks_table
    if cs_table is not None:
        out["alignment_cs"] = cs_table
    return out


def extract_alignment_cs(
    alignments: pa.Table, tags: pa.Table
) -> pa.Table:
    """Pull the cs tag rows out of a tag shard into ``ALIGNMENT_CS_TABLE``.

    Only emits rows for alignments that survived filtering (i.e. appear
    in ``alignments``). Returns an empty schema-shaped table when no cs
    rows are present.
    """
    if tags.num_rows == 0 or alignments.num_rows == 0:
        return ALIGNMENT_CS_TABLE.empty_table()
    cs_mask = pc.equal(tags.column("tag"), "cs")
    cs_rows = tags.filter(cs_mask)
    if cs_rows.num_rows == 0:
        return ALIGNMENT_CS_TABLE.empty_table()
    # Restrict to alignment_ids present in the surviving set.
    keep_mask = pc.is_in(
        cs_rows.column("alignment_id"),
        value_set=alignments.column("alignment_id"),
    )
    cs_rows = cs_rows.filter(keep_mask)
    if cs_rows.num_rows == 0:
        return ALIGNMENT_CS_TABLE.empty_table()
    return pa.table(
        {
            "alignment_id": cs_rows.column("alignment_id"),
            "cs_string": cs_rows.column("value"),
        },
        schema=ALIGNMENT_CS_TABLE,
    )


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
    "extract_alignment_blocks",
    "extract_alignment_cs",
    "fused_decode_filter_overlap_worker",
    "serialise_gene_set",
]
