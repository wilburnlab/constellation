"""Phase 2 — genome-guided fingerprint clustering.

Consumes ``READ_FINGERPRINT_TABLE`` (re-derived at runtime from
``alignment_blocks/`` + ``introns.parquet``, so the
``--intron-tolerance-bp`` knob is a clustering parameter — sweeping it
re-runs ``cluster_junctions`` against the raw per-position rows
in ``introns.parquet`` and rebuilds fingerprints in one shot), the
trimmed transcript-window sequences from S1 demux, and per-alignment
block summaries; produces the splicing-topology-resolved clusters that
Phase 4 will measure de novo Phase 3 against.

Granularity is the **transcript / isoform** level — finer than gene,
coarser than per-base variant. A gene with N isoforms produces ≥ N
clusters (truncated reads, novel isoforms, and basecaller-junction-
errors can yield more); reads agreeing on canonical-intron-id splicing
topology collapse into one cluster regardless of substitution noise.

Algorithm (roadmap §2.1):

1. Group reads by ``fingerprint_hash`` per (contig, strand).
2. Strand-aware 5'/3' span-coherence drift filter — separates
   alt-TSS / alt-polyA outliers from the within-cluster majority.
   Drift-filtered reads are *kept* in the membership table with
   ``role='drift_filtered'`` so Phase 4 can ask where Phase 3
   landed them.
3. Layer-0 dereplicate trimmed transcript-window sequences within
   each surviving cluster; pick the most-abundant unique sequence's
   read as the representative (tiebreak by mean ``dorado_quality``).
4. Compute per-member fit-to-cluster scores (drift_5p_bp,
   drift_3p_bp, match_rate, indel_rate, n_aligned_bp). Multi-
   parameter — downstream consumers can build any weighted
   goodness-of-fit on top.
5. Optional ``build_consensus``: per-position weighted-PWM majority
   vote on the genome coordinate frame via the shared
   ``align/consensus.py`` primitive.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_dataset
import torch

from constellation.sequencing.align.consensus import build_consensus
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)


# Sentinel used in place of null sample_id while sorting / walking
# partition boundaries — torch can diff a plain int64 tensor but not
# a nullable Arrow column directly. Output sample_id is converted
# back to nullable on the cluster row.
_SAMPLE_NULL_SENTINEL: int = -1 << 60


def _build_alignment_metrics(alignment_blocks: pa.Table) -> pa.Table:
    """Sum n_match / n_mismatch / n_insert / n_delete per alignment_id.

    Returns a 6-column Arrow table keyed on ``alignment_id``:
    ``n_match_sum, n_mismatch_sum, n_insert_sum, n_delete_sum,
    cs_aware``. The prior dict-returning version did 5 × to_pylist over
    ~200M-row block tables and accumulated nested Python dicts —
    catastrophic at scale. Now a single Arrow group_by aggregate replaces
    the inner Python row loop with a multithreaded C++ kernel call.

    ``n_match`` / ``n_mismatch`` are nullable on the input
    (cs:long absent → CIGAR-only block). To preserve the prior
    "cs_aware=False propagates if ANY contributing block is missing"
    semantics, we coerce missing values to 0 for summation but track
    cs-awareness via a per-block int8 indicator min-aggregated per
    alignment (any 0 wins).
    """
    out_schema = pa.schema(
        [
            pa.field("alignment_id", pa.int64(), nullable=False),
            pa.field("n_match_sum", pa.int64(), nullable=False),
            pa.field("n_mismatch_sum", pa.int64(), nullable=False),
            pa.field("n_insert_sum", pa.int64(), nullable=False),
            pa.field("n_delete_sum", pa.int64(), nullable=False),
            pa.field("cs_aware", pa.bool_(), nullable=False),
        ]
    )
    if alignment_blocks.num_rows == 0:
        return out_schema.empty_table()

    cs_aware_per_block = pc.and_(
        pc.is_valid(alignment_blocks.column("n_match")),
        pc.is_valid(alignment_blocks.column("n_mismatch")),
    )
    aug = pa.table(
        {
            "alignment_id": alignment_blocks.column("alignment_id"),
            "n_match": pc.fill_null(alignment_blocks.column("n_match"), 0),
            "n_mismatch": pc.fill_null(
                alignment_blocks.column("n_mismatch"), 0
            ),
            "n_insert": alignment_blocks.column("n_insert"),
            "n_delete": alignment_blocks.column("n_delete"),
            "cs_aware_int": pc.cast(cs_aware_per_block, pa.int8()),
        }
    )
    aggregated = aug.group_by("alignment_id").aggregate(
        [
            ("n_match", "sum"),
            ("n_mismatch", "sum"),
            ("n_insert", "sum"),
            ("n_delete", "sum"),
            ("cs_aware_int", "min"),
        ]
    )
    return pa.table(
        {
            "alignment_id": aggregated.column("alignment_id"),
            "n_match_sum": pc.cast(
                aggregated.column("n_match_sum"), pa.int64()
            ),
            "n_mismatch_sum": pc.cast(
                aggregated.column("n_mismatch_sum"), pa.int64()
            ),
            "n_insert_sum": pc.cast(
                aggregated.column("n_insert_sum"), pa.int64()
            ),
            "n_delete_sum": pc.cast(
                aggregated.column("n_delete_sum"), pa.int64()
            ),
            "cs_aware": pc.equal(
                aggregated.column("cs_aware_int_min"), pa.scalar(1, type=pa.int8())
            ),
        },
        schema=out_schema,
    )


def _build_alignment_lookup(alignments: pa.Table) -> pa.Table:
    """Primary-alignment metadata keyed on ``read_id``.

    Returns a 3-column Arrow table: ``(read_id, alignment_id,
    ref_start)``. The prior dict-of-dicts version cost 3 × to_pylist
    over the ~200M-row alignments table and a 200M-iter dict-of-dicts
    build. The replacement is a single Arrow filter + select that
    plugs directly into the join in
    :func:`cluster_by_fingerprint`'s prologue.
    """
    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    return alignments.filter(primary_mask).select(
        ["read_id", "alignment_id", "ref_start"]
    )


def cluster_by_fingerprint(
    fingerprints: pa.Table,
    reads: pa.Table,
    *,
    alignments: pa.Table,
    alignment_blocks: pa.Table | None = None,
    alignment_cs: pa.Table | pa_dataset.Dataset | None = None,
    genome: GenomeReference | None = None,
    read_to_sample: pa.Table | None = None,
    max_5p_drift: int = 25,
    max_3p_drift: int = 75,
    min_cluster_size: int = 1,
    build_consensus_seq: bool = False,
    drop_drift_filtered: bool = False,
    per_sample_clusters: bool = False,
    cluster_id_seed: int = 0,
    consensus_threads: int = 1,
    progress: Callable[[str], None] | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Group reads into clusters by quantised splicing-topology fingerprint.

    Parameters
    ----------
    fingerprints : pa.Table
        ``READ_FINGERPRINT_TABLE``-shaped. Already derived against the
        clustering-time ``INTRON_TABLE`` (which carries the per-row
        ``intron_id`` cluster assignments computed at the chosen
        ``intron_tolerance_bp``).
    reads : pa.Table
        Must carry columns ``read_id``, ``sequence`` (the **trimmed
        transcript window**, NOT the raw read), and optionally
        ``dorado_quality`` (mean phred over the trimmed window;
        Layer-0 tiebreak).
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped — supplies primary-alignment lookup
        from ``read_id`` to ``(alignment_id, ref_start)``.
    alignment_blocks : pa.Table, optional
        ``ALIGNMENT_BLOCK_TABLE``-shaped. When provided, populates
        ``match_rate`` / ``indel_rate`` / ``n_aligned_bp`` on
        cluster_membership rows. Required when
        ``build_consensus_seq=True``.
    alignment_cs : pa.Table, optional
        ``ALIGNMENT_CS_TABLE``-shaped. Required when
        ``build_consensus_seq=True``.
    genome : GenomeReference, optional
        Required when ``build_consensus_seq=True``; supplies the
        reference window for PWM accumulation.
    read_to_sample : pa.Table, optional
        2-column Arrow table with ``read_id`` (string) + ``sample_id``
        (int64) for ``per_sample_clusters`` mode. Reads without a sample
        mapping fall into the "unassigned" partition (``sample_id``
        column null in the output).
    max_5p_drift, max_3p_drift : int
        Strand-aware bp tolerance for the span-coherence filter.
        Defaults 25 / 75 mirror the lab observations on TSS / polyA
        scatter.
    min_cluster_size : int, default 1
        Drop clusters whose surviving (post-drift-filter) member
        count is below this. Singletons are kept by default.
    build_consensus_seq : bool, default False
        Populate ``consensus_sequence`` per cluster via the shared
        weighted-PWM kernel. Requires ``alignment_blocks``,
        ``alignment_cs``, and ``genome``.
    drop_drift_filtered : bool, default False
        When True, drift-filtered reads are excluded from
        ``cluster_membership``. Default keeps them with
        ``role='drift_filtered'``.
    per_sample_clusters : bool, default False
        When True, partition by sample_id in addition to (contig,
        strand). Default treats clusters as spanning samples.
    cluster_id_seed : int, default 0
        Starting cluster_id. Caller packs ``(worker_idx << 32)`` here
        when running parallel partitions.
    progress : Callable[[str], None] | None, default None
        Optional one-arg callable that the function calls at each
        internal stage boundary + every K clusters inside the per-
        cluster loop. The CLI wires it up under ``--progress``; tests
        pass ``None``. Each message is a short free-form string; the
        callable is responsible for prefixing / flushing.

    Returns
    -------
    (clusters, membership) : (pa.Table, pa.Table)
        Schemas: ``TRANSCRIPT_CLUSTER_TABLE``, ``CLUSTER_MEMBERSHIP_TABLE``.
    """
    if build_consensus_seq:
        if alignment_blocks is None or alignment_cs is None or genome is None:
            raise ValueError(
                "build_consensus_seq=True requires alignment_blocks, "
                "alignment_cs, and genome"
            )
    if fingerprints.num_rows == 0:
        return (
            TRANSCRIPT_CLUSTER_TABLE.empty_table(),
            CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        )

    # ── Pre-index each source table by fingerprint row ──────────
    # The prior implementation chained left-outer joins to build a
    # wide ``enriched`` table (~80 GB at PromethION scale with sequence
    # strings) and then sorted it globally — that crashed with
    # ``ArrowInvalid: offset overflow while concatenating arrays`` at
    # 200M reads because ``sequence`` blew past Arrow's 2 GiB int32
    # string-buffer limit during the post-sort ``take`` + ``combine_chunks``.
    #
    # Replacement: resolve per-fingerprint row indices into each source
    # table once via ``pc.index_in`` (Arrow hash-join under the hood),
    # sort only a narrow all-numeric keys table (~8 GB at 200M rows,
    # offset-overflow-immune), and ``pc.take`` per cluster from each
    # source. Source tables (notably ``reads.sequence``) never go through
    # the heavy sort. See sequencing/CLAUDE.md "Resolve-stage pitfalls".
    def _p(msg: str) -> None:
        if progress is not None:
            progress(f"  cluster_by_fingerprint: {msg}")

    _p("build primary_lookup")
    primary_lookup = _build_alignment_lookup(alignments)
    _p(f"primary_lookup: {primary_lookup.num_rows:,} rows")
    metrics_table: pa.Table | None = None
    if alignment_blocks is not None:
        _p("build alignment_metrics")
        metrics_table = _build_alignment_metrics(alignment_blocks)
        _p(f"alignment_metrics: {metrics_table.num_rows:,} rows")

    n = fingerprints.num_rows
    fp_read_ids = fingerprints.column("read_id")
    has_quality = "dorado_quality" in reads.schema.names

    # Row indices into each source. Nulls propagate where there's no match.
    _p("pc.index_in: reads")
    reads_idx = pc.index_in(fp_read_ids, reads.column("read_id"))
    _p("pc.index_in: primary_lookup")
    primary_idx = pc.index_in(fp_read_ids, primary_lookup.column("read_id"))
    if metrics_table is not None:
        _p("pc.index_in: metrics (via primary alignment_id)")
        aid_per_fp = pc.take(
            primary_lookup.column("alignment_id"), primary_idx
        )
        metrics_idx = pc.index_in(
            aid_per_fp, metrics_table.column("alignment_id")
        )
    else:
        metrics_idx = None

    # sample_id per fingerprint row.
    if per_sample_clusters and read_to_sample is not None:
        _p("pc.index_in: read_to_sample")
        rts_idx = pc.index_in(fp_read_ids, read_to_sample.column("read_id"))
        sample_id_col = pc.take(
            read_to_sample.column("sample_id"), rts_idx
        )
    else:
        sample_id_col = pa.nulls(n, type=pa.int64())
    sample_id_for_sort = pc.fill_null(sample_id_col, _SAMPLE_NULL_SENTINEL)
    _p("source indices resolved")

    # contig_id → name lookup (small — keep as dict for consensus path).
    contig_id_to_name: dict[int, str] = {}
    if genome is not None:
        for cid, name in zip(
            genome.contigs.column("contig_id").to_pylist(),
            genome.contigs.column("name").to_pylist(),
            strict=True,
        ):
            contig_id_to_name[int(cid)] = str(name)

    # ── Consensus prefetches (only when --build-consensus) ──────
    # Two per-cluster fixed costs in the consensus path that we lift
    # out of the loop entirely:
    #   1. `genome.sequence_of(contig_id)` is uncached and decodes
    #      the entire chromosome (~200 MB for mouse chr1) on each
    #      call (see sequencing/CLAUDE.md pitfall #1). For ~46k
    #      clusters across ~22 contigs, that's ~46k × 200 ms = ~2.5h
    #      of pure chromosome-decoding overhead. Prefetched once per
    #      unique contig that actually appears in any cluster.
    #   2. `alignment_cs.to_table(filter=...)` with Dataset filter
    #      pushdown scans the ~50 GB alignment_cs/ parquet dir for
    #      1-N matching rows per cluster. Replaced with a single
    #      up-front filter to the fingerprinted-aid set + per-cluster
    #      Arrow `index_in` + `take` (sub-ms each).
    contig_seq_cache: dict[int, str] = {}
    alignment_cs_table: pa.Table | None = None
    cs_idx_per_fp: pa.Array | pa.ChunkedArray | None = None
    if build_consensus_seq and genome is not None and alignment_cs is not None:
        needed_contig_ids = pc.unique(
            fingerprints.column("contig_id")
        ).to_pylist()
        _p(
            f"consensus prefetch: contig sequences for "
            f"{len(needed_contig_ids)} contigs"
        )
        for cid in needed_contig_ids:
            contig_seq_cache[int(cid)] = genome.sequence_of(int(cid))
        _p("consensus prefetch: contig_seq_cache populated")

        fp_aids = pc.take(
            primary_lookup.column("alignment_id"), primary_idx
        )
        fp_aids_unique = pc.unique(fp_aids)
        _p(
            f"consensus prefetch: materializing alignment_cs filtered "
            f"to {len(fp_aids_unique):,} fingerprinted alignment_ids"
        )
        if isinstance(alignment_cs, pa_dataset.Dataset):
            alignment_cs_table = alignment_cs.to_table(
                columns=["alignment_id", "cs_string"],
                filter=pc.is_in(
                    pa_dataset.field("alignment_id"),
                    value_set=fp_aids_unique,
                ),
            )
        else:
            alignment_cs_table = alignment_cs.select(
                ["alignment_id", "cs_string"]
            ).filter(
                pc.is_in(
                    alignment_cs.column("alignment_id"),
                    value_set=fp_aids_unique,
                )
            )
        # cs_string is `string` (int32 offsets) in the on-disk schema.
        # At PromethION scale, cs:long across millions of alignments
        # totals 10s of GB and overflows int32 string-buffer offsets
        # during the downstream combine_chunks + per-cluster pc.take
        # (the same offset-overflow class that bit reads.sequence in
        # the CLI). Cast to large_string per-chunk *before* combine_
        # chunks — the cast is offset-arithmetic only (no data copy)
        # and each input chunk individually fits under the 2 GiB
        # string cap, so the cast itself doesn't overflow.
        if pa.types.is_string(
            alignment_cs_table.schema.field("cs_string").type
        ):
            alignment_cs_table = alignment_cs_table.set_column(
                alignment_cs_table.schema.get_field_index("cs_string"),
                "cs_string",
                pc.cast(
                    alignment_cs_table.column("cs_string"),
                    pa.large_string(),
                ),
            )
            _p("consensus prefetch: cs_string cast string → large_string")
        alignment_cs_table = alignment_cs_table.combine_chunks()
        _p(
            f"consensus prefetch: alignment_cs_table ready "
            f"({alignment_cs_table.num_rows:,} rows, "
            f"{alignment_cs_table.nbytes / 1e9:.2f} GB)"
        )
        # cs row-index per fingerprint, computed once. Avoids the prior
        # per-cluster `pc.index_in(member_aids, alignment_cs_table
        # .column("alignment_id"))` call — pyarrow rebuilds the
        # value-set hash table on every call, which at PromethION
        # scale meant ~1 sec per cluster × 46k clusters ≈ 13 hours of
        # pure hash-table-build wall time. One up-front pc.index_in
        # over fp_aids builds the hash once and probes N_fingerprints
        # times; the result rides through narrow_keys → sorted_keys so
        # the per-cluster work becomes a free slice + a small pc.take
        # from alignment_cs_table.cs_string.
        _p(
            "consensus prefetch: pc.index_in fp_aids → "
            "alignment_cs_table rows"
        )
        cs_idx_per_fp = pc.index_in(
            fp_aids, alignment_cs_table.column("alignment_id")
        )
        _p("consensus prefetch: cs_idx_per_fp ready")

    # ── Narrow keys table for sort ──────────────────────────────
    # All-numeric except for ``strand`` (1-char strings: 200M × 1 byte
    # is well under int32 offset limits). Carries the per-source row
    # indices alongside the sort keys so per-cluster takes are O(cluster
    # size).
    narrow_cols: dict[str, pa.ChunkedArray | pa.Array] = {
        "row_idx": pa.array(np.arange(n, dtype=np.int64)),
        "contig_id": fingerprints.column("contig_id"),
        "strand": fingerprints.column("strand"),
        "sample_id": sample_id_for_sort,
        "fingerprint_hash": fingerprints.column("fingerprint_hash"),
        "reads_idx": reads_idx,
        "primary_idx": primary_idx,
    }
    if metrics_idx is not None:
        narrow_cols["metrics_idx"] = metrics_idx
    if cs_idx_per_fp is not None:
        narrow_cols["cs_idx"] = cs_idx_per_fp
    narrow_keys = pa.table(narrow_cols)
    _p(f"narrow_keys built ({narrow_keys.num_rows:,} rows)")

    _p("sort_by + combine_chunks")
    sorted_keys = narrow_keys.sort_by(
        [
            ("contig_id", "ascending"),
            ("strand", "ascending"),
            ("sample_id", "ascending"),
            ("fingerprint_hash", "ascending"),
        ]
    ).combine_chunks()
    _p("sort done")
    boundaries = _cluster_boundaries(sorted_keys)
    n_boundaries = len(boundaries)
    if n_boundaries > 0:
        sizes_np = np.array([sz for _, sz in boundaries], dtype=np.int64)
        _p(
            f"boundary walk: {n_boundaries:,} clusters, "
            f"size min={int(sizes_np.min())} median={int(np.median(sizes_np))} "
            f"p99={int(np.percentile(sizes_np, 99))} "
            f"max={int(sizes_np.max())}"
        )
        del sizes_np
    else:
        _p("boundary walk: 0 clusters")

    # Combine_chunks the source tables once before the per-cluster
    # loop. pyarrow's `pc.take` has per-call dispatch overhead roughly
    # proportional to chunk count, even when the index array is 1
    # element — and we'll call pc.take ~10×/cluster × N_clusters. With
    # multi-shard parquet sources that each split into many chunks,
    # this dispatch cost dominates the loop wall time (a 1-row cluster
    # takes ~7s with chunked sources vs. ~10ms with single-chunk
    # sources at mouse scale).
    #
    # Each combine is guarded by an .nbytes size check — at PromethION
    # scale with poor fingerprint coverage, `reads.sequence` is a few
    # GB and safely combinable; at full coverage it could exceed
    # available RAM and the guard falls back to chunked + slow takes.
    _COMBINE_BYTES_LIMIT = 64 * 1024 ** 3  # 64 GB hard cap per source

    def _maybe_combine(t: pa.Table, name: str) -> pa.Table:
        try:
            nb = int(t.nbytes)
        except Exception:
            nb = -1
        if 0 <= nb <= _COMBINE_BYTES_LIMIT:
            _p(f"combine_chunks: {name} ({nb / 1e9:.2f} GB)")
            return t.combine_chunks()
        _p(
            f"combine_chunks: SKIP {name} ({nb / 1e9:.2f} GB > "
            f"{_COMBINE_BYTES_LIMIT / 1e9:.0f} GB cap) — falling back to "
            f"per-call ChunkedArray dispatch (slow)"
        )
        return t

    fingerprints = _maybe_combine(fingerprints, "fingerprints")
    if reads.num_rows > 0:
        reads = _maybe_combine(reads, "reads")
    primary_lookup = _maybe_combine(primary_lookup, "primary_lookup")
    if metrics_table is not None:
        metrics_table = _maybe_combine(metrics_table, "metrics_table")

    # Hoist source-column references out of the loop.
    sk_row_idx = sorted_keys.column("row_idx")
    sk_reads_idx = sorted_keys.column("reads_idx")
    sk_primary_idx = sorted_keys.column("primary_idx")
    sk_sample_id = sorted_keys.column("sample_id")
    sk_metrics_idx = (
        sorted_keys.column("metrics_idx") if metrics_table is not None else None
    )
    sk_cs_idx = (
        sorted_keys.column("cs_idx") if cs_idx_per_fp is not None else None
    )
    reads_seq = reads.column("sequence") if reads.num_rows > 0 else None
    reads_qual = (
        reads.column("dorado_quality")
        if reads.num_rows > 0 and has_quality
        else None
    )
    pri_aid = primary_lookup.column("alignment_id")
    pri_rstart = primary_lookup.column("ref_start")
    if metrics_table is not None:
        m_match = metrics_table.column("n_match_sum")
        m_mismatch = metrics_table.column("n_mismatch_sum")
        m_insert = metrics_table.column("n_insert_sum")
        m_delete = metrics_table.column("n_delete_sum")
        m_csaware = metrics_table.column("cs_aware")
    _p("source column refs hoisted — entering per-cluster loop")

    # ── Per-cluster assembly + emission ─────────────────────────
    cluster_rows: list[dict[str, Any]] = []
    membership_tables: list[pa.Table] = []
    next_cluster_id = int(cluster_id_seed)
    log_every = max(1, n_boundaries // 100)  # ~1% increments

    sample_null_sentinel_scalar = pa.scalar(
        _SAMPLE_NULL_SENTINEL, type=pa.int64()
    )

    import time as _time
    loop_t0 = _time.perf_counter()
    last_progress_t = loop_t0
    PROGRESS_INTERVAL_S = 30.0  # heartbeat between % milestones
    SLOW_CLUSTER_WARN_S = 60.0  # per-cluster wall-time warning
    DETAIL_FIRST_N = 5  # detailed per-cluster timing for the first N

    for i, (start, size) in enumerate(boundaries):
        # Heartbeat: % milestone OR ≥30 s since last progress line.
        if progress is not None and i > 0:
            now = _time.perf_counter()
            since_progress = now - last_progress_t
            if i % log_every == 0 or since_progress >= PROGRESS_INTERVAL_S:
                elapsed = now - loop_t0
                rate = i / elapsed if elapsed > 0 else 0.0
                running_member_count = sum(
                    t.num_rows for t in membership_tables
                )
                _p(
                    f"  per-cluster loop: {i:,} / {n_boundaries:,} "
                    f"({100 * i / n_boundaries:.1f}%) — "
                    f"emitted={len(cluster_rows):,} "
                    f"membership={running_member_count:,} "
                    f"rate={rate:.1f} clusters/s loop_elapsed={elapsed:.1f}s"
                )
                last_progress_t = now

        cluster_t0 = _time.perf_counter()
        detail = progress is not None and i < DETAIL_FIRST_N
        if detail:
            _p(f"  cluster {i}: start size={size}")

        # Per-cluster slice of sort keys + source-row indices.
        _ts = _time.perf_counter()
        row_indices = sk_row_idx.slice(start, size)
        reads_indices = sk_reads_idx.slice(start, size)
        primary_indices = sk_primary_idx.slice(start, size)
        sample_filled_slice = sk_sample_id.slice(start, size)
        t_slice = _time.perf_counter() - _ts

        # Take from each source. ``pc.take`` propagates nulls.
        _ts = _time.perf_counter()
        fp_slice = fingerprints.take(row_indices)
        t_fp_take = _time.perf_counter() - _ts

        _ts = _time.perf_counter()
        if reads_seq is not None:
            seq_col = pc.take(reads_seq, reads_indices)
            quality_col = (
                pc.take(reads_qual, reads_indices)
                if reads_qual is not None
                else pa.nulls(size, type=pa.float32())
            )
        else:
            seq_col = pa.nulls(size, type=pa.string())
            quality_col = pa.nulls(size, type=pa.float32())
        t_reads_take = _time.perf_counter() - _ts

        _ts = _time.perf_counter()
        aid_col = pc.take(pri_aid, primary_indices)
        rstart_col = pc.take(pri_rstart, primary_indices)
        t_primary_take = _time.perf_counter() - _ts

        # Un-sentinel sample_id for the emitted slice.
        _ts = _time.perf_counter()
        sample_col = pc.if_else(
            pc.equal(sample_filled_slice, sample_null_sentinel_scalar),
            pa.scalar(None, type=pa.int64()),
            sample_filled_slice,
        )
        t_sample = _time.perf_counter() - _ts

        slice_cols: dict[str, pa.Array | pa.ChunkedArray] = {
            "read_id": fp_slice.column("read_id"),
            "contig_id": fp_slice.column("contig_id"),
            "strand": fp_slice.column("strand"),
            "n_blocks": fp_slice.column("n_blocks"),
            "span_start": fp_slice.column("span_start"),
            "span_end": fp_slice.column("span_end"),
            "fingerprint_hash": fp_slice.column("fingerprint_hash"),
            "junction_signature": fp_slice.column("junction_signature"),
            "sequence": seq_col,
            "dorado_quality": quality_col,
            "alignment_id": aid_col,
            "ref_start": rstart_col,
            "sample_id": sample_col,
        }
        _ts = _time.perf_counter()
        if metrics_table is not None and sk_metrics_idx is not None:
            metrics_indices = sk_metrics_idx.slice(start, size)
            slice_cols["n_match_sum"] = pc.take(m_match, metrics_indices)
            slice_cols["n_mismatch_sum"] = pc.take(m_mismatch, metrics_indices)
            slice_cols["n_insert_sum"] = pc.take(m_insert, metrics_indices)
            slice_cols["n_delete_sum"] = pc.take(m_delete, metrics_indices)
            slice_cols["cs_aware"] = pc.take(m_csaware, metrics_indices)
        t_metrics_take = _time.perf_counter() - _ts

        # Pre-computed cs row index per fingerprint rides through into
        # _emit_cluster as a slice_table column so the per-cluster
        # consensus path doesn't have to call pc.index_in itself.
        if sk_cs_idx is not None:
            slice_cols["cs_idx"] = sk_cs_idx.slice(start, size)

        _ts = _time.perf_counter()
        slice_table = pa.table(slice_cols)
        t_table_build = _time.perf_counter() - _ts

        if detail:
            _p(
                f"  cluster {i}: slice_table built "
                f"({_time.perf_counter() - cluster_t0:.2f}s) — "
                f"slice={t_slice * 1000:.1f}ms "
                f"fp_take={t_fp_take * 1000:.1f}ms "
                f"reads_take={t_reads_take * 1000:.1f}ms "
                f"primary_take={t_primary_take * 1000:.1f}ms "
                f"sample={t_sample * 1000:.1f}ms "
                f"metrics_take={t_metrics_take * 1000:.1f}ms "
                f"table_build={t_table_build * 1000:.1f}ms"
            )

        emit_t0 = _time.perf_counter()
        emitted = _emit_cluster(
            slice_table,
            cluster_id=next_cluster_id,
            max_5p_drift=max_5p_drift,
            max_3p_drift=max_3p_drift,
            min_cluster_size=min_cluster_size,
            build_consensus_seq=build_consensus_seq,
            drop_drift_filtered=drop_drift_filtered,
            genome=genome,
            alignment_cs=alignment_cs,
            contig_id_to_name=contig_id_to_name,
            has_metrics=metrics_table is not None,
            has_quality=has_quality,
            contig_seq_cache=contig_seq_cache or None,
            alignment_cs_table=alignment_cs_table,
            consensus_threads=consensus_threads,
            verbose_progress=_p if detail else None,
        )
        cluster_dt = _time.perf_counter() - cluster_t0
        emit_dt = _time.perf_counter() - emit_t0

        if progress is not None and i < DETAIL_FIRST_N:
            _p(
                f"  cluster {i}: done in {cluster_dt:.2f}s "
                f"(_emit_cluster={emit_dt:.2f}s)"
            )
        elif progress is not None and cluster_dt >= SLOW_CLUSTER_WARN_S:
            _p(
                f"  slow cluster {i}: size={size:,} "
                f"took {cluster_dt:.1f}s (_emit_cluster={emit_dt:.1f}s)"
            )
            last_progress_t = _time.perf_counter()

        if emitted is None:
            continue
        cluster_row, members_table = emitted
        cluster_rows.append(cluster_row)
        if members_table.num_rows > 0:
            membership_tables.append(members_table)
        next_cluster_id += 1

    total_members = sum(t.num_rows for t in membership_tables)
    _p(
        f"per-cluster loop done — emitted {len(cluster_rows):,} clusters / "
        f"{total_members:,} membership rows"
    )

    if not cluster_rows:
        return (
            TRANSCRIPT_CLUSTER_TABLE.empty_table(),
            CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        )
    _p("concat_tables: clusters + membership")
    clusters = pa.Table.from_pylist(cluster_rows, schema=TRANSCRIPT_CLUSTER_TABLE)
    if membership_tables:
        membership = pa.concat_tables(membership_tables)
    else:
        membership = CLUSTER_MEMBERSHIP_TABLE.empty_table()
    _p("output tables built")
    return clusters, membership


def _cluster_boundaries(enriched_sorted: pa.Table) -> list[tuple[int, int]]:
    """Walk consecutive ``(contig_id, strand, sample_id, fingerprint_hash)``
    runs in the already-sorted enriched table.

    Each unique 4-tuple = one cluster. ``strand`` is dictionary-encoded
    to an int32 indices array for the torch boundary diff; nulls in
    ``sample_id`` are filled with a sentinel so the diff treats them
    consistently.
    """
    n = enriched_sorted.num_rows
    if n == 0:
        return []

    def _col_np(col: pa.ChunkedArray | pa.Array) -> np.ndarray:
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        return col.to_numpy(zero_copy_only=False)

    contig = _col_np(enriched_sorted.column("contig_id"))
    fp = _col_np(enriched_sorted.column("fingerprint_hash"))
    sample_filled = pc.fill_null(
        enriched_sorted.column("sample_id"), _SAMPLE_NULL_SENTINEL
    )
    sample = _col_np(sample_filled)
    strand_dict = pc.dictionary_encode(enriched_sorted.column("strand"))
    if isinstance(strand_dict, pa.ChunkedArray):
        strand_dict = strand_dict.combine_chunks()
    strand_idx = strand_dict.indices.to_numpy(zero_copy_only=False)

    contig_t = torch.from_numpy(contig.astype(np.int64, copy=False))
    fp_t = torch.from_numpy(fp.astype(np.int64, copy=False))
    sample_t = torch.from_numpy(sample.astype(np.int64, copy=False))
    strand_t = torch.from_numpy(strand_idx.astype(np.int64, copy=False))

    change_mask = torch.zeros(n, dtype=torch.bool)
    change_mask[0] = True
    if n > 1:
        change_mask[1:] = (
            (contig_t[1:] != contig_t[:-1])
            | (strand_t[1:] != strand_t[:-1])
            | (sample_t[1:] != sample_t[:-1])
            | (fp_t[1:] != fp_t[:-1])
        )
    starts = torch.nonzero(change_mask, as_tuple=False).squeeze(1).tolist()
    starts.append(n)
    return [(starts[i], starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]


def _build_metric_cols(
    table: pa.Table, *, has_metrics: bool, n_rows: int
) -> tuple[pa.Array, pa.Array, pa.Array]:
    """Vectorized per-row metrics → (match_rate, indel_rate, n_aligned_bp).

    Mirrors the original ``_metrics_from_member_row`` + the metric
    branches in ``_make_membership_row``, but as a single Arrow
    expression chain. Per-row branches collapse to ``pc.if_else`` on
    boolean masks; the result is three columns of length ``n_rows``
    that go directly into the membership table.

    Semantics (per row):
      n_aligned_bp = n_match + n_mismatch + n_insert + n_delete when
          (alignment_id is valid AND cs_aware is valid), else 0.
      match_rate = n_match / (n_match + n_mismatch) when (valid_metrics
          AND cs_aware AND n_match+n_mismatch > 0), else null.
      indel_rate = (n_insert + n_delete) / n_aligned when (valid_metrics
          AND cs_aware AND n_aligned > 0), else null.
    """
    null_float32 = pa.scalar(None, type=pa.float32())
    if not has_metrics or n_rows == 0:
        return (
            pa.nulls(n_rows, type=pa.float32()),
            pa.nulls(n_rows, type=pa.float32()),
            pa.array([0] * n_rows, type=pa.int32()),
        )
    aid_valid = pc.is_valid(table.column("alignment_id"))
    cs_aware_valid = pc.is_valid(table.column("cs_aware"))
    valid = pc.and_(aid_valid, cs_aware_valid)
    # cs_aware (bool) — fill null=False so the AND/divide flow is safe.
    cs_aware_bool = pc.fill_null(table.column("cs_aware"), False)

    n_match = pc.cast(pc.fill_null(table.column("n_match_sum"), 0), pa.int64())
    n_mismatch = pc.cast(
        pc.fill_null(table.column("n_mismatch_sum"), 0), pa.int64()
    )
    n_insert = pc.cast(
        pc.fill_null(table.column("n_insert_sum"), 0), pa.int64()
    )
    n_delete = pc.cast(
        pc.fill_null(table.column("n_delete_sum"), 0), pa.int64()
    )
    mm_sum = pc.add(n_match, n_mismatch)
    n_aligned = pc.add(pc.add(pc.add(n_match, n_mismatch), n_insert), n_delete)

    match_rate_raw = pc.divide(
        pc.cast(n_match, pa.float32()), pc.cast(mm_sum, pa.float32())
    )
    indel_rate_raw = pc.divide(
        pc.cast(pc.add(n_insert, n_delete), pa.float32()),
        pc.cast(n_aligned, pa.float32()),
    )
    match_ok = pc.and_(
        pc.and_(valid, cs_aware_bool), pc.greater(mm_sum, 0)
    )
    indel_ok = pc.and_(
        pc.and_(valid, cs_aware_bool), pc.greater(n_aligned, 0)
    )
    match_rate = pc.if_else(match_ok, match_rate_raw, null_float32)
    indel_rate = pc.if_else(indel_ok, indel_rate_raw, null_float32)
    n_aligned_bp = pc.cast(
        pc.if_else(valid, n_aligned, pa.scalar(0, type=pa.int64())),
        pa.int32(),
    )
    return match_rate, indel_rate, n_aligned_bp


def _emit_cluster(
    slice_table: pa.Table,
    *,
    cluster_id: int,
    max_5p_drift: int,
    max_3p_drift: int,
    min_cluster_size: int,
    build_consensus_seq: bool,
    drop_drift_filtered: bool,
    genome: GenomeReference | None,
    alignment_cs: pa.Table | pa_dataset.Dataset | None,
    contig_id_to_name: dict[int, str],
    has_metrics: bool,
    has_quality: bool,
    contig_seq_cache: dict[int, str] | None = None,
    alignment_cs_table: pa.Table | None = None,
    consensus_threads: int = 1,
    verbose_progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], pa.Table] | None:
    """Per-cluster work: drift filter + Layer-0 derep + optional
    consensus + cluster row + membership table.

    Returns ``(cluster_row_dict, membership_table)`` where
    ``membership_table`` is shaped per ``CLUSTER_MEMBERSHIP_TABLE``
    (possibly zero rows). Returns ``None`` when the cluster is
    filtered out (e.g. ``min_cluster_size``).

    Operates entirely on Arrow columns / numpy without ever
    materialising ``slice_table.to_pylist()`` — the prior Python list
    of dicts hit ~40 GB of Python heap on the 2.15M-read mega-cluster
    and was the dominant per-cluster cost at PromethION scale.
    """
    import time as _t

    _vp = verbose_progress  # local alias for terser checks below
    _t0 = _t.perf_counter() if _vp is not None else 0.0

    n_total = slice_table.num_rows
    if n_total == 0:
        return None

    # Scalar per-cluster metadata via direct column[0] access — cheap.
    c = slice_table.column
    contig_id = int(c("contig_id")[0].as_py())
    strand = str(c("strand")[0].as_py())
    sample_id_raw = c("sample_id")[0].as_py()
    fp_hash = int(c("fingerprint_hash")[0].as_py())

    # ── Singleton fast path ────────────────────────────────────
    # 50% of clusters at PromethION scale are size=1. The single read
    # is the representative; no drift, no duplicates, no derep loop.
    # Gated off when build_consensus_seq is on so we don't bypass the
    # consensus invocation.
    if (
        n_total == 1
        and not build_consensus_seq
        and min_cluster_size <= 1
    ):
        read_id_s = str(c("read_id")[0].as_py())
        span_start_s = int(c("span_start")[0].as_py())
        span_end_s = int(c("span_end")[0].as_py())
        match_rate_col, indel_rate_col, n_aligned_col = _build_metric_cols(
            slice_table, has_metrics=has_metrics, n_rows=1
        )
        membership_table_s = pa.Table.from_arrays(
            [
                pa.array([cluster_id], type=pa.int64()),
                pa.array([read_id_s], type=pa.string()),
                pa.array(["representative"], type=pa.string()),
                pa.array([None], type=pa.int32()),
                pa.array([None], type=pa.int32()),
                match_rate_col,
                indel_rate_col,
                n_aligned_col,
            ],
            schema=CLUSTER_MEMBERSHIP_TABLE,
        )
        cluster_row_s = {
            "cluster_id": int(cluster_id),
            "representative_read_id": read_id_s,
            "n_reads": 1,
            "identity_threshold": None,
            "consensus_sequence": None,
            "predicted_protein": None,
            "orf_start": None,
            "orf_end": None,
            "orf_strand": None,
            "codon_table": None,
            "mode": "genome-guided",
            "contig_id": contig_id,
            "strand": strand,
            "span_start": span_start_s,
            "span_end": span_end_s,
            "fingerprint_hash": fp_hash,
            "n_unique_sequences": 1,
            "sample_id": (
                None if sample_id_raw is None else int(sample_id_raw)
            ),
        }
        if _vp is not None:
            _vp(
                f"    _emit_cluster: SINGLETON FAST PATH "
                f"({(_t.perf_counter() - _t0) * 1000:.2f}ms)"
            )
        return cluster_row_s, membership_table_s

    # ── Vectorized drift filter ──────────────────────────────
    span_start_np = slice_table.column("span_start").to_numpy(
        zero_copy_only=False
    )
    span_end_np = slice_table.column("span_end").to_numpy(
        zero_copy_only=False
    )
    if n_total > 1:
        med_start = float(np.median(span_start_np))
        med_end = float(np.median(span_end_np))
        if strand == "-":
            drift_5p_np = (span_end_np - med_end).astype(np.int64)
            drift_3p_np = (span_start_np - med_start).astype(np.int64)
        else:
            drift_5p_np = (span_start_np - med_start).astype(np.int64)
            drift_3p_np = (span_end_np - med_end).astype(np.int64)
        kept_mask_np = (np.abs(drift_5p_np) <= max_5p_drift) & (
            np.abs(drift_3p_np) <= max_3p_drift
        )
    else:
        # Singleton on this path (build_consensus_seq=True) — no median
        # to compare; drift is conceptually null.
        drift_5p_np = np.zeros(n_total, dtype=np.int64)
        drift_3p_np = np.zeros(n_total, dtype=np.int64)
        kept_mask_np = np.array([True])

    kept_indices_np = np.nonzero(kept_mask_np)[0]
    drifted_indices_np = np.nonzero(~kept_mask_np)[0]
    n_kept = int(len(kept_indices_np))
    n_drifted = int(len(drifted_indices_np))
    if _vp is not None:
        _vp(
            f"    _emit_cluster: drift filter "
            f"({(_t.perf_counter() - _t0) * 1000:.2f}ms) "
            f"kept={n_kept} drifted={n_drifted}"
        )
        _t0 = _t.perf_counter()

    if n_kept < min_cluster_size:
        return None

    kept_indices_pa = pa.array(kept_indices_np, type=pa.int64())
    kept_table = slice_table.take(kept_indices_pa)

    # ── Vectorized Layer-0 derep via Arrow group_by(sequence) ────
    # Replaces the prior `_layer0_derep` Python dict + blake2b loop.
    # For mega-clusters with mostly-identical sequences, this collapses
    # ~2M reads to ~100K-500K unique-sequence groups in C++-speed Arrow.
    seq_col = kept_table.column("sequence")
    seq_valid_mask = pc.is_valid(seq_col)
    seq_for_derep = kept_table.filter(seq_valid_mask).select(
        ["read_id", "sequence", "dorado_quality"]
        if has_quality
        else ["read_id", "sequence"]
    )
    rep_id: str
    rep_seq: str | None
    n_unique_out: int
    if seq_for_derep.num_rows == 0:
        # No sequenced reads — fall back to first kept read by id.
        rep_id = str(kept_table.column("read_id")[0].as_py())
        rep_seq = None
        n_unique_out = 0
    else:
        agg_specs = [("read_id", "min"), ("read_id", "count")]
        if has_quality:
            agg_specs.append(("dorado_quality", "mean"))
        derep = seq_for_derep.group_by("sequence").aggregate(agg_specs)
        counts_np = derep.column("read_id_count").to_numpy(
            zero_copy_only=False
        )
        if has_quality:
            qual_raw = derep.column("dorado_quality_mean").to_numpy(
                zero_copy_only=False
            )
            qual_np = np.nan_to_num(qual_raw, nan=0.0)
        else:
            qual_np = np.zeros(len(counts_np), dtype=np.float64)
        # Pick representative: max count, tiebreak max mean quality.
        # `np.lexsort` is ascending; primary key is the last argument.
        order = np.lexsort((qual_np, counts_np))
        best_idx = int(order[-1])
        rep_id = str(derep.column("read_id_min")[best_idx].as_py())
        rep_seq = str(derep.column("sequence")[best_idx].as_py())
        n_unique_out = int(derep.num_rows)
    if _vp is not None:
        _vp(
            f"    _emit_cluster: layer0_derep "
            f"({(_t.perf_counter() - _t0) * 1000:.2f}ms) "
            f"n_unique={n_unique_out}"
        )
        _t0 = _t.perf_counter()

    # ── Consensus (optional) ──────────────────────────────────
    consensus_seq: str | None = None
    kept_span_start_np = kept_table.column("span_start").to_numpy(
        zero_copy_only=False
    )
    kept_span_end_np = kept_table.column("span_end").to_numpy(
        zero_copy_only=False
    )
    if build_consensus_seq and genome is not None and alignment_cs is not None:
        contig_name = contig_id_to_name.get(contig_id)
        if contig_name is not None:
            if (
                contig_seq_cache is not None
                and contig_id in contig_seq_cache
            ):
                contig_seq = contig_seq_cache[contig_id]
            else:
                contig_seq = genome.sequence_of(contig_id)
            window_start = int(kept_span_start_np.min())
            window_end = int(kept_span_end_np.max())
            window_seq = contig_seq[window_start:window_end]
            aid_col = kept_table.column("alignment_id")
            rs_col = kept_table.column("ref_start")
            aid_valid_mask = pc.is_valid(aid_col)
            rs_valid_mask = pc.is_valid(rs_col)
            both_valid_mask = pc.and_(aid_valid_mask, rs_valid_mask)
            valid_count = int(pc.sum(pc.cast(both_valid_mask, pa.int64())).as_py() or 0)
            if valid_count > 0:
                aid_valid_col = aid_col.filter(both_valid_mask)
                rs_valid_col = rs_col.filter(both_valid_mask)
                member_aids = aid_valid_col.to_pylist()
                member_ref_starts = rs_valid_col.to_pylist()
                member_weights = [1.0] * len(member_aids)
                member_aids_arr = pa.array(
                    member_aids, type=pa.int64()
                )
                if (
                    alignment_cs_table is not None
                    and "cs_idx" in slice_table.column_names
                ):
                    # Pre-computed path: cs_idx column carries the row
                    # index into alignment_cs_table for each fingerprint.
                    # Per-cluster work is a small filter + take, NOT a
                    # 14M-row hash-table rebuild via pc.index_in. This
                    # single change drops consensus per cluster from
                    # ~1 sec → ~ms at PromethION scale.
                    cs_idx_col = kept_table.column("cs_idx")
                    cs_idx_valid = cs_idx_col.filter(both_valid_mask)
                    cs_strings_col = pc.take(
                        alignment_cs_table.column("cs_string"),
                        cs_idx_valid,
                    )
                    cs_for_cluster = pa.table(
                        {
                            "alignment_id": member_aids_arr,
                            "cs_string": cs_strings_col,
                        }
                    )
                elif alignment_cs_table is not None:
                    # Fallback when caller didn't pre-compute cs_idx
                    # (test fixtures, Jupyter usage). Pays the per-call
                    # index_in cost.
                    cs_indices = pc.index_in(
                        member_aids_arr,
                        alignment_cs_table.column("alignment_id"),
                    )
                    cs_strings_col = pc.take(
                        alignment_cs_table.column("cs_string"),
                        cs_indices,
                    )
                    cs_for_cluster = pa.table(
                        {
                            "alignment_id": member_aids_arr,
                            "cs_string": cs_strings_col,
                        }
                    )
                elif isinstance(alignment_cs, pa_dataset.Dataset):
                    cs_for_cluster = alignment_cs.to_table(
                        columns=["alignment_id", "cs_string"],
                        filter=pc.is_in(
                            pa_dataset.field("alignment_id"),
                            value_set=member_aids_arr,
                        ),
                    )
                else:
                    cs_for_cluster = alignment_cs
                consensus_seq = build_consensus(
                    member_alignment_ids=member_aids,
                    member_weights=member_weights,
                    member_ref_starts=member_ref_starts,
                    alignment_cs=cs_for_cluster,
                    reference_sequence=window_seq,
                    reference_start=window_start,
                    threads=int(consensus_threads),
                )
    if _vp is not None:
        _vp(
            f"    _emit_cluster: consensus "
            f"({(_t.perf_counter() - _t0) * 1000:.2f}ms) "
            f"build={build_consensus_seq}"
        )
        _t0 = _t.perf_counter()

    # ── Cluster row ──────────────────────────────────────────
    cluster_row = {
        "cluster_id": int(cluster_id),
        "representative_read_id": str(rep_id),
        "n_reads": int(n_kept),
        "identity_threshold": None,
        "consensus_sequence": consensus_seq,
        "predicted_protein": None,
        "orf_start": None,
        "orf_end": None,
        "orf_strand": None,
        "codon_table": None,
        "mode": "genome-guided",
        "contig_id": contig_id,
        "strand": strand,
        "span_start": int(kept_span_start_np.min()),
        "span_end": int(kept_span_end_np.max()),
        "fingerprint_hash": int(fp_hash),
        "n_unique_sequences": int(n_unique_out) if n_unique_out > 0 else 1,
        "sample_id": (None if sample_id_raw is None else int(sample_id_raw)),
    }

    # ── Vectorized membership table construction ─────────────
    # kept members: role = representative | duplicate | member.
    # drifted members (kept iff not drop_drift_filtered): role = drift_filtered.
    kept_read_id_col = kept_table.column("read_id")
    if rep_seq is not None:
        is_rep_mask = pc.equal(
            kept_read_id_col, pa.scalar(rep_id, type=pa.string())
        )
        is_dup_mask = pc.and_(
            pc.equal(
                kept_table.column("sequence"),
                pa.scalar(rep_seq, type=pa.string()),
            ),
            pc.invert(is_rep_mask),
        )
        is_rep_np = is_rep_mask.to_numpy(zero_copy_only=False).astype(bool)
        is_dup_np = pc.fill_null(is_dup_mask, False).to_numpy(
            zero_copy_only=False
        ).astype(bool)
    else:
        # No sequences → no duplicates possible; rep tagged by id only.
        kept_read_ids_py = kept_read_id_col.to_pylist()
        is_rep_np = np.array(
            [rid == rep_id for rid in kept_read_ids_py], dtype=bool
        )
        is_dup_np = np.zeros(n_kept, dtype=bool)
    kept_roles_np = np.where(
        is_rep_np,
        "representative",
        np.where(is_dup_np, "duplicate", "member"),
    )
    # Drift columns: null on rep/dup rows AND on singletons (n_total==1).
    if n_total == 1:
        kept_drift_null_mask = np.ones(n_kept, dtype=bool)
    else:
        kept_drift_null_mask = ~(kept_roles_np == "member")
    kept_drift_5p_vals = drift_5p_np[kept_indices_np].astype(np.int32)
    kept_drift_3p_vals = drift_3p_np[kept_indices_np].astype(np.int32)
    kept_drift_5p_pa = pa.array(
        kept_drift_5p_vals, type=pa.int32(), mask=kept_drift_null_mask
    )
    kept_drift_3p_pa = pa.array(
        kept_drift_3p_vals, type=pa.int32(), mask=kept_drift_null_mask
    )
    kept_roles_pa = pa.array(kept_roles_np.tolist(), type=pa.string())
    kept_cluster_id_pa = pa.array(
        np.full(n_kept, int(cluster_id), dtype=np.int64), type=pa.int64()
    )
    kept_match_rate, kept_indel_rate, kept_n_aligned = _build_metric_cols(
        kept_table, has_metrics=has_metrics, n_rows=n_kept
    )
    kept_membership = pa.Table.from_arrays(
        [
            kept_cluster_id_pa,
            kept_read_id_col,
            kept_roles_pa,
            kept_drift_5p_pa,
            kept_drift_3p_pa,
            kept_match_rate,
            kept_indel_rate,
            kept_n_aligned,
        ],
        schema=CLUSTER_MEMBERSHIP_TABLE,
    )

    if not drop_drift_filtered and n_drifted > 0:
        drifted_indices_pa = pa.array(drifted_indices_np, type=pa.int64())
        drifted_table = slice_table.take(drifted_indices_pa)
        drifted_read_id_col = drifted_table.column("read_id")
        drifted_roles_pa = pa.array(
            ["drift_filtered"] * n_drifted, type=pa.string()
        )
        drifted_cluster_id_pa = pa.array(
            np.full(n_drifted, int(cluster_id), dtype=np.int64),
            type=pa.int64(),
        )
        drifted_drift_5p_pa = pa.array(
            drift_5p_np[drifted_indices_np].astype(np.int32),
            type=pa.int32(),
        )
        drifted_drift_3p_pa = pa.array(
            drift_3p_np[drifted_indices_np].astype(np.int32),
            type=pa.int32(),
        )
        d_match, d_indel, d_naligned = _build_metric_cols(
            drifted_table, has_metrics=has_metrics, n_rows=n_drifted
        )
        drifted_membership = pa.Table.from_arrays(
            [
                drifted_cluster_id_pa,
                drifted_read_id_col,
                drifted_roles_pa,
                drifted_drift_5p_pa,
                drifted_drift_3p_pa,
                d_match,
                d_indel,
                d_naligned,
            ],
            schema=CLUSTER_MEMBERSHIP_TABLE,
        )
        membership_table = pa.concat_tables(
            [kept_membership, drifted_membership]
        )
    else:
        membership_table = kept_membership

    if _vp is not None:
        _vp(
            f"    _emit_cluster: membership table "
            f"({(_t.perf_counter() - _t0) * 1000:.2f}ms) "
            f"n_members={membership_table.num_rows}"
        )

    return cluster_row, membership_table


__all__ = ["cluster_by_fingerprint"]
