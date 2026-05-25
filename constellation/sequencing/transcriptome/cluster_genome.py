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

import hashlib
import statistics
from typing import Any

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


_LAYER0_DIGEST_SIZE = 8


def _seq_hash(sequence: str) -> int:
    """blake2b-8 of an ASCII trimmed-window sequence."""
    h = hashlib.blake2b(digest_size=_LAYER0_DIGEST_SIZE)
    h.update(sequence.encode("ascii", errors="replace"))
    return int.from_bytes(h.digest(), "little", signed=False)


def _strand_signed_drift(
    span_start: int,
    span_end: int,
    median_start: float,
    median_end: float,
    strand: str,
) -> tuple[int, int]:
    """Strand-aware (5'_drift_bp, 3'_drift_bp).

    On ``+`` strand: 5' = span_start, 3' = span_end.
    On ``-`` strand: 5' = span_end, 3' = span_start.
    Sign is read - median (positive = read extends further than the
    median in that orientation). On ``?`` strand we treat as ``+``;
    minimap2 in splice-aware mode uses ``+``/``-``, so this only
    matters on assembly-vs-genome runs without splice-site disambiguation.
    """
    if strand == "-":
        drift_5p = int(span_end - median_end)
        drift_3p = int(span_start - median_start)
    else:
        drift_5p = int(span_start - median_start)
        drift_3p = int(span_end - median_end)
    return drift_5p, drift_3p


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


def _layer0_derep(
    read_ids: list[str],
    sequences: dict[str, str],
    qualities: dict[str, float],
) -> tuple[str, int, dict[str, int], dict[str, int]]:
    """Pick the representative read by Layer-0 derep.

    Returns ``(rep_read_id, n_unique_sequences, hash_of_read,
    count_per_hash)``. ``hash_of_read`` maps each read_id to its
    sequence hash so the caller can role-tag duplicates. Ties on
    abundance break on mean ``dorado_quality``. If a read has no
    sequence in ``sequences``, it's skipped (the cluster's
    representative comes from the surviving sequenced reads).
    """
    hash_of_read: dict[str, int] = {}
    count_per_hash: dict[int, int] = {}
    quality_sum_per_hash: dict[int, float] = {}
    quality_count_per_hash: dict[int, int] = {}
    exemplar_per_hash: dict[int, str] = {}
    for rid in read_ids:
        seq = sequences.get(rid)
        if seq is None or not seq:
            continue
        h = _seq_hash(seq)
        hash_of_read[rid] = h
        count_per_hash[h] = count_per_hash.get(h, 0) + 1
        q = qualities.get(rid)
        if q is not None:
            quality_sum_per_hash[h] = quality_sum_per_hash.get(h, 0.0) + float(q)
            quality_count_per_hash[h] = quality_count_per_hash.get(h, 0) + 1
        exemplar_per_hash.setdefault(h, rid)
    if not count_per_hash:
        # No sequences available — fall back to first read by id.
        return read_ids[0], 0, hash_of_read, {}
    # Pick rep: max count, tiebreak by mean quality.
    def _score(h: int) -> tuple[int, float]:
        cnt = count_per_hash[h]
        if quality_count_per_hash.get(h, 0) > 0:
            mean_q = quality_sum_per_hash[h] / quality_count_per_hash[h]
        else:
            mean_q = 0.0
        return (cnt, mean_q)
    best_hash = max(count_per_hash.keys(), key=_score)
    rep_read_id = exemplar_per_hash[best_hash]
    n_unique = len(count_per_hash)
    # Map hash → str-keyed count for caller convenience (they don't
    # care about the int hash, just the abundances per role tag).
    count_per_hash_str = {str(h): cnt for h, cnt in count_per_hash.items()}
    return rep_read_id, n_unique, hash_of_read, count_per_hash_str


def _make_membership_row(
    *,
    cluster_id: int,
    read_id: str,
    role: str,
    drift_5p_bp: int | None,
    drift_3p_bp: int | None,
    metrics: dict[str, int] | None,
) -> dict[str, Any]:
    if metrics is None:
        match_rate = None
        indel_rate = None
        n_aligned = 0
    else:
        n_match = int(metrics.get("n_match", 0))
        n_mismatch = int(metrics.get("n_mismatch", 0))
        n_insert = int(metrics.get("n_insert", 0))
        n_delete = int(metrics.get("n_delete", 0))
        n_aligned = n_match + n_mismatch + n_insert + n_delete
        cs_aware = bool(metrics.get("cs_aware", False))
        if cs_aware and (n_match + n_mismatch) > 0:
            match_rate = float(n_match) / float(n_match + n_mismatch)
        else:
            match_rate = None
        if cs_aware and n_aligned > 0:
            indel_rate = float(n_insert + n_delete) / float(n_aligned)
        else:
            indel_rate = None
    return {
        "cluster_id": int(cluster_id),
        "read_id": str(read_id),
        "role": role,
        "drift_5p_bp": (None if drift_5p_bp is None else int(drift_5p_bp)),
        "drift_3p_bp": (None if drift_3p_bp is None else int(drift_3p_bp)),
        "match_rate": match_rate,
        "indel_rate": indel_rate,
        "n_aligned_bp": int(n_aligned),
    }


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
    primary_lookup = _build_alignment_lookup(alignments)
    metrics_table: pa.Table | None = (
        _build_alignment_metrics(alignment_blocks)
        if alignment_blocks is not None
        else None
    )

    n = fingerprints.num_rows
    fp_read_ids = fingerprints.column("read_id")
    has_quality = "dorado_quality" in reads.schema.names

    # Row indices into each source. Nulls propagate where there's no match.
    reads_idx = pc.index_in(fp_read_ids, reads.column("read_id"))
    primary_idx = pc.index_in(fp_read_ids, primary_lookup.column("read_id"))
    if metrics_table is not None:
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
        rts_idx = pc.index_in(fp_read_ids, read_to_sample.column("read_id"))
        sample_id_col = pc.take(
            read_to_sample.column("sample_id"), rts_idx
        )
    else:
        sample_id_col = pa.nulls(n, type=pa.int64())
    sample_id_for_sort = pc.fill_null(sample_id_col, _SAMPLE_NULL_SENTINEL)

    # contig_id → name lookup (small — keep as dict for consensus path).
    contig_id_to_name: dict[int, str] = {}
    if genome is not None:
        for cid, name in zip(
            genome.contigs.column("contig_id").to_pylist(),
            genome.contigs.column("name").to_pylist(),
            strict=True,
        ):
            contig_id_to_name[int(cid)] = str(name)

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
    narrow_keys = pa.table(narrow_cols)

    sorted_keys = narrow_keys.sort_by(
        [
            ("contig_id", "ascending"),
            ("strand", "ascending"),
            ("sample_id", "ascending"),
            ("fingerprint_hash", "ascending"),
        ]
    ).combine_chunks()
    boundaries = _cluster_boundaries(sorted_keys)

    # ── Per-cluster assembly + emission ─────────────────────────
    cluster_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    next_cluster_id = int(cluster_id_seed)

    sample_null_sentinel_scalar = pa.scalar(
        _SAMPLE_NULL_SENTINEL, type=pa.int64()
    )

    for start, size in boundaries:
        # Per-cluster slice of sort keys + source-row indices.
        row_indices = sorted_keys.column("row_idx").slice(start, size)
        reads_indices = sorted_keys.column("reads_idx").slice(start, size)
        primary_indices = sorted_keys.column("primary_idx").slice(start, size)
        sample_filled_slice = sorted_keys.column("sample_id").slice(
            start, size
        )

        # Take from each source. ``pc.take`` propagates nulls.
        fp_slice = fingerprints.take(row_indices)
        if reads.num_rows > 0:
            seq_col = pc.take(reads.column("sequence"), reads_indices)
            if has_quality:
                quality_col = pc.take(
                    reads.column("dorado_quality"), reads_indices
                )
            else:
                quality_col = pa.nulls(size, type=pa.float32())
        else:
            seq_col = pa.nulls(size, type=pa.string())
            quality_col = pa.nulls(size, type=pa.float32())

        aid_col = pc.take(
            primary_lookup.column("alignment_id"), primary_indices
        )
        rstart_col = pc.take(
            primary_lookup.column("ref_start"), primary_indices
        )

        # Un-sentinel sample_id for the emitted slice.
        sample_col = pc.if_else(
            pc.equal(sample_filled_slice, sample_null_sentinel_scalar),
            pa.scalar(None, type=pa.int64()),
            sample_filled_slice,
        )

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
        if metrics_table is not None:
            metrics_indices = sorted_keys.column("metrics_idx").slice(
                start, size
            )
            slice_cols["n_match_sum"] = pc.take(
                metrics_table.column("n_match_sum"), metrics_indices
            )
            slice_cols["n_mismatch_sum"] = pc.take(
                metrics_table.column("n_mismatch_sum"), metrics_indices
            )
            slice_cols["n_insert_sum"] = pc.take(
                metrics_table.column("n_insert_sum"), metrics_indices
            )
            slice_cols["n_delete_sum"] = pc.take(
                metrics_table.column("n_delete_sum"), metrics_indices
            )
            slice_cols["cs_aware"] = pc.take(
                metrics_table.column("cs_aware"), metrics_indices
            )
        slice_table = pa.table(slice_cols)

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
        )
        if emitted is None:
            continue
        cluster_row, members_rows = emitted
        cluster_rows.append(cluster_row)
        membership_rows.extend(members_rows)
        next_cluster_id += 1

    if not cluster_rows:
        return (
            TRANSCRIPT_CLUSTER_TABLE.empty_table(),
            CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        )
    clusters = pa.Table.from_pylist(cluster_rows, schema=TRANSCRIPT_CLUSTER_TABLE)
    membership = pa.Table.from_pylist(
        membership_rows, schema=CLUSTER_MEMBERSHIP_TABLE
    )
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
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Per-cluster work: drift filter + Layer-0 derep + optional
    consensus + cluster row + membership rows.

    ``slice_table`` is a small Arrow slice (typically tens of rows)
    covering one ``(contig, strand, sample, fingerprint)`` cluster. We
    ``to_pylist()`` it once — bounded by cluster size, not by total
    read count — and the rest of the per-cluster work is row-shaped by
    construction (median, hashing, representative selection).
    """
    members = slice_table.to_pylist()
    n_total = len(members)
    if n_total == 0:
        return None
    contig_id = int(members[0]["contig_id"])
    strand = str(members[0]["strand"])
    sample_id_raw = members[0]["sample_id"]
    fp_hash = int(members[0]["fingerprint_hash"])

    # Drift filter (singletons skip — no median to compare).
    if n_total > 1:
        med_start = statistics.median(int(m["span_start"]) for m in members)
        med_end = statistics.median(int(m["span_end"]) for m in members)
        kept: list[dict[str, Any]] = []
        drifted: list[dict[str, Any]] = []
        for m in members:
            drift_5p, drift_3p = _strand_signed_drift(
                int(m["span_start"]),
                int(m["span_end"]),
                med_start,
                med_end,
                strand,
            )
            m["_drift_5p"] = drift_5p
            m["_drift_3p"] = drift_3p
            if abs(drift_5p) > max_5p_drift or abs(drift_3p) > max_3p_drift:
                drifted.append(m)
            else:
                kept.append(m)
    else:
        kept = members
        drifted = []
        for m in members:
            m["_drift_5p"] = None
            m["_drift_3p"] = None

    if len(kept) < min_cluster_size:
        return None

    # Layer-0 derep — sequences and qualities are now columns on the
    # enriched slice, not external dicts. Use _layer0_derep with
    # adapter dicts built per-cluster (bounded by cluster size).
    kept_ids = [str(m["read_id"]) for m in kept]
    seq_lookup: dict[str, str] = {}
    for m in kept:
        seq = m.get("sequence")
        if seq is not None:
            seq_lookup[str(m["read_id"])] = str(seq)
    quality_lookup: dict[str, float] = {}
    if has_quality:
        for m in kept:
            q = m.get("dorado_quality")
            if q is not None:
                quality_lookup[str(m["read_id"])] = float(q)
    rep_id, n_unique, hash_of_read, _counts = _layer0_derep(
        kept_ids, seq_lookup, quality_lookup
    )

    consensus_seq: str | None = None
    kept_starts_list = [int(m["span_start"]) for m in kept]
    kept_ends_list = [int(m["span_end"]) for m in kept]
    if build_consensus_seq and genome is not None and alignment_cs is not None:
        window_start = min(kept_starts_list)
        window_end = max(kept_ends_list)
        contig_name = contig_id_to_name.get(contig_id)
        if contig_name is not None:
            contig_seq = genome.sequence_of(contig_id)
            window_seq = contig_seq[window_start:window_end]
            member_aids: list[int] = []
            member_weights: list[float] = []
            member_ref_starts: list[int] = []
            for m in kept:
                aid = m.get("alignment_id")
                rs = m.get("ref_start")
                if aid is None or rs is None:
                    continue
                member_aids.append(int(aid))
                member_weights.append(1.0)
                member_ref_starts.append(int(rs))
            if member_aids:
                # Dataset handle → materialise only this cluster's rows
                # via filter pushdown. ``alignment_cs`` at PromethION
                # scale is ~50 GB; the per-cluster slice is KB. Keeping
                # the dataset open instead of pre-materialising the
                # whole table is what makes ``--build-consensus``
                # tractable at full scale.
                if isinstance(alignment_cs, pa_dataset.Dataset):
                    cs_for_cluster = alignment_cs.to_table(
                        columns=["alignment_id", "cs_string"],
                        filter=pc.is_in(
                            pa_dataset.field("alignment_id"),
                            value_set=pa.array(
                                member_aids, type=pa.int64()
                            ),
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
                )

    cluster_row = {
        "cluster_id": int(cluster_id),
        "representative_read_id": str(rep_id),
        "n_reads": int(len(kept)),
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
        "span_start": int(min(kept_starts_list)),
        "span_end": int(max(kept_ends_list)),
        "fingerprint_hash": int(fp_hash),
        "n_unique_sequences": int(n_unique) if n_unique > 0 else 1,
        "sample_id": (None if sample_id_raw is None else int(sample_id_raw)),
    }

    membership_rows: list[dict[str, Any]] = []
    rep_seq_hash = hash_of_read.get(rep_id)
    for m in kept:
        rid = str(m["read_id"])
        role = "member"
        if rid == rep_id:
            role = "representative"
        elif (
            rep_seq_hash is not None and hash_of_read.get(rid) == rep_seq_hash
        ):
            role = "duplicate"
        metrics = _metrics_from_member_row(m) if has_metrics else None
        drift_5p = m.get("_drift_5p") if role == "member" else None
        drift_3p = m.get("_drift_3p") if role == "member" else None
        membership_rows.append(
            _make_membership_row(
                cluster_id=cluster_id,
                read_id=rid,
                role=role,
                drift_5p_bp=drift_5p,
                drift_3p_bp=drift_3p,
                metrics=metrics,
            )
        )

    if not drop_drift_filtered:
        for m in drifted:
            rid = str(m["read_id"])
            metrics = _metrics_from_member_row(m) if has_metrics else None
            membership_rows.append(
                _make_membership_row(
                    cluster_id=cluster_id,
                    read_id=rid,
                    role="drift_filtered",
                    drift_5p_bp=m.get("_drift_5p"),
                    drift_3p_bp=m.get("_drift_3p"),
                    metrics=metrics,
                )
            )

    return cluster_row, membership_rows


def _metrics_from_member_row(
    member: dict[str, Any],
) -> dict[str, Any] | None:
    """Repack the joined-in metric columns into the ``metrics`` dict
    that :func:`_make_membership_row` expects. Returns ``None`` when
    the alignment had no metric rows joined (e.g. unmapped read or
    metrics_table not supplied).
    """
    aid = member.get("alignment_id")
    if aid is None:
        return None
    cs_aware = member.get("cs_aware")
    if cs_aware is None:
        return None
    return {
        "n_match": int(member.get("n_match_sum") or 0),
        "n_mismatch": int(member.get("n_mismatch_sum") or 0),
        "n_insert": int(member.get("n_insert_sum") or 0),
        "n_delete": int(member.get("n_delete_sum") or 0),
        "cs_aware": bool(cs_aware),
    }


__all__ = ["cluster_by_fingerprint"]
