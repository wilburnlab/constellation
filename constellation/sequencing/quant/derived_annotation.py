"""Data-informed exon discovery + gene rollup from clustered introns.

Given the canonical ``INTRON_TABLE`` (post-:func:`cluster_junctions`)
and the per-position ``COVERAGE_TABLE``, derive an ``Annotation`` whose
``FEATURE_TABLE`` contains data-supported gene + exon features. Both
inputs originate from the always-on resolve-stage outputs of
``transcriptome align``; the function chain is:

    aggregate_junctions  →  cluster_junctions  →  INTRON_TABLE
                                                          \\
    build_pileup  →  COVERAGE_TABLE                        \\
                            \\                              \\
                             ▼                               ▼
                              build_derived_annotation(...)
                                              │
                                              ▼
                                      (Annotation,
                                       BLOCK_EXON_ASSIGNMENT_TABLE,
                                       EXON_PSI_TABLE)

Granularity contract:
- Exons are minimal segments — each cut introduced by a trusted intron
  donor or acceptor inside a covered region produces a new exon
  segment. Reads agreeing on splice topology share segment membership.
- Boundary calls are threshold-driven: a position is a boundary iff
  it appears as donor_pos or acceptor_pos of an intron whose total
  cluster ``read_count`` (summed across all member position pairs)
  meets ``min_intron_read_count``.
- Gene rollup is connected-components over exons, where edges follow
  trusted-intron splice junctions (donor==exon_a.end AND
  acceptor==exon_b.start, same contig + strand).
- Coverage thresholds the candidate exon body but does NOT contribute
  to boundary determination — basecaller noise on coverage doesn't
  introduce false boundaries.

v1 limitations (deferred to v2):
- Pure single-exon genes (no trusted introns) are not discovered.
  Multi-exon genes with at least one trusted intron pull in their
  flanking exons as expected.
- Coverage is unstranded (build_pileup does not stratify by alignment
  strand); per-strand exon derivation uses unstranded coverage as a
  depth-thresholding substrate. Loci with reads on both strands at
  similar depth could over-emit. v2 lifts this with stranded pile-up.
- Differential exon-abundance (negbin LRT across samples) is deferred;
  v1 emits the per-(exon, sample) inclusion / exclusion / PSI substrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq
import torch

from constellation.core.graph.network import Network
from constellation.sequencing.schemas.alignment import (
    BLOCK_EXON_ASSIGNMENT_TABLE,
)
from constellation.sequencing.schemas.quant import EXON_PSI_TABLE
from constellation.sequencing.schemas.reference import FEATURE_TABLE


# Type alias for callers that pass either an in-memory table or a path
# to a Parquet dataset (file or directory). Tests + Jupyter callers pass
# a table; the CLI passes a path so the resolve stage never holds the
# full BLOCK_EXON_ASSIGNMENT_TABLE in RAM.
BlockAssignmentsLike = Union[pa.Table, Path, str]


# Sentinel sample_id when no read_to_sample mapping is supplied or a
# read has no resolved sample. Matches COVERAGE_TABLE convention.
_UNSTRATIFIED_SAMPLE_ID: int = -1


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _trusted_seed_rows(
    introns: pa.Table, *, min_intron_read_count: int
) -> pa.Table:
    """Return seed rows of intron clusters whose total read_count
    meets the threshold.

    Filters in two steps:
      1. ``group_by(intron_id).sum(read_count)`` → trusted intron_ids.
      2. ``filter(is_intron_seed=True AND intron_id IN trusted_ids)``.
    """
    if introns.num_rows == 0:
        return introns.schema.empty_table()
    totals = introns.group_by("intron_id").aggregate(
        [("read_count", "sum")]
    )
    trusted_ids_arr = totals.filter(
        pc.greater_equal(
            totals.column("read_count_sum"),
            int(min_intron_read_count),
        )
    ).column("intron_id")
    seeds = introns.filter(pc.equal(introns.column("is_intron_seed"), True))
    return seeds.filter(pc.is_in(seeds.column("intron_id"), trusted_ids_arr))


def _coalesce_covered_intervals(
    coverage: pa.Table, *, min_exon_depth: int
) -> dict[int, list[tuple[int, int]]]:
    """Per-contig list of [start, end) intervals with depth >= threshold.

    Coalesces abutting / overlapping intervals across sample partitions —
    a region covered at depth N in sample A and depth M in sample B
    both pass the threshold and merge into one covered span.
    """
    if coverage.num_rows == 0:
        return {}
    filt = coverage.filter(
        pc.greater_equal(coverage.column("depth"), int(min_exon_depth))
    )
    if filt.num_rows == 0:
        return {}
    contig_ids = filt.column("contig_id").to_pylist()
    starts = filt.column("start").to_pylist()
    ends = filt.column("end").to_pylist()
    by_contig: dict[int, list[tuple[int, int]]] = {}
    for c, s, e in zip(contig_ids, starts, ends, strict=True):
        by_contig.setdefault(int(c), []).append((int(s), int(e)))
    coalesced: dict[int, list[tuple[int, int]]] = {}
    for c, ivs in by_contig.items():
        ivs.sort()
        merged: list[tuple[int, int]] = []
        cur_start, cur_end = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_end:  # abutting or overlapping
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
        coalesced[c] = merged
    return coalesced


def _empty_features_table() -> pa.Table:
    return FEATURE_TABLE.empty_table()


def _empty_block_assignments() -> pa.Table:
    return BLOCK_EXON_ASSIGNMENT_TABLE.empty_table()


def _empty_exon_psi() -> pa.Table:
    return EXON_PSI_TABLE.empty_table()


# ──────────────────────────────────────────────────────────────────────
# derive_exons
# ──────────────────────────────────────────────────────────────────────


def derive_exons(
    coverage: pa.Table,
    introns: pa.Table,
    contigs: pa.Table,
    *,
    min_exon_depth: int = 5,
    min_intron_read_count: int = 3,
) -> pa.Table:
    """Build a FEATURE_TABLE-shaped ``type='exon'`` table from coverage
    + clustered introns.

    Parameters
    ----------
    coverage : pa.Table
        ``COVERAGE_TABLE``-shaped — RLE depth per (contig, sample).
    introns : pa.Table
        ``INTRON_TABLE``-shaped — clustered intron site evidence.
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — accepted as a placeholder for v1.5
        per-strand validation; not used in v1.
    min_exon_depth : int, default 5
        Minimum coverage depth (per RLE row) required for a region
        to be considered as part of a candidate exon.
    min_intron_read_count : int, default 3
        Minimum total reads supporting an intron cluster (summed across
        cluster member positions) to treat it as a trusted boundary.

    Returns
    -------
    pa.Table conforming to ``FEATURE_TABLE`` with one row per derived
    exon (``type='exon'``). ``feature_id`` is assigned sequentially
    starting from 0; ``parent_id`` is null and gets populated by
    :func:`roll_up_genes`. ``source='constellation_derived'``.
    """
    del contigs  # reserved for v1.5 stranded coverage validation

    if introns.num_rows == 0 or coverage.num_rows == 0:
        return _empty_features_table()

    trusted_seeds = _trusted_seed_rows(
        introns, min_intron_read_count=min_intron_read_count
    )
    if trusted_seeds.num_rows == 0:
        # No trusted introns → no boundaries → no exons in v1.
        return _empty_features_table()

    coalesced = _coalesce_covered_intervals(
        coverage, min_exon_depth=min_exon_depth
    )
    if not coalesced:
        return _empty_features_table()

    # Group trusted introns by (contig_id, strand). Keep both the set
    # of (donor, acceptor) tuples (for the "drop pieces == intron"
    # rule) and the set of cut positions (donors ∪ acceptors).
    intron_pairs_by_partition: dict[tuple[int, str], set[tuple[int, int]]] = {}
    cuts_by_partition: dict[tuple[int, str], set[int]] = {}
    for r in trusted_seeds.to_pylist():
        key = (int(r["contig_id"]), str(r["strand"]))
        d = int(r["donor_pos"])
        a = int(r["acceptor_pos"])
        intron_pairs_by_partition.setdefault(key, set()).add((d, a))
        cuts = cuts_by_partition.setdefault(key, set())
        cuts.add(d)
        cuts.add(a)

    exon_records: list[dict] = []
    next_feature_id = 0

    # Iterate per (contig, strand). For each covered interval on the
    # contig, segment using cuts of that strand's intron set; drop
    # pieces whose [start, end) exactly matches a trusted intron's span.
    for (contig_id, strand) in sorted(intron_pairs_by_partition.keys()):
        intron_pairs = intron_pairs_by_partition[(contig_id, strand)]
        cut_positions = cuts_by_partition[(contig_id, strand)]
        intervals = coalesced.get(contig_id, [])
        for iv_start, iv_end in intervals:
            interior_cuts = sorted(
                p for p in cut_positions if iv_start < p < iv_end
            )
            edges = [iv_start] + interior_cuts + [iv_end]
            for i in range(len(edges) - 1):
                p_start = edges[i]
                p_end = edges[i + 1]
                if p_end <= p_start:
                    continue
                if (p_start, p_end) in intron_pairs:
                    # This piece exactly matches a trusted intron span
                    # — drop. Pieces inside larger introns (e.g. nested
                    # alt-3'SS subspans) are NOT dropped; they may be
                    # exons in some other isoform.
                    continue
                exon_records.append(
                    {
                        "feature_id": next_feature_id,
                        "contig_id": int(contig_id),
                        "start": int(p_start),
                        "end": int(p_end),
                        "strand": str(strand),
                        "type": "exon",
                        "name": None,
                        "parent_id": None,  # set by roll_up_genes
                        "source": "constellation_derived",
                        "score": None,
                        "phase": None,
                        "attributes_json": None,
                    }
                )
                next_feature_id += 1

    if not exon_records:
        return _empty_features_table()
    return pa.Table.from_pylist(exon_records, schema=FEATURE_TABLE)


# ──────────────────────────────────────────────────────────────────────
# roll_up_genes
# ──────────────────────────────────────────────────────────────────────


def roll_up_genes(
    exons: pa.Table,
    introns: pa.Table,
    *,
    min_intron_read_count: int = 3,
) -> pa.Table:
    """Combine exons + connected-component-derived genes into a single
    FEATURE_TABLE.

    Parameters
    ----------
    exons : pa.Table
        ``FEATURE_TABLE``-shaped, ``type='exon'`` only — the output of
        :func:`derive_exons`.
    introns : pa.Table
        ``INTRON_TABLE``-shaped — used to build splice-junction edges
        between exons. Only trusted intron seeds (above
        ``min_intron_read_count``) contribute edges.
    min_intron_read_count : int, default 3
        Same threshold as :func:`derive_exons`.

    Returns
    -------
    pa.Table conforming to ``FEATURE_TABLE`` with ``type='gene'`` rows
    appended to the input exons. Each exon's ``parent_id`` is rewritten
    to point at its gene's ``feature_id``. Gene ``feature_id``s are
    assigned starting after the highest exon ``feature_id``.
    """
    if exons.num_rows == 0:
        return _empty_features_table()

    trusted_seeds = _trusted_seed_rows(
        introns, min_intron_read_count=min_intron_read_count
    )

    # Index exons by (contig, strand, end) for donor matching and by
    # (contig, strand, start) for acceptor matching.
    exon_meta: dict[int, dict] = {
        int(r["feature_id"]): r for r in exons.to_pylist()
    }
    exon_by_end: dict[tuple[int, str, int], list[int]] = {}
    exon_by_start: dict[tuple[int, str, int], list[int]] = {}
    for fid, r in exon_meta.items():
        cid = int(r["contig_id"])
        strand = str(r["strand"])
        exon_by_end.setdefault((cid, strand, int(r["end"])), []).append(fid)
        exon_by_start.setdefault((cid, strand, int(r["start"])), []).append(fid)

    edge_pairs: list[tuple[int, int]] = []
    for r in trusted_seeds.to_pylist():
        cid = int(r["contig_id"])
        strand = str(r["strand"])
        d = int(r["donor_pos"])
        a = int(r["acceptor_pos"])
        srcs = exon_by_end.get((cid, strand, d), [])
        dsts = exon_by_start.get((cid, strand, a), [])
        for src in srcs:
            for dst in dsts:
                if src != dst:
                    edge_pairs.append((src, dst))

    nodes_table = pa.table(
        {"id": pa.array(list(exon_meta.keys()), type=pa.int64())}
    )
    edges_table = pa.table(
        {
            "src": pa.array(
                [int(e[0]) for e in edge_pairs], type=pa.int64()
            ),
            "dst": pa.array(
                [int(e[1]) for e in edge_pairs], type=pa.int64()
            ),
        }
    )
    net = Network(nodes_table, edges_table, directed=False)
    components = net.connected_components()

    max_exon_id = (
        max(exon_meta.keys()) if exon_meta else -1
    )
    next_gene_id = max_exon_id + 1

    gene_records: list[dict] = []
    exon_to_gene: dict[int, int] = {}
    for component in components:
        gene_id = next_gene_id
        next_gene_id += 1
        first_exon = exon_meta[int(component[0])]
        contig_id = int(first_exon["contig_id"])
        strand = str(first_exon["strand"])
        starts = [int(exon_meta[int(e)]["start"]) for e in component]
        ends = [int(exon_meta[int(e)]["end"]) for e in component]
        gene_records.append(
            {
                "feature_id": int(gene_id),
                "contig_id": contig_id,
                "start": int(min(starts)),
                "end": int(max(ends)),
                "strand": strand,
                "type": "gene",
                "name": None,
                "parent_id": None,
                "source": "constellation_derived",
                "score": None,
                "phase": None,
                "attributes_json": None,
            }
        )
        for e in component:
            exon_to_gene[int(e)] = int(gene_id)

    # Update exon parent_ids and emit combined table.
    new_exon_records: list[dict] = []
    for r in exons.to_pylist():
        new_r = dict(r)
        new_r["parent_id"] = exon_to_gene.get(int(r["feature_id"]))
        new_exon_records.append(new_r)

    combined = gene_records + new_exon_records
    combined.sort(key=lambda r: int(r["feature_id"]))
    return pa.Table.from_pylist(combined, schema=FEATURE_TABLE)


# ──────────────────────────────────────────────────────────────────────
# assign_blocks_to_exons
# ──────────────────────────────────────────────────────────────────────


def assign_blocks_to_exons(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    derived_features: pa.Table,
    contigs: pa.Table,
    *,
    output_path: Path | None = None,
    batch_size: int = 10_000_000,
) -> pa.Table | dict[str, int]:
    """Build ``BLOCK_EXON_ASSIGNMENT_TABLE`` — long-form M:N edge table
    between alignment blocks and derived exons.

    Arrow joins resolve ``(alignment_id → ref_name → contig_id)``
    once; per-contig the algorithm bridges to PyTorch CPU tensors and
    uses ``torch.searchsorted`` + ``torch.repeat_interleave`` to
    vectorise the interval-overlap expansion over millions of blocks.
    Replaces the pre-refactor 1.6B-row ``to_pylist()`` + nested Python
    loop that pinned a single thread for hours at mouse scale.

    Parameters
    ----------
    alignment_blocks : pa.Table
        ``ALIGNMENT_BLOCK_TABLE``-shaped.
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped — supplies ``ref_name`` for the
        contig lookup. Secondary / supplementary alignments are
        excluded so each block contributes at most once per exon.
    derived_features : pa.Table
        ``FEATURE_TABLE``-shaped output of :func:`roll_up_genes`. Only
        ``type='exon'`` rows are used.
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — supplies the ``ref_name → contig_id``
        resolution.
    output_path : Path | None, default None
        If provided, stream per-contig per-batch RecordBatches to this
        Parquet file and return a summary stats dict. If omitted,
        accumulate batches and return them as a single in-memory table
        (tests + Jupyter path).
    batch_size : int, default 10_000_000
        Blocks per inner vectorisation batch. Per-batch intermediate
        memory ≈ ``batch_size × avg_candidates_per_block × 7 columns
        × 8 bytes`` (~1.7 GB at the default with 3 avg candidates).

    Returns
    -------
    pa.Table conforming to ``BLOCK_EXON_ASSIGNMENT_TABLE`` (when
    ``output_path`` is None), or ``dict[str, int]`` summary stats
    ``{"n_rows_written": int, "n_contigs": int}`` (when streaming).
    One row per ``(alignment_id, block_index, data_exon_id)`` overlap
    edge with ``overlap_bp > 0``.
    """
    if alignment_blocks.num_rows == 0 or derived_features.num_rows == 0:
        return _finalise_empty_block_assignments(output_path)

    exons = derived_features.filter(
        pc.equal(derived_features.column("type"), "exon")
    ).select(["feature_id", "contig_id", "start", "end"])
    if exons.num_rows == 0:
        return _finalise_empty_block_assignments(output_path)

    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    primary = alignments.filter(primary_mask).select(
        ["alignment_id", "ref_name"]
    )

    # Attach contig_id to every primary alignment, then propagate to
    # every block via the join chain.
    contig_lookup = contigs.select(["contig_id", "name"]).rename_columns(
        ["contig_id", "ref_name"]
    )
    primary_with_contig = primary.join(
        contig_lookup, keys="ref_name", join_type="inner"
    ).select(["alignment_id", "contig_id"])
    blocks_with_contig = alignment_blocks.join(
        primary_with_contig, keys="alignment_id", join_type="inner"
    )
    if blocks_with_contig.num_rows == 0:
        return _finalise_empty_block_assignments(output_path)
    blocks_with_contig = blocks_with_contig.combine_chunks()

    writer: pq.ParquetWriter | None = None
    out_batches: list[pa.RecordBatch] = []
    n_rows_written = 0
    n_contigs = 0
    try:
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(
                str(output_path), schema=BLOCK_EXON_ASSIGNMENT_TABLE
            )

        unique_contigs = (
            pc.unique(blocks_with_contig.column("contig_id")).to_pylist()
        )
        for contig_id in unique_contigs:
            contig_filter = pc.equal(
                blocks_with_contig.column("contig_id"), contig_id
            )
            c_blocks = blocks_with_contig.filter(contig_filter)
            c_exons = exons.filter(pc.equal(exons.column("contig_id"), contig_id))
            if c_blocks.num_rows == 0 or c_exons.num_rows == 0:
                continue
            c_blocks = c_blocks.combine_chunks()
            c_exons = c_exons.combine_chunks()

            blk_aid = torch.from_numpy(
                c_blocks.column("alignment_id").chunk(0).to_numpy(
                    zero_copy_only=True
                )
            )
            blk_bidx = torch.from_numpy(
                c_blocks.column("block_index").chunk(0).to_numpy(
                    zero_copy_only=True
                )
            )
            blk_bs = torch.from_numpy(
                c_blocks.column("ref_start").chunk(0).to_numpy(
                    zero_copy_only=True
                )
            )
            blk_be = torch.from_numpy(
                c_blocks.column("ref_end").chunk(0).to_numpy(
                    zero_copy_only=True
                )
            )

            exon_starts = torch.from_numpy(
                c_exons.column("start").chunk(0).to_numpy(zero_copy_only=True)
            )
            exon_ends = torch.from_numpy(
                c_exons.column("end").chunk(0).to_numpy(zero_copy_only=True)
            )
            exon_ids = torch.from_numpy(
                c_exons.column("feature_id").chunk(0).to_numpy(
                    zero_copy_only=True
                )
            )
            sort_perm = torch.argsort(exon_starts, stable=True)
            es = exon_starts[sort_perm].contiguous()
            ee = exon_ends[sort_perm].contiguous()
            eid = exon_ids[sort_perm].contiguous()

            n_blocks = blk_aid.numel()
            batch_emitted = 0
            for chunk_start in range(0, n_blocks, batch_size):
                chunk_end = min(chunk_start + batch_size, n_blocks)
                b_aid = blk_aid[chunk_start:chunk_end]
                b_bidx = blk_bidx[chunk_start:chunk_end]
                b_bs = blk_bs[chunk_start:chunk_end]
                b_be = blk_be[chunk_start:chunk_end]

                batch_rb = _expand_block_exon_overlaps_batch(
                    b_aid, b_bidx, b_bs, b_be, es, ee, eid
                )
                if batch_rb is None or batch_rb.num_rows == 0:
                    continue
                batch_emitted += batch_rb.num_rows
                n_rows_written += batch_rb.num_rows
                if writer is not None:
                    writer.write_batch(batch_rb)
                else:
                    out_batches.append(batch_rb)
            if batch_emitted > 0:
                n_contigs += 1
    finally:
        if writer is not None:
            writer.close()

    if output_path is not None:
        return {"n_rows_written": n_rows_written, "n_contigs": n_contigs}
    if not out_batches:
        return _empty_block_assignments()
    return pa.Table.from_batches(out_batches, schema=BLOCK_EXON_ASSIGNMENT_TABLE)


def _finalise_empty_block_assignments(
    output_path: Path | None,
) -> pa.Table | dict[str, int]:
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_empty_block_assignments(), str(output_path))
        return {"n_rows_written": 0, "n_contigs": 0}
    return _empty_block_assignments()


def _expand_block_exon_overlaps_batch(
    blk_aid: torch.Tensor,
    blk_bidx: torch.Tensor,
    blk_bs: torch.Tensor,
    blk_be: torch.Tensor,
    exon_starts_sorted: torch.Tensor,
    exon_ends_sorted: torch.Tensor,
    exon_ids_sorted: torch.Tensor,
) -> pa.RecordBatch | None:
    """Vectorised per-batch interval overlap.

    ``torch.searchsorted`` over the sorted exon-starts gives an
    inclusive-of-zero upper bound on candidate exons per block (exons
    with ``start < block_end``); the post-filter ``exon_end > block_start``
    completes the half-open overlap predicate. The expansion via
    ``torch.repeat_interleave`` turns the (block, candidate-slice) shape
    into a flat (block_repeat, exon_idx) pair list — all without a
    Python-level loop over blocks.
    """
    n_batch = blk_bs.numel()
    if n_batch == 0:
        return None

    hi = torch.searchsorted(exon_starts_sorted, blk_be, right=False)
    # candidate count per block (lo is always 0 because we filter
    # exon_end > block_start as a post-step).
    counts = hi.to(torch.int64)
    total = int(counts.sum().item())
    if total == 0:
        return None

    cum = torch.cumsum(counts, dim=0)
    starts_offset = cum - counts
    block_repeats = torch.repeat_interleave(
        torch.arange(n_batch, dtype=torch.int64), counts
    )
    within = torch.arange(total, dtype=torch.int64) - torch.repeat_interleave(
        starts_offset, counts
    )
    exon_idx = within  # since lo == 0

    cand_es = exon_starts_sorted[exon_idx]
    cand_ee = exon_ends_sorted[exon_idx]
    cand_eid = exon_ids_sorted[exon_idx]
    cand_bs = blk_bs[block_repeats]
    cand_be = blk_be[block_repeats]
    cand_aid = blk_aid[block_repeats]
    cand_bidx = blk_bidx[block_repeats]

    overlap_mask = cand_ee > cand_bs
    if not bool(overlap_mask.any()):
        return None

    cand_es = cand_es[overlap_mask]
    cand_ee = cand_ee[overlap_mask]
    cand_eid = cand_eid[overlap_mask]
    cand_bs = cand_bs[overlap_mask]
    cand_be = cand_be[overlap_mask]
    cand_aid = cand_aid[overlap_mask]
    cand_bidx = cand_bidx[overlap_mask]

    ovl_starts = torch.maximum(cand_es, cand_bs)
    ovl_ends = torch.minimum(cand_ee, cand_be)
    ovl_bp = (ovl_ends - ovl_starts).to(torch.int32)
    block_len = (cand_be - cand_bs).to(torch.float32)
    exon_len = (cand_ee - cand_es).to(torch.float32)
    block_fraction = ovl_bp.to(torch.float32) / block_len
    exon_fraction = ovl_bp.to(torch.float32) / exon_len

    return pa.RecordBatch.from_arrays(
        [
            pa.array(cand_aid.numpy(), type=pa.int64()),
            pa.array(cand_bidx.numpy(), type=pa.int32()),
            pa.array(cand_eid.numpy(), type=pa.int64()),
            pa.array(ovl_bp.numpy(), type=pa.int32()),
            pa.array(block_fraction.numpy(), type=pa.float32()),
            pa.array(exon_fraction.numpy(), type=pa.float32()),
        ],
        schema=BLOCK_EXON_ASSIGNMENT_TABLE,
    )


# ──────────────────────────────────────────────────────────────────────
# compute_exon_psi
# ──────────────────────────────────────────────────────────────────────


def compute_exon_psi(
    block_assignments: BlockAssignmentsLike,
    alignments: pa.Table,
    derived_features: pa.Table,
    contigs: pa.Table,
    read_to_sample: pa.Table | None = None,
) -> pa.Table:
    """Per-(derived_exon, sample) inclusion / exclusion / PSI substrate.

    Definitions:

    - **Gene-spanning alignment**: ``alignment.ref_start <= gene.start AND
      alignment.ref_end >= gene.end`` on the same contig.
    - **Inclusion**: gene-spanning alignment with at least one block
      overlapping the exon (a row in ``block_assignments`` with
      ``overlap_bp > 0`` for ``(alignment_id, exon_id)``).
    - **Exclusion**: gene-spanning alignment with NO block overlapping
      the exon. By construction such an alignment must have spliced
      past the exon (a single un-spliced block spanning the gene
      would intersect every exon).
    - **Ambiguous**: alignment that overlaps the gene but doesn't
      fully span it — diagnostic only; not in the PSI ratio.

    PSI = ``n_inclusion_reads / (n_inclusion_reads + n_exclusion_reads)``.

    Parameters
    ----------
    block_assignments : pa.Table | Path | str
        Either an in-memory ``BLOCK_EXON_ASSIGNMENT_TABLE``-shaped table
        (tests + Jupyter callers) or a path to a Parquet file / dataset
        directory produced by :func:`assign_blocks_to_exons` in
        streaming mode (production callers). Only distinct
        ``(alignment_id, data_exon_id)`` pairs are needed downstream;
        the path variant is stream-scanned in batches so the on-disk
        50+ GB table at mouse scale never materialises in RAM.
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped.
    derived_features : pa.Table
        ``FEATURE_TABLE``-shaped output of :func:`roll_up_genes` —
        contains both gene + exon rows.
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — supplies the ``ref_name → contig_id``
        resolution.
    read_to_sample : pa.Table | None
        Optional 2-column Arrow table with ``read_id`` (string) +
        ``sample_id`` (int64). When omitted, every alignment maps to
        ``sample_id = -1`` (matches ``COVERAGE_TABLE``'s unstratified
        sentinel).

    Returns
    -------
    pa.Table conforming to ``EXON_PSI_TABLE``. One row per
    ``(data_exon_id, sample_id)`` pair with at least one
    gene-spanning or partial-spanning alignment.
    """
    exons = derived_features.filter(
        pc.equal(derived_features.column("type"), "exon")
    ).select(["feature_id", "contig_id", "start", "end", "parent_id"])
    genes = derived_features.filter(
        pc.equal(derived_features.column("type"), "gene")
    ).select(["feature_id", "contig_id", "start", "end"])
    if exons.num_rows == 0 or genes.num_rows == 0:
        return _empty_exon_psi()

    # Filter to primary alignments; attach contig_id + sample_id via
    # Arrow joins. The prior implementation built an alignment-id→metadata
    # Python dict at this point — ~200M dicts of dicts at mouse scale.
    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    primary = alignments.filter(primary_mask).select(
        ["alignment_id", "read_id", "ref_name", "ref_start", "ref_end"]
    )
    if primary.num_rows == 0:
        return _empty_exon_psi()

    contig_lookup = contigs.select(["contig_id", "name"]).rename_columns(
        ["contig_id", "ref_name"]
    )
    primary = primary.join(contig_lookup, keys="ref_name", join_type="inner")
    if primary.num_rows == 0:
        return _empty_exon_psi()

    if read_to_sample is not None:
        primary = primary.join(
            read_to_sample, keys="read_id", join_type="left outer"
        )
        primary = primary.set_column(
            primary.column_names.index("sample_id"),
            "sample_id",
            pc.fill_null(
                primary.column("sample_id"), _UNSTRATIFIED_SAMPLE_ID
            ),
        )
    else:
        primary = primary.append_column(
            "sample_id",
            pa.array(
                np.full(primary.num_rows, _UNSTRATIFIED_SAMPLE_ID, dtype=np.int64),
                type=pa.int64(),
            ),
        )

    # ── Gene-spanning classification per contig ─────────────────────
    aln_gene_role = _classify_alignments_per_gene(primary, genes)
    if aln_gene_role.num_rows == 0:
        return _empty_exon_psi()

    # Cross-join with exons on parent_id → enumerate (alignment, exon, role)
    # for each gene's exons.
    exon_membership = exons.select(["feature_id", "parent_id"]).rename_columns(
        ["data_exon_id", "gene_id"]
    )
    aln_exon_role = aln_gene_role.join(
        exon_membership, keys="gene_id", join_type="inner"
    )
    if aln_exon_role.num_rows == 0:
        return _empty_exon_psi()

    # Extract distinct (alignment_id, data_exon_id) overlap pairs from
    # block_assignments. Accepts either an in-memory table or a path
    # (Parquet file / dataset dir) that we stream-scan in batches.
    overlap_pairs = _extract_distinct_overlap_pairs(block_assignments)

    # Left-join indicates "did this (alignment, exon) actually overlap
    # at the block level". We use a sentinel-column trick: add a
    # constant `_has_overlap` int8 to overlap_pairs; after the left
    # join, that column is null iff there was no overlap.
    overlap_pairs_marked = overlap_pairs.append_column(
        "_has_overlap",
        pa.array(np.ones(overlap_pairs.num_rows, dtype=np.int8), type=pa.int8()),
    )
    joined = aln_exon_role.join(
        overlap_pairs_marked,
        keys=["alignment_id", "data_exon_id"],
        join_type="left outer",
    )
    has_overlap = pc.is_valid(joined.column("_has_overlap"))
    is_spanning = pc.equal(joined.column("role"), "spanning")
    is_partial = pc.equal(joined.column("role"), "partial")

    inc_indicator = pc.cast(pc.and_(is_spanning, has_overlap), pa.int32())
    exc_indicator = pc.cast(
        pc.and_(is_spanning, pc.invert(has_overlap)), pa.int32()
    )
    amb_indicator = pc.cast(is_partial, pa.int32())

    enriched = joined.append_column("inc_i", inc_indicator)
    enriched = enriched.append_column("exc_i", exc_indicator)
    enriched = enriched.append_column("amb_i", amb_indicator)

    aggregated = enriched.group_by(["data_exon_id", "sample_id"]).aggregate(
        [("inc_i", "sum"), ("exc_i", "sum"), ("amb_i", "sum")]
    )
    if aggregated.num_rows == 0:
        return _empty_exon_psi()

    aggregated = aggregated.sort_by(
        [
            ("data_exon_id", "ascending"),
            ("sample_id", "ascending"),
        ]
    )

    inc = pc.cast(aggregated.column("inc_i_sum"), pa.int32())
    exc = pc.cast(aggregated.column("exc_i_sum"), pa.int32())
    amb = pc.cast(aggregated.column("amb_i_sum"), pa.int32())
    denom = pc.add(inc, exc)
    denom_positive = pc.greater(denom, 0)
    psi = pc.if_else(
        denom_positive,
        pc.divide(pc.cast(inc, pa.float32()), pc.cast(denom, pa.float32())),
        pa.scalar(None, type=pa.float32()),
    )

    return pa.table(
        {
            "data_exon_id": aggregated.column("data_exon_id"),
            "sample_id": aggregated.column("sample_id"),
            "n_inclusion_reads": inc,
            "n_exclusion_reads": exc,
            "n_ambiguous_reads": amb,
            "psi": psi,
        },
        schema=EXON_PSI_TABLE,
    )


def _classify_alignments_per_gene(
    primary: pa.Table, genes: pa.Table
) -> pa.Table:
    """Build ``(alignment_id, gene_id, role, sample_id)`` rows.

    ``role`` is ``'spanning'`` (alignment fully contains gene),
    ``'partial'`` (alignment overlaps gene but doesn't span). Per-contig
    chunked broadcast in torch: sort alignments by ref_start once per
    contig; then for chunks of ~K genes, build a ``(K × M_alignments)``
    boolean grid and emit non-zero entries via ``torch.nonzero``. K is
    auto-tuned so peak intermediate bytes stay near 1 GB.
    """
    if primary.num_rows == 0 or genes.num_rows == 0:
        return _empty_aln_gene_role()

    # 1 GB target / (1 byte/bool × n_alignments_per_contig) → genes per chunk.
    # Bounded between 1 and 256 to keep per-iteration overhead reasonable.
    _GENE_CHUNK_TARGET_BYTES = 1_073_741_824

    out_aid: list[torch.Tensor] = []
    out_gid: list[torch.Tensor] = []
    out_sid: list[torch.Tensor] = []
    out_role: list[torch.Tensor] = []
    role_spanning = 0
    role_partial = 1

    gene_contigs = pc.unique(genes.column("contig_id")).to_pylist()
    for contig_id in gene_contigs:
        c_genes = genes.filter(pc.equal(genes.column("contig_id"), contig_id))
        c_primary = primary.filter(
            pc.equal(primary.column("contig_id"), contig_id)
        )
        if c_genes.num_rows == 0 or c_primary.num_rows == 0:
            continue
        c_primary = c_primary.combine_chunks()
        c_genes = c_genes.combine_chunks()

        a_start = torch.from_numpy(
            c_primary.column("ref_start").chunk(0).to_numpy(zero_copy_only=True)
        )
        a_end = torch.from_numpy(
            c_primary.column("ref_end").chunk(0).to_numpy(zero_copy_only=True)
        )
        a_id = torch.from_numpy(
            c_primary.column("alignment_id").chunk(0).to_numpy(
                zero_copy_only=True
            )
        )
        a_sid = torch.from_numpy(
            c_primary.column("sample_id").chunk(0).to_numpy(
                zero_copy_only=True
            )
        )
        order = torch.argsort(a_start, stable=True)
        a_start = a_start[order].contiguous()
        a_end = a_end[order].contiguous()
        a_id = a_id[order].contiguous()
        a_sid = a_sid[order].contiguous()
        M = a_start.numel()

        g_start = torch.from_numpy(
            c_genes.column("start").chunk(0).to_numpy(zero_copy_only=True)
        )
        g_end = torch.from_numpy(
            c_genes.column("end").chunk(0).to_numpy(zero_copy_only=True)
        )
        g_id = torch.from_numpy(
            c_genes.column("feature_id").chunk(0).to_numpy(zero_copy_only=True)
        )
        n_genes = g_start.numel()

        chunk_size = max(1, min(256, _GENE_CHUNK_TARGET_BYTES // max(M, 1)))
        for cs in range(0, n_genes, chunk_size):
            ce = min(cs + chunk_size, n_genes)
            cg_start = g_start[cs:ce]
            cg_end = g_end[cs:ce]
            cg_id = g_id[cs:ce]

            # Broadcast: (K, M) boolean matrices for spanning + overlap.
            spanning = (a_start.unsqueeze(0) <= cg_start.unsqueeze(1)) & (
                a_end.unsqueeze(0) >= cg_end.unsqueeze(1)
            )
            overlapping = (a_start.unsqueeze(0) < cg_end.unsqueeze(1)) & (
                a_end.unsqueeze(0) > cg_start.unsqueeze(1)
            )
            partials = overlapping & ~spanning

            if bool(spanning.any()):
                sp_idx = torch.nonzero(spanning, as_tuple=False)
                k_idx = sp_idx[:, 0]
                m_idx = sp_idx[:, 1]
                out_aid.append(a_id[m_idx])
                out_gid.append(cg_id[k_idx])
                out_sid.append(a_sid[m_idx])
                out_role.append(
                    torch.full(
                        (sp_idx.shape[0],), role_spanning, dtype=torch.int8
                    )
                )
            if bool(partials.any()):
                pt_idx = torch.nonzero(partials, as_tuple=False)
                k_idx = pt_idx[:, 0]
                m_idx = pt_idx[:, 1]
                out_aid.append(a_id[m_idx])
                out_gid.append(cg_id[k_idx])
                out_sid.append(a_sid[m_idx])
                out_role.append(
                    torch.full(
                        (pt_idx.shape[0],), role_partial, dtype=torch.int8
                    )
                )

    if not out_aid:
        return _empty_aln_gene_role()

    aid_t = torch.cat(out_aid)
    gid_t = torch.cat(out_gid)
    sid_t = torch.cat(out_sid)
    role_t = torch.cat(out_role)
    role_strs = pc.if_else(
        pc.equal(pa.array(role_t.numpy(), type=pa.int8()), role_spanning),
        "spanning",
        "partial",
    )
    return pa.table(
        {
            "alignment_id": pa.array(aid_t.numpy(), type=pa.int64()),
            "gene_id": pa.array(gid_t.numpy(), type=pa.int64()),
            "sample_id": pa.array(sid_t.numpy(), type=pa.int64()),
            "role": role_strs,
        }
    )


def _empty_aln_gene_role() -> pa.Table:
    return pa.table(
        {
            "alignment_id": pa.array([], type=pa.int64()),
            "gene_id": pa.array([], type=pa.int64()),
            "sample_id": pa.array([], type=pa.int64()),
            "role": pa.array([], type=pa.string()),
        }
    )


def _extract_distinct_overlap_pairs(
    block_assignments: BlockAssignmentsLike,
) -> pa.Table:
    """Return a 2-column distinct ``(alignment_id, data_exon_id)`` table.

    Accepts an in-memory table or a Parquet path. The path variant
    stream-scans the dataset in batches and group-by-deduplicates each
    batch before the final cross-batch dedup — peak resident is bounded
    by the per-batch row count + the cardinality of distinct pairs (~7
    GB at mouse scale), not the full ~50 GB block_assignments file.
    """
    empty = pa.table(
        {
            "alignment_id": pa.array([], type=pa.int64()),
            "data_exon_id": pa.array([], type=pa.int64()),
        }
    )
    if isinstance(block_assignments, pa.Table):
        if block_assignments.num_rows == 0:
            return empty
        return (
            block_assignments.select(["alignment_id", "data_exon_id"])
            .group_by(["alignment_id", "data_exon_id"])
            .aggregate([])
        )
    if isinstance(block_assignments, (str, Path)):
        ds = pa_dataset.dataset(str(block_assignments))
        per_batch: list[pa.Table] = []
        for batch in ds.to_batches(columns=["alignment_id", "data_exon_id"]):
            if batch.num_rows == 0:
                continue
            tbl = pa.Table.from_batches([batch])
            per_batch.append(
                tbl.group_by(["alignment_id", "data_exon_id"]).aggregate([])
            )
        if not per_batch:
            return empty
        return (
            pa.concat_tables(per_batch)
            .group_by(["alignment_id", "data_exon_id"])
            .aggregate([])
        )
    raise TypeError(
        f"block_assignments must be pa.Table or path, got {type(block_assignments)!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────


def build_derived_annotation(
    coverage: pa.Table | Path | str,
    introns: pa.Table,
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    contigs: pa.Table,
    *,
    read_to_sample: pa.Table | None = None,
    block_assignments_output_path: Path | None = None,
    min_exon_depth: int = 5,
    min_intron_read_count: int = 3,
):
    """Resolve-stage entry point — chains :func:`derive_exons` →
    :func:`roll_up_genes` → :func:`assign_blocks_to_exons` →
    :func:`compute_exon_psi`.

    Parameters
    ----------
    coverage : pa.Table | Path | str
        ``COVERAGE_TABLE`` data, either in-memory (tests + Jupyter) or
        a path to the Parquet file written by :func:`build_pileup` in
        streaming mode (production). Path variant is read once via
        ``pq.read_table`` — coverage post-RLE is small enough that
        in-memory consumption inside ``derive_exons`` is fine.
    introns, alignment_blocks, alignments, contigs
        Standard Arrow inputs.
    read_to_sample : pa.Table | None
        2-column ``(read_id, sample_id)`` Arrow table for per-sample
        PSI stratification. None → unstratified (sample_id = -1).
    block_assignments_output_path : Path | None
        When provided, :func:`assign_blocks_to_exons` streams the M:N
        block × exon edge table to this Parquet file and
        :func:`compute_exon_psi` reads it back as a dataset. When
        None, the edge table stays in memory and is returned in the
        result tuple (tests + Jupyter ergonomic).

    Returns
    -------
    (annotation, block_assignments, exon_psi)
        - ``annotation`` : :class:`Annotation` with FEATURE_TABLE
          containing gene + exon rows. Empty when no trusted introns
          are present.
        - ``block_assignments`` : ``BLOCK_EXON_ASSIGNMENT_TABLE``-shaped
          table if ``block_assignments_output_path`` is None; otherwise
          the same `Path` (production callers consume from disk).
        - ``exon_psi`` : ``EXON_PSI_TABLE``-shaped.
    """
    # Local import to avoid circular import at module load time.
    from constellation.sequencing.annotation.annotation import Annotation

    metadata = {
        "min_exon_depth": int(min_exon_depth),
        "min_intron_read_count": int(min_intron_read_count),
        "source": "constellation_derived",
    }

    if isinstance(coverage, (str, Path)):
        coverage_table = pq.read_table(str(coverage))
    else:
        coverage_table = coverage

    exons = derive_exons(
        coverage_table, introns, contigs,
        min_exon_depth=min_exon_depth,
        min_intron_read_count=min_intron_read_count,
    )
    if exons.num_rows == 0:
        empty_annotation = Annotation(
            features=_empty_features_table(),
            metadata_extras=metadata,
        )
        empty_ba: pa.Table | Path
        if block_assignments_output_path is not None:
            # Still write an empty Parquet so the CLI manifest can
            # reference the path uniformly.
            Path(block_assignments_output_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            pq.write_table(
                _empty_block_assignments(),
                str(block_assignments_output_path),
            )
            empty_ba = Path(block_assignments_output_path)
        else:
            empty_ba = _empty_block_assignments()
        return (
            empty_annotation,
            empty_ba,
            _empty_exon_psi(),
        )

    full_features = roll_up_genes(
        exons, introns, min_intron_read_count=min_intron_read_count
    )

    if block_assignments_output_path is not None:
        # Streaming mode: write to disk; compute_exon_psi reads back
        # as a dataset.
        assign_blocks_to_exons(
            alignment_blocks, alignments, full_features, contigs,
            output_path=Path(block_assignments_output_path),
        )
        block_assignments_for_psi: BlockAssignmentsLike = Path(
            block_assignments_output_path
        )
        block_assignments_return: pa.Table | Path = Path(
            block_assignments_output_path
        )
    else:
        block_assignments_table = assign_blocks_to_exons(
            alignment_blocks, alignments, full_features, contigs,
        )
        block_assignments_for_psi = block_assignments_table
        block_assignments_return = block_assignments_table

    exon_psi = compute_exon_psi(
        block_assignments_for_psi, alignments, full_features, contigs,
        read_to_sample=read_to_sample,
    )
    annotation = Annotation(
        features=full_features,
        metadata_extras=metadata,
    )
    return annotation, block_assignments_return, exon_psi


__all__ = [
    "derive_exons",
    "roll_up_genes",
    "assign_blocks_to_exons",
    "compute_exon_psi",
    "build_derived_annotation",
]
