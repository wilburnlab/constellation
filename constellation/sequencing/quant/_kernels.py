"""Per-table kernels — filter + gene-overlap, used by every Mode A path.

Two pure functions over Arrow tables; no I/O. Both ``Alignments.filter``
(in-memory) and the fused per-chunk worker
(:func:`sequencing.quant.genome_count.fused_decode_filter_overlap_worker`)
call these — single source of truth for filter predicates and the
sort+searchsorted gene-overlap join.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.alignments.alignments import _aligned_fraction_array


# ──────────────────────────────────────────────────────────────────────
# Filter
# ──────────────────────────────────────────────────────────────────────


def apply_filter_predicates(
    alignments_table: pa.Table,
    *,
    min_length: int | None = None,
    min_aligned_fraction: float | None = None,
    min_mapq: int | None = None,
    primary_only: bool = False,
) -> tuple[pa.Table, list[dict[str, int]]]:
    """Return ``(filtered, audit)`` with each predicate applied as a
    distinct stage so the audit trail surfaces per-filter row counts.

    Vectorised throughout — no Python row loops except the per-row
    CIGAR walk inside :func:`_aligned_fraction_array` (only invoked
    when ``min_aligned_fraction`` is set).
    """
    table = alignments_table
    audit: list[dict[str, int]] = []

    def _record(stage: str, before: int, after_table: pa.Table) -> pa.Table:
        after = after_table.num_rows
        audit.append({"stage": stage, "kept": after, "dropped": before - after})
        return after_table

    if primary_only:
        before = table.num_rows
        mask = pc.and_(
            pc.invert(table.column("is_secondary")),
            pc.invert(table.column("is_supplementary")),
        )
        table = _record("primary_only", before, table.filter(mask))

    if min_length is not None:
        before = table.num_rows
        length = pc.subtract(table.column("ref_end"), table.column("ref_start"))
        mask = pc.greater_equal(length, int(min_length))
        table = _record(f"min_length>={min_length}", before, table.filter(mask))

    if min_mapq is not None:
        before = table.num_rows
        mask = pc.greater_equal(table.column("mapq"), int(min_mapq))
        table = _record(f"min_mapq>={min_mapq}", before, table.filter(mask))

    if min_aligned_fraction is not None:
        before = table.num_rows
        af = _aligned_fraction_array(table)
        mask = pc.and_(
            pc.is_valid(af),
            pc.greater_equal(af, float(min_aligned_fraction)),
        )
        table = _record(
            f"min_aligned_fraction>={min_aligned_fraction}",
            before,
            table.filter(mask),
        )

    return table, audit


# ──────────────────────────────────────────────────────────────────────
# Gene overlap
# ──────────────────────────────────────────────────────────────────────


GENE_ASSIGNMENT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("alignment_id", pa.int64(), nullable=False),
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("gene_id", pa.int64(), nullable=False),
        pa.field("overlap_fraction", pa.float32(), nullable=False),
    ]
)


def gene_set_from_annotation(annotation, genome) -> pa.Table:
    """Project ``Annotation.features_of_type('gene')`` plus contig
    names from ``GenomeReference`` into the (contig_name, start, end,
    strand, gene_id) shape consumed by :func:`compute_gene_overlap`.

    Built once at the parent level and broadcast to fused workers via
    Arrow IPC. Bounded-small (~5–50k rows even for mammalian genomes).
    """
    contigs = genome.contigs
    contig_id_to_name = dict(
        zip(
            contigs.column("contig_id").to_pylist(),
            contigs.column("name").to_pylist(),
        )
    )
    genes = annotation.features_of_type("gene")
    feature_ids = genes.column("feature_id").to_pylist()
    contig_ids = genes.column("contig_id").to_pylist()
    starts = genes.column("start").to_pylist()
    ends = genes.column("end").to_pylist()
    strands = genes.column("strand").to_pylist()
    rows: list[dict[str, object]] = []
    for fid, cid, s, e, strand in zip(feature_ids, contig_ids, starts, ends, strands):
        name = contig_id_to_name.get(int(cid))
        if name is None:
            continue  # gene on a contig not in the genome — skip silently
        rows.append(
            {
                "gene_id": int(fid),
                "contig_name": name,
                "start": int(s),
                "end": int(e),
                "strand": strand,
            }
        )
    schema = pa.schema(
        [
            pa.field("gene_id", pa.int64(), nullable=False),
            pa.field("contig_name", pa.string(), nullable=False),
            pa.field("start", pa.int64(), nullable=False),
            pa.field("end", pa.int64(), nullable=False),
            pa.field("strand", pa.string(), nullable=False),
        ]
    )
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def compute_gene_overlap(
    alignments_chunk: pa.Table,
    gene_set: pa.Table,
    *,
    min_overlap_fraction: float = 0.5,
    allow_antisense: bool = False,
) -> pa.Table:
    """Per-chunk sort+searchsorted sweep-join.

    For each alignment, finds the gene with the largest overlap on the
    same contig (and matching strand unless ``allow_antisense``); drops
    alignments where the best overlap is below ``min_overlap_fraction``
    of the alignment's reference span (``ref_end - ref_start``).

    Returns ``(alignment_id, read_id, gene_id, overlap_fraction)``;
    one row per matched alignment, schema fixed to GENE_ASSIGNMENT_SCHEMA.
    """
    if alignments_chunk.num_rows == 0:
        return GENE_ASSIGNMENT_SCHEMA.empty_table()

    # Pull the columns we need into NumPy. Strings stay as Python lists —
    # NumPy's object-dtype string handling here is no faster than a
    # plain list, and we only iterate once per alignment.
    a_ids = np.asarray(
        alignments_chunk.column("alignment_id").to_numpy(zero_copy_only=False)
    )
    a_reads = alignments_chunk.column("read_id").to_pylist()
    a_contigs = alignments_chunk.column("ref_name").to_pylist()
    a_starts = np.asarray(
        alignments_chunk.column("ref_start").to_numpy(zero_copy_only=False)
    )
    a_ends = np.asarray(
        alignments_chunk.column("ref_end").to_numpy(zero_copy_only=False)
    )
    a_strands = alignments_chunk.column("strand").to_pylist()

    if gene_set.num_rows == 0:
        return GENE_ASSIGNMENT_SCHEMA.empty_table()

    g_ids = np.asarray(gene_set.column("gene_id").to_numpy(zero_copy_only=False))
    g_contigs = gene_set.column("contig_name").to_pylist()
    g_starts = np.asarray(gene_set.column("start").to_numpy(zero_copy_only=False))
    g_ends = np.asarray(gene_set.column("end").to_numpy(zero_copy_only=False))
    g_strands = gene_set.column("strand").to_pylist()

    # Per-contig partition — group genes by contig name, sort within.
    by_contig: dict[str, dict[str, np.ndarray]] = {}
    contig_keys = sorted(set(g_contigs))
    for c in contig_keys:
        idx = np.array([i for i, gc in enumerate(g_contigs) if gc == c])
        if idx.size == 0:
            continue
        order = np.argsort(g_starts[idx], kind="stable")
        sorted_idx = idx[order]
        by_contig[c] = {
            "gene_ids": g_ids[sorted_idx],
            "starts": g_starts[sorted_idx],
            "ends": g_ends[sorted_idx],
            "strands": np.array([g_strands[i] for i in sorted_idx], dtype=object),
        }

    out_alignment_ids: list[int] = []
    out_read_ids: list[str] = []
    out_gene_ids: list[int] = []
    out_fractions: list[float] = []

    for i in range(len(a_ids)):
        contig = a_contigs[i]
        bucket = by_contig.get(contig)
        if bucket is None:
            continue
        a_start = int(a_starts[i])
        a_end = int(a_ends[i])
        ref_span = a_end - a_start
        if ref_span <= 0:
            continue
        a_strand = a_strands[i]
        starts = bucket["starts"]
        ends = bucket["ends"]
        gene_ids = bucket["gene_ids"]
        gene_strands = bucket["strands"]

        # Genes whose start < a_end can possibly overlap; restrict to
        # this prefix via searchsorted on the sorted starts array.
        upper = int(np.searchsorted(starts, a_end, side="left"))
        if upper == 0:
            continue
        # Among those, only the ones whose end > a_start actually overlap.
        candidate_starts = starts[:upper]
        candidate_ends = ends[:upper]
        candidate_ids = gene_ids[:upper]
        candidate_strands = gene_strands[:upper]
        end_mask = candidate_ends > a_start
        if not end_mask.any():
            continue
        c_starts = candidate_starts[end_mask]
        c_ends = candidate_ends[end_mask]
        c_ids = candidate_ids[end_mask]
        c_strands = candidate_strands[end_mask]

        # Strand check — gene strand '.' is unstranded, always passes.
        if not allow_antisense:
            keep = np.array(
                [
                    (gs == "." or gs == a_strand)
                    for gs in c_strands
                ]
            )
            if not keep.any():
                continue
            c_starts = c_starts[keep]
            c_ends = c_ends[keep]
            c_ids = c_ids[keep]

        overlap = np.minimum(c_ends, a_end) - np.maximum(c_starts, a_start)
        # Should be > 0 given the end_mask guard, but clamp for safety.
        overlap = np.maximum(overlap, 0)
        best = int(np.argmax(overlap))
        best_overlap = int(overlap[best])
        fraction = best_overlap / ref_span
        if fraction < min_overlap_fraction:
            continue

        out_alignment_ids.append(int(a_ids[i]))
        out_read_ids.append(a_reads[i])
        out_gene_ids.append(int(c_ids[best]))
        out_fractions.append(float(fraction))

    if not out_alignment_ids:
        return GENE_ASSIGNMENT_SCHEMA.empty_table()
    return pa.Table.from_pydict(
        {
            "alignment_id": out_alignment_ids,
            "read_id": out_read_ids,
            "gene_id": out_gene_ids,
            "overlap_fraction": out_fractions,
        },
        schema=GENE_ASSIGNMENT_SCHEMA,
    )


__all__ = [
    "apply_filter_predicates",
    "compute_gene_overlap",
    "gene_set_from_annotation",
    "GENE_ASSIGNMENT_SCHEMA",
]
